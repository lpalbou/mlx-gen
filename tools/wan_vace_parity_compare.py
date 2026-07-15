"""Compare the native MLX Wan VACE port against exported diffusers reference tensors.

Loads the reference bundle from tools/wan_vace_parity_export.py, feeds the SAME inputs through
the MLX runtime stages, and reports per-stage max-abs / rel errors. Structural porting errors
produce O(1) differences; bf16-weight vs fp32-reference differences stay in the 1e-2 band.

Usage:
  uv run python tools/wan_vace_parity_compare.py \
      --reference validation_outputs/wan_vace_parity/reference.safetensors
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_reference(path: Path) -> dict[str, mx.array]:
    return {key: value for key, value in mx.load(str(path)).items()}


def report(name: str, ours: mx.array, reference: mx.array, results: dict) -> None:
    ours_np = np.asarray(ours.astype(mx.float32))
    reference_np = np.asarray(reference.astype(mx.float32))
    if ours_np.shape != reference_np.shape:
        results[name] = {"status": "SHAPE_MISMATCH", "ours": list(ours_np.shape), "ref": list(reference_np.shape)}
        print(f"  {name}: SHAPE MISMATCH ours={ours_np.shape} ref={reference_np.shape}")
        return
    abs_err = np.abs(ours_np - reference_np)
    denom = np.maximum(np.abs(reference_np), 1e-3)
    rel = abs_err / denom
    results[name] = {
        "max_abs": round(float(abs_err.max()), 6),
        "mean_abs": round(float(abs_err.mean()), 6),
        "p99_rel": round(float(np.quantile(rel, 0.99)), 6),
    }
    print(f"  {name}: max_abs={abs_err.max():.5f} mean_abs={abs_err.mean():.6f} p99_rel={np.quantile(rel, 0.99):.5f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--fp32", action="store_true", help="Load MLX weights in float32 for a tight structural check.")
    args = parser.parse_args()

    from mflux.models.common.config.model_config import ModelConfig

    if args.fp32:
        ModelConfig.precision = mx.float32

    from mflux.models.wan.variants import WanVace

    manifest = json.loads(args.reference.with_suffix(".json").read_text())
    reference = load_reference(args.reference)
    results: dict = {"manifest": manifest}

    model = WanVace(model_config=None, model_path="Wan-AI/Wan2.1-VACE-1.3B-diffusers")
    guidance = float(manifest["guidance"])

    video = reference["video"].astype(mx.float32)
    mask = reference["mask"].astype(mx.float32)

    print("stage: VAE conditioning encode")
    binary_mask = mx.where(mask > 0.5, 1.0, 0.0)
    inactive = model.vae.encode_normalized(video * (1 - binary_mask)).astype(mx.float32)
    reactive = model.vae.encode_normalized(video * binary_mask).astype(mx.float32)
    mx.eval(inactive, reactive)
    report("inactive_latents", inactive, reference["inactive_latents"], results)
    report("reactive_latents", reactive, reference["reactive_latents"], results)

    reference_latent = model.vae.encode_normalized(reference["reference_image"].astype(mx.float32)).astype(mx.float32)
    reference_latent = mx.concatenate([reference_latent, mx.zeros_like(reference_latent)], axis=1)
    mx.eval(reference_latent)
    report("reference_latent_32ch", reference_latent, reference["reference_latent_32ch"], results)

    print("stage: mask channels")
    mask_channels = model._prepare_mask_channels(mask=mask, latent_frames=inactive.shape[2], num_reference_images=1)
    mx.eval(mask_channels)
    report("mask_channels", mask_channels, reference["mask_channels"], results)

    conditioning = mx.concatenate([inactive, reactive], axis=1)
    conditioning = mx.concatenate([reference_latent, conditioning], axis=2)
    control = mx.concatenate([conditioning, mask_channels], axis=1)
    mx.eval(control)
    report("control", control, reference["control"], results)

    print("stage: transformer forward (first timestep)")
    timesteps = [float(t) for t in np.asarray(reference["timesteps"])]
    control_cast = control.astype(ModelConfig.precision)
    scales = [1.0] * len(model.transformer.vace_layers)
    prompt_embeds = reference["prompt_embeds"].astype(ModelConfig.precision)
    negative_embeds = reference["negative_prompt_embeds"].astype(ModelConfig.precision)
    initial_latents = reference["initial_latents"].astype(mx.float32)

    first_noise_pred = model.transformer(
        hidden_states=initial_latents.astype(ModelConfig.precision),
        timestep=mx.array([timesteps[0]]),
        encoder_hidden_states=prompt_embeds,
        control_hidden_states=control_cast,
        control_hidden_states_scale=scales,
    )
    mx.eval(first_noise_pred)
    report("first_noise_pred", first_noise_pred, reference["first_noise_pred"], results)

    print("stage: bounded denoise loop")
    scheduler = model._create_scheduler(flow_shift=3.0, solver="unipc")
    scheduler.set_timesteps(int(manifest["num_steps"]))
    latents = initial_latents
    for timestep in scheduler.timesteps.tolist():
        timestep_batch = model._batch_timestep(batch_size=1, timestep=timestep)
        noise_pred = model.transformer(
            hidden_states=latents.astype(ModelConfig.precision),
            timestep=timestep_batch,
            encoder_hidden_states=prompt_embeds,
            control_hidden_states=control_cast,
            control_hidden_states_scale=scales,
        )
        noise_uncond = model.transformer(
            hidden_states=latents.astype(ModelConfig.precision),
            timestep=timestep_batch,
            encoder_hidden_states=negative_embeds,
            control_hidden_states=control_cast,
            control_hidden_states_scale=scales,
        )
        noise_pred = noise_uncond + guidance * (noise_pred - noise_uncond)
        latents = scheduler.step(noise_pred.astype(mx.float32), timestep, latents, return_dict=False)[0]
        mx.eval(latents)
    report("final_latents_with_ref", latents, reference["final_latents_with_ref"], results)

    print("stage: VAE decode")
    decoded = model.vae.decode_normalized_latents(latents.astype(mx.float32)[:, :, 1:])
    decoded = mx.clip(decoded, -1.0, 1.0)
    mx.eval(decoded)
    report("decoded", decoded, reference["decoded"], results)

    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"results written to {args.json}")


if __name__ == "__main__":
    main()
