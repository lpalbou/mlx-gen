"""Export Wan2.1-VACE-1.3B reference tensors (torch/diffusers, CPU float32) for MLX parity.

Produces a safetensors bundle with deterministic inputs and per-stage reference outputs:
conditioning latents, mask channels, control tensor, a single transformer forward, a bounded
scheduler loop, and decoded pixels. The MLX side loads the same inputs and compares stage by
stage (tools/wan_vace_parity_compare.py).

Usage:
  PYTHONPATH=/Users/albou/projects/gh/diffusers/src uv run --with accelerate --with ftfy \
      python tools/wan_vace_parity_export.py --output validation_outputs/wan_vace_parity/reference.safetensors
"""

import argparse
import json
from pathlib import Path

import torch

HEIGHT = 48
WIDTH = 80
NUM_FRAMES = 9
NUM_STEPS = 2
GUIDANCE = 5.0
SEED = 20260706
MODEL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B-diffusers/snapshots/ec4d2cb062b548996b179d493fdd05340de702a1"
)


def build_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    video = torch.rand((1, 3, NUM_FRAMES, HEIGHT, WIDTH), generator=generator) * 2.0 - 1.0
    mask = torch.zeros((1, 3, NUM_FRAMES, HEIGHT, WIDTH))
    mask[:, :, :, :, WIDTH // 2 :] = 1.0
    reference_image = torch.rand((1, 3, 1, HEIGHT, WIDTH), generator=generator) * 2.0 - 1.0
    prompt_embeds = torch.randn((1, 512, 4096), generator=generator) * 0.05
    negative_prompt_embeds = torch.randn((1, 512, 4096), generator=generator) * 0.05
    initial_latents = torch.randn((1, 16, (NUM_FRAMES - 1) // 4 + 1 + 1, HEIGHT // 8, WIDTH // 8), generator=generator)
    return {
        "video": video,
        "mask": mask,
        "reference_image": reference_image,
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "initial_latents": initial_latents,
    }


def encode_mode(vae, pixels: torch.Tensor) -> torch.Tensor:
    latent_dist = vae.encode(pixels).latent_dist
    latents = latent_dist.mode()
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    return (latents - latents_mean) / latents_std


def prepare_mask_channels(mask: torch.Tensor, latent_frames: int, num_refs: int) -> torch.Tensor:
    num_frames, height, width = mask.shape[2], mask.shape[3], mask.shape[4]
    new_height, new_width = height // 8, width // 8
    single = mask[0, 0]
    single = single.view(num_frames, new_height, 8, new_width, 8)
    single = single.permute(2, 4, 0, 1, 3).flatten(0, 1)
    single = torch.nn.functional.interpolate(
        single.unsqueeze(0), size=(latent_frames, new_height, new_width), mode="nearest-exact"
    ).squeeze(0)
    if num_refs > 0:
        padding = torch.zeros_like(single[:, :num_refs])
        single = torch.cat([padding, single], dim=1)
    return single.unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACETransformer3DModel
    from safetensors.torch import save_file

    torch.set_grad_enabled(False)
    inputs = build_inputs()

    vae = AutoencoderKLWan.from_pretrained(MODEL_SNAPSHOT / "vae", torch_dtype=torch.float32)
    transformer = WanVACETransformer3DModel.from_pretrained(MODEL_SNAPSHOT / "transformer", torch_dtype=torch.float32)
    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_SNAPSHOT / "scheduler")

    binary_mask = torch.where(inputs["mask"] > 0.5, 1.0, 0.0)
    inactive = encode_mode(vae, inputs["video"] * (1 - binary_mask))
    reactive = encode_mode(vae, inputs["video"] * binary_mask)
    conditioning = torch.cat([inactive, reactive], dim=1)

    reference_latent = encode_mode(vae, inputs["reference_image"])
    reference_latent = torch.cat([reference_latent, torch.zeros_like(reference_latent)], dim=1)
    conditioning = torch.cat([reference_latent, conditioning], dim=2)

    latent_frames = inactive.shape[2]
    mask_channels = prepare_mask_channels(inputs["mask"], latent_frames, num_refs=1)
    control = torch.cat([conditioning, mask_channels], dim=1)

    scheduler.set_timesteps(NUM_STEPS)
    timesteps = scheduler.timesteps
    conditioning_scale = control.new_ones(len(transformer.config.vace_layers))

    first_noise_pred = transformer(
        hidden_states=inputs["initial_latents"],
        timestep=timesteps[0].expand(1),
        encoder_hidden_states=inputs["prompt_embeds"],
        control_hidden_states=control,
        control_hidden_states_scale=conditioning_scale,
        return_dict=False,
    )[0]

    latents = inputs["initial_latents"].clone()
    for t in timesteps:
        noise_pred = transformer(
            hidden_states=latents,
            timestep=t.expand(1),
            encoder_hidden_states=inputs["prompt_embeds"],
            control_hidden_states=control,
            control_hidden_states_scale=conditioning_scale,
            return_dict=False,
        )[0]
        noise_uncond = transformer(
            hidden_states=latents,
            timestep=t.expand(1),
            encoder_hidden_states=inputs["negative_prompt_embeds"],
            control_hidden_states=control,
            control_hidden_states_scale=conditioning_scale,
            return_dict=False,
        )[0]
        noise_pred = noise_uncond + GUIDANCE * (noise_pred - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    final_latents = latents[:, :, 1:]
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    decoded = vae.decode(final_latents * latents_std + latents_mean, return_dict=False)[0]

    tensors = {
        **inputs,
        "inactive_latents": inactive,
        "reactive_latents": reactive,
        "reference_latent_32ch": reference_latent,
        "mask_channels": mask_channels,
        "control": control,
        "timesteps": timesteps.to(torch.float32),
        "first_noise_pred": first_noise_pred,
        "final_latents_with_ref": latents,
        "decoded": decoded.clamp(-1.0, 1.0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(args.output))
    manifest = {
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "num_steps": NUM_STEPS,
        "guidance": GUIDANCE,
        "seed": SEED,
        "snapshot": str(MODEL_SNAPSHOT),
        "shapes": {k: list(v.shape) for k, v in tensors.items()},
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(f"exported {len(tensors)} tensors to {args.output}")
    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)}")


if __name__ == "__main__":
    main()
