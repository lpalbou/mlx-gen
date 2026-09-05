"""Direct loader for the MiniMax-H3 audio VAE (verification and standalone use)."""

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.minimax_h3.model.h3_audio_vae.h3_audio_vae import H3AudioVAE
from mflux.models.minimax_h3.model.h3_precision import disable_tf32
from mflux.models.minimax_h3.weights.h3_weight_mapping import MiniMaxH3WeightMapping

convert_audio_vae_state = MiniMaxH3WeightMapping.convert_audio_vae_state
fold_weight_norm = MiniMaxH3WeightMapping.fold_weight_norm


def audio_vae_kwargs(config: dict) -> dict:
    return dict(
        encoder_dim=config["encoder_dim"],
        encoder_rates=tuple(config["encoder_rates"]),
        latent_dim=config["latent_dim"],
        latent_channels=config["latent_channels"],
        num_attention_heads=config["num_attention_heads"],
        decoder_dim=config["decoder_dim"],
        decoder_rates=tuple(config["decoder_rates"]),
        decoder_kernel_sizes=tuple(config["decoder_kernel_sizes"]),
        resblock_kernel_sizes=tuple(config["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(tuple(d) for d in config["resblock_dilation_sizes"]),
        sampling_rate=config["sampling_rate"],
        latents_mean=config.get("latents_mean"),
        latents_std=config.get("latents_std"),
    )


def check_state_matches(model, weights: dict[str, mx.array], name: str) -> None:
    expected = {key: value.shape for key, value in tree_flatten(model.parameters())}
    missing = sorted(set(expected) - set(weights))
    extra = sorted(set(weights) - set(expected))
    mismatched = sorted(k for k in set(expected) & set(weights) if tuple(weights[k].shape) != tuple(expected[k]))
    if missing or extra or mismatched:
        raise ValueError(
            f"{name} weight mismatch: missing={len(missing)} {missing[:3]}; extra={len(extra)} {extra[:3]}; "
            f"shape-mismatch={len(mismatched)} {[(k, tuple(weights[k].shape), tuple(expected[k])) for k in mismatched[:3]]}"
        )


def load_h3_audio_vae(root: str | Path) -> H3AudioVAE:
    disable_tf32()
    root = Path(root)
    model = H3AudioVAE(**audio_vae_kwargs(json.loads((root / "config.json").read_text())))
    weights = convert_audio_vae_state(mx.load(str(root / "diffusion_pytorch_model.safetensors")))
    check_state_matches(model, weights, "H3 audio VAE")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model
