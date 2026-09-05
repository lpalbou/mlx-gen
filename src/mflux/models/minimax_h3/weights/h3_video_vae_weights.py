"""Direct loader for the MiniMax-H3 visual VAE (verification and standalone use)."""

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten

from mflux.models.minimax_h3.model.h3_precision import disable_tf32
from mflux.models.minimax_h3.model.h3_video_vae.h3_video_vae import H3VideoVAE
from mflux.models.minimax_h3.weights.h3_audio_vae_weights import check_state_matches
from mflux.models.minimax_h3.weights.h3_weight_mapping import MiniMaxH3WeightMapping


def convert_video_vae_state(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    return {key: MiniMaxH3WeightMapping.video_vae_transform(value) for key, value in weights.items()}


def video_vae_kwargs(config: dict) -> dict:
    kwargs = {k: config[k] for k in H3VideoVAE.__init__.__code__.co_varnames[1:] if k in config}
    for key in ("block_out_channels", "spatial_downsample_factors", "temporal_downsample_factors"):
        kwargs[key] = tuple(kwargs[key])
    return kwargs


def load_h3_video_vae(root: str | Path, dtype: mx.Dtype | None = None) -> H3VideoVAE:
    disable_tf32()
    root = Path(root)
    model = H3VideoVAE(**video_vae_kwargs(json.loads((root / "config.json").read_text())))
    weights: dict[str, mx.array] = {}
    for shard in sorted(root.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    weights = convert_video_vae_state(weights)
    if dtype is not None:
        weights = {
            k: (v.astype(dtype) if v.dtype in (mx.float32, mx.bfloat16, mx.float16) else v) for k, v in weights.items()
        }
    check_state_matches(model, weights, "H3 video VAE")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model
