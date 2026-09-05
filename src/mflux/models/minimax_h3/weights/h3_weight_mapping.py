"""Weight mapping for the MiniMax-H3 components.

The transformer is a key-for-key passthrough of the diffusers checkpoint. The text encoder keeps
Qwen3-VL's own names under `model.language_model.`, of which only the embeddings and decoder layers
`0 .. 49` are needed. The visual VAE is a passthrough with 3D kernels moved to MLX's channel-last
layout, and the audio VAE folds torch `weight_norm` pairs into plain kernels.
"""

import mlx.core as mx

from mflux.models.common.weights.mapping.weight_mapping import WeightTarget

TEXT_ENCODER_PREFIX = "model.language_model."
TEXT_ENCODER_NUM_LAYERS = 50
_TEXT_LAYER_TENSORS = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)
_AUDIO_TRANSPOSED_CONV_PREFIX = "decoder.ups."


class MiniMaxH3WeightMapping:
    @staticmethod
    def get_text_encoder_mapping() -> list[WeightTarget]:
        targets = [
            WeightTarget(to_pattern="embed_tokens.weight", from_pattern=[f"{TEXT_ENCODER_PREFIX}embed_tokens.weight"])
        ]
        targets.extend(
            WeightTarget(
                to_pattern=f"layers.{{layer}}.{name}",
                from_pattern=[f"{TEXT_ENCODER_PREFIX}layers.{{layer}}.{name}"],
            )
            for name in _TEXT_LAYER_TENSORS
        )
        return targets

    @staticmethod
    def text_encoder_key_is_needed(key: str, num_layers: int = TEXT_ENCODER_NUM_LAYERS) -> bool:
        """Whether a checkpoint key feeds the truncated conditioner (embeddings and layers below `num_layers`)."""
        if not key.startswith(TEXT_ENCODER_PREFIX):
            return False
        rest = key[len(TEXT_ENCODER_PREFIX) :]
        if rest == "embed_tokens.weight":
            return True
        if rest.startswith("layers."):
            return int(rest.split(".")[1]) < num_layers
        return False

    @staticmethod
    def video_vae_transform(tensor: mx.array) -> mx.array:
        """torch `(out, in, kT, kH, kW)` 3D kernels to MLX `(out, kT, kH, kW, in)`; everything else untouched."""
        return tensor.transpose(0, 2, 3, 4, 1) if tensor.ndim == 5 else tensor

    @staticmethod
    def fold_weight_norm(weight_g: mx.array, weight_v: mx.array) -> mx.array:
        norm = mx.sqrt(
            mx.sum(mx.square(weight_v.astype(mx.float32)), axis=tuple(range(1, weight_v.ndim)), keepdims=True)
        )
        return (weight_g.astype(mx.float32) * weight_v.astype(mx.float32) / norm).astype(weight_v.dtype)

    @staticmethod
    def convert_audio_vae_state(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Fold `weight_g` / `weight_v` pairs and move 1D kernels to MLX's `(out, k, in)` layout."""
        converted: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key.endswith(".weight_g"):
                continue
            if key.endswith(".weight_v"):
                base = key[: -len(".weight_v")]
                value = MiniMaxH3WeightMapping.fold_weight_norm(weights[base + ".weight_g"], value)
                key = base + ".weight"
            if value.ndim == 3 and key.endswith(".weight") and "filter" not in key:
                # torch Conv1d (out, in, k) -> (out, k, in); ConvTranspose1d (in, out, k) -> (out, k, in).
                value = (
                    value.transpose(1, 2, 0)
                    if key.startswith(_AUDIO_TRANSPOSED_CONV_PREFIX)
                    else value.transpose(0, 2, 1)
                )
            converted[key] = value
        return converted
