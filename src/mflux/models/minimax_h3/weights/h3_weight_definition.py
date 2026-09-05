from typing import List

import mlx.core as mx

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.minimax_h3.weights.h3_weight_mapping import (
    TEXT_ENCODER_NUM_LAYERS,
    TEXT_ENCODER_PREFIX,
    MiniMaxH3WeightMapping,
)


class MiniMaxH3WeightDefinition:
    """Components of `MiniMaxAI/MiniMax-H3`: Qwen3-VL conditioner (first 50 layers), the omni
    transformer, the visual VAE and the audio VAE."""

    # `_keep_in_fp32_modules` of the diffusers transformer: the input/output heads and the timestep MLP.
    TRANSFORMER_FP32_PREFIXES = ("proj_in.", "audio_proj_in.", "time_embedder.", "proj_out.", "audio_proj_out.")
    # Quantization-sensitive paths that stay bf16 in a q8/q4 package: q8 on the AdaLN table projections turns the
    # real one-block output into noise (relative RMS error 1.17 for `adaln_proj`, 0.23 for `norm_out`), while every
    # other module family quantizes to ~7e-3. Measured 2026-09-04 against the diffusers fp32 reference.
    TRANSFORMER_QUANTIZATION_SENSITIVE_FRAGMENTS = (".adaln_proj.", "norm_out.")
    TOKENIZER_NAME = "minimax_h3"

    def __init__(self, model_config: ModelConfig | None = None):
        self.model_config = model_config or ModelConfig.minimax_h3()

    @staticmethod
    def for_config(model_config: ModelConfig) -> "MiniMaxH3WeightDefinition":
        return MiniMaxH3WeightDefinition(model_config)

    def get_components(self=None) -> List[ComponentDefinition]:
        if isinstance(self, MiniMaxH3WeightDefinition):
            return self.components()
        return MiniMaxH3WeightDefinition().components()

    def components(self) -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                loading_mode="multi_glob",
                precision=ModelConfig.precision,
                num_layers=TEXT_ENCODER_NUM_LAYERS,
                mapping_getter=MiniMaxH3WeightMapping.get_text_encoder_mapping,
                weight_prefix_filters=[TEXT_ENCODER_PREFIX],
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                loading_mode="multi_glob",
                precision=ModelConfig.precision,
                precision_override=MiniMaxH3WeightDefinition.transformer_precision_override,
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                loading_mode="multi_glob",
                precision=ModelConfig.precision,
                bulk_transform=MiniMaxH3WeightMapping.video_vae_transform,
                skip_quantization=True,
            ),
            ComponentDefinition(
                name="audio_vae",
                hf_subdir="audio_vae",
                loading_mode="single",
                # The reference decodes audio in fp32; bf16 leaves audible quantization noise.
                precision=mx.float32,
                skip_quantization=True,
            ),
        ]

    @staticmethod
    def transformer_precision_override(key: str) -> mx.Dtype | None:
        if MiniMaxH3WeightDefinition.is_transformer_fp32_path(key):
            return mx.float32
        return None

    @staticmethod
    def quantization_predicate(path: str, module, bits: int | None = None):
        if not hasattr(module, "to_quantized"):
            return False
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2 or weight.shape[-1] % 64:
            return False
        # The fp32 keep-set stays exact; it is a negligible share of the parameters anyway. Module paths carry no
        # trailing dot (`proj_out`), parameter keys do (`proj_out.weight`).
        if MiniMaxH3WeightDefinition.is_transformer_fp32_path(path):
            return False
        if MiniMaxH3WeightDefinition.is_transformer_quantization_sensitive_path(path):
            return False
        return True

    @staticmethod
    def is_transformer_quantization_sensitive_path(path: str) -> bool:
        dotted = f"{path}."
        return any(
            fragment in dotted for fragment in MiniMaxH3WeightDefinition.TRANSFORMER_QUANTIZATION_SENSITIVE_FRAGMENTS
        )

    @staticmethod
    def is_transformer_fp32_path(path: str) -> bool:
        return any(
            path == prefix[:-1] or path.startswith(prefix)
            for prefix in MiniMaxH3WeightDefinition.TRANSFORMER_FP32_PREFIXES
        )

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name=MiniMaxH3WeightDefinition.TOKENIZER_NAME,
                hf_subdir="tokenizer",
                tokenizer_class="Qwen2Tokenizer",
                encoder_class=LanguageTokenizer,
                # The prompt is fed verbatim: no chat template, no special tokens, no padding.
                max_length=8192,
                padding="longest",
                add_special_tokens=False,
                download_patterns=["tokenizer/*"],
            ),
        ]

    def get_download_patterns(self=None) -> List[str]:
        return [
            "model_index.json",
            "modular_model_index.json",
            "scheduler/*.json",
            "audio_scheduler/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "vae/*.safetensors",
            "vae/*.json",
            "audio_vae/*.safetensors",
            "audio_vae/*.json",
        ]
