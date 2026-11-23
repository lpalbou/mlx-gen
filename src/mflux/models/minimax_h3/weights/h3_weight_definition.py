from typing import List

import mlx.core as mx

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.minimax_h3.weights.h3_weight_mapping import (
    TEXT_ENCODER_NUM_LAYERS,
    TEXT_ENCODER_PREFIX,
    VISION_NUM_BLOCKS,
    VISION_PREFIX,
    MiniMaxH3WeightMapping,
)


class MiniMaxH3WeightDefinition:
    """Components of `MiniMaxAI/MiniMax-H3`: Qwen3-VL conditioner (first 50 layers), the omni
    transformer, the visual VAE and the audio VAE."""

    # `_keep_in_fp32_modules` of the diffusers transformer: the input/output heads and the timestep MLP.
    TRANSFORMER_FP32_PREFIXES = ("proj_in.", "audio_proj_in.", "time_embedder.", "proj_out.", "audio_proj_out.")
    # Left as released at every width. The final modulation is 29M parameters and scales the whole output; the
    # Qwen3-VL vision tower is 0.6B, and quantizing it at any width flips a few very-high-norm image tokens.
    UNQUANTIZED_PREFIXES = ("norm_out.", "visual.")
    # The two policies with measurements behind them (docs/quantization.md, "MiniMax-H3 q8 And Mixed q4").
    # 8 is 8-bit throughout. 4 is the low-memory mix: a 4-bit conditioner and a 6-bit transformer, because no
    # transformer family tolerates 4-bit better than another, so there is nothing a 4/8 mix could protect.
    QUANTIZATION_BITS = (4, 8)
    TRANSFORMER_MIN_BITS = 6
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
                num_blocks=VISION_NUM_BLOCKS,
                mapping_getter=MiniMaxH3WeightMapping.get_text_encoder_mapping,
                weight_prefix_filters=[TEXT_ENCODER_PREFIX, VISION_PREFIX],
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
        if f"{path}.".startswith(MiniMaxH3WeightDefinition.UNQUANTIZED_PREFIXES):
            return False
        if bits is None:
            return True
        return {"bits": MiniMaxH3WeightDefinition.module_bits(path, bits)}

    @staticmethod
    def module_bits(path: str, bits: int) -> int:
        # `path` is a module path inside its component: `language_model...` in the conditioner, anything else
        # in the transformer (AdaLN projections, text projection and token refiner included).
        if path.startswith("language_model"):
            return bits
        return max(bits, MiniMaxH3WeightDefinition.TRANSFORMER_MIN_BITS)

    @staticmethod
    def component_bits(component_name: str, bits: int) -> int:
        return MiniMaxH3WeightDefinition.module_bits(
            "language_model" if component_name == "text_encoder" else "transformer_blocks", bits
        )

    @staticmethod
    def require_supported_bits(bits: int | None) -> None:
        if bits is not None and bits not in MiniMaxH3WeightDefinition.QUANTIZATION_BITS:
            raise ValueError(
                f"MiniMax-H3 has two quantization policies, 8 and 4 (a 4-bit conditioner with a "
                f"{MiniMaxH3WeightDefinition.TRANSFORMER_MIN_BITS}-bit transformer); --quantize {bits} is not one "
                "of them."
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
        # What a run reads. Path resolution treats every entry as required, so only files that both the source
        # snapshot and every prepared package carry belong here.
        return [
            "model_index.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "vae/*.safetensors",
            "vae/*.json",
            "audio_vae/*.safetensors",
            "audio_vae/*.json",
        ]

    @staticmethod
    def get_optional_download_patterns() -> List[str]:
        # Fetched when present, never required: the license and its notice travel with the weights
        # (sections III.1 and III.4 of the MiniMax H3 Community License), the rest documents the release.
        return [
            "LICENSE",
            "NOTICE",
            "README.md",
            "modular_model_index.json",
            "scheduler/*.json",
            "audio_scheduler/*.json",
        ]
