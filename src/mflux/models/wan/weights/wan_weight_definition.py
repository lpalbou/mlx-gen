from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.wan.weights.wan_weight_mapping import WanWeightMapping


class WanWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                num_layers=30,
                loading_mode="multi_glob",
                precision=ModelConfig.precision,
                mapping_getter=WanWeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                loading_mode="single",
                precision=ModelConfig.precision,
                mapping_getter=WanWeightMapping.get_vae_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="wan",
                hf_subdir="tokenizer",
                tokenizer_class="T5TokenizerFast",
                max_length=512,
                padding="max_length",
                add_special_tokens=True,
                download_patterns=["tokenizer/*"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "model_index.json",
            "scheduler/*.json",
            "tokenizer/*",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "vae/*.safetensors",
            "vae/*.json",
        ]

    @staticmethod
    def quantization_predicate(path: str, module, bits: int | None = None) -> bool:
        del path, bits
        return hasattr(module, "to_quantized")
