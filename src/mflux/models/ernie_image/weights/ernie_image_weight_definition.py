from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.ernie_image.tokenizer import ErnieImageTokenizer
from mflux.models.ernie_image.weights.ernie_image_weight_mapping import ErnieImageWeightMapping


class ErnieImageWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=ErnieImageWeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                num_layers=36,
                precision=ModelConfig.precision,
                mapping_getter=ErnieImageWeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                num_layers=26,
                loading_mode="single",
                precision=ModelConfig.precision,
                mapping_getter=ErnieImageWeightMapping.get_text_encoder_mapping,
                weight_prefix_filters=["language_model.model."],
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="ernie",
                hf_subdir="tokenizer",
                tokenizer_class="AutoTokenizer",
                encoder_class=ErnieImageTokenizer,
                max_length=512,
                padding="longest",
                add_special_tokens=True,
                download_patterns=["tokenizer/*"],
            ),
            TokenizerDefinition(
                name="ernie_prompt_enhancer",
                hf_subdir="pe_tokenizer",
                tokenizer_class="AutoTokenizer",
                encoder_class=ErnieImageTokenizer,
                max_length=512,
                padding="longest",
                add_special_tokens=True,
                download_patterns=["pe_tokenizer/*"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "LICENSE",
            "README.md",
            "model_index.json",
            "scheduler/**",
            "tokenizer/**",
            "pe_tokenizer/**",
            "text_encoder/**",
            "transformer/**",
            "vae/**",
            "pe/**",
        ]

    @staticmethod
    def quantization_predicate(path: str, module, bits: int | None = None) -> bool:
        del path, module, bits
        return False
