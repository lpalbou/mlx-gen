from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.wan.weights.wan_weight_mapping import WanWeightMapping


class WanWeightDefinition:
    def __init__(self, model_config: ModelConfig | None = None):
        self.model_config = model_config or ModelConfig.wan2_2_ti2v_5b()

    @staticmethod
    def for_config(model_config: ModelConfig) -> "WanWeightDefinition":
        return WanWeightDefinition(model_config)

    @staticmethod
    def config(model_config: ModelConfig | None = None) -> "WanWeightDefinition":
        return WanWeightDefinition(model_config)

    @staticmethod
    def _transformer_config(model_config: ModelConfig) -> dict:
        return model_config.transformer_overrides

    @staticmethod
    def _num_layers(model_config: ModelConfig) -> int:
        return int(WanWeightDefinition._transformer_config(model_config).get("num_layers", 30))

    @staticmethod
    def _has_transformer_2(model_config: ModelConfig) -> bool:
        return bool(WanWeightDefinition._transformer_config(model_config).get("has_transformer_2", False))

    @staticmethod
    def _vae_variant(model_config: ModelConfig) -> str:
        return str(WanWeightDefinition._transformer_config(model_config).get("vae_variant", "wan22_ti2v"))

    def get_components(self=None) -> List[ComponentDefinition]:
        if isinstance(self, WanWeightDefinition):
            return self.components()
        return WanWeightDefinition().components()

    def components(self) -> List[ComponentDefinition]:
        num_layers = self._num_layers(self.model_config)
        vae_variant = self._vae_variant(self.model_config)
        components = [
            self._transformer_component("transformer", "transformer", num_layers),
        ]
        if self._has_transformer_2(self.model_config):
            components.append(self._transformer_component("transformer_2", "transformer_2", num_layers))
        components.append(
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                loading_mode="single",
                precision=ModelConfig.precision,
                mapping_getter=lambda: WanWeightMapping.get_vae_mapping(variant=vae_variant),
                skip_quantization=True,
            )
        )
        return components

    @staticmethod
    def _transformer_component(name: str, hf_subdir: str, num_layers: int) -> ComponentDefinition:
        return ComponentDefinition(
            name=name,
            hf_subdir=hf_subdir,
            num_layers=num_layers,
            loading_mode="multi_glob",
            precision=ModelConfig.precision,
            mapping_getter=lambda: WanWeightMapping.get_transformer_mapping(num_layers=num_layers),
        )

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

    def get_download_patterns(self=None) -> List[str]:
        if isinstance(self, WanWeightDefinition):
            return self.download_patterns()
        return WanWeightDefinition().download_patterns()

    def download_patterns(self) -> List[str]:
        patterns = [
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
        if self._has_transformer_2(self.model_config):
            patterns.extend(["transformer_2/*.safetensors", "transformer_2/*.json"])
        return patterns

    @staticmethod
    def quantization_predicate(path: str, module, bits: int | None = None) -> bool:
        if not hasattr(module, "to_quantized"):
            return False
        if bits == 8 and WanWeightDefinition._is_q8_sensitive_transformer_path(path):
            return False
        return True

    @staticmethod
    def _is_q8_sensitive_transformer_path(path: str) -> bool:
        return path == "proj_out" or path.startswith("condition_embedder.")
