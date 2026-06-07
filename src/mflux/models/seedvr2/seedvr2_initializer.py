from pathlib import Path

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.seedvr2.model.seedvr2_transformer.transformer import SeedVR2Transformer
from mflux.models.seedvr2.model.seedvr2_vae.vae import SeedVR2VAE
from mflux.models.seedvr2.weights.seedvr2_weight_definition import SeedVR2WeightDefinition


class SeedVR2Initializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None = None,
        model_path: str | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        runtime_config = SeedVR2Initializer._model_config_for_source(model_config, path)
        root_path = SeedVR2Initializer._resolve_weight_root(path, runtime_config)
        weight_definition = SeedVR2WeightDefinition.resolve(runtime_config, root_path=root_path)
        SeedVR2Initializer._init_config(model, runtime_config)
        weights = SeedVR2Initializer._load_weights(root_path, weight_definition)
        SeedVR2Initializer._init_text_embedding(model, weights)
        SeedVR2Initializer._init_models(model, runtime_config)
        SeedVR2Initializer._apply_weights(model, weights, quantize, weight_definition)

    @staticmethod
    def _model_config_for_source(model_config: ModelConfig, source: str) -> ModelConfig:
        if source == model_config.model_name:
            return model_config
        return ModelConfig(
            priority=model_config.priority,
            aliases=model_config.aliases,
            model_name=source,
            base_model=model_config.base_model,
            controlnet_model=model_config.controlnet_model,
            custom_transformer_model=model_config.custom_transformer_model,
            num_train_steps=model_config.num_train_steps,
            max_sequence_length=model_config.max_sequence_length,
            supports_guidance=model_config.supports_guidance,
            requires_sigma_shift=model_config.requires_sigma_shift,
            transformer_overrides=model_config.transformer_overrides,
            text_encoder_overrides=model_config.text_encoder_overrides,
            inference_aliases=model_config.inference_aliases,
            sigma_base_shift=model_config.sigma_base_shift,
            sigma_max_shift=model_config.sigma_max_shift,
            sigma_base_seq_len=model_config.sigma_base_seq_len,
            sigma_max_seq_len=model_config.sigma_max_seq_len,
            sigma_shift_terminal=model_config.sigma_shift_terminal,
        )

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig(vae_encode_tiled=False, vae_decode_tiles_per_dim=0)

    @staticmethod
    def _resolve_weight_root(model_path: str, model_config: ModelConfig) -> Path:
        patterns = SeedVR2WeightDefinition.get_download_patterns_for_source(model_config, model_path)
        root_path = PathResolution.resolve(model_path, patterns=patterns)
        if root_path is None:
            raise ValueError("SeedVR2 requires a resolved model path.")
        return root_path

    @staticmethod
    def _load_weights(model_path: Path, weight_definition) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=weight_definition,
            model_path=str(model_path),
        )

    @staticmethod
    def _init_text_embedding(model, weights: LoadedWeights) -> None:
        embedding_component = weights.components.get("text_embedding")
        if isinstance(embedding_component, dict):
            model.text_embedding = embedding_component.get("embedding")
        else:
            model.text_embedding = None

    @staticmethod
    def _init_models(model, model_config: ModelConfig) -> None:
        model.vae = SeedVR2VAE()
        model.transformer = SeedVR2Transformer(**(model_config.transformer_overrides or {}))

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None, weight_definition) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=weight_definition,
            models={
                "transformer": model.transformer,
                "vae": model.vae,
            },
        )
