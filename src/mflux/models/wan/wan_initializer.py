from pathlib import Path

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.wan.model.wan_transformer import WanTransformer
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.weights import WanWeightDefinition


class WanInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        root_path = PathResolution.resolve(path=path, patterns=WanWeightDefinition.get_download_patterns())
        WanInitializer._init_config(model, model_config, root_path)
        weights = WanInitializer._load_weights(str(root_path))
        WanInitializer._init_tokenizers(model, str(root_path))
        WanInitializer._init_models(model)
        WanInitializer._apply_weights(model, weights, quantize)

    @staticmethod
    def _init_config(model, model_config: ModelConfig, root_path: Path) -> None:
        model.model_config = model_config
        model.root_path = root_path
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=WanWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=WanWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.transformer = WanTransformer()
        model.vae = Wan2_2_VAE()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=WanWeightDefinition,
            models={
                "transformer": model.transformer,
                "vae": model.vae,
            },
        )
