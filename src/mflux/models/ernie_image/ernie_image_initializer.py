from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.ernie_image.model.ernie_image_transformer import ErnieImageTransformer2DModel
from mflux.models.ernie_image.model.ernie_image_vae import ErnieImageVAE
from mflux.models.ernie_image.model.mistral3_text_encoder import Mistral3TextEncoder
from mflux.models.ernie_image.weights.ernie_image_weight_definition import ErnieImageWeightDefinition


class ErnieImageInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        del lora_paths, lora_scales
        if quantize is not None:
            raise ValueError("ERNIE quantization is not enabled yet. Prepare or generate ERNIE in BF16 first.")
        path = model_path if model_path else model_config.model_name
        ErnieImageInitializer._init_config(model, model_config)
        weights = ErnieImageInitializer._load_weights(path)
        ErnieImageInitializer._init_tokenizers(model, path)
        ErnieImageInitializer._init_models(model)
        ErnieImageInitializer._apply_weights(model, weights, quantize)
        model.lora_paths = None
        model.lora_scales = None

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=ErnieImageWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=ErnieImageWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = ErnieImageVAE()
        model.transformer = ErnieImageTransformer2DModel()
        model.text_encoder = Mistral3TextEncoder()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=ErnieImageWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )
