from pathlib import Path

from mlx.utils import tree_flatten

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.download_policy import DownloadRequiredError
from mflux.models.common.resolution.config_resolution import ConfigResolution
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
        model.seedvr2_resident_weight_bytes = SeedVR2WeightDefinition.estimate_resident_weight_bytes(
            root_path=root_path,
            weight_definition=weight_definition,
        )
        # The weights are bound lazily and read by the first restoration; the run-time memory
        # checks count them as resident only from then on.
        model.seedvr2_weights_read = False
        weights = SeedVR2Initializer._load_weights(root_path, weight_definition)
        SeedVR2Initializer._init_text_embedding(model, weights)
        SeedVR2Initializer._init_models(model, runtime_config)
        SeedVR2Initializer._record_checkpoint_provenance(model, weight_definition, root_path)
        SeedVR2Initializer._assert_weight_coverage(model, weights)
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
        model.seedvr2_checkpoint_variant = None
        model.seedvr2_source_layout = None

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
        model.vae.set_causal_slicing(split_size=4)
        model.transformer = SeedVR2Transformer(**(model_config.transformer_overrides or {}))

    @staticmethod
    def _record_checkpoint_provenance(model, weight_definition, root_path: Path) -> None:
        name = getattr(weight_definition, "__name__", "")
        if name == "SeedVR2WeightDefinition3BOfficial":
            model.seedvr2_checkpoint_variant = "3b"
            model.seedvr2_source_layout = "official"
            return
        if name == "SeedVR2WeightDefinition7BOfficial":
            model.seedvr2_checkpoint_variant = "7b"
            model.seedvr2_source_layout = "official"
            return
        if name == "SeedVR2WeightDefinition7BOfficialSharp":
            model.seedvr2_checkpoint_variant = "7b-sharp"
            model.seedvr2_source_layout = "official"
            return
        if name == "SeedVR2WeightDefinition3BPrepared":
            model.seedvr2_checkpoint_variant = "3b"
            model.seedvr2_source_layout = "prepared"
            return
        if name == "SeedVR2WeightDefinition7BPrepared":
            aliases = {alias.lower() for alias in getattr(model.model_config, "aliases", [])}
            model.seedvr2_checkpoint_variant = "7b-sharp" if "seedvr2-7b-sharp" in aliases else "7b"
            model.seedvr2_source_layout = "prepared"
            return

        aliases = {alias.lower() for alias in getattr(model.model_config, "aliases", [])}
        if (
            "seedvr2-7b-sharp" in aliases
            or "seedvr2_ema_7b_sharp" in ConfigResolution.identifying_name(str(root_path)).lower()
        ):
            model.seedvr2_checkpoint_variant = "7b-sharp"
        elif "seedvr2-7b" in aliases:
            model.seedvr2_checkpoint_variant = "7b"
        else:
            model.seedvr2_checkpoint_variant = "3b"
        model.seedvr2_source_layout = "mlx-native"

    @staticmethod
    def _assert_weight_coverage(model, weights: LoadedWeights) -> None:
        stored_q = weights.meta_data.quantization_level
        for component_name, component_model in (("transformer", model.transformer), ("vae", model.vae)):
            component_weights = weights.components.get(component_name)
            if component_weights is None:
                continue
            expected = {key for key, _ in tree_flatten(component_model.parameters())}
            provided = {
                key
                for key, _ in tree_flatten(component_weights)
                if not (stored_q is not None and (key.endswith(".scales") or key.endswith(".biases")))
            }
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            if missing or extra:
                missing_preview = ", ".join(missing[:5]) if missing else "none"
                extra_preview = ", ".join(extra[:5]) if extra else "none"
                raise ValueError(
                    f"SeedVR2 {component_name} weight coverage mismatch: "
                    f"missing={len(missing)} ({missing_preview}); "
                    f"extra={len(extra)} ({extra_preview})"
                )

    @staticmethod
    def estimate_resident_weight_bytes(model_config: ModelConfig, model_path: str | None = None) -> int:
        path = model_path if model_path else model_config.model_name
        runtime_config = SeedVR2Initializer._model_config_for_source(model_config, path)
        try:
            root_path = SeedVR2Initializer._resolve_weight_root(path, runtime_config)
            weight_definition = SeedVR2WeightDefinition.resolve(runtime_config, root_path=root_path)
            return SeedVR2WeightDefinition.estimate_resident_weight_bytes(
                root_path=root_path,
                weight_definition=weight_definition,
            )
        except (DownloadRequiredError, FileNotFoundError, ValueError, OSError):
            return SeedVR2Initializer._default_resident_weight_bytes(runtime_config)

    @staticmethod
    def _default_resident_weight_bytes(model_config: ModelConfig) -> int:
        aliases = {alias.lower() for alias in getattr(model_config, "aliases", [])}
        if "seedvr2-7b" in aliases or "seedvr2-7b-sharp" in aliases:
            return 34 * (1000**3)
        return 15 * (1000**3)

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
