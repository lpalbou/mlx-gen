import gc
import json
from pathlib import Path

import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.lora_compatibility import LoRACompatibility
from mflux.models.common.lora.mapping.lora_loader import LoRAApplicationError, LoRAApplicationResult, LoRALoader
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.wan.model.wan_transformer import WanTransformer
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.weights import WanWeightDefinition
from mflux.models.wan.weights.wan_lora_mapping import WanLoRAMapping


class WanInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_target_roles: list[str] | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        LoRACompatibility.validate_for_model_config(
            model_config=model_config,
            selected_model=path,
            lora_paths=lora_paths,
        )
        weight_definition = WanWeightDefinition.for_config(model_config)
        root_path = PathResolution.resolve(path=path, patterns=weight_definition.get_download_patterns())
        WanInitializer._validate_source_config(root_path, model_config)
        WanInitializer._init_config(model, model_config, root_path, weight_definition)
        WanInitializer._init_tokenizers(model, str(root_path), weight_definition)
        WanInitializer._init_models(model, model_config)
        WanInitializer._load_and_apply_weights(model, root_path, quantize, weight_definition)
        WanInitializer._apply_lora(
            model,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            lora_target_roles=lora_target_roles,
        )

    @staticmethod
    def _init_config(model, model_config: ModelConfig, root_path: Path, weight_definition: WanWeightDefinition) -> None:
        model.model_config = model_config
        model.root_path = root_path
        model.weight_definition = weight_definition
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.prompt_embed_cache = {}
        model.image_condition_cache = {}

    @staticmethod
    def _load_weights(model_path: str, weight_definition: WanWeightDefinition) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=weight_definition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str, weight_definition: WanWeightDefinition) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=weight_definition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model, model_config: ModelConfig) -> None:
        transformer_kwargs = WanInitializer._transformer_kwargs(model_config)
        model.transformer = WanTransformer(**transformer_kwargs)
        model.transformer_2 = (
            WanTransformer(**transformer_kwargs)
            if model_config.transformer_overrides.get("has_transformer_2", False)
            else None
        )
        model.vae = Wan2_2_VAE(**WanInitializer._vae_kwargs(model_config))

    @staticmethod
    def _apply_weights(
        model,
        weights: LoadedWeights,
        quantize: int | None,
        weight_definition: WanWeightDefinition,
    ) -> None:
        models = {
            "transformer": model.transformer,
            "vae": model.vae,
        }
        if model.transformer_2 is not None:
            models["transformer_2"] = model.transformer_2
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=weight_definition,
            models=models,
        )

    @staticmethod
    def _load_and_apply_weights(
        model,
        root_path: Path,
        quantize: int | None,
        weight_definition: WanWeightDefinition,
    ) -> None:
        bits = None
        bits_resolved = False

        for component in weight_definition.get_components():
            component_model = getattr(model, component.model_attr or component.name, None)
            if component_model is None:
                continue

            component_weights, q_level, version = WeightLoader._load_component(
                root_path=root_path,
                component=component,
            )
            WanInitializer._normalize_runtime_sensitive_q8_paths(
                component_name=component.name,
                component_weights=component_weights,
                q_level=q_level,
            )
            WanInitializer._validate_component_quantization_layout(component.name, component_weights, q_level)
            loaded_weights = LoadedWeights(
                components={component.name: component_weights},
                meta_data=MetaData(quantization_level=q_level, mflux_version=version),
            )
            component_bits = WeightApplier.apply_and_quantize_single(
                weights=loaded_weights,
                model=component_model,
                component=component,
                quantize_arg=quantize,
                quantization_predicate=weight_definition.quantization_predicate,
            )
            mismatch_error = None
            if component.skip_quantization and component_bits is None:
                pass
            elif not bits_resolved:
                bits = component_bits
                bits_resolved = True
            elif component_bits != bits:
                mismatch_error = ValueError(
                    "Wan component quantization mismatch: "
                    f"{component.name} resolved to {component_bits}, but earlier components resolved to {bits}."
                )
            del loaded_weights
            del component_weights
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
            if mismatch_error is not None:
                raise mismatch_error

        model.bits = bits

    @staticmethod
    def _apply_lora(
        model,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
        lora_target_roles: list[str] | None,
    ) -> None:
        resolved_roles = WanInitializer._resolve_lora_roles(
            model,
            lora_paths=lora_paths,
            lora_target_roles=lora_target_roles,
        )
        if not lora_paths:
            model.lora_application_result = LoRAApplicationResult(resolved_paths=[], resolved_scales=[], reports=())
            model.lora_application_reports = ()
            model.lora_paths = []
            model.lora_scales = []
            model.lora_target_roles = []
            return

        if lora_scales is not None and len(lora_scales) != len(lora_paths):
            raise LoRAApplicationError(
                f"Number of LoRA scales ({len(lora_scales)}) must match number of LoRA files ({len(lora_paths)})."
            )

        results: list[LoRAApplicationResult] = []
        for index, lora_path in enumerate(lora_paths):
            role = resolved_roles[index]
            transformer = WanInitializer._transformer_for_role(model, role)
            result = LoRALoader.load_and_apply_lora_detailed(
                lora_mapping=WanLoRAMapping.get_mapping(),
                transformer=transformer,
                lora_paths=[lora_path],
                lora_scales=None if lora_scales is None else [lora_scales[index]],
                role=role,
                state_dict_transform=WanInitializer._transform_wan_lora_state_dict,
            )
            results.append(result)

        model.lora_application_result = LoRAApplicationResult(
            resolved_paths=[path for result in results for path in result.resolved_paths],
            resolved_scales=[scale for result in results for scale in result.resolved_scales],
            reports=tuple(report for result in results for report in result.reports),
        )
        model.lora_application_reports = model.lora_application_result.reports
        model.lora_paths = model.lora_application_result.resolved_paths
        model.lora_scales = model.lora_application_result.resolved_scales
        model.lora_target_roles = resolved_roles

    @staticmethod
    def _resolve_lora_roles(
        model,
        *,
        lora_paths: list[str] | None,
        lora_target_roles: list[str] | None,
    ) -> list[str]:
        if not lora_paths:
            if lora_target_roles:
                raise LoRAApplicationError("--lora-target-roles requires --lora-paths.")
            return []

        if model.transformer_2 is None:
            if lora_target_roles is None:
                return ["transformer"] * len(lora_paths)
            if len(lora_target_roles) != len(lora_paths):
                raise LoRAApplicationError(
                    "--lora-target-roles must provide one role per LoRA file for Wan generation."
                )
            invalid = [role for role in lora_target_roles if role != "transformer"]
            if invalid:
                raise LoRAApplicationError(
                    "Wan TI2V-5B uses one transformer; valid --lora-target-roles value is only 'transformer'."
                )
            return list(lora_target_roles)

        if lora_target_roles is None:
            raise LoRAApplicationError(
                "Wan A14B LoRAs require explicit --lora-target-roles so MLX-Gen knows whether each file "
                "targets the high-noise or low-noise denoiser."
            )
        if len(lora_target_roles) != len(lora_paths):
            raise LoRAApplicationError("--lora-target-roles must provide one role per LoRA file for Wan generation.")

        valid_roles = {"high_noise_transformer", "low_noise_transformer"}
        invalid = [role for role in lora_target_roles if role not in valid_roles]
        if invalid:
            raise LoRAApplicationError(
                "Wan A14B valid --lora-target-roles values are 'high_noise_transformer' and "
                "'low_noise_transformer'."
            )
        return list(lora_target_roles)

    @staticmethod
    def _transformer_for_role(model, role: str) -> WanTransformer:
        if role == "transformer":
            return model.transformer
        if role == "high_noise_transformer":
            return model.transformer
        if role == "low_noise_transformer":
            if model.transformer_2 is None:
                raise LoRAApplicationError("Selected Wan model does not expose a low-noise transformer.")
            return model.transformer_2
        raise LoRAApplicationError(f"Unsupported Wan LoRA target role: {role}.")

    @staticmethod
    def _transform_wan_lora_state_dict(weights: dict[str, mx.array], transformer: WanTransformer) -> dict[str, mx.array]:
        if not WanInitializer._transformer_uses_image_projections(transformer):
            return weights
        if WanInitializer._state_dict_has_image_projection_lora(weights):
            return weights
        expanded = dict(weights)
        WanInitializer._expand_t2v_lora_for_i2v(expanded)
        return expanded

    @staticmethod
    def _transformer_uses_image_projections(transformer: WanTransformer) -> bool:
        if not transformer.blocks:
            return False
        return getattr(transformer.blocks[0].attn2, "add_k_proj", None) is not None

    @staticmethod
    def _state_dict_has_image_projection_lora(weights: dict[str, mx.array]) -> bool:
        image_markers = ("add_k_proj", "add_v_proj", "k_img", "v_img", "cross_attn_k_img", "cross_attn_v_img")
        return any(any(marker in key for marker in image_markers) for key in weights)

    @staticmethod
    def _expand_t2v_lora_for_i2v(weights: dict[str, mx.array]) -> None:
        reference_pairs = [
            (
                "transformer.blocks.",
                ".attn2.to_k.lora_A.weight",
                ".attn2.to_k.lora_B.weight",
                ".attn2.add_k_proj.lora_A.weight",
                ".attn2.add_k_proj.lora_B.weight",
                ".attn2.add_v_proj.lora_A.weight",
                ".attn2.add_v_proj.lora_B.weight",
            ),
            (
                "blocks.",
                ".attn2.to_k.lora_A.weight",
                ".attn2.to_k.lora_B.weight",
                ".attn2.add_k_proj.lora_A.weight",
                ".attn2.add_k_proj.lora_B.weight",
                ".attn2.add_v_proj.lora_A.weight",
                ".attn2.add_v_proj.lora_B.weight",
            ),
            (
                "diffusion_model.blocks.",
                ".cross_attn.k.lora_A.weight",
                ".cross_attn.k.lora_B.weight",
                ".cross_attn.k_img.lora_A.weight",
                ".cross_attn.k_img.lora_B.weight",
                ".cross_attn.v_img.lora_A.weight",
                ".cross_attn.v_img.lora_B.weight",
            ),
            (
                "diffusion_model.blocks.",
                ".cross_attn.k.lora_down.weight",
                ".cross_attn.k.lora_up.weight",
                ".cross_attn.k_img.lora_down.weight",
                ".cross_attn.k_img.lora_up.weight",
                ".cross_attn.v_img.lora_down.weight",
                ".cross_attn.v_img.lora_up.weight",
            ),
            (
                "lora_unet_blocks_",
                "_cross_attn_k.lora_A.weight",
                "_cross_attn_k.lora_B.weight",
                "_cross_attn_k_img.lora_A.weight",
                "_cross_attn_k_img.lora_B.weight",
                "_cross_attn_v_img.lora_A.weight",
                "_cross_attn_v_img.lora_B.weight",
            ),
            (
                "lora_unet_blocks_",
                "_cross_attn_k.lora_down.weight",
                "_cross_attn_k.lora_up.weight",
                "_cross_attn_k_img.lora_down.weight",
                "_cross_attn_k_img.lora_up.weight",
                "_cross_attn_v_img.lora_down.weight",
                "_cross_attn_v_img.lora_up.weight",
            ),
        ]

        for prefix, ref_a_suffix, ref_b_suffix, add_k_a_suffix, add_k_b_suffix, add_v_a_suffix, add_v_b_suffix in reference_pairs:
            WanInitializer._expand_projection_family(
                weights,
                prefix=prefix,
                ref_a_suffix=ref_a_suffix,
                ref_b_suffix=ref_b_suffix,
                add_k_a_suffix=add_k_a_suffix,
                add_k_b_suffix=add_k_b_suffix,
                add_v_a_suffix=add_v_a_suffix,
                add_v_b_suffix=add_v_b_suffix,
            )

    @staticmethod
    def _expand_projection_family(
        weights: dict[str, mx.array],
        *,
        prefix: str,
        ref_a_suffix: str,
        ref_b_suffix: str,
        add_k_a_suffix: str,
        add_k_b_suffix: str,
        add_v_a_suffix: str,
        add_v_b_suffix: str,
    ) -> None:
        prefixes = [key[: -len(ref_a_suffix)] for key in list(weights.keys()) if key.startswith(prefix) and key.endswith(ref_a_suffix)]
        for key_prefix in prefixes:
            ref_a = f"{key_prefix}{ref_a_suffix}"
            ref_b = f"{key_prefix}{ref_b_suffix}"
            if ref_a not in weights or ref_b not in weights:
                continue
            weights.setdefault(f"{key_prefix}{add_k_a_suffix}", mx.zeros_like(weights[ref_a]))
            weights.setdefault(f"{key_prefix}{add_k_b_suffix}", mx.zeros_like(weights[ref_b]))
            weights.setdefault(f"{key_prefix}{add_v_a_suffix}", mx.zeros_like(weights[ref_a]))
            weights.setdefault(f"{key_prefix}{add_v_b_suffix}", mx.zeros_like(weights[ref_b]))

    @staticmethod
    def _validate_component_quantization_layout(
        component_name: str, component_weights: dict, q_level: int | None
    ) -> None:
        if q_level != 8 or component_name not in ("transformer", "transformer_2"):
            return

        incompatible_paths = WanInitializer._q8_sensitive_weight_paths(component_weights)
        if not incompatible_paths:
            return

        examples = ", ".join(incompatible_paths[:3])
        if len(incompatible_paths) > 3:
            examples += ", ..."
        raise ValueError(
            "Wan q8 checkpoint uses an incompatible older quantization layout: "
            f"{component_name} stores q8 tensors for BF16-only paths ({examples}). "
            "Regenerate or re-download the checkpoint with the current Wan q8 policy; "
            "conditioning and output projection layers must remain BF16."
        )

    @staticmethod
    def _normalize_runtime_sensitive_q8_paths(
        component_name: str,
        component_weights: dict,
        q_level: int | None,
    ) -> None:
        if q_level != 8 or component_name not in ("transformer", "transformer_2"):
            return

        normalized_paths: list[str] = []
        WanInitializer._normalize_runtime_sensitive_q8_paths_recursive(
            node=component_weights,
            path="",
            normalized_paths=normalized_paths,
        )
        if normalized_paths:
            preview = ", ".join(normalized_paths[:3])
            if len(normalized_paths) > 3:
                preview += ", ..."
            print(
                "⚠️  Normalizing Wan q8 runtime-sensitive paths to BF16 at load: "
                f"{preview}"
            )

    @staticmethod
    def _normalize_runtime_sensitive_q8_paths_recursive(
        node,
        *,
        path: str,
        normalized_paths: list[str],
    ) -> None:
        if isinstance(node, list):
            for index, value in enumerate(node):
                next_path = f"{path}.{index}" if path else str(index)
                WanInitializer._normalize_runtime_sensitive_q8_paths_recursive(
                    value,
                    path=next_path,
                    normalized_paths=normalized_paths,
                )
            return

        if not isinstance(node, dict):
            return

        if WanInitializer._is_quantized_linear_state(node) and WanInitializer._is_runtime_sensitive_q8_path(path):
            node["weight"] = WanInitializer._dequantized_linear_weight(node)
            node.pop("scales", None)
            node.pop("biases", None)
            normalized_paths.append(path)
            return

        for key, value in node.items():
            next_path = f"{path}.{key}" if path else str(key)
            WanInitializer._normalize_runtime_sensitive_q8_paths_recursive(
                value,
                path=next_path,
                normalized_paths=normalized_paths,
            )

    @staticmethod
    def _is_quantized_linear_state(node: dict) -> bool:
        return {
            "weight",
            "scales",
            "biases",
        }.issubset(node.keys())

    @staticmethod
    def _dequantized_linear_weight(node: dict) -> mx.array:
        bits = 8
        input_dims = int(node["weight"].shape[1]) * (32 // bits)
        scale_columns = int(node["scales"].shape[1])
        if scale_columns <= 0 or input_dims % scale_columns != 0:
            raise ValueError(
                "Cannot infer Wan q8 group size for runtime normalization: "
                f"weight={tuple(node['weight'].shape)}, scales={tuple(node['scales'].shape)}."
            )
        group_size = input_dims // scale_columns
        return mx.dequantize(
            node["weight"],
            node["scales"],
            node["biases"],
            group_size=group_size,
            bits=bits,
        ).astype(ModelConfig.precision)

    @staticmethod
    def _is_runtime_sensitive_q8_path(path: str) -> bool:
        return path.endswith(
            (
                ".attn1.to_q",
                ".attn1.to_k",
                ".attn1.to_v",
                ".attn1.to_out.0",
                ".attn2.to_q",
                ".attn2.to_k",
                ".attn2.to_v",
                ".attn2.to_out.0",
                ".attn2.add_k_proj",
                ".attn2.add_v_proj",
                ".ffn.net.0",
                ".ffn.net.1",
            )
        )

    @staticmethod
    def _q8_sensitive_weight_paths(weights: dict, prefix: str = "") -> list[str]:
        paths = []
        for key, value in weights.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                if ("scales" in value or "biases" in value) and WanWeightDefinition._is_q8_sensitive_transformer_path(
                    path
                ):
                    paths.append(path)
                paths.extend(WanInitializer._q8_sensitive_weight_paths(value, path))
        return paths

    @staticmethod
    def _transformer_kwargs(model_config: ModelConfig) -> dict:
        allowed = {
            "patch_size",
            "num_attention_heads",
            "attention_head_dim",
            "in_channels",
            "out_channels",
            "text_dim",
            "freq_dim",
            "ffn_dim",
            "num_layers",
            "cross_attn_norm",
            "eps",
            "added_kv_proj_dim",
            "rope_max_seq_len",
        }
        kwargs = {key: value for key, value in model_config.transformer_overrides.items() if key in allowed}
        if "patch_size" in kwargs:
            kwargs["patch_size"] = tuple(kwargs["patch_size"])
        return kwargs

    @staticmethod
    def _vae_kwargs(model_config: ModelConfig) -> dict:
        return dict(model_config.transformer_overrides.get("vae_config", {}))

    @staticmethod
    def _validate_source_config(root_path: Path | None, model_config: ModelConfig) -> None:
        if root_path is None:
            return

        overrides = model_config.transformer_overrides
        model_index = WanInitializer._read_json(root_path / "model_index.json")
        if model_index is not None:
            WanInitializer._validate_model_index(root_path, model_index, overrides)

        transformer_config = WanInitializer._read_json(root_path / "transformer" / "config.json")
        if transformer_config is not None:
            WanInitializer._validate_transformer_config(root_path, transformer_config, overrides, "transformer")

        transformer_2_config = WanInitializer._read_json(root_path / "transformer_2" / "config.json")
        has_transformer_2 = bool(overrides.get("has_transformer_2", False))
        if transformer_2_config is not None and not has_transformer_2:
            raise ValueError(
                "Wan source/config mismatch: "
                f"{root_path} contains transformer_2/config.json, but {model_config.model_name} is configured "
                "as a single-transformer Wan model."
            )
        if transformer_2_config is not None:
            WanInitializer._validate_transformer_config(root_path, transformer_2_config, overrides, "transformer_2")

        vae_config = WanInitializer._read_json(root_path / "vae" / "config.json")
        if vae_config is not None:
            WanInitializer._validate_vae_config(root_path, vae_config, overrides)

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.exists():
            return None
        with path.open("rt") as json_file:
            return json.load(json_file)

    @staticmethod
    def _validate_model_index(root_path: Path, model_index: dict, overrides: dict) -> None:
        expected_has_transformer_2 = bool(overrides.get("has_transformer_2", False))
        actual_transformer_2 = model_index.get("transformer_2")
        actual_has_transformer_2 = isinstance(actual_transformer_2, list) and any(
            value is not None for value in actual_transformer_2
        )
        if actual_transformer_2 is not None and actual_has_transformer_2 != expected_has_transformer_2:
            WanInitializer._raise_source_mismatch(
                root_path=root_path,
                key="model_index.transformer_2",
                actual=actual_transformer_2,
                expected="present" if expected_has_transformer_2 else "absent",
            )

        for key in ("boundary_ratio", "expand_timesteps"):
            if key in model_index and key in overrides and model_index[key] != overrides[key]:
                WanInitializer._raise_source_mismatch(
                    root_path=root_path,
                    key=f"model_index.{key}",
                    actual=model_index[key],
                    expected=overrides[key],
                )

    @staticmethod
    def _validate_transformer_config(
        root_path: Path, transformer_config: dict, overrides: dict, component: str
    ) -> None:
        for key in ("in_channels", "out_channels", "num_layers", "num_attention_heads", "ffn_dim"):
            WanInitializer._validate_config_key(root_path, transformer_config, overrides, f"{component}.{key}", key)
        WanInitializer._validate_config_key(
            root_path, transformer_config, overrides, f"{component}.patch_size", "patch_size"
        )

    @staticmethod
    def _validate_vae_config(root_path: Path, vae_config: dict, overrides: dict) -> None:
        expected_vae_config = overrides.get("vae_config", {})
        for key in (
            "base_dim",
            "decoder_base_dim",
            "z_dim",
            "in_channels",
            "out_channels",
            "patch_size",
            "scale_factor_spatial",
            "scale_factor_temporal",
            "is_residual",
        ):
            WanInitializer._validate_config_key(root_path, vae_config, expected_vae_config, f"vae.{key}", key)

    @staticmethod
    def _validate_config_key(root_path: Path, source: dict, expected_source: dict, label: str, key: str) -> None:
        if key not in source or key not in expected_source:
            return
        if source[key] != expected_source[key]:
            WanInitializer._raise_source_mismatch(
                root_path=root_path,
                key=label,
                actual=source[key],
                expected=expected_source[key],
            )

    @staticmethod
    def _raise_source_mismatch(root_path: Path, key: str, actual, expected) -> None:
        raise ValueError(
            "Wan source/config mismatch: "
            f"{root_path} has {key}={actual!r}, but the selected Wan runtime expects {expected!r}. "
            "Pass the exact Wan model/config that matches this checkpoint; MLX-Gen will not fall back silently."
        )
