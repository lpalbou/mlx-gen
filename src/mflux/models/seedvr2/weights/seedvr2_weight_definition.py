from pathlib import Path
from typing import List

import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.seedvr2.weights.seedvr2_weight_mapping import SeedVR2WeightMapping


class SeedVR2WeightDefinition3B:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir=".",
                num_blocks=32,
                loading_mode="mlx_native",
                mapping_getter=lambda: SeedVR2WeightMapping.get_transformer_mapping(num_blocks=32),
                weight_files=["seedvr2_ema_3b_fp16.safetensors"],
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir=".",
                num_blocks=4,
                loading_mode="mlx_native",
                mapping_getter=SeedVR2WeightMapping.get_vae_mapping,
                weight_files=["ema_vae_fp16.safetensors"],
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "seedvr2_ema_3b_fp16.safetensors",
            "ema_vae_fp16.safetensors",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return SeedVR2WeightDefinition.quantization_predicate(path, module)


class SeedVR2WeightDefinition3BOfficial:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir=".",
                num_blocks=32,
                loading_mode="torch_checkpoint",
                precision=mx.float16,
                mapping_getter=lambda: SeedVR2WeightMapping.get_transformer_mapping(num_blocks=32),
                weight_files=["seedvr2_ema_3b.pth"],
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir=".",
                num_blocks=4,
                loading_mode="torch_checkpoint",
                precision=mx.float16,
                mapping_getter=SeedVR2WeightMapping.get_vae_mapping,
                weight_files=["ema_vae.pth"],
            ),
            ComponentDefinition(
                name="text_embedding",
                hf_subdir=".",
                loading_mode="torch_tensor",
                mapping_getter=None,
                skip_quantization=True,
                weight_files=["pos_emb.pt"],
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "seedvr2_ema_3b.pth",
            "ema_vae.pth",
            "pos_emb.pt",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return SeedVR2WeightDefinition.quantization_predicate(path, module)


class SeedVR2WeightDefinition3BPrepared:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                num_blocks=32,
                loading_mode="mlx_native",
                mapping_getter=lambda: SeedVR2WeightMapping.get_transformer_mapping(num_blocks=32),
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                num_blocks=4,
                loading_mode="mlx_native",
                mapping_getter=SeedVR2WeightMapping.get_vae_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "transformer/*.safetensors",
            "transformer/model.safetensors.index.json",
            "vae/*.safetensors",
            "vae/model.safetensors.index.json",
            "README.md",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return SeedVR2WeightDefinition.quantization_predicate(path, module)


class SeedVR2WeightDefinition7B:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir=".",
                num_blocks=36,
                loading_mode="mlx_native",
                mapping_getter=lambda: SeedVR2WeightMapping.get_transformer_mapping(num_blocks=36),
                weight_files=["seedvr2_ema_7b_fp16.safetensors"],
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir=".",
                num_blocks=4,
                loading_mode="mlx_native",
                mapping_getter=SeedVR2WeightMapping.get_vae_mapping,
                weight_files=["ema_vae_fp16.safetensors"],
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "seedvr2_ema_7b_fp16.safetensors",
            "ema_vae_fp16.safetensors",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return SeedVR2WeightDefinition.quantization_predicate(path, module)


class SeedVR2WeightDefinition7BOfficial:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir=".",
                num_blocks=36,
                loading_mode="torch_checkpoint",
                precision=mx.float16,
                mapping_getter=lambda: SeedVR2WeightMapping.get_transformer_mapping(num_blocks=36),
                weight_files=["seedvr2_ema_7b.pth"],
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir=".",
                num_blocks=4,
                loading_mode="torch_checkpoint",
                precision=mx.float16,
                mapping_getter=SeedVR2WeightMapping.get_vae_mapping,
                weight_files=["ema_vae.pth"],
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "seedvr2_ema_7b.pth",
            "ema_vae.pth",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return SeedVR2WeightDefinition.quantization_predicate(path, module)


class SeedVR2WeightDefinition7BPrepared:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                num_blocks=36,
                loading_mode="mlx_native",
                mapping_getter=lambda: SeedVR2WeightMapping.get_transformer_mapping(num_blocks=36),
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                num_blocks=4,
                loading_mode="mlx_native",
                mapping_getter=SeedVR2WeightMapping.get_vae_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return SeedVR2WeightDefinition3BPrepared.get_download_patterns()

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return SeedVR2WeightDefinition.quantization_predicate(path, module)


class SeedVR2WeightDefinition:
    @staticmethod
    def resolve(model_config, root_path: Path | str | None = None):
        aliases = {a.lower() for a in getattr(model_config, "aliases", [])}
        if root_path is not None and (Path(root_path) / "transformer" / "model.safetensors.index.json").exists():
            if "seedvr2-7b" in aliases:
                return SeedVR2WeightDefinition7BPrepared
            return SeedVR2WeightDefinition3BPrepared
        if root_path is not None and (Path(root_path) / "seedvr2_ema_3b.pth").exists():
            return SeedVR2WeightDefinition3BOfficial
        if root_path is not None and (Path(root_path) / "seedvr2_ema_7b.pth").exists():
            return SeedVR2WeightDefinition7BOfficial
        if SeedVR2WeightDefinition._source_is_official_3b(getattr(model_config, "model_name", None)):
            return SeedVR2WeightDefinition3BOfficial
        if SeedVR2WeightDefinition._source_is_official_7b(getattr(model_config, "model_name", None)):
            return SeedVR2WeightDefinition7BOfficial

        if "seedvr2-7b" in aliases:
            return SeedVR2WeightDefinition7B
        return SeedVR2WeightDefinition3B

    @staticmethod
    def for_saving(model_config):
        aliases = {a.lower() for a in getattr(model_config, "aliases", [])}
        if "seedvr2-7b" in aliases:
            return SeedVR2WeightDefinition7BPrepared
        return SeedVR2WeightDefinition3BPrepared

    @staticmethod
    def get_download_patterns_for_source(model_config, source: str | None) -> List[str]:
        aliases = {a.lower() for a in getattr(model_config, "aliases", [])}
        if SeedVR2WeightDefinition._source_is_official_3b(source):
            return SeedVR2WeightDefinition3BOfficial.get_download_patterns()
        if SeedVR2WeightDefinition._source_is_official_7b(source):
            return SeedVR2WeightDefinition7BOfficial.get_download_patterns()

        if source is not None:
            local_path = Path(source).expanduser()
            if local_path.is_dir():
                return SeedVR2WeightDefinition.resolve(model_config, root_path=local_path).get_download_patterns()

        normalized = (source or "").lower()
        if normalized == "numz/seedvr2_comfyui":
            if "seedvr2-7b" in aliases:
                return SeedVR2WeightDefinition7B.get_download_patterns()
            return SeedVR2WeightDefinition3B.get_download_patterns()
        if "abstractframework/seedvr2" in normalized or normalized.endswith("-4bit") or normalized.endswith("-8bit"):
            if "seedvr2-7b" in normalized or "seedvr2-7b" in aliases:
                return SeedVR2WeightDefinition7BPrepared.get_download_patterns()
            return SeedVR2WeightDefinition3BPrepared.get_download_patterns()

        return SeedVR2WeightDefinition.resolve(model_config).get_download_patterns()

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return SeedVR2WeightDefinition3BOfficial.get_components()

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return SeedVR2WeightDefinition3BOfficial.get_download_patterns()

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            return False

        if not hasattr(module, "to_quantized"):
            return False

        if isinstance(module, nn.Linear):
            if hasattr(module, "weight") and module.weight.shape[-1] % 64 != 0:
                return False

        return True

    @staticmethod
    def _source_is_official_3b(source: str | None) -> bool:
        if source is None:
            return False
        normalized = source.lower()
        return normalized in {
            "bytedance-seed/seedvr2-3b",
            "bytedance-seed/seedvr2_3b",
        }

    @staticmethod
    def _source_is_official_7b(source: str | None) -> bool:
        if source is None:
            return False
        normalized = source.lower()
        return normalized in {
            "bytedance-seed/seedvr2-7b",
            "bytedance-seed/seedvr2_7b",
        }
