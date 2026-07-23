import json
import logging
import pickle
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx.utils import tree_unflatten
from safetensors import safe_open

from mflux.cli.defaults.defaults import MFLUX_CACHE_DIR
from mflux.models.common.download_policy import (
    downloads_enabled,
    raise_direct_url_download_required,
    raise_download_required,
)
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_prefetcher import WeightPrefetcher
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.utils.runtime_memory import RuntimeMemory

if TYPE_CHECKING:
    from mflux.models.common.weights.loading.weight_definition import WeightDefinitionType

logger = logging.getLogger(__name__)


class WeightLoader:
    @staticmethod
    def load_single(
        component: ComponentDefinition,
        repo_id: str,
        file_pattern: str = "*.safetensors",
    ) -> LoadedWeights:
        # Bare Python-API loads never pass through the CLI's runtime-memory setup,
        # so the machine-derived MLX cache-limit default applies here (0094).
        RuntimeMemory.apply_default_cache_limit_once()
        # Deferred: huggingface_hub pulls httpx+rich (~0.5 s) and is only needed
        # once weights actually resolve, never on the import path (0088).
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError

        patterns = [file_pattern, "config.json"]
        try:
            root_path = Path(snapshot_download(repo_id=repo_id, allow_patterns=patterns, local_files_only=True))
        except LocalEntryNotFoundError:
            if not downloads_enabled():
                raise_download_required(repo_id)
            root_path = Path(snapshot_download(repo_id=repo_id, allow_patterns=patterns))
        weights, q_level, version = WeightLoader._load_component(root_path, component)
        return LoadedWeights(
            components={component.name: weights},
            meta_data=MetaData(quantization_level=q_level, mflux_version=version),
        )

    @staticmethod
    def load(
        weight_definition: "WeightDefinitionType",
        model_path: str | None = None,
    ) -> LoadedWeights:
        # Bare Python-API loads never pass through the CLI's runtime-memory setup,
        # so the machine-derived MLX cache-limit default applies here (0094).
        RuntimeMemory.apply_default_cache_limit_once()
        root_path = PathResolution.resolve(
            path=model_path,
            patterns=weight_definition.get_download_patterns(),
        )

        # 2. Load each component (with caching for shared sources)
        components = {}
        quantization_level = None
        mflux_version = None
        raw_weights_cache: dict[tuple, dict] = {}  # Cache by (path, loading_mode, weight_files)

        for component in weight_definition.get_components():
            weights, q_level, version = WeightLoader._load_component(root_path, component, raw_weights_cache)
            components[component.name] = weights

            # Track metadata from first component that has it
            if quantization_level is None and q_level is not None:
                quantization_level = q_level
                mflux_version = version

        return LoadedWeights(
            components=components,
            meta_data=MetaData(
                quantization_level=quantization_level,
                mflux_version=mflux_version,
            ),
        )

    @staticmethod
    def _load_component(
        root_path: Path | None,
        component: ComponentDefinition,
        raw_weights_cache: dict[tuple, dict] | None = None,
    ) -> tuple[dict, int | None, str | None]:
        # Handle direct URL downloads (e.g., Apple CDN for DepthPro)
        if component.download_url is not None:
            file_path = WeightLoader._download_from_url(component.download_url, component.name)
            raw_weights = WeightLoader._load_weights_file(file_path, component.loading_mode)
        else:
            if root_path is None:
                raise ValueError(f"No root_path and no download_url for component: {component.name}")
            component_path = root_path / component.hf_subdir

            # Try mflux saved format first
            weights, q_level, version = WeightLoader._try_load_mflux_format(component_path)
            if weights is not None:
                if component.skip_quantization:
                    q_level = None
                return weights, q_level, version

            # Check cache for shared loading (e.g., FIBO VLM decoder + visual from same source)
            cache_key = (str(component_path), component.loading_mode, tuple(component.weight_files or []))
            if raw_weights_cache is not None and cache_key in raw_weights_cache:
                raw_weights = raw_weights_cache[cache_key]
            else:
                # Fall back to HuggingFace format with mapping
                raw_weights = WeightLoader._load_safetensors(
                    component_path, component.loading_mode, component.weight_files
                )
                # Cache for potential reuse by other components
                if raw_weights_cache is not None:
                    raw_weights_cache[cache_key] = raw_weights

        # Apply prefix filtering if specified (e.g., filter "model.language_model" vs "model.visual")
        if component.weight_prefix_filters is not None:
            raw_weights = {
                k: v
                for k, v in raw_weights.items()
                if any(k.startswith(prefix) for prefix in component.weight_prefix_filters)
            }

        # Apply precision conversion if specified
        if component.precision is not None:
            raw_weights = WeightLoader._convert_precision(raw_weights, component.precision)

        # Passthrough mode: apply bulk transform and unflatten (no key mapping)
        if component.mapping_getter is None:
            if component.bulk_transform is not None:
                raw_weights = {k: component.bulk_transform(v) for k, v in raw_weights.items()}
            return tree_unflatten(list(raw_weights.items())), None, None

        # Standard mode: apply declarative weight mapping
        mapped_weights = WeightMapper.apply_mapping(
            hf_weights=raw_weights,
            mapping=component.mapping_getter(),
            num_blocks=component.num_blocks,
            num_layers=component.num_layers,
        )
        return mapped_weights, None, None

    @staticmethod
    def _try_load_mflux_format(path: Path) -> tuple[dict | None, int | None, str | None]:
        if not path.exists():
            return None, None, None

        shard_files = WeightLoader._mflux_shard_files(path)
        if not shard_files:
            return None, None, None

        metadata = WeightLoader._load_safetensors_metadata(shard_files[0])
        quantization_level_str = metadata.get("quantization_level")
        mflux_version = metadata.get("mflux_version")

        # If no mflux metadata, this isn't our format
        if quantization_level_str is None and mflux_version is None:
            return None, None, None

        # Convert quantization level from string to int
        if quantization_level_str in (None, "None", "null", ""):
            quantization_level = None
        else:
            quantization_level = int(quantization_level_str)

        # No prefetch here: prepared packages are written by ModelSaver in
        # module-tree order (tree_flatten), so first materialization walks the
        # file near-sequentially and OS readahead already runs at SSD speed.
        # Measured on the Klein 9B q8 package (page-cold, residency 0.000):
        # prefetch cost +1.7-2.1 s end-to-end with zero first-generate benefit
        # (0093 cycle-2 ruling). HF-repo paths below keep the prefetch — their
        # file layout diverges from apply/quantize order.
        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            all_weights.update(mx.load(str(shard)))

        unflattened = tree_unflatten(list(all_weights.items()))
        return unflattened, quantization_level, mflux_version

    @staticmethod
    def _load_safetensors_metadata(path: Path) -> dict[str, str]:
        with safe_open(str(path), framework="numpy") as handle:
            return dict(handle.metadata() or {})

    @staticmethod
    def _mflux_shard_files(path: Path) -> list[Path]:
        index_path = path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as file:
                index_data = json.load(file)
            weight_map = index_data.get("weight_map")
            if isinstance(weight_map, dict) and weight_map:
                return [
                    path / shard_name
                    for shard_name in sorted(set(weight_map.values()))
                    if isinstance(shard_name, str) and not shard_name.startswith("._")
                ]
        return sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))

    @staticmethod
    def _download_from_url(url: str, component_name: str) -> Path:
        cache_dir = MFLUX_CACHE_DIR / component_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Extract filename from URL
        filename = url.split("/")[-1]
        file_path = cache_dir / filename

        if not file_path.exists():
            if not downloads_enabled():
                raise_direct_url_download_required(component_name, url)
            logger.info(f"Downloading {component_name} weights from {url}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                logger.info(f"Downloaded to {file_path}")
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logger.error(f"Failed to download: {e}")
                logger.info(f"Please manually download from: {url}")
                raise FileNotFoundError(f"Model file not found at {file_path}") from e

        return file_path

    @staticmethod
    def _load_weights_file(file_path: Path, loading_mode: str) -> dict[str, mx.array]:
        WeightPrefetcher.prefetch([file_path])
        if loading_mode == "torch_checkpoint":
            return WeightLoader._load_torch_checkpoint(file_path)
        elif loading_mode in ("mlx_native", "single"):
            data = mx.load(str(file_path), return_metadata=True)
            return dict(data[0].items())
        else:
            raise ValueError(f"Unsupported loading mode for single file: {loading_mode}")

    @staticmethod
    def _load_torch_checkpoint(file_path: Path) -> dict[str, mx.array]:
        torch = WeightLoader._torch_module()
        try:
            pt_weights = torch.load(file_path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as exc:
            raise ValueError(
                f"Unsafe or unsupported Torch checkpoint at {file_path}. "
                "MLX-Gen only loads tensor/state-dict checkpoints with torch weights_only=True; "
                "convert legacy pickle checkpoints to safetensors or an MLX-Gen prepared package."
            ) from exc
        try:
            return WeightLoader._torch_object_to_mx_weights(pt_weights)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unsupported Torch checkpoint payload at {file_path}. "
                "MLX-Gen only loads plain tensors or tensor-only state dictionaries; "
                "use safetensors, an official supported checkpoint, or an MLX-Gen prepared package."
            ) from exc

    @staticmethod
    def _load_safetensors(path: Path, loading_mode: str, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if loading_mode == "mlx_native":
            return WeightLoader._load_mlx_native(path, weight_files)
        elif loading_mode == "torch_convert":
            return WeightLoader._load_torch_convert(path, weight_files)
        elif loading_mode == "multi_json":
            return WeightLoader._load_multi_json(path)
        elif loading_mode == "torch_bfloat16":
            return WeightLoader._load_torch_bfloat16(path)
        elif loading_mode == "single":
            return WeightLoader._load_single(path)
        elif loading_mode == "multi_glob":
            return WeightLoader._load_multi_glob(path)
        elif loading_mode == "torch_checkpoint":
            return WeightLoader._load_torch_checkpoint_directory(path, weight_files)
        elif loading_mode == "torch_tensor":
            return WeightLoader._load_torch_tensor_directory(path, weight_files)
        else:
            raise ValueError(f"Unknown loading mode: {loading_mode}")

    @staticmethod
    def _load_torch_checkpoint_directory(path: Path, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        shard_files = WeightLoader._resolve_weight_files(path, weight_files, "*.pth")
        WeightPrefetcher.prefetch(shard_files)
        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            all_weights.update(WeightLoader._load_torch_checkpoint(shard))
        return all_weights

    @staticmethod
    def _load_torch_tensor_directory(path: Path, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        shard_files = WeightLoader._resolve_weight_files(path, weight_files, "*.pt")
        if len(shard_files) != 1:
            raise ValueError(f"Expected exactly one tensor file in {path}, found {len(shard_files)}.")
        WeightPrefetcher.prefetch(shard_files)
        loaded = WeightLoader._load_torch_checkpoint(shard_files[0])
        if len(loaded) == 1:
            return loaded
        raise ValueError(f"Expected a tensor checkpoint in {shard_files[0]}, found {len(loaded)} tensors.")

    @staticmethod
    def _resolve_weight_files(path: Path, weight_files: list[str] | None, fallback_pattern: str) -> list[Path]:
        if weight_files:
            missing = [f for f in weight_files if not (path / f).exists()]
            if missing:
                raise FileNotFoundError(f"Missing specified weight files in {path}: {missing}")
            return [path / f for f in weight_files]

        shard_files = sorted(f for f in path.glob(fallback_pattern) if not f.name.startswith("._"))
        if not shard_files:
            raise FileNotFoundError(f"No {fallback_pattern} files found in {path}")
        return shard_files

    @staticmethod
    def _torch_object_to_mx_weights(obj) -> dict[str, mx.array]:
        torch = WeightLoader._torch_module()
        if isinstance(obj, torch.Tensor):
            return {"embedding": WeightLoader._torch_tensor_to_mx(obj)}

        if isinstance(obj, dict):
            for key in ("state_dict", "model", "module"):
                nested = obj.get(key)
                if isinstance(nested, dict):
                    obj = nested
                    break
            tensor_weights = {
                str(k): WeightLoader._torch_tensor_to_mx(v)
                for k, v in obj.items()
                if isinstance(v, torch.Tensor)
            }
            if not tensor_weights:
                raise ValueError("Torch checkpoint does not contain tensor weights.")
            non_tensor_keys = [str(k) for k, v in obj.items() if not isinstance(v, torch.Tensor)]
            if non_tensor_keys:
                preview = ", ".join(non_tensor_keys[:5])
                suffix = "..." if len(non_tensor_keys) > 5 else ""
                raise ValueError(f"Torch checkpoint contains non-tensor entries: {preview}{suffix}")
            return tensor_weights

        raise TypeError(f"Unsupported torch checkpoint object: {type(obj).__name__}")

    @staticmethod
    def _torch_tensor_to_mx(tensor) -> mx.array:
        torch = WeightLoader._torch_module()
        if tensor.dtype == torch.bfloat16:
            return mx.array(tensor.detach().cpu().to(torch.float32).numpy()).astype(mx.bfloat16)
        return mx.array(tensor.detach().cpu().numpy())

    @staticmethod
    def _load_mlx_native(path: Path, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if weight_files:
            # Load only specified files
            missing = [f for f in weight_files if not (path / f).exists()]
            if missing:
                raise FileNotFoundError(f"Missing specified weight files in {path}: {missing}")
            shard_files = [path / f for f in weight_files]
        else:
            # Fall back to loading all safetensors files
            shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
            if not shard_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")

        WeightPrefetcher.prefetch(shard_files)
        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            weights = mx.load(str(shard))
            all_weights.update(weights)

        return all_weights

    @staticmethod
    def _load_torch_convert(path: Path, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if weight_files:
            # Load only specified files
            missing = [f for f in weight_files if not (path / f).exists()]
            if missing:
                raise FileNotFoundError(f"Missing specified weight files in {path}: {missing}")
            shard_files = [path / f for f in weight_files]
        else:
            # Fall back to loading all safetensors files
            shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
            if not shard_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")

        WeightPrefetcher.prefetch(shard_files)
        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            torch_weights = WeightLoader._torch_load_file(str(shard))
            for key, tensor in torch_weights.items():
                all_weights[key] = WeightLoader._torch_tensor_to_mx(tensor)

        return all_weights

    @staticmethod
    def _load_multi_json(path: Path) -> dict[str, mx.array]:
        index_path = path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        # Group weights by file
        files_to_load: dict[str, list[str]] = {}
        for param_name, file_name in index["weight_map"].items():
            if file_name not in files_to_load:
                files_to_load[file_name] = []
            files_to_load[file_name].append(param_name)

        WeightPrefetcher.prefetch([path / file_name for file_name in files_to_load])
        all_weights: dict[str, mx.array] = {}
        for file_name, param_names in files_to_load.items():
            file_path = path / file_name

            # Use mx.load which handles bfloat16 natively
            file_weights = mx.load(str(file_path))

            for param_name in param_names:
                if param_name in file_weights:
                    all_weights[param_name] = file_weights[param_name]

        return all_weights

    @staticmethod
    def _load_torch_bfloat16(path: Path) -> dict[str, mx.array]:
        index_path = path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        weight_files = sorted(set(index["weight_map"].values()))

        WeightPrefetcher.prefetch([path / wf for wf in weight_files])
        all_weights: dict[str, mx.array] = {}
        for wf in weight_files:
            file_path = path / wf
            data = WeightLoader._torch_load_file(str(file_path))
            for k, v in data.items():
                all_weights[k] = WeightLoader._torch_tensor_to_mx(v)

        return all_weights

    @staticmethod
    def _load_single(path: Path) -> dict[str, mx.array]:
        safetensors_files = [f for f in path.glob("*.safetensors") if not f.name.startswith("._")]
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        weights_file = safetensors_files[0]
        WeightPrefetcher.prefetch([weights_file])
        data = mx.load(str(weights_file), return_metadata=True)
        return dict(data[0].items())

    @staticmethod
    def _load_multi_glob(path: Path) -> dict[str, mx.array]:
        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        WeightPrefetcher.prefetch(shard_files)
        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            data, _ = mx.load(str(shard), return_metadata=True)
            all_weights.update(dict(data.items()))

        return all_weights

    @staticmethod
    def _convert_precision(weights: dict[str, mx.array], precision: mx.Dtype) -> dict[str, mx.array]:
        return {k: v if v.dtype == precision else v.astype(precision) for k, v in weights.items()}

    @staticmethod
    def _torch_module() -> Any:
        import torch

        return torch

    @staticmethod
    def _torch_load_file(path: str) -> dict:
        from safetensors.torch import load_file as torch_load_file

        return torch_load_file(path)
