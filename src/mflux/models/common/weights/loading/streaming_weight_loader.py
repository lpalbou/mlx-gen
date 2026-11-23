"""Stream a Hugging Face-layout component into its module one shard at a time.

`WeightLoader` + `WeightApplier` load a whole component before quantizing it, which for the two
30B-class MiniMax-H3 components would hold 50+ GB of bf16 next to the growing q8 copy. Here each
shard is mapped, quantized into the already-quantized module structure and released before the
next one is read, so the peak is one shard above the final footprint. Prepared MLX-Gen packages
(`mlxgen prepare`) are already in the module's layout and stream shard by shard as well.
"""

import gc
import json
import struct
from pathlib import Path
from typing import Callable

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import safe_open

from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.utils.runtime_memory import RuntimeMemory


class StreamingWeightLoader:
    @staticmethod
    def load_into(
        model: nn.Module,
        component_root: Path,
        component: ComponentDefinition,
        quantize_arg: int | None,
        quantization_predicate: Callable | None = None,
        post_transform: Callable[[dict[str, mx.array]], dict[str, mx.array]] | None = None,
        keep: Callable[[str], bool] | None = None,
        evaluate: bool = True,
    ) -> int | None:
        """Load `component` from `component_root / component.hf_subdir` into `model`; returns the resolved bits."""
        # `keep` selects tensors by their name in the module; the rest are never read. With `evaluate=False`
        # every tensor stays a pending file read, so a weight costs memory from its first use only.
        # Same once-per-process MLX cache-limit default as `WeightLoader` (Python-API hosts skip the CLI setup).
        RuntimeMemory.apply_default_cache_limit_once()
        component_path = Path(component_root) / component.hf_subdir
        prepared_shards = StreamingWeightLoader._prepared_shards(component_path)
        if prepared_shards:
            return StreamingWeightLoader._load_prepared(
                model, component, prepared_shards, quantize_arg, keep=keep, evaluate=evaluate
            )

        files = WeightLoader._resolve_weight_files(component_path, component.weight_files, "*.safetensors")
        bits = None
        if quantize_arg is not None and not component.skip_quantization:
            bits = int(quantize_arg)
            predicate = quantization_predicate or (lambda path, module: hasattr(module, "to_quantized"))
            # Quantizes the (lazy, never evaluated) initial parameters: only the module structure matters here,
            # the real tensors are quantized shard by shard below and replace them before anything is evaluated.
            nn.quantize(
                model, class_predicate=WeightApplier.quantization_predicate_for_bits(predicate, bits), bits=bits
            )
        quantized_modules = {
            path: module
            for path, module in model.named_modules()
            if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
        }

        for file in files:
            raw = dict(mx.load(str(file)).items())
            if component.weight_prefix_filters is not None:
                raw = {k: v for k, v in raw.items() if k.startswith(tuple(component.weight_prefix_filters))}
            if component.precision is not None:
                raw = WeightLoader._convert_precision(
                    raw, component.precision, precision_override=component.precision_override
                )
            if component.mapping_getter is None:
                mapped = {k: component.bulk_transform(v) for k, v in raw.items()} if component.bulk_transform else raw
            else:
                nested = WeightMapper.apply_mapping(
                    hf_weights=raw,
                    mapping=component.mapping_getter(),
                    num_blocks=component.num_blocks,
                    num_layers=component.num_layers,
                )
                mapped = dict(tree_flatten(nested))
            if post_transform is not None:
                mapped = post_transform(mapped)
            if keep is not None:
                mapped = {key: tensor for key, tensor in mapped.items() if keep(key)}

            flat: list[tuple[str, mx.array]] = []
            for key, tensor in mapped.items():
                parent, _, leaf = key.rpartition(".")
                module = quantized_modules.get(parent)
                if module is not None and leaf == "weight":
                    weight, scales, biases = StreamingWeightLoader.quantize(tensor, module.group_size, module.bits)
                    flat.extend(((key, weight), (f"{parent}.scales", scales), (f"{parent}.biases", biases)))
                else:
                    flat.append((key, tensor))
            model.update(tree_unflatten(flat), strict=False)
            if evaluate:
                mx.eval(*[tensor for _, tensor in flat])
            del raw, mapped, flat
            if evaluate:
                gc.collect()
                mx.clear_cache()
        return bits

    # The width whose indices are chosen again on the stored grid; one index is one byte of the packed weight.
    REGRIDDED_BITS = 8

    @staticmethod
    def quantize(tensor: mx.array, group_size: int, bits: int) -> tuple[mx.array, mx.array, mx.array]:
        weight, scales, biases = mx.quantize(tensor, group_size=group_size, bits=bits)
        if bits != StreamingWeightLoader.REGRIDDED_BITS or tensor.ndim != 2 or scales.dtype == mx.float32:
            return weight, scales, biases
        # `mx.quantize` picks every index on a float32 grid, then returns that grid's scales and biases at the
        # tensor's own precision. With 256 levels the bf16 rounding of the grid is as large as a step, so about
        # one index in five is no longer the nearest. Picking them again on the grid that is stored lowers the
        # weight error by a quarter at the same size; at 6 bits and below the steps are too wide for it to matter.
        rows, inputs = tensor.shape
        groups = tensor.astype(mx.float32).reshape(rows, inputs // group_size, group_size)
        step = scales.astype(mx.float32)[..., None]
        offset = biases.astype(mx.float32)[..., None]
        indices = mx.clip(mx.round((groups - offset) / mx.where(step == 0, 1.0, step)), 0, 255)
        return indices.astype(mx.uint8).reshape(-1).view(mx.uint32).reshape(weight.shape), scales, biases

    @staticmethod
    def _prepared_shards(component_path: Path) -> list[Path]:
        """Shards of a prepared MLX-Gen package (identified by its metadata), or [] for a Hugging Face snapshot."""
        if not component_path.exists():
            return []
        shards = WeightLoader._mflux_shard_files(component_path)
        if not shards:
            return []
        metadata = WeightLoader._load_safetensors_metadata(shards[0])
        return shards if "mflux_version" in metadata or "quantization_level" in metadata else []

    # MLX's default affine group size, which every MLX-Gen package is written with.
    PREPARED_GROUP_SIZE = 64

    @staticmethod
    def _load_prepared(
        model: nn.Module,
        component: ComponentDefinition,
        shards: list[Path],
        quantize_arg: int | None,
        keep: Callable[[str], bool] | None = None,
        evaluate: bool = True,
    ) -> int | None:
        # A prepared package already holds the module's own tensors, so its shards stream one at a time (the
        # whole-component path would hold a 64 GiB package twice). Each module is rebuilt at the width its
        # tensors were stored with, read from the shard headers, so a mixed-width package loads as written
        # whatever the current quantization policy says.
        stored = WeightLoader._load_safetensors_metadata(shards[0]).get("quantization_level")
        stored_bits = None if stored in (None, "", "None", "null") or component.skip_quantization else int(stored)
        if quantize_arg is not None and stored_bits is not None and int(quantize_arg) != stored_bits:
            print(f"⚠️  {component.name}: stored q{stored_bits} package; ignoring --quantize {quantize_arg}.")
        shapes = StreamingWeightLoader._stored_shapes(shards)
        if keep is not None:
            shapes = {key: shape for key, shape in shapes.items() if keep(key)}
        StreamingWeightLoader._quantize_as_stored(model, component, shapes)
        StreamingWeightLoader._require_stored_parameters(model, component, shapes)
        for shard in shards:
            weights = {key: value for key, value in mx.load(str(shard)).items() if keep is None or keep(key)}
            model.update(tree_unflatten(list(weights.items())), strict=False)
            if evaluate:
                mx.eval(*weights.values())
            del weights
            if evaluate:
                gc.collect()
                mx.clear_cache()
        return stored_bits

    @staticmethod
    def component_files(component_path: Path, component: ComponentDefinition) -> tuple[list[Path], bool]:
        # The weight files of a component, and whether they are a prepared package (stored as the module holds
        # them) rather than a Hugging Face snapshot.
        prepared = StreamingWeightLoader._prepared_shards(Path(component_path))
        if prepared:
            return prepared, True
        files = WeightLoader._resolve_weight_files(Path(component_path), component.weight_files, "*.safetensors")
        return [Path(file) for file in files], False

    @staticmethod
    def is_prepared(component_path: Path) -> bool:
        return bool(StreamingWeightLoader._prepared_shards(Path(component_path)))

    @staticmethod
    def stored_bits(component_path: Path) -> int | None:
        # The width a prepared package records for a component; None for a snapshot or an unquantized package.
        shards = StreamingWeightLoader._prepared_shards(Path(component_path))
        if not shards:
            return None
        stored = WeightLoader._load_safetensors_metadata(shards[0]).get("quantization_level")
        return None if stored in (None, "", "None", "null") else int(stored)

    @staticmethod
    def tensor_index(files: list[Path], keep: Callable[[str], bool] | None = None) -> dict[str, tuple[Path, int]]:
        # Where each tensor is and how many bytes it holds as stored, from the safetensors headers alone.
        index: dict[str, tuple[Path, int]] = {}
        for file in files:
            with open(file, "rb") as handle:
                (header_length,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(header_length))
            for key, entry in header.items():
                if key != "__metadata__" and (keep is None or keep(key)):
                    begin, end = entry["data_offsets"]
                    index[key] = (Path(file), end - begin)
        return index

    @staticmethod
    def tensor_bytes(files: list[Path], keep: Callable[[str], bool] | None = None) -> int:
        return sum(size for _, size in StreamingWeightLoader.tensor_index(files, keep).values())

    @staticmethod
    def file_signatures(files) -> dict[Path, tuple[int, int, int]]:
        signatures = {}
        for file in files:
            status = Path(file).stat()
            signatures[Path(file)] = (status.st_ino, status.st_size, status.st_mtime_ns)
        return signatures

    @staticmethod
    def require_unchanged(signatures: dict[Path, tuple[int, int, int]]) -> None:
        # A tensor that has not been read yet takes whatever its file holds when it is read. A file replaced
        # since the model was loaded would hand it other weights without an error.
        for file, recorded in signatures.items():
            try:
                status = file.stat()
                current = (status.st_ino, status.st_size, status.st_mtime_ns)
            except OSError:
                current = None
            if current != recorded:
                raise RuntimeError(
                    f"{str(file)!r} changed on disk after the model was loaded, and weights it has not read yet "
                    "would come from the new file. Load the model again."
                )

    @staticmethod
    def _stored_shapes(shards: list[Path]) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {}
        for shard in shards:
            with safe_open(str(shard), framework="numpy") as handle:
                for key in handle.keys():
                    shapes[key] = tuple(handle.get_slice(key).get_shape())
        return shapes

    @staticmethod
    def _quantize_as_stored(
        model: nn.Module, component: ComponentDefinition, shapes: dict[str, tuple[int, ...]]
    ) -> None:
        group_size = StreamingWeightLoader.PREPARED_GROUP_SIZE
        stored_paths = {key[: -len(".scales")] for key in shapes if key.endswith(".scales")}
        matched: set[str] = set()

        def as_stored(path: str, module):
            if path not in stored_paths or not hasattr(module, "to_quantized"):
                return False
            packed, scales = shapes.get(f"{path}.weight"), shapes[f"{path}.scales"]
            inputs = module.weight.shape[-1]
            # Packed uint32 columns hold `inputs * bits / 32`; the scales hold one column per group of inputs.
            if packed is None or scales[-1] * group_size != inputs or (packed[-1] * 32) % inputs:
                raise ValueError(
                    f"{component.name}: the stored tensors of {path!r} (weight {packed}, scales {scales}) do not fit "
                    f"its {inputs} inputs at a group size of {group_size}. The package was written for a different "
                    "model layout."
                )
            matched.add(path)
            return {"bits": packed[-1] * 32 // inputs, "group_size": group_size}

        nn.quantize(model, class_predicate=as_stored)
        unknown = sorted(stored_paths - matched)
        if unknown:
            raise ValueError(
                f"{component.name}: the package stores quantized tensors for {len(unknown)} modules this model "
                f"does not have (first: {unknown[0]!r}). The package was written for a different model layout."
            )

    @staticmethod
    def _require_stored_parameters(
        model: nn.Module, component: ComponentDefinition, shapes: dict[str, tuple[int, ...]]
    ) -> None:
        # `update(strict=False)` would leave a parameter the package lacks at its random initial value, and
        # take one of another shape as it comes.
        expected = {key: tuple(value.shape) for key, value in tree_flatten(model.parameters())}
        missing = sorted(key for key in expected if key not in shapes)
        if missing:
            raise ValueError(
                f"{component.name}: the package is missing {len(missing)} tensors this model needs "
                f"(first: {missing[0]!r}). It is incomplete or was written for a different model layout."
            )
        mismatched = sorted(key for key, shape in expected.items() if shapes[key] != shape)
        if mismatched:
            key = mismatched[0]
            raise ValueError(
                f"{component.name}: {key!r} is stored as {shapes[key]} and this model expects {expected[key]}. "
                "The package was written for a different model layout."
            )
