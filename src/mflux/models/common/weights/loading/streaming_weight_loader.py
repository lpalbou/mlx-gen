"""Stream a Hugging Face-layout component into its module one shard at a time.

`WeightLoader` + `WeightApplier` load a whole component before quantizing it, which for the two
30B-class MiniMax-H3 components would hold 50+ GB of bf16 next to the growing q8 copy. Here each
shard is mapped, quantized into the already-quantized module structure and released before the
next one is read, so the peak is one shard above the final footprint. Prepared MLX-Gen packages
(`mlxgen prepare`) are already in the module's layout and stream shard by shard as well.
"""

import gc
from pathlib import Path
from typing import Callable

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

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
    ) -> int | None:
        """Load `component` from `component_root / component.hf_subdir` into `model`; returns the resolved bits."""
        # Same once-per-process MLX cache-limit default as `WeightLoader` (Python-API hosts skip the CLI setup).
        RuntimeMemory.apply_default_cache_limit_once()
        component_path = Path(component_root) / component.hf_subdir
        prepared_shards = StreamingWeightLoader._prepared_shards(component_path)
        if prepared_shards:
            return StreamingWeightLoader._load_prepared(
                model, component, prepared_shards, quantize_arg, quantization_predicate
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

            flat: list[tuple[str, mx.array]] = []
            for key, tensor in mapped.items():
                parent, _, leaf = key.rpartition(".")
                module = quantized_modules.get(parent)
                if module is not None and leaf == "weight":
                    weight, scales, biases = mx.quantize(tensor, group_size=module.group_size, bits=module.bits)
                    flat.extend(((key, weight), (f"{parent}.scales", scales), (f"{parent}.biases", biases)))
                else:
                    flat.append((key, tensor))
            model.update(tree_unflatten(flat), strict=False)
            mx.eval(*[tensor for _, tensor in flat])
            del raw, mapped, flat
            gc.collect()
            mx.clear_cache()
        return bits

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

    @staticmethod
    def _load_prepared(
        model: nn.Module,
        component: ComponentDefinition,
        shards: list[Path],
        quantize_arg: int | None,
        quantization_predicate: Callable | None,
    ) -> int | None:
        """A prepared MLX-Gen package already holds the module's own tensors (quantized or not); stream its shards.

        The whole-component path would hold a 75 GB MiniMax-H3 package in memory twice over while applying it.
        """
        metadata = WeightLoader._load_safetensors_metadata(shards[0])
        stored = metadata.get("quantization_level")
        stored_bits = None if stored in (None, "None") else int(stored)
        if component.skip_quantization:
            stored_bits = None
        elif quantize_arg is not None and stored_bits is not None and int(quantize_arg) != stored_bits:
            print(f"⚠️  {component.name}: stored q{stored_bits} package; ignoring --quantize {quantize_arg}.")
        if stored_bits is not None:
            predicate = quantization_predicate or (lambda path, module: hasattr(module, "to_quantized"))
            nn.quantize(
                model,
                class_predicate=WeightApplier.quantization_predicate_for_bits(predicate, stored_bits),
                bits=stored_bits,
            )
        for shard in shards:
            weights = dict(mx.load(str(shard)).items())
            model.update(tree_unflatten(list(weights.items())), strict=False)
            mx.eval(*weights.values())
            del weights
            gc.collect()
            mx.clear_cache()
        return stored_bits
