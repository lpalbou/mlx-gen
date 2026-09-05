"""Stream a Hugging Face-layout component into its module one shard at a time.

`WeightLoader` + `WeightApplier` load a whole component before quantizing it, which for the two
30B-class MiniMax-H3 components would hold 50+ GB of bf16 next to the growing q8 copy. Here each
shard is mapped, quantized into the already-quantized module structure and released before the
next one is read, so the peak is one shard above the final footprint. Prepared MLX-Gen packages
(`mlxgen prepare`) are already in the module's layout and go through `WeightApplier` unchanged.
"""

import gc
from pathlib import Path
from typing import Callable

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper


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
        component_path = Path(component_root) / component.hf_subdir
        stored_weights, stored_q_level, version = WeightLoader._try_load_mflux_format(component_path)
        if stored_weights is not None:
            loaded = LoadedWeights(
                components={component.name: stored_weights},
                meta_data=MetaData(quantization_level=stored_q_level, mflux_version=version),
            )
            return WeightApplier.apply_and_quantize_single(
                weights=loaded,
                model=model,
                component=component,
                quantize_arg=quantize_arg,
                quantization_predicate=quantization_predicate,
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
