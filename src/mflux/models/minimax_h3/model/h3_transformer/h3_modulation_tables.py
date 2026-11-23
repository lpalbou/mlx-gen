import gc
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common.weights.loading.streaming_weight_loader import StreamingWeightLoader
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.minimax_h3.model.h3_transformer.h3_embedding import H3AdaLayerNormModulation


class H3ModulationTables:
    # The per-block AdaLN projections take the timestep embedding alone, and a run uses a handful of
    # timesteps that are all known before its first step. So each block's modulation is computed once per
    # schedule, straight from disk, and the projections (13B of the transformer's 33B parameters) are never
    # held in memory. The values are the ones the resident projection would produce.
    KEY_FRAGMENT = ".adaln_proj."
    MODULATION_PARTS = 6
    KEPT_SCHEDULES = 2

    def __init__(self, transformer_path: Path, component: ComponentDefinition, bits: int | None):
        files, prepared = StreamingWeightLoader.component_files(transformer_path, component)
        index = StreamingWeightLoader.tensor_index(files, H3ModulationTables.holds)
        self.files = {key: file for key, (file, _) in index.items()}
        self.signatures = StreamingWeightLoader.file_signatures(set(self.files.values()))
        # A package holds the projections as the model computes with them; a source snapshot is converted
        # as its load converts every other tensor.
        self.precision = None if prepared else component.precision
        # The width a source snapshot's projections are quantized to at load; a package is read as stored.
        self.bits = bits
        self._schedules: dict[bytes, list[list[tuple[mx.array, ...]]]] = {}

    @staticmethod
    def holds(key: str) -> bool:
        return H3ModulationTables.KEY_FRAGMENT in key

    def for_schedule(self, transformer: nn.Module, timesteps: list[mx.array]) -> list[list[tuple[mx.array, ...]]]:
        # `[step][block]`: the six tables each block takes at each step, for the distinct timesteps of that step.
        key = b"".join(
            len(values).to_bytes(4, "little") + values.tobytes()
            for values in (np.asarray(timestep, dtype=np.float32) for timestep in timesteps)
        )
        tables = self._schedules.pop(key, None)
        if tables is None:
            tables = self._compute(transformer, timesteps)
            while len(self._schedules) >= H3ModulationTables.KEPT_SCHEDULES:
                # The schedule used longest ago goes first.
                self._schedules.pop(next(iter(self._schedules)))
        self._schedules[key] = tables
        return tables

    def _compute(self, transformer: nn.Module, timesteps: list[mx.array]) -> list[list[tuple[mx.array, ...]]]:
        StreamingWeightLoader.require_unchanged(self.signatures)
        embeddings = [transformer.timestep_embedding(timestep) for timestep in timesteps]
        mx.eval(*embeddings)
        steps: list[list[tuple[mx.array, ...]]] = [[] for _ in timesteps]
        for block_index in range(len(transformer.transformer_blocks)):
            linear, hidden_size = self._projection(block_index)
            block_tables = [H3AdaLayerNormModulation.tables(linear, embedding, hidden_size) for embedding in embeddings]
            mx.eval(*[table for tables in block_tables for table in tables])
            for step, tables in zip(steps, block_tables):
                step.append(tables)
            del linear
        gc.collect()
        mx.clear_cache()
        return steps

    def _projection(self, block_index: int) -> tuple[nn.Module, int]:
        prefix = f"transformer_blocks.{block_index}.adaln_proj.linear."
        opened: dict[Path, dict[str, mx.array]] = {}
        tensors = {}
        for leaf in ("weight", "bias", "scales", "biases"):
            file = self.files.get(prefix + leaf)
            if file is not None:
                if file not in opened:
                    opened[file] = mx.load(str(file))
                tensors[leaf] = opened[file][prefix + leaf]
        if "weight" not in tensors or "bias" not in tensors:
            raise ValueError(f"The transformer weights hold no AdaLN projection for block {block_index}.")
        group_size = StreamingWeightLoader.PREPARED_GROUP_SIZE
        if "scales" in tensors:
            inputs = tensors["scales"].shape[-1] * group_size
            linear = nn.QuantizedLinear(
                group_size, 1, bias=True, group_size=group_size, bits=tensors["weight"].shape[-1] * 32 // inputs
            )
            linear.weight, linear.scales, linear.biases = tensors["weight"], tensors["scales"], tensors["biases"]
            linear.bias = tensors["bias"]
        elif self.bits is not None:
            linear = nn.QuantizedLinear(group_size, 1, bias=True, group_size=group_size, bits=self.bits)
            linear.weight, linear.scales, linear.biases = StreamingWeightLoader.quantize(
                self._as_loaded(tensors["weight"]), group_size, self.bits
            )
            linear.bias = self._as_loaded(tensors["bias"])
        else:
            linear = nn.Linear(1, 1, bias=True)
            linear.weight = self._as_loaded(tensors["weight"])
            linear.bias = self._as_loaded(tensors["bias"])
        return linear, int(tensors["bias"].shape[0]) // (H3ModulationTables.MODULATION_PARTS * 3)

    def _as_loaded(self, tensor: mx.array) -> mx.array:
        return tensor if self.precision is None or tensor.dtype == self.precision else tensor.astype(self.precision)
