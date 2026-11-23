from pathlib import Path
from typing import Callable

import mlx.core as mx
from mlx import nn

from mflux.models.common.weights.loading.streaming_weight_loader import StreamingWeightLoader
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition


class H3BlockStream:
    # Keeps the transformer blocks out of memory: every block tensor is a pending file read, and once a
    # block has run its tensors are replaced by new pending reads, which releases what the call loaded.
    # One block stands in memory at a time, at the price of reading the blocks from disk at every step.
    PREFIX = "transformer_blocks."

    def __init__(
        self, transformer_path: Path, component: ComponentDefinition, keep: Callable[[str], bool] | None = None
    ):
        files, prepared = StreamingWeightLoader.component_files(transformer_path, component)
        index = StreamingWeightLoader.tensor_index(
            files, lambda key: key.startswith(H3BlockStream.PREFIX) and (keep is None or keep(key))
        )
        self.precision = None if prepared else component.precision
        self.bytes_per_step = sum(size for _, size in index.values())
        self.blocks: dict[int, dict[Path, list[str]]] = {}
        for key, (file, _) in index.items():
            self.blocks.setdefault(int(key.split(".")[1]), {}).setdefault(file, []).append(key)
        self.signatures = StreamingWeightLoader.file_signatures({file for file, _ in index.values()})

    @staticmethod
    def streams_as_stored(transformer_path: Path, quantize: int | None) -> bool:
        # A block can be read again at every step only when the file holds it as the model computes with it:
        # a prepared package, or a source snapshot loaded without quantization.
        return StreamingWeightLoader.is_prepared(transformer_path) or quantize is None

    def rebind(self, index: int, block: nn.Module) -> None:
        files = self.blocks[index]
        StreamingWeightLoader.require_unchanged({file: self.signatures[file] for file in files})
        for file, keys in files.items():
            stored = mx.load(str(file))
            for key in keys:
                tensor = stored[key]
                if self.precision is not None and tensor.dtype != self.precision:
                    tensor = tensor.astype(self.precision)
                module, leaf = H3BlockStream._target(block, key.split(".")[2:])
                setattr(module, leaf, tensor)

    @staticmethod
    def _target(block: nn.Module, parts: list[str]) -> tuple[nn.Module, str]:
        module = block
        for part in parts[:-1]:
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        # An adapter wraps the linear it trains: one adapter holds it as `linear`, fused ones as `base_linear`.
        return getattr(module, "linear", getattr(module, "base_linear", module)), parts[-1]
