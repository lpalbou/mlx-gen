from collections.abc import Callable, Hashable

import mlx.core as mx
from mlx import nn


class CompiledPredictCache:
    # mx.compile bakes closed-over weight arrays into the traced graph as constants,
    # so a cached callable is only valid while the module it closes over keeps its
    # identity AND its weights. Argument shape/dtype changes retrace inside
    # mx.compile itself and need no key here; the key covers python-level branch
    # structure (e.g. CFG negative-embeds present or not). Weight MUTATION on the
    # same module (LoRA load, training updates, LoRA bake at save) must call
    # clear() explicitly — module identity cannot detect it (0095; wan 0090 d12
    # established the drop-before-release discipline this follows).
    def __init__(self) -> None:
        self._entries: dict[Hashable, Callable] = {}
        self._weights_token: object | None = None

    def get_or_build(self, *, key: Hashable, weights_token: object, build: Callable[[], Callable]) -> Callable:
        if weights_token is not self._weights_token:
            # Module replaced (release/reload): every cached trace closes over
            # stale arrays and would keep them alive; drop them all.
            self._entries = {}
            self._weights_token = weights_token
        compiled = self._entries.get(key)
        if compiled is None:
            compiled = build()
            self._entries[key] = compiled
        return compiled

    def clear(self) -> None:
        self._entries = {}
        self._weights_token = None

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def compile_over_weights(predict: Callable, module: nn.Module) -> Callable:
        # The evaluation must happen before the compiled function is first called. A trace
        # (one per input shape) owns its own copy of every array that is not yet evaluated
        # when it is taken, and keeps it for its lifetime. Weights are loaded lazily, so
        # without this every canvas size held another copy of the transformer. Evaluated
        # arrays are shared by all traces: one copy, the module's own. It moves the weight
        # read out of the first step (about 0.35 s once per process on klein 9B); do not
        # remove it or defer it into the denoise loop.
        mx.eval(module.parameters())
        return mx.compile(predict)
