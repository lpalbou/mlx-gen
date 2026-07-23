from collections.abc import Callable, Hashable


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
