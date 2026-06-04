from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from mflux.callbacks.callback import AfterLoopCallback, BeforeLoopCallback, InLoopCallback, InterruptCallback
from mflux.callbacks.progress import ProgressCallback, ProgressEvent

if TYPE_CHECKING:
    from mflux.callbacks.generation_context import GenerationContext
    from mflux.models.common.config.config import Config


class CallbackRegistry:
    def __init__(self):
        self.in_loop: list[InLoopCallback] = []
        self.before_loop: list[BeforeLoopCallback] = []
        self.interrupt: list[InterruptCallback] = []
        self.after_loop: list[AfterLoopCallback] = []
        self._progress: list[tuple[ProgressCallback, str | None]] = []

    def register(self, callback) -> None:
        if hasattr(callback, "call_before_loop"):
            self.before_loop.append(callback)
        if hasattr(callback, "call_in_loop"):
            self.in_loop.append(callback)
        if hasattr(callback, "call_after_loop"):
            self.after_loop.append(callback)
        if hasattr(callback, "call_interrupt"):
            self.interrupt.append(callback)

    def subscribe_progress(
        self,
        callback: ProgressCallback,
        *,
        task: str | None = None,
    ) -> Callable[[], None]:
        subscription = (callback, task)
        self._progress.append(subscription)

        def unsubscribe() -> None:
            if subscription in self._progress:
                self._progress.remove(subscription)

        return unsubscribe

    def emit_progress(
        self,
        event: ProgressEvent,
    ) -> None:
        for callback, task in list(self._progress):
            if task is None or task == event.task:
                callback(event)

    def has_progress_subscribers(self, *, task: str | None = None) -> bool:
        return any(task_filter is None or task_filter == task for _, task_filter in self._progress)

    def start(
        self,
        seed: int,
        prompt: str,
        config: Config,
        *,
        task: str | None = None,
    ) -> GenerationContext:
        from mflux.callbacks.generation_context import GenerationContext

        return GenerationContext(self, seed, prompt, config, task=task)

    def before_loop_callbacks(self) -> list[BeforeLoopCallback]:
        return self.before_loop

    def in_loop_callbacks(self) -> list[InLoopCallback]:
        return self.in_loop

    def after_loop_callbacks(self) -> list[AfterLoopCallback]:
        return self.after_loop

    def interrupt_callbacks(self) -> list[InterruptCallback]:
        return self.interrupt
