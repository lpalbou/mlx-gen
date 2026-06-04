from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import PIL.Image
import tqdm

from mflux.callbacks.progress import ProgressEvent

if TYPE_CHECKING:
    from mflux.callbacks.callback_registry import CallbackRegistry
    from mflux.models.common.config.config import Config


class GenerationContext:
    def __init__(
        self,
        registry: CallbackRegistry,
        seed: int,
        prompt: str,
        config: Config,
        *,
        task: str | None = None,
    ):
        self._registry = registry
        self._seed = seed
        self._prompt = prompt
        self._config = config
        self._task = task
        self._progress_step = 0

    def before_loop(
        self,
        latents: mx.array,
        *,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
    ) -> None:
        for subscriber in self._registry.before_loop_callbacks():
            subscriber.call_before_loop(
                seed=self._seed,
                prompt=self._prompt,
                latents=latents,
                config=self._config,
                canny_image=canny_image,
                depth_image=depth_image,
            )
        self._progress_step = 0
        self._emit_progress(phase="start", step=0, timestep=None)

    def in_loop(self, t: int, latents: mx.array, time_steps: tqdm = None) -> None:
        time_steps = time_steps or self._config.time_steps
        self._progress_step = min(self._total_steps(), self._progress_step + 1)
        if self._has_progress_listener():
            mx.eval(latents)
        self._emit_progress(phase="denoise", step=self._progress_step, timestep=t)
        for subscriber in self._registry.in_loop_callbacks():
            subscriber.call_in_loop(
                t=t,
                seed=self._seed,
                prompt=self._prompt,
                latents=latents,
                config=self._config,
                time_steps=time_steps,
            )

    def after_loop(self, latents: mx.array) -> None:
        self._progress_step = self._total_steps()
        for subscriber in self._registry.after_loop_callbacks():
            subscriber.call_after_loop(
                seed=self._seed,
                prompt=self._prompt,
                latents=latents,
                config=self._config,
            )
        self._emit_progress(phase="complete", step=self._progress_step, timestep=None)

    def interruption(self, t: int, latents: mx.array, time_steps: tqdm = None) -> None:
        time_steps = time_steps or self._config.time_steps
        self._emit_progress(phase="interrupted", step=self._progress_step, timestep=t)
        for subscriber in self._registry.interrupt_callbacks():
            subscriber.call_interrupt(
                t=t,
                seed=self._seed,
                prompt=self._prompt,
                latents=latents,
                config=self._config,
                time_steps=time_steps,
            )

    def _emit_progress(self, *, phase: str, step: int, timestep: int | float | None) -> None:
        if not self._has_progress_listener():
            return
        event = ProgressEvent(
            phase=phase,
            step=step,
            total_steps=self._total_steps(),
            task=self._resolved_task(),
            timestep=timestep,
        )
        self._registry.emit_progress(event)

    def _total_steps(self) -> int:
        return max(0, int(self._config.num_inference_steps) - int(self._config.init_time_step))

    def _resolved_task(self) -> str:
        if self._task is not None:
            return self._task
        image_path = getattr(self._config, "image_path", None)
        image_strength = getattr(self._config, "image_strength", None)
        if image_path is not None and image_strength is not None and image_strength > 0.0:
            return "image-to-image"
        return "text-to-image"

    def _has_progress_listener(self) -> bool:
        return self._registry.has_progress_subscribers(task=self._resolved_task())
