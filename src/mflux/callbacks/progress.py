from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ProgressEvent:
    phase: str
    frame: int | None = None
    total_frames: int | None = None
    step: int = 0
    total_steps: int = 0
    task: str | None = None
    timestep: int | float | None = None
    seed: int | None = None
    item_index: int | None = None
    item_count: int | None = None
    output_path: str | None = None
    input_name: str | None = None
    # Resolved output dimensions, carried on early events (Wan emits them at
    # phase="start") so hosts learn final geometry before container probing.
    width: int | None = None
    height: int | None = None

    @property
    def progress(self) -> float:
        return self.step_progress

    @property
    def step_progress(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return min(1.0, max(0.0, self.step / self.total_steps))

    @property
    def frame_progress(self) -> float | None:
        if self.frame is None or self.total_frames is None or self.total_frames <= 0:
            return None
        return min(1.0, max(0.0, self.frame / self.total_frames))


ProgressCallback = Callable[[ProgressEvent], None]
