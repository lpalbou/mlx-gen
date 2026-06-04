from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True, kw_only=True)
class TensorHealthReport:
    name: str
    phase: str
    shape: tuple[int, ...]
    dtype: str
    total_values: int
    non_finite_values: int
    nan_values: int
    inf_values: int
    finite_min: float | None = None
    finite_max: float | None = None
    step: int | None = None
    total_steps: int | None = None
    timestep: int | float | None = None
    frame: int | None = None
    total_frames: int | None = None
    denoiser: str | None = None
    guidance: float | None = None

    def message(self) -> str:
        parts = [
            "Non-finite tensor values detected",
            f"phase={self.phase}",
            f"tensor={self.name}",
            f"shape={self.shape}",
            f"dtype={self.dtype}",
            f"non_finite={self.non_finite_values}/{self.total_values}",
            f"nan={self.nan_values}",
            f"inf={self.inf_values}",
        ]
        if self.finite_min is None or self.finite_max is None:
            parts.append("finite_range=none")
        else:
            parts.append(f"finite_range={self.finite_min:g}..{self.finite_max:g}")
        if self.step is not None:
            if self.total_steps is None:
                parts.append(f"step={self.step}")
            else:
                parts.append(f"step={self.step}/{self.total_steps}")
        if self.timestep is not None:
            parts.append(f"timestep={self.timestep}")
        if self.frame is not None:
            if self.total_frames is None:
                parts.append(f"frame={self.frame}")
            else:
                parts.append(f"frame={self.frame}/{self.total_frames}")
        if self.denoiser is not None:
            parts.append(f"denoiser={self.denoiser}")
        if self.guidance is not None:
            parts.append(f"guidance={self.guidance:g}")
        parts.append("generation aborted before saving invalid output")
        return "; ".join(parts) + "."


class TensorHealthError(ValueError):
    def __init__(self, report: TensorHealthReport):
        self.report = report
        super().__init__(report.message())


class TensorHealth:
    @staticmethod
    def ensure_finite(
        tensor,
        *,
        name: str,
        phase: str,
        step: int | None = None,
        total_steps: int | None = None,
        timestep: int | float | None = None,
        frame: int | None = None,
        total_frames: int | None = None,
        denoiser: str | None = None,
        guidance: float | None = None,
    ) -> None:
        if isinstance(tensor, np.ndarray):
            TensorHealth._ensure_finite_numpy(
                tensor,
                name=name,
                phase=phase,
                step=step,
                total_steps=total_steps,
                timestep=timestep,
                frame=frame,
                total_frames=total_frames,
                denoiser=denoiser,
                guidance=guidance,
            )
            return

        TensorHealth._ensure_finite_mlx(
            tensor,
            name=name,
            phase=phase,
            step=step,
            total_steps=total_steps,
            timestep=timestep,
            frame=frame,
            total_frames=total_frames,
            denoiser=denoiser,
            guidance=guidance,
        )

    @staticmethod
    def should_check_step(step: int, total_steps: int, interval: int | None) -> bool:
        if interval is None or interval <= 0:
            return False
        return step == 1 or step == total_steps or step % interval == 0

    @staticmethod
    def _ensure_finite_numpy(
        tensor: np.ndarray,
        *,
        name: str,
        phase: str,
        dtype: str | None = None,
        step: int | None,
        total_steps: int | None,
        timestep: int | float | None,
        frame: int | None,
        total_frames: int | None,
        denoiser: str | None,
        guidance: float | None,
    ) -> None:
        finite = np.isfinite(tensor)
        if finite.all():
            return
        nan_values = int(np.isnan(tensor).sum())
        inf_values = int(np.isinf(tensor).sum())
        finite_values = tensor[finite]
        finite_min = float(np.min(finite_values)) if finite_values.size else None
        finite_max = float(np.max(finite_values)) if finite_values.size else None
        TensorHealth._raise(
            name=name,
            phase=phase,
            shape=tuple(int(value) for value in tensor.shape),
            dtype=dtype or str(tensor.dtype),
            total_values=int(tensor.size),
            non_finite_values=int(tensor.size - finite.sum()),
            nan_values=nan_values,
            inf_values=inf_values,
            finite_min=finite_min,
            finite_max=finite_max,
            step=step,
            total_steps=total_steps,
            timestep=timestep,
            frame=frame,
            total_frames=total_frames,
            denoiser=denoiser,
            guidance=guidance,
        )

    @staticmethod
    def _ensure_finite_mlx(
        tensor: mx.array,
        *,
        name: str,
        phase: str,
        step: int | None,
        total_steps: int | None,
        timestep: int | float | None,
        frame: int | None,
        total_frames: int | None,
        denoiser: str | None,
        guidance: float | None,
    ) -> None:
        finite = mx.isfinite(tensor)
        if bool(mx.all(finite).item()):
            return
        nan_values = int(mx.sum(mx.isnan(tensor)).item())
        inf_values = int(mx.sum(mx.isinf(tensor)).item())
        non_finite_values = int(mx.sum(mx.logical_not(finite)).item())
        total_values = int(np.prod(tensor.shape))
        finite_values = total_values - non_finite_values
        finite_min = None
        finite_max = None
        if finite_values > 0:
            tensor_float = tensor.astype(mx.float32)
            finite_min = float(mx.min(mx.where(finite, tensor_float, mx.inf)).item())
            finite_max = float(mx.max(mx.where(finite, tensor_float, -mx.inf)).item())
        TensorHealth._raise(
            name=name,
            phase=phase,
            shape=tuple(int(value) for value in tensor.shape),
            dtype=str(tensor.dtype),
            total_values=total_values,
            non_finite_values=non_finite_values,
            nan_values=nan_values,
            inf_values=inf_values,
            finite_min=finite_min,
            finite_max=finite_max,
            step=step,
            total_steps=total_steps,
            timestep=timestep,
            frame=frame,
            total_frames=total_frames,
            denoiser=denoiser,
            guidance=guidance,
        )

    @staticmethod
    def _raise(
        *,
        name: str,
        phase: str,
        shape: tuple[int, ...],
        dtype: str,
        total_values: int,
        non_finite_values: int,
        nan_values: int,
        inf_values: int,
        finite_min: float | None,
        finite_max: float | None,
        step: int | None,
        total_steps: int | None,
        timestep: int | float | None,
        frame: int | None,
        total_frames: int | None,
        denoiser: str | None,
        guidance: float | None,
    ) -> None:
        raise TensorHealthError(
            TensorHealthReport(
                name=name,
                phase=phase,
                shape=shape,
                dtype=dtype,
                total_values=total_values,
                non_finite_values=non_finite_values,
                nan_values=nan_values,
                inf_values=inf_values,
                finite_min=finite_min,
                finite_max=finite_max,
                step=step,
                total_steps=total_steps,
                timestep=timestep,
                frame=frame,
                total_frames=total_frames,
                denoiser=denoiser,
                guidance=guidance,
            )
        )
