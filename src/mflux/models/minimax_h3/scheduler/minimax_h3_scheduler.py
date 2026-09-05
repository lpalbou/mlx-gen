"""Rectified-flow Euler scheduler for MiniMax-H3 (diffusers `MiniMaxH3Scheduler`).

Two conventions differ from the usual flow-matching scheduler and are kept exactly:
the transformer predicts a data-ward velocity (`x0 = x_t + sigma * v`), and timesteps are
`t = 1 - sigma` in `[0, 1]` with `t = 1` meaning clean. The sigma grid starts from
`linspace(1, 0, num_inference_steps)` — the terminal zero counts toward the requested
step count, so `num_inference_steps` grid points drive `num_inference_steps - 1` model
evaluations — is pushed through the exponential shift, and collapses consecutive
duplicates. MiniMax-H3 runs two instances per request: `shift=12.0` (video) and
`shift=3.0` (audio) on the same grid, stepped in lockstep by index.
"""

import mlx.core as mx
import numpy as np


def _aten_linspace_f32(start: float, end: float, steps: int) -> np.ndarray:
    """`torch.linspace(start, end, steps, dtype=float32)` reproduced bit-for-bit.

    ATen fills the first half as `start + i * step` and the second half as `end - (steps - 1 - i) * step`
    as fused multiply-adds, which is what fixes which shifted sigmas survive `unique_consecutive`.
    """
    if steps == 1:
        return np.array([start], dtype=np.float32)
    start32, end32 = np.float32(start), np.float32(end)
    step = np.float32((end32 - start32) / np.float32(steps - 1))
    index = np.arange(steps, dtype=np.float32)
    halfway = steps // 2
    # ATen (clang/arm64) contracts `start + i * step` into a single-rounding FMA; a float64 intermediate
    # reproduces that rounding exactly for float32 operands of this size (verified on 600 sigma grids).
    first = (np.float64(start32) + np.float64(step) * index.astype(np.float64)).astype(np.float32)
    second = (np.float64(end32) - np.float64(step) * (np.float32(steps - 1) - index).astype(np.float64)).astype(
        np.float32
    )
    return np.where(index < halfway, first, second).astype(np.float32)


class MiniMaxH3Scheduler:
    def __init__(self, shift: float = 12.0):
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self.shift = float(shift)
        self.sigmas = np.zeros(0, dtype=np.float32)
        self.timesteps = np.zeros(0, dtype=np.float32)
        self.num_inference_steps = 0

    def set_shift(self, shift: float) -> None:
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self.shift = float(shift)

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Build `num_inference_steps` sigma grid points (terminal `0` included) and the `t = 1 - sigma` timesteps."""
        if num_inference_steps < 2:
            raise ValueError(f"`num_inference_steps` must be at least 2 grid points, got {num_inference_steps}.")
        base = _aten_linspace_f32(1.0, 0.0, int(num_inference_steps))
        shift = np.float32(self.shift)
        sigmas = (shift * base / (np.float32(1.0) + (shift - np.float32(1.0)) * base)).astype(np.float32)
        keep = np.concatenate([[True], sigmas[1:] != sigmas[:-1]])
        self.sigmas = sigmas[keep]
        self.timesteps = (np.float32(1.0) - self.sigmas[:-1]).astype(np.float32)
        self.num_inference_steps = int(self.timesteps.shape[0])

    def scale_noise(self, sample: mx.array, timestep: float, noise: mx.array) -> mx.array:
        """Forward process in H3's convention: `x_t = t * x0 + (1 - t) * noise` (`t = 1` returns the sample)."""
        t = mx.array(timestep, dtype=sample.dtype)
        return t * sample + (1.0 - t) * noise

    def step(self, model_output: mx.array, step_index: int, sample: mx.array) -> mx.array:
        """One Euler step from grid point `step_index` to `step_index + 1`, evaluated in float32."""
        timestep = np.float32(self.timesteps[step_index])
        sigma_from_timestep = mx.array(np.float32(1.0) - timestep, dtype=mx.float32)
        sample32 = sample.astype(mx.float32)
        denoised = sample32 + sigma_from_timestep * model_output.astype(mx.float32)
        sigma = np.float32(self.sigmas[step_index])
        sigma_next = np.float32(self.sigmas[step_index + 1])
        ratio = mx.array(np.float32(sigma_next / sigma), dtype=mx.float32)
        prev_sample = ratio * sample32 + (1.0 - ratio) * denoised
        return prev_sample.astype(sample.dtype)
