from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class WanEulerSchedulerOutput:
    prev_sample: mx.array


class WanEulerScheduler:
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        flow_shift: float = 5.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.flow_shift = flow_shift
        self.timesteps = mx.array([], dtype=mx.float32)
        self.sigmas = mx.array([], dtype=mx.float32)
        self.step_index: int | None = None

    def set_timesteps(self, num_inference_steps: int) -> None:
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be greater than zero.")
        timesteps = np.linspace(self.num_train_timesteps, 0, num_inference_steps + 1, dtype=np.float32)
        sigmas = timesteps / self.num_train_timesteps
        sigmas = self.flow_shift * sigmas / (1 + (self.flow_shift - 1) * sigmas)

        self.timesteps = mx.array((sigmas[:-1] * self.num_train_timesteps).astype(np.float32), dtype=mx.float32)
        self.sigmas = mx.array(sigmas.astype(np.float32), dtype=mx.float32)
        self.step_index = None

    def step(
        self,
        model_output: mx.array,
        timestep: float | mx.array,
        sample: mx.array,
        return_dict: bool = True,
    ) -> WanEulerSchedulerOutput | tuple[mx.array]:
        if self.step_index is None:
            self.step_index = self._index_for_timestep(timestep)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        prev_sample = sample.astype(mx.float32) + (sigma_next - sigma) * model_output.astype(mx.float32)
        self.step_index += 1

        if not return_dict:
            return (prev_sample,)
        return WanEulerSchedulerOutput(prev_sample=prev_sample)

    def _index_for_timestep(self, timestep: float | mx.array) -> int:
        timestep_value = self._timestep_to_float(timestep)
        timesteps = np.array(self.timesteps)
        matches = np.where(np.isclose(timesteps, timestep_value, rtol=1e-6, atol=1e-5))[0]
        if len(matches) == 0:
            return len(timesteps) - 1
        if len(matches) > 1:
            return int(matches[1])
        return int(matches[0])

    @staticmethod
    def _timestep_to_float(timestep: float | mx.array) -> float:
        if hasattr(timestep, "item"):
            return float(timestep.item())
        return float(timestep)
