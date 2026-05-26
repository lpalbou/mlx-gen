from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class WanUniPCSchedulerOutput:
    prev_sample: mx.array


class WanUniPCMultistepScheduler:
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        flow_shift: float = 5.0,
        solver_order: int = 2,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
    ):
        if solver_order != 2:
            raise ValueError("Wan2.2 TI2V currently supports UniPC solver_order=2.")
        if solver_type != "bh2":
            raise ValueError("Wan2.2 TI2V currently supports UniPC solver_type='bh2'.")
        self.num_train_timesteps = num_train_timesteps
        self.flow_shift = flow_shift
        self.solver_order = solver_order
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.num_inference_steps: int | None = None
        self.timesteps = mx.array([], dtype=mx.int64)
        self.sigmas = mx.array([], dtype=mx.float32)
        self.model_outputs: list[mx.array | None] = [None] * solver_order
        self.timestep_list: list[int | None] = [None] * solver_order
        self.lower_order_nums = 0
        self.last_sample: mx.array | None = None
        self.step_index: int | None = None
        self.this_order = 1

    def set_timesteps(self, num_inference_steps: int) -> None:
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be greater than zero.")
        sigmas = np.linspace(1, 1 / self.num_train_timesteps, num_inference_steps + 1)[:-1]
        sigmas = self.flow_shift * sigmas / (1 + (self.flow_shift - 1) * sigmas)
        if abs(sigmas[0] - 1) < 1e-6:
            sigmas[0] -= 1e-6
        timesteps = (sigmas * self.num_train_timesteps).astype(np.int64)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        self.timesteps = mx.array(timesteps, dtype=mx.int64)
        self.sigmas = mx.array(sigmas, dtype=mx.float32)
        self.num_inference_steps = num_inference_steps
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.step_index = None
        self.this_order = 1

    def step(
        self,
        model_output: mx.array,
        timestep: int | mx.array,
        sample: mx.array,
        return_dict: bool = True,
    ) -> WanUniPCSchedulerOutput | tuple[mx.array]:
        if self.num_inference_steps is None:
            raise ValueError("set_timesteps must be called before step.")
        if self.step_index is None:
            self.step_index = self._index_for_timestep(timestep)

        use_corrector = self.step_index > 0 and self.last_sample is not None
        converted = self.convert_model_output(model_output=model_output, sample=sample)
        if use_corrector:
            sample = self._multistep_uni_c_bh_update(
                this_model_output=converted,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = converted
        self.timestep_list[-1] = self._timestep_to_int(timestep)

        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)

        self.last_sample = sample
        prev_sample = self._multistep_uni_p_bh_update(sample=sample, order=self.this_order)
        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        self.step_index += 1

        if not return_dict:
            return (prev_sample,)
        return WanUniPCSchedulerOutput(prev_sample=prev_sample)

    def convert_model_output(self, model_output: mx.array, sample: mx.array) -> mx.array:
        if self.step_index is None:
            raise ValueError("step_index is not initialized.")
        sigma = self.sigmas[self.step_index]
        return sample - sigma * model_output

    @staticmethod
    def sigma_to_alpha_sigma(sigma: mx.array) -> tuple[mx.array, mx.array]:
        return 1 - sigma, sigma

    def _multistep_uni_p_bh_update(self, *, sample: mx.array, order: int) -> mx.array:
        if self.step_index is None:
            raise ValueError("step_index is not initialized.")
        model_outputs = self.model_outputs
        m0 = model_outputs[-1]
        if m0 is None:
            raise ValueError("current model output is missing.")

        sigma_t_raw = self.sigmas[self.step_index + 1]
        sigma_s0_raw = self.sigmas[self.step_index]
        alpha_t, sigma_t = self.sigma_to_alpha_sigma(sigma_t_raw)
        alpha_s0, sigma_s0 = self.sigma_to_alpha_sigma(sigma_s0_raw)
        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s0 = mx.log(alpha_s0) - mx.log(sigma_s0)
        h = lambda_t - lambda_s0

        d1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_outputs[-(i + 1)]
            if mi is None:
                continue
            alpha_si, sigma_si = self.sigma_to_alpha_sigma(self.sigmas[si])
            lambda_si = mx.log(alpha_si) - mx.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            d1s.append((mi - m0) / rk)

        hh = -h
        h_phi_1 = mx.expm1(hh)
        b_h = mx.expm1(hh)

        x_t = sigma_t / sigma_s0 * sample - alpha_t * h_phi_1 * m0
        if d1s:
            d1s_stack = mx.stack(d1s, axis=1)
            if order == 2:
                rhos_p = mx.array([0.5], dtype=sample.dtype)
            else:
                raise ValueError("Only UniPC order <= 2 is implemented for Wan2.2 TI2V.")
            x_t = x_t - alpha_t * b_h * self._weighted_history_sum(rhos_p, d1s_stack)
        return x_t.astype(sample.dtype)

    def _multistep_uni_c_bh_update(
        self,
        *,
        this_model_output: mx.array,
        last_sample: mx.array,
        this_sample: mx.array,
        order: int,
    ) -> mx.array:
        if self.step_index is None:
            raise ValueError("step_index is not initialized.")
        model_outputs = self.model_outputs
        m0 = model_outputs[-1]
        if m0 is None:
            raise ValueError("previous model output is missing.")

        sigma_t_raw = self.sigmas[self.step_index]
        sigma_s0_raw = self.sigmas[self.step_index - 1]
        alpha_t, sigma_t = self.sigma_to_alpha_sigma(sigma_t_raw)
        alpha_s0, sigma_s0 = self.sigma_to_alpha_sigma(sigma_s0_raw)
        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s0 = mx.log(alpha_s0) - mx.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_outputs[-(i + 1)]
            if mi is None:
                continue
            alpha_si, sigma_si = self.sigma_to_alpha_sigma(self.sigmas[si])
            lambda_si = mx.log(alpha_si) - mx.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(float(rk.item()))
            d1s.append((mi - m0) / rk)
        rks.append(1.0)

        hh = -h
        h_phi_1 = mx.expm1(hh)
        b_h = mx.expm1(hh)

        if order == 1:
            rhos_c = mx.array([0.5], dtype=this_sample.dtype)
        elif order == 2:
            rhos_c = self._solve_order2_rhos(rks=rks, h=hh, b_h=b_h, dtype=this_sample.dtype)
        else:
            raise ValueError("Only UniPC order <= 2 is implemented for Wan2.2 TI2V.")

        x_t = sigma_t / sigma_s0 * last_sample - alpha_t * h_phi_1 * m0
        corr_res = 0
        if d1s:
            corr_res = self._weighted_history_sum(rhos_c[:-1], mx.stack(d1s, axis=1))
        d1_t = this_model_output - m0
        x_t = x_t - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)
        return x_t.astype(this_sample.dtype)

    @staticmethod
    def _weighted_history_sum(weights: mx.array, values: mx.array) -> mx.array:
        shape = (1, weights.shape[0], *([1] * (values.ndim - 2)))
        return mx.sum(values * weights.reshape(shape), axis=1)

    @staticmethod
    def _solve_order2_rhos(*, rks: list[float], h: mx.array, b_h: mx.array, dtype: mx.Dtype) -> mx.array:
        h_value = float(h.item())
        b_h_value = float(b_h.item())
        h_phi_1 = np.expm1(h_value)
        h_phi_k = h_phi_1 / h_value - 1
        b = [h_phi_k / b_h_value]
        h_phi_k = h_phi_k / h_value - 1 / 2
        b.append(h_phi_k * 2 / b_h_value)
        r = np.array([[1.0, 1.0], [rks[0], 1.0]], dtype=np.float64)
        return mx.array(np.linalg.solve(r, np.array(b, dtype=np.float64)), dtype=dtype)

    def _index_for_timestep(self, timestep: int | mx.array) -> int:
        timestep_int = self._timestep_to_int(timestep)
        timesteps = np.array(self.timesteps)
        matches = np.where(timesteps == timestep_int)[0]
        if len(matches) == 0:
            return len(timesteps) - 1
        if len(matches) > 1:
            return int(matches[1])
        return int(matches[0])

    @staticmethod
    def _timestep_to_int(timestep: int | mx.array) -> int:
        if hasattr(timestep, "item"):
            return int(timestep.item())
        return int(timestep)
