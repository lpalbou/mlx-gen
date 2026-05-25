import mlx.core as mx


class ErnieImageScheduler:
    def __init__(self, num_inference_steps: int, shift: float = 4.0, num_train_timesteps: int = 1000):
        raw_sigmas = mx.linspace(1.0, 0.0, num_inference_steps + 1, dtype=mx.float32)[:-1]
        sigmas = shift * raw_sigmas / (1 + (shift - 1) * raw_sigmas)
        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)

    def step(self, noise: mx.array, timestep: int, latents: mx.array) -> mx.array:
        dt = (self.sigmas[timestep + 1] - self.sigmas[timestep]).astype(latents.dtype)
        return latents + noise.astype(latents.dtype) * dt
