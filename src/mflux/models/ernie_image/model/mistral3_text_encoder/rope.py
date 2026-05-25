import math

import mlx.core as mx
from mlx import nn


class Mistral3YarnRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        base: float = 1000000.0,
        factor: float = 16.0,
        original_max_position_embeddings: int = 16384,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float | None = 1.0,
        mscale_all_dim: float | None = 1.0,
        attention_factor: float | None = None,
        truncate: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.factor = factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.attention_factor = self._attention_factor(factor, mscale, mscale_all_dim, attention_factor)
        self.inv_freq = self._compute_inv_freq(truncate=truncate)

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        freqs = mx.expand_dims(position_ids.astype(mx.float32), axis=-1) * self.inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self.attention_factor
        sin = mx.sin(emb) * self.attention_factor
        return cos.astype(x.dtype), sin.astype(x.dtype)

    def _compute_inv_freq(self, truncate: bool) -> mx.array:
        pos_freqs = self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.factor * pos_freqs)
        low, high = self._find_correction_range(truncate=truncate)
        extrapolation_factor = 1 - self._linear_ramp_factor(low, high, self.dim // 2)
        return inv_freq_interpolation * (1 - extrapolation_factor) + inv_freq_extrapolation * extrapolation_factor

    def _find_correction_range(self, truncate: bool) -> tuple[float, float]:
        low = self._find_correction_dim(self.beta_fast)
        high = self._find_correction_dim(self.beta_slow)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, self.dim - 1)

    def _find_correction_dim(self, num_rotations: float) -> float:
        return (self.dim * math.log(self.original_max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(self.base)
        )

    @staticmethod
    def _linear_ramp_factor(min_value: float, max_value: float, dim: int) -> mx.array:
        if min_value == max_value:
            max_value += 0.001
        linear = (mx.arange(dim, dtype=mx.float32) - min_value) / (max_value - min_value)
        return mx.minimum(mx.maximum(linear, 0), 1)

    @staticmethod
    def _attention_factor(
        factor: float,
        mscale: float | None,
        mscale_all_dim: float | None,
        attention_factor: float | None,
    ) -> float:
        if attention_factor is not None:
            return attention_factor
        if mscale and mscale_all_dim:
            return Mistral3YarnRotaryEmbedding._mscale(factor, mscale) / Mistral3YarnRotaryEmbedding._mscale(
                factor, mscale_all_dim
            )
        return Mistral3YarnRotaryEmbedding._mscale(factor)

    @staticmethod
    def _mscale(scale: float, mscale: float = 1.0) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0
