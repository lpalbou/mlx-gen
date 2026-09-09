import math

import mlx.core as mx
from mlx import nn

from mflux.models.minimax_h3.model.h3_precision import linear_input_dtype


class H3RotaryPosEmbed(nn.Module):
    """3-axis rotary embedding over `(t, h, w)`; one `rope_freq_dim`-long `inv_freq` shared by the three axes."""

    def __init__(self, rope_freq_dim: int = 16, rope_theta: float = 10000.0):
        super().__init__()
        self.rope_freq_dim = rope_freq_dim
        exponent = mx.arange(0, 2 * rope_freq_dim, 2, dtype=mx.float32) / (2 * rope_freq_dim)
        self._inv_freq = 1.0 / (rope_theta**exponent)

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        freqs = position_ids.astype(mx.float32)[:, :, None] * self._inv_freq[None, None, :]  # (S, 3, F)
        freqs = freqs.reshape(position_ids.shape[0], 3 * self.rope_freq_dim)
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(freqs), mx.sin(freqs)


class H3TimestepProjection(nn.Module):
    """diffusers `Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)` on timesteps in `[0, 1]`."""

    def __init__(self, num_channels: int, max_period: int = 10000):
        super().__init__()
        self.num_channels = num_channels
        self.max_period = max_period

    def __call__(self, timesteps: mx.array) -> mx.array:
        half_dim = self.num_channels // 2
        exponent = -math.log(self.max_period) * mx.arange(0, half_dim, dtype=mx.float32) / half_dim
        emb = timesteps.astype(mx.float32)[:, None] * mx.exp(exponent)[None, :]
        return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)


class H3TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, out_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.linear_2 = nn.Linear(time_embed_dim, out_dim, bias=True)

    def __call__(self, sample: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(sample)))


class H3AdaLayerNormModulation(nn.Module):
    """Projects the timestep embedding to six `(timestep, modality)` modulation tables for one block.

    Rows are laid out `[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]`, addressed by
    `timestep_indices * 3 + token_tags`.
    """

    def __init__(self, time_embed_dim: int, hidden_size: int, modality_num: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(time_embed_dim, 6 * hidden_size * modality_num, bias=True)

    def __call__(self, temb: mx.array) -> tuple[mx.array, ...]:
        modulation = self.linear(nn.silu(temb.astype(mx.float32)).astype(linear_input_dtype(self.linear)))
        modulation = modulation.reshape(-1, 6 * self.hidden_size)
        return tuple(mx.split(modulation, 6, axis=-1))


class H3AdaLayerNormOut(nn.Module):
    """Final RMSNorm of the packed sequence, shift/scale modulated per row by its timestep."""

    def __init__(self, hidden_size: int, time_embed_dim: int, eps: float):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array, temb: mx.array, timestep_indices: mx.array) -> mx.array:
        shift, scale = mx.split(
            self.linear(nn.silu(temb.astype(mx.float32)).astype(linear_input_dtype(self.linear))), 2, axis=-1
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + mx.take(scale, timestep_indices, axis=0)) + mx.take(
            shift, timestep_indices, axis=0
        )
