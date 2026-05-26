import math

import mlx.core as mx
from mlx import nn

from mflux.models.wan.model.wan_transformer.wan_activation import WanActivation


class WanTimestepProjection(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0.0,
        scale: float = 1.0,
        max_period: int = 10000,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def __call__(self, timesteps: mx.array) -> mx.array:
        if timesteps.ndim != 1:
            raise ValueError(f"Timesteps should be 1D, got {timesteps.shape}")
        half_dim = self.num_channels // 2
        exponent = -math.log(self.max_period) * mx.arange(0, half_dim, dtype=mx.float32)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = timesteps[:, None].astype(mx.float32) * mx.exp(exponent)[None, :]
        emb = self.scale * emb
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        if self.flip_sin_to_cos:
            emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
        if self.num_channels % 2 == 1:
            emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
        return emb


class WanTimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        sample = nn.silu(sample)
        return self.linear_2(sample)


class WanTextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, caption: mx.array) -> mx.array:
        hidden_states = self.linear_1(caption)
        hidden_states = WanActivation.gelu_tanh(hidden_states)
        return self.linear_2(hidden_states)


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ):
        super().__init__()
        self.timesteps_proj = WanTimestepProjection(
            num_channels=time_freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedder = WanTimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.time_proj = nn.Linear(dim, time_proj_dim, bias=True)
        self.text_embedder = WanTextProjection(text_embed_dim, dim)

    def __call__(
        self,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        timestep_seq_len: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.reshape(-1, timestep_seq_len, timestep.shape[-1])
        temb = self.time_embedder(timestep).astype(encoder_hidden_states.dtype)
        timestep_proj = self.time_proj(nn.silu(temb))
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        return temb, timestep_proj, encoder_hidden_states


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim
        freqs_cos = []
        freqs_sin = []
        for dim in (t_dim, h_dim, w_dim):
            cos, sin = self._get_1d_rotary_pos_embed(dim=dim, max_seq_len=max_seq_len, theta=theta)
            freqs_cos.append(cos)
            freqs_sin.append(sin)
        self.freqs_cos = mx.concatenate(freqs_cos, axis=1)
        self.freqs_sin = mx.concatenate(freqs_sin, axis=1)

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        _, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
        cos_t, cos_h, cos_w = mx.split(self.freqs_cos, [self.t_dim, self.t_dim + self.h_dim], axis=1)
        sin_t, sin_h, sin_w = mx.split(self.freqs_sin, [self.t_dim, self.t_dim + self.h_dim], axis=1)

        cos_f = mx.broadcast_to(cos_t[:ppf].reshape(ppf, 1, 1, -1), (ppf, pph, ppw, self.t_dim))
        cos_h = mx.broadcast_to(cos_h[:pph].reshape(1, pph, 1, -1), (ppf, pph, ppw, self.h_dim))
        cos_w = mx.broadcast_to(cos_w[:ppw].reshape(1, 1, ppw, -1), (ppf, pph, ppw, self.w_dim))
        sin_f = mx.broadcast_to(sin_t[:ppf].reshape(ppf, 1, 1, -1), (ppf, pph, ppw, self.t_dim))
        sin_h = mx.broadcast_to(sin_h[:pph].reshape(1, pph, 1, -1), (ppf, pph, ppw, self.h_dim))
        sin_w = mx.broadcast_to(sin_w[:ppw].reshape(1, 1, ppw, -1), (ppf, pph, ppw, self.w_dim))

        freqs_cos = mx.concatenate([cos_f, cos_h, cos_w], axis=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = mx.concatenate([sin_f, sin_h, sin_w], axis=-1).reshape(1, ppf * pph * ppw, 1, -1)
        return freqs_cos, freqs_sin

    @staticmethod
    def _get_1d_rotary_pos_embed(dim: int, max_seq_len: int, theta: float) -> tuple[mx.array, mx.array]:
        positions = mx.arange(max_seq_len, dtype=mx.float32)
        freqs = mx.arange(0, dim, 2, dtype=mx.float32) / dim
        freqs = 1.0 / (theta**freqs)
        freqs = positions[:, None] * freqs[None, :]
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        cos = mx.repeat(cos, repeats=2, axis=1)
        sin = mx.repeat(sin, repeats=2, axis=1)
        return cos, sin
