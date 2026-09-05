import math

import mlx.core as mx
from mlx import nn

from mflux.models.minimax_h3.model.h3_precision import rowwise


def apply_rotary_emb(hidden_states: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the leading `rotary_dim` channels of every head (rotate-half convention), pass the rest through.

    `hidden_states` is `(B, S, H, D)`; `cos` / `sin` are `(S, rotary_dim)`.
    """
    rotary_dim = cos.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    cos = cos.astype(hidden_states.dtype)[None, :, None, :]
    sin = sin.astype(hidden_states.dtype)[None, :, None, :]
    half = rotary_dim // 2
    x1, x2 = rotary[..., :half], rotary[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return mx.concatenate([rotary * cos + rotated * sin, passthrough], axis=-1)


class H3Attention(nn.Module):
    """Full self-attention over the packed sequence with per-head query/key RMSNorm. No cross-attention."""

    def __init__(self, hidden_size: int, heads: int, dim_head: int, qk_norm_eps: float = 1e-5):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        inner_dim = heads * dim_head
        self.scale = 1.0 / math.sqrt(dim_head)
        self.to_q = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.to_out = [nn.Linear(inner_dim, hidden_size, bias=False)]

    def __call__(self, hidden_states: mx.array, rotary_emb: tuple[mx.array, mx.array] | None = None) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        # `rowwise`: q8 linears see up to ~38k packed rows at 768p (see `h3_precision.QUANTIZED_MATMUL_MAX_ROWS`).
        query = rowwise(self.to_q, hidden_states).reshape(batch_size, seq_len, self.heads, self.head_dim)
        key = rowwise(self.to_k, hidden_states).reshape(batch_size, seq_len, self.heads, self.head_dim)
        value = rowwise(self.to_v, hidden_states).reshape(batch_size, seq_len, self.heads, self.head_dim)
        query = self.norm_q(query)
        key = self.norm_k(key)
        if rotary_emb is not None:
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)
        attention = mx.fast.scaled_dot_product_attention(
            query.transpose(0, 2, 1, 3),
            key.transpose(0, 2, 1, 3),
            value.transpose(0, 2, 1, 3),
            scale=self.scale,
        )
        attention = attention.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.heads * self.head_dim)
        return rowwise(self.to_out[0], attention.astype(query.dtype))


class H3SwiGLU(nn.Module):
    """diffusers `SwiGLU`: one projection to `2 * inner`, split into value and gate, `value * silu(gate)`."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states, gate = mx.split(rowwise(self.proj, hidden_states), 2, axis=-1)
        return hidden_states * nn.silu(gate)


class H3FeedForward(nn.Module):
    """diffusers `FeedForward(activation_fn="swiglu", bias=False)`; `net.1` is the parameter-free dropout slot."""

    def __init__(self, dim: int, inner_dim: int, bias: bool = False):
        super().__init__()
        self.net = [H3SwiGLU(dim, inner_dim, bias=bias), nn.Identity(), nn.Linear(inner_dim, dim, bias=bias)]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return rowwise(self.net[2], self.net[0](hidden_states))
