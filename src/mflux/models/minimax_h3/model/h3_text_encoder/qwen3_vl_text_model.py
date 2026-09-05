"""Qwen3-VL language model, as MiniMax-H3 uses it for text conditioning.

MiniMax-H3 conditions on `hidden_states[50]` of Qwen3-VL-32B — the output of decoder layer 49,
before the final norm — so only the first `num_hidden_layers` layers requested are built and
the final norm is never applied. Rotary embeddings are the 3-axis interleaved M-RoPE of
Qwen3-VL (`mrope_section`); for pure text all three axes carry the same positions.
Module names follow `Qwen3VLTextModel` (`embed_tokens`, `layers.N.self_attn.q_proj`, ...).
"""

import math

import mlx.core as mx
from mlx import nn


class Qwen3VLTextRotaryEmbedding:
    def __init__(self, head_dim: int, rope_theta: float, mrope_section: tuple[int, ...]):
        self.mrope_section = tuple(mrope_section)
        self._inv_freq = 1.0 / (rope_theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        """`position_ids` is `(3, B, S)` (t, h, w); returns cos/sin of shape `(B, S, head_dim)`."""
        freqs = position_ids.astype(mx.float32)[:, :, :, None] * self._inv_freq[None, None, None, :]  # (3, B, S, D/2)
        thw = freqs[0]
        # Interleaved recomposition: axis 1 (h) fills indices 1, 4, 7, ...; axis 2 (w) fills 2, 5, 8, ... up to 3 * section.
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            index = mx.arange(offset, length, 3)
            thw[..., index] = freqs[axis][..., index]
        emb = mx.concatenate([thw, thw], axis=-1)
        return mx.cos(emb), mx.sin(emb)


def _rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


class Qwen3VLTextAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, rms_norm_eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        batch, seq_len, _ = x.shape
        query = self.q_norm(self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        key = self.k_norm(self.k_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        value = self.v_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = cos[:, None].astype(query.dtype), sin[:, None].astype(query.dtype)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        out = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=1.0 / math.sqrt(self.head_dim), mask="causal"
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1))


class Qwen3VLTextMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3VLTextDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps)
        self.mlp = Qwen3VLTextMLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3VLTextModel(nn.Module):
    """Decoder layers `0 .. num_hidden_layers - 1`; `__call__` returns the pre-norm hidden state after the last one."""

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 5120,
        num_hidden_layers: int = 50,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 25600,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 5_000_000.0,
        mrope_section: tuple[int, ...] = (24, 20, 20),
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Qwen3VLTextDecoderLayer(
                hidden_size, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, rms_norm_eps
            )
            for _ in range(num_hidden_layers)
        ]
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(head_dim, rope_theta, mrope_section)

    @staticmethod
    def text_position_ids(batch: int, seq_len: int) -> mx.array:
        """Pure-text M-RoPE positions: the same `arange` on all three axes, `(3, B, S)`."""
        return mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32)[None, None, :], (3, batch, seq_len))

    def __call__(
        self,
        input_ids: mx.array,
        position_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        deepstack_visual_embeds: list[mx.array] | None = None,
        visual_positions: mx.array | None = None,
    ) -> mx.array:
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if position_ids is None:
            position_ids = self.text_position_ids(x.shape[0], x.shape[1])
        cos, sin = self.rotary_emb(position_ids)
        for index, layer in enumerate(self.layers):
            x = layer(x, cos, sin)
            if deepstack_visual_embeds is not None and index < len(deepstack_visual_embeds):
                # DeepStack: add the visual features of stage `index` to the vision-token rows.
                x[:, visual_positions, :] = x[:, visual_positions, :] + deepstack_visual_embeds[index].astype(x.dtype)
        return x
