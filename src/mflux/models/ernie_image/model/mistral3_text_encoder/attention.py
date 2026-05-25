import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


class Mistral3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3072,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        llama_4_scaling_beta: float = 0.1,
        original_max_position_embeddings: int = 16384,
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scale = 1.0 / math.sqrt(head_dim)
        self.llama_4_scaling_beta = llama_4_scaling_beta
        self.original_max_position_embeddings = original_max_position_embeddings

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None,
        position_ids: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value = self.v_proj(hidden_states).reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        query = mx.transpose(query, axes=(0, 2, 1, 3))
        key = mx.transpose(key, axes=(0, 2, 1, 3))
        value = mx.transpose(value, axes=(0, 2, 1, 3))

        query, key = Mistral3Attention._apply_rotary_pos_emb(query, key, *position_embeddings)
        query = query * self._llama_4_attention_scale(position_ids).astype(query.dtype)

        if self.num_key_value_groups > 1:
            key = self._repeat_kv(key, self.num_key_value_groups)
            value = self._repeat_kv(value, self.num_key_value_groups)

        attention_output = scaled_dot_product_attention(
            query.astype(mx.float32),
            key.astype(mx.float32),
            value.astype(mx.float32),
            scale=self.scale,
            mask=attention_mask,
        )
        attention_output = attention_output.astype(query.dtype)
        attention_output = mx.transpose(attention_output, axes=(0, 2, 1, 3)).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        return self.o_proj(attention_output)

    def _llama_4_attention_scale(self, position_ids: mx.array) -> mx.array:
        scale = 1 + self.llama_4_scaling_beta * mx.log(
            1 + mx.floor(position_ids.astype(mx.float32) / self.original_max_position_embeddings)
        )
        return scale[:, None, :, None]

    @staticmethod
    def _repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
        batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = mx.expand_dims(hidden_states, axis=2)
        hidden_states = mx.broadcast_to(hidden_states, (batch_size, num_key_value_heads, n_rep, seq_len, head_dim))
        return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, seq_len, head_dim)

    @staticmethod
    def _apply_rotary_pos_emb(
        query: mx.array,
        key: mx.array,
        cos: mx.array,
        sin: mx.array,
        unsqueeze_dim: int = 1,
    ) -> tuple[mx.array, mx.array]:
        cos = mx.expand_dims(cos, axis=unsqueeze_dim)
        sin = mx.expand_dims(sin, axis=unsqueeze_dim)
        query_embed = (query * cos) + (Mistral3Attention._rotate_half(query) * sin)
        key_embed = (key * cos) + (Mistral3Attention._rotate_half(key) * sin)
        return query_embed, key_embed

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)
