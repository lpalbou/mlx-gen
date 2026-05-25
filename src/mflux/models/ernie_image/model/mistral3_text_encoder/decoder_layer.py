import mlx.core as mx
from mlx import nn

from mflux.models.ernie_image.model.mistral3_text_encoder.attention import Mistral3Attention
from mflux.models.ernie_image.model.mistral3_text_encoder.mlp import Mistral3MLP
from mflux.models.ernie_image.model.mistral3_text_encoder.rms_norm import Mistral3RMSNorm


class Mistral3DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3072,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9216,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-5,
        llama_4_scaling_beta: float = 0.1,
        original_max_position_embeddings: int = 16384,
    ):
        super().__init__()
        self.input_layernorm = Mistral3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Mistral3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Mistral3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            llama_4_scaling_beta=llama_4_scaling_beta,
            original_max_position_embeddings=original_max_position_embeddings,
        )
        self.mlp = Mistral3MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        position_ids: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
