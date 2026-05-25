import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.ernie_image.model.mistral3_text_encoder.decoder_layer import Mistral3DecoderLayer
from mflux.models.ernie_image.model.mistral3_text_encoder.rms_norm import Mistral3RMSNorm
from mflux.models.ernie_image.model.mistral3_text_encoder.rope import Mistral3YarnRotaryEmbedding


class Mistral3TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 3072,
        num_hidden_layers: int = 26,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9216,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1000000.0,
        rope_factor: float = 16.0,
        original_max_position_embeddings: int = 16384,
        llama_4_scaling_beta: float = 0.1,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Mistral3DecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                llama_4_scaling_beta=llama_4_scaling_beta,
                original_max_position_embeddings=original_max_position_embeddings,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = Mistral3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Mistral3YarnRotaryEmbedding(
            dim=head_dim,
            base=rope_theta,
            factor=rope_factor,
            original_max_position_embeddings=original_max_position_embeddings,
        )

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        position_ids = mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32)[None, :], (batch_size, seq_len))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = self._causal_mask(attention_mask, hidden_states, seq_len)

        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            all_hidden_states.append(hidden_states)

        return all_hidden_states[-2].astype(ModelConfig.precision)

    @staticmethod
    def _causal_mask(attention_mask: mx.array | None, hidden_states: mx.array, seq_len: int) -> mx.array:
        batch_size = hidden_states.shape[0]
        dtype = hidden_states.dtype
        index = mx.arange(seq_len, dtype=mx.int32)
        allowed = index[None, :] <= index[:, None]
        causal_mask = mx.where(
            allowed,
            mx.zeros((seq_len, seq_len), dtype=dtype),
            mx.full((seq_len, seq_len), -float("inf"), dtype=dtype),
        )
        causal_mask = causal_mask[None, None, :, :]
        if attention_mask is None:
            return causal_mask
        padding_mask = mx.where(
            attention_mask[:, None, None, :] == 1,
            mx.zeros((batch_size, 1, 1, seq_len), dtype=dtype),
            mx.full((batch_size, 1, 1, seq_len), -float("inf"), dtype=dtype),
        )
        return causal_mask + padding_mask
