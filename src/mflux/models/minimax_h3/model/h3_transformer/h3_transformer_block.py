import mlx.core as mx
from mlx import nn

from mflux.models.minimax_h3.model.h3_transformer.h3_attention import H3Attention, H3FeedForward
from mflux.models.minimax_h3.model.h3_transformer.h3_embedding import H3AdaLayerNormModulation


class H3TokenRefinerBlock(nn.Module):
    """Plain pre-norm block refining the projected text stream: no AdaLN, no rotary embedding."""

    def __init__(self, hidden_size: int, heads: int, dim_head: int, ffn_dim: int, norm_eps: float, qk_norm_eps: float):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = H3Attention(hidden_size, heads, dim_head, qk_norm_eps=qk_norm_eps)
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = H3FeedForward(hidden_size, ffn_dim)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.ff(self.norm2(hidden_states))


class H3TokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dim_head: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
    ):
        super().__init__()
        self.refiner_blocks = [
            H3TokenRefinerBlock(hidden_size, heads, dim_head, ffn_dim, norm_eps, qk_norm_eps) for _ in range(num_layers)
        ]
        self.final_norm = nn.RMSNorm(hidden_size, eps=final_norm_eps)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class H3TransformerBlock(nn.Module):
    """Pre-norm self-attention and feed-forward, each modulated by AdaLN rows selected per packed row."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dim_head: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = H3Attention(hidden_size, heads, dim_head, qk_norm_eps=qk_norm_eps)
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = H3FeedForward(hidden_size, ffn_dim)
        self.adaln_proj = H3AdaLayerNormModulation(time_embed_dim, hidden_size)

    def __call__(
        self, hidden_states: mx.array, temb: mx.array, adaln_indices: mx.array, rotary_emb: tuple[mx.array, mx.array]
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            mx.take(table, adaln_indices, axis=0) for table in self.adaln_proj(temb)
        )
        normed = self.norm1(hidden_states) * (1.0 + scale_msa) + shift_msa
        hidden_states = hidden_states + gate_msa * self.attn(normed, rotary_emb)
        normed = self.norm2(hidden_states) * (1.0 + scale_mlp) + shift_mlp
        return hidden_states + gate_mlp * self.ff(normed)
