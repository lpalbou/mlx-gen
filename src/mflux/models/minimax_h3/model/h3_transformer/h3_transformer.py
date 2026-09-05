"""MiniMax-H3 omni transformer (diffusers `MiniMaxH3Transformer3DModel`).

One stack of blocks over a single packed sequence holding text, conditioning video rows,
audio rows and target video rows. Attention is dense self-attention with no mask; modality
enters only through the input projections, the per-row AdaLN table row and the two output
heads. The caller builds the layout (see `latent_creator/h3_layout.py`).
"""

import mlx.core as mx
from mlx import nn

from mflux.models.minimax_h3.model.h3_transformer.h3_embedding import (
    H3AdaLayerNormOut,
    H3RotaryPosEmbed,
    H3TimestepEmbedding,
    H3TimestepProjection,
)
from mflux.models.minimax_h3.model.h3_transformer.h3_transformer_block import H3TokenRefiner, H3TransformerBlock

MODALITY_NUM = 3


class MiniMaxH3Transformer(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.in_channels = in_channels
        self.audio_in_channels = audio_in_channels
        video_patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.proj_in = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_proj_in = nn.Linear(audio_in_channels, hidden_size, bias=True)
        self.context_embedder = nn.Linear(text_dim, hidden_size, bias=True)
        self.time_proj = H3TimestepProjection(freq_dim)
        self.time_embedder = H3TimestepEmbedding(freq_dim, time_embed_hidden_dim, time_embed_dim)
        self.rope = H3RotaryPosEmbed(rope_freq_dim, rope_theta)
        self.token_refiner = H3TokenRefiner(
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            ffn_dim,
            num_refiner_layers,
            norm_eps,
            qk_norm_eps,
            final_norm_eps,
        )
        self.transformer_blocks = [
            H3TransformerBlock(
                hidden_size, num_attention_heads, attention_head_dim, ffn_dim, time_embed_dim, norm_eps, qk_norm_eps
            )
            for _ in range(num_layers)
        ]
        self.norm_out = H3AdaLayerNormOut(hidden_size, time_embed_dim, final_norm_eps)
        self.proj_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_proj_out = nn.Linear(hidden_size, audio_in_channels, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        audio_hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        timestep_indices: mx.array,
        token_tags: mx.array,
        position_ids: mx.array,
        video_indices: mx.array,
        audio_indices: mx.array,
        text_indices: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Return the data-ward velocity of the video rows `(B, Nv, C*patch)` and audio rows `(B, Na, Ca)`."""
        sequence_length = position_ids.shape[0]
        rotary_emb = self.rope(position_ids)

        video_embeds = self.proj_in(hidden_states.astype(self.proj_in.weight.dtype))
        audio_embeds = self.audio_proj_in(audio_hidden_states.astype(self.audio_proj_in.weight.dtype))
        text_embeds = self.context_embedder(encoder_hidden_states.astype(self.context_embedder.weight.dtype))
        text_embeds = self.token_refiner(text_embeds)

        packed = mx.zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]), dtype=text_embeds.dtype)
        packed[:, text_indices, :] = text_embeds
        packed[:, video_indices, :] = video_embeds.astype(text_embeds.dtype)
        packed[:, audio_indices, :] = audio_embeds.astype(text_embeds.dtype)

        temb = self.time_embedder(self.time_proj(timestep).astype(self.time_embedder.linear_1.weight.dtype))
        adaln_indices = timestep_indices * MODALITY_NUM + token_tags

        for block in self.transformer_blocks:
            packed = block(packed, temb, adaln_indices, rotary_emb)

        packed = self.norm_out(packed, temb, timestep_indices).astype(self.proj_out.weight.dtype)
        video_output = mx.take(self.proj_out(packed), video_indices, axis=1)
        audio_output = mx.take(self.audio_proj_out(packed), audio_indices, axis=1)
        return video_output, audio_output
