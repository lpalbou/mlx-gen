import math

import mlx.core as mx
from mlx import nn

from mflux.models.wan.model.wan_transformer.wan_embedding import WanRotaryPosEmbed, WanTimeTextImageEmbedding
from mflux.models.wan.model.wan_transformer.wan_transformer_block import WanTransformerBlock


class WanTransformer(nn.Module):
    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        in_channels: int = 48,
        out_channels: int | None = 48,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 14336,
        num_layers: int = 30,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels,
            self.inner_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=self.inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=self.inner_dim * 6,
            text_embed_dim=text_dim,
        )
        self.blocks = [
            WanTransformerBlock(
                self.inner_dim,
                ffn_dim,
                num_attention_heads,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
                added_kv_proj_dim=added_kv_proj_dim,
            )
            for _ in range(num_layers)
        ]
        self.norm_out = nn.LayerNorm(self.inner_dim, eps=eps, affine=False)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels * math.prod(patch_size), bias=True)
        self.scale_shift_table = mx.random.normal((1, 2, self.inner_dim)) / self.inner_dim**0.5

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)
        hidden_states = self._patch_embed(hidden_states)

        if timestep.ndim == 2:
            timestep_seq_len = timestep.shape[1]
            timestep = timestep.reshape(-1)
        else:
            timestep_seq_len = None

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            timestep_seq_len=timestep_seq_len,
        )
        if timestep_seq_len is not None:
            timestep_proj = timestep_proj.reshape(batch_size, timestep_seq_len, 6, -1)
        else:
            timestep_proj = timestep_proj.reshape(batch_size, 6, -1)

        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        hidden_states = self._project_out(hidden_states, temb)
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = mx.transpose(hidden_states, (0, 7, 1, 4, 2, 5, 3, 6))
        return hidden_states.reshape(batch_size, -1, num_frames, height, width)

    def _patch_embed(self, hidden_states: mx.array) -> mx.array:
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 4, 1))
        hidden_states = self.patch_embedding(hidden_states)
        batch_size, frames, height, width, channels = hidden_states.shape
        return hidden_states.reshape(batch_size, frames * height * width, channels)

    def _project_out(self, hidden_states: mx.array, temb: mx.array) -> mx.array:
        if temb.ndim == 3:
            shift, scale = mx.split(self.scale_shift_table[None, :, :, :] + temb[:, :, None, :], 2, axis=2)
            shift = mx.squeeze(shift, axis=2)
            scale = mx.squeeze(scale, axis=2)
        else:
            shift, scale = mx.split(self.scale_shift_table + temb[:, None, :], 2, axis=1)
        hidden_states = self.norm_out(hidden_states.astype(mx.float32)) * (1 + scale) + shift
        return self.proj_out(hidden_states.astype(temb.dtype))
