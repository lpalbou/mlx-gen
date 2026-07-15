import math
import os
from dataclasses import dataclass

import mlx.core as mx
from mlx import nn

from mflux.models.wan.model.wan_transformer.wan_embedding import WanRotaryPosEmbed, WanTimeTextImageEmbedding
from mflux.models.wan.model.wan_transformer.wan_transformer_block import (
    WanTransformerBlock,
    WanVACETransformerBlock,
)
from mflux.utils.tensor_health import TensorHealth


@dataclass(frozen=True, kw_only=True)
class WanBlockHealthContext:
    step: int | None = None
    total_steps: int | None = None
    timestep: int | float | None = None
    denoiser: str | None = None
    guidance: float | None = None


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
        vace_layers: list[int] | None = None,
        vace_in_channels: int = 96,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.vace_layers = list(vace_layers) if vace_layers else None

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
        if self.vace_layers is not None:
            if max(self.vace_layers) >= num_layers:
                raise ValueError(f"VACE layers {self.vace_layers} exceed the transformer layer count {num_layers}.")
            if 0 not in self.vace_layers:
                raise ValueError("VACE layers must include layer 0.")
            self.vace_patch_embedding = nn.Conv3d(
                vace_in_channels,
                self.inner_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            )
            self.vace_blocks = [
                WanVACETransformerBlock(
                    self.inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                    added_kv_proj_dim=added_kv_proj_dim,
                    apply_input_projection=index == 0,
                )
                for index in range(len(self.vace_layers))
            ]
        self.norm_out = nn.LayerNorm(self.inner_dim, eps=eps, affine=False)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels * math.prod(patch_size), bias=True)
        self.scale_shift_table = mx.random.normal((1, 2, self.inner_dim)) / self.inner_dim**0.5

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        clear_cache_each_block: bool = False,
        block_health_context: WanBlockHealthContext | None = None,
        control_hidden_states: mx.array | None = None,
        control_hidden_states_scale: list[float] | None = None,
    ) -> mx.array:
        if (control_hidden_states is not None) != (self.vace_layers is not None):
            raise ValueError(
                "control_hidden_states must be provided exactly when the transformer is configured with VACE layers. "
                "If you are running a Wan VACE checkpoint, use the WanVace runtime "
                "(CLI: mlxgen-generate-wan --model wan2.1-vace-1.3b)."
            )
        batch_size, _, num_frames, height, width = hidden_states.shape
        if hidden_states.shape[1] != self.in_channels:
            raise ValueError(
                "Wan transformer input channel mismatch: "
                f"got {hidden_states.shape[1]} channels, expected {self.in_channels}."
            )
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)
        hidden_states = self._patch_embed(hidden_states)
        self._check_block_health(
            enabled=self._block_health_enabled(),
            name="patch_embedding",
            tensor=hidden_states,
            context=block_health_context,
        )

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
        self._check_block_health(
            enabled=self._block_health_enabled(),
            name="condition_embedder.temb",
            tensor=temb,
            context=block_health_context,
        )
        self._check_block_health(
            enabled=self._block_health_enabled(),
            name="condition_embedder.timestep_proj",
            tensor=timestep_proj,
            context=block_health_context,
        )
        self._check_block_health(
            enabled=self._block_health_enabled(),
            name="condition_embedder.encoder_hidden_states",
            tensor=encoder_hidden_states,
            context=block_health_context,
        )

        control_hints = None
        if control_hidden_states is not None:
            control_hints = self._vace_control_hints(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                control_hidden_states=control_hidden_states,
                control_hidden_states_scale=control_hidden_states_scale,
                timestep_proj=timestep_proj,
                rotary_emb=rotary_emb,
            )

        block_health_enabled = self._block_health_enabled()
        for block_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                block_name=f"blocks.{block_index}",
                block_health_context=block_health_context,
            )
            if control_hints is not None and block_index in control_hints:
                hint, scale = control_hints[block_index]
                hidden_states = hidden_states + hint * scale
            self._check_block_health(
                enabled=block_health_enabled,
                name=f"blocks.{block_index}.hidden_states",
                tensor=hidden_states,
                context=block_health_context,
            )
            if clear_cache_each_block:
                mx.eval(hidden_states)
                mx.clear_cache()

        hidden_states = self._project_out(hidden_states, temb)
        self._check_block_health(
            enabled=block_health_enabled,
            name="proj_out",
            tensor=hidden_states,
            context=block_health_context,
        )
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

    def _vace_control_hints(
        self,
        *,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        control_hidden_states: mx.array,
        control_hidden_states_scale: list[float] | None,
        timestep_proj: mx.array,
        rotary_emb: tuple[mx.array, mx.array],
    ) -> dict[int, tuple[mx.array, float]]:
        scales = control_hidden_states_scale
        if scales is None:
            scales = [1.0] * len(self.vace_layers)
        if len(scales) != len(self.vace_layers):
            raise ValueError(
                f"Length of control_hidden_states_scale {len(scales)} must equal "
                f"the number of VACE layers {len(self.vace_layers)}."
            )
        control = mx.transpose(control_hidden_states, (0, 2, 3, 4, 1))
        control = self.vace_patch_embedding(control)
        batch_size, frames, height, width, channels = control.shape
        control = control.reshape(batch_size, frames * height * width, channels)
        # The reference zero-pads the control sequence to the main sequence length so that
        # block 0's proj_in(control) + hidden_states is shape-compatible.
        padding = hidden_states.shape[1] - control.shape[1]
        if padding < 0:
            raise ValueError(
                f"VACE control sequence ({control.shape[1]}) is longer than the main sequence "
                f"({hidden_states.shape[1]})."
            )
        if padding > 0:
            control = mx.concatenate(
                [control, mx.zeros((batch_size, padding, channels), dtype=control.dtype)],
                axis=1,
            )
        hints: dict[int, tuple[mx.array, float]] = {}
        for index, block in enumerate(self.vace_blocks):
            conditioning_states, control = block(
                hidden_states,
                encoder_hidden_states,
                control,
                timestep_proj,
                rotary_emb,
            )
            hints[self.vace_layers[index]] = (conditioning_states, float(scales[index]))
        return hints

    def _project_out(self, hidden_states: mx.array, temb: mx.array) -> mx.array:
        if temb.ndim == 3:
            shift, scale = mx.split(self.scale_shift_table[None, :, :, :] + temb[:, :, None, :], 2, axis=2)
            shift = mx.squeeze(shift, axis=2)
            scale = mx.squeeze(scale, axis=2)
        else:
            shift, scale = mx.split(self.scale_shift_table + temb[:, None, :], 2, axis=1)
        hidden_states = self.norm_out(hidden_states.astype(mx.float32)) * (1 + scale) + shift
        return self.proj_out(hidden_states.astype(temb.dtype))

    @staticmethod
    def _block_health_enabled() -> bool:
        return os.environ.get("MFLUX_WAN_BLOCK_HEALTH", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
            "blocks",
            "detail",
            "detailed",
            "all",
        }

    @staticmethod
    def _check_block_health(
        *,
        enabled: bool,
        name: str,
        tensor: mx.array,
        context: WanBlockHealthContext | None,
    ) -> None:
        if not enabled:
            return
        TensorHealth.ensure_finite(
            tensor,
            name=f"wan.transformer.{name}",
            phase="wan-transformer-block",
            step=None if context is None else context.step,
            total_steps=None if context is None else context.total_steps,
            timestep=None if context is None else context.timestep,
            denoiser=None if context is None else context.denoiser,
            guidance=None if context is None else context.guidance,
        )
