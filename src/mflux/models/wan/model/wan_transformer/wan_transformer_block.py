import os
from typing import Any

import mlx.core as mx
from mlx import nn

from mflux.models.wan.model.wan_transformer.wan_activation import WanActivation
from mflux.models.wan.model.wan_transformer.wan_attention import WanAttention
from mflux.utils.tensor_health import TensorHealth


class WanFeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.net = [
            nn.Linear(dim, inner_dim, bias=True),
            nn.Linear(inner_dim, dim, bias=True),
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.net[0](hidden_states)
        hidden_states = WanActivation.gelu_tanh(hidden_states)
        return self.net[1](hidden_states)


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
        )
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
        )
        self.norm2 = nn.LayerNorm(dim, eps=eps, affine=True) if cross_attn_norm else None
        self.ffn = WanFeedForward(dim, ffn_dim)
        self.norm3 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.scale_shift_table = mx.random.normal((1, 6, dim)) / dim**0.5

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb: mx.array,
        rotary_emb: tuple[mx.array, mx.array],
        block_name: str | None = None,
        block_health_context: Any | None = None,
    ) -> mx.array:
        block_name = block_name or "block"
        block_health_enabled = self._block_health_enabled()
        if temb.ndim == 4:
            modulation = self.scale_shift_table[None, :, :, :] + temb.astype(mx.float32)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = mx.split(
                modulation,
                6,
                axis=2,
            )
            shift_msa = mx.squeeze(shift_msa, axis=2)
            scale_msa = mx.squeeze(scale_msa, axis=2)
            gate_msa = mx.squeeze(gate_msa, axis=2)
            c_shift_msa = mx.squeeze(c_shift_msa, axis=2)
            c_scale_msa = mx.squeeze(c_scale_msa, axis=2)
            c_gate_msa = mx.squeeze(c_gate_msa, axis=2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = mx.split(
                self.scale_shift_table + temb.astype(mx.float32),
                6,
                axis=1,
            )

        norm_hidden_states = self.norm1(hidden_states.astype(mx.float32)) * (1 + scale_msa) + shift_msa
        self._check_tensor_health(
            enabled=block_health_enabled,
            name=f"{block_name}.norm1",
            tensor=norm_hidden_states,
            context=block_health_context,
        )
        attn_output = self.attn1(
            norm_hidden_states.astype(hidden_states.dtype),
            rotary_emb=rotary_emb,
            attention_name=f"{block_name}.attn1",
            block_health_context=block_health_context,
        )
        self._check_tensor_health(
            enabled=block_health_enabled,
            name=f"{block_name}.attn1",
            tensor=attn_output,
            context=block_health_context,
        )
        hidden_states = (hidden_states.astype(mx.float32) + attn_output * gate_msa).astype(hidden_states.dtype)
        self._check_tensor_health(
            enabled=block_health_enabled,
            name=f"{block_name}.attn1_residual",
            tensor=hidden_states,
            context=block_health_context,
        )

        norm_hidden_states = hidden_states.astype(mx.float32)
        if self.norm2 is not None:
            norm_hidden_states = self.norm2(norm_hidden_states)
        self._check_tensor_health(
            enabled=block_health_enabled,
            name=f"{block_name}.norm2",
            tensor=norm_hidden_states,
            context=block_health_context,
        )
        attn_output = self.attn2(
            norm_hidden_states.astype(hidden_states.dtype),
            encoder_hidden_states,
            attention_name=f"{block_name}.attn2",
            block_health_context=block_health_context,
        )
        self._check_tensor_health(
            enabled=block_health_enabled,
            name=f"{block_name}.attn2",
            tensor=attn_output,
            context=block_health_context,
        )
        hidden_states = hidden_states + attn_output
        self._check_tensor_health(
            enabled=block_health_enabled,
            name=f"{block_name}.attn2_residual",
            tensor=hidden_states,
            context=block_health_context,
        )

        norm_hidden_states = self.norm3(hidden_states.astype(mx.float32)) * (1 + c_scale_msa) + c_shift_msa
        self._check_tensor_health(
            enabled=block_health_enabled,
            name=f"{block_name}.norm3",
            tensor=norm_hidden_states,
            context=block_health_context,
        )
        ff_output = self.ffn(norm_hidden_states.astype(hidden_states.dtype))
        self._check_tensor_health(
            enabled=block_health_enabled,
            name=f"{block_name}.ffn",
            tensor=ff_output,
            context=block_health_context,
        )
        return (hidden_states.astype(mx.float32) + ff_output.astype(mx.float32) * c_gate_msa).astype(
            hidden_states.dtype
        )

    @staticmethod
    def _block_health_enabled() -> bool:
        return os.environ.get("MFLUX_WAN_BLOCK_HEALTH", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
            "detail",
            "detailed",
            "all",
        }

    @staticmethod
    def _check_tensor_health(*, enabled: bool, name: str, tensor: mx.array, context: Any | None) -> None:
        if not enabled:
            return
        TensorHealth.ensure_finite(
            tensor,
            name=f"wan.transformer.{name}",
            phase="wan-transformer-block",
            step=getattr(context, "step", None),
            total_steps=getattr(context, "total_steps", None),
            timestep=getattr(context, "timestep", None),
            denoiser=getattr(context, "denoiser", None),
            guidance=getattr(context, "guidance", None),
        )
