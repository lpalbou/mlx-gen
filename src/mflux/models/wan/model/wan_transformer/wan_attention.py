import os
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.utils.tensor_health import TensorHealth


class WanAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float = 1e-5,
        added_kv_proj_dim: int | None = None,
        cross_attention_dim_head: int | None = None,
    ):
        super().__init__()
        self.inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads
        self.scale = self.dim_head**-0.5

        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = [nn.Linear(self.inner_dim, dim, bias=True)]
        self.norm_q = nn.RMSNorm(self.inner_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.inner_dim, eps=eps)

        self.add_k_proj = None
        self.add_v_proj = None
        self.norm_added_k = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = nn.RMSNorm(self.inner_dim, eps=eps)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
        rotary_emb: tuple[mx.array, mx.array] | None = None,
        attention_name: str | None = None,
        block_health_context: Any | None = None,
    ) -> mx.array:
        attention_name = attention_name or "attention"
        health_enabled = self._block_health_enabled()
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        encoder_hidden_states_img = None
        if self.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        self._check_tensor_health(
            enabled=health_enabled,
            name=f"{attention_name}.to_q",
            tensor=query,
            context=block_health_context,
        )
        self._check_tensor_health(
            enabled=health_enabled,
            name=f"{attention_name}.to_k",
            tensor=key,
            context=block_health_context,
        )
        self._check_tensor_health(
            enabled=health_enabled,
            name=f"{attention_name}.to_v",
            tensor=value,
            context=block_health_context,
        )

        query = self.norm_q(query).reshape(query.shape[0], query.shape[1], self.heads, self.dim_head)
        key = self.norm_k(key).reshape(key.shape[0], key.shape[1], self.heads, self.dim_head)
        value = value.reshape(value.shape[0], value.shape[1], self.heads, self.dim_head)
        self._check_tensor_health(
            enabled=health_enabled,
            name=f"{attention_name}.norm_q",
            tensor=query,
            context=block_health_context,
        )
        self._check_tensor_health(
            enabled=health_enabled,
            name=f"{attention_name}.norm_k",
            tensor=key,
            context=block_health_context,
        )

        if rotary_emb is not None:
            query = self._apply_rotary_emb(query, *rotary_emb)
            key = self._apply_rotary_emb(key, *rotary_emb)
            self._check_tensor_health(
                enabled=health_enabled,
                name=f"{attention_name}.rotary_q",
                tensor=query,
                context=block_health_context,
            )
            self._check_tensor_health(
                enabled=health_enabled,
                name=f"{attention_name}.rotary_k",
                tensor=key,
                context=block_health_context,
            )

        query = mx.transpose(query, (0, 2, 1, 3))
        key = mx.transpose(key, (0, 2, 1, 3))
        value = mx.transpose(value, (0, 2, 1, 3))
        hidden_states = scaled_dot_product_attention(query, key, value, scale=self.scale)
        self._check_tensor_health(
            enabled=health_enabled,
            name=f"{attention_name}.sdpa",
            tensor=hidden_states,
            context=block_health_context,
        )
        hidden_states = mx.transpose(hidden_states, (0, 2, 1, 3)).reshape(
            hidden_states.shape[0], hidden_states.shape[2], self.inner_dim
        )

        if encoder_hidden_states_img is not None:
            image_hidden_states = self._image_attention(
                query,
                encoder_hidden_states_img,
                attention_name=attention_name,
                block_health_context=block_health_context,
            )
            self._check_tensor_health(
                enabled=health_enabled,
                name=f"{attention_name}.image_attention",
                tensor=image_hidden_states,
                context=block_health_context,
            )
            hidden_states = hidden_states + image_hidden_states
            self._check_tensor_health(
                enabled=health_enabled,
                name=f"{attention_name}.combined_attention",
                tensor=hidden_states,
                context=block_health_context,
            )

        hidden_states = self.to_out[0](hidden_states)
        self._check_tensor_health(
            enabled=health_enabled,
            name=f"{attention_name}.to_out",
            tensor=hidden_states,
            context=block_health_context,
        )
        return hidden_states

    def _image_attention(
        self,
        query: mx.array,
        encoder_hidden_states_img: mx.array,
        *,
        attention_name: str,
        block_health_context: Any | None,
    ) -> mx.array:
        if self.add_k_proj is None or self.add_v_proj is None or self.norm_added_k is None:
            raise ValueError("Image attention requested without added key/value projections.")
        health_enabled = self._block_health_enabled()
        key_img = self.norm_added_k(self.add_k_proj(encoder_hidden_states_img))
        value_img = self.add_v_proj(encoder_hidden_states_img)
        self._check_tensor_health(
            enabled=health_enabled,
            name=f"{attention_name}.add_k_proj",
            tensor=key_img,
            context=block_health_context,
        )
        self._check_tensor_health(
            enabled=health_enabled,
            name=f"{attention_name}.add_v_proj",
            tensor=value_img,
            context=block_health_context,
        )
        key_img = key_img.reshape(key_img.shape[0], key_img.shape[1], self.heads, self.dim_head)
        value_img = value_img.reshape(value_img.shape[0], value_img.shape[1], self.heads, self.dim_head)
        key_img = mx.transpose(key_img, (0, 2, 1, 3))
        value_img = mx.transpose(value_img, (0, 2, 1, 3))
        hidden_states = scaled_dot_product_attention(query, key_img, value_img, scale=self.scale)
        self._check_tensor_health(
            enabled=health_enabled,
            name=f"{attention_name}.image_sdpa",
            tensor=hidden_states,
            context=block_health_context,
        )
        return mx.transpose(hidden_states, (0, 2, 1, 3)).reshape(
            hidden_states.shape[0], hidden_states.shape[2], self.inner_dim
        )

    @staticmethod
    def _apply_rotary_emb(
        hidden_states: mx.array,
        freqs_cos: mx.array,
        freqs_sin: mx.array,
    ) -> mx.array:
        x = hidden_states.reshape(*hidden_states.shape[:-1], hidden_states.shape[-1] // 2, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        out = mx.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        return out.reshape(hidden_states.shape)

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
