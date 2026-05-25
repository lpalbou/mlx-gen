import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.common.config import ModelConfig
from mflux.models.ernie_image.model.mistral3_text_encoder.rms_norm import Mistral3RMSNorm


class ErnieImageEmbedND3(nn.Module):
    def __init__(self, dim: int = 128, theta: int = 256, axes_dim: tuple[int, int, int] = (32, 48, 48)):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: mx.array) -> mx.array:
        emb = mx.concatenate(
            [ErnieImageEmbedND3._rope(ids[..., axis], self.axes_dim[axis], self.theta) for axis in range(3)],
            axis=-1,
        )
        emb = mx.expand_dims(emb, axis=2)
        return mx.stack([emb, emb], axis=-1).reshape(*emb.shape[:-1], -1)

    @staticmethod
    def _rope(pos: mx.array, dim: int, theta: int) -> mx.array:
        scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
        omega = 1.0 / (theta**scale)
        return mx.expand_dims(pos.astype(mx.float32), axis=-1) * omega


class ErnieImagePatchEmbedDynamic(nn.Module):
    def __init__(self, in_channels: int = 128, embed_dim: int = 4096, patch_size: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.proj(hidden_states)
        batch_size, height, width, dim = hidden_states.shape
        return hidden_states.reshape(batch_size, height * width, dim)


class ErnieImageAttention(nn.Module):
    def __init__(
        self,
        query_dim: int = 4096,
        heads: int = 32,
        dim_head: int = 128,
        eps: float = 1e-6,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = 1.0 / math.sqrt(dim_head)
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.norm_q = Mistral3RMSNorm(dim_head, eps=eps) if qk_layernorm else None
        self.norm_k = Mistral3RMSNorm(dim_head, eps=eps) if qk_layernorm else None
        self.to_out = [nn.Linear(self.inner_dim, query_dim, bias=False)]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        image_rotary_emb: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.to_q(hidden_states).reshape(batch_size, seq_len, self.heads, self.head_dim)
        key = self.to_k(hidden_states).reshape(batch_size, seq_len, self.heads, self.head_dim)
        value = self.to_v(hidden_states).reshape(batch_size, seq_len, self.heads, self.head_dim)

        if self.norm_q is not None:
            query = self.norm_q(query.astype(mx.float32)).astype(ModelConfig.precision)
        if self.norm_k is not None:
            key = self.norm_k(key.astype(mx.float32)).astype(ModelConfig.precision)

        if image_rotary_emb is not None:
            query = ErnieImageAttention._apply_rotary_emb(query, image_rotary_emb)
            key = ErnieImageAttention._apply_rotary_emb(key, image_rotary_emb)

        query = mx.transpose(query, (0, 2, 1, 3))
        key = mx.transpose(key, (0, 2, 1, 3))
        value = mx.transpose(value, (0, 2, 1, 3))
        hidden_states = scaled_dot_product_attention(
            query.astype(mx.float32),
            key.astype(mx.float32),
            value.astype(mx.float32),
            scale=self.scale,
            mask=attention_mask,
        )
        hidden_states = mx.transpose(hidden_states.astype(query.dtype), (0, 2, 1, 3)).reshape(
            batch_size, seq_len, self.inner_dim
        )
        return self.to_out[0](hidden_states)

    @staticmethod
    def _apply_rotary_emb(hidden_states: mx.array, freqs_cis: mx.array) -> mx.array:
        rot_dim = freqs_cis.shape[-1]
        x = hidden_states[..., :rot_dim]
        x_pass = hidden_states[..., rot_dim:]
        cos = mx.cos(freqs_cis).astype(x.dtype)
        sin = mx.sin(freqs_cis).astype(x.dtype)
        x1, x2 = mx.split(x, 2, axis=-1)
        x_rotated = mx.concatenate([-x2, x1], axis=-1)
        x = x * cos + x_rotated * sin
        if x_pass.shape[-1] == 0:
            return x
        return mx.concatenate([x, x_pass], axis=-1)


class ErnieImageFeedForward(nn.Module):
    def __init__(self, hidden_size: int = 4096, ffn_hidden_size: int = 12288):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        gate = self._gelu(self.gate_proj(hidden_states).astype(mx.float32))
        up = self.up_proj(hidden_states).astype(mx.float32)
        return self.linear_fc2((up * gate).astype(hidden_states.dtype))

    @staticmethod
    def _gelu(x: mx.array) -> mx.array:
        return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


class ErnieImageSharedAdaLNBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 32,
        ffn_hidden_size: int = 12288,
        eps: float = 1e-6,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.adaLN_sa_ln = Mistral3RMSNorm(hidden_size, eps=eps)
        self.self_attention = ErnieImageAttention(
            query_dim=hidden_size,
            dim_head=hidden_size // num_heads,
            heads=num_heads,
            eps=eps,
            qk_layernorm=qk_layernorm,
        )
        self.adaLN_mlp_ln = Mistral3RMSNorm(hidden_size, eps=eps)
        self.mlp = ErnieImageFeedForward(hidden_size, ffn_hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        rotary_pos_emb: mx.array,
        temb: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array],
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb

        residual = hidden_states
        hidden_states = self.adaLN_sa_ln(hidden_states)
        hidden_states = (hidden_states.astype(mx.float32) * (1 + scale_msa.astype(mx.float32))) + shift_msa.astype(
            mx.float32
        )
        attn_input = mx.transpose(hidden_states.astype(ModelConfig.precision), (1, 0, 2))
        attn_output = self.self_attention(
            attn_input,
            attention_mask=attention_mask,
            image_rotary_emb=rotary_pos_emb,
        )
        attn_output = mx.transpose(attn_output, (1, 0, 2))
        hidden_states = residual + (gate_msa.astype(mx.float32) * attn_output.astype(mx.float32)).astype(
            residual.dtype
        )

        residual = hidden_states
        hidden_states = self.adaLN_mlp_ln(hidden_states)
        hidden_states = (hidden_states.astype(mx.float32) * (1 + scale_mlp.astype(mx.float32))) + shift_mlp.astype(
            mx.float32
        )
        return residual + (gate_mlp.astype(mx.float32) * self.mlp(hidden_states).astype(mx.float32)).astype(
            residual.dtype
        )


class ErnieImageTimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        sample = nn.silu(sample)
        return self.linear_2(sample)


class ErnieImageAdaLNModulation(nn.Module):
    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.linear(nn.silu(hidden_states))


class ErnieImageAdaLNContinuous(nn.Module):
    def __init__(self, hidden_size: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, affine=False)
        self.linear = nn.Linear(hidden_size, hidden_size * 2, bias=True)

    def __call__(self, hidden_states: mx.array, conditioning: mx.array) -> mx.array:
        scale, shift = mx.split(self.linear(conditioning), 2, axis=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1 + mx.expand_dims(scale, axis=0)) + mx.expand_dims(shift, axis=0)


class ErnieImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_layers: int = 36,
        ffn_hidden_size: int = 12288,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        text_in_dim: int = 3072,
        rope_theta: int = 256,
        rope_axes_dim: tuple[int, int, int] = (32, 48, 48),
        eps: float = 1e-6,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_in_dim = text_in_dim

        self.x_embedder = ErnieImagePatchEmbedDynamic(in_channels, hidden_size, patch_size)
        self.text_proj = nn.Linear(text_in_dim, hidden_size, bias=False) if text_in_dim != hidden_size else None
        self.time_embedding = ErnieImageTimestepEmbedding(hidden_size)
        self.pos_embed = ErnieImageEmbedND3(dim=self.head_dim, theta=rope_theta, axes_dim=rope_axes_dim)
        self.adaLN_modulation = ErnieImageAdaLNModulation(hidden_size)
        self.layers = [
            ErnieImageSharedAdaLNBlock(hidden_size, num_attention_heads, ffn_hidden_size, eps, qk_layernorm)
            for _ in range(num_layers)
        ]
        self.final_norm = ErnieImageAdaLNContinuous(hidden_size, eps)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        text_bth: mx.array,
        text_lens: mx.array,
    ) -> mx.array:
        batch_size, _, height, width = hidden_states.shape
        patch_size = self.patch_size
        height_p = height // patch_size
        width_p = width // patch_size
        num_image_tokens = height_p * width_p

        image_sbh = mx.transpose(self.x_embedder(hidden_states), (1, 0, 2))
        if self.text_proj is not None and text_bth.size > 0:
            text_bth = self.text_proj(text_bth)
        max_text_len = text_bth.shape[1]
        text_sbh = mx.transpose(text_bth, (1, 0, 2))
        hidden_states = mx.concatenate([image_sbh, text_sbh], axis=0)
        sequence_len = hidden_states.shape[0]

        rotary_pos_emb = self.pos_embed(
            self._position_ids(
                batch_size=batch_size,
                num_image_tokens=num_image_tokens,
                height_p=height_p,
                width_p=width_p,
                max_text_len=max_text_len,
                text_lens=text_lens,
            )
        )
        attention_mask = self._attention_mask(
            batch_size=batch_size,
            num_image_tokens=num_image_tokens,
            max_text_len=max_text_len,
            text_lens=text_lens,
            dtype=hidden_states.dtype,
        )

        conditioning = self.time_embedding(self._timestep_embedding(timestep.astype(mx.float32), self.hidden_size))
        conditioning = conditioning.astype(ModelConfig.precision)
        temb = tuple(
            mx.broadcast_to(mx.expand_dims(t, axis=0), (sequence_len, t.shape[0], t.shape[1]))
            for t in mx.split(self.adaLN_modulation(conditioning), 6, axis=-1)
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary_pos_emb, temb, attention_mask)

        hidden_states = self.final_norm(hidden_states, conditioning).astype(hidden_states.dtype)
        patches = mx.transpose(self.final_linear(hidden_states)[:num_image_tokens], (1, 0, 2))
        patches = patches.reshape(batch_size, height_p, width_p, patch_size, patch_size, self.out_channels)
        patches = mx.transpose(patches, (0, 5, 1, 3, 2, 4))
        return patches.reshape(batch_size, self.out_channels, height, width)

    @staticmethod
    def _timestep_embedding(timesteps: mx.array, dim: int) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / half)
        args = timesteps[:, None].astype(mx.float32) * freqs[None, :]
        emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
        if dim % 2 == 1:
            emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
        return emb

    @staticmethod
    def _position_ids(
        batch_size: int,
        num_image_tokens: int,
        height_p: int,
        width_p: int,
        max_text_len: int,
        text_lens: mx.array,
    ) -> mx.array:
        if max_text_len > 0:
            text_positions = mx.broadcast_to(
                mx.arange(max_text_len, dtype=mx.float32).reshape(1, max_text_len, 1),
                (batch_size, max_text_len, 1),
            )
            text_ids = mx.concatenate([text_positions, mx.zeros((batch_size, max_text_len, 2))], axis=-1)
        else:
            text_ids = mx.zeros((batch_size, 0, 3), dtype=mx.float32)

        y, x = mx.meshgrid(mx.arange(height_p, dtype=mx.float32), mx.arange(width_p, dtype=mx.float32), indexing="ij")
        grid_yx = mx.stack([y, x], axis=-1).reshape(num_image_tokens, 2)
        text_offsets = mx.broadcast_to(text_lens.astype(mx.float32).reshape(batch_size, 1, 1), (batch_size, num_image_tokens, 1))
        image_grid = mx.broadcast_to(grid_yx.reshape(1, num_image_tokens, 2), (batch_size, num_image_tokens, 2))
        image_ids = mx.concatenate([text_offsets, image_grid], axis=-1)
        return mx.concatenate([image_ids, text_ids], axis=1)

    @staticmethod
    def _attention_mask(
        batch_size: int,
        num_image_tokens: int,
        max_text_len: int,
        text_lens: mx.array,
        dtype: mx.Dtype,
    ) -> mx.array:
        image_valid = mx.ones((batch_size, num_image_tokens), dtype=mx.bool_)
        if max_text_len > 0:
            text_valid = mx.arange(max_text_len)[None, :] < text_lens[:, None]
        else:
            text_valid = mx.zeros((batch_size, 0), dtype=mx.bool_)
        valid = mx.concatenate([image_valid, text_valid], axis=1)
        return mx.where(
            valid[:, None, None, :],
            mx.zeros((batch_size, 1, 1, num_image_tokens + max_text_len), dtype=dtype),
            mx.full((batch_size, 1, 1, num_image_tokens + max_text_len), -float("inf"), dtype=dtype),
        )
