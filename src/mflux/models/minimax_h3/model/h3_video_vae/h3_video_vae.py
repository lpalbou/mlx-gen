"""MiniMax-H3 visual VAE (diffusers `AutoencoderKLMiniMaxH3`).

A temporally causal 3D-CNN encoder (16x spatial, 4x temporal, 24 latent channels) paired with
a non-causal ViT decoder. Pixels use ImageNet-normalized RGB over a [0, 1] base range, and
latents are normalized per channel with `latents_mean` / `latents_std` by the caller.
Spatial tiling is on by default, as released: the reference frames are the blended-tile ones.

Module and parameter names follow the checkpoint. Internally tensors are channels-last
`(batch, frames, height, width, channels)` as MLX convolutions expect; the public `encode`
and `decode` take and return the reference `(batch, channels, frames, height, width)` layout.
"""

import math

import mlx.core as mx
from mlx import nn

from mflux.models.minimax_h3.model.h3_transformer.h3_attention import H3FeedForward


def _reflect_pad_axis(x: mx.array, axis: int, before: int, after: int) -> mx.array:
    """torch `reflect` padding (edge excluded) along one axis, built from reversed slices."""
    parts = []
    if before > 0:
        index = [slice(None)] * x.ndim
        index[axis] = slice(before, 0, -1)
        parts.append(x[tuple(index)])
    parts.append(x)
    if after > 0:
        index = [slice(None)] * x.ndim
        index[axis] = slice(-2, -2 - after, -1)
        parts.append(x[tuple(index)])
    return mx.concatenate(parts, axis=axis) if len(parts) > 1 else x


class H3CausalConv3d(nn.Conv3d):
    """Conv3d with symmetric reflect spatial padding and causal (front-only, zero) temporal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
    ):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding

    def __call__(self, x: mx.array) -> mx.array:
        if self.spatial_padding > 0:
            x = _reflect_pad_axis(x, 2, self.spatial_padding, self.spatial_padding)
            x = _reflect_pad_axis(x, 3, self.spatial_padding, self.spatial_padding)
        if self.temporal_padding > 0:
            x = mx.pad(x, [(0, 0), (self.temporal_padding, 0), (0, 0), (0, 0), (0, 0)])
        return super().__call__(x)


class H3FrameGroupNorm(nn.Module):
    """GroupNorm with statistics computed per frame (time folded into the batch axis)."""

    def __init__(self, num_groups: int, channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        batch, frames, height, width, channels = x.shape
        dtype = x.dtype
        grouped = x.astype(mx.float32).reshape(
            batch * frames, height * width, self.num_groups, channels // self.num_groups
        )
        mean = grouped.mean(axis=(1, 3), keepdims=True)
        var = grouped.var(axis=(1, 3), keepdims=True)
        normed = ((grouped - mean) * mx.rsqrt(var + self.eps)).reshape(batch, frames, height, width, channels)
        return (normed * self.weight + self.bias).astype(dtype)


class H3ResnetBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_num_groups: int = 32, norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = H3FrameGroupNorm(norm_num_groups, in_channels, eps=norm_eps)
        self.conv1 = H3CausalConv3d(in_channels, out_channels, 3, spatial_padding=1, temporal_padding=2)
        self.norm2 = H3FrameGroupNorm(norm_num_groups, out_channels, eps=norm_eps)
        self.conv2 = H3CausalConv3d(out_channels, out_channels, 3, spatial_padding=1, temporal_padding=2)
        self.conv_shortcut = H3CausalConv3d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.conv1(nn.silu(self.norm1(x)))
        x = self.conv2(nn.silu(self.norm2(x)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + x


class H3Downsample3d(nn.Module):
    """Strided 3x3x3 convolution; a spatial stride of 2 is preceded by a bottom/right reflect pad of 1."""

    def __init__(self, in_channels: int, out_channels: int, temporal_stride: int = 1, spatial_stride: int = 2):
        super().__init__()
        self.spatial_stride = spatial_stride
        self.conv = H3CausalConv3d(
            in_channels, out_channels, 3, stride=(temporal_stride, spatial_stride, spatial_stride), temporal_padding=2
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.spatial_stride == 2:
            x = _reflect_pad_axis(x, 2, 0, 1)
            x = _reflect_pad_axis(x, 3, 0, 1)
        return self.conv(x)


class H3DownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        temporal_downsample_factor,
        spatial_downsample_factor,
        norm_num_groups=32,
        norm_eps=1e-6,
    ):
        super().__init__()
        self.resnets = [
            H3ResnetBlock3d(in_channels if i == 0 else out_channels, out_channels, norm_num_groups, norm_eps)
            for i in range(num_layers)
        ]
        self.downsamplers = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsamplers = [
                H3Downsample3d(out_channels, out_channels, temporal_downsample_factor, spatial_downsample_factor)
            ]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)
        return x


class H3VideoEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        block_out_channels,
        layers_per_block,
        spatial_downsample_factors,
        temporal_downsample_factors,
        norm_num_groups,
        norm_eps,
    ):
        super().__init__()
        self.conv_in = H3CausalConv3d(in_channels, block_out_channels[0], 3, spatial_padding=1, temporal_padding=2)
        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down_blocks = [
            H3DownBlock3d(
                block_in_channels[i],
                block_out_channels[i],
                layers_per_block,
                temporal_downsample_factors[i],
                spatial_downsample_factors[i],
                norm_num_groups,
                norm_eps,
            )
            for i in range(len(block_out_channels))
        ]
        self.norm_out = H3FrameGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps)
        self.conv_out = H3CausalConv3d(block_out_channels[-1], out_channels, 3, spatial_padding=1, temporal_padding=2)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        return self.conv_out(nn.silu(self.norm_out(x)))


class H3VideoRotaryPosEmbed(nn.Module):
    """3-axis rotary embedding for the ViT decoder over length-normalized `[-1, 1)` coordinates."""

    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3):
        super().__init__()
        self._inv_freq = 1.0 / theta ** mx.arange(0, 1, 2 * num_axes / dim, dtype=mx.float32)

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self._inv_freq[None, None, None, :]  # (B, N, 3, F)
        angles = angles.reshape(angles.shape[0], angles.shape[1], -1)
        angles = mx.concatenate([angles, angles], axis=-1)[:, :, None, :]  # (B, N, 1, rotary_dim)
        return mx.cos(angles), mx.sin(angles)


def _rms_norm_no_affine(x: mx.array, eps: float) -> mx.array:
    x32 = x.astype(mx.float32)
    return (x32 * mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)).astype(x.dtype)


class H3VideoAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-5):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = [nn.Linear(inner_dim, dim, bias=True)]

    def __call__(self, x: mx.array, rotary_emb: tuple[mx.array, mx.array]) -> mx.array:
        batch, seq_len, _ = x.shape
        query = _rms_norm_no_affine(self.to_q(x).reshape(batch, seq_len, self.heads, self.dim_head), self.eps)
        key = _rms_norm_no_affine(self.to_k(x).reshape(batch, seq_len, self.heads, self.dim_head), self.eps)
        value = self.to_v(x).reshape(batch, seq_len, self.heads, self.dim_head)
        cos, sin = (t.astype(query.dtype) for t in rotary_emb)
        rotary_dim = cos.shape[-1]

        def rotate(t: mx.array) -> mx.array:
            rot, rest = t[..., :rotary_dim], t[..., rotary_dim:]
            first, second = mx.split(rot, 2, axis=-1)
            return mx.concatenate([rot * cos + mx.concatenate([-second, first], axis=-1) * sin, rest], axis=-1)

        query, key = rotate(query), rotate(key)
        out = mx.fast.scaled_dot_product_attention(
            query.transpose(0, 2, 1, 3),
            key.transpose(0, 2, 1, 3),
            value.transpose(0, 2, 1, 3),
            scale=1.0 / math.sqrt(self.dim_head),
        )
        return self.to_out[0](out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1))


class H3VideoTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ffn_mult: int = 4, eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps)
        self.attn = H3VideoAttention(dim, heads, dim_head, eps)
        self.scale1 = mx.zeros((dim,))
        self.norm2 = nn.RMSNorm(dim, eps=eps)
        self.ff = H3FeedForward(dim, dim * ffn_mult, bias=True)
        self.scale2 = mx.zeros((dim,))

    def __call__(self, x: mx.array, rotary_emb: tuple[mx.array, mx.array]) -> mx.array:
        x = x + self.attn(self.norm1(x), rotary_emb) * self.scale1
        return x + self.ff(self.norm2(x)) * self.scale2


class H3VideoViTDecoder3d(nn.Module):
    """Every latent voxel becomes a token; learned register tokens plus one zero token are appended at
    position 0, attended over, dropped, and each remaining token expands into a `4 x 16 x 16` pixel block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        patch_size,
        patch_size_t,
        num_layers,
        num_attention_heads,
        attention_head_dim,
        num_register_tokens,
        ffn_mult,
        rope_theta,
        rope_dim_ratio,
        norm_eps,
    ):
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.rope = H3VideoRotaryPosEmbed(int(attention_head_dim * rope_dim_ratio), theta=rope_theta)
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = mx.zeros((1, num_register_tokens, dim))
        self.transformer_blocks = [
            H3VideoTransformerBlock(dim, num_attention_heads, attention_head_dim, ffn_mult, norm_eps)
            for _ in range(num_layers)
        ]
        self.norm_out = nn.LayerNorm(dim, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def __call__(self, latents: mx.array) -> mx.array:
        """`(B, F, H, W, C)` latents to `(B, F*pt, H*p, W*p, out_channels)` pixels."""
        batch, frames, height, width, channels = latents.shape
        tokens = self.proj_in(latents.reshape(batch, frames * height * width, channels))
        num_patches = tokens.shape[1]
        registers = mx.broadcast_to(
            self.register_tokens.astype(tokens.dtype), (batch, self.num_register_tokens, tokens.shape[-1])
        )
        tokens = mx.concatenate([tokens, registers, mx.zeros_like(tokens[:, :1, :])], axis=1)

        grids = [2.0 * (mx.arange(size, dtype=mx.float32) + 0.5) / size - 1.0 for size in (frames, height, width)]
        position_ids = mx.stack(mx.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 3)
        position_ids = mx.concatenate([position_ids, mx.zeros((self.num_register_tokens + 1, 3))], axis=0)
        rotary_emb = self.rope(mx.broadcast_to(position_ids[None], (batch,) + position_ids.shape))

        for block in self.transformer_blocks:
            tokens = block(tokens, rotary_emb)
        tokens = self.proj_out(self.norm_out(tokens))[:, :num_patches, :]
        p, pt = self.patch_size, self.patch_size_t
        tokens = tokens.reshape(batch, frames, height, width, self.out_channels, pt, p, p)
        tokens = tokens.transpose(0, 1, 5, 2, 6, 3, 7, 4)  # (B, F, pt, H, p, W, p, C)
        return tokens.reshape(batch, frames * pt, height * p, width * p, self.out_channels)


class H3VideoVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 24,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        decoder_num_layers: int = 36,
        decoder_num_attention_heads: int = 32,
        decoder_attention_head_dim: int = 64,
        decoder_num_register_tokens: int = 4,
        decoder_ffn_mult: int = 4,
        decoder_rope_theta: float = 100.0,
        decoder_rope_dim_ratio: float = 0.75,
        decoder_norm_eps: float = 1e-5,
        clip_length: int = 17,
        token_drop: int = 3,
        latents_mean: list[float] | None = None,
        latents_std: list[float] | None = None,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.clip_length = clip_length
        self.token_drop = token_drop
        self.spatial_compression_ratio = math.prod(spatial_downsample_factors)
        self.temporal_compression_ratio = math.prod(temporal_downsample_factors)
        self._latents_mean = mx.array(latents_mean, dtype=mx.float32) if latents_mean else None
        self._latents_std = mx.array(latents_std, dtype=mx.float32) if latents_std else None
        self.encoder = H3VideoEncoder3d(
            in_channels,
            2 * latent_channels,
            block_out_channels,
            layers_per_block,
            spatial_downsample_factors,
            temporal_downsample_factors,
            norm_num_groups,
            norm_eps,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        self.decoder = H3VideoViTDecoder3d(
            latent_channels,
            out_channels,
            self.spatial_compression_ratio,
            self.temporal_compression_ratio,
            decoder_num_layers,
            decoder_num_attention_heads,
            decoder_attention_head_dim,
            decoder_num_register_tokens,
            decoder_ffn_mult,
            decoder_rope_theta,
            decoder_rope_dim_ratio,
            decoder_norm_eps,
        )
        self.frame_pre_padding = (-clip_length) % self.temporal_compression_ratio
        self.tokens_chunk_size = math.ceil(clip_length / self.temporal_compression_ratio)
        self.token_overlap = (-token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.temporal_compression_ratio - self.frame_pre_padding, 0)
        self.use_tiling = True
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64

    @property
    def latents_mean(self) -> mx.array:
        return self._latents_mean

    @property
    def latents_std(self) -> mx.array:
        return self._latents_std

    # --- tiling -----------------------------------------------------------------------------
    def _split_tiles(self, length: int, tile_size: int, min_overlap: int) -> tuple[list[int], list[int], list[int]]:
        if tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // self.spatial_compression_ratio):
            overlaps[i % (num_tiles - 1)] += self.spatial_compression_ratio
        starts = [0]
        for i in range(num_tiles - 1):
            starts.append(starts[-1] + tile_size - overlaps[i])
        return starts, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(a: mx.array, b: mx.array, blend_extent: int, axis: int) -> mx.array:
        blend_extent = min(a.shape[axis], b.shape[axis], blend_extent)
        shape = [1] * a.ndim
        shape[axis] = blend_extent
        positions = mx.arange(blend_extent, dtype=b.dtype).reshape(shape)
        weight_b = positions / blend_extent
        index_a = [slice(None)] * a.ndim
        index_a[axis] = slice(-blend_extent, None)
        index_b = [slice(None)] * b.ndim
        index_b[axis] = slice(0, blend_extent)
        blended = a[tuple(index_a)] * (1 - weight_b) + b[tuple(index_b)] * weight_b
        if blend_extent == b.shape[axis]:
            return blended
        index_rest = [slice(None)] * b.ndim
        index_rest[axis] = slice(blend_extent, None)
        return mx.concatenate([blended, b[tuple(index_rest)]], axis=axis)

    def _stitch_tiles(
        self, tiles: list[list[mx.array]], height_overlaps: list[int], width_overlaps: list[int]
    ) -> mx.array:
        rows = []
        for i, row in enumerate(tiles):
            stitched = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], axis=2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], axis=3)
                if i < len(tiles) - 1:
                    tile = tile[:, :, : -height_overlaps[i], :, :]
                if j < len(row) - 1:
                    tile = tile[:, :, :, : -width_overlaps[j], :]
                stitched.append(tile)
            rows.append(mx.concatenate(stitched, axis=3))
        return mx.concatenate(rows, axis=2)

    def _encode_clip(self, x: mx.array) -> mx.array:
        if not self.use_tiling:
            return self.quant_conv(self.encoder(x))
        height, width = x.shape[2], x.shape[3]
        y_idx, y_len, y_ov = self._split_tiles(height, self.tile_sample_min_height, self.tile_sample_min_overlap_height)
        x_idx, x_len, x_ov = self._split_tiles(width, self.tile_sample_min_width, self.tile_sample_min_overlap_width)
        rows = [
            [self.quant_conv(self.encoder(x[:, :, i : i + ih, j : j + jw, :])) for j, jw in zip(x_idx, x_len)]
            for i, ih in zip(y_idx, y_len)
        ]
        ratio = self.spatial_compression_ratio
        return self._stitch_tiles(rows, [o // ratio for o in y_ov], [o // ratio for o in x_ov])

    def _decode_clip(self, z: mx.array) -> mx.array:
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))
        ratio = self.spatial_compression_ratio
        height, width = z.shape[2] * ratio, z.shape[3] * ratio
        y_idx, y_len, y_ov = self._split_tiles(height, self.tile_sample_min_height, self.tile_sample_min_overlap_height)
        x_idx, x_len, x_ov = self._split_tiles(width, self.tile_sample_min_width, self.tile_sample_min_overlap_width)
        rows = [
            [
                self.decoder(
                    self.post_quant_conv(z[:, :, i // ratio : (i + ih) // ratio, j // ratio : (j + jw) // ratio, :])
                )
                for j, jw in zip(x_idx, x_len)
            ]
            for i, ih in zip(y_idx, y_len)
        ]
        return self._stitch_tiles(rows, y_ov, x_ov)

    # --- temporal chunking --------------------------------------------------------------------
    def _encode(self, x: mx.array) -> mx.array:
        num_frames = x.shape[1]
        if num_frames == 1:
            return self._encode_clip(x)
        if num_frames % self.clip_length != 0:
            pad = (-num_frames) % self.clip_length
            x = mx.concatenate([x, mx.repeat(x[:, -1:], pad, axis=1)], axis=1)
        moments = mx.concatenate(
            [
                self._encode_clip(x[:, i * self.clip_length : (i + 1) * self.clip_length])
                for i in range(x.shape[1] // self.clip_length)
            ],
            axis=1,
        )
        if self.token_drop > 0:
            moments = moments[:, : -self.token_drop]
        return moments

    def _decode(self, z: mx.array) -> mx.array:
        chunk = self.tokens_chunk_size
        ratio = self.temporal_compression_ratio
        chunk_num_frames = chunk * ratio
        num_tokens = z.shape[1] + self.token_drop
        pad_tokens = (-num_tokens) % chunk
        num_chunks = (num_tokens + pad_tokens) // chunk - int(self.token_drop > 0)
        if pad_tokens > 0:
            z = mx.concatenate([z, mx.repeat(z[:, -1:], pad_tokens, axis=1)], axis=1)
        decoded, overlap = [], None
        for i in range(num_chunks):
            start = i * chunk
            clip = self._decode_clip(z[:, start : start + chunk + self.token_overlap])
            for j in range(int(self.token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                piece = clip[:, frame_start : frame_start + chunk_num_frames][:, self.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        piece = self._blend(overlap, piece, self.frame_overlap, axis=1)
                    decoded.append(piece)
                else:
                    overlap = piece
        if overlap is not None:
            decoded.append(overlap)
        out = mx.concatenate(decoded, axis=1)
        if pad_tokens > 0:
            intra_tail = self.clip_length % ratio
            before = z.shape[1] - pad_tokens
            pad_frames = sum(
                intra_tail if intra_tail and (before + k) % chunk == 0 else ratio for k in range(pad_tokens)
            )
            out = out[:, :-pad_frames]
        return out

    # --- public API in the reference (B, C, F, H, W) layout ------------------------------------
    def encode(self, pixels: mx.array) -> tuple[mx.array, mx.array]:
        """ImageNet-normalized `(B, 3, F, H, W)` pixels to posterior `(mean, logvar)`, each `(B, 24, F', H', W')`."""
        moments = self._encode(pixels.transpose(0, 2, 3, 4, 1)).transpose(0, 4, 1, 2, 3)
        return mx.split(moments, 2, axis=1)

    def decode(self, latents: mx.array) -> mx.array:
        """Denormalized `(B, 24, F', H', W')` latents to ImageNet-normalized `(B, 3, F, H, W)` pixels."""
        return self._decode(latents.transpose(0, 2, 3, 4, 1)).transpose(0, 4, 1, 2, 3)
