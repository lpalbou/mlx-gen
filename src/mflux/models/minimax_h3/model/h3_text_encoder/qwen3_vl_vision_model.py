"""Qwen3-VL vision tower and image preprocessing, as MiniMax-H3 uses them for keyframes.

The processor (`Qwen2VLImageProcessor` semantics) turns an image into `16x16` patches duplicated over
a temporal pair, ordered by `2x2` spatial-merge blocks. The tower embeds the patches, adds a bilinearly
resampled learned position grid, runs 27 pre-norm blocks with a 2-axis rotary embedding, and merges
every `2x2` block into one token of the language model's width. DeepStack mergers on blocks 8, 16 and 24
produce the features the language model adds at its first three layers.
"""

import math

import mlx.core as mx
import numpy as np
import PIL.Image
from mlx import nn

from mflux.models.minimax_h3.model.h3_precision import linear_input_dtype

IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
MIN_PIXELS = 65536
MAX_PIXELS = 16777216


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen2-VL's resize rule: multiples of `factor`, pixel count within bounds, aspect ratio preserved."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess_image(
    image: PIL.Image.Image,
    patch_size: int = 16,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Image to `(num_patches, channels * temporal * patch * patch)` float32 patches and its `(t, h, w)` grid."""
    image = image.convert("RGB")
    height, width = smart_resize(image.height, image.width, patch_size * merge_size, min_pixels, max_pixels)
    if (width, height) != image.size:
        image = image.resize((width, height), PIL.Image.Resampling.BICUBIC)
    pixels = (np.asarray(image, dtype=np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD  # (H, W, C)
    pixels = pixels.transpose(2, 0, 1)  # (C, H, W)
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = pixels.reshape(
        3, grid_h // merge_size, merge_size, patch_size, grid_w // merge_size, merge_size, patch_size
    )
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)  # (gh/m, gw/m, m, m, C, p, p)
    patches = np.repeat(patches[:, :, :, :, :, None, :, :], temporal_patch_size, axis=5)  # (…, C, T, p, p)
    return np.ascontiguousarray(patches.reshape(grid_h * grid_w, -1)), (1, grid_h, grid_w)


def vision_position_ids(grid_thw: tuple[int, int, int], merge_size: int) -> np.ndarray:
    """`(N, 2)` (h, w) patch coordinates in spatial-merge-block order, repeated over the temporal grid."""
    t, h, w = grid_thw
    hpos, wpos = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    block = (h // merge_size, merge_size, w // merge_size, merge_size)
    hpos = hpos.reshape(block).transpose(0, 2, 1, 3).reshape(-1)
    wpos = wpos.reshape(block).transpose(0, 2, 1, 3).reshape(-1)
    return np.tile(np.stack([hpos, wpos], axis=-1), (t, 1))


def _axis_taps(index: np.ndarray, size: int, side: int) -> tuple[np.ndarray, np.ndarray]:
    src = index.astype(np.float64) * (side - 1) / max(size - 1, 1)  # align_corners=True
    floor = np.floor(src)
    taps = floor[:, None].astype(np.int64) + np.arange(2)[None, :]
    weights = np.clip(1.0 - np.abs(src[:, None] - taps), 0.0, None)
    return np.clip(taps, 0, side - 1), weights


def position_embedding_taps(
    grid_thw: tuple[int, int, int], side: int, merge_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear (align_corners) gather indices and weights resampling the `side x side` learned grid to the patch grid."""
    t, h, w = grid_thw
    within = np.arange(t * h * w) % (h * w)
    blocks_w = w // merge_size
    in_col = within % merge_size
    in_row = (within // merge_size) % merge_size
    block_col = (within // (merge_size * merge_size)) % blocks_w
    block_row = within // (merge_size * merge_size * blocks_w)
    row = block_row * merge_size + in_row
    col = block_col * merge_size + in_col
    h_taps, h_weights = _axis_taps(row, h, side)
    w_taps, w_weights = _axis_taps(col, w, side)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices, weights.astype(np.float32)


def _rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


class Qwen3VLVisionPatchEmbed(nn.Module):
    """The reference `Conv3d` over one patch, expressed as a linear map of the flattened patch."""

    def __init__(self, in_channels: int, patch_size: int, temporal_patch_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_channels * temporal_patch_size * patch_size * patch_size, embed_dim, bias=True)

    def __call__(self, patches: mx.array) -> mx.array:
        return self.proj(patches.astype(linear_input_dtype(self.proj)))


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, hidden_size: int, merge_size: int, out_hidden_size: int, use_postshuffle_norm: bool):
        super().__init__()
        self.hidden_size = hidden_size * merge_size * merge_size
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(self.hidden_size, out_hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x.reshape(-1, self.hidden_size) if self.use_postshuffle_norm else x).reshape(-1, self.hidden_size)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        seq_len = x.shape[0]
        qkv = self.qkv(x).reshape(seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (N, H, D)
        cos, sin = cos[:, None, :], sin[:, None, :]
        query32, key32 = query.astype(mx.float32), key.astype(mx.float32)
        query = (query32 * cos + _rotate_half(query32) * sin).astype(query.dtype)
        key = (key32 * cos + _rotate_half(key32) * sin).astype(key.dtype)
        out = mx.fast.scaled_dot_product_attention(
            query.transpose(1, 0, 2)[None],
            key.transpose(1, 0, 2)[None],
            value.transpose(1, 0, 2)[None],
            scale=self.head_dim**-0.5,
        )
        return self.proj(out[0].transpose(1, 0, 2).reshape(seq_len, -1))


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))  # `gelu_pytorch_tanh`


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(hidden_size, num_heads)
        self.mlp = Qwen3VLVisionMLP(hidden_size, intermediate_size)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class Qwen3VLVisionModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1152,
        depth: int = 27,
        num_heads: int = 16,
        intermediate_size: int = 4304,
        in_channels: int = 3,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        num_position_embeddings: int = 2304,
        out_hidden_size: int = 5120,
        deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24),
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.num_grid_per_side = int(num_position_embeddings**0.5)
        self.deepstack_visual_indexes = tuple(deepstack_visual_indexes)
        head_dim = hidden_size // num_heads
        self.patch_embed = Qwen3VLVisionPatchEmbed(in_channels, patch_size, temporal_patch_size, hidden_size)
        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
        # 2-axis rope: `head_dim // 2` frequencies per axis pair, h and w halves, then duplicated to the full head.
        rope_dim = head_dim // 2
        self._inv_freq = 1.0 / (rope_theta ** (mx.arange(0, rope_dim, 2, dtype=mx.float32) / rope_dim))
        self.blocks = [Qwen3VLVisionBlock(hidden_size, num_heads, intermediate_size) for _ in range(depth)]
        self.merger = Qwen3VLVisionPatchMerger(
            hidden_size, spatial_merge_size, out_hidden_size, use_postshuffle_norm=False
        )
        self.deepstack_merger_list = [
            Qwen3VLVisionPatchMerger(hidden_size, spatial_merge_size, out_hidden_size, use_postshuffle_norm=True)
            for _ in deepstack_visual_indexes
        ]

    def __call__(self, patches: mx.array, grid_thw: tuple[int, int, int]) -> tuple[mx.array, list[mx.array]]:
        """Patches `(N, C*T*p*p)` of one image to merged tokens `(N/4, out)` and the DeepStack feature list."""
        x = self.patch_embed(patches)
        indices, weights = position_embedding_taps(grid_thw, self.num_grid_per_side, self.spatial_merge_size)
        pos = (self.pos_embed(mx.array(indices)) * mx.array(weights)[:, :, None]).sum(axis=1)
        x = x + pos.astype(x.dtype)
        positions = mx.array(vision_position_ids(grid_thw, self.spatial_merge_size), dtype=mx.float32)  # (N, 2)
        freqs = (positions[:, :, None] * self._inv_freq[None, None, :]).reshape(positions.shape[0], -1)  # (N, D/2)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos, sin = mx.cos(emb), mx.sin(emb)
        deepstack_features = []
        for index, block in enumerate(self.blocks):
            x = block(x, cos, sin)
            if index in self.deepstack_visual_indexes:
                deepstack_features.append(self.deepstack_merger_list[self.deepstack_visual_indexes.index(index)](x))
        return self.merger(x), deepstack_features
