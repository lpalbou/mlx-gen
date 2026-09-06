"""Packed-sequence geometry for MiniMax-H3.

MiniMax-H3 denoises one packed 1-D sequence holding the text conditioning, the
keyframe conditioning rows, the audio rows and the video rows. This module owns the
arithmetic that describes that sequence — canvas and frame-count rules, the
`(t, h, w)` rotary coordinates of every row, the per-row modality tags, the row
indices of each modality, and the per-step timestep plan — following the diffusers
`modular_pipelines/minimax_h3` blocks exactly. Rotary grids are computed in float64
as upstream does and cast to float32 where the transformer consumes them.
"""

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

FPS = 24
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
MODALITY_NUM = 3
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2
# The rate the released audio VAE writes; a run reads the real value off the VAE's own config.
AUDIO_SAMPLE_RATE = 32000
MIN_ASPECT_RATIO = 1 / 4
MAX_ASPECT_RATIO = 4
MIN_DURATION_SECONDS = 5.0
MAX_DURATION_SECONDS = 15.0
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
KEYFRAME_NOISE_AUG = 0.999
KEYFRAME_ENCODE_SEED = 42
TEXT_ENCODER_LAYER = 50
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    canvas_multiple: int = 32,
    short_edge: int = CANVAS_SHORT_EDGE,
    max_pixels: int = CANVAS_MAX_PIXELS,
) -> tuple[int, int]:
    """Resolve a display aspect ratio into a MiniMax-H3 `(height, width)` canvas."""
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")
    ratio = aspect_width / aspect_height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:{1 / MIN_ASPECT_RATIO:g} to {MAX_ASPECT_RATIO:g}:1, "
            f"got {aspect_width}:{aspect_height} ({ratio:g})."
        )
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale
    multiple = canvas_multiple
    return max(multiple, round(height / multiple) * multiple), max(multiple, round(width / multiple) * multiple)


MIN_DURATION_SECONDS = 5.0
MAX_DURATION_SECONDS = 15.0
# The duration a request generates is the one of the *aligned* count, so the ceiling holds for that:
# 346 frames would pass a check on the request and then round up to 362, i.e. 15.083 s. The largest
# `17 * n + 5` inside the window is therefore 345 (14.375 s), not 360 or 362.
MIN_NUM_FRAMES = 124
MAX_NUM_FRAMES = 345


def valid_frame_counts() -> tuple[int, ...]:
    """Every frame count MiniMax-H3 accepts: the `17 * n + 5` grid inside the 5 to 15 second window."""
    return tuple(range(MIN_NUM_FRAMES, MAX_NUM_FRAMES + 1, 17))


def align_num_frames(num_frames: int, frames_per_chunk: int = 17, latents_per_chunk: int = 5) -> int:
    """Snap a frame count up to the next `17 * n + 5` the video VAE can encode."""
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    while num_frames % frames_per_chunk != latents_per_chunk:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int, frames_per_chunk: int = 17, latents_per_chunk: int = 5) -> int:
    """`17 * n + 5` pixel frames map to `5 * n + 2` latent frames."""
    if num_frames % frames_per_chunk != latents_per_chunk:
        raise ValueError(
            f"`num_frames` must be of the form {frames_per_chunk} * n + {latents_per_chunk}, got {num_frames}."
        )
    return (num_frames - latents_per_chunk) // frames_per_chunk * latents_per_chunk + 2


def audio_latent_num_frames(
    num_frames: int, fps: float = FPS, latents_per_second: int = AUDIO_LATENTS_PER_SECOND
) -> int:
    return int(round(num_frames / fps * latents_per_second))


def patchify_video_latents(latents: mx.array, patch_size: tuple[int, int, int]) -> mx.array:
    """`(B, C, F, H, W)` latents to `(B * patches, C * prod(patch))` rows, frame-major then row-major."""
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}.")
    latents = latents.reshape(
        batch_size, channels, num_frames // patch_t, patch_t, height // patch_h, patch_h, width // patch_w, patch_w
    )
    latents = latents.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w)


def unpatchify_video_rows(
    rows: mx.array,
    channels: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int],
) -> mx.array:
    """Inverse of `patchify_video_latents` for a single batch item: rows to `(1, C, F, H, W)`."""
    patch_t, patch_h, patch_w = patch_size
    rows = rows.reshape(
        -1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(-1, channels, num_latent_frames, latent_height, latent_width)


def pack_audio_rows(audio_latents: mx.array) -> mx.array:
    """`(channels, C, T)` audio latents to channel-major `(channels * T, C)` rows."""
    return audio_latents.transpose(0, 2, 1).reshape(-1, audio_latents.shape[1])


def unpack_audio_rows(rows: mx.array, audio_channels: int, num_audio_latents: int) -> mx.array:
    """Channel-major `(channels * T, C)` rows back to `(channels, C, T)` latents."""
    return rows.reshape(audio_channels, num_audio_latents, rows.shape[-1]).transpose(0, 2, 1)


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE


def _temporal_position_grid(num_latent_frames: int, origin: float) -> np.ndarray:
    spans = np.array(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
            for i in range(num_latent_frames)
        ],
        dtype=np.float64,
    )
    return origin + np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(spans[:-1])])


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[np.ndarray, np.ndarray]:
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    hh, ww = np.meshgrid(height_grid, width_grid, indexing="ij")
    return np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1), width_grid


@dataclass(frozen=True)
class H3PackedLayout:
    """Structural description of one packed MiniMax-H3 sequence, shared by every batch item."""

    position_ids: mx.array  # (seq_len, 3) float32 rotary coordinates (t, h, w)
    token_tags: mx.array  # (seq_len,) int32 modality tag per row
    video_indices: mx.array  # rows of the video stream, conditioning rows first
    audio_indices: mx.array  # rows of the audio stream, reference rows first
    text_indices: mx.array  # rows of the text stream
    num_condition_video_rows: int
    num_condition_audio_rows: int

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])

    @property
    def num_text_tokens(self) -> int:
        return int(self.text_indices.shape[0])


def build_packed_sequence(
    text_token_tags: np.ndarray,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    audio_channels: int = AUDIO_CHANNELS,
    keyframe_anchors: tuple[str, ...] = (),
) -> H3PackedLayout:
    """Build the `[text | keyframe conditions | target audio | target video]` layout of `t2va` / `fl2va`."""
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = int(text_token_tags.shape[0])
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * audio_channels
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows
    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    position_ids = np.zeros((sequence_length, 3), dtype=np.float64)
    position_ids[:num_text_tokens, 0] = np.arange(num_text_tokens, dtype=np.float64)

    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
            for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
                spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
            anchor_time = float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    audio_time = float(num_text_tokens) + np.arange(num_audio_latents, dtype=np.float64)
    position_ids[audio_start:video_start, 0] = np.tile(audio_time, audio_channels)
    position_ids[audio_start:video_start, 2] = np.concatenate(
        [
            np.full(num_audio_latents, float(width_grid[0]), dtype=np.float64),
            np.full(num_audio_rows - num_audio_latents, float(width_grid[-1]), dtype=np.float64),
        ]
    )

    video_positions = np.empty((num_latent_frames, rows_per_frame, 3), dtype=np.float64)
    video_positions[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_positions[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_positions.reshape(-1, 3)

    video_indices = np.concatenate([np.arange(condition_start, audio_start), np.arange(video_start, sequence_length)])
    audio_indices = np.arange(audio_start, video_start)
    text_indices = np.arange(num_text_tokens)
    token_tags = np.empty(sequence_length, dtype=np.int32)
    token_tags[text_indices] = text_token_tags.astype(np.int32)
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG

    return H3PackedLayout(
        position_ids=mx.array(position_ids.astype(np.float32)),
        token_tags=mx.array(token_tags),
        video_indices=mx.array(video_indices.astype(np.int32)),
        audio_indices=mx.array(audio_indices.astype(np.int32)),
        text_indices=mx.array(text_indices.astype(np.int32)),
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=0,
    )


def build_row_timesteps(
    layout: H3PackedLayout,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Per-row timesteps reduced to the transformer's `(distinct timesteps, per-row index)` pair.

    Generated video and audio rows step down their own schedules, conditioning rows stay pinned at
    their noise-augmentation level, and text rows inherit the video timestep.
    """
    video_indices = np.array(layout.video_indices)
    audio_indices = np.array(layout.audio_indices)
    row_timesteps = np.full(layout.sequence_length, video_timestep, dtype=np.float32)
    row_timesteps[video_indices[: layout.num_condition_video_rows]] = condition_video_timestep
    row_timesteps[audio_indices[layout.num_condition_audio_rows :]] = audio_timestep
    row_timesteps[audio_indices[: layout.num_condition_audio_rows]] = condition_audio_timestep
    unique_timesteps, inverse = np.unique(row_timesteps, return_inverse=True)
    return mx.array(unique_timesteps.astype(np.float32)), mx.array(inverse.reshape(-1).astype(np.int32))
