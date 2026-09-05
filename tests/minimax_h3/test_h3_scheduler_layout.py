"""Fast contracts of the MiniMax-H3 scheduler, packed layout and prompt composition (no weights)."""

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.minimax_h3.latent_creator.h3_layout import (
    AUDIO_TAG,
    TEXT_TAG,
    VIDEO_TAG,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    pack_audio_rows,
    patchify_video_latents,
    resolve_canvas_size,
    unpack_audio_rows,
    unpatchify_video_rows,
    video_latent_num_frames,
)
from mflux.models.minimax_h3.scheduler.minimax_h3_scheduler import MiniMaxH3Scheduler
from mflux.models.minimax_h3.variants.minimax_h3 import compose_prompt

# torch.linspace + MiniMaxH3Scheduler (diffusers 0.40) grids, float32, verified bit-exact on 600 grids.
TORCH_SIGMAS_N9_SHIFT12 = [
    1.0,
    0.9882352948188782,
    0.9729729890823364,
    0.9523809552192688,
    0.9230769276618958,
    0.8780487775802612,
    0.800000011920929,
    0.6315789222717285,
    0.0,
]
TORCH_SIGMAS_N9_SHIFT3 = [
    1.0,
    0.9545454382896423,
    0.8999999761581421,
    0.8333333134651184,
    0.75,
    0.6428571343421936,
    0.5,
    0.30000001192092896,
    0.0,
]


@pytest.mark.fast
@pytest.mark.parametrize("shift, expected", [(12.0, TORCH_SIGMAS_N9_SHIFT12), (3.0, TORCH_SIGMAS_N9_SHIFT3)])
def test_sigma_grid_matches_torch_bit_for_bit(shift, expected):
    scheduler = MiniMaxH3Scheduler(shift=shift)
    scheduler.set_timesteps(9)
    assert scheduler.sigmas.dtype == np.float32
    assert scheduler.sigmas.tolist() == expected
    # N grid points drive N - 1 transformer evaluations; the timestep the model sees is 1 - sigma.
    assert len(scheduler.timesteps) == 8
    assert np.array_equal(scheduler.timesteps, (np.float32(1.0) - scheduler.sigmas[:-1]).astype(np.float32))


@pytest.mark.fast
def test_step_follows_the_rectified_flow_line_exactly():
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(9)
    x0 = mx.random.normal((6, 4))
    noise = mx.random.normal((6, 4))
    sigma = float(scheduler.sigmas[3])
    x_t = scheduler.scale_noise(x0, 1.0 - sigma, noise)
    velocity = x0 - noise  # the exact data-ward velocity of a straight path
    x_next = scheduler.step(velocity, 3, x_t)
    expected = scheduler.scale_noise(x0, 1.0 - float(scheduler.sigmas[4]), noise)
    assert float(mx.abs(x_next - expected).max()) < 1e-6


@pytest.mark.fast
def test_layout_partitions_rows_and_tags_each_stream():
    num_text, frames, height, width, audio = 5, 7, 4, 6, 8
    layout = build_packed_sequence(np.full((num_text,), TEXT_TAG, dtype=np.int32), frames, height, width, audio)
    video_rows = frames * (height // 2) * (width // 2)
    assert layout.sequence_length == num_text + audio * 2 + video_rows
    indices = np.concatenate(
        [np.array(layout.text_indices), np.array(layout.audio_indices), np.array(layout.video_indices)]
    )
    assert sorted(indices.tolist()) == list(range(layout.sequence_length))
    tags = np.array(layout.token_tags)
    assert (tags[np.array(layout.text_indices)] == TEXT_TAG).all()
    assert (tags[np.array(layout.audio_indices)] == AUDIO_TAG).all()
    assert (tags[np.array(layout.video_indices)] == VIDEO_TAG).all()
    assert layout.num_condition_video_rows == 0 and layout.num_condition_audio_rows == 0
    assert layout.position_ids.shape == (layout.sequence_length, 3)

    unique, inverse = build_row_timesteps(layout, 0.2, 0.5, 0.999, 1.0)
    rows = np.array(unique)[np.array(inverse)]
    assert unique.shape == (2,) and np.allclose(rows[np.array(layout.video_indices)], 0.2)
    assert np.allclose(rows[np.array(layout.audio_indices)], 0.5) and np.allclose(
        rows[np.array(layout.text_indices)], 0.2
    )


@pytest.mark.fast
def test_patchify_and_audio_packing_round_trip():
    latents = mx.random.normal((1, 24, 7, 4, 6))
    rows = patchify_video_latents(latents, (1, 2, 2))
    assert rows.shape == (7 * 2 * 3, 24 * 4)
    assert mx.array_equal(unpatchify_video_rows(rows, 24, 7, 4, 6, (1, 2, 2)), latents)
    audio = mx.random.normal((2, 32, 8))
    packed = pack_audio_rows(audio)
    assert packed.shape == (16, 32)
    assert mx.array_equal(unpack_audio_rows(packed, 2, 8), audio)


@pytest.mark.fast
def test_frame_and_canvas_contracts():
    assert align_num_frames(120) == 124 and align_num_frames(124) == 124 and align_num_frames(125) == 141
    assert video_latent_num_frames(124) == 37 and audio_latent_num_frames(124) == 207
    assert resolve_canvas_size(16, 9, 32) == (768, 1344)
    assert resolve_canvas_size(9, 16, 32) == (1344, 768)
    assert resolve_canvas_size(1, 1, 32) == (768, 768)


@pytest.mark.fast
def test_compose_prompt_wraps_plain_text_and_keeps_structured_prompts():
    assert compose_prompt("A fox", "wind", "piano") == (
        "integrated_multimodal_description: A fox\n\noverall_soundscape: wind\n\nnon_diegetic_music: piano"
    )
    structured = "integrated_multimodal_description: [Shot 1] A fox.\n\noverall_soundscape: Wind."
    assert compose_prompt(structured) == structured
    assert compose_prompt("  A fox  ") == "integrated_multimodal_description: A fox"
