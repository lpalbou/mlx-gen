"""Qwen3-VL vision-side contracts used by MiniMax-H3 first-frame conditioning."""

import numpy as np
import PIL.Image
import pytest

from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_model import rope_index
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_vision_model import (
    position_embedding_taps,
    preprocess_image,
    smart_resize,
    vision_position_ids,
)


@pytest.mark.fast
def test_preprocess_keeps_canvas_sized_images_and_orders_patches_by_merge_block():
    image = PIL.Image.fromarray((np.random.default_rng(0).random((544, 960, 3)) * 255).astype(np.uint8))
    patches, grid = preprocess_image(image)
    assert grid == (1, 34, 60) and patches.shape == (34 * 60, 3 * 2 * 16 * 16)
    pixels = (np.asarray(image, dtype=np.float32) / 255.0 - 0.5) / 0.5
    # first patch = top-left 16x16 block, second patch = the block to its right (same merge window), duplicated in time
    first = patches[0].reshape(3, 2, 16, 16)
    assert np.allclose(first[:, 0], pixels[:16, :16].transpose(2, 0, 1)) and np.array_equal(first[:, 0], first[:, 1])
    second = patches[1].reshape(3, 2, 16, 16)
    assert np.allclose(second[:, 0], pixels[:16, 16:32].transpose(2, 0, 1))
    third = patches[2].reshape(3, 2, 16, 16)
    assert np.allclose(third[:, 0], pixels[16:32, :16].transpose(2, 0, 1))
    assert smart_resize(544, 960, 32, 65536, 16777216) == (544, 960)
    assert smart_resize(100, 100, 32, 65536, 16777216) == (256, 256)


@pytest.mark.fast
def test_vision_position_ids_and_taps_follow_merge_block_order():
    positions = vision_position_ids((1, 4, 6), 2)
    assert positions.shape == (24, 2)
    assert positions[:4].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    indices, weights = position_embedding_taps((1, 4, 6), side=48, merge_size=2)
    assert indices.shape == (24, 4) and np.allclose(weights.sum(axis=1), 1.0)


@pytest.mark.fast
def test_rope_index_lays_out_text_then_image_grid():
    ids = np.array([5, 6, 152] + [150] * 6 + [153, 7])
    positions = rope_index(ids, 150, [(1, 4, 6)], 2)
    assert positions[:, :3].tolist() == [[0, 1, 2]] * 3
    assert positions[0, 3:9].tolist() == [3] * 6  # temporal axis constant
    assert positions[1, 3:9].tolist() == [3, 3, 3, 4, 4, 4]  # rows
    assert positions[2, 3:9].tolist() == [3, 4, 5, 3, 4, 5]  # columns
    assert positions[:, 9:].tolist() == [[6, 7]] * 3  # text resumes after max(h, w) // merge
