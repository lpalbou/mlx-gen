import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.wan.variants.wan2_2_ti2v import Wan2_2_TI2V


def test_wan_prompt_embed_cache_reuses_first_load(monkeypatch):
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.prompt_embed_cache = {}
    observed: list[tuple[tuple[str, ...], int]] = []
    expected = mx.array([[1.0]], dtype=mx.float32)

    def fake_load(self, *, cleaned, max_sequence_length):
        observed.append((tuple(cleaned), max_sequence_length))
        return expected

    monkeypatch.setattr(Wan2_2_TI2V, "_load_t5_prompt_embeds", fake_load)

    first = model._get_t5_prompt_embeds(["hello"], max_sequence_length=512)
    second = model._get_t5_prompt_embeds(["hello"], max_sequence_length=512)

    assert observed == [(("hello",), 512)]
    np.testing.assert_array_equal(np.array(first), np.array(second))


def test_wan_prompt_embed_cache_stays_bounded(monkeypatch):
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.prompt_embed_cache = {}

    monkeypatch.setattr(
        Wan2_2_TI2V,
        "_load_t5_prompt_embeds",
        lambda self, *, cleaned, max_sequence_length: mx.array([[len(cleaned[0]) + max_sequence_length]], dtype=mx.float32),
    )

    model._get_t5_prompt_embeds(["alpha"], max_sequence_length=512)
    model._get_t5_prompt_embeds(["beta"], max_sequence_length=512)
    model._get_t5_prompt_embeds(["gamma"], max_sequence_length=512)

    assert len(model.prompt_embed_cache) == 2


def test_wan_first_frame_condition_cache_reuses_same_source(monkeypatch, tmp_path):
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.image_condition_cache = {}
    observed: list[tuple[str, int, int]] = []
    image_path = tmp_path / "source.png"
    Image.new("RGB", (4, 4), color="white").save(image_path)
    expected = mx.array([1.0], dtype=mx.float32)

    def fake_load(self, *, image_path, height, width):
        observed.append((str(image_path), height, width))
        return expected

    monkeypatch.setattr(Wan2_2_TI2V, "_load_first_frame_condition", fake_load)

    first = model._encode_first_frame_condition(image_path=image_path, height=64, width=96)
    second = model._encode_first_frame_condition(image_path=image_path, height=64, width=96)

    assert observed == [(str(image_path), 64, 96)]
    np.testing.assert_array_equal(np.array(first), np.array(second))


def test_wan_video_condition_cache_reuses_same_source(monkeypatch, tmp_path):
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.image_condition_cache = {}
    observed: list[tuple[str, int, int, int, int]] = []
    image_path = tmp_path / "source.png"
    Image.new("RGB", (4, 4), color="white").save(image_path)
    expected = mx.array([2.0], dtype=mx.float32)

    def fake_load(self, *, image_path, height, width, num_frames, batch_size):
        observed.append((str(image_path), height, width, num_frames, batch_size))
        return expected

    monkeypatch.setattr(Wan2_2_TI2V, "_load_video_condition", fake_load)

    first = model._encode_video_condition(
        image_path=image_path,
        height=64,
        width=96,
        num_frames=9,
        batch_size=1,
    )
    second = model._encode_video_condition(
        image_path=image_path,
        height=64,
        width=96,
        num_frames=9,
        batch_size=1,
    )

    assert observed == [(str(image_path), 64, 96, 9, 1)]
    np.testing.assert_array_equal(np.array(first), np.array(second))
