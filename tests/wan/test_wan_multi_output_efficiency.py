import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.wan.prompt_embed_store import WanPromptEmbedStore
from mflux.models.wan.variants.wan2_2_ti2v import Wan2_2_TI2V


def _fake_tokenize(self, *, cleaned, max_sequence_length):
    # Deterministic per-prompt token ids so cache keys differ per prompt.
    seed = sum(ord(ch) for ch in "".join(cleaned)) % 97
    return {
        "input_ids": np.full((len(cleaned), 8), seed, dtype=np.int64),
        "attention_mask": np.ones((len(cleaned), 8), dtype=np.int64),
    }


def _bare_model(tmp_path, *, disk_cache_enabled=False):
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.prompt_embed_cache = {}
    model._prompt_embed_store = WanPromptEmbedStore(
        enabled=disk_cache_enabled, cache_dir=tmp_path / "prompt-cache"
    )
    model._prompt_embed_fingerprint = "test-fingerprint"
    model._keep_text_encoder_resident = False
    model._resident_text_encoder = None
    return model


def test_wan_prompt_embed_cache_reuses_first_load(monkeypatch, tmp_path):
    model = _bare_model(tmp_path)
    observed: list[int] = []
    expected = mx.array([[1.0]], dtype=mx.float32)

    def fake_load(self, *, text_inputs, max_sequence_length):
        observed.append(max_sequence_length)
        return expected

    monkeypatch.setattr(Wan2_2_TI2V, "_tokenize_prompts", _fake_tokenize)
    monkeypatch.setattr(Wan2_2_TI2V, "_load_t5_prompt_embeds", fake_load)

    first = model._get_t5_prompt_embeds(["hello"], max_sequence_length=512)
    second = model._get_t5_prompt_embeds(["hello"], max_sequence_length=512)

    assert observed == [512]
    np.testing.assert_array_equal(np.array(first), np.array(second))


def test_wan_prompt_embed_cache_stays_bounded(monkeypatch, tmp_path):
    model = _bare_model(tmp_path)

    monkeypatch.setattr(Wan2_2_TI2V, "_tokenize_prompts", _fake_tokenize)
    monkeypatch.setattr(
        Wan2_2_TI2V,
        "_load_t5_prompt_embeds",
        lambda self, *, text_inputs, max_sequence_length: mx.array(
            [[float(text_inputs["input_ids"][0, 0]) + max_sequence_length]], dtype=mx.float32
        ),
    )

    model._get_t5_prompt_embeds(["alpha"], max_sequence_length=512)
    model._get_t5_prompt_embeds(["beta"], max_sequence_length=512)
    model._get_t5_prompt_embeds(["gamma"], max_sequence_length=512)

    assert len(model.prompt_embed_cache) == 2


def test_wan_prompt_embed_disk_cache_hit_skips_loader(monkeypatch, tmp_path):
    # A disk hit must serve the embeds WITHOUT calling the torch loader; a
    # miss must store them for the next process.
    model = _bare_model(tmp_path, disk_cache_enabled=True)
    loads: list[int] = []
    expected = mx.array([[42.0]], dtype=mx.float32)

    def fake_load(self, *, text_inputs, max_sequence_length):
        loads.append(max_sequence_length)
        return expected

    monkeypatch.setattr(Wan2_2_TI2V, "_tokenize_prompts", _fake_tokenize)
    monkeypatch.setattr(Wan2_2_TI2V, "_load_t5_prompt_embeds", fake_load)

    first = model._get_t5_prompt_embeds(["hello"], max_sequence_length=512)
    assert loads == [512]

    # New model instance = fresh in-process cache, same disk store: disk hit.
    second_model = _bare_model(tmp_path, disk_cache_enabled=True)
    monkeypatch.setattr(
        Wan2_2_TI2V,
        "_load_t5_prompt_embeds",
        lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("loader must not run on a disk hit")),
    )
    second = second_model._get_t5_prompt_embeds(["hello"], max_sequence_length=512)

    # The hit path re-casts to the runtime precision (bf16), which numpy
    # cannot view directly: compare through float32.
    np.testing.assert_array_equal(
        np.array(first.astype(mx.float32)), np.array(second.astype(mx.float32))
    )


def test_wan_prompt_embed_disk_key_varies_by_prompt(monkeypatch, tmp_path):
    model = _bare_model(tmp_path, disk_cache_enabled=True)
    monkeypatch.setattr(Wan2_2_TI2V, "_tokenize_prompts", _fake_tokenize)

    key_a = model._prompt_embed_disk_key(
        text_inputs=_fake_tokenize(model, cleaned=["a"], max_sequence_length=8), max_sequence_length=512
    )
    key_b = model._prompt_embed_disk_key(
        text_inputs=_fake_tokenize(model, cleaned=["b"], max_sequence_length=8), max_sequence_length=512
    )

    assert key_a is not None and key_b is not None
    assert key_a != key_b


def test_wan_resident_text_encoder_loads_once_across_prompts(monkeypatch, tmp_path):
    # keep_text_encoder_resident=True must reuse ONE encoder instance across
    # different-prompt encodes (the storyboard scene-chaining case). The
    # transformers package is replaced at the sys.modules level so the
    # function-local `from transformers import UMT5EncoderModel` resolves to
    # the fake regardless of lazy-module behavior, and no HF loading runs.
    import sys
    import types

    import torch

    class FakeEncoderOutput:
        def __init__(self, batch, seq):
            self.last_hidden_state = torch.zeros((batch, seq, 4), dtype=torch.bfloat16)

    created = []

    class FakeEncoder:
        def __init__(self):
            created.append(self)
            self.forward_calls = 0

        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask):
            self.forward_calls += 1
            return FakeEncoderOutput(input_ids.shape[0], input_ids.shape[1])

    class FakeUMT5:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return FakeEncoder()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.UMT5EncoderModel = FakeUMT5
    fake_utils = types.ModuleType("transformers.utils")
    fake_utils.logging = types.SimpleNamespace(
        get_verbosity=lambda: 0,
        set_verbosity_error=lambda: None,
        set_verbosity=lambda level: None,
    )
    fake_transformers.utils = fake_utils
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_utils)

    model = _bare_model(tmp_path)
    model._keep_text_encoder_resident = True
    (tmp_path / "text_encoder").mkdir()
    model.root_path = tmp_path
    monkeypatch.setattr(Wan2_2_TI2V, "_tokenize_prompts", _fake_tokenize)

    model._get_t5_prompt_embeds(["scene one"], max_sequence_length=8)
    model._get_t5_prompt_embeds(["scene two, different prompt"], max_sequence_length=8)

    assert len(created) == 1
    assert created[0].forward_calls == 2
    assert model._resident_text_encoder is created[0]


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
