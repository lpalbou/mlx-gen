import gc
import weakref
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import UMT5Config, UMT5EncoderModel

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.wan.prompt_embed_store import WanPromptEmbedStore
from mflux.models.wan.variants.wan2_2_ti2v import Wan2_2_TI2V
from mflux.models.wan.wan_text_encoder import WanTextEncoder
from mflux.models.wan.wan_text_encoder_loader import WanTextEncoderLoader

pytestmark = pytest.mark.fast


class TinyUMT5:
    # A randomly initialised UMT5 encoder of the real architecture, stored like the Wan snapshots: bfloat16
    # safetensors in two shards, the token table only as `shared.weight`. The float32 model holds the same values.
    @staticmethod
    def write(path) -> UMT5EncoderModel:
        torch.manual_seed(0)
        config = UMT5Config(
            vocab_size=96,
            d_model=32,
            d_kv=8,
            d_ff=48,
            num_layers=3,
            num_heads=4,
            feed_forward_proj="gated-gelu",
            dense_act_fn="gelu_new",
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
        )
        model = UMT5EncoderModel(config).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(0.0, 0.5)
                parameter.copy_(parameter.to(torch.bfloat16).float())
        path.mkdir(parents=True, exist_ok=True)
        config.save_pretrained(path)
        tensors = {
            name: tensor.to(torch.bfloat16).contiguous()
            for name, tensor in model.state_dict().items()
            if name != "encoder.embed_tokens.weight"
        }
        names = sorted(tensors)
        save_file({name: tensors[name] for name in names[::2]}, str(path / "model-00001-of-00002.safetensors"))
        save_file({name: tensors[name] for name in names[1::2]}, str(path / "model-00002-of-00002.safetensors"))
        return model

    @staticmethod
    def encoder(path) -> WanTextEncoder:
        encoder = WanTextEncoder(path)
        encoder.device = torch.device("cpu")
        return encoder

    @staticmethod
    def whole_model(model: UMT5EncoderModel, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return (
                model(torch.from_numpy(input_ids).long(), torch.from_numpy(attention_mask).long())
                .last_hidden_state.float()
                .numpy()
            )


def _inputs(lengths, padded_length):
    rng = np.random.default_rng(7)
    input_ids = np.zeros((len(lengths), padded_length), dtype=np.int64)
    attention_mask = np.zeros((len(lengths), padded_length), dtype=np.int64)
    for row, length in enumerate(lengths):
        input_ids[row, :length] = rng.integers(2, 96, size=length)
        attention_mask[row, :length] = 1
    return input_ids, attention_mask


def test_streamed_encode_matches_the_whole_float32_model_on_the_real_tokens(tmp_path):
    model = TinyUMT5.write(tmp_path / "text_encoder")
    input_ids, attention_mask = _inputs([9, 5], padded_length=24)

    encoder = TinyUMT5.encoder(tmp_path / "text_encoder")
    streamed = encoder.encode(input_ids, attention_mask)
    reference = TinyUMT5.whole_model(model, input_ids, attention_mask)

    # No weight outlives the call: the encoder is back to its storage-less structure.
    assert all(parameter.is_meta for parameter in encoder._model.parameters())

    # The batch runs at its longest real row: the padding positions after it are never computed.
    assert streamed.shape == (2, 9, 32)
    for row, length in enumerate([9, 5]):
        np.testing.assert_allclose(streamed[row, :length], reference[row, :length], rtol=0, atol=1e-5)


def test_streamed_encode_is_the_whole_model_bit_for_bit_without_padding(tmp_path):
    model = TinyUMT5.write(tmp_path / "text_encoder")
    input_ids, attention_mask = _inputs([11, 11], padded_length=11)

    streamed = TinyUMT5.encoder(tmp_path / "text_encoder").encode(input_ids, attention_mask)

    np.testing.assert_array_equal(streamed, TinyUMT5.whole_model(model, input_ids, attention_mask))


def test_a_float32_source_encodes_exactly_as_its_package(tmp_path):
    # The package stores bfloat16 weights with the feed-forward output layers in float32; the source stores
    # float32 everywhere. Both must give the same embeddings.
    model = TinyUMT5.write(tmp_path / "package")
    torch.manual_seed(1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.5)
    state = {name: tensor for name, tensor in model.state_dict().items() if name != "encoder.embed_tokens.weight"}
    layouts = {
        "source": {name: tensor.contiguous() for name, tensor in state.items()},
        "package": {
            name: (tensor if ".wo." in name else tensor.to(torch.bfloat16)).contiguous()
            for name, tensor in state.items()
        },
    }
    encoded = {}
    input_ids, attention_mask = _inputs([9, 5], padded_length=12)
    for layout, tensors in layouts.items():
        path = tmp_path / layout
        path.mkdir(exist_ok=True)
        model.config.save_pretrained(path)
        save_file(tensors, str(path / "model.safetensors"))
        encoded[layout] = TinyUMT5.encoder(path).encode(input_ids, attention_mask)

    np.testing.assert_array_equal(encoded["source"], encoded["package"])


def test_a_failed_encode_leaves_the_encoder_empty_and_reusable(tmp_path):
    TinyUMT5.write(tmp_path / "text_encoder")
    input_ids, attention_mask = _inputs([7, 4], padded_length=10)
    encoder = TinyUMT5.encoder(tmp_path / "text_encoder")
    expected = encoder.encode(input_ids, attention_mask)

    def fail(module, args, output):
        raise RuntimeError("block failed")

    failing = encoder._model.encoder.block[1].register_forward_hook(fail)
    with pytest.raises(RuntimeError, match="block failed"):
        encoder.encode(input_ids, attention_mask)
    failing.remove()

    assert all(parameter.is_meta for parameter in encoder._model.parameters())
    for block in encoder._model.encoder.block:
        assert not block._forward_pre_hooks and not block._forward_hooks
    np.testing.assert_array_equal(encoder.encode(input_ids, attention_mask), expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the Metal device")
def test_streamed_encode_on_the_gpu_matches_the_whole_float32_model(tmp_path):
    model = TinyUMT5.write(tmp_path / "text_encoder")
    input_ids, attention_mask = _inputs([9, 5], padded_length=24)

    encoder = WanTextEncoder(tmp_path / "text_encoder")
    assert encoder.device.type == "mps"
    streamed = encoder.encode(input_ids, attention_mask)
    reference = TinyUMT5.whole_model(model, input_ids, attention_mask)

    assert all(parameter.is_meta for parameter in encoder._model.parameters())
    assert torch.mps.current_allocated_memory() == 0
    for row, length in enumerate([9, 5]):
        np.testing.assert_allclose(streamed[row, :length], reference[row, :length], rtol=0, atol=1e-4)


def _bare_model(tmp_path, *, keep_resident: bool):
    return SimpleNamespace(
        model_config=ModelConfig.wan2_2_ti2v_5b(),
        _resident_text_encoder=None,
        _keep_text_encoder_resident=keep_resident,
        _component_subdir_path=lambda component: tmp_path / component,
    )


def _counting_loader(monkeypatch):
    loaded = []

    def load(*, text_encoder_path):
        loaded.append(TinyUMT5.encoder(text_encoder_path))
        return loaded[-1]

    monkeypatch.setattr(WanTextEncoderLoader, "load", staticmethod(load))
    return loaded


def test_engine_encode_pads_to_the_sequence_length_and_releases_the_encoder(monkeypatch, tmp_path):
    model_reference = TinyUMT5.write(tmp_path / "text_encoder")
    loaded = _counting_loader(monkeypatch)
    input_ids, attention_mask = _inputs([6, 3], padded_length=16)
    model = _bare_model(tmp_path, keep_resident=False)

    embeds = Wan2_2_TI2V._load_t5_prompt_embeds(
        model, text_inputs={"input_ids": input_ids, "attention_mask": attention_mask}, max_sequence_length=16
    )

    assert embeds.shape == (2, 16, 32)
    assert embeds.dtype == ModelConfig.precision
    expected = np.zeros((2, 16, 32), dtype=np.float32)
    reference = TinyUMT5.whole_model(model_reference, input_ids, attention_mask)
    for row, length in enumerate([6, 3]):
        expected[row, :length] = reference[row, :length]
    rounded = np.array(mx.array(expected).astype(ModelConfig.precision).astype(mx.float32))
    np.testing.assert_allclose(np.array(embeds.astype(mx.float32)), rounded, rtol=1e-2, atol=1e-6)
    assert not np.array(embeds.astype(mx.float32))[0, 6:].any()
    assert not np.array(embeds.astype(mx.float32))[1, 3:].any()

    released = weakref.ref(loaded.pop())
    gc.collect()
    assert model._resident_text_encoder is None
    assert released() is None


def test_resident_text_encoder_is_built_once_across_prompts(monkeypatch, tmp_path):
    TinyUMT5.write(tmp_path / "text_encoder")
    loaded = _counting_loader(monkeypatch)
    model = _bare_model(tmp_path, keep_resident=True)

    for lengths in ([4, 2], [7, 5]):
        input_ids, attention_mask = _inputs(lengths, padded_length=8)
        Wan2_2_TI2V._load_t5_prompt_embeds(
            model, text_inputs={"input_ids": input_ids, "attention_mask": attention_mask}, max_sequence_length=8
        )

    assert len(loaded) == 1
    assert model._resident_text_encoder is loaded[0]


def test_prompt_cache_key_no_longer_matches_entries_of_the_bfloat16_cpu_encoder():
    text_inputs = {
        "input_ids": np.array([[1, 2]], dtype=np.int64),
        "attention_mask": np.array([[1, 1]], dtype=np.int64),
    }
    model = SimpleNamespace(
        _prompt_embed_store=SimpleNamespace(enabled=True),
        _prompt_embed_fingerprint="same-encoder",
        model_config=ModelConfig.wan2_2_ti2v_5b(),
    )
    # The key 0.40.1 wrote for these tokens: the same fields without the encode policy.
    old_key = WanPromptEmbedStore.compute_key(
        encoder_fingerprint="same-encoder",
        input_ids_bytes=str((1, 2)).encode("utf-8") + text_inputs["input_ids"].tobytes(),
        attention_mask_bytes=str((1, 2)).encode("utf-8") + text_inputs["attention_mask"].tobytes(),
        max_sequence_length=2,
        precision=str(ModelConfig.precision),
    )

    new_key = Wan2_2_TI2V._prompt_embed_disk_key(model, text_inputs=text_inputs, max_sequence_length=2)

    assert new_key == WanPromptEmbedStore.compute_key(
        encoder_fingerprint=f"same-encoder:{WanTextEncoderLoader.ENCODE_POLICY_ID}",
        input_ids_bytes=str((1, 2)).encode("utf-8") + text_inputs["input_ids"].tobytes(),
        attention_mask_bytes=str((1, 2)).encode("utf-8") + text_inputs["attention_mask"].tobytes(),
        max_sequence_length=2,
        precision=str(ModelConfig.precision),
    )
    assert new_key != old_key
