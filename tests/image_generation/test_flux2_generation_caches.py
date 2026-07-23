from types import SimpleNamespace

import mlx.core as mx
import pytest

import mflux.models.flux2.variants.txt2img.flux2_klein as flux2_klein_module
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.utils.compiled_predict_cache import CompiledPredictCache


class _FakeTokenizer:
    def tokenize(self, prompt, max_length=512):
        return SimpleNamespace(
            input_ids=mx.zeros((1, 8), dtype=mx.int32),
            attention_mask=mx.ones((1, 8), dtype=mx.int32),
        )


class _FakeTextEncoder:
    def __init__(self):
        self.encode_calls = 0

    def get_prompt_embeds(self, input_ids, attention_mask, hidden_state_layers):
        self.encode_calls += 1
        return mx.ones((1, 8, 16), dtype=mx.float32)


class _FakeVAE:
    def decode_packed_latents(self, packed_latents):
        # Identity decode keeps the final latents observable for equality checks.
        return packed_latents


def _install_fake_flux2(monkeypatch):
    text_encoder = _FakeTextEncoder()

    def fake_init(model, model_config, quantize, model_path=None, lora_paths=None, lora_scales=None):
        model.prompt_cache = {}
        model.compiled_predict_cache = CompiledPredictCache()
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.tokenizers = {"qwen3": _FakeTokenizer()}
        model.text_encoder = text_encoder
        model.vae = _FakeVAE()
        model.transformer = lambda **kwargs: mx.zeros_like(kwargs["hidden_states"])
        model.bits = None
        model.lora_paths = None
        model.lora_scales = None

    monkeypatch.setattr(Flux2Initializer, "init", staticmethod(fake_init))
    monkeypatch.setattr(flux2_klein_module.ImageUtil, "to_image", staticmethod(lambda **kwargs: kwargs))
    monkeypatch.setattr(flux2_klein_module.LoRALoader, "extra_metadata_for_model", staticmethod(lambda model: {}))
    return text_encoder


def _generate(model, seed=11, prompt="a red fox"):
    return model.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=2,
        height=64,
        width=64,
        guidance=1.0,
    )


class TestFlux2PromptCacheWiring:
    def test_identical_prompt_second_generation_skips_reencode(self, monkeypatch):
        text_encoder = _install_fake_flux2(monkeypatch)
        model = Flux2Klein(model_config=ModelConfig.flux2_klein_4b())

        _generate(model, seed=1)
        _generate(model, seed=2)

        assert text_encoder.encode_calls == 1
        assert len(model.prompt_cache) == 1

    def test_different_prompt_reencodes(self, monkeypatch):
        text_encoder = _install_fake_flux2(monkeypatch)
        model = Flux2Klein(model_config=ModelConfig.flux2_klein_4b())

        _generate(model, prompt="a red fox")
        _generate(model, prompt="a blue fox")

        assert text_encoder.encode_calls == 2

    def test_edit_encode_prompt_pair_uses_the_cache(self, monkeypatch):
        text_encoder = _install_fake_flux2(monkeypatch)
        model = Flux2KleinEdit(model_config=ModelConfig.flux2_klein_4b())

        model._encode_prompt_pair(prompt="same edit", negative_prompt="", guidance=1.0)
        model._encode_prompt_pair(prompt="same edit", negative_prompt="", guidance=1.0)

        assert text_encoder.encode_calls == 1


class TestFlux2PromptCacheBounds:
    def test_hit_miss_and_lru_eviction(self):
        text_encoder = _FakeTextEncoder()
        tokenizer = _FakeTokenizer()
        cache: dict = {}

        def encode(prompt):
            return Flux2PromptEncoder.encode_prompt(
                prompt=prompt,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt_cache=cache,
            )

        first_embeds, first_ids = encode("prompt-0")
        cached_embeds, cached_ids = encode("prompt-0")
        assert text_encoder.encode_calls == 1
        assert cached_embeds is first_embeds
        assert cached_ids is first_ids

        # Refresh prompt-0, then overflow the bound: prompt-1 (now the oldest) must go.
        for index in range(1, Flux2PromptEncoder.PROMPT_CACHE_MAX_ENTRIES + 1):
            encode(f"prompt-{index}")
        encode("prompt-0")
        encode(f"prompt-{Flux2PromptEncoder.PROMPT_CACHE_MAX_ENTRIES + 1}")

        assert len(cache) == Flux2PromptEncoder.PROMPT_CACHE_MAX_ENTRIES
        keys = list(cache)
        assert any("'prompt-0'" in key for key in keys)
        assert not any("'prompt-1'" in key for key in keys)

    def test_cache_key_covers_encode_parameters(self):
        text_encoder = _FakeTextEncoder()
        cache: dict = {}
        Flux2PromptEncoder.encode_prompt(
            prompt="p", tokenizer=_FakeTokenizer(), text_encoder=text_encoder,
            max_sequence_length=512, prompt_cache=cache,
        )
        Flux2PromptEncoder.encode_prompt(
            prompt="p", tokenizer=_FakeTokenizer(), text_encoder=text_encoder,
            max_sequence_length=256, prompt_cache=cache,
        )
        assert text_encoder.encode_calls == 2
        assert len(cache) == 2


class TestFlux2CompiledPredictReuse:
    def test_same_shape_calls_reuse_one_compiled_predict(self, monkeypatch):
        _install_fake_flux2(monkeypatch)
        model = Flux2Klein(model_config=ModelConfig.flux2_klein_4b())

        _generate(model, seed=1)
        first_entries = dict(model.compiled_predict_cache._entries)
        _generate(model, seed=2)
        second_entries = dict(model.compiled_predict_cache._entries)

        assert len(model.compiled_predict_cache) == 1
        assert first_entries == second_entries  # same callable objects, no rebuild

    def test_transformer_replacement_invalidates_compiled_predict(self, monkeypatch):
        _install_fake_flux2(monkeypatch)
        model = Flux2Klein(model_config=ModelConfig.flux2_klein_4b())

        _generate(model, seed=1)
        first_entries = dict(model.compiled_predict_cache._entries)
        model.transformer = lambda **kwargs: mx.zeros_like(kwargs["hidden_states"])
        _generate(model, seed=2)
        second_entries = dict(model.compiled_predict_cache._entries)

        assert len(model.compiled_predict_cache) == 1
        assert all(second_entries[key] is not first_entries[key] for key in first_entries)


class TestFlux2LowRamPromptCacheInteraction:
    # 0095 claim: --low-ram multi-seed keeps prompt embeds usable across the
    # MemorySaver text-encoder release. The encode of seed 1 fills the cache
    # BEFORE before_loop drops the encoder; seed 2 with the same prompt must hit
    # the cache and never touch the (now None) encoder.
    def test_same_prompt_multi_seed_survives_text_encoder_release(self, monkeypatch):
        from mflux.callbacks.instances.memory_saver import MemorySaver

        monkeypatch.setenv("MFLUX_RUNTIME_MEMORY_TELEMETRY", "0")
        text_encoder = _install_fake_flux2(monkeypatch)
        model = Flux2Klein(model_config=ModelConfig.flux2_klein_4b())
        model.callbacks.register(MemorySaver(model=model, keep_transformer=True))

        _generate(model, seed=1)
        assert model.text_encoder is None  # MemorySaver released it in before_loop

        _generate(model, seed=2)

        assert text_encoder.encode_calls == 1
        assert len(model.prompt_cache) == 1

    def test_new_prompt_after_text_encoder_release_fails_loud(self, monkeypatch):
        from mflux.callbacks.instances.memory_saver import MemorySaver

        monkeypatch.setenv("MFLUX_RUNTIME_MEMORY_TELEMETRY", "0")
        _install_fake_flux2(monkeypatch)
        model = Flux2Klein(model_config=ModelConfig.flux2_klein_4b())
        model.callbacks.register(MemorySaver(model=model, keep_transformer=True))

        _generate(model, seed=1, prompt="a red fox")

        with pytest.raises(AttributeError):
            _generate(model, seed=2, prompt="a different prompt")


class _CancelForTest(RuntimeError):
    pass


class TestFlux2AbortSafety:
    # Contract for embedding hosts (BlackPixel cooperative cancel): an exception
    # raised inside a progress callback must propagate out of generate_image and
    # leave the instance reusable, with prompt/compile caches consistent.
    def test_exception_from_progress_callback_propagates_and_model_stays_reusable(self, monkeypatch):
        text_encoder = _install_fake_flux2(monkeypatch)
        model = Flux2Klein(model_config=ModelConfig.flux2_klein_4b())
        state = {"raise_on_denoise": True}

        def progress_callback(event):
            if state["raise_on_denoise"] and event.phase == "denoise" and event.step >= 2:
                raise _CancelForTest("host cancel")

        model.callbacks.subscribe_progress(progress_callback)

        with pytest.raises(_CancelForTest):
            _generate(model, seed=7)

        state["raise_on_denoise"] = False
        aborted_then_retried = _generate(model, seed=7)

        # Caches stayed consistent: the retry hit the materialized prompt entry and
        # the compiled predict (the fake encoder is shared with the clean model
        # below, so assert the count before that second instance encodes).
        assert text_encoder.encode_calls == 1
        assert len(model.prompt_cache) == 1
        assert len(model.compiled_predict_cache) == 1

        clean_model = Flux2Klein(model_config=ModelConfig.flux2_klein_4b())
        clean = _generate(clean_model, seed=7)

        assert mx.array_equal(aborted_then_retried["decoded_latents"], clean["decoded_latents"])
