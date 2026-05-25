import math

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.ernie_image.model.mistral3_text_encoder.attention import Mistral3Attention
from mflux.models.ernie_image.model.mistral3_text_encoder.rope import Mistral3YarnRotaryEmbedding
from mflux.models.ernie_image.model.mistral3_text_encoder.text_encoder import Mistral3TextEncoder
from mflux.models.ernie_image.scheduler import ErnieImageScheduler
from mflux.models.ernie_image.tokenizer import ErnieImageTokenizer
from mflux.models.ernie_image.weights.ernie_image_weight_definition import ErnieImageWeightDefinition
from mflux.models.ernie_image.weights.ernie_image_weight_mapping import ErnieImageWeightMapping


class AddLayer:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, hidden_states, attention_mask, position_ids, position_embeddings):
        return hidden_states + self.value


class DummyRawTokenizer:
    bos_token_id = 7
    pad_token_id = 99

    def __call__(self, prompt, add_special_tokens, truncation, padding, max_length):
        del truncation, padding, max_length
        if prompt == "":
            return {"input_ids": []}
        ids = [ord(c) % 10 for c in prompt]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids]
        return {"input_ids": ids}


@pytest.mark.fast
def test_yarn_rope_inv_freq_matches_reference_formula():
    rope = Mistral3YarnRotaryEmbedding(dim=8, base=10000.0, factor=4.0, original_max_position_embeddings=128)
    actual = np.array(rope.inv_freq)

    dim = 8
    base = 10000.0
    factor = 4.0
    beta_fast = 32.0
    beta_slow = 1.0
    original_max_position_embeddings = 128
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    def correction_dim(num_rotations):
        return (dim * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
    extrapolation_factor = 1 - ramp
    expected = inv_freq_interpolation * (1 - extrapolation_factor) + inv_freq_extrapolation * extrapolation_factor

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.fast
def test_llama4_attention_scale_only_changes_after_original_context():
    attention = Mistral3Attention(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        llama_4_scaling_beta=0.1,
        original_max_position_embeddings=4,
    )
    scale = attention._llama_4_attention_scale(mx.array([[0, 3, 4, 8]], dtype=mx.int32))

    expected = np.array([1.0, 1.0, 1.0 + 0.1 * math.log(2.0), 1.0 + 0.1 * math.log(3.0)])
    np.testing.assert_allclose(np.array(scale).reshape(-1), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.fast
def test_repeat_kv_matches_attention_head_expansion():
    hidden_states = mx.array([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    repeated = Mistral3Attention._repeat_kv(hidden_states, n_rep=2)

    expected = np.array([[[[1.0, 2.0]], [[1.0, 2.0]], [[3.0, 4.0]], [[3.0, 4.0]]]])
    np.testing.assert_array_equal(np.array(repeated), expected)


@pytest.mark.fast
def test_text_encoder_returns_penultimate_layer_hidden_state():
    encoder = Mistral3TextEncoder(
        vocab_size=4,
        hidden_size=4,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=8,
        head_dim=4,
    )
    encoder.embed_tokens.weight = mx.zeros((4, 4), dtype=mx.float32)
    encoder.layers = [AddLayer(1.0), AddLayer(10.0)]

    output = encoder(mx.array([[1, 2]], dtype=mx.int32))

    np.testing.assert_array_equal(np.array(output.astype(mx.float32)), np.ones((1, 2, 4), dtype=np.float32))


@pytest.mark.fast
def test_ernie_tokenizer_matches_diffusers_bos_fallback_and_longest_padding():
    tokenizer = ErnieImageTokenizer(DummyRawTokenizer())
    output = tokenizer.tokenize(["A", ""])

    np.testing.assert_array_equal(np.array(output.input_ids), np.array([[7, 5], [7, 99]], dtype=np.int32))
    np.testing.assert_array_equal(np.array(output.attention_mask), np.array([[1, 1], [1, 0]], dtype=np.int32))


@pytest.mark.fast
def test_causal_mask_applies_padding_to_key_positions():
    hidden_states = mx.zeros((1, 3, 4), dtype=mx.float32)
    attention_mask = mx.array([[1, 1, 0]], dtype=mx.int32)
    mask = Mistral3TextEncoder._causal_mask(attention_mask, hidden_states, seq_len=3)
    values = np.array(mask[0, 0])

    assert values[0, 0] == 0
    assert np.isneginf(values[0, 1])
    assert np.isneginf(values[0, 2])
    assert values[1, 0] == 0
    assert values[1, 1] == 0
    assert np.isneginf(values[1, 2])


@pytest.mark.fast
def test_text_encoder_weight_mapping_uses_mistral3_language_prefix():
    mapping = ErnieImageWeightMapping.get_text_encoder_mapping()
    from_patterns = {pattern for target in mapping for pattern in target.from_pattern}

    assert "language_model.model.embed_tokens.weight" in from_patterns
    assert "language_model.model.layers.{layer}.self_attn.q_proj.weight" in from_patterns
    assert "model.layers.{layer}.self_attn.q_norm.weight" not in from_patterns


@pytest.mark.fast
def test_transformer_weight_mapping_uses_diffusers_ernie_names():
    mapping = ErnieImageWeightMapping.get_transformer_mapping()
    pairs = {(target.to_pattern, tuple(target.from_pattern)) for target in mapping}

    assert ("x_embedder.proj.weight", ("x_embedder.proj.weight",)) in pairs
    assert ("adaLN_modulation.linear.weight", ("adaLN_modulation.1.weight",)) in pairs
    assert ("layers.{layer}.self_attention.to_out.0.weight", ("layers.{layer}.self_attention.to_out.0.weight",)) in pairs
    assert ("layers.{layer}.mlp.linear_fc2.weight", ("layers.{layer}.mlp.linear_fc2.weight",)) in pairs


@pytest.mark.fast
def test_ernie_weight_definition_downloads_full_source_snapshot():
    patterns = set(ErnieImageWeightDefinition.get_download_patterns())

    assert "text_encoder/**" in patterns
    assert "transformer/**" in patterns
    assert "vae/**" in patterns
    assert "pe/**" in patterns


@pytest.mark.fast
def test_ernie_weight_definition_loads_full_generation_components():
    components = {component.name for component in ErnieImageWeightDefinition.get_components()}

    assert components == {"vae", "transformer", "text_encoder"}


@pytest.mark.fast
def test_ernie_weight_definition_uses_custom_tokenizer():
    tokenizer_definition = ErnieImageWeightDefinition.get_tokenizers()[0]

    assert tokenizer_definition.encoder_class is ErnieImageTokenizer
    assert tokenizer_definition.padding == "longest"


@pytest.mark.fast
def test_ernie_scheduler_matches_diffusers_shifted_custom_sigmas():
    scheduler = ErnieImageScheduler(num_inference_steps=4)
    raw = np.array([1.0, 0.75, 0.5, 0.25], dtype=np.float32)
    shifted = 4.0 * raw / (1.0 + 3.0 * raw)

    np.testing.assert_allclose(np.array(scheduler.sigmas), np.concatenate([shifted, [0.0]]), rtol=1e-6)
    np.testing.assert_allclose(np.array(scheduler.timesteps), shifted * 1000.0, rtol=1e-6)
