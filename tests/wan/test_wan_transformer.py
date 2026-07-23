import mlx.core as mx
import numpy as np
import pytest

from mflux.models.wan.model.wan_transformer import WanTransformer
from mflux.models.wan.model.wan_transformer.wan_embedding import WanRotaryPosEmbed
from mflux.models.wan.variants import Wan2_2_TI2V


def _tiny_transformer(num_layers: int = 1) -> WanTransformer:
    return WanTransformer(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=4,
        out_channels=4,
        text_dim=12,
        freq_dim=8,
        ffn_dim=16,
        num_layers=num_layers,
        cross_attn_norm=True,
        rope_max_seq_len=8,
    )


def test_wan_rotary_pos_embed_matches_patch_token_grid_shape():
    rope = WanRotaryPosEmbed(attention_head_dim=8, patch_size=(1, 2, 2), max_seq_len=8)
    hidden_states = mx.zeros((1, 4, 2, 4, 6), dtype=mx.float32)

    cos, sin = rope(hidden_states)

    assert cos.shape == (1, 12, 1, 8)
    assert sin.shape == (1, 12, 1, 8)
    np.testing.assert_allclose(np.array(cos[0, 0, 0]), np.ones((8,), dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(sin[0, 0, 0]), np.zeros((8,), dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_wan_rotary_pos_embed_shape_cache_is_bitwise_identical_to_fresh_compute():
    # F7 (0090): shapes are constant within a run, so the embeds cache per token grid.
    # A cache hit must be bitwise identical to a fresh instance's computation, and a
    # different shape must never serve stale tensors.
    rope = WanRotaryPosEmbed(attention_head_dim=8, patch_size=(1, 2, 2), max_seq_len=8)
    fresh = WanRotaryPosEmbed(attention_head_dim=8, patch_size=(1, 2, 2), max_seq_len=8)
    hidden_states = mx.zeros((1, 4, 2, 4, 6), dtype=mx.float32)

    first_cos, first_sin = rope(hidden_states)
    cached_cos, cached_sin = rope(hidden_states)
    fresh_cos, fresh_sin = fresh(hidden_states)
    mx.eval(first_cos, first_sin, cached_cos, cached_sin, fresh_cos, fresh_sin)

    assert (2, 2, 3) in rope._freqs_cache
    np.testing.assert_array_equal(np.array(cached_cos), np.array(first_cos))
    np.testing.assert_array_equal(np.array(cached_sin), np.array(first_sin))
    np.testing.assert_array_equal(np.array(cached_cos), np.array(fresh_cos))
    np.testing.assert_array_equal(np.array(cached_sin), np.array(fresh_sin))

    # Different grid (3, 2, 2): same 12-token count as (2, 2, 3) but a distinct
    # cache key, so it must be computed fresh, not served from the cache.
    other_cos, other_sin = rope(mx.zeros((1, 4, 3, 4, 4), dtype=mx.float32))
    assert other_cos.shape == (1, 12, 1, 8)
    assert other_sin.shape == (1, 12, 1, 8)
    assert not np.array_equal(np.array(other_cos), np.array(first_cos))
    assert len(rope._freqs_cache) == 2


def test_wan_rotary_pos_embed_cache_stays_out_of_parameters_and_is_bounded():
    from mlx.utils import tree_flatten

    rope = WanRotaryPosEmbed(attention_head_dim=8, patch_size=(1, 2, 2), max_seq_len=8)
    for frames in (1, 2, 3, 4, 5, 6):
        rope(mx.zeros((1, 4, frames, 4, 4), dtype=mx.float32))

    # The cache never leaks into the weight/quantization traversal and stays bounded
    # for long-lived hosts that chain several resolutions in one process. The bound
    # is 2 grids (cycle-3 review): a grid pair is ~114 MB f32 at A14B 121f@1280x720
    # per expert, and a rebuild costs only ~33 ms, so retention stays small.
    parameter_keys = [key for key, _ in tree_flatten(rope.parameters())]
    assert all("_freqs_cache" not in key for key in parameter_keys)
    assert len(rope._freqs_cache) == 2
    # FIFO by insertion order: the two most recently computed grids remain.
    assert set(rope._freqs_cache.keys()) == {(5, 2, 2), (6, 2, 2)}


def test_wan_transformer_tiny_forward_preserves_video_shape():
    model = _tiny_transformer()
    hidden_states = mx.zeros((1, 4, 2, 4, 4), dtype=mx.float32)
    timestep = mx.array([999.0], dtype=mx.float32)
    encoder_hidden_states = mx.zeros((1, 5, 12), dtype=mx.float32)

    output = model(hidden_states, timestep, encoder_hidden_states)
    mx.eval(output)

    assert output.shape == hidden_states.shape


def test_wan_transformer_tiny_forward_supports_expanded_timesteps():
    model = _tiny_transformer(num_layers=0)
    hidden_states = mx.zeros((1, 4, 2, 4, 4), dtype=mx.float32)
    timestep = mx.full((1, 8), 999.0, dtype=mx.float32)
    encoder_hidden_states = mx.zeros((1, 5, 12), dtype=mx.float32)

    output = model(hidden_states, timestep, encoder_hidden_states)
    mx.eval(output)

    assert output.shape == hidden_states.shape


@pytest.mark.parametrize(
    "timestep_factory",
    [
        # Scalar convention (A14B experts): one timestep per batch item.
        lambda: mx.array([999.0], dtype=mx.float32),
        # Expanded convention (TI2V-5B): per-patch-token timesteps.
        lambda: mx.full((1, 8), 999.0, dtype=mx.float32),
    ],
    ids=["scalar_a14b", "expanded_5b"],
)
def test_wan_transformer_compiled_matches_eager_within_kernel_tolerance(timestep_factory):
    # 0090 d12 parity pin: compiled output is NOT bitwise-identical to eager
    # (kernel fusion reorders float ops, measured ~5e-4 on real weights), so
    # --compile-transformer stays opt-in. This bounds the divergence.
    mx.random.seed(11)
    model = _tiny_transformer(num_layers=2)
    hidden_states = mx.random.normal((1, 4, 2, 4, 4), dtype=mx.float32)
    timestep = timestep_factory()
    encoder_hidden_states = mx.random.normal((1, 5, 12), dtype=mx.float32)

    eager = model(hidden_states, timestep, encoder_hidden_states)
    mx.eval(eager)
    compiled_fn = Wan2_2_TI2V._compile_denoiser(model)
    compiled = compiled_fn(hidden_states, timestep, encoder_hidden_states)
    mx.eval(compiled)

    assert compiled.shape == eager.shape
    max_delta = float(mx.max(mx.abs(compiled - eager)).item())
    assert max_delta < 5e-3, f"compiled-vs-eager divergence too large: {max_delta}"


def test_wan_transformer_can_clear_cache_after_each_block(monkeypatch):
    model = _tiny_transformer(num_layers=2)
    hidden_states = mx.zeros((1, 4, 2, 4, 4), dtype=mx.float32)
    timestep = mx.array([999.0], dtype=mx.float32)
    encoder_hidden_states = mx.zeros((1, 5, 12), dtype=mx.float32)
    clear_calls = []

    monkeypatch.setattr(
        "mflux.models.wan.model.wan_transformer.wan_transformer.mx.clear_cache",
        lambda: clear_calls.append(True),
    )

    output = model(
        hidden_states,
        timestep,
        encoder_hidden_states,
        clear_cache_each_block=True,
    )
    mx.eval(output)

    assert output.shape == hidden_states.shape
    assert clear_calls == [True, True]
