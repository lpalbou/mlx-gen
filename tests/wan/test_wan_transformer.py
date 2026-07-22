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
