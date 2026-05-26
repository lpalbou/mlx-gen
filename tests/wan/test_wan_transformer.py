import mlx.core as mx
import numpy as np

from mflux.models.wan.model.wan_transformer import WanTransformer
from mflux.models.wan.model.wan_transformer.wan_embedding import WanRotaryPosEmbed


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
