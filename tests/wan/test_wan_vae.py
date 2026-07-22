import mlx.core as mx
import numpy as np

from mflux.models.common.config import ModelConfig
from mflux.models.fibo.model.fibo_vae.common.wan_2_2_rms_norm import Wan2_2_RMSNorm
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.weights.wan_weight_mapping import WanWeightMapping
from mflux.utils.video_util import VideoUtil


def test_wan_vae_patchify_unpatchify_roundtrip_5d():
    x = mx.arange(1 * 3 * 2 * 4 * 6, dtype=mx.float32).reshape(1, 3, 2, 4, 6)

    patched = Wan2_2_VAE.patchify(x, patch_size=2)
    unpatched = Wan2_2_VAE.unpatchify(patched, patch_size=2)

    assert patched.shape == (1, 12, 2, 2, 3)
    np.testing.assert_array_equal(np.array(unpatched), np.array(x))


def test_wan_vae_decode_normalized_latents_expands_temporal_dimension():
    vae = Wan2_2_VAE()
    decoded = vae.decode_normalized_latents(mx.zeros((1, 48, 2, 4, 4), dtype=mx.float32))
    mx.eval(decoded)

    assert decoded.shape == (1, 3, 5, 64, 64)
    assert float(mx.min(decoded).item()) >= -1.0
    assert float(mx.max(decoded).item()) <= 1.0


def test_wan_vae_decode_low_ram_clears_cache_after_each_latent_slice(monkeypatch):
    vae = Wan2_2_VAE()
    clear_calls = []

    monkeypatch.setattr("mflux.models.wan.model.wan_vae.wan_2_2_vae.gc.collect", lambda: None)
    monkeypatch.setattr("mflux.models.wan.model.wan_vae.wan_2_2_vae.mx.synchronize", lambda: None)
    monkeypatch.setattr(
        "mflux.models.wan.model.wan_vae.wan_2_2_vae.mx.clear_cache",
        lambda: clear_calls.append(True),
    )

    decoded = vae.decode_normalized_latents(
        mx.zeros((1, 48, 2, 4, 4), dtype=mx.float32),
        clear_cache_each_slice=True,
    )
    mx.eval(decoded)

    assert decoded.shape == (1, 3, 5, 64, 64)
    assert clear_calls == [True, True]


def test_wan_vae_streamed_slice_decode_is_bitwise_identical_to_full_decode():
    # 0089 e3 parity pin: Wan2_2_TI2V now decodes exclusively through
    # iter_decode_normalized_latent_slices. decode_normalized_latents is the
    # still-shipped non-streamed path (WanVace uses it), so comparing against it
    # is an honest same-code-generation comparison, not a test-only copy.
    # Both paths share post_quant_conv + the causal per-frame decoder with the
    # same feat_cache handoff; the streamed path just unpatchifies/clips per
    # slice, which is frame-independent - hence BITWISE equality is required.
    mx.random.seed(7)
    vae = Wan2_2_VAE()
    latents = mx.random.normal((1, 48, 3, 4, 4), dtype=mx.float32)

    full = vae.decode_normalized_latents(latents)
    mx.eval(full)
    streamed = mx.concatenate(list(vae.iter_decode_normalized_latent_slices(latents)), axis=2)
    mx.eval(streamed)

    assert streamed.shape == full.shape == (1, 3, 9, 64, 64)
    np.testing.assert_array_equal(np.array(streamed), np.array(full))


def test_wan_vae_encode_normalized_first_frame_matches_i2v_condition_shape():
    vae = Wan2_2_VAE()
    image = mx.zeros((1, 3, 64, 64), dtype=mx.float32)

    condition = vae.encode_normalized(image)
    mx.eval(condition)

    assert condition.shape == (1, 48, 1, 4, 4)


def test_wan_vae_encode_normalized_video_handles_17_frame_clip():
    vae = Wan2_2_VAE()
    video = mx.zeros((1, 3, 17, 64, 64), dtype=mx.float32)

    latents = vae.encode_normalized(video)
    mx.eval(latents)

    assert latents.shape == (1, 48, 5, 4, 4)


def test_wan_vae_rms_norm_normalizes_bf16_inputs_in_fp32():
    norm = Wan2_2_RMSNorm(dim=4, images=False)
    x = mx.array(
        np.linspace(-12.0, 12.0, 1 * 4 * 2 * 2 * 2, dtype=np.float32).reshape(1, 4, 2, 2, 2)
    ).astype(mx.bfloat16)

    output = norm(x)
    mx.eval(output)

    x_float = x.astype(mx.float32)
    expected = x_float / mx.maximum(
        mx.sqrt(mx.sum(x_float * x_float, axis=1, keepdims=True)),
        mx.array(norm.eps, dtype=mx.float32),
    )
    expected = expected * norm.scale * norm.weight.reshape(1, -1, 1, 1, 1).astype(mx.float32)
    np.testing.assert_allclose(
        np.array(output.astype(mx.float32)),
        np.array(expected.astype(mx.bfloat16).astype(mx.float32)),
    )


def test_wan_vae_mapping_includes_encoder_shortcut_weights():
    targets = {target.to_pattern for target in WanWeightMapping.get_vae_mapping()}

    assert "encoder.down_blocks.1.resnets.0.conv_shortcut.conv3d.weight" in targets
    assert "encoder.down_blocks.2.resnets.0.conv_shortcut.conv3d.weight" in targets


def test_wan_vae_decode_can_save_mp4(tmp_path):
    vae = Wan2_2_VAE()
    decoded = vae.decode_normalized_latents(mx.zeros((1, 48, 2, 4, 4), dtype=mx.float32))
    video = VideoUtil.to_video(
        decoded_latents=decoded,
        fps=8,
        model_config=ModelConfig.wan2_2_ti2v_5b(),
        seed=11,
        prompt="wan vae latent smoke",
        steps=1,
        guidance=5.0,
        quantization=0,
        generation_time=0.1,
    )
    output_path = tmp_path / "wan_vae_smoke.mp4"

    video.save(output_path)

    assert video.num_frames == 5
    assert output_path.exists()
    assert VideoUtil.extract_frame(output_path).size == (64, 64)
