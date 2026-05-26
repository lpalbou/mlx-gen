import mlx.core as mx
import numpy as np

from mflux.models.common.config import ModelConfig
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


def test_wan_vae_encode_normalized_first_frame_matches_i2v_condition_shape():
    vae = Wan2_2_VAE()
    image = mx.zeros((1, 3, 64, 64), dtype=mx.float32)

    condition = vae.encode_normalized(image)
    mx.eval(condition)

    assert condition.shape == (1, 48, 1, 4, 4)


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
