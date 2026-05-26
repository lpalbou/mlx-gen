import mlx.core as mx
import numpy as np

from mflux.models.common.config import ModelConfig
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
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
