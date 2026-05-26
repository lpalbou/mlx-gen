import json

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.utils.generated_video import GeneratedVideo
from mflux.utils.video_util import VideoUtil


def _solid_frame(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (32, 24), color)


def test_generated_video_saves_mp4_and_metadata(tmp_path):
    output_path = tmp_path / "generated.mp4"
    video = GeneratedVideo(
        frames=[_solid_frame((255, 0, 0)), _solid_frame((0, 255, 0)), _solid_frame((0, 0, 255))],
        fps=12,
        model_config=ModelConfig.wan2_2_ti2v_5b(),
        seed=42,
        prompt="test video prompt",
        steps=2,
        guidance=5.0,
        precision=mx.bfloat16,
        quantization=0,
        generation_time=1.23,
        height=24,
        width=32,
    )

    video.save(path=output_path, export_json_metadata=True)

    assert output_path.exists()
    metadata = json.loads(output_path.with_suffix(".metadata.json").read_text())
    assert metadata["model"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert metadata["task"] == "text-to-video"
    assert metadata["frames"] == 3
    assert metadata["fps"] == 12
    assert metadata["duration_seconds"] == 0.25

    first_frame = VideoUtil.extract_frame(output_path)
    assert first_frame.size == (32, 24)


def test_generated_video_respects_no_replace(tmp_path):
    output_path = tmp_path / "generated.mp4"
    output_path.write_bytes(b"existing")
    video = GeneratedVideo(
        frames=[_solid_frame((255, 0, 0))],
        fps=8,
        model_config=ModelConfig.wan2_2_ti2v_5b(),
        seed=1,
        prompt="test",
        steps=1,
        guidance=5.0,
        precision=mx.bfloat16,
        quantization=0,
        generation_time=0.1,
        height=24,
        width=32,
    )

    video.save(path=output_path, overwrite=False)

    assert output_path.read_bytes() == b"existing"
    assert (tmp_path / "generated_1.mp4").exists()


def test_video_util_converts_decoded_latents_to_video(tmp_path):
    decoded = mx.array(np.zeros((1, 3, 2, 16, 16), dtype=np.float32))
    output_path = tmp_path / "latents.mp4"

    video = VideoUtil.to_video(
        decoded_latents=decoded,
        fps=4,
        model_config=ModelConfig.wan2_2_ti2v_5b(),
        seed=7,
        prompt="latent smoke",
        steps=1,
        guidance=5.0,
        quantization=0,
        generation_time=0.2,
    )
    video.save(output_path)

    assert video.num_frames == 2
    assert video.first_frame().size == (16, 16)
    assert output_path.exists()
