import json
import shutil
import subprocess

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.utils.generated_video import GeneratedVideo
from mflux.utils.video_health import VideoHealth, VideoHealthError
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
        guidance_2=3.0,
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
    assert metadata["guidance_2"] == 3.0
    assert metadata["video_health"]["frames"]["frame_count"] == 3
    assert metadata["video_health"]["file"]["frame_count"] == 3
    assert metadata["video_health"]["file"]["width"] == 32
    assert metadata["video_health"]["file"]["height"] == 24

    first_frame = VideoUtil.extract_frame(output_path)
    assert first_frame.size == (32, 24)
    first_frame_rgb = np.array(first_frame)
    assert first_frame_rgb[..., 0].mean() > first_frame_rgb[..., 1].mean()
    assert first_frame_rgb[..., 0].mean() > first_frame_rgb[..., 2].mean()
    assert _video_codec_name(output_path) in (None, "h264")


def test_generated_video_metadata_records_i2v_source_and_requested_dimensions():
    video = GeneratedVideo(
        frames=[_solid_frame((255, 0, 0))],
        fps=12,
        model_config=ModelConfig.wan2_2_i2v_a14b(),
        seed=42,
        prompt="test video prompt",
        steps=2,
        guidance=4.0,
        guidance_2=3.0,
        precision=mx.bfloat16,
        quantization=8,
        generation_time=1.23,
        height=336,
        width=448,
        task="image-to-video",
        image_path="source.png",
        source_width=320,
        source_height=240,
        requested_width=512,
        requested_height=288,
    )

    metadata = video._get_metadata()

    assert metadata["width"] == 448
    assert metadata["height"] == 336
    assert metadata["requested_width"] == 512
    assert metadata["requested_height"] == 288
    assert metadata["source_image_width"] == 320
    assert metadata["source_image_height"] == 240


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


def test_generated_video_rejects_all_black_output_before_save(tmp_path):
    output_path = tmp_path / "black.mp4"
    video = GeneratedVideo(
        frames=[_solid_frame((0, 0, 0)), _solid_frame((0, 0, 0))],
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

    with pytest.raises(VideoHealthError, match="effectively black"):
        video.save(path=output_path)

    assert not output_path.exists()


def test_video_health_rejects_black_file_postflight(tmp_path):
    output_path = tmp_path / "black-postflight.mp4"
    VideoUtil.save_video(
        frames=[_solid_frame((0, 0, 0)), _solid_frame((0, 0, 0))], path=output_path, fps=8, validate_health=False
    )

    with pytest.raises(VideoHealthError, match="effectively black"):
        VideoHealth.validate_file(
            output_path,
            expected_width=32,
            expected_height=24,
            expected_frames=2,
            expected_fps=8,
        )


def test_video_health_reports_valid_saved_video(tmp_path):
    output_path = tmp_path / "healthy.mp4"
    VideoUtil.save_video(
        frames=[_solid_frame((255, 0, 0)), _solid_frame((0, 255, 0)), _solid_frame((0, 0, 255))],
        path=output_path,
        fps=12,
    )

    report = VideoHealth.validate_file(
        output_path,
        expected_width=32,
        expected_height=24,
        expected_frames=3,
        expected_fps=12,
    )

    assert report.frame_count == 3
    assert report.width == 32
    assert report.height == 24
    assert report.luma_max > report.luma_min


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


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_video_util_rejects_non_finite_decoded_latents(invalid_value):
    decoded_np = np.zeros((1, 3, 2, 16, 16), dtype=np.float32)
    decoded_np[0, 0, 0, 0, 0] = invalid_value

    with pytest.raises(ValueError, match="Non-finite tensor values"):
        VideoUtil.to_video(
            decoded_latents=mx.array(decoded_np),
            fps=4,
            model_config=ModelConfig.wan2_2_ti2v_5b(),
            seed=7,
            prompt="latent smoke",
            steps=1,
            guidance=5.0,
            quantization=0,
            generation_time=0.2,
        )


def _video_codec_name(path) -> str | None:
    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None
