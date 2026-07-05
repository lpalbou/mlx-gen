import json
import shutil
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRAApplicationResult, LoRAFileReport
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
        flow_shift=5.0,
        solver="euler",
        precision=mx.bfloat16,
        quantization=0,
        generation_time=1.23,
        height=24,
        width=32,
    )

    video.save(path=output_path, export_json_metadata=True)

    assert output_path.exists()
    metadata = json.loads(output_path.with_suffix(".metadata.json").read_text())
    assert metadata["metadata_schema_version"] == 1
    assert metadata["model"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert metadata["task"] == "text-to-video"
    assert metadata["frames"] == 3
    assert metadata["fps"] == 12
    assert metadata["duration_seconds"] == 0.25
    assert metadata["guidance_2"] == 3.0
    assert metadata["flow_shift"] == 5.0
    assert metadata["solver"] == "euler"
    assert metadata["video_path"] is None
    assert metadata["video_health"]["frames"]["frame_count"] == 3
    assert metadata["video_health"]["file"]["frame_count"] == 3
    assert metadata["video_health"]["file"]["width"] == 32
    assert metadata["video_health"]["file"]["height"] == 24

    first_frame = VideoUtil.extract_frame(output_path)
    assert first_frame.size == (32, 24)
    first_frame_rgb = VideoUtil._pil_rgb_to_array(first_frame)
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


def test_generated_video_metadata_records_video_source_path():
    video = GeneratedVideo(
        frames=[_solid_frame((255, 0, 0))],
        fps=29.97,
        model_config=ModelConfig.seedvr2_3b(),
        seed=42,
        prompt="",
        steps=1,
        guidance=1.0,
        precision=mx.bfloat16,
        quantization=8,
        generation_time=0.5,
        height=240,
        width=320,
        task="video-to-video",
        video_path="source.mp4",
        extra_metadata={"audio_copied": False},
    )

    metadata = video._get_metadata()

    assert metadata["video_path"] == "source.mp4"
    assert metadata["fps"] == 29.97
    assert metadata["audio_copied"] is False


def test_generated_video_records_lora_application_metadata():
    report = LoRAFileReport(
        requested_path="wouterverweirder/wan_2_2_5B_woven_fabric_02-lora:wan_2_2_5B_woven_fabric_02.safetensors",
        resolved_path="/tmp/wan.safetensors",
        scale=0.9,
        role="transformer",
        total_key_count=600,
        matched_key_count=600,
        unmatched_key_count=0,
        applied_target_count=300,
    )
    result = LoRAApplicationResult(
        resolved_paths=["/tmp/wan.safetensors"],
        resolved_scales=[0.9],
        reports=(report,),
    )
    video = GeneratedVideo(
        frames=[_solid_frame((255, 0, 0))],
        fps=12,
        model_config=ModelConfig.wan2_2_ti2v_5b(),
        seed=42,
        prompt="test video prompt",
        steps=2,
        guidance=5.0,
        guidance_2=None,
        flow_shift=5.0,
        solver="unipc",
        precision=mx.bfloat16,
        quantization=8,
        generation_time=1.23,
        height=24,
        width=32,
        lora_paths=["/tmp/wan.safetensors"],
        lora_scales=[0.9],
        extra_metadata={
            **result.extra_metadata(),
            "lora_target_roles": ["transformer"],
        },
    )

    metadata = video._get_metadata()

    assert metadata["lora_paths"] == ["/tmp/wan.safetensors"]
    assert metadata["lora_scales"] == [0.9]
    assert metadata["solver"] == "unipc"
    assert metadata["lora_target_roles"] == ["transformer"]
    assert metadata["lora_applied_file_count"] == 1
    assert metadata["lora_applied_target_count"] == 300


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


def test_video_util_lazy_decoded_latents_save_uses_batches(tmp_path, monkeypatch):
    decoded = mx.array(np.zeros((1, 3, 3, 16, 16), dtype=np.float32))
    output_path = tmp_path / "latents-batched.mp4"
    observed = {}

    def fake_save_video_batches(
        frame_batches,
        path,
        fps,
        metadata=None,
        export_json_metadata=False,
        overwrite=True,
        validate_health=True,
        source_audio_copy=None,
    ):
        batches = list(frame_batches)
        observed["batch_lengths"] = [len(batch) for batch in batches]
        observed["metadata_frames"] = metadata["frames"]
        return path

    monkeypatch.setattr(VideoUtil, "save_video_batches", fake_save_video_batches)

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
        materialize_frames=False,
    )

    assert video.num_frames == 3
    assert video._frames is None

    video.save(output_path)

    assert observed == {"batch_lengths": [3], "metadata_frames": 3}
    assert video._frames is None


def test_video_util_reads_bounded_clip(tmp_path):
    output_path = tmp_path / "clip.mp4"
    frames = [
        _solid_frame((255, 0, 0)),
        _solid_frame((0, 255, 0)),
        _solid_frame((0, 0, 255)),
        _solid_frame((255, 255, 0)),
    ]
    VideoUtil.save_video(frames=frames, path=output_path, fps=4)

    clip = VideoUtil.read_video_clip(output_path, start_seconds=0.25, max_frames=2)

    assert clip.source_width == 32
    assert clip.source_height == 24
    assert clip.clip_start_frame == 1
    assert clip.clip_frame_count == 2
    assert len(clip.frames) == 2
    assert abs(clip.fps - 4.0) < 0.1


def _rainbow_clip_path(tmp_path, *, frame_count: int = 8, fps: int = 8):
    # Distinct hues let assertions identify WHICH source frames were sampled after lossy encode.
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]
    output_path = tmp_path / "rainbow.mp4"
    frames = [_solid_frame(palette[index % len(palette)]) for index in range(frame_count)]
    VideoUtil.save_video(frames=frames, path=output_path, fps=fps)
    return output_path, palette


def _center_color(frame):
    width, height = frame.size
    return frame.getpixel((width // 2, height // 2))


def _assert_color_close(actual, expected, tolerance=60):
    assert all(abs(a - e) <= tolerance for a, e in zip(actual, expected)), (actual, expected)


def test_video_util_resamples_clip_to_target_fps(tmp_path):
    clip_path, palette = _rainbow_clip_path(tmp_path, frame_count=8, fps=8)

    clip = VideoUtil.read_video_clip(clip_path, max_frames=4, target_fps=4.0)

    assert clip.sampled_fps == 4.0
    assert abs(clip.fps - 8.0) < 0.1
    assert clip.clip_frame_count == 4
    # 4 fps sampling of an 8 fps source picks every second source frame.
    for output_index, source_index in enumerate([0, 2, 4, 6]):
        _assert_color_close(_center_color(clip.frames[output_index]), palette[source_index])


def test_video_util_resample_skips_matching_fps(tmp_path):
    clip_path, palette = _rainbow_clip_path(tmp_path, frame_count=4, fps=4)

    clip = VideoUtil.read_video_clip(clip_path, max_frames=4, target_fps=4.0)

    assert clip.sampled_fps is None
    assert clip.clip_frame_count == 4
    for index in range(4):
        _assert_color_close(_center_color(clip.frames[index]), palette[index])


def test_video_util_resample_pyav_fallback_matches_ffmpeg(tmp_path, monkeypatch):
    clip_path, palette = _rainbow_clip_path(tmp_path, frame_count=8, fps=8)

    monkeypatch.setattr("mflux.utils.video_util.shutil.which", lambda name: None)
    clip = VideoUtil.read_video_clip(clip_path, max_frames=4, target_fps=4.0)

    assert clip.sampled_fps == 4.0
    assert clip.clip_frame_count == 4
    for output_index, source_index in enumerate([0, 2, 4, 6]):
        _assert_color_close(_center_color(clip.frames[output_index]), palette[source_index])


def test_video_util_resample_short_source_yields_fewer_frames(tmp_path):
    clip_path, _ = _rainbow_clip_path(tmp_path, frame_count=4, fps=8)  # 0.5s of source

    clip = VideoUtil.read_video_clip(clip_path, max_frames=4, target_fps=4.0)

    assert clip.clip_frame_count < 4


def test_save_video_copies_source_audio_and_records_fields(tmp_path, monkeypatch):
    from mflux.utils.video_util import AudioCopyResult, SourceAudioCopySpec

    observed = {}

    def fake_copy(**kwargs):
        observed.update(kwargs)
        return AudioCopyResult(audio_present=True, audio_copied=True, copy_mode="stream_copy", reason=None)

    monkeypatch.setattr(VideoUtil, "copy_source_audio_to_video", staticmethod(fake_copy))
    output_path = tmp_path / "with_audio.mp4"
    saved = VideoUtil.save_video(
        frames=[_solid_frame((255, 0, 0)) for _ in range(4)],
        path=output_path,
        fps=4,
        metadata={"frame_count": 4},
        export_json_metadata=True,
        source_audio_copy=SourceAudioCopySpec(
            source_video_path=tmp_path / "source.mp4",
            clip_start_seconds=0.0,
            clip_duration_seconds=1.0,
        ),
    )

    assert observed["clip_duration_seconds"] == 1.0
    assert observed["restored_video_path"] == saved
    sidecar = json.loads((tmp_path / "with_audio.metadata.json").read_text())
    assert sidecar["audio_copied"] is True
    assert sidecar["audio_copy_mode"] == "stream_copy"
    assert "video_health" in sidecar


def test_save_video_batches_copies_source_audio_and_records_fields(tmp_path, monkeypatch):
    from mflux.utils.video_util import AudioCopyResult, SourceAudioCopySpec

    def fake_copy(**kwargs):
        return AudioCopyResult(audio_present=True, audio_copied=True, copy_mode="stream_copy", reason=None)

    monkeypatch.setattr(VideoUtil, "copy_source_audio_to_video", staticmethod(fake_copy))
    output_path = tmp_path / "batched_audio.mp4"
    VideoUtil.save_video_batches(
        frame_batches=[[_solid_frame((0, 255, 0)) for _ in range(4)]],
        path=output_path,
        fps=4,
        metadata={"frame_count": 4},
        export_json_metadata=True,
        source_audio_copy=SourceAudioCopySpec(
            source_video_path=tmp_path / "source.mp4",
            clip_start_seconds=0.0,
            clip_duration_seconds=1.0,
        ),
    )

    sidecar = json.loads((tmp_path / "batched_audio.metadata.json").read_text())
    assert sidecar["audio_copied"] is True
    assert "video_health" in sidecar


def test_save_video_audio_copy_is_best_effort_on_failure(tmp_path, monkeypatch, capsys):
    from mflux.utils.video_util import SourceAudioCopySpec

    def raising_copy(**kwargs):
        raise RuntimeError("source vanished")

    monkeypatch.setattr(VideoUtil, "copy_source_audio_to_video", staticmethod(raising_copy))
    output_path = tmp_path / "silent.mp4"
    saved = VideoUtil.save_video(
        frames=[_solid_frame((0, 0, 255)) for _ in range(4)],
        path=output_path,
        fps=4,
        metadata={"frame_count": 4},
        export_json_metadata=True,
        source_audio_copy=SourceAudioCopySpec(
            source_video_path=tmp_path / "gone.mp4",
            clip_start_seconds=0.0,
            clip_duration_seconds=1.0,
        ),
    )

    assert saved.exists()
    output = capsys.readouterr().out
    assert "Source audio could not be preserved" in output
    assert "ffmpeg" in output
    sidecar = json.loads((tmp_path / "silent.metadata.json").read_text())
    assert sidecar["audio_copied"] is False
    assert "RuntimeError" in sidecar["audio_copy_reason"]


def test_generated_video_save_requests_audio_copy_for_v2v(tmp_path, monkeypatch):
    observed = {}

    def fake_save_video(**kwargs):
        observed.update(kwargs)
        return tmp_path / "out.mp4"

    monkeypatch.setattr(VideoUtil, "save_video", staticmethod(fake_save_video))
    video = GeneratedVideo(
        frames=[_solid_frame((255, 0, 0)) for _ in range(16)],
        model_config=ModelConfig.wan2_2_ti2v_5b(),
        seed=1,
        prompt="v2v audio",
        steps=1,
        guidance=5.0,
        precision="bfloat16",
        quantization=0,
        generation_time=1.0,
        height=24,
        width=32,
        fps=16,
        task="video-to-video",
        video_path=tmp_path / "source.mp4",
    )

    video.save(tmp_path / "out.mp4")

    spec = observed["source_audio_copy"]
    assert spec is not None
    assert Path(spec.source_video_path) == tmp_path / "source.mp4"
    assert spec.clip_start_seconds == 0.0
    assert spec.clip_duration_seconds == pytest.approx(1.0)

    # Non-V2V saves must not attempt audio copy.
    observed.clear()
    t2v = GeneratedVideo(
        frames=[_solid_frame((255, 0, 0)) for _ in range(16)],
        model_config=ModelConfig.wan2_2_ti2v_5b(),
        seed=1,
        prompt="t2v silent",
        steps=1,
        guidance=5.0,
        precision="bfloat16",
        quantization=0,
        generation_time=1.0,
        height=24,
        width=32,
        fps=16,
        task="text-to-video",
    )
    t2v.save(tmp_path / "out2.mp4")
    assert observed["source_audio_copy"] is None


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
