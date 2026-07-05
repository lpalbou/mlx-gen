import shutil
import subprocess

import pytest

from mflux.utils.video_util import AudioCopyResult, SourceVideoInfo, VideoUtil


@pytest.mark.fast
def test_copy_source_audio_to_video_skips_when_ffmpeg_is_missing(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    restored = tmp_path / "restored.mp4"
    source.touch()
    restored.touch()
    inspected: list[str] = []

    def fake_inspect(path):
        inspected.append(str(path))
        return SourceVideoInfo(
            fps=12.0,
            source_width=64,
            source_height=48,
            source_frame_count=12,
            source_duration_seconds=1.0,
            audio_present=True,
        )

    monkeypatch.setattr("mflux.utils.video_util.VideoUtil.inspect_video", staticmethod(fake_inspect))
    monkeypatch.setattr("mflux.utils.video_util.shutil.which", lambda name: None)

    result = VideoUtil.copy_source_audio_to_video(
        source_video_path=source,
        restored_video_path=restored,
        clip_start_seconds=0.0,
        clip_duration_seconds=1.0,
    )

    assert result == AudioCopyResult(
        audio_present=True,
        audio_copied=False,
        copy_mode=None,
        reason="ffmpeg_not_found",
    )
    assert inspected == [str(source)]


@pytest.mark.fast
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for audio copy-through tests")
def test_copy_source_audio_to_video_muxes_audio_onto_restored_clip(tmp_path):
    source = tmp_path / "source.mp4"
    restored = tmp_path / "restored.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            "0.5",
            "-t",
            "1.0",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(restored),
        ],
        check=True,
    )

    result = VideoUtil.copy_source_audio_to_video(
        source_video_path=source,
        restored_video_path=restored,
        clip_start_seconds=0.5,
        clip_duration_seconds=1.0,
    )

    inspected = VideoUtil.inspect_video(restored)
    assert result == AudioCopyResult(
        audio_present=True,
        audio_copied=True,
        copy_mode="ffmpeg_copy_video_aac_audio",
        reason=None,
    )
    assert inspected.audio_present is True
    assert inspected.source_frame_count == 12
    assert inspected.fps == pytest.approx(12.0, abs=1e-6)
    assert inspected.source_duration_seconds == pytest.approx(1.0, abs=0.05)
    assert VideoUtil._audio_duration_seconds(restored) == pytest.approx(1.0, abs=0.05)


@pytest.mark.fast
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for audio copy-through tests")
def test_copy_source_audio_keeps_all_video_frames_on_short_clips(tmp_path):
    # Regression: -shortest truncated stream-copied video packets at AAC audio EOF, dropping the
    # trailing frames of short clips (17f @ 16fps lost 2 frames) and failing the frame-count check.
    source = tmp_path / "source.mp4"
    restored = tmp_path / "restored.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "1.6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=16",
            "-frames:v",
            "17",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(restored),
        ],
        check=True,
    )

    result = VideoUtil.copy_source_audio_to_video(
        source_video_path=source,
        restored_video_path=restored,
        clip_start_seconds=0.0,
        clip_duration_seconds=17 / 16,
    )

    inspected = VideoUtil.inspect_video(restored)
    assert result.audio_copied is True, result.reason
    assert inspected.source_frame_count == 17
    assert inspected.audio_present is True
