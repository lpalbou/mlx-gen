from __future__ import annotations

import gc
import json
import logging
import shutil
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import PIL.Image

log = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class VideoHealthReport:
    source: str
    frame_count: int
    width: int
    height: int
    fps: float | None
    luma_min: float
    luma_max: float
    luma_mean: float
    mean_temporal_delta: float | None

    @property
    def is_all_black(self) -> bool:
        return self.luma_max <= VideoHealth.BLACK_LUMA_THRESHOLD

    @property
    def is_all_white(self) -> bool:
        return self.luma_min >= VideoHealth.WHITE_LUMA_THRESHOLD

    @property
    def is_temporally_static(self) -> bool:
        return self.mean_temporal_delta is not None and self.mean_temporal_delta <= VideoHealth.STATIC_DELTA_THRESHOLD

    def to_metadata(self) -> dict:
        return {
            "source": self.source,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "luma_min": self.luma_min,
            "luma_max": self.luma_max,
            "luma_mean": self.luma_mean,
            "mean_temporal_delta": self.mean_temporal_delta,
        }


class VideoHealthError(ValueError):
    pass


class VideoHealth:
    BLACK_LUMA_THRESHOLD = 2.0
    WHITE_LUMA_THRESHOLD = 253.0
    STATIC_DELTA_THRESHOLD = 0.5

    @staticmethod
    def validate_frames(
        frames: list[PIL.Image.Image],
        *,
        fps: int,
        expected_width: int | None = None,
        expected_height: int | None = None,
        strict_visual: bool = True,
    ) -> VideoHealthReport:
        if not frames:
            raise VideoHealthError("Video health check failed: no frames were provided.")
        if fps <= 0:
            raise VideoHealthError("Video health check failed: fps must be greater than zero.")

        width, height = frames[0].size
        if expected_width is not None and width != expected_width:
            raise VideoHealthError(f"Video health check failed: expected width {expected_width}, got {width}.")
        if expected_height is not None and height != expected_height:
            raise VideoHealthError(f"Video health check failed: expected height {expected_height}, got {height}.")

        stats = _LumaStats()
        for index, frame in enumerate(frames):
            if frame.size != (width, height):
                raise VideoHealthError(
                    f"Video health check failed: frame {index + 1}/{len(frames)} has size {frame.size}, "
                    f"expected {(width, height)}."
                )
            stats.add(VideoHealth._luma_array(frame))

        report = stats.report(
            source="frames",
            width=width,
            height=height,
            fps=float(fps),
        )
        VideoHealth._validate_visual_report(report, strict_visual=strict_visual)
        return report

    @staticmethod
    def validate_file(
        path: str | Path,
        *,
        expected_width: int | None = None,
        expected_height: int | None = None,
        expected_frames: int | None = None,
        expected_fps: int | float | None = None,
        strict_visual: bool = True,
    ) -> VideoHealthReport:
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            return VideoHealth._validate_file(
                path,
                expected_width=expected_width,
                expected_height=expected_height,
                expected_frames=expected_frames,
                expected_fps=expected_fps,
                strict_visual=strict_visual,
            )
        finally:
            if gc_was_enabled:
                gc.enable()

    @staticmethod
    def _validate_file(
        path: str | Path,
        *,
        expected_width: int | None = None,
        expected_height: int | None = None,
        expected_frames: int | None = None,
        expected_fps: int | float | None = None,
        strict_visual: bool = True,
    ) -> VideoHealthReport:
        file_path = Path(path)
        if not file_path.exists():
            raise VideoHealthError(f"Video health check failed: {file_path} does not exist.")
        if file_path.stat().st_size <= 0:
            raise VideoHealthError(f"Video health check failed: {file_path} is empty.")

        metadata = VideoHealth._inspect_file(file_path)
        if metadata is None:
            raise VideoHealthError(f"Video health check failed: could not inspect {file_path}.")
        width, height, fps, stream_frame_count = metadata

        stats = VideoHealth._decode_file_luma_stats(file_path)
        if stats is None:
            raise VideoHealthError(
                f"Video health check failed: could not decode frames safely in a child process for {file_path}."
            )
        if stats.frame_count == 0:
            raise VideoHealthError(f"Video health check failed: {file_path} contains no decodable frames.")
        if expected_width is not None and width != expected_width:
            raise VideoHealthError(f"Video health check failed: expected width {expected_width}, got {width}.")
        if expected_height is not None and height != expected_height:
            raise VideoHealthError(f"Video health check failed: expected height {expected_height}, got {height}.")
        if expected_frames is not None and stats.frame_count != expected_frames:
            raise VideoHealthError(
                f"Video health check failed: expected {expected_frames} frames, decoded {stats.frame_count}."
            )
        if expected_fps is not None and fps is not None and abs(fps - float(expected_fps)) > 0.1:
            raise VideoHealthError(f"Video health check failed: expected fps {expected_fps:g}, got {fps:g}.")

        report = stats.report(
            source=str(file_path),
            width=width,
            height=height,
            fps=fps,
        )
        VideoHealth._validate_visual_report(report, strict_visual=strict_visual)
        return report

    @staticmethod
    def _validate_visual_report(report: VideoHealthReport, *, strict_visual: bool) -> None:
        if strict_visual and report.is_all_black:
            raise VideoHealthError(
                "Video health check failed: every decoded frame is effectively black "
                f"(luma_range={report.luma_min:g}..{report.luma_max:g})."
            )
        if strict_visual and report.is_all_white:
            raise VideoHealthError(
                "Video health check failed: every decoded frame is effectively white "
                f"(luma_range={report.luma_min:g}..{report.luma_max:g})."
            )
        if report.is_temporally_static and report.frame_count > 1:
            log.warning(
                "Generated video is nearly static: source=%s, frames=%s, luma_range=%g..%g, mean_delta=%g",
                report.source,
                report.frame_count,
                report.luma_min,
                report.luma_max,
                report.mean_temporal_delta or 0.0,
            )

    @staticmethod
    def _luma_array(frame: PIL.Image.Image) -> np.ndarray:
        # Buffer-protocol conversion; getdata() would build ~1M Python tuples per frame
        # (measured ~350x slower) and is removed in Pillow 14.
        rgb_image = frame.convert("RGB")
        rgb = np.asarray(rgb_image, dtype=np.float32)
        return VideoHealth._luma_rgb_array(rgb)

    @staticmethod
    def _luma_rgb_array(rgb: np.ndarray) -> np.ndarray:
        if rgb.dtype != np.float32:
            rgb = np.asarray(rgb, dtype=np.float32)
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    @staticmethod
    def _decode_file_luma_stats(file_path: Path) -> _LumaStats | None:
        script = r"""
import json
import sys

import av
import numpy as np

path = sys.argv[1]
frame_count = 0
luma_min = float("inf")
luma_max = float("-inf")
luma_mean_sum = 0.0
temporal_delta_sum = 0.0
temporal_delta_count = 0
previous_luma = None
with av.open(path) as container:
    video_stream = container.streams.video[0]
    try:
        video_stream.thread_type = "NONE"
    except Exception:
        pass
    for frame in container.decode(video_stream):
        rgb = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.float32)
        luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        frame_count += 1
        luma_min = min(luma_min, float(np.min(luma)))
        luma_max = max(luma_max, float(np.max(luma)))
        luma_mean_sum += float(np.mean(luma))
        if previous_luma is not None:
            temporal_delta_sum += float(np.mean(np.abs(luma - previous_luma)))
            temporal_delta_count += 1
        previous_luma = luma
print(json.dumps({
    "frame_count": frame_count,
    "luma_min": luma_min,
    "luma_max": luma_max,
    "luma_mean_sum": luma_mean_sum,
    "temporal_delta_sum": temporal_delta_sum,
    "temporal_delta_count": temporal_delta_count,
}))
"""
        try:
            result = VideoHealth._run_subprocess_with_gc_suspended(
                [sys.executable, "-c", script, str(file_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("Video health child decoder failed for %s: %s", file_path, exc)
            return None
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            if stderr:
                log.warning("Video health child decoder exited with %s for %s: %s", result.returncode, file_path, stderr)
            return None
        try:
            payload = json.loads(result.stdout)
            return _LumaStats(
                frame_count=int(payload["frame_count"]),
                luma_min=float(payload["luma_min"]),
                luma_max=float(payload["luma_max"]),
                luma_mean_sum=float(payload["luma_mean_sum"]),
                temporal_delta_sum=float(payload["temporal_delta_sum"]),
                temporal_delta_count=int(payload["temporal_delta_count"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log.warning("Video health child decoder returned invalid stats for %s: %s", file_path, exc)
            return None

    @staticmethod
    def _inspect_file(file_path: Path) -> tuple[int, int, float | None, int | None] | None:
        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path is not None:
            command = [
                ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(file_path),
            ]
            result = VideoHealth._run_subprocess_with_gc_suspended(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                with suppress(json.JSONDecodeError, KeyError, TypeError, ValueError, StopIteration):
                    stream = json.loads(result.stdout)["streams"][0]
                    fps = VideoHealth._rate_to_float(VideoHealth._fraction_from_string(stream.get("avg_frame_rate")))
                    if fps is None or fps <= 0:
                        fps = VideoHealth._rate_to_float(VideoHealth._fraction_from_string(stream.get("r_frame_rate")))
                    frame_count = VideoHealth._optional_int(stream.get("nb_frames"))
                    duration = VideoHealth._optional_float(stream.get("duration"))
                    if frame_count is None and duration is not None and fps is not None:
                        frame_count = max(1, int(round(duration * fps)))
                    return int(stream["width"]), int(stream["height"]), fps, frame_count

        return VideoHealth._inspect_file_pyav(file_path)

    @staticmethod
    def _inspect_file_pyav(file_path: Path) -> tuple[int, int, float | None, int | None] | None:
        try:
            import av

            with av.open(str(file_path)) as container:
                if len(container.streams.video) == 0:
                    return None
                video_stream = container.streams.video[0]
                fps = VideoHealth._rate_to_float(video_stream.average_rate or video_stream.base_rate)
                stream_frame_count = int(video_stream.frames) if video_stream.frames else None
                return int(video_stream.width), int(video_stream.height), fps, stream_frame_count
        except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def _fraction_from_string(value: str | None) -> Fraction | None:
        if not value or value == "0/0":
            return None
        with suppress(ZeroDivisionError, ValueError):
            return Fraction(value)
        return None

    @staticmethod
    def _optional_int(value) -> int | None:
        if value in (None, "N/A"):
            return None
        with suppress(TypeError, ValueError):
            return int(value)
        return None

    @staticmethod
    def _optional_float(value) -> float | None:
        if value in (None, "N/A"):
            return None
        with suppress(TypeError, ValueError):
            return float(value)
        return None

    @staticmethod
    def _rate_to_float(rate: Fraction | int | float | None) -> float | None:
        if rate is None:
            return None
        return float(rate)

    @staticmethod
    def _run_subprocess_with_gc_suspended(*args, **kwargs) -> subprocess.CompletedProcess:
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            return subprocess.run(*args, **kwargs)
        finally:
            if gc_was_enabled:
                gc.enable()


@dataclass
class _LumaStats:
    frame_count: int = 0
    luma_min: float = float("inf")
    luma_max: float = float("-inf")
    luma_mean_sum: float = 0.0
    temporal_delta_sum: float = 0.0
    temporal_delta_count: int = 0
    previous_luma: np.ndarray | None = None

    @staticmethod
    def neutral(frame_count: int) -> "_LumaStats":
        return _LumaStats(
            frame_count=frame_count,
            luma_min=0.0,
            luma_max=255.0,
            luma_mean_sum=127.5 * frame_count,
            temporal_delta_sum=0.0,
            temporal_delta_count=0,
        )

    def add(self, luma: np.ndarray) -> None:
        self.frame_count += 1
        self.luma_min = min(self.luma_min, float(np.min(luma)))
        self.luma_max = max(self.luma_max, float(np.max(luma)))
        self.luma_mean_sum += float(np.mean(luma))
        if self.previous_luma is not None:
            self.temporal_delta_sum += float(np.mean(np.abs(luma - self.previous_luma)))
            self.temporal_delta_count += 1
        self.previous_luma = luma

    def report(self, *, source: str, width: int, height: int, fps: float | None) -> VideoHealthReport:
        if self.frame_count <= 0:
            raise VideoHealthError(f"Video health check failed: {source} contains no frames.")
        return VideoHealthReport(
            source=source,
            frame_count=self.frame_count,
            width=width,
            height=height,
            fps=fps,
            luma_min=self.luma_min,
            luma_max=self.luma_max,
            luma_mean=self.luma_mean_sum / self.frame_count,
            mean_temporal_delta=(
                self.temporal_delta_sum / self.temporal_delta_count if self.temporal_delta_count else None
            ),
        )
