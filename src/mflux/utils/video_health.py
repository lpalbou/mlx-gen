from __future__ import annotations

import logging
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
        file_path = Path(path)
        if not file_path.exists():
            raise VideoHealthError(f"Video health check failed: {file_path} does not exist.")
        if file_path.stat().st_size <= 0:
            raise VideoHealthError(f"Video health check failed: {file_path} is empty.")

        import av

        with av.open(str(file_path)) as container:
            if len(container.streams.video) == 0:
                raise VideoHealthError(f"Video health check failed: {file_path} has no video stream.")
            video_stream = container.streams.video[0]
            width = int(video_stream.width)
            height = int(video_stream.height)
            fps = VideoHealth._rate_to_float(video_stream.average_rate or video_stream.base_rate)
            stats = _LumaStats()
            for frame in container.decode(video_stream):
                stats.add(VideoHealth._luma_array(PIL.Image.fromarray(frame.to_ndarray(format="rgb24"))))

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
        rgb = np.array(frame.convert("RGB"), dtype=np.float32)
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    @staticmethod
    def _rate_to_float(rate: Fraction | int | float | None) -> float | None:
        if rate is None:
            return None
        return float(rate)


@dataclass
class _LumaStats:
    frame_count: int = 0
    luma_min: float = float("inf")
    luma_max: float = float("-inf")
    luma_mean_sum: float = 0.0
    temporal_delta_sum: float = 0.0
    temporal_delta_count: int = 0
    previous_luma: np.ndarray | None = None

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
