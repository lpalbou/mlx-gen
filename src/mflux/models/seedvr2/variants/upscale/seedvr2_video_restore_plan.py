from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.seedvr2.seedvr2_initializer import SeedVR2Initializer
from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util
from mflux.utils.scale_factor import ScaleFactor
from mflux.utils.video_util import DecodedVideoClip, VideoUtil

GB = 1000**3


class SeedVR2VideoRestoreRefused(ValueError):
    SCALE_ABOVE_SAFE_PROFILE = "scale_above_safe_profile"
    BEYOND_MEMORY_PROFILE = "beyond_memory_profile"
    BELOW_TEMPORAL_QUALITY_FLOOR = "below_temporal_quality_floor"

    def __init__(self, message: str, *, reason: str, warnings: tuple[str, ...] = ()):
        super().__init__(message)
        self.reason = reason
        self.warnings = warnings or (message,)


@dataclass(frozen=True)
class SeedVR2VideoRestorePlan:
    CACHE_LIMIT_BYTES: ClassVar[int] = 8 * 1000**3
    DEFAULT_TEMPORAL_CHUNK_SIZE: ClassVar[int] = 49
    DEFAULT_TEMPORAL_CHUNK_OVERLAP: ClassVar[int] = 16

    variant: str
    resolution: int | ScaleFactor
    start_seconds: float
    max_frames: int | None
    requested_frames: int
    source_width: int
    source_height: int
    audio_present: bool
    target_height: int
    target_width: int
    restore_mode: str
    route_reason: str
    effective_chunk_size: int
    effective_chunk_overlap: int
    chunk_frame_limit: int
    chunk_pixel_volume: int
    enforce_memory_budget: bool
    risk_level: str
    warnings: tuple[str, ...]
    low_ram_required: bool = True
    keep_transformer: bool = True
    cache_limit_bytes: int = CACHE_LIMIT_BYTES

    @classmethod
    def for_video(
        cls,
        video_path: str | Path,
        *,
        model_config: ModelConfig = ModelConfig.seedvr2_3b(),
        model_path: str | None = None,
        start_seconds: float = 0.0,
        max_frames: int | None = None,
        resolution: int | str | ScaleFactor | None = None,
        temporal_chunk_size: int | None = None,
        temporal_chunk_overlap: int | None = None,
        force_unsafe_memory_profile: bool = False,
        weights_resident: bool = False,
    ) -> "SeedVR2VideoRestorePlan":
        chunk_size_was_explicit = temporal_chunk_size is not None
        chunk_overlap_was_explicit = temporal_chunk_overlap is not None
        if not chunk_size_was_explicit:
            temporal_chunk_size = cls.DEFAULT_TEMPORAL_CHUNK_SIZE
        if not chunk_overlap_was_explicit:
            temporal_chunk_overlap = cls.DEFAULT_TEMPORAL_CHUNK_OVERLAP
        cls.validate_temporal_chunk_options(temporal_chunk_size, temporal_chunk_overlap)
        resolution = SeedVR2VideoRestorePlan._resolve_resolution(resolution)
        source_probe = VideoUtil.read_video_clip(video_path, start_seconds=start_seconds, max_frames=1)
        requested_frames = SeedVR2VideoRestorePlan._requested_frame_count(source_probe, max_frames)

        variant = SeedVR2VideoRestorePlan._variant_name(model_config)
        height, width = SeedVR2Util.resolved_video_frame_size(
            source_width=source_probe.source_width,
            source_height=source_probe.source_height,
            resolution=resolution,
        )
        area = height * width
        min_side = min(source_probe.source_width, source_probe.source_height)
        scale = (
            float(resolution.get_scaled_value(min_side)) / float(min_side)
            if isinstance(resolution, ScaleFactor)
            else float(resolution) / float(min_side)
        )
        weight_bytes = (
            0
            if weights_resident
            else SeedVR2Initializer.estimate_resident_weight_bytes(model_config=model_config, model_path=model_path)
        )
        budget_bytes = SeedVR2Util.video_restore_budget_bytes(
            weights_resident=weights_resident, weight_bytes=weight_bytes
        )
        safe_chunk_limit = SeedVR2VideoRestorePlan._safe_chunk_frame_limit(
            model_config=model_config,
            height=height,
            width=width,
            requested_frames=requested_frames,
            budget_bytes=budget_bytes,
        )
        # (reason, message) pairs in the order the CLI has always reported them.
        refusals: list[tuple[str, str]] = []

        if chunk_size_was_explicit:
            chunk_size_error = SeedVR2VideoRestorePlan._explicit_chunk_size_error(
                requested_frames=requested_frames,
                temporal_chunk_size=temporal_chunk_size,
            )
            if chunk_size_error is not None:
                raise ValueError(chunk_size_error)

        requested_chunk_size = SeedVR2VideoRestorePlan._requested_streaming_chunk_size(
            requested_frames=requested_frames,
            temporal_chunk_size=temporal_chunk_size,
            chunk_size_was_explicit=chunk_size_was_explicit,
            safe_chunk_limit=safe_chunk_limit,
            force_unsafe_memory_profile=force_unsafe_memory_profile,
        )
        effective_chunk_size = SeedVR2VideoRestorePlan._aligned_chunk_size(requested_chunk_size)
        if force_unsafe_memory_profile:
            route_reason = "unsafe_override"
        else:
            route_reason = "safe_streaming"
            if scale > 1.0:
                refusals.append(
                    (
                        SeedVR2VideoRestoreRefused.SCALE_ABOVE_SAFE_PROFILE,
                        "SeedVR2 safe video mode only supports source-size restoration or smaller output. "
                        "Use 1x or override with --force-unsafe-video-memory.",
                    )
                )
            frame_size_refusal = SeedVR2VideoRestorePlan._frame_size_refusal(
                model_config=model_config,
                source_probe=source_probe,
                height=height,
                width=width,
                requested_frames=requested_frames,
                budget_bytes=budget_bytes,
            )
            if frame_size_refusal is not None:
                refusals.append((SeedVR2VideoRestoreRefused.BEYOND_MEMORY_PROFILE, frame_size_refusal))
                warnings = tuple(message for _, message in refusals)
                raise SeedVR2VideoRestoreRefused(refusals[0][1], reason=refusals[0][0], warnings=warnings)
            if safe_chunk_limit < requested_frames and effective_chunk_size > safe_chunk_limit:
                refusals.append(
                    (
                        SeedVR2VideoRestoreRefused.BEYOND_MEMORY_PROFILE,
                        "This SeedVR2 video chunk request exceeds the supported safe memory profile. "
                        "Use --force-unsafe-video-memory to run it explicitly, or reduce source size, resolution, "
                        "or clip length.",
                    )
                )
        effective_chunk_overlap = SeedVR2VideoRestorePlan._effective_chunk_overlap(
            chunk_size=effective_chunk_size,
            requested_frames=requested_frames,
            requested_overlap=temporal_chunk_overlap,
            overlap_was_explicit=chunk_overlap_was_explicit,
        )
        temporal_quality_error = SeedVR2Util.streamed_video_temporal_quality_error(
            frame_count=requested_frames,
            chunk_size=effective_chunk_size,
            overlap=effective_chunk_overlap,
        )
        if temporal_quality_error is not None:
            reason = SeedVR2VideoRestoreRefused.BELOW_TEMPORAL_QUALITY_FLOOR
            # The quality floor is not a memory limit, so the unsafe switch does not lift it.
            if force_unsafe_memory_profile:
                raise SeedVR2VideoRestoreRefused(temporal_quality_error, reason=reason)
            refusals.append((reason, temporal_quality_error))

        warnings = tuple(message for _, message in refusals)
        if refusals and not force_unsafe_memory_profile:
            raise SeedVR2VideoRestoreRefused(refusals[0][1], reason=refusals[0][0], warnings=warnings)

        risk_level = "low"
        if scale > 1.0 or variant in {"7b", "7b-sharp"} or force_unsafe_memory_profile:
            risk_level = "medium"
        if warnings:
            risk_level = "high"

        return cls(
            variant=variant,
            resolution=resolution,
            start_seconds=start_seconds,
            max_frames=max_frames,
            requested_frames=requested_frames,
            source_width=source_probe.source_width,
            source_height=source_probe.source_height,
            audio_present=source_probe.audio_present,
            target_height=height,
            target_width=width,
            restore_mode="streaming",
            route_reason=route_reason,
            effective_chunk_size=effective_chunk_size,
            effective_chunk_overlap=effective_chunk_overlap,
            chunk_frame_limit=safe_chunk_limit,
            chunk_pixel_volume=effective_chunk_size * area,
            enforce_memory_budget=route_reason != "unsafe_override",
            risk_level=risk_level,
            warnings=warnings,
        )

    @staticmethod
    def validate_temporal_chunk_options(temporal_chunk_size: int, temporal_chunk_overlap: int) -> None:
        if temporal_chunk_size <= 0:
            raise ValueError("--temporal-chunk-size must be greater than zero.")
        if temporal_chunk_overlap < 0:
            raise ValueError("--temporal-chunk-overlap must be greater than or equal to zero.")
        if temporal_chunk_overlap >= temporal_chunk_size:
            raise ValueError("--temporal-chunk-overlap must be smaller than --temporal-chunk-size.")

    @staticmethod
    def _resolve_resolution(resolution: int | str | ScaleFactor | None) -> int | ScaleFactor:
        if resolution is None:
            return ScaleFactor(1)
        if isinstance(resolution, str):
            text = resolution.strip()
            return int(text) if text.isdigit() else ScaleFactor.parse(text)
        return resolution

    @staticmethod
    def _requested_frame_count(source_probe: DecodedVideoClip, max_frames: int | None) -> int:
        if source_probe.source_frame_count is not None:
            available_frames = max(0, source_probe.source_frame_count - source_probe.clip_start_frame)
        elif source_probe.source_duration_seconds is not None:
            available_frames = max(
                1, int(round(source_probe.source_duration_seconds * source_probe.fps)) - source_probe.clip_start_frame
            )
        else:
            raise ValueError("SeedVR2 video restore requires a finite source frame count or duration.")
        return min(max_frames, available_frames) if max_frames is not None else available_frames

    @staticmethod
    def _variant_name(model_config: ModelConfig) -> str:
        aliases = {alias.lower() for alias in (model_config.aliases or [])}
        if "seedvr2-7b-sharp" in aliases:
            return "7b-sharp"
        if "seedvr2-7b" in aliases:
            return "7b"
        return "3b"

    @staticmethod
    def _memory_profile(model_config: ModelConfig) -> tuple[int, str]:
        overrides = model_config.transformer_overrides or {}
        inner_dim = int(overrides.get("vid_dim", 2560))
        text_attention_mode = str(overrides.get("text_attention_mode", "window_pool"))
        return inner_dim, text_attention_mode

    @staticmethod
    def _working_set_bytes(model_config: ModelConfig, *, frame_count: int, height: int, width: int) -> int:
        inner_dim, text_attention_mode = SeedVR2VideoRestorePlan._memory_profile(model_config)
        return SeedVR2Util.estimate_video_restore_working_set_bytes(
            frame_count=SeedVR2Util.padded_video_frame_count(frame_count),
            height=height,
            width=width,
            inner_dim=inner_dim,
            text_attention_mode=text_attention_mode,
        )

    @staticmethod
    def _safe_chunk_frame_limit(
        *,
        model_config: ModelConfig,
        height: int,
        width: int,
        requested_frames: int,
        budget_bytes: int,
    ) -> int:
        if requested_frames <= 0 or height <= 0 or width <= 0:
            raise ValueError("requested_frames, height, and width must be greater than zero.")
        # The largest chunk that fits, not the first that does not: the VAE's working set is not
        # monotonic in the chunk length (a 33-frame chunk needs less than a 29-frame one).
        max_safe = 0
        for candidate in range(1, requested_frames + 1):
            estimate = SeedVR2VideoRestorePlan._working_set_bytes(
                model_config, frame_count=candidate, height=height, width=width
            )
            if estimate <= budget_bytes:
                max_safe = candidate
        return max_safe

    @staticmethod
    def _smallest_chunk_working_set_bytes(
        model_config: ModelConfig, *, height: int, width: int, requested_frames: int
    ) -> int:
        # The chunks the profile allows: the whole clip up to the temporal quality floor, any
        # 4n+1 length from the floor up to the clip otherwise.
        floor = SeedVR2Util.VIDEO_MIN_PRODUCTION_STREAMING_CHUNK_FRAMES
        if requested_frames <= floor:
            candidates = [requested_frames]
        else:
            candidates = range(floor, SeedVR2Util.padded_video_frame_count(requested_frames) + 1, 4)
        return min(
            SeedVR2VideoRestorePlan._working_set_bytes(model_config, frame_count=frames, height=height, width=width)
            for frames in candidates
        )

    @staticmethod
    def _frame_size_refusal(
        *,
        model_config: ModelConfig,
        source_probe: DecodedVideoClip,
        height: int,
        width: int,
        requested_frames: int,
        budget_bytes: int,
    ) -> str | None:
        needed = SeedVR2VideoRestorePlan._smallest_chunk_working_set_bytes(
            model_config, height=height, width=width, requested_frames=requested_frames
        )
        if needed <= budget_bytes:
            return None
        message = (
            f"SeedVR2 video restore cannot fit a {width}x{height} output frame in this machine's video memory "
            f"budget ({budget_bytes / GB:.1f} GB): the VAE works on whole frames, so this size needs about "
            f"{needed / GB:.1f} GB whatever the temporal chunk size. "
        )
        for shortest_edge in range(min(height, width) - 16, 15, -16):
            fit_height, fit_width = SeedVR2Util.resolved_video_frame_size(
                source_width=source_probe.source_width,
                source_height=source_probe.source_height,
                resolution=shortest_edge,
            )
            fit_bytes = SeedVR2VideoRestorePlan._smallest_chunk_working_set_bytes(
                model_config, height=fit_height, width=fit_width, requested_frames=requested_frames
            )
            if fit_bytes <= budget_bytes:
                return message + (
                    f"The largest output that fits is {fit_width}x{fit_height} ({fit_width * fit_height} pixels "
                    f"per frame): use --resolution {shortest_edge} or smaller, or --force-unsafe-video-memory to "
                    "run this size anyway."
                )
        return message + "No output size fits this budget: free memory, or use --force-unsafe-video-memory."

    @staticmethod
    def _aligned_chunk_size(max_frames: int) -> int:
        if max_frames <= 1:
            return 1
        candidate = max_frames
        while candidate > 1 and (candidate - 1) % 4 != 0:
            candidate -= 1
        return max(candidate, 1)

    @staticmethod
    def _aligned_chunk_overlap(chunk_size: int, requested_overlap: int) -> int:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        overlap_cap = max(0, chunk_size // 3)
        overlap = min(requested_overlap, overlap_cap)
        overlap -= overlap % 4
        return max(0, overlap)

    @staticmethod
    def _explicit_chunk_size_error(*, requested_frames: int, temporal_chunk_size: int) -> str | None:
        if temporal_chunk_size <= 1:
            return None
        if (temporal_chunk_size - 1) % 4 != 0:
            return (
                "--temporal-chunk-size must satisfy 4n+1 for SeedVR2 video restore. "
                f"Got {temporal_chunk_size}; use an explicit aligned value such as "
                f"{SeedVR2Util.padded_video_frame_count(min(temporal_chunk_size, requested_frames))}."
            )
        return None

    @staticmethod
    def _explicit_chunk_overlap(chunk_size: int, requested_overlap: int) -> int:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if requested_overlap < 0:
            raise ValueError("--temporal-chunk-overlap must be greater than or equal to zero.")
        if requested_overlap >= chunk_size:
            raise ValueError("--temporal-chunk-overlap must be smaller than the effective temporal chunk size.")
        if requested_overlap % 4 != 0:
            raise ValueError("--temporal-chunk-overlap must be a multiple of 4 for SeedVR2 video restore.")
        return requested_overlap

    @staticmethod
    def _effective_chunk_overlap(
        *,
        chunk_size: int,
        requested_frames: int,
        requested_overlap: int,
        overlap_was_explicit: bool,
    ) -> int:
        if chunk_size >= requested_frames:
            return 0
        if overlap_was_explicit:
            return SeedVR2VideoRestorePlan._explicit_chunk_overlap(chunk_size, requested_overlap)
        return SeedVR2VideoRestorePlan._aligned_chunk_overlap(chunk_size, requested_overlap)

    @staticmethod
    def _requested_streaming_chunk_size(
        *,
        requested_frames: int,
        temporal_chunk_size: int,
        chunk_size_was_explicit: bool,
        safe_chunk_limit: int,
        force_unsafe_memory_profile: bool,
    ) -> int:
        if requested_frames <= 0:
            raise ValueError("requested_frames must be greater than zero.")
        if chunk_size_was_explicit:
            if temporal_chunk_size >= requested_frames:
                return SeedVR2Util.padded_video_frame_count(requested_frames)
            return temporal_chunk_size
        if force_unsafe_memory_profile or safe_chunk_limit >= requested_frames:
            return SeedVR2Util.padded_video_frame_count(requested_frames)
        return max(safe_chunk_limit, 1)
