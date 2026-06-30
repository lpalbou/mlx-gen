import fcntl
import gc
import json
import sys
import traceback
from copy import copy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import mlx.core as mx

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.output_paths import format_output_template, normalize_output_template, resolve_output_path
from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print, emit_cli_failure_event_for_argv
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.download_policy import DownloadRequiredError, is_huggingface_repo_id
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.seedvr2.latent_creator.seedvr2_latent_creator import SeedVR2LatentCreator
from mflux.models.seedvr2.seedvr2_initializer import SeedVR2Initializer
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.runtime_memory import RuntimeMemory
from mflux.utils.scale_factor import ScaleFactor
from mflux.utils.video_util import VideoUtil

SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

SUPPORTED_VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}

DEFAULT_SEEDVR2_VIDEO_CACHE_LIMIT_GB = 8.0
SEEDVR2_OFFICIAL_BOUNDED_FRAME_LIMIT = 121


@dataclass(frozen=True)
class SeedVR2VideoRestorePlan:
    variant: str
    requested_frames: int
    target_height: int
    target_width: int
    restore_mode: str
    route_reason: str
    effective_chunk_size: int | None
    effective_chunk_overlap: int | None
    chunk_frame_limit: int | None
    chunk_pixel_volume: int | None
    low_ram_required: bool
    low_ram_effective: bool
    cache_limit_gb: float | None
    risk_level: str
    warnings: tuple[str, ...]


class SeedVR2VideoRunLock:
    def __init__(self):
        self.path = ui_defaults.MFLUX_CACHE_DIR / "locks" / "seedvr2-video.lock"
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(
                "Another SeedVR2 video restore is already running in mlx-gen. "
                "Wait for it to finish or override with --force-unsafe-video-memory."
            ) from exc
        self.handle.write(f"{datetime.now().isoformat()} pid-lock\n")
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is None:
            return
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES


def _resolve_seedvr2_model(model_arg: str | None, model_path: str | None) -> tuple[ModelConfig, str | None]:
    if model_arg is None:
        return ModelConfig.seedvr2_3b(), model_path

    normalized = model_arg.lower()
    if normalized in {"seedvr2", "seedvr2-3b"}:
        return ModelConfig.seedvr2_3b(), None
    if normalized == "seedvr2-7b":
        return ModelConfig.seedvr2_7b(), None
    if normalized == "seedvr2-7b-sharp":
        return ModelConfig.seedvr2_7b_sharp(), None
    if normalized in {"bytedance-seed/seedvr2-3b", "bytedance-seed/seedvr2_3b"}:
        return ModelConfig.seedvr2_3b(), model_arg
    if normalized in {"bytedance-seed/seedvr2-7b", "bytedance-seed/seedvr2_7b"}:
        return ModelConfig.seedvr2_7b(), model_arg
    if normalized == "numz/seedvr2_comfyui":
        return ModelConfig.seedvr2_3b(), model_arg
    if normalized.startswith("abstractframework/seedvr2-3b-"):
        return ModelConfig.seedvr2_3b(), model_arg
    if normalized.startswith("abstractframework/seedvr2-7b-"):
        return ModelConfig.seedvr2_7b(), model_arg

    requested_model_path = model_path
    if requested_model_path is None and Path(model_arg).expanduser().exists():
        requested_model_path = model_arg

    if requested_model_path is not None:
        path = Path(requested_model_path).expanduser()
        if path.is_dir():
            has_3b = (path / "seedvr2_ema_3b_fp16.safetensors").exists()
            has_official_3b = (path / "seedvr2_ema_3b.pth").exists()
            has_7b = (path / "seedvr2_ema_7b_fp16.safetensors").exists()
            has_official_7b = (path / "seedvr2_ema_7b.pth").exists()
            has_official_7b_sharp = (path / "seedvr2_ema_7b_sharp.pth").exists()
            if has_official_7b_sharp and not (has_3b or has_official_3b or has_7b or has_official_7b):
                return ModelConfig.seedvr2_7b_sharp(), requested_model_path
            if (has_7b or has_official_7b or has_official_7b_sharp) and not (has_3b or has_official_3b):
                if has_official_7b_sharp and ("seedvr2-7b-sharp" in normalized or not (has_7b or has_official_7b)):
                    return ModelConfig.seedvr2_7b_sharp(), requested_model_path
                return ModelConfig.seedvr2_7b(), requested_model_path
            if (has_3b or has_official_3b) and not (has_7b or has_official_7b):
                return ModelConfig.seedvr2_3b(), requested_model_path
            if (path / "transformer" / "model.safetensors.index.json").exists():
                if (
                    "seedvr2-7b-sharp" in normalized
                    or "7b-sharp" in path.name.lower()
                    or "7b_sharp" in path.name.lower()
                ):
                    return ModelConfig.seedvr2_7b_sharp(), requested_model_path
                if "seedvr2-7b" in normalized or "7b" in path.name.lower():
                    return ModelConfig.seedvr2_7b(), requested_model_path
                return ModelConfig.seedvr2_3b(), requested_model_path

    if is_huggingface_repo_id(model_arg):
        raise ValueError(
            "Unsupported SeedVR2 model handle "
            f"{model_arg!r}. Use seedvr2, seedvr2-3b, seedvr2-7b, seedvr2-7b-sharp, "
            "ByteDance-Seed/SeedVR2-3B, ByteDance-Seed/SeedVR2-7B, "
            "AbstractFramework/seedvr2-3b-8bit, AbstractFramework/seedvr2-3b-4bit, "
            "AbstractFramework/seedvr2-7b-8bit, AbstractFramework/seedvr2-7b-4bit, "
            "or an explicit local SeedVR2 path."
        )

    source = (requested_model_path or model_arg).lower()
    if "seedvr2_ema_7b_sharp" in source or "seedvr2-7b-sharp" in source or "seedvr2_7b_sharp" in source:
        return ModelConfig.seedvr2_7b_sharp(), requested_model_path
    if "seedvr2_ema_7b" in source or "seedvr2-7b" in source:
        return ModelConfig.seedvr2_7b(), requested_model_path
    return ModelConfig.seedvr2_3b(), requested_model_path


def _expand_image_paths(image_paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for image_path in image_paths:
        if image_path.is_dir():
            dir_images = sorted(
                [path for path in image_path.iterdir() if _is_image_file(path)],
                key=lambda path: path.name.lower(),
            )
            if not dir_images:
                print(f"No images found in directory: {image_path}", file=sys.stderr)
            expanded.extend(dir_images)
        else:
            expanded.append(image_path)
    return expanded


def _expand_video_paths(video_paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for video_path in video_paths:
        if video_path.is_dir():
            dir_videos = sorted(
                [path for path in video_path.iterdir() if _is_video_file(path)],
                key=lambda path: path.name.lower(),
            )
            if not dir_videos:
                print(f"No videos found in directory: {video_path}", file=sys.stderr)
            expanded.extend(dir_videos)
        else:
            expanded.append(video_path)
    return expanded


def _provided_options(argv: list[str]) -> set[str]:
    provided = set()
    aliases = {
        "-m": "--model",
        "-i": "--image-path",
        "-s": "--seed",
        "-r": "--resolution",
        "-q": "--quantize",
    }
    expects_value = {
        "--model",
        "--image-path",
        "--video-path",
        "--seed",
        "--resolution",
        "--quantize",
        "--softness",
        "--color-correction",
        "--start-seconds",
        "--max-frames",
        "--temporal-chunk-size",
        "--temporal-chunk-overlap",
        "--mlx-cache-limit-gb",
        "--output",
    }
    index = 0
    while index < len(argv):
        arg = aliases.get(argv[index], argv[index])
        if not arg.startswith("-"):
            index += 1
            continue
        provided.add(arg)
        if arg in expects_value:
            index += 2
            continue
        index += 1
    return provided


def _validate_batch_output_collisions(
    *,
    output_pattern: str,
    image_paths: list[Path],
    video_paths: list[Path],
    seeds: list[int],
    replace: bool,
) -> None:
    if not replace:
        return
    planned_outputs: dict[Path, list[str]] = {}
    for source_path in [*image_paths, *video_paths]:
        for seed in seeds:
            rendered = Path(
                format_output_template(
                    output_pattern,
                    seed=seed,
                    input_name=source_path.stem,
                )
            )
            planned_outputs.setdefault(rendered, []).append(f"{source_path.name} (seed {seed})")
    collisions = {path: sources for path, sources in planned_outputs.items() if len(sources) > 1}
    if collisions:
        details = "; ".join(
            f"{path} <- {', '.join(sources)}"
            for path, sources in sorted(collisions.items(), key=lambda item: str(item[0]))
        )
        raise ValueError(
            "SeedVR2 would write multiple results to the same output path with --replace true. "
            "Use distinct source basenames, choose a different --output template, or pass "
            f"--replace false. Collisions: {details}"
        )


def _requested_video_frame_count(source_probe, max_frames: int | None) -> int:
    if source_probe.source_frame_count is not None:
        available_frames = max(0, source_probe.source_frame_count - source_probe.clip_start_frame)
    elif source_probe.source_duration_seconds is not None:
        available_frames = max(
            1, int(round(source_probe.source_duration_seconds * source_probe.fps)) - source_probe.clip_start_frame
        )
    else:
        raise ValueError("SeedVR2 video restore requires a finite source frame count or duration.")
    return min(max_frames, available_frames) if max_frames is not None else available_frames


def _seedvr2_variant_name(model_config: ModelConfig) -> str:
    aliases = {alias.lower() for alias in (model_config.aliases or [])}
    if "seedvr2-7b-sharp" in aliases:
        return "7b-sharp"
    if "seedvr2-7b" in aliases:
        return "7b"
    return "3b"


def _seedvr2_resolution_scale(resolution: int | ScaleFactor, min_side: int) -> float:
    if isinstance(resolution, ScaleFactor):
        return float(resolution.get_scaled_value(min_side)) / float(min_side)
    return float(resolution) / float(min_side)


def _estimate_seedvr2_output_size(
    *,
    source_width: int,
    source_height: int,
    resolution: int | ScaleFactor,
) -> tuple[int, int]:
    min_side = min(source_width, source_height)
    if isinstance(resolution, ScaleFactor):
        target_res = resolution.get_scaled_value(min_side)
    else:
        target_res = resolution
    scale = target_res / min_side
    width = max(2, (int(source_width * scale) // 2) * 2)
    height = max(2, (int(source_height * scale) // 2) * 2)
    return height, width


def _aligned_chunk_size(max_frames: int) -> int:
    if max_frames <= 1:
        return 1
    candidate = max_frames
    while candidate > 1 and (candidate - 1) % 4 != 0:
        candidate -= 1
    return max(candidate, 1)


def _aligned_chunk_overlap(chunk_size: int, requested_overlap: int) -> int:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    overlap_cap = max(0, chunk_size // 3)
    overlap = min(requested_overlap, overlap_cap)
    overlap -= overlap % 4
    return max(0, overlap)


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
        return _explicit_chunk_overlap(chunk_size, requested_overlap)
    return _aligned_chunk_overlap(chunk_size, requested_overlap)


def _streaming_temporal_quality_error(*, frame_count: int, chunk_size: int, overlap: int) -> str | None:
    return SeedVR2Util.streamed_video_temporal_quality_error(
        frame_count=frame_count,
        chunk_size=chunk_size,
        overlap=overlap,
    )


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


def _seedvr2_memory_profile(model_config: ModelConfig) -> tuple[int, str]:
    overrides = model_config.transformer_overrides or {}
    inner_dim = int(overrides.get("vid_dim", 2560))
    text_attention_mode = str(overrides.get("text_attention_mode", "window_pool"))
    return inner_dim, text_attention_mode


def _safe_chunk_frame_limit(
    *,
    model_config: ModelConfig,
    resolved_model_path: str | None,
    height: int,
    width: int,
    requested_frames: int,
    requested_overlap: int,
) -> int:
    if requested_frames <= 0 or height <= 0 or width <= 0:
        raise ValueError("requested_frames, height, and width must be greater than zero.")
    inner_dim, text_attention_mode = _seedvr2_memory_profile(model_config)
    resident_weight_bytes = SeedVR2Initializer.estimate_resident_weight_bytes(
        model_config=model_config,
        model_path=resolved_model_path,
    )
    budget_bytes = SeedVR2Util.host_safe_video_memory_budget_bytes(reserve_bytes=resident_weight_bytes)

    max_safe = 0
    for candidate in range(1, requested_frames + 1):
        estimate = SeedVR2Util.estimate_video_restore_working_set_bytes(
            frame_count=SeedVR2Util.padded_video_frame_count(candidate),
            height=height,
            width=width,
            inner_dim=inner_dim,
            text_attention_mode=text_attention_mode,
        )
        if estimate <= budget_bytes:
            max_safe = candidate
        else:
            break
    return max_safe


def _plan_seedvr2_video_restore(
    *,
    model_config: ModelConfig,
    resolved_model_path: str | None = None,
    source_probe,
    requested_frames: int,
    resolution: int | ScaleFactor,
    temporal_chunk_size: int,
    temporal_chunk_overlap: int,
    chunk_size_was_explicit: bool,
    chunk_overlap_was_explicit: bool,
    low_ram_requested: bool,
    cache_limit_gb: float | None,
    force_unsafe_memory_profile: bool,
) -> SeedVR2VideoRestorePlan:
    variant = _seedvr2_variant_name(model_config)
    height, width = _estimate_seedvr2_output_size(
        source_width=source_probe.source_width,
        source_height=source_probe.source_height,
        resolution=resolution,
    )
    area = height * width
    scale = _seedvr2_resolution_scale(resolution, min(source_probe.source_width, source_probe.source_height))
    safe_chunk_limit = _safe_chunk_frame_limit(
        model_config=model_config,
        resolved_model_path=resolved_model_path,
        height=height,
        width=width,
        requested_frames=requested_frames,
        requested_overlap=temporal_chunk_overlap,
    )
    warnings: list[str] = []
    low_ram_required = True

    if low_ram_required and not low_ram_requested and not force_unsafe_memory_profile:
        warnings.append("SeedVR2 video restore requires --low-ram on the supported safe profile.")

    if chunk_size_was_explicit:
        chunk_size_error = _explicit_chunk_size_error(
            requested_frames=requested_frames,
            temporal_chunk_size=temporal_chunk_size,
        )
        if chunk_size_error is not None:
            raise ValueError(chunk_size_error)

    requested_chunk_size = _requested_streaming_chunk_size(
        requested_frames=requested_frames,
        temporal_chunk_size=temporal_chunk_size,
        chunk_size_was_explicit=chunk_size_was_explicit,
        safe_chunk_limit=safe_chunk_limit,
        force_unsafe_memory_profile=force_unsafe_memory_profile,
    )
    if force_unsafe_memory_profile:
        restore_mode = "streaming"
        route_reason = "unsafe_override"
        effective_chunk_size = _aligned_chunk_size(requested_chunk_size)
        effective_chunk_overlap = _effective_chunk_overlap(
            chunk_size=effective_chunk_size,
            requested_frames=requested_frames,
            requested_overlap=temporal_chunk_overlap,
            overlap_was_explicit=chunk_overlap_was_explicit,
        )
        temporal_quality_error = _streaming_temporal_quality_error(
            frame_count=requested_frames,
            chunk_size=effective_chunk_size,
            overlap=effective_chunk_overlap,
        )
        if temporal_quality_error is not None:
            raise ValueError(temporal_quality_error)
        chunk_pixel_volume = effective_chunk_size * area
    else:
        restore_mode = "streaming"
        route_reason = "safe_streaming"
        effective_chunk_size = _aligned_chunk_size(requested_chunk_size)
        if scale > 1.0:
            warnings.append(
                "SeedVR2 safe video mode only supports source-size restoration or smaller output. "
                "Use 1x or override with --force-unsafe-video-memory."
            )
        if safe_chunk_limit < requested_frames and effective_chunk_size > safe_chunk_limit:
            warnings.append(
                "This SeedVR2 video chunk request exceeds the supported safe memory profile. "
                "Use --force-unsafe-video-memory to run it explicitly, or reduce source size, resolution, "
                "or clip length."
            )
        effective_chunk_overlap = _effective_chunk_overlap(
            chunk_size=effective_chunk_size,
            requested_frames=requested_frames,
            requested_overlap=temporal_chunk_overlap,
            overlap_was_explicit=chunk_overlap_was_explicit,
        )
        temporal_quality_error = _streaming_temporal_quality_error(
            frame_count=requested_frames,
            chunk_size=effective_chunk_size,
            overlap=effective_chunk_overlap,
        )
        if temporal_quality_error is not None:
            warnings.append(temporal_quality_error)
        chunk_pixel_volume = effective_chunk_size * area

    risk_level = "low"
    if scale > 1.0 or variant in {"7b", "7b-sharp"} or force_unsafe_memory_profile:
        risk_level = "medium"
    if warnings:
        risk_level = "high"

    return SeedVR2VideoRestorePlan(
        variant=variant,
        requested_frames=requested_frames,
        target_height=height,
        target_width=width,
        restore_mode=restore_mode,
        route_reason=route_reason,
        effective_chunk_size=effective_chunk_size,
        effective_chunk_overlap=effective_chunk_overlap,
        chunk_frame_limit=safe_chunk_limit,
        chunk_pixel_volume=chunk_pixel_volume,
        low_ram_required=low_ram_required,
        low_ram_effective=low_ram_requested,
        cache_limit_gb=cache_limit_gb,
        risk_level=risk_level,
        warnings=tuple(warnings),
    )


def _apply_cache_limit_if_needed(args) -> None:
    CallbackManager.apply_runtime_memory_options(args)


def _seedvr2_video_runtime_diagnostics() -> dict:
    return RuntimeMemory.snapshot("seedvr2-failure", synchronize=True).to_metadata()


def _write_seedvr2_failure_manifest(
    *,
    output_path: str | Path,
    video_path: Path,
    model: str | None,
    seed: int,
    resolution: int | ScaleFactor,
    softness: float,
    color_correction_mode: str,
    drop_audio: bool,
    start_seconds: float,
    max_frames: int | None,
    plan: SeedVR2VideoRestorePlan,
    error: BaseException,
) -> Path:
    failure_path = Path(output_path).with_suffix(".failure.json")
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(),
                "status": "failed",
                "error_type": error.__class__.__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "run": {
                    "model": model,
                    "task": "video-to-video",
                    "seed": seed,
                    "video_path": str(video_path),
                    "resolution": str(resolution),
                    "softness": round(float(softness), 3),
                    "color_correction_mode": color_correction_mode,
                    "drop_audio": drop_audio,
                    "start_seconds": round(float(start_seconds), 3),
                    "max_frames": max_frames,
                    "restore_plan": asdict(plan),
                    "output": str(output_path),
                },
                "runtime_diagnostics": _seedvr2_video_runtime_diagnostics(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return failure_path


def _print_seedvr2_video_preflight(
    video_path: Path,
    source_probe,
    plan: SeedVR2VideoRestorePlan,
    *,
    json_events: bool,
) -> None:
    cli_print(
        "SeedVR2 video preflight: "
        f"model={plan.variant} "
        f"source={source_probe.source_width}x{source_probe.source_height} "
        f"target={plan.target_width}x{plan.target_height} "
        f"frames={plan.requested_frames} "
        f"mode={plan.restore_mode} "
        f"reason={plan.route_reason} "
        f"low_ram={plan.low_ram_effective} "
        f"cache_limit_gb={plan.cache_limit_gb or 'none'} "
        f"video={video_path.name}",
        json_events=json_events,
    )
    if plan.restore_mode == "streaming":
        cli_print(
            "SeedVR2 streaming plan: "
            f"chunk_size={plan.effective_chunk_size} "
            f"chunk_overlap={plan.effective_chunk_overlap} "
            f"safe_chunk_frame_limit={plan.chunk_frame_limit} "
            f"chunk_pixel_volume={plan.chunk_pixel_volume}",
            json_events=json_events,
        )
    for warning in plan.warnings:
        cli_print(f"SeedVR2 warning: {warning}", json_events=json_events)


def _run_seedvr2_video_restore(
    *,
    model: SeedVR2,
    args,
    video_path: Path,
    source_probe,
    plan: SeedVR2VideoRestorePlan,
    output_pattern: str,
    seed: int,
) -> None:
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    events = CliRuntimeEventStream(
        enabled=bool(args.json_events),
        command="mlxgen upscale",
        model=model.model_config.model_name,
        seed=seed,
    )
    unsubscribe = None
    try:
        output_path = resolve_output_path(
            output_pattern,
            overwrite=args.replace,
            seed=seed,
            input_name=video_path.stem,
        )
        events.set_output_path(output_path)
        unsubscribe = events.subscribe_model(
            model,
            map_complete_to_generated=False,
            suppress_terminal_phases={"failed"},
        )
        _print_seedvr2_video_preflight(video_path, source_probe, plan, json_events=bool(args.json_events))
        if source_probe.audio_present:
            if args.drop_audio:
                cli_print(
                    "SeedVR2 note: source audio detected; --drop-audio was requested, so the restored MP4 will stay silent.",
                    json_events=bool(args.json_events),
                )
            else:
                cli_print(
                    "SeedVR2 note: source audio detected; the CLI will preserve the matching source audio segment. "
                    "If that cannot be proven safe, the run fails. Pass --drop-audio to allow a silent output intentionally.",
                    json_events=bool(args.json_events),
                )
        runtime_metadata = {
            "restore_mode": plan.restore_mode,
            "restore_mode_reason": plan.route_reason,
            "low_ram_requested": bool(args.low_ram),
            "low_ram_effective": bool(plan.low_ram_effective),
            "mlx_cache_limit_gb": args.mlx_cache_limit_gb,
            "requested_temporal_chunk_size": args.temporal_chunk_size,
            "requested_temporal_chunk_overlap": args.temporal_chunk_overlap,
            "effective_temporal_chunk_size": plan.effective_chunk_size,
            "effective_temporal_chunk_overlap": plan.effective_chunk_overlap,
            "seedvr2_risk_level": plan.risk_level,
            "seedvr2_target_height": plan.target_height,
            "seedvr2_target_width": plan.target_width,
            "drop_audio_requested": bool(args.drop_audio),
        }
        try:
            result_path = model.restore_video_to_path(
                seed=seed,
                video_path=video_path,
                resolution=args.resolution,
                softness=args.softness,
                start_seconds=args.start_seconds,
                max_frames=args.max_frames,
                output_path=output_path,
                export_json_metadata=args.metadata,
                overwrite=True,
                temporal_chunk_size=(
                    plan.effective_chunk_size if plan.effective_chunk_size is not None else args.temporal_chunk_size
                ),
                temporal_chunk_overlap=(
                    plan.effective_chunk_overlap
                    if plan.effective_chunk_overlap is not None
                    else args.temporal_chunk_overlap
                ),
                color_correction_mode=args.color_correction,
                drop_audio=args.drop_audio,
                restore_metadata=runtime_metadata,
                enforce_memory_budget=plan.route_reason != "unsafe_override",
            )
        except Exception as exc:
            failure_path = _write_seedvr2_failure_manifest(
                output_path=output_path,
                video_path=video_path,
                model=args.model,
                seed=seed,
                resolution=args.resolution,
                softness=args.softness,
                color_correction_mode=args.color_correction,
                drop_audio=args.drop_audio,
                start_seconds=args.start_seconds,
                max_frames=args.max_frames,
                plan=plan,
                error=exc,
            )
            events.emit_failed(task="video-to-video", error=exc, diagnostics_path=failure_path)
            cli_print(f"SeedVR2 failure manifest saved at: {failure_path}", json_events=bool(args.json_events))
            raise
        events.set_output_path(result_path)
        cli_print(f"Video saved successfully at: {result_path}", json_events=bool(args.json_events))
    finally:
        if unsubscribe is not None:
            unsubscribe()
        if gc_was_enabled:
            gc.enable()


def _load_seedvr2_model(
    *,
    parser: CommandLineParser,
    args,
    resolved_model_path: str | None,
    model_config: ModelConfig,
) -> SeedVR2:
    try:
        model = SeedVR2(
            quantize=args.quantize,
            model_path=resolved_model_path,
            model_config=model_config,
        )
    except DownloadRequiredError as exc:
        if getattr(args, "json_events", False):
            emit_cli_failure_event_for_argv(
                prog=parser.prog,
                argv=sys.argv[1:],
                error=exc,
                output_path=args.output,
            )
            cli_print(str(exc), json_events=True, error=True)
            raise SystemExit(1) from None
        parser.error(str(exc))
    return model


def _run_video_with_fresh_model(
    *,
    parser: CommandLineParser,
    args,
    resolved_model_path: str | None,
    model_config: ModelConfig,
    video_path: Path,
    source_probe,
    plan: SeedVR2VideoRestorePlan,
    output_pattern: str,
    seed: int,
) -> None:
    model = _load_seedvr2_model(
        parser=parser,
        args=args,
        resolved_model_path=resolved_model_path,
        model_config=model_config,
    )
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=SeedVR2LatentCreator,
    )
    if memory_saver is not None:
        memory_saver.keep_transformer = True
    try:
        _run_seedvr2_video_restore(
            model=model,
            args=args,
            video_path=video_path,
            source_probe=source_probe,
            plan=plan,
            output_pattern=output_pattern,
            seed=seed,
        )
    finally:
        if memory_saver:
            cli_print(memory_saver.memory_stats(), json_events=bool(args.json_events))
        del model
        mx.clear_cache()


def main():
    # 1. Parse command line arguments
    parser = CommandLineParser(description="Upscale an image using SeedVR2 diffusion-based super-resolution.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_seedvr2_upscale_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()
    provided_options = _provided_options(sys.argv[1:])

    image_paths = _expand_image_paths(args.image_path) if args.image_path else []
    video_paths = _expand_video_paths(args.video_path) if args.video_path else []
    if not image_paths and not video_paths:
        cli_print("No images or videos to upscale.", json_events=bool(args.json_events))
        return
    if image_paths and (args.start_seconds != 0.0 or args.max_frames is not None):
        parser.error("--start-seconds and --max-frames are only supported with --video-path.")
    if video_paths and args.vae_tiling:
        parser.error(
            "--vae-tiling is not supported for SeedVR2 video restore. Use --low-ram and temporal chunking instead."
        )
    if args.temporal_chunk_size <= 0:
        parser.error("--temporal-chunk-size must be greater than zero.")
    if args.temporal_chunk_overlap < 0:
        parser.error("--temporal-chunk-overlap must be greater than or equal to zero.")
    if args.temporal_chunk_overlap >= args.temporal_chunk_size:
        parser.error("--temporal-chunk-overlap must be smaller than --temporal-chunk-size.")
    safe_video_mode = bool(video_paths) and not args.force_unsafe_video_memory
    if video_paths and "--resolution" not in provided_options:
        args.resolution = ScaleFactor(1)
    if video_paths and not args.low_ram:
        cli_print("SeedVR2 video mode: enabling --low-ram automatically.", json_events=bool(args.json_events))
        args.low_ram = True

    try:
        model_config, resolved_model_path = _resolve_seedvr2_model(args.model, args.model_path)
    except ValueError as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    video_probes: dict[Path, object] = {}
    video_restore_plans: dict[Path, SeedVR2VideoRestorePlan] = {}
    try:
        if video_paths:
            if args.mlx_cache_limit_gb is None:
                args.mlx_cache_limit_gb = DEFAULT_SEEDVR2_VIDEO_CACHE_LIMIT_GB
            elif safe_video_mode and args.mlx_cache_limit_gb > DEFAULT_SEEDVR2_VIDEO_CACHE_LIMIT_GB:
                cli_print(
                    "SeedVR2 safe video mode: clamping --mlx-cache-limit-gb to "
                    f"{DEFAULT_SEEDVR2_VIDEO_CACHE_LIMIT_GB:g}.",
                    json_events=bool(args.json_events),
                )
                args.mlx_cache_limit_gb = DEFAULT_SEEDVR2_VIDEO_CACHE_LIMIT_GB
            cache_limit_gb = args.mlx_cache_limit_gb
            for video_path in video_paths:
                source_probe = VideoUtil.read_video_clip(
                    video_path,
                    start_seconds=args.start_seconds,
                    max_frames=1,
                )
                video_probes[video_path] = source_probe
                requested_frames = _requested_video_frame_count(source_probe, args.max_frames)
                plan = _plan_seedvr2_video_restore(
                    model_config=model_config,
                    resolved_model_path=resolved_model_path,
                    source_probe=source_probe,
                    requested_frames=requested_frames,
                    resolution=args.resolution,
                    temporal_chunk_size=args.temporal_chunk_size,
                    temporal_chunk_overlap=args.temporal_chunk_overlap,
                    chunk_size_was_explicit="--temporal-chunk-size" in provided_options,
                    chunk_overlap_was_explicit="--temporal-chunk-overlap" in provided_options,
                    low_ram_requested=args.low_ram,
                    cache_limit_gb=cache_limit_gb,
                    force_unsafe_memory_profile=args.force_unsafe_video_memory,
                )
                if plan.warnings and not args.force_unsafe_video_memory:
                    parser.error(plan.warnings[0])
                video_restore_plans[video_path] = plan
            _apply_cache_limit_if_needed(args)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        output_pattern = normalize_output_template(
            args.output,
            is_video=bool(video_paths),
            include_seed=len(args.seed) > 1,
            include_input_name=len(image_paths) > 1 or len(video_paths) > 1,
        )
        _validate_batch_output_collisions(
            output_pattern=output_pattern,
            image_paths=image_paths,
            video_paths=video_paths,
            seeds=args.seed,
            replace=args.replace,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        if image_paths and not video_paths:
            _apply_cache_limit_if_needed(args)
        if image_paths:
            model = _load_seedvr2_model(
                parser=parser,
                args=args,
                resolved_model_path=resolved_model_path,
                model_config=model_config,
            )
            image_tiling_config = model.tiling_config
            if args.vae_tiling:
                model.tiling_config = TilingConfig(vae_encode_tile_size=768, vae_encode_tile_overlap=128)
            callback_args = args
            if args.low_ram:
                callback_args = copy(args)
                callback_args.low_ram = False
            memory_saver = CallbackManager.register_callbacks(
                args=callback_args,
                model=model,
                latent_creator=SeedVR2LatentCreator,
            )
            if args.low_ram and not args.vae_tiling:
                # Keep SeedVR2 image --low-ram to cache control without silently changing VAE encode behavior.
                model.tiling_config = image_tiling_config
            try:
                for image_path in image_paths:
                    for seed in args.seed:
                        events = CliRuntimeEventStream(
                            enabled=bool(args.json_events),
                            command="mlxgen upscale",
                            model=model.model_config.model_name,
                            seed=seed,
                        )
                        output_path = resolve_output_path(
                            output_pattern,
                            overwrite=args.replace,
                            seed=seed,
                            input_name=image_path.stem,
                        )
                        events.set_output_path(output_path)
                        unsubscribe = events.subscribe_model(model, map_complete_to_generated=True)
                        try:
                            result = model.generate_image(
                                seed=seed,
                                image_path=image_path,
                                resolution=args.resolution,
                                softness=args.softness,
                                color_correction_mode=args.color_correction,
                            )
                            events.emit_save()
                            result.save(
                                output_path,
                                export_json_metadata=args.metadata,
                                overwrite=True,
                                embed_metadata=args.embed_metadata,
                            )
                            events.emit_complete()
                        except Exception as exc:
                            events.emit_failed(error=exc)
                            raise
                        finally:
                            if unsubscribe is not None:
                                unsubscribe()
            finally:
                if memory_saver:
                    cli_print(memory_saver.memory_stats(), json_events=bool(args.json_events))
        elif video_paths:
            with SeedVR2VideoRunLock():
                for video_path in video_paths:
                    for seed in args.seed:
                        _run_video_with_fresh_model(
                            parser=parser,
                            args=args,
                            resolved_model_path=resolved_model_path,
                            model_config=model_config,
                            video_path=video_path,
                            source_probe=video_probes[video_path],
                            plan=video_restore_plans[video_path],
                            output_pattern=output_pattern,
                            seed=seed,
                        )
    except StopImageGenerationException as exc:
        cli_print(str(exc), json_events=bool(args.json_events))


if __name__ == "__main__":
    main()
