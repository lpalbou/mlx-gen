import fcntl
import gc
import json
import sys
import traceback
from copy import copy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import mlx.core as mx

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.output_paths import format_output_template, normalize_output_template, resolve_output_path
from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print, emit_cli_failure_event_for_argv
from mflux.models.common.cli.restore_dispatch import resolve_restore_family
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.download_policy import DownloadRequiredError, is_huggingface_repo_id
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.seedvr2.latent_creator.seedvr2_latent_creator import SeedVR2LatentCreator
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.models.seedvr2.variants.upscale.seedvr2_video_restore_plan import SeedVR2VideoRestorePlan
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.runtime_memory import RuntimeMemory
from mflux.utils.scale_factor import ScaleFactor

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

DEFAULT_SEEDVR2_VIDEO_CACHE_LIMIT_GB = SeedVR2VideoRestorePlan.CACHE_LIMIT_BYTES / 1000**3

# The options SeedVR2 cannot honour for a given input kind, as data rather than as an
# ad-hoc if-chain. These are the refusals that state a *capability* of the route -- the
# image route has no clip window, and the video route has no VAE tiling -- as opposed to
# the numeric argument-shape checks below them in `main`, which validate the shape of a
# value the route does accept. The restoration capability record is the intended home for
# these strings; keeping them in one ordered mapping is what lets them move there verbatim
# without re-typing, and keeps `mlxgen capabilities` and this parser quoting one text.
#
# Unlike SwiftVR, whose refusals are unconditional and therefore family-level, every
# SeedVR2 refusal is conditional on the input kind, so the mapping is keyed by input kind
# first. Insertion order is the reporting order: a command that breaks several rules still
# reports the same first error it reported before this mapping existed.
SEEDVR2_UNSUPPORTED_OPTIONS: dict[str, dict[str, str]] = {
    "image": {
        "--start-seconds/--max-frames": ("--start-seconds and --max-frames are only supported with --video-path."),
    },
    "video": {
        "--vae-tiling": (
            "--vae-tiling is not supported for SeedVR2 video restore. Use --low-ram and temporal chunking instead."
        ),
    },
}

# How each rule above is detected on the parsed namespace. Detection is argparse's
# business and stays here; the message text is the contract's. Keyed by the same rule id,
# so a rule cannot be declared without a detector or detected without a message.
_SEEDVR2_OPTION_DETECTORS: dict[str, Callable[[object], bool]] = {
    "--start-seconds/--max-frames": lambda args: args.start_seconds != 0.0 or args.max_frames is not None,
    "--vae-tiling": lambda args: bool(args.vae_tiling),
}

# Options whose presence changes the plan. Whether one was given is read from what argparse
# parsed, never from the argv strings, so `--opt value`, `--opt=value` and an unambiguous
# abbreviation agree.
_SEEDVR2_GIVEN_OPTION_DESTS = ("resolution", "temporal_chunk_size", "temporal_chunk_overlap")


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
                "Another video restore is already running in mlx-gen. "
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


def _reject_unsupported_options(parser, args, *, input_kind: str) -> None:
    """Refuse the options this family cannot honour for ``input_kind``.

    Reports the first violated rule in declaration order, which is the order the
    equivalent if-chain reported before these rules became data.
    """
    for rule_id, message in SEEDVR2_UNSUPPORTED_OPTIONS[input_kind].items():
        if _SEEDVR2_OPTION_DETECTORS[rule_id](args):
            parser.error(message)


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
    low_ram_effective: bool,
    cache_limit_gb: float | None,
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
                    "restore_plan": {
                        # The limit in force is the CLI's, stated once.
                        **{key: value for key, value in asdict(plan).items() if key != "cache_limit_bytes"},
                        "low_ram_effective": low_ram_effective,
                        "cache_limit_gb": cache_limit_gb,
                    },
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
    plan: SeedVR2VideoRestorePlan,
    *,
    low_ram: bool,
    cache_limit_gb: float | None,
    json_events: bool,
) -> None:
    cli_print(
        "SeedVR2 video preflight: "
        f"model={plan.variant} "
        f"source={plan.source_width}x{plan.source_height} "
        f"target={plan.target_width}x{plan.target_height} "
        f"frames={plan.requested_frames} "
        f"mode={plan.restore_mode} "
        f"reason={plan.route_reason} "
        f"low_ram={low_ram} "
        f"cache_limit_gb={cache_limit_gb or 'none'} "
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
        _print_seedvr2_video_preflight(
            video_path,
            plan,
            low_ram=args.low_ram,
            cache_limit_gb=args.mlx_cache_limit_gb,
            json_events=bool(args.json_events),
        )
        if plan.audio_present:
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
            "low_ram_effective": bool(args.low_ram),
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
                resolution=plan.resolution,
                softness=args.softness,
                start_seconds=plan.start_seconds,
                max_frames=plan.max_frames,
                output_path=output_path,
                export_json_metadata=args.metadata,
                overwrite=True,
                temporal_chunk_size=plan.effective_chunk_size,
                temporal_chunk_overlap=plan.effective_chunk_overlap,
                color_correction_mode=args.color_correction,
                drop_audio=args.drop_audio,
                restore_metadata=runtime_metadata,
                enforce_memory_budget=plan.enforce_memory_budget,
                validate_health=not args.no_validate_health,
            )
        except Exception as exc:
            failure_path = _write_seedvr2_failure_manifest(
                output_path=output_path,
                video_path=video_path,
                model=args.model,
                seed=seed,
                resolution=plan.resolution,
                softness=args.softness,
                color_correction_mode=args.color_correction,
                drop_audio=args.drop_audio,
                start_seconds=args.start_seconds,
                max_frames=args.max_frames,
                plan=plan,
                low_ram_effective=bool(args.low_ram),
                cache_limit_gb=args.mlx_cache_limit_gb,
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
        memory_saver.keep_transformer = plan.keep_transformer
    try:
        _run_seedvr2_video_restore(
            model=model,
            args=args,
            video_path=video_path,
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
    parser = CommandLineParser(
        description=(
            "Restore or upscale an image or video. SeedVR2 provides diffusion-based "
            "super-resolution with scaling; SwiftVR provides one-step restoration at the "
            "source resolution."
        )
    )
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    for action in parser._actions:
        if action.dest == "model":
            action.help = (
                "Restoration model handle. SeedVR2: seedvr2, seedvr2-3b, seedvr2-7b, seedvr2-7b-sharp, an "
                "official SeedVR2 Hugging Face repo, or an AbstractFramework SeedVR2 package. "
                "SwiftVR: swiftvr or swiftvr-5b. A local path also works."
            )
            break
    parser.add_seedvr2_upscale_arguments()
    parser.add_output_arguments()
    # These parse to None when omitted; the declared defaults (the ones --help states) are
    # applied right after, in this one place.
    declared_defaults = {dest: parser.get_default(dest) for dest in _SEEDVR2_GIVEN_OPTION_DESTS}
    parser.set_defaults(**dict.fromkeys(declared_defaults))
    args = parser.parse_args()
    given_options = {dest for dest in declared_defaults if getattr(args, dest) is not None}
    for dest, default in declared_defaults.items():
        if dest not in given_options:
            setattr(args, dest, default)

    image_paths = _expand_image_paths(args.image_path) if args.image_path else []
    video_paths = _expand_video_paths(args.video_path) if args.video_path else []
    if not image_paths and not video_paths:
        cli_print("No images or videos to upscale.", json_events=bool(args.json_events))
        return
    # `--image-path` and `--video-path` are a required mutually exclusive group, so
    # exactly one of these runs, and it runs in the same position the two inlined rules
    # used to occupy.
    if image_paths:
        _reject_unsupported_options(parser, args, input_kind="image")
    if video_paths:
        _reject_unsupported_options(parser, args, input_kind="video")
    # Below here the checks validate the *shape* of a value the route does accept, which
    # is argument validation rather than a capability of the family, so they stay inline.
    # The plan owns this rule; it is checked here too so a bad shape is reported before
    # anything is printed or probed, for image inputs as well, as it always was.
    try:
        SeedVR2VideoRestorePlan.validate_temporal_chunk_options(args.temporal_chunk_size, args.temporal_chunk_overlap)
    except ValueError as exc:
        parser.error(str(exc))
    safe_video_mode = bool(video_paths) and not args.force_unsafe_video_memory
    if video_paths and not args.low_ram:
        cli_print("SeedVR2 video mode: enabling --low-ram automatically.", json_events=bool(args.json_events))
        args.low_ram = True

    # `mlxgen upscale` already picked this main() by family, but this entry point is also
    # reachable directly as `mflux-upscale-seedvr2`. Resolving through the shared
    # dispatcher keeps both paths on one classifier and lets an unknown handle name both
    # restoration families. `_resolve_seedvr2_model` is still what resolves every SeedVR2
    # handle and still what raises for an unknown one.
    try:
        route = resolve_restore_family(args.model, args.model_path)
        if route.family != "seedvr2":
            raise ValueError(
                f"{args.model!r} names the {route.family} model family, which this entry point cannot "
                "run. Use `mlxgen upscale` instead, which dispatches by --model."
            )
        model_config, resolved_model_path = route.model_config, route.model_path
    except ValueError as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    video_restore_plans: dict[Path, SeedVR2VideoRestorePlan] = {}
    try:
        if video_paths:
            if args.mlx_cache_limit_gb is None:
                args.mlx_cache_limit_gb = DEFAULT_SEEDVR2_VIDEO_CACHE_LIMIT_GB
            elif safe_video_mode and (
                # -1 (explicit unlimited, 0094) also exceeds the certified safe
                # profile bound; the clamp notice below keeps the override visible.
                args.mlx_cache_limit_gb < 0 or args.mlx_cache_limit_gb > DEFAULT_SEEDVR2_VIDEO_CACHE_LIMIT_GB
            ):
                cli_print(
                    "SeedVR2 safe video mode: clamping --mlx-cache-limit-gb to "
                    f"{DEFAULT_SEEDVR2_VIDEO_CACHE_LIMIT_GB:g}.",
                    json_events=bool(args.json_events),
                )
                args.mlx_cache_limit_gb = DEFAULT_SEEDVR2_VIDEO_CACHE_LIMIT_GB
            for video_path in video_paths:
                # Options the user did not type go in as None, so the plan applies the profile's
                # defaults (1x, 49/16). A refusal is a ValueError carrying the first warning.
                video_restore_plans[video_path] = SeedVR2VideoRestorePlan.for_video(
                    video_path,
                    model_config=model_config,
                    model_path=resolved_model_path,
                    start_seconds=args.start_seconds,
                    max_frames=args.max_frames,
                    resolution=args.resolution if "resolution" in given_options else None,
                    temporal_chunk_size=args.temporal_chunk_size if "temporal_chunk_size" in given_options else None,
                    temporal_chunk_overlap=(
                        args.temporal_chunk_overlap if "temporal_chunk_overlap" in given_options else None
                    ),
                    force_unsafe_memory_profile=args.force_unsafe_video_memory,
                )
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
                                num_inference_steps=args.steps,
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
                            plan=video_restore_plans[video_path],
                            output_pattern=output_pattern,
                            seed=seed,
                        )
    except StopImageGenerationException as exc:
        cli_print(str(exc), json_events=bool(args.json_events))


if __name__ == "__main__":
    main()
