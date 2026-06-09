import argparse
import gc
import json
import random
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mlx.core as mx
from tqdm import tqdm

from mflux.callbacks import ProgressEvent
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import boolean_flag_value, positive_float
from mflux.models.common.config import ModelConfig
from mflux.models.wan.variants import Wan2_2_TI2V
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError
from mflux.utils.prompt_util import PromptUtil

WAN_DEFAULT_WIDTH = Wan2_2_TI2V.RECOMMENDED_WIDTH
WAN_DEFAULT_HEIGHT = Wan2_2_TI2V.RECOMMENDED_HEIGHT
WAN_DEFAULT_FRAMES = Wan2_2_TI2V.RECOMMENDED_FRAMES
WAN_DEFAULT_FPS = Wan2_2_TI2V.RECOMMENDED_FPS
GENERIC_WAN_ALIASES = {"wan", "wan-video"}


def main() -> None:
    parser = _parser()
    provided_options = _provided_options(sys.argv[1:])
    args = parser.parse_args()
    provided_options.update(_apply_metadata_defaults(args))
    _validate_args(parser, args)
    _apply_seed_defaults(args)

    try:
        model_config, model_path = _resolve_model(args.model)
        _apply_model_defaults(args, model_config, provided_options)
        _apply_runtime_memory_options(args)
        model = Wan2_2_TI2V(
            model_config=model_config,
            quantize=args.quantize,
            model_path=model_path,
        )
        single_seed = len(args.seed) == 1
        release_inactive_denoiser = single_seed and bool(
            model_config.transformer_overrides.get("has_transformer_2", False)
        )
        release_denoisers_before_decode = args.low_ram and single_seed
        for seed in args.seed:
            progress = _WanCliProgress(enabled=args.progress)
            output_path = args.output.format(seed=seed)
            prompt = ""
            try:
                prompt = PromptUtil.read_prompt(args)
                video = model.generate_video(
                    seed=seed,
                    prompt=prompt,
                    width=args.width,
                    height=args.height,
                    num_frames=args.frames,
                    fps=args.fps,
                    guidance=args.guidance,
                    guidance_2=args.guidance_2,
                    flow_shift=args.flow_shift,
                    num_inference_steps=args.steps,
                    negative_prompt=args.negative_prompt,
                    image_path=args.image_path,
                    max_sequence_length=args.max_sequence_length,
                    progress_callback=progress if args.progress else None,
                    release_inactive_denoiser=release_inactive_denoiser,
                    release_denoisers_before_decode=release_denoisers_before_decode,
                    clear_cache_each_step=args.low_ram,
                    clear_cache_each_transformer_block=args.low_ram,
                    tensor_health_check_interval=args.tensor_health_check_interval,
                )
                print(f"Saving video to: {output_path}")
                _emit_cli_video_progress(progress, phase="save", video=video)
                saved_path = video.save(
                    path=output_path,
                    export_json_metadata=args.metadata,
                    overwrite=args.replace,
                )
                _emit_cli_video_progress(progress, phase="complete", video=video)
                print(f"Saved video to: {saved_path or output_path}")
                del video
                gc.collect()
                mx.clear_cache()
            except Exception as exc:
                _write_failure_manifest(output_path=output_path, args=args, seed=seed, prompt=prompt, error=exc)
                _emit_cli_failure_progress(
                    progress,
                    total_frames=args.frames,
                    total_steps=args.steps,
                    task="image-to-video" if args.image_path is not None else "text-to-video",
                )
                raise
            finally:
                progress.close()
    except (
        ModelConfigError,
        PromptFileReadError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
        NotImplementedError,
    ) as exc:
        print(exc)
        raise SystemExit(1) from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxgen-generate-wan",
        description="Generate a video using supported Wan2.2 models.",
    )
    parser.add_argument("--model", "-m", required=True, help="Wan model alias, Hugging Face repo, or local path.")
    parser.add_argument("--image-path", default=None, help="Input image for Wan image-to-video models.")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", type=str, help="Text prompt for video generation.")
    prompt_group.add_argument("--prompt-file", type=Path, help="Path to a text file containing the prompt.")
    parser.add_argument(
        "--negative-prompt",
        "--negative",
        dest="negative_prompt",
        type=str,
        default="",
        help="Negative prompt used when guidance > 1.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=WAN_DEFAULT_WIDTH,
        help=(
            "Video width. For text-to-video, adjusted up to a model-specific patch multiple. "
            "For image-to-video, used as a size target while preserving the input image aspect ratio."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=WAN_DEFAULT_HEIGHT,
        help=(
            "Video height. For text-to-video, adjusted up to a model-specific patch multiple. "
            "For image-to-video, used as a size target while preserving the input image aspect ratio."
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=WAN_DEFAULT_FRAMES,
        help="Number of frames. Adjusted to 4n + 1.",
    )
    parser.add_argument("--fps", type=int, default=WAN_DEFAULT_FPS, help="Output video frame rate.")
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps.")
    parser.add_argument("--guidance", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--flow-shift",
        type=positive_float,
        default=None,
        help=(
            "Wan flow-matching schedule shift. Defaults to the selected model config. "
            "Wan references recommend 5.0 for 720p-class TI2V-5B and 3.0 for 480p-class runs."
        ),
    )
    parser.add_argument(
        "--guidance-2",
        type=float,
        default=None,
        help=(
            "Optional low-noise guidance scale for Wan A14B transformer_2. "
            "When omitted with --guidance, follows --guidance; when both are omitted, uses model defaults."
        ),
    )
    parser.add_argument("--seed", "-s", type=int, default=None, nargs="+", help="One or more random seeds.")
    parser.add_argument("--auto-seeds", type=int, default=-1, help="Generate N random seeds.")
    parser.add_argument("--quantize", "-q", type=int, choices=ui_defaults.QUANTIZE_CHOICES, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=512, help="UMT5 prompt token length.")
    parser.add_argument("--metadata", action="store_true", help="Export video metadata as JSON.")
    parser.add_argument("--output", type=str, default="video.mp4", help='Output path. Default is "video.mp4".')
    parser.add_argument(
        "--progress",
        type=boolean_flag_value,
        nargs="?",
        const=True,
        default=True,
        help="Show denoise-step progress with the requested frame count as context. Default is true.",
    )
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument(
        "--replace",
        type=boolean_flag_value,
        nargs="?",
        const=True,
        default=True,
        help="Replace the target output when it already exists. Default is true.",
    )
    parser.add_argument("--no-replace", action="store_false", dest="replace")
    parser.add_argument("--config-from-metadata", "-C", type=Path, default=None)
    parser.add_argument(
        "--battery-percentage-stop-limit", "-B", type=int, default=ui_defaults.BATTERY_PERCENTAGE_STOP_LIMIT
    )
    parser.add_argument(
        "--low-ram",
        action="store_true",
        help=(
            "Reduce peak memory by clearing MLX cache between transformer blocks and denoise steps, "
            "then releasing denoisers before decode. May reduce throughput."
        ),
    )
    parser.add_argument(
        "--mlx-cache-limit-gb",
        type=positive_float,
        default=None,
        help="Limit MLX cache size in GB. With --low-ram, defaults to 1 GB when omitted.",
    )
    parser.add_argument(
        "--tensor-health-check-interval",
        type=_positive_int,
        default=None,
        help=(
            "Check Wan denoise latents for NaN/Inf every N steps. Disabled by default to preserve "
            "the normal MLX lazy execution path; use 1 for every-step diagnostics."
        ),
    )
    parser.add_argument(
        "--failure-diagnostics",
        action="store_true",
        help=(
            "Add MLX allocator memory and tensor-health details to the failure manifest. "
            "This does not serialize full video latents."
        ),
    )
    return parser


def _apply_metadata_defaults(args: argparse.Namespace) -> set[str]:
    provided_options = set()
    if args.config_from_metadata is None:
        return provided_options
    metadata = json.loads(args.config_from_metadata.read_text())
    if args.prompt is None and args.prompt_file is None:
        args.prompt = metadata.get("prompt")
    if args.negative_prompt == "":
        args.negative_prompt = metadata.get("negative_prompt") or ""
        if args.negative_prompt:
            provided_options.add("--negative-prompt")
    if args.seed is None and metadata.get("seed") is not None:
        args.seed = [int(metadata["seed"])]
        provided_options.add("--seed")
    if args.quantize is None:
        args.quantize = metadata.get("quantize")
        if args.quantize is not None:
            provided_options.add("--quantize")
    if args.image_path is None and metadata.get("image_path") is not None:
        args.image_path = metadata.get("image_path")
        provided_options.add("--image-path")
    for name in ("width", "height", "frames", "fps", "steps", "guidance", "guidance_2", "flow_shift"):
        value = metadata.get(name)
        if value is not None and getattr(args, name) == _parser().get_default(name):
            setattr(args, name, value)
            provided_options.add(f"--{name.replace('_', '-')}")
    return provided_options


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.prompt is None and args.prompt_file is None:
        parser.error("Either --prompt or --prompt-file is required, or provide prompt in --config-from-metadata.")
    if args.fps <= 0:
        parser.error("--fps must be greater than zero.")
    if args.steps <= 0:
        parser.error("--steps must be greater than zero.")
    if args.max_sequence_length <= 0:
        parser.error("--max-sequence-length must be greater than zero.")
    if args.image_path is not None and not Path(args.image_path).exists():
        parser.error(f"--image-path does not exist: {args.image_path}")


def _apply_seed_defaults(args: argparse.Namespace) -> None:
    if args.seed is None and args.auto_seeds > 0:
        args.seed = random.sample(range(int(1e7) + 1), args.auto_seeds)
    if args.seed is None:
        args.seed = [int(time.time())]
    if len(args.seed) > 1:
        output_path = Path(args.output)
        args.output = str(output_path.with_stem(output_path.stem + "_seed_{seed}"))


def _resolve_model(model: str) -> tuple[ModelConfig, str | None]:
    try:
        model_config = ModelConfig.from_name(model)
    except ModelConfigError as exc:
        if "wan" in model.lower():
            raise ModelConfigError(
                f"Cannot infer a supported Wan model config from {model}. "
                "Use an exact supported Wan repo or a local prepared folder whose name includes a specific Wan alias."
            ) from exc
        raise
    if _uses_only_generic_wan_alias(model=model, model_config=model_config):
        raise ModelConfigError(
            f"Cannot infer a supported Wan model config from {model}. "
            "Use an exact supported Wan repo or a local prepared folder whose name includes a specific Wan alias."
        )
    model_path = model if model_config.base_model is not None else None
    return model_config, model_path


def _uses_only_generic_wan_alias(model: str, model_config: ModelConfig) -> bool:
    if model_config.base_model is None:
        return False
    model_key = model.lower().replace("_", "-")
    specific_aliases = [alias for alias in model_config.aliases if alias not in GENERIC_WAN_ALIASES]
    return not any(alias.lower() in model_key for alias in specific_aliases)


def _apply_model_defaults(args: argparse.Namespace, model_config: ModelConfig, provided_options: set[str]) -> None:
    wan_config = model_config.transformer_overrides
    option_map = {
        "width": ("default_width", "--width"),
        "height": ("default_height", "--height"),
        "frames": ("default_frames", "--frames"),
        "fps": ("default_fps", "--fps"),
        "steps": ("default_steps", "--steps"),
        "guidance": ("default_guidance", "--guidance"),
        "flow_shift": ("flow_shift", "--flow-shift"),
    }
    for attr, (config_key, option_name) in option_map.items():
        if option_name in provided_options:
            continue
        value = wan_config.get(config_key)
        if value is not None:
            setattr(args, attr, value)
    if "--negative-prompt" not in provided_options and not args.negative_prompt:
        args.negative_prompt = wan_config.get("default_negative_prompt", "")
    if (
        "--guidance-2" not in provided_options
        and "--guidance" not in provided_options
        and wan_config.get("default_guidance_2") is not None
    ):
        args.guidance_2 = wan_config["default_guidance_2"]


def _apply_runtime_memory_options(args: argparse.Namespace) -> None:
    cache_limit_bytes = _resolve_cache_limit_bytes(args.mlx_cache_limit_gb)
    if args.low_ram and cache_limit_bytes is None:
        cache_limit_bytes = 1000**3
    if cache_limit_bytes is not None:
        mx.set_cache_limit(cache_limit_bytes)
        mx.clear_cache()
        mx.reset_peak_memory()


def _resolve_cache_limit_bytes(mlx_cache_limit_gb: float | None) -> int | None:
    if mlx_cache_limit_gb is None:
        return None
    return int(mlx_cache_limit_gb * (1000**3))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _emit_cli_video_progress(progress: "_WanCliProgress", *, phase: str, video) -> None:
    total_frames = getattr(video, "num_frames", 0)
    total_steps = getattr(video, "steps", 0)
    progress(
        ProgressEvent(
            phase=phase,
            frame=total_frames,
            total_frames=total_frames,
            step=total_steps,
            total_steps=total_steps,
            task=getattr(video, "task", None),
        )
    )


def _emit_cli_failure_progress(
    progress: "_WanCliProgress",
    *,
    total_frames: int,
    total_steps: int,
    task: str,
) -> None:
    progress(
        ProgressEvent(
            phase="failed",
            frame=0,
            total_frames=total_frames,
            step=0,
            total_steps=total_steps,
            task=task,
        )
    )


def _write_failure_manifest(
    *,
    output_path: str,
    args: argparse.Namespace,
    seed: int,
    prompt: str,
    error: BaseException,
) -> Path:
    failure_path = Path(output_path).with_suffix(".failure.json")
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_report = getattr(error, "report", None)
    manifest = {
        "created_at": datetime.now().isoformat(),
        "status": "failed",
        "error_type": error.__class__.__name__,
        "error": str(error),
        "tensor_health": asdict(tensor_report) if tensor_report is not None else None,
        "run": {
            "model": args.model,
            "task": "image-to-video" if args.image_path is not None else "text-to-video",
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": args.negative_prompt or None,
            "image_path": str(args.image_path) if args.image_path is not None else None,
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "steps": args.steps,
            "guidance": args.guidance,
            "guidance_2": args.guidance_2,
            "flow_shift": args.flow_shift,
            "fps": args.fps,
            "output": output_path,
            "low_ram": bool(args.low_ram),
            "tensor_health_check_interval": args.tensor_health_check_interval,
            "failure_diagnostics": bool(args.failure_diagnostics),
        },
    }
    if args.failure_diagnostics:
        manifest["runtime_diagnostics"] = _runtime_diagnostics()
    failure_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return failure_path


def _runtime_diagnostics() -> dict:
    def _mlx_memory(name: str):
        try:
            return int(getattr(mx, name)())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    return {
        "mlx_active_memory_bytes": _mlx_memory("get_active_memory"),
        "mlx_peak_memory_bytes": _mlx_memory("get_peak_memory"),
        "mlx_cache_memory_bytes": _mlx_memory("get_cache_memory"),
    }


def _provided_options(argv: list[str]) -> set[str]:
    provided = set()
    aliases = {
        "-m": "--model",
        "-s": "--seed",
        "-q": "--quantize",
        "-C": "--config-from-metadata",
        "-B": "--battery-percentage-stop-limit",
        "--negative": "--negative-prompt",
    }
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        provided.add(aliases.get(option, option))
    return provided


class _WanCliProgress:
    _lock_configured = False

    def __init__(self, enabled: bool):
        self.enabled = enabled
        if enabled:
            self._configure_thread_lock()
        self._bar: tqdm | None = None
        self._last_step = 0

    def __call__(self, event: ProgressEvent) -> None:
        if not self.enabled:
            return
        if self._bar is None:
            self._bar = tqdm(total=event.total_steps, desc="Denoising video", unit="step")
        delta = max(0, event.step - self._last_step)
        if delta:
            self._bar.update(delta)
            self._last_step = event.step
        self._bar.set_postfix_str(f"{event.phase}; {event.total_frames} frames")
        if event.phase in {"complete", "failed"}:
            self.close()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    @classmethod
    def _configure_thread_lock(cls) -> None:
        if not cls._lock_configured:
            tqdm.set_lock(threading.RLock())
            cls._lock_configured = True


if __name__ == "__main__":
    main()
