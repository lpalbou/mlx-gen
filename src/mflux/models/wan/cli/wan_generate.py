import argparse
import gc
import json
import sys
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mflux.callbacks import ProgressEvent
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.output_paths import normalize_output_template, resolve_output_path
from mflux.cli.parser.parsers import boolean_flag_value, image_strength_value, positive_float
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print
from mflux.cli.seed_values import resolve_seed_values
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.wan.variants import Wan2_2_TI2V, WanVace
from mflux.utils.dimension_resolver import CANVAS_POLICY_CHOICES, RESIZE_MODE_CHOICES, RESIZE_MODE_RESIZE
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError
from mflux.utils.prompt_util import PromptUtil
from mflux.utils.runtime_memory import RuntimeMemory

WAN_DEFAULT_WIDTH = Wan2_2_TI2V.RECOMMENDED_WIDTH
WAN_DEFAULT_HEIGHT = Wan2_2_TI2V.RECOMMENDED_HEIGHT
WAN_DEFAULT_FRAMES = Wan2_2_TI2V.RECOMMENDED_FRAMES
WAN_DEFAULT_FPS = Wan2_2_TI2V.RECOMMENDED_FPS
WAN_DEFAULT_VIDEO_STRENGTH = 0.8
GENERIC_WAN_ALIASES = {"wan", "wan-video"}


def main() -> None:
    parser = _parser()
    provided_options = _provided_options(sys.argv[1:])
    args = parser.parse_args()
    LoRALoader.set_debug_enabled(bool(getattr(args, "debug", False)))
    provided_options.update(_apply_metadata_defaults(args))
    _validate_args(parser, args)
    try:
        _apply_seed_defaults(args)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        model_config, model_path = _resolve_model(args.model)
        _apply_model_defaults(args, model_config, provided_options)
        _validate_model_runtime_args(args=args, model_config=model_config)
        _apply_runtime_memory_options(args)
        is_vace = bool(model_config.transformer_overrides.get("supports_vace", False))
        model_class = WanVace if is_vace else Wan2_2_TI2V
        model = model_class(
            model_config=model_config,
            quantize=args.quantize,
            model_path=model_path,
            lora_paths=args.lora_paths,
            lora_scales=args.lora_scales,
            lora_target_roles=args.lora_target_roles,
            keep_text_encoder_resident=args.keep_text_encoder,
            prompt_embed_disk_cache=not args.no_prompt_cache,
        )
        single_seed = len(args.seed) == 1
        release_denoisers_before_decode = args.low_ram and single_seed
        # Single-seed CLI runs keep the pre-0089 rule: release the inactive
        # expert regardless of quantization — the process exits after one
        # item, so no reload can ever be paid and bf16 checkpoints lose
        # nothing (cycle-2 review: the model's reload-gated auto default
        # silently dropped this release for bf16/runtime-quantized runs).
        # Multi-seed and Python-API callers use the model-owned default.
        release_inactive_denoiser_arg = args.release_inactive_denoiser
        if (
            release_inactive_denoiser_arg is None
            and single_seed
            and bool(model_config.transformer_overrides.get("has_transformer_2", False))
        ):
            release_inactive_denoiser_arg = True
        for seed in args.seed:
            progress = _WanCliProgress(enabled=args.progress and not args.json_events)
            output_path = resolve_output_path(args.output, overwrite=args.replace, seed=seed)
            events = CliRuntimeEventStream(
                enabled=bool(args.json_events),
                command="mlxgen generate",
                model=model_config.model_name,
                seed=seed,
            )
            events.set_output_path(output_path)
            prompt = ""
            try:
                prompt = PromptUtil.read_prompt(args)
                generate_kwargs = dict(
                    seed=seed,
                    prompt=prompt,
                    width=args.width,
                    height=args.height,
                    num_frames=args.frames,
                    fps=args.fps,
                    guidance=args.guidance,
                    guidance_2=args.guidance_2,
                    flow_shift=args.flow_shift,
                    solver=args.solver,
                    num_inference_steps=args.steps,
                    negative_prompt=args.negative_prompt,
                    image_path=args.image_path,
                    video_path=args.video_path,
                    video_strength=args.video_strength,
                    video_mask_path=args.video_mask_path,
                    canvas_policy=args.canvas_policy,
                    resize_mode=args.resize_mode,
                    max_sequence_length=args.max_sequence_length,
                    progress_callback=events.handle_progress
                    if events.enabled
                    else (progress if args.progress else None),
                    # Explicit user intent > single-seed CLI rule > None
                    # (= model-owned auto default, 0089 e4).
                    release_inactive_denoiser=release_inactive_denoiser_arg,
                    release_denoisers_before_decode=release_denoisers_before_decode,
                    clear_cache_each_step=args.low_ram,
                    clear_cache_each_transformer_block=args.low_ram,
                    tensor_health_check_interval=args.tensor_health_check_interval,
                    compile_transformer=args.compile_transformer,
                )
                if is_vace:
                    generate_kwargs["reference_image_paths"] = args.reference_image_paths
                    if args.conditioning_scale is not None:
                        generate_kwargs["conditioning_scale"] = args.conditioning_scale
                    if args.vace_masked_region is not None:
                        generate_kwargs["masked_region_mode"] = args.vace_masked_region
                video = model.generate_video(**generate_kwargs)
                cli_print(f"Saving video to: {output_path}", json_events=bool(args.json_events))
                # The save event carries the output's own fps/frames/dimensions
                # so hosts can skip a metadata probe decode entirely (0087).
                events.emit_save(
                    task=getattr(video, "task", None),
                    health_check="skipped" if args.no_validate_health else None,
                    fps=getattr(video, "fps", None),
                    width=getattr(video, "width", None),
                    height=getattr(video, "height", None),
                    total_frames=getattr(video, "num_frames", None),
                )
                _emit_cli_video_progress(progress, phase="save", video=video)
                saved_path = video.save(
                    path=output_path,
                    export_json_metadata=args.metadata,
                    overwrite=True,
                    validate_health=not args.no_validate_health,
                )
                events.set_output_path(saved_path or output_path)
                _emit_cli_video_progress(progress, phase="complete", video=video)
                events.emit_complete(task=getattr(video, "task", None))
                cli_print(f"Saved video to: {saved_path or output_path}", json_events=bool(args.json_events))
                del video
                gc.collect()
                mx.clear_cache()
            except Exception as exc:
                failure_path = _write_failure_manifest(
                    output_path=output_path, args=args, seed=seed, prompt=prompt, error=exc
                )
                _emit_cli_failure_progress(
                    progress,
                    total_frames=args.frames,
                    total_steps=args.steps,
                    task=_requested_task(args),
                )
                events.emit_failed(
                    task=_requested_task(args),
                    error=exc,
                    diagnostics_path=failure_path,
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
        cli_print(str(exc), json_events=bool(getattr(args, "json_events", False)), error=True)
        raise SystemExit(1) from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxgen-generate-wan",
        description="Generate a video using supported Wan2.2 models.",
    )
    parser.add_argument("--model", "-m", required=True, help="Wan model alias, Hugging Face repo, or local path.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for internal generation details such as LoRA fusion targets.",
    )
    parser.add_argument("--image-path", default=None, help="Input image for Wan image-to-video models.")
    parser.add_argument(
        "--video-path",
        default=None,
        help="Input source video for Wan video-to-video model configs when explicitly supported.",
    )
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
            "Video width. For text-to-video and video-to-video, adjusted up to a model-specific patch multiple. "
            "For image-to-video, used as a size target while preserving the input image aspect ratio."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=WAN_DEFAULT_HEIGHT,
        help=(
            "Video height. For text-to-video and video-to-video, adjusted up to a model-specific patch multiple. "
            "For image-to-video, used as a size target while preserving the input image aspect ratio."
        ),
    )
    parser.add_argument(
        "--canvas-policy",
        choices=CANVAS_POLICY_CHOICES,
        default=None,
        help=(
            "How the output canvas is resolved from a source input. Default keeps each route's behavior: "
            "image-to-video resolves a source-ratio canvas near the requested size; text/video-to-video honor "
            "the requested (multiple-adjusted) canvas. Pass exact-resize to honor the requested canvas on "
            "image-to-video, or source-aspect to derive a source-ratio canvas on video-to-video."
        ),
    )
    parser.add_argument(
        "--resize-mode",
        choices=RESIZE_MODE_CHOICES,
        default=RESIZE_MODE_RESIZE,
        help=(
            "How source pixels (image-to-video first frame, video-to-video frames, and their masks) map onto "
            "the canvas: resize stretches to fill (default), crop center-crops without distortion, pad "
            "letterboxes the full source onto the canvas without distortion (black bars)."
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
    parser.add_argument(
        "--solver",
        choices=("unipc", "euler"),
        default=None,
        help=(
            "Wan denoiser solver. Defaults to the model profile. LightX2V 4-step Wan LoRAs are designed around "
            "the euler path. Public Wan video-to-video currently requires unipc."
        ),
    )
    parser.add_argument("--guidance", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--video-strength",
        type=image_strength_value,
        default=None,
        help=(
            "Denoising strength in (0, 1] for plain prompt-guided Wan video-to-video. "
            f"Default for that route is {WAN_DEFAULT_VIDEO_STRENGTH}."
        ),
    )
    parser.add_argument(
        "--video-mask-path",
        default=None,
        help=(
            "Mask for masked Wan video-to-video. White marks the region the model may change; "
            "black regions are preserved from the source video. Takes a static image, or - on Wan "
            "VACE models - an animated mask video whose per-frame trajectory carries the object's "
            "motion into the conditioning. Requires --video-path."
        ),
    )
    parser.add_argument(
        "--reference-image",
        dest="reference_image_paths",
        action="append",
        default=None,
        help=(
            "Reference image for Wan VACE models: injects the pictured subject/object into the "
            "generation. Repeatable. Only supported on VACE model configs."
        ),
    )
    parser.add_argument(
        "--conditioning-scale",
        type=float,
        default=None,
        help="Wan VACE conditioning strength applied to every VACE layer. Default: 1.0.",
    )
    parser.add_argument(
        "--vace-masked-region",
        choices=["generate", "repaint"],
        default=None,
        help=(
            "Wan VACE only: what the model sees inside the white mask region. 'generate' "
            "(default) gray-fills it per the official VACE inpainting convention, freeing the "
            "model to synthesize new structure; 'repaint' keeps the source content as "
            "conditioning, preserving structure and changing style/color only."
        ),
    )
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
    parser.add_argument("--auto-seeds", type=int, default=-1, help="Generate N random seeds between 0 and 10,000,000.")
    parser.add_argument("--quantize", "-q", type=int, choices=ui_defaults.QUANTIZE_CHOICES, default=None)
    parser.add_argument(
        "--lora-paths",
        type=str,
        nargs="*",
        default=None,
        help="Wan LoRA paths: local files, Hugging Face repos, or repo:file.safetensors entries.",
    )
    parser.add_argument(
        "--lora-scales",
        type=float,
        nargs="*",
        default=None,
        help="Per-file Wan LoRA scaling factors. Must match --lora-paths when provided.",
    )
    parser.add_argument(
        "--lora-target-roles",
        nargs="*",
        default=None,
        help=(
            "Per-file Wan LoRA target roles. Use 'transformer' for TI2V-5B. Use "
            "'high_noise_transformer' and/or 'low_noise_transformer' for Wan A14B."
        ),
    )
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
        "--json-events",
        action="store_true",
        help="Emit machine-readable JSONL runtime events to stdout and move human CLI text to stderr.",
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
        "--no-validate-health",
        action="store_true",
        help=(
            "Skip the post-save full-file health re-decode. For embedded hosts that probe the saved "
            "file themselves; the skip is recorded as health_check=skipped in metadata and the save event."
        ),
    )
    parser.add_argument(
        "--keep-text-encoder",
        action="store_true",
        help=(
            "Keep the UMT5 text encoder resident between generations in this process instead of "
            "loading and releasing it per prompt encode. Trades ~11 GB resident RAM for skipping "
            "the reload on every new prompt (useful for hosts that chain scene generations)."
        ),
    )
    parser.add_argument(
        "--no-prompt-cache",
        action="store_true",
        help=(
            "Disable the exact disk cache for UMT5 prompt embeds. By default, identical "
            "(model, tokenizer, prompt, length, precision) encodes are served from a small "
            "safetensors cache instead of reloading the text encoder."
        ),
    )
    parser.add_argument(
        "--release-inactive-denoiser",
        dest="release_inactive_denoiser",
        action="store_true",
        default=None,
        help=(
            "Force releasing the Wan A14B high-noise transformer (~14 GB) after its per-item "
            "denoise phase and reloading it for the next item. The default is automatic: on for "
            "dual-transformer models loaded from disk-prequantized packages (reload is a cheap "
            "mmap read), off when quantizing at load time (each reload would re-quantize 14B)."
        ),
    )
    parser.add_argument(
        "--no-release-inactive-denoiser",
        dest="release_inactive_denoiser",
        action="store_false",
        default=None,
        help="Keep both Wan A14B transformers resident for the whole run (~28 GB).",
    )
    parser.add_argument(
        "--compile-transformer",
        action="store_true",
        help=(
            "Opt-in: run the Wan denoiser(s) as compiled MLX graphs (~2-6%% per step). "
            "Output is NOT bit-identical to the eager default (compiled kernels differ by ~5e-4). "
            "Ignored with a printed notice when --low-ram, --tensor-health-check-interval, or "
            "MFLUX_WAN_BLOCK_HEALTH require the eager path."
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
    if args.seed is None and args.auto_seeds == -1 and metadata.get("seed") is not None:
        args.seed = [int(metadata["seed"])]
        provided_options.add("--seed")
    if args.quantize is None:
        args.quantize = metadata.get("quantize")
        if args.quantize is not None:
            provided_options.add("--quantize")
    if args.image_path is None and metadata.get("image_path") is not None:
        args.image_path = metadata.get("image_path")
        provided_options.add("--image-path")
    if args.video_path is None and metadata.get("video_path") is not None:
        args.video_path = metadata.get("video_path")
        provided_options.add("--video-path")
    if args.lora_paths is None and metadata.get("lora_paths") is not None:
        args.lora_paths = metadata.get("lora_paths")
        provided_options.add("--lora-paths")
    if args.lora_scales is None and metadata.get("lora_scales") is not None:
        args.lora_scales = metadata.get("lora_scales")
        provided_options.add("--lora-scales")
    if args.lora_target_roles is None and metadata.get("lora_target_roles") is not None:
        args.lora_target_roles = metadata.get("lora_target_roles")
        provided_options.add("--lora-target-roles")
    for name in (
        "width",
        "height",
        "frames",
        "fps",
        "steps",
        "guidance",
        "guidance_2",
        "flow_shift",
        "solver",
        "canvas_policy",
        "resize_mode",
    ):
        value = metadata.get(name)
        if value is not None and getattr(args, name) == _parser().get_default(name):
            setattr(args, name, value)
            provided_options.add(f"--{name.replace('_', '-')}")
    if args.video_strength is None and metadata.get("video_strength") is not None:
        args.video_strength = metadata.get("video_strength")
    if args.video_mask_path is None and metadata.get("video_mask_path") is not None:
        args.video_mask_path = metadata.get("video_mask_path")
        provided_options.add("--video-mask-path")
    if args.reference_image_paths is None and metadata.get("reference_image_paths"):
        args.reference_image_paths = list(metadata.get("reference_image_paths"))
        provided_options.add("--reference-image")
    if args.conditioning_scale is None and metadata.get("conditioning_scale") is not None:
        args.conditioning_scale = float(metadata.get("conditioning_scale"))
        provided_options.add("--conditioning-scale")
    if args.vace_masked_region is None and metadata.get("masked_region_mode") is not None:
        args.vace_masked_region = str(metadata.get("masked_region_mode"))
        provided_options.add("--vace-masked-region")
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
    if args.image_path is not None and args.video_path is not None:
        parser.error("--image-path and --video-path cannot be used together.")
    if args.image_path is not None and not Path(args.image_path).exists():
        parser.error(f"--image-path does not exist: {args.image_path}")
    if args.video_path is not None and not Path(args.video_path).exists():
        parser.error(f"--video-path does not exist: {args.video_path}")
    if args.video_strength is not None and args.video_path is None:
        parser.error("--video-strength requires --video-path.")
    if args.video_mask_path is not None and args.video_path is None:
        parser.error("--video-mask-path requires --video-path.")
    if args.video_mask_path is not None and not Path(args.video_mask_path).exists():
        parser.error(f"--video-mask-path does not exist: {args.video_mask_path}")
    if args.lora_scales is not None and not args.lora_paths:
        parser.error("--lora-scales requires --lora-paths.")
    if args.lora_target_roles is not None and not args.lora_paths:
        parser.error("--lora-target-roles requires --lora-paths.")


def _validate_model_runtime_args(*, args: argparse.Namespace, model_config: ModelConfig) -> None:
    is_vace = bool(model_config.transformer_overrides.get("supports_vace", False))
    if args.image_path is not None and not bool(
        model_config.transformer_overrides.get("supports_image_to_video", True)
    ):
        raise ValueError(f"{model_config.model_name} does not support image-to-video input.")
    if args.video_path is not None and not bool(
        model_config.transformer_overrides.get("supports_video_to_video", False)
    ):
        raise ValueError(f"{model_config.model_name} does not support video-to-video input.")
    if args.video_path is not None and args.solver is not None and str(args.solver).strip().lower() != "unipc":
        raise ValueError("Wan video-to-video currently requires --solver unipc.")
    if args.reference_image_paths and not is_vace:
        raise ValueError(
            f"--reference-image requires a Wan VACE model config; {model_config.model_name} does not support it."
        )
    if args.conditioning_scale is not None and not is_vace:
        raise ValueError(
            f"--conditioning-scale requires a Wan VACE model config; {model_config.model_name} does not support it."
        )
    if args.vace_masked_region is not None and not is_vace:
        raise ValueError(
            f"--vace-masked-region requires a Wan VACE model config; {model_config.model_name} does not support it."
        )
    if is_vace and args.video_strength is not None:
        raise ValueError(
            "--video-strength is not supported on Wan VACE models: VACE has no SDEdit warm start. "
            "Use --video-mask-path and --conditioning-scale to control the edit."
        )
    if args.reference_image_paths:
        for path in args.reference_image_paths:
            _probe_reference_image(image_path=path)
    if args.video_path is not None:
        _probe_source_video(video_path=args.video_path, requested_frames=args.frames, requested_fps=args.fps)
    if args.video_mask_path is not None:
        _probe_video_mask(mask_path=args.video_mask_path, is_vace=is_vace)


def _probe_reference_image(*, image_path: str) -> None:
    from PIL import Image

    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"--reference-image is not a readable image: {image_path} ({exc})") from exc


def _probe_video_mask(*, mask_path: str, is_vace: bool = False) -> None:
    # Fail on unreadable or all-black masks before the multi-minute model weight load.
    from PIL import Image

    from mflux.models.wan.variants.wan_vace import WanVace
    from mflux.utils.video_util import VideoUtil

    if WanVace._is_video_mask(mask_path):
        if not is_vace:
            raise ValueError(
                "Animated video masks for --video-mask-path are only supported on Wan VACE models; "
                "the masked A14B video-to-video route takes one static image mask."
            )
        try:
            clip = VideoUtil.inspect_video(mask_path)
        except Exception as exc:
            raise ValueError(f"--video-mask-path is not a readable video: {mask_path} ({exc})") from exc
        if not (clip.source_frame_count or 0):
            raise ValueError(f"--video-mask-path has no frames: {mask_path}")
        return
    try:
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"--video-mask-path is not a readable image: {mask_path} ({exc})") from exc
    if not bool((mask >= 128).any()):
        raise ValueError(
            f"--video-mask-path has no editable (white) region: {mask_path}. "
            "White marks the region the model may change; an all-black mask would edit nothing."
        )


def _probe_source_video(
    *, video_path: str, requested_frames: int | None, requested_fps: int | float | None = None
) -> None:
    # Fail on unreadable/short sources before the multi-minute model weight load.
    from mflux.utils.video_util import VideoUtil

    try:
        source_info = VideoUtil.inspect_video(video_path)
    except Exception as exc:
        raise ValueError(f"--video-path is not a readable video: {video_path} ({exc})") from exc
    if requested_frames is None:
        return
    source_duration = getattr(source_info, "source_duration_seconds", None)
    source_frame_count = source_info.source_frame_count
    if requested_fps and source_duration is not None:
        # Resampling samples the output timeline, so the requirement is duration-based. One
        # target-frame period of slack absorbs the fps filter's round=near boundary behavior.
        needed_seconds = int(requested_frames) / float(requested_fps)
        slack = 1.0 / float(requested_fps)
        if source_duration + slack < needed_seconds:
            raise ValueError(
                f"Wan video-to-video needs {needed_seconds:.2f}s of source video "
                f"({requested_frames} frames at {float(requested_fps):.3g} fps), "
                f"but {video_path} is only {float(source_duration):.2f}s long."
            )
    elif source_frame_count is not None and source_frame_count < int(requested_frames):
        raise ValueError(
            f"Wan video-to-video requires at least {requested_frames} source frames, "
            f"but {video_path} only has {source_frame_count}."
        )


def _apply_seed_defaults(args: argparse.Namespace) -> None:
    args.seed = resolve_seed_values(
        seed_values=args.seed,
        auto_seeds=args.auto_seeds,
    )
    if len(args.seed) > 1:
        args.output = normalize_output_template(args.output, include_seed=True)


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
    if "--solver" not in provided_options and args.solver is None:
        args.solver = wan_config.get("default_solver", "unipc")
    if (
        "--video-strength" not in provided_options
        and args.video_path is not None
        and args.video_strength is None
        and not wan_config.get("supports_vace", False)
    ):
        # VACE has no SDEdit warm start; injecting a strength default would be false metadata.
        args.video_strength = WAN_DEFAULT_VIDEO_STRENGTH


def _apply_runtime_memory_options(args: argparse.Namespace) -> None:
    RuntimeMemory.apply_mlx_cache_limit(args.mlx_cache_limit_gb, low_ram=args.low_ram)


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
    output_path: str | Path,
    args: argparse.Namespace,
    seed: int,
    prompt: str,
    error: BaseException,
) -> Path:
    resolved_output_path = str(output_path)
    failure_path = Path(resolved_output_path).with_suffix(".failure.json")
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
            "task": _requested_task(args),
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": args.negative_prompt or None,
            "image_path": str(args.image_path) if args.image_path is not None else None,
            "video_path": str(args.video_path) if args.video_path is not None else None,
            "video_strength": args.video_strength,
            "video_mask_path": str(args.video_mask_path) if args.video_mask_path is not None else None,
            "reference_image_paths": [str(path) for path in args.reference_image_paths]
            if args.reference_image_paths
            else None,
            "conditioning_scale": args.conditioning_scale,
            "vace_masked_region": args.vace_masked_region,
            "width": args.width,
            "height": args.height,
            "canvas_policy": args.canvas_policy,
            "resize_mode": args.resize_mode,
            "frames": args.frames,
            "steps": args.steps,
            "guidance": args.guidance,
            "guidance_2": args.guidance_2,
            "flow_shift": args.flow_shift,
            "solver": args.solver,
            "fps": args.fps,
            "output": resolved_output_path,
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
    return RuntimeMemory.snapshot("wan-failure", synchronize=True).to_metadata()


def _requested_task(args: argparse.Namespace) -> str:
    if args.video_path is not None:
        return "video-to-video"
    if args.image_path is not None:
        return "image-to-video"
    return "text-to-video"


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
