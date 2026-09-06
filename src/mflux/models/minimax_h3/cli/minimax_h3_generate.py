import argparse
import gc
import threading
from pathlib import Path

import mlx.core as mx
from tqdm import tqdm

from mflux.callbacks import ProgressEvent
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.output_paths import normalize_output_template, resolve_output_path
from mflux.cli.parser.parsers import boolean_flag_value, cache_limit_gb_value
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print
from mflux.cli.seed_values import resolve_seed_values
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.minimax_h3.variants.minimax_h3 import MiniMaxH3
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError
from mflux.utils.prompt_util import PromptUtil
from mflux.utils.runtime_memory import RuntimeMemory


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    LoRALoader.set_debug_enabled(bool(args.debug))
    try:
        args.seed = resolve_seed_values(seed_values=args.seed, auto_seeds=args.auto_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if len(args.seed) > 1:
        args.output = normalize_output_template(args.output, include_seed=True)
    RuntimeMemory.apply_mlx_cache_limit(args.mlx_cache_limit_gb)

    try:
        model_config, model_path = _resolve_model(args.model, args.base_model)
        if model_path is not None:
            cli_print(
                f"Loading {model_path} as {model_config.aliases[0]} (base model {model_config.base_model}).",
                json_events=bool(args.json_events),
            )
        model = MiniMaxH3(
            model_config=model_config,
            quantize=args.quantize,
            model_path=model_path,
            lora_paths=args.lora_paths,
            lora_scales=args.lora_scales,
        )
        for seed in args.seed:
            progress = _CliProgress(enabled=args.progress and not args.json_events)
            output_path = resolve_output_path(args.output, overwrite=args.replace, seed=seed)
            events = CliRuntimeEventStream(
                enabled=bool(args.json_events), command="mlxgen generate", model=model_config.model_name, seed=seed
            )
            events.set_output_path(output_path)
            prompt = ""
            try:
                prompt = PromptUtil.read_prompt(args)
                video = model.generate_video(
                    seed=seed,
                    prompt=prompt,
                    soundscape=args.soundscape,
                    music=args.music,
                    width=args.width,
                    height=args.height,
                    num_frames=args.frames,
                    num_inference_steps=args.steps,
                    video_shift=args.video_shift,
                    audio_shift=args.audio_shift,
                    image_path=args.image_path,
                    generate_audio=not args.no_audio,
                    release_text_encoder=args.release_text_encoder,
                    progress_callback=events.handle_progress
                    if events.enabled
                    else (progress if args.progress else None),
                )
                cli_print(f"Saving video to: {output_path}", json_events=bool(args.json_events))
                events.emit_save(
                    task=video.task,
                    health_check="skipped" if args.no_validate_health else None,
                    fps=video.fps,
                    width=video.width,
                    height=video.height,
                    total_frames=video.num_frames,
                )
                saved_path = video.save(
                    path=output_path,
                    export_json_metadata=args.metadata,
                    overwrite=True,
                    validate_health=not args.no_validate_health,
                )
                events.set_output_path(saved_path or output_path)
                events.emit_complete(task=video.task)
                cli_print(f"Saved video to: {saved_path or output_path}", json_events=bool(args.json_events))
                del video
                gc.collect()
                mx.clear_cache()
            except Exception as exc:
                events.emit_failed(task="text-to-video", error=exc, diagnostics_path=None)
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


def _resolve_model(model: str, base_model: str | None = None) -> tuple[ModelConfig, str | None]:
    """Catalog entries resolve by alias; a repo id or local package resolves by `--base-model` (or its name)."""
    model_config = ModelConfig.from_name(model, base_model=base_model)
    is_catalog_entry = model.lower() in {alias.lower() for alias in model_config.aliases}
    model_path = None if is_catalog_entry else model
    return model_config, model_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxgen-generate-minimax-h3",
        description=(
            "Generate a video with a synchronized stereo soundtrack using MiniMax-H3. Prompts follow the "
            "H3 structured format (integrated_multimodal_description / overall_soundscape / non_diegetic_music); "
            "a plain description is wrapped automatically."
        ),
    )
    parser.add_argument(
        "--model", "-m", required=True, help="minimax-h3, minimax-h3-turbo, a Hugging Face repo, or a local path."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Catalog entry a prepared package or repo id runs as (minimax-h3, minimax-h3-turbo, minimax-h3-turbo-544p).",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", type=str, help="Visual description (or a complete structured H3 prompt).")
    prompt_group.add_argument("--prompt-file", type=Path, help="Path to a text file containing the prompt.")
    parser.add_argument(
        "--soundscape", type=str, default=None, help="`overall_soundscape` section: diegetic sound design."
    )
    parser.add_argument("--music", type=str, default=None, help="`non_diegetic_music` section: score description.")
    parser.add_argument(
        "--image-path",
        default=None,
        help="Keyframe the video starts from (image-to-video). The canvas follows its aspect ratio unless --width/--height are given.",
    )
    parser.add_argument(
        "--width", type=int, default=None, help="Canvas width, multiple of 32 (default: 16:9 at 768p short edge)."
    )
    parser.add_argument("--height", type=int, default=None, help="Canvas height, multiple of 32.")
    parser.add_argument(
        "--frames", type=int, default=None, help="Frame count at 24 fps, rounded up to 17n+5 (default 124, 5.2 s)."
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Transformer evaluations (default 50; Turbo 8-step LoRA: 8)."
    )
    parser.add_argument("--video-shift", type=float, default=None, help="Video flow shift (default 12; Turbo 768p: 6).")
    parser.add_argument("--audio-shift", type=float, default=None, help="Audio flow shift (default 3).")
    parser.add_argument("--no-audio", action="store_true", help="Skip the audio decode and write a silent clip.")
    parser.add_argument(
        "--release-text-encoder",
        action="store_true",
        help="Drop the Qwen3-VL conditioner once the prompt is encoded, lowering the peak by its resident size. "
        "Cached prompt embeddings stay usable; a later run with a new prompt reloads it.",
    )
    parser.add_argument("--seed", "-s", type=int, default=None, nargs="+", help="One or more random seeds.")
    parser.add_argument("--auto-seeds", type=int, default=-1, help="Generate N random seeds between 0 and 10,000,000.")
    parser.add_argument("--quantize", "-q", type=int, choices=ui_defaults.QUANTIZE_CHOICES, default=None)
    parser.add_argument("--lora-paths", type=str, nargs="*", default=None, help="LoRA files (PEFT diffusers layout).")
    parser.add_argument("--lora-scales", type=float, nargs="*", default=None, help="Per-LoRA scales (default 1.0).")
    parser.add_argument(
        "--mlx-cache-limit-gb",
        type=cache_limit_gb_value,
        default=None,
        help="Cap the MLX free-buffer cache in GB (default: total RAM / 8, clamped to 1-8 GiB; -1 for unlimited).",
    )
    parser.add_argument("--metadata", action="store_true", help="Export video metadata as JSON.")
    parser.add_argument("--output", type=str, default="video.mp4", help='Output path. Default is "video.mp4".')
    parser.add_argument("--json-events", action="store_true", help="Emit machine-readable runtime events on stdout.")
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
    parser.add_argument("--no-validate-health", action="store_true", help="Skip the post-save decode check.")
    parser.add_argument("--debug", action="store_true", help="Verbose LoRA loading diagnostics.")
    return parser


class _CliProgress:
    _lock_configured = False

    def __init__(self, enabled: bool):
        self.enabled = enabled
        if enabled and not _CliProgress._lock_configured:
            tqdm.set_lock(threading.RLock())
            _CliProgress._lock_configured = True
        self._bar: tqdm | None = None
        self._last_step = 0

    def __call__(self, event: ProgressEvent) -> None:
        if not self.enabled:
            return
        if self._bar is None:
            self._bar = tqdm(total=event.total_steps, desc="Denoising video+audio", unit="step")
        delta = max(0, event.step - self._last_step)
        if delta:
            self._bar.update(delta)
            self._last_step = event.step
        self._bar.set_postfix_str(f"{event.phase}; {event.total_frames} frames")
        if event.phase in {"generated", "complete", "failed"}:
            self.close()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


if __name__ == "__main__":
    main()
