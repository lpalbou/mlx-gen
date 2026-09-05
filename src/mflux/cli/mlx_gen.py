import argparse
import json
import shlex
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.router_options import ForwardPolicy, add_router_options, reemit_options
from mflux.cli.runtime_events import cli_print, emit_cli_failure_event_for_argv
from mflux.models.common.config import ModelConfig
from mflux.models.common.download_policy import allow_downloads, is_huggingface_repo_id
from mflux.models.common.lora.lora_compatibility import LoRACompatibility
from mflux.models.common.lora.mapping.lora_loader import LoRAApplicationError
from mflux.release.validation_registry import (
    default_validation_profile_id_for_model,
    get_model_validation,
    get_validation_profile,
    list_validation_profiles,
)
from mflux.task_inference import TaskInferenceError, get_model_capabilities, normalize_task, resolve_generation_plan
from mflux.utils.box_values import BoxValueError, BoxValues
from mflux.utils.exceptions import ModelConfigError
from mflux.utils.version_util import VersionUtil


@dataclass(frozen=True)
class RouterInvocation:
    target_name: str
    target_main: Callable[[], None]
    argv: list[str]


@dataclass(frozen=True)
class _Route:
    target_name: str
    target_main: Callable[[], None]
    image_argument: str | None
    video_argument: str | None
    requires_image: bool
    model_override: str | None = None
    control_model: str | None = None
    # Whether the selected route's capability row publishes supports_outpaint_fill. Only those
    # backends declare --outpaint-fill/--outpaint-fill-color; a fixed-canvas route is given the
    # flags' one legal value or nothing at all, so they are dropped rather than forwarded.
    accepts_outpaint_fill: bool = False


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _top_level_parser().print_help()
        return

    if argv and argv[0] in {"download", "prepare", "capabilities", "validation", "upscale"}:
        _run_model_command(argv)
        return

    invocation = _resolve_invocation(_normalize_command(argv))
    previous_argv = sys.argv
    try:
        sys.argv = invocation.argv
        try:
            invocation.target_main()
        except (FileNotFoundError, LoRAApplicationError) as exc:
            if "--json-events" in invocation.argv:
                emit_cli_failure_event_for_argv(
                    prog="mlxgen generate",
                    argv=invocation.argv[1:],
                    error=exc,
                )
            cli_print(str(exc), json_events="--json-events" in invocation.argv, error=True)
            raise SystemExit(1) from None
    finally:
        sys.argv = previous_argv


def _resolve_invocation(argv: list[str]) -> RouterInvocation:
    args, forwarded = _parse_router_args(argv)
    images = _collect_images(args)
    videos = _collect_videos(args)
    reference_images = _collect_reference_images(args)
    # The SVI anchor (0103) IS the image-to-video source for plan resolution:
    # it fills the image slot without ever being emitted as --image-path (the
    # wan backend owns the --svi-anchor-image contract).
    svi_argv = _svi_forwarded_argv(args=args, images=images)
    route = _resolve_route(
        args,
        image_count=len(images) + (1 if args.svi_anchor_image else 0),
        video_count=len(videos),
        reference_image_count=len(reference_images),
    )
    if route.requires_image and not images:
        _parser().error(f"{route.target_name} requires --image or --images.")
    reframe_argv = _reframe_forwarded_argv(args=args, images=images, forwarded=forwarded)
    outpaint_argv = _outpaint_forwarded_argv(args=args, images=images, forwarded=forwarded, route=route)

    normalized_argv = [route.target_name]
    # Consumed options are re-emitted from the ROUTER_OPTIONS descriptor table so a consumed flag
    # can never be silently dropped again; the round-trip test locks each descriptor's fate.
    for option in reemit_options():
        value = getattr(args, option.dest)
        if option.use_route_model_override:
            value = route.model_override or value
        emit_allowed = not option.route_gated or _route_accepts_base_model(route)
        if option.policy is ForwardPolicy.REEMIT_FLAG:
            if value and emit_allowed:
                if option.emit_flag is None:
                    raise AssertionError(f"{option.dest} has no re-emission flag.")
                normalized_argv.append(option.emit_flag)
        elif option.policy is ForwardPolicy.REEMIT_VALUES:
            if value and emit_allowed:
                if option.emit_flag is None:
                    raise AssertionError(f"{option.dest} has no re-emission flag.")
                for item in value:
                    normalized_argv.extend([option.emit_flag, str(item)])
        elif value is not None and emit_allowed:
            if option.emit_flag is None:
                raise AssertionError(f"{option.dest} has no re-emission flag.")
            normalized_argv.extend([option.emit_flag, str(value)])
        # --controlnet-model is INJECTED (never consumed): it rides between --model and
        # --base-model to preserve the emission order pinned by the exact-argv tests.
        if option.dest == "model" and route.control_model is not None:
            if args.requested_controlnet_model is not None and args.requested_controlnet_model != route.control_model:
                _parser().error(
                    "--controlnet-model conflicts with the exact ControlNet route selected by mlxgen generate. "
                    "Use the documented route, or call the backend command directly if you need a different "
                    "ControlNet package."
                )
            if args.requested_controlnet_model is None:
                normalized_argv.extend(["--controlnet-model", route.control_model])
        # Image/video path emission is route-dependent (TRANSFORMED); it rides after --base-model.
        if option.dest == "base_model":
            if route.image_argument is not None and images:
                normalized_argv.append(route.image_argument)
            normalized_argv.extend(images)
            if route.video_argument is not None and videos:
                normalized_argv.append(route.video_argument)
            normalized_argv.extend(videos)
    normalized_argv.extend(forwarded)
    normalized_argv.extend(reframe_argv)
    normalized_argv.extend(outpaint_argv)
    normalized_argv.extend(svi_argv)

    return RouterInvocation(
        target_name=route.target_name,
        target_main=route.target_main,
        argv=normalized_argv,
    )


def _route_accepts_base_model(route: _Route) -> bool:
    return route.target_name in {
        "mflux-generate-bonsai",
        "mflux-generate-fibo",
        "mflux-generate-fibo-edit",
        "mflux-generate-flux2",
        "mflux-generate-flux2-edit",
        "mflux-generate-ernie-image",
        "mflux-generate-qwen",
        "mflux-generate-qwen-edit",
        "mflux-generate-z-image",
        "mflux-generate-z-image-turbo",
        "mlxgen-generate-minimax-h3",
    }


def _parse_router_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = _parser()
    args, forwarded = parser.parse_known_args(argv)
    metadata = _read_metadata_config(parser, argv)
    if args.model is None:
        args.model = metadata.get("model")
    if args.base_model is None:
        args.base_model = metadata.get("base_model")
    _reject_prepare_path_in_generate(parser, args, forwarded)
    args.metadata_images = _metadata_images(metadata)
    args.metadata_videos = _metadata_videos(metadata)
    args.metadata_reference_images = _metadata_reference_images(metadata)
    if args.reference_image_paths is None:
        args.reference_image_paths = list(args.metadata_reference_images)
    try:
        args.task = normalize_task(args.task)
    except TaskInferenceError as exc:
        parser.error(str(exc))
    args.has_image_strength = _option_was_provided(argv, "--image-strength") or _metadata_has_positive_number(
        metadata,
        "image_strength",
    )
    args.has_video_strength = _option_was_provided(argv, "--video-strength") or _metadata_has_positive_number(
        metadata,
        "video_strength",
    )
    if args.video_strength is None and metadata.get("video_strength") is not None:
        # Backfill + re-emit so the backend parser validates metadata-sourced strength at parse
        # time (before the model load) exactly like an argv-sourced value.
        args.video_strength = metadata.get("video_strength")
    if args.video_mask_path is None:
        args.video_mask_path = metadata.get("video_mask_path")
    args.has_video_mask = args.video_mask_path is not None
    if args.svi_anchor_image is None and metadata.get("svi_anchor_image_path") is not None:
        args.svi_anchor_image = str(metadata.get("svi_anchor_image_path"))
    if args.reframe_padding is None:
        args.reframe_padding = metadata.get("reframe_padding")
    if args.outpaint_padding is None:
        args.outpaint_padding = metadata.get("outpaint_padding") or metadata.get("image_outpaint_padding")
    args.has_mask = _has_mask_option(argv, metadata)
    args.has_control_image = _has_controlnet_image_option(argv, metadata)
    args.has_outpaint = args.outpaint_padding is not None or _has_outpaint_option(argv, metadata)
    args.has_negative_prompt = bool(
        _option_value(argv, "--negative-prompt") or _option_value(argv, "--negative") or metadata.get("negative_prompt")
    )
    args.has_reframe = args.reframe_padding is not None
    args.has_lora = _has_lora_option(argv, metadata)
    args.lora_paths = _lora_paths_from_options(argv, metadata)
    args.requested_controlnet_model = _option_value(argv, "--controlnet-model") or metadata.get("controlnet_model")
    args.requested_controlnet_strength = _option_value(argv, "--controlnet-strength")
    if args.requested_controlnet_strength is None and metadata.get("controlnet_strength") is not None:
        args.requested_controlnet_strength = str(metadata["controlnet_strength"])
    args.has_explicit_controlnet_model = args.requested_controlnet_model is not None
    args.has_explicit_controlnet_strength = args.requested_controlnet_strength is not None
    if args.has_lora and _has_lora_scales_without_paths(argv, metadata):
        parser.error("--lora-scales requires --lora-paths.")
    if args.has_reframe and args.has_outpaint:
        parser.error("--reframe-padding and --outpaint-padding are different workflows and cannot be used together.")
    if args.model is None:
        parser.error("--model is required so mlx-gen can choose the right backend, unless metadata provides it.")
    return args, forwarded


def _reject_prepare_path_in_generate(
    parser: argparse.ArgumentParser, args: argparse.Namespace, forwarded: list[str]
) -> None:
    path = _option_value(forwarded, "--path")
    if path is None:
        return

    quantize = _option_value(forwarded, "--quantize") or _option_value(forwarded, "-q")
    prepare_command = ["mlxgen", "prepare"]
    if args.model is not None:
        prepare_command.extend(["--model", args.model])
    prepare_command.extend(["--path", path])
    if quantize is not None:
        prepare_command.extend(["--quantize", quantize])

    parser.error(
        "--path prepares a local model folder and is not a generation option. "
        f"Use `{shlex.join(prepare_command)}` to create the model folder. "
        "Use --output to choose the generated image path."
    )


def _option_value(argv: list[str], option_name: str) -> str | None:
    for index, token in enumerate(argv):
        if token == option_name:
            if index + 1 >= len(argv):
                return ""
            return argv[index + 1]
        if token.startswith(f"{option_name}="):
            return token.split("=", 1)[1]
    return None


def _normalize_command(argv: list[str]) -> list[str]:
    if argv and argv[0] in {"generate", "gen"}:
        return argv[1:]
    return argv


def _top_level_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxgen",
        usage="mlxgen [generate|upscale|capabilities|validation|download|prepare] ...",
        description=(
            f"{VersionUtil.format_cli_release_label()}\n\n"
            "Prepare local model assets and generate images or videos with MLX-Gen."
        ),
        epilog=(
            "Commands:\n"
            "  generate    Generate images or videos, and edit images, from a prepared or cached model.\n"
            "  upscale     Restore or upscale images and videos with SeedVR2 or SwiftVR.\n"
            "  capabilities Inspect model generation tasks, modes, and option support.\n"
            "  validation   Inspect release-validation evidence for exact model/package rows.\n"
            "  download     Explicitly download a model snapshot into the Hugging Face cache.\n"
            "  prepare      Create a reusable local MLX-Gen model folder, optionally quantized.\n"
            "\n"
            "Examples:\n"
            "  mlxgen generate --model z-image-turbo --prompt 'A puffin standing on a cliff'\n"
            "  mlxgen generate --model wan2.2-ti2v-5b --prompt 'A city timelapse'\n"
            "  mlxgen upscale --model seedvr2-3b --image-path input.png --resolution 2x --output upscaled.png\n"
            "  mlxgen capabilities --model flux2-klein-4b\n"
            "  mlxgen capabilities --model swiftvr\n"
            "  mlxgen validation --model AbstractFramework/qwen-image-edit-2509-8bit\n"
            "  mlxgen download --model Qwen/Qwen-Image\n"
            "  mlxgen prepare --model Qwen/Qwen-Image --path ./models/qwen-image-8bit --quantize 8\n"
            "\n"
            "Use 'mlxgen <command> --help' for command-specific options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser


def _option_was_provided(argv: list[str], option_name: str) -> bool:
    for token in argv:
        if token == option_name or token.startswith(f"{option_name}="):
            return True
    return False


def _metadata_has_positive_number(metadata: dict, key: str) -> bool:
    value = metadata.get(key)
    if value is None:
        return False
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _has_mask_option(argv: list[str], metadata: dict) -> bool:
    return (
        _option_was_provided(argv, "--mask-path")
        or _option_was_provided(argv, "--masked-image-path")
        or metadata.get("mask_path") is not None
        or metadata.get("masked_image_path") is not None
    )


def _has_outpaint_option(argv: list[str], metadata: dict) -> bool:
    return (
        _option_was_provided(argv, "--outpaint-padding")
        or _option_was_provided(argv, "--image-outpaint-padding")
        or metadata.get("outpaint_padding") is not None
        or metadata.get("image_outpaint_padding") is not None
    )


def _has_controlnet_image_option(argv: list[str], metadata: dict) -> bool:
    return _option_was_provided(argv, "--controlnet-image-path") or metadata.get("controlnet_image_path") is not None


def _has_lora_option(argv: list[str], metadata: dict) -> bool:
    return (
        _option_was_provided(argv, "--lora-paths")
        or _option_was_provided(argv, "--lora-scales")
        or _option_was_provided(argv, "--lora-style")
        or bool(metadata.get("lora_paths"))
        or bool(metadata.get("lora_scales"))
    )


def _has_lora_scales_without_paths(argv: list[str], metadata: dict) -> bool:
    has_scales = _option_was_provided(argv, "--lora-scales") or bool(metadata.get("lora_scales"))
    has_paths = _option_was_provided(argv, "--lora-paths") or bool(metadata.get("lora_paths"))
    return has_scales and not has_paths


def _lora_paths_from_options(argv: list[str], metadata: dict) -> list[str]:
    paths = _option_values(argv, "--lora-paths")
    metadata_paths = metadata.get("lora_paths")
    if isinstance(metadata_paths, str):
        paths.append(metadata_paths)
    elif isinstance(metadata_paths, list):
        paths.extend(str(path) for path in metadata_paths)
    return paths


def _option_values(argv: list[str], option_name: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith(f"{option_name}="):
            values.append(token.split("=", 1)[1])
            index += 1
            continue
        if token == option_name:
            index += 1
            while index < len(argv) and not argv[index].startswith("--"):
                values.append(argv[index])
                index += 1
            continue
        index += 1
    return values


def _read_metadata_config(parser: argparse.ArgumentParser, argv: list[str]) -> dict:
    metadata_path = _metadata_path(argv)
    if metadata_path is None:
        return {}
    try:
        with metadata_path.open("rt") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"Could not read --config-from-metadata {metadata_path}: {exc}")
    if not isinstance(metadata, dict):
        parser.error(f"--config-from-metadata {metadata_path} must contain a JSON object.")
    return metadata


def _metadata_path(argv: list[str]) -> Path | None:
    for index, token in enumerate(argv):
        if token in {"--config-from-metadata", "-C"}:
            if index + 1 >= len(argv):
                return None
            return Path(argv[index + 1])
        if token.startswith("--config-from-metadata="):
            return Path(token.split("=", 1)[1])
    return None


def _metadata_images(metadata: dict) -> list[str]:
    image_paths = metadata.get("image_paths")
    if isinstance(image_paths, list):
        return [str(path) for path in image_paths]
    if isinstance(image_paths, str):
        return [image_paths]

    image_path = metadata.get("image_path")
    if isinstance(image_path, str):
        return [image_path]
    return []


def _metadata_videos(metadata: dict) -> list[str]:
    video_paths = metadata.get("video_paths")
    if isinstance(video_paths, list):
        return [str(path) for path in video_paths]
    if isinstance(video_paths, str):
        return [video_paths]

    video_path = metadata.get("video_path")
    if isinstance(video_path, str):
        return [video_path]
    return []


def _metadata_reference_images(metadata: dict) -> list[str]:
    paths = metadata.get("reference_image_paths")
    if isinstance(paths, list):
        return [str(path) for path in paths]
    if isinstance(paths, str):
        return [paths]
    return []


def _parser() -> argparse.ArgumentParser:
    parser = CommandLineParser(
        prog="mlxgen generate",
        description=(
            "Generate images or videos with MLX-Gen. The command routes to the right model backend from --model "
            "and from whether input images or a source video are supplied."
        ),
        epilog=(
            "Common generation options are forwarded to the selected backend, including --prompt, "
            "--prompt-file, --width, --height, --steps, --guidance, --seed, --auto-seeds, "
            "--negative-prompt/--negative, --canvas-policy, --resize-mode (latent image-to-image and "
            "Wan video routes), --quantize, --lora-paths, --lora-scales, "
            "--mask-path, --controlnet-image-path, --controlnet-strength, --metadata, "
            "--config-from-metadata/-C, --output, --replace, --frames, --fps, --guidance-2, "
            "--flow-shift, --video-strength, --video-mask-path, --reframe-padding, --outpaint-padding, --outpaint-fill, --outpaint-fill-color, --low-ram, --debug, "
            "--tensor-health-check-interval, --json-events, --embed-metadata, --no-validate-health (video routes), "
            "--keep-text-encoder, --no-prompt-cache, --compile-transformer, and "
            "--release-inactive-denoiser/--no-release-inactive-denoiser (Wan routes), "
            "and --progress/--no-progress.\n"
            "Restore or upscale an existing source video with `mlxgen upscale --video-path ...`.\n"
            "Plain prompt-guided video-to-video is available on exact Wan video-to-video routes such as "
            "`Wan2.2-T2V-A14B`; keep `--solver unipc` for that public path.\n"
            "Wan video routes also accept --failure-diagnostics for failure manifests.\n\n"
            "Exact VACE and Bernini routes accept repeatable --reference-image conditioning.\n"
            "Use --mask-path for localized masked edit or inpaint on models that support masked edit or inpaint.\n"
            "Use --controlnet-image-path for structured control on a text-to-image route; it is not the same as "
            "source-image editing.\n"
            "--outpaint-passes (auto, 1, 2) decides whether a --outpaint-padding request that pads both axes "
            "deeply runs as two single-axis passes; auto splits past the route's published depth."
        ),
    )
    add_router_options(parser)
    return parser


def _run_model_command(argv: list[str]) -> None:
    command = argv[0]
    if command == "capabilities":
        _show_capabilities(argv[1:])
        return
    if command == "validation":
        _show_validation(argv[1:])
        return
    if command == "upscale":
        _upscale_image(argv[1:])
        return
    if command == "download":
        _download_model(argv[1:])
        return
    if command == "prepare":
        _prepare_model(argv[1:])
        return
    raise AssertionError("unreachable")


def _upscale_image(argv: list[str]) -> None:
    from mflux.models.common.cli.restore_dispatch import upscale_main

    previous_argv = sys.argv
    try:
        sys.argv = ["mlxgen upscale", *argv]
        upscale_main()
    finally:
        sys.argv = previous_argv


def _show_capabilities(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="mlxgen capabilities",
        description=(
            "Inspect the public tasks, internal modes, and option support for a model. Generation "
            "routes are listed under 'capabilities'; promptless restoration routes (mlxgen upscale) "
            "are listed under 'restoration'. An empty array means the model is not routable through "
            "that command, not that the model is unsupported."
        ),
    )
    parser.add_argument("--model", "-m", required=True, help="Model alias, Hugging Face repo, or local model path.")
    parser.add_argument("--base-model", default=None, help="Base model hint for custom repositories or local paths.")
    parser.add_argument(
        "--family",
        choices=[
            "qwen",
            "flux2",
            "fibo",
            "z-image",
            "ernie-image",
            "wan",
            "minimax-h3",
            "bonsai",
            "seedvr2",
            "swiftvr",
        ],
        default=None,
        help="Override model-family detection for local paths or custom repo names.",
    )
    args = parser.parse_args(argv)
    try:
        model_config = _model_config(args.model, base_model=args.base_model)
        capabilities = get_model_capabilities(
            model=args.model,
            model_config=model_config,
            family=args.family,
            base_model=args.base_model,
        )
    except TaskInferenceError as exc:
        parser.error(str(exc))
    print(json.dumps(capabilities.to_dict(), indent=2, sort_keys=True))


def _show_validation(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="mlxgen validation",
        description=(
            "Inspect release-validation evidence for exact model/package rows. This is separate from "
            "route capabilities and does not control mlxgen generate."
        ),
    )
    parser.add_argument("--model", "-m", default=None, help="Model alias, Hugging Face repo, or local model path.")
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Validation profile id. Defaults to the first profile with evidence for the requested model, "
            "or the current I2I edit 5x4 profile when no model-specific evidence exists."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available validation profiles instead of returning profile/model rows.",
    )
    args = parser.parse_args(argv)
    try:
        if args.list:
            payload = {
                "profiles": [
                    {
                        "id": profile.id,
                        "title": profile.title,
                        "canonical_source": profile.canonical_source,
                        "description": profile.description,
                        "record_count": len(profile.records),
                    }
                    for profile in list_validation_profiles()
                ]
            }
        elif args.model:
            payload = get_model_validation(
                args.model,
                profile_id=args.profile or default_validation_profile_id_for_model(args.model),
            ).to_dict()
        else:
            payload = get_validation_profile(args.profile or get_validation_profile().id).to_dict()
    except KeyError as exc:
        parser.error(str(exc))
    print(json.dumps(payload, indent=2, sort_keys=True))


def _download_model(argv: list[str]) -> None:
    # Deferred: huggingface_hub pulls httpx+rich (~0.5 s); only the explicit
    # download command needs it, never router dispatch (0088).
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(
        prog="mlxgen download",
        description="Explicitly download a Hugging Face model snapshot into the local cache.",
    )
    parser.add_argument("--model", "-m", required=True, help="Model alias or Hugging Face repo id.")
    parser.add_argument("--base-model", default=None, help="Base model hint for custom repositories.")
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download the full repository instead of the MLX-Gen weight/tokenizer patterns.",
    )
    args = parser.parse_args(argv)

    if _is_depth_pro_download(args.model):
        _download_depth_pro()
        return

    model_config = _model_config(args.model, base_model=args.base_model)
    repo_id = model_config.model_name if model_config is not None else args.model
    if not is_huggingface_repo_id(repo_id):
        parser.error(f"--model must resolve to a Hugging Face repo id for download. Got: {repo_id!r}")

    if (
        args.all_files
        and model_config is not None
        and model_config.transformer_overrides.get("supports_bernini_renderer")
    ):
        parser.error(
            "Bernini-R uses a pinned component subset. Omit --all-files so MLX-Gen downloads "
            "only the required official runtime files."
        )

    download_sources: list[tuple[str, list[str] | None, str | None, int | None]] = (
        _factored_download_sources(model_config) if not args.all_files else []
    )
    if not download_sources:
        download_sources = [
            (
                repo_id,
                None if args.all_files else _download_patterns(model_config, repo_id),
                None,
                # expected_download_bytes counts the pattern-scoped subset a run needs, so
                # it does not describe an --all-files snapshot; the preflight skips rather
                # than checking against a number it knows is too small.
                None if args.all_files else _expected_download_bytes(model_config),
            )
        ]
    if model_config is not None and any(expected for *_, expected in download_sources):
        _preflight_download_space(parser, model_config, download_sources)
    downloaded_paths = []
    with allow_downloads():
        for source_repo, patterns, revision, _expected_bytes in download_sources:
            revision_label = f" at {revision}" if revision is not None else ""
            print(f"Downloading {source_repo}{revision_label} into the Hugging Face cache...")
            if revision is None:
                downloaded_paths.append(snapshot_download(repo_id=source_repo, allow_patterns=patterns))
            else:
                downloaded_paths.append(
                    snapshot_download(repo_id=source_repo, allow_patterns=patterns, revision=revision)
                )
    for path in downloaded_paths:
        print(f"Downloaded snapshot: {path}")
    turbo_lora = model_config.transformer_overrides.get("turbo_lora") if model_config is not None else None
    if turbo_lora:
        from mflux.models.common.resolution.lora_resolution import LoraResolution

        with allow_downloads():
            print(f"Downloaded Turbo LoRA: {LoraResolution.resolve(turbo_lora)}")
    print("You can now run generation without a runtime download, for example:")
    if model_config is not None and model_config.transformer_overrides.get("supports_bernini_renderer"):
        print(
            "  mlxgen generate --model "
            f"{shlex.quote(repo_id)} --reference-image reference.png "
            "--prompt 'Bring the referenced subject to life' --output video.mp4"
        )
    elif model_config is not None and _is_swiftvr(
        set(model_config.aliases), _model_key(repo_id, model_config.base_model)
    ):
        print(f"  mlxgen upscale --model {shlex.quote(repo_id)} --video-path input.mp4 --output restored.mp4")
    elif model_config is not None and _is_minimax_h3(
        set(model_config.aliases), _model_key(repo_id, model_config.base_model)
    ):
        print(
            f"  mlxgen generate --model {shlex.quote(model_config.aliases[0])} "
            "--prompt 'A red fox trots through fresh snow at dawn' --soundscape 'Snow crunching, distant crows' "
            "--output video.mp4"
        )
    elif model_config is not None and _is_seedvr2(
        set(model_config.aliases), _model_key(repo_id, model_config.base_model)
    ):
        print(
            "  mlxgen upscale --model "
            f"{shlex.quote(repo_id)} --image-path input.png --resolution 2x --output upscaled.png"
        )
    else:
        print(
            "  mlxgen generate --model "
            f"{shlex.quote(repo_id)} --prompt 'A product photo of a ceramic teapot' --output image.png"
        )


def _expected_download_bytes(model_config: ModelConfig | None) -> int | None:
    """Catalog byte count for a single-repository download, or ``None`` when unknown."""
    if model_config is None:
        return None
    expected = model_config.transformer_overrides.get("expected_download_bytes")
    return expected if isinstance(expected, int) and expected > 0 else None


def _preflight_download_space(
    parser: argparse.ArgumentParser,
    model_config: ModelConfig,
    download_sources: list[tuple[str, list[str] | None, str | None, int | None]],
) -> None:
    """Refuse a download that cannot fit, for any source carrying an expected size.

    Sources with no declared size return early and are never guessed at: a preflight that
    invents a number would either block a download that fits or wave through one that does
    not.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    from mflux.models.common.resolution.path_resolution import PathResolution

    overrides = model_config.transformer_overrides
    missing_bytes = 0
    missing_sources = []
    for source_repo, patterns, revision, expected_bytes in download_sources:
        if PathResolution._find_complete_cached_snapshot(source_repo, patterns, revision=revision) is not None:
            continue
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            return
        missing_bytes += expected_bytes
        missing_sources.append(source_repo)

    if not missing_sources:
        return

    cache_path = Path(HF_HUB_CACHE).expanduser()
    existing_path = cache_path
    while not existing_path.exists() and existing_path != existing_path.parent:
        existing_path = existing_path.parent
    free_bytes = shutil.disk_usage(existing_path).free
    headroom_bytes = overrides.get("download_headroom_bytes", 0)
    if not isinstance(headroom_bytes, int) or headroom_bytes < 0:
        headroom_bytes = 0
    required_bytes = missing_bytes + headroom_bytes
    if free_bytes >= required_bytes:
        return

    missing_label = ", ".join(missing_sources)
    parser.error(
        f"Insufficient free space for the missing model sources ({missing_label}). "
        f"Need at least {required_bytes / 1024**3:.2f} GiB including download headroom; "
        f"only {free_bytes / 1024**3:.2f} GiB is available at {existing_path}. "
        "Already complete pinned sources are not counted."
    )


def _factored_download_sources(model_config: ModelConfig | None) -> list[tuple[str, list[str], str | None, int | None]]:
    if model_config is None:
        return []
    overrides = model_config.transformer_overrides
    base_source = overrides.get("component_base_model")
    transformer_source = model_config.custom_transformer_model
    if not isinstance(base_source, str) or not base_source or not transformer_source:
        return []

    from mflux.models.wan.weights import WanWeightDefinition

    definition = WanWeightDefinition.for_config(model_config)
    grouped: dict[tuple[str, str | None], tuple[list[str], int | None]] = {}
    for source, patterns, revision, expected_bytes in (
        (
            base_source,
            definition.get_base_download_patterns(),
            overrides.get("expected_component_base_revision"),
            overrides.get("expected_component_base_download_bytes"),
        ),
        (
            transformer_source,
            definition.get_transformer_download_patterns(),
            overrides.get("expected_transformer_revision"),
            overrides.get("expected_transformer_download_bytes"),
        ),
    ):
        key = (source, revision)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = (list(patterns), expected_bytes if isinstance(expected_bytes, int) else None)
            continue
        merged_patterns, merged_bytes = existing
        for pattern in patterns:
            if pattern not in merged_patterns:
                merged_patterns.append(pattern)
        if isinstance(expected_bytes, int) and expected_bytes > 0:
            merged_bytes = (merged_bytes or 0) + expected_bytes
        grouped[key] = (merged_patterns, merged_bytes)
    return [
        (source, patterns, revision, expected_bytes)
        for (source, revision), (patterns, expected_bytes) in grouped.items()
    ]


def _is_depth_pro_download(model: str) -> bool:
    normalized = model.strip().lower().replace("_", "-")
    return normalized in {"depth-pro", "apple/depth-pro"}


def _download_depth_pro() -> None:
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.depth_pro.weights.depth_pro_weight_definition import DepthProWeightDefinition

    component = DepthProWeightDefinition.get_components()[0]
    if component.download_url is None:
        raise RuntimeError("Depth Pro download URL is not configured.")

    print("Downloading Depth Pro weights into the MLX-Gen cache...")
    with allow_downloads():
        path = WeightLoader._download_from_url(component.download_url, component.name)
    print(f"Downloaded Depth Pro weights: {path}")


def _prepare_model(argv: list[str]) -> None:
    from mflux.models.common.cli.save import main as save_main

    previous_argv = sys.argv
    try:
        sys.argv = ["mlxgen prepare", *argv]
        with allow_downloads():
            save_main()
    finally:
        sys.argv = previous_argv


def _download_patterns(model_config: ModelConfig | None, repo_id: str) -> list[str] | None:
    if model_config is None:
        return None

    aliases = set(model_config.aliases)
    model_key = _model_key(model_config.model_name, model_config.base_model, repo_id)
    weight_definition = _weight_definition_for(aliases, model_key, model_config)
    if weight_definition is None:
        return None

    if _is_seedvr2(aliases, model_key):
        patterns = list(weight_definition.get_download_patterns_for_source(model_config, repo_id))
    else:
        patterns = list(weight_definition.get_download_patterns())
    for tokenizer in weight_definition.get_tokenizers():
        patterns.extend(tokenizer.download_patterns or [f"{tokenizer.hf_subdir}/**"])
    return sorted(set(patterns))


def _weight_definition_for(aliases: set[str], model_key: str, model_config: ModelConfig | None = None):
    if _is_qwen(aliases, model_key):
        from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

        return QwenWeightDefinition
    if _is_flux2(aliases, model_key):
        from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

        return Flux2KleinWeightDefinition
    if _is_bonsai(aliases, model_key):
        from mflux.models.bonsai_image.weights import BonsaiImageWeightDefinition

        return BonsaiImageWeightDefinition
    if _is_fibo(aliases, model_key):
        from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition

        return FIBOWeightDefinition
    if _is_z_image(aliases, model_key):
        from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

        return ZImageWeightDefinition
    if _is_ernie(aliases, model_key):
        from mflux.models.ernie_image.weights.ernie_image_weight_definition import ErnieImageWeightDefinition

        return ErnieImageWeightDefinition
    if _is_swiftvr(aliases, model_key):
        from mflux.models.swiftvr.weights.swiftvr_weight_definition import SwiftVRWeightDefinition

        return SwiftVRWeightDefinition
    if _is_minimax_h3(aliases, model_key):
        from mflux.models.minimax_h3.weights import MiniMaxH3WeightDefinition

        return MiniMaxH3WeightDefinition
    if _is_wan(aliases, model_key):
        from mflux.models.wan.weights import WanWeightDefinition

        if model_config is not None:
            return WanWeightDefinition.for_config(model_config)
        return WanWeightDefinition
    if _is_seedvr2(aliases, model_key):
        from mflux.models.seedvr2.weights.seedvr2_weight_definition import SeedVR2WeightDefinition

        return SeedVR2WeightDefinition
    return None


def _collect_images(args: argparse.Namespace) -> list[str]:
    images = []
    if args.image_path is not None:
        images.append(args.image_path)
    images.extend(args.images)
    for group in args.image_groups:
        images.extend(group)
    return images or args.metadata_images


def _collect_videos(args: argparse.Namespace) -> list[str]:
    videos = []
    if args.video_path is not None:
        videos.append(args.video_path)
    videos.extend(args.videos)
    return videos or args.metadata_videos


def _collect_reference_images(args: argparse.Namespace) -> list[str]:
    return [str(path) for path in (args.reference_image_paths or args.metadata_reference_images)]


def _svi_forwarded_argv(args: argparse.Namespace, images: list[str]) -> list[str]:
    if args.svi_anchor_image is None:
        return []
    if images:
        _parser().error(
            "--svi-anchor-image conflicts with --image/--image-path: SVI conditioning derives the whole "
            "clip from the anchor and the optional --svi-motion-latent."
        )
    return ["--svi-anchor-image", str(args.svi_anchor_image)]


def _reframe_forwarded_argv(args: argparse.Namespace, images: list[str], forwarded: list[str]) -> list[str]:
    if not args.has_reframe:
        return []
    _validate_padding_value(args.reframe_padding, option_name="--reframe-padding")
    if len(images) != 1:
        _parser().error("--reframe-padding requires exactly one --image or --image-path.")
    if _option_was_provided(forwarded, "--width") or _option_was_provided(forwarded, "--height"):
        _parser().error(
            "--reframe-padding computes --width and --height from the source image; do not pass either option."
        )
    if _option_was_provided(forwarded, "--canvas-policy"):
        _parser().error("--reframe-padding uses --canvas-policy exact-resize; do not pass --canvas-policy.")

    return ["--reframe-padding", args.reframe_padding]


def _outpaint_forwarded_argv(
    args: argparse.Namespace, images: list[str], forwarded: list[str], route: _Route
) -> list[str]:
    if not args.has_outpaint:
        # The fill options configure the outpaint canvas and are meaningless without it;
        # _validate_outpaint_fill_options has already rejected them when they were passed.
        return []
    fill_argv = _outpaint_fill_forwarded_argv(args) if route.accepts_outpaint_fill else []
    # The pass planner is the shared layer's, so every outpaint backend declares --outpaint-passes.
    if args.outpaint_passes is not None:
        fill_argv.extend(["--outpaint-passes", str(args.outpaint_passes)])
    _validate_padding_value(args.outpaint_padding, option_name="--outpaint-padding")
    if len(images) != 1:
        _parser().error("--outpaint-padding requires exactly one --image or --image-path.")
    if _option_was_provided(forwarded, "--width") or _option_was_provided(forwarded, "--height"):
        _parser().error(
            "--outpaint-padding computes --width and --height from the source image; do not pass either option."
        )
    if _option_was_provided(forwarded, "--canvas-policy"):
        _parser().error("--outpaint-padding uses --canvas-policy exact-resize; do not pass --canvas-policy.")

    return ["--outpaint-padding", args.outpaint_padding, *fill_argv]


def _outpaint_fill_forwarded_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    if args.outpaint_fill is not None:
        argv.extend(["--outpaint-fill", str(args.outpaint_fill)])
    if args.outpaint_fill_color is not None:
        argv.extend(["--outpaint-fill-color", str(args.outpaint_fill_color)])
    return argv


def _validate_padding_value(padding_value: str | None, *, option_name: str) -> None:
    if padding_value is None:
        _parser().error(f"{option_name} requires a padding value.")
    try:
        padding = BoxValues.parse(padding_value)
    except BoxValueError as exc:
        _parser().error(str(exc))
    try:
        values = [
            _padding_part_number(padding.top),
            _padding_part_number(padding.right),
            _padding_part_number(padding.bottom),
            _padding_part_number(padding.left),
        ]
    except ValueError as exc:
        _parser().error(str(exc))
    if any(value < 0 for value in values):
        _parser().error(f"{option_name} values must be zero or positive.")
    if not any(value > 0 for value in values):
        _parser().error(f"{option_name} must add pixels on at least one side.")


def _padding_part_number(value: int | str) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(value.strip("%"))
    except ValueError as exc:
        raise ValueError(f"Invalid padding value: {value}") from exc


def _resolve_route(
    args: argparse.Namespace,
    image_count: int,
    video_count: int,
    reference_image_count: int = 0,
) -> _Route:
    has_images = image_count > 0
    has_videos = video_count > 0
    model_config = _model_config(args.model, base_model=args.base_model)
    _validate_svi_route_request(args, model_config=model_config)
    plan = _resolve_generation_plan(
        args,
        image_count=image_count,
        video_count=video_count,
        reference_image_count=reference_image_count,
        model_config=model_config,
    )
    if args.svi_anchor_image is not None and plan.family != "wan":
        # Backstop for identities whose family only resolves during planning.
        _parser().error(
            "--svi-anchor-image requires a Wan A14B image-to-video model: SVI 2.0 Pro conditioning "
            f"does not exist on the resolved {plan.family} route."
        )
    _validate_controlnet_route_options(args, plan=plan)
    capability = _capability_row(args=args, model_config=model_config, capability_id=plan.capability_id)
    _validate_outpaint_fill_options(args, capability=capability)
    _validate_negative_prompt_option(args, capability=capability)
    _validate_lora_compatibility(args=args, model_config=model_config)
    _validate_family_override(args, model_config=model_config, plan=plan)
    route = _route_for_plan(
        plan.handler_id,
        has_image=has_images,
        has_video=has_videos,
        model_override=plan.model_override,
        control_model=plan.control_model,
    )
    return replace(route, accepts_outpaint_fill=capability is not None and capability.supports_outpaint_fill)


def _validate_svi_route_request(args: argparse.Namespace, *, model_config: ModelConfig | None) -> None:
    # Fail with the SVI truth BEFORE plan resolution can produce a misleading
    # image-to-image error for the anchor-sourced image count (0103).
    if args.svi_anchor_image is None or model_config is None:
        return
    try:
        family = get_model_capabilities(model_config=model_config).family
    except TaskInferenceError:
        return
    if family != "wan":
        _parser().error(
            "--svi-anchor-image requires a Wan A14B image-to-video model: SVI 2.0 Pro conditioning "
            f"does not exist on {family} routes."
        )


def _validate_controlnet_route_options(args: argparse.Namespace, *, plan) -> None:
    if not args.has_explicit_controlnet_model and not args.has_explicit_controlnet_strength:
        return
    if _plan_accepts_explicit_controlnet_options(plan):
        return
    if args.has_explicit_controlnet_model:
        _parser().error(
            "--controlnet-model is only supported on exact ControlNet routes selected by mlxgen generate. "
            "Use --controlnet-image-path for structured control, or a validated masked control-inpaint row."
        )
    _parser().error(
        "--controlnet-strength is only supported on exact ControlNet routes selected by mlxgen generate. "
        "Use --controlnet-image-path for structured control, or a validated masked control-inpaint row."
    )


def _plan_accepts_explicit_controlnet_options(plan) -> bool:
    return plan.capability_id in {"qwen.control", "qwen.control-inpaint"}


def _validate_negative_prompt_option(args: argparse.Namespace, *, capability) -> None:
    """Refuse --negative-prompt on a route whose capability row does not support it.

    Whether a negative prompt means anything is a property of the resolved route and, for FLUX.2
    Klein, of the weights: base Klein runs true classifier-free guidance and takes one, distilled
    Klein is step-distilled and has no guidance branch. Failing here names the route instead of
    letting the backend answer after the parse.
    """
    if capability is None or not getattr(args, "has_negative_prompt", False):
        return
    if capability.supports_negative_prompt:
        return
    detail = ""
    if capability.id.startswith("flux2."):
        detail = (
            " FLUX.2 Klein distilled weights are step-distilled and run no classifier-free guidance branch to "
            "steer; FLUX.2 Klein base models accept it with --guidance above 1.0."
        )
    _parser().error(
        f"--negative-prompt is not supported on {capability.id} for {args.model}.{detail} Drop --negative-prompt."
    )


def _validate_outpaint_fill_options(args: argparse.Namespace, *, capability) -> None:
    """Reject --outpaint-fill / --outpaint-fill-color the selected route cannot honour.

    `mlxgen generate` declares both options once for every model, so whether they mean anything
    is a property of the resolved route, not of the parser. Without this check the flags reach a
    backend that never declared them and argparse answers with "unrecognized arguments" under a
    program name the caller never typed.
    """
    if args.outpaint_fill is None and args.outpaint_fill_color is None and args.outpaint_passes is None:
        return
    if not args.has_outpaint:
        if args.outpaint_fill is None and args.outpaint_fill_color is None:
            _parser().error(
                "--outpaint-passes configures the --outpaint-padding run. Pass --outpaint-padding, or drop it."
            )
        _parser().error(
            "--outpaint-fill and --outpaint-fill-color configure the --outpaint-padding conditioning "
            "canvas. Pass --outpaint-padding, or drop these options."
        )
    if capability is None:
        return

    from mflux.outpaint import OutpaintError, check_outpaint_fill_options, outpaint_contract

    try:
        check_outpaint_fill_options(
            contract=outpaint_contract(capability=capability),
            fill=args.outpaint_fill,
            fill_color_requested=args.outpaint_fill_color is not None,
            passes=args.outpaint_passes,
        )
    except OutpaintError as exc:
        _parser().error(str(exc))


def _capability_row(
    args: argparse.Namespace,
    *,
    model_config: ModelConfig | None,
    capability_id: str,
):
    try:
        capabilities = get_model_capabilities(
            model=args.model,
            model_config=model_config,
            family=args.family,
            base_model=args.base_model,
        )
    except TaskInferenceError:
        return None
    for capability in capabilities.capabilities:
        if capability.id == capability_id:
            return capability
    return None


def _validate_lora_compatibility(args: argparse.Namespace, model_config: ModelConfig | None) -> None:
    if not args.has_lora or model_config is None:
        return
    try:
        LoRACompatibility.validate_for_model_config(
            model_config=model_config,
            selected_model=args.model,
            lora_paths=args.lora_paths,
        )
    except LoRAApplicationError as exc:
        _parser().error(str(exc))


def _resolve_generation_plan(
    args: argparse.Namespace,
    *,
    image_count: int,
    video_count: int,
    reference_image_count: int = 0,
    model_config: ModelConfig | None,
):
    try:
        return resolve_generation_plan(
            model=args.model,
            model_config=model_config,
            family=args.family,
            base_model=args.base_model,
            image_count=image_count,
            video_count=video_count,
            reference_image_count=reference_image_count,
            task=args.task,
            i2i_mode=args.i2i_mode,
            has_image_strength=args.has_image_strength,
            has_video_strength=args.has_video_strength,
            has_video_mask=args.has_video_mask,
            has_mask=args.has_mask,
            has_control_image=args.has_control_image,
            has_outpaint=args.has_outpaint,
            has_reframe=args.has_reframe,
            has_lora=args.has_lora,
        )
    except TaskInferenceError as exc:
        _parser().error(str(exc))


def _validate_family_override(
    args: argparse.Namespace,
    *,
    model_config: ModelConfig | None,
    plan,
) -> None:
    if args.family is None or model_config is not None:
        if args.family is not None and model_config is not None:
            try:
                actual_family = get_model_capabilities(model_config=model_config).family
            except TaskInferenceError as exc:
                _parser().error(str(exc))
            if actual_family != args.family:
                _parser().error(
                    f"--family {args.family} conflicts with model {args.model!r}, "
                    f"which resolves to family {actual_family}."
                )
        return
    handlers_requiring_model_config = {
        "flux2.generate",
        "flux2.edit",
        "ernie-image.generate",
        "fibo.generate",
        "z-image.generate",
        "z-image-turbo.generate",
        "fibo.edit",
        "bonsai.generate",
    }
    if plan.handler_id not in handlers_requiring_model_config:
        return
    _parser().error(
        f"--family {args.family} is not enough to configure model path {args.model!r} for {plan.mode}. "
        "Pass --base-model with a supported model alias so the backend can build the correct model config."
    )


def _route_for_plan(
    handler_id: str,
    *,
    has_image: bool,
    has_video: bool,
    model_override: str | None,
    control_model: str | None,
) -> _Route:
    if handler_id == "qwen.generate":
        return _qwen_route(control_model=control_model)
    if handler_id == "qwen.edit":
        return _qwen_edit_route(model_override=model_override)
    if handler_id == "flux2.generate":
        return _flux2_route()
    if handler_id == "flux2.edit":
        return _flux2_edit_route()
    if handler_id == "fibo.generate":
        return _fibo_route()
    if handler_id == "fibo.edit":
        return _fibo_edit_route(model_override=model_override)
    if handler_id == "z-image.generate":
        return _z_image_route()
    if handler_id == "z-image-turbo.generate":
        return _z_image_turbo_route()
    if handler_id == "ernie-image.generate":
        return _ernie_route()
    if handler_id == "wan.generate":
        return _wan_route(has_image=has_image, has_video=has_video)
    if handler_id == "minimax-h3.generate":
        return _minimax_h3_route(has_image=has_image)
    if handler_id == "bonsai.generate":
        return _bonsai_route()
    _parser().error(f"Unsupported generation handler {handler_id!r}.")


def _model_config(model: str | None, base_model: str | None = None) -> ModelConfig | None:
    if model is None:
        return None
    try:
        return ModelConfig.from_name(model, base_model=base_model)
    except ModelConfigError:
        return None


def _model_key(*parts: str | None) -> str:
    return " ".join(part for part in parts if part).lower().replace("_", "-")


def _has_alias(aliases: set[str], *needles: str) -> bool:
    return bool(aliases.intersection(needles))


def _is_qwen(aliases: set[str], model_key: str) -> bool:
    return (
        _has_alias(
            aliases,
            "qwen-image",
            "qwen-image-edit",
            "qwen-image-edit-2509",
            "qwen-image-edit-2511",
        )
        or "qwen" in model_key
    )


def _is_qwen_edit(aliases: set[str], model_key: str) -> bool:
    return _has_alias(
        aliases,
        "qwen-image-edit",
        "qwen-image-edit-2509",
        "qwen-image-edit-2511",
    ) or ("qwen" in model_key and "edit" in model_key)


def _is_flux2(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("flux2") or alias.startswith("klein") for alias in aliases) or any(
        token in model_key for token in ("flux2", "flux.2", "klein")
    )


def _is_bonsai(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("bonsai") for alias in aliases) or "bonsai" in model_key


def _is_fibo(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("fibo") for alias in aliases) or "fibo" in model_key


def _is_fibo_edit(aliases: set[str], model_key: str) -> bool:
    return _has_alias(aliases, "fibo-edit", "fibo-edit-rmbg") or ("fibo" in model_key and "edit" in model_key)


def _is_z_image(aliases: set[str], model_key: str) -> bool:
    return _has_alias(aliases, "z-image") or "z-image" in model_key or "zimage" in model_key


def _is_z_image_turbo(aliases: set[str], model_key: str) -> bool:
    return _has_alias(aliases, "z-image-turbo") or (
        ("z-image" in model_key or "zimage" in model_key) and "turbo" in model_key
    )


def _is_ernie(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("ernie") for alias in aliases) or "ernie" in model_key


def _is_wan(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("wan") for alias in aliases) or "wan" in model_key


def _is_minimax_h3(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("minimax-h3") for alias in aliases) or "minimax-h3" in model_key


def _is_seedvr2(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("seedvr2") for alias in aliases) or "seedvr2" in model_key


def _is_swiftvr(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("swiftvr") for alias in aliases) or "swiftvr" in model_key


def _qwen_route(control_model: str | None = None) -> _Route:
    from mflux.models.qwen.cli.qwen_image_generate import main as target_main

    return _Route(
        "mflux-generate-qwen",
        target_main,
        "--image-path",
        None,
        requires_image=False,
        control_model=control_model,
    )


def _qwen_edit_route(model_override: str | None = None) -> _Route:
    from mflux.models.qwen.cli.qwen_image_edit_generate import main as target_main

    return _Route(
        "mflux-generate-qwen-edit",
        target_main,
        "--image-paths",
        None,
        requires_image=True,
        model_override=model_override,
    )


def _flux2_route() -> _Route:
    from mflux.models.flux2.cli.flux2_generate import main as target_main

    return _Route("mflux-generate-flux2", target_main, "--image-path", None, requires_image=False)


def _flux2_edit_route() -> _Route:
    from mflux.models.flux2.cli.flux2_edit_generate import main as target_main

    return _Route("mflux-generate-flux2-edit", target_main, "--image-paths", None, requires_image=True)


def _fibo_route() -> _Route:
    from mflux.models.fibo.cli.fibo_generate import main as target_main

    return _Route("mflux-generate-fibo", target_main, "--image-path", None, requires_image=False)


def _fibo_edit_route(model_override: str | None = None) -> _Route:
    from mflux.models.fibo.cli.fibo_edit import main as target_main

    return _Route(
        "mflux-generate-fibo-edit",
        target_main,
        "--image-path",
        None,
        requires_image=True,
        model_override=model_override,
    )


def _z_image_route() -> _Route:
    from mflux.models.z_image.cli.z_image_generate import main as target_main

    return _Route("mflux-generate-z-image", target_main, "--image-path", None, requires_image=False)


def _z_image_turbo_route() -> _Route:
    from mflux.models.z_image.cli.z_image_turbo_generate import main as target_main

    return _Route("mflux-generate-z-image-turbo", target_main, "--image-path", None, requires_image=False)


def _ernie_route() -> _Route:
    from mflux.models.ernie_image.cli.ernie_image_generate import main as target_main

    return _Route(
        "mflux-generate-ernie-image",
        target_main,
        image_argument="--image-path",
        video_argument=None,
        requires_image=False,
    )


def _wan_route(has_image: bool, has_video: bool) -> _Route:
    from mflux.models.wan.cli.wan_generate import main as target_main

    return _Route(
        "mlxgen-generate-wan",
        target_main,
        image_argument="--image-path" if has_image else None,
        video_argument="--video-path" if has_video else None,
        requires_image=False,
    )


def _minimax_h3_route(has_image: bool) -> _Route:
    from mflux.models.minimax_h3.cli.minimax_h3_generate import main as target_main

    return _Route(
        "mlxgen-generate-minimax-h3",
        target_main,
        image_argument="--image-path" if has_image else None,
        video_argument=None,
        requires_image=False,
    )


def _bonsai_route() -> _Route:
    from mflux.models.bonsai_image.cli.bonsai_image_generate import main as target_main

    return _Route("mflux-generate-bonsai", target_main, image_argument=None, video_argument=None, requires_image=False)


if __name__ == "__main__":
    main()
