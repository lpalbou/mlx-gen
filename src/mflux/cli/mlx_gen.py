import argparse
import json
import shlex
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

from mflux.models.common.config import ModelConfig
from mflux.models.common.download_policy import allow_downloads, is_huggingface_repo_id
from mflux.utils.exceptions import ModelConfigError


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
    requires_image: bool
    model_override: str | None = None


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _top_level_parser().print_help()
        return

    if argv and argv[0] in {"download", "prepare"}:
        _run_model_command(argv)
        return

    invocation = _resolve_invocation(_normalize_command(argv))
    previous_argv = sys.argv
    try:
        sys.argv = invocation.argv
        try:
            invocation.target_main()
        except FileNotFoundError as exc:
            print(exc)
            raise SystemExit(1) from None
    finally:
        sys.argv = previous_argv


def _resolve_invocation(argv: list[str]) -> RouterInvocation:
    args, forwarded = _parse_router_args(argv)
    images = _collect_images(args)
    route = _resolve_route(args, image_count=len(images))
    if route.requires_image and not images:
        _parser().error(f"{route.target_name} requires --image or --images.")

    forwarded_model = route.model_override or args.model
    normalized_argv = [route.target_name]
    if forwarded_model is not None:
        normalized_argv.extend(["--model", forwarded_model])
    if route.image_argument is not None and images:
        normalized_argv.append(route.image_argument)
        normalized_argv.extend(images)
    normalized_argv.extend(forwarded)

    return RouterInvocation(
        target_name=route.target_name,
        target_main=route.target_main,
        argv=normalized_argv,
    )


def _parse_router_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = _parser()
    args, forwarded = parser.parse_known_args(argv)
    metadata = _read_metadata_config(parser, argv)
    if args.model is None:
        args.model = metadata.get("model")
    _reject_prepare_path_in_generate(parser, args, forwarded)
    args.metadata_images = _metadata_images(metadata)
    args.task = _normalize_task(args.task)
    args.has_image_strength = _option_was_provided(argv, "--image-strength")
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
        usage="mlxgen [generate|download|prepare] ...",
        description="Prepare local model assets and generate or edit images with MLX-Gen.",
        epilog=(
            "Commands:\n"
            "  generate    Generate or edit images and videos from a prepared or cached model.\n"
            "  download    Explicitly download a model snapshot into the Hugging Face cache.\n"
            "  prepare     Create a reusable local MLX-Gen model folder, optionally quantized.\n"
            "\n"
            "Examples:\n"
            "  mlxgen generate --model z-image-turbo --prompt 'A puffin standing on a cliff'\n"
            "  mlxgen generate --model wan2.2-ti2v-5b --task text-to-video --prompt 'A city timelapse'\n"
            "  mlxgen download --model Qwen/Qwen-Image\n"
            "  mlxgen prepare --model Qwen/Qwen-Image --path ./models/qwen-image-8bit --quantize 8\n"
            "\n"
            "Use 'mlxgen <command> --help' for command-specific options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser


def _normalize_task(task: str) -> str:
    aliases = {
        "txt2img": "text-to-image",
        "img2img": "image-to-image",
        "txt2vid": "text-to-video",
        "t2v": "text-to-video",
        "img2vid": "image-to-video",
        "i2v": "image-to-video",
    }
    return aliases.get(task, task)


def _option_was_provided(argv: list[str], option_name: str) -> bool:
    for token in argv:
        if token == option_name or token.startswith(f"{option_name}="):
            return True
    return False


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxgen generate",
        description=(
            "Generate or edit images and videos with MLX-Gen. The command routes to the right model backend "
            "from --model and from whether input images are supplied."
        ),
        epilog=(
            "Common generation options are forwarded to the selected backend, including --prompt, "
            "--prompt-file, --width, --height, --steps, --guidance, --seed, --auto-seeds, "
            "--negative-prompt, --quantize, --lora-paths, --lora-scales, --metadata, "
            "--config-from-metadata/-C, --output, --replace, --frames, and --fps."
        ),
    )
    parser.add_argument("--model", "-m", type=str, help="Model alias, Hugging Face repo, or local model path.")
    parser.add_argument(
        "--family",
        choices=["qwen", "flux2", "fibo", "z-image", "ernie-image", "wan"],
        default=None,
        help="Override model-family detection for local paths or custom repo names.",
    )
    parser.add_argument(
        "--task",
        choices=[
            "auto",
            "text-to-image",
            "txt2img",
            "image-to-image",
            "img2img",
            "edit",
            "text-to-video",
            "txt2vid",
            "t2v",
            "image-to-video",
            "img2vid",
            "i2v",
        ],
        default="auto",
        help="Override automatic routing. Default: auto.",
    )
    parser.add_argument(
        "--image",
        "--input-image",
        "-i",
        dest="images",
        action="append",
        default=[],
        help="Input image for image-to-image or edit. Repeat for multi-image edit.",
    )
    parser.add_argument(
        "--images",
        "--input-images",
        "--image-paths",
        dest="image_groups",
        nargs="+",
        action="append",
        default=[],
        help="One or more input images for editing.",
    )
    parser.add_argument(
        "--image-path",
        dest="image_path",
        default=None,
        help="Compatibility alias for a single input image.",
    )
    return parser


def _run_model_command(argv: list[str]) -> None:
    command = argv[0]
    if command == "download":
        _download_model(argv[1:])
        return
    if command == "prepare":
        _prepare_model(argv[1:])
        return
    raise AssertionError("unreachable")


def _download_model(argv: list[str]) -> None:
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

    patterns = None if args.all_files else _download_patterns(model_config, repo_id)
    print(f"Downloading {repo_id} into the Hugging Face cache...")
    with allow_downloads():
        path = snapshot_download(repo_id=repo_id, allow_patterns=patterns)
    print(f"Downloaded snapshot: {path}")
    print("You can now run generation without a runtime download, for example:")
    print(
        "  mlxgen generate --model "
        f"{shlex.quote(repo_id)} --prompt 'A product photo of a ceramic teapot' --output image.png"
    )


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
    weight_definition = _weight_definition_for(aliases, model_key)
    if weight_definition is None:
        return None

    patterns = list(weight_definition.get_download_patterns())
    for tokenizer in weight_definition.get_tokenizers():
        patterns.extend(tokenizer.download_patterns or [f"{tokenizer.hf_subdir}/**"])
    return sorted(set(patterns))


def _weight_definition_for(aliases: set[str], model_key: str):
    if _is_qwen(aliases, model_key):
        from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

        return QwenWeightDefinition
    if _is_flux2(aliases, model_key):
        from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

        return Flux2KleinWeightDefinition
    if _is_fibo(aliases, model_key):
        from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition

        return FIBOWeightDefinition
    if _is_z_image(aliases, model_key):
        from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

        return ZImageWeightDefinition
    if _is_ernie(aliases, model_key):
        from mflux.models.ernie_image.weights.ernie_image_weight_definition import ErnieImageWeightDefinition

        return ErnieImageWeightDefinition
    if _is_wan(aliases, model_key):
        from mflux.models.wan.weights import WanWeightDefinition

        return WanWeightDefinition
    return None


def _collect_images(args: argparse.Namespace) -> list[str]:
    images = []
    if args.image_path is not None:
        images.append(args.image_path)
    images.extend(args.images)
    for group in args.image_groups:
        images.extend(group)
    return images or args.metadata_images


def _resolve_route(args: argparse.Namespace, image_count: int) -> _Route:
    model_config = _model_config(args.model)
    aliases = set(model_config.aliases) if model_config is not None else set()
    model_name = model_config.model_name if model_config is not None else args.model
    model_key = _model_key(model_name, args.model)
    has_images = image_count > 0
    has_multiple_images = image_count > 1
    explicit_img2img = args.task == "image-to-image" or args.has_image_strength

    if args.family == "qwen":
        if args.task == "edit" or _is_qwen_edit(aliases, model_key) or (has_multiple_images and not explicit_img2img):
            return _qwen_edit_route()
        return _qwen_route()

    if args.family == "flux2":
        if args.task == "edit" or (args.task == "auto" and has_images and not explicit_img2img):
            return _flux2_edit_route()
        return _flux2_route()

    if args.family == "fibo":
        if args.task == "edit" or _is_fibo_edit(aliases, model_key):
            return _fibo_edit_route()
        return _fibo_route()

    if args.family == "z-image":
        if _is_z_image_turbo(aliases, model_key):
            return _z_image_turbo_route()
        return _z_image_route()

    if args.family == "ernie-image":
        _reject_ernie_unsupported_inputs(args, image_count)
        return _ernie_route()

    if args.family == "wan":
        _reject_wan_unsupported_inputs(args, image_count)
        return _wan_route(has_image=has_images)

    if _is_qwen_edit(aliases, model_key) or (args.task == "edit" and _is_qwen(aliases, model_key)):
        return _qwen_edit_route(model_override=None if _is_qwen_edit(aliases, model_key) else "qwen-image-edit")

    if _is_qwen(aliases, model_key):
        if args.task == "text-to-image":
            return _qwen_route()
        if explicit_img2img or (args.task == "auto" and not has_multiple_images):
            return _qwen_route()
        return _qwen_edit_route(model_override="qwen-image-edit")

    if _is_flux2(aliases, model_key):
        if args.task == "edit" or (args.task == "auto" and has_images and not explicit_img2img):
            return _flux2_edit_route()
        return _flux2_route()

    if _is_fibo_edit(aliases, model_key) or (args.task == "edit" and _is_fibo(aliases, model_key)):
        return _fibo_edit_route(model_override=None if _is_fibo_edit(aliases, model_key) else "fibo-edit")

    if _is_fibo(aliases, model_key):
        return _fibo_route()

    if _is_z_image_turbo(aliases, model_key):
        return _z_image_turbo_route()

    if _is_z_image(aliases, model_key):
        return _z_image_route()

    if _is_ernie(aliases, model_key):
        _reject_ernie_unsupported_inputs(args, image_count)
        return _ernie_route()

    if _is_wan(aliases, model_key):
        _reject_wan_unsupported_inputs(args, image_count)
        return _wan_route(has_image=has_images)

    _parser().error(
        f"Could not infer a supported backend from --model {args.model!r}. "
        "Use a model name containing qwen, flux2/flux.2/klein, fibo, z-image, ernie, or wan, or pass --family."
    )
    raise AssertionError("unreachable")


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
    return _has_alias(aliases, "qwen-image", "qwen-image-edit") or "qwen" in model_key


def _is_qwen_edit(aliases: set[str], model_key: str) -> bool:
    return _has_alias(aliases, "qwen-image-edit") or ("qwen" in model_key and "edit" in model_key)


def _is_flux2(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("flux2") or alias.startswith("klein") for alias in aliases) or any(
        token in model_key for token in ("flux2", "flux.2", "klein")
    )


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


def _reject_ernie_unsupported_inputs(args: argparse.Namespace, image_count: int) -> None:
    if args.task == "edit":
        _parser().error(
            "ERNIE Image Turbo supports text-to-image and experimental single-image image-to-image only. "
            "Multi-image edit is not supported."
        )
    if image_count > 1:
        _parser().error(
            "ERNIE Image Turbo supports only one input image. "
            "Pass a single --image/--image-path for experimental image-to-image."
        )
    if args.task == "text-to-image" and image_count:
        _parser().error("ERNIE Image Turbo text-to-image cannot be combined with --image/--image-path.")
    if args.task == "image-to-image" and image_count == 0:
        _parser().error("ERNIE Image Turbo image-to-image requires --image or --image-path.")
    if args.has_image_strength and image_count == 0:
        _parser().error("ERNIE Image Turbo --image-strength requires --image or --image-path.")


def _reject_wan_unsupported_inputs(args: argparse.Namespace, image_count: int) -> None:
    if args.task in {"edit", "text-to-image", "image-to-image"}:
        _parser().error(
            "Wan2.2 TI2V supports text-to-video and image-to-video tasks. "
            "Use --task text-to-video or --task image-to-video."
        )
    if args.task == "image-to-video" or image_count:
        _parser().error(
            "Wan2.2 image-to-video is not enabled yet. Text-to-video works now; "
            "I2V needs the Diffusers first-frame latent conditioning path."
        )


def _qwen_route() -> _Route:
    from mflux.models.qwen.cli.qwen_image_generate import main as target_main

    return _Route("mflux-generate-qwen", target_main, "--image-path", requires_image=False)


def _qwen_edit_route(model_override: str | None = None) -> _Route:
    from mflux.models.qwen.cli.qwen_image_edit_generate import main as target_main

    return _Route(
        "mflux-generate-qwen-edit", target_main, "--image-paths", requires_image=True, model_override=model_override
    )


def _flux2_route() -> _Route:
    from mflux.models.flux2.cli.flux2_generate import main as target_main

    return _Route("mflux-generate-flux2", target_main, "--image-path", requires_image=False)


def _flux2_edit_route() -> _Route:
    from mflux.models.flux2.cli.flux2_edit_generate import main as target_main

    return _Route("mflux-generate-flux2-edit", target_main, "--image-paths", requires_image=True)


def _fibo_route() -> _Route:
    from mflux.models.fibo.cli.fibo_generate import main as target_main

    return _Route("mflux-generate-fibo", target_main, "--image-path", requires_image=False)


def _fibo_edit_route(model_override: str | None = None) -> _Route:
    from mflux.models.fibo.cli.fibo_edit import main as target_main

    return _Route(
        "mflux-generate-fibo-edit", target_main, "--image-path", requires_image=True, model_override=model_override
    )


def _z_image_route() -> _Route:
    from mflux.models.z_image.cli.z_image_generate import main as target_main

    return _Route("mflux-generate-z-image", target_main, "--image-path", requires_image=False)


def _z_image_turbo_route() -> _Route:
    from mflux.models.z_image.cli.z_image_turbo_generate import main as target_main

    return _Route("mflux-generate-z-image-turbo", target_main, "--image-path", requires_image=False)


def _ernie_route() -> _Route:
    from mflux.models.ernie_image.cli.ernie_image_generate import main as target_main

    return _Route(
        "mflux-generate-ernie-image",
        target_main,
        image_argument="--image-path",
        requires_image=False,
    )


def _wan_route(has_image: bool) -> _Route:
    from mflux.models.wan.cli.wan_generate import main as target_main

    return _Route(
        "mlxgen-generate-wan",
        target_main,
        image_argument="--image-path" if has_image else None,
        requires_image=False,
    )


if __name__ == "__main__":
    main()
