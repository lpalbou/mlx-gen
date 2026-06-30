import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.output_paths import resolve_output_path
from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.variants.edit.qwen_image_edit import (
    QwenImageEdit,
    QwenImageEdit as _QwenImageEditImplementation,
)
from mflux.utils.dimension_resolver import CANVAS_POLICY_EXACT_RESIZE
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.outpaint_util import OutpaintCanvas, OutpaintUtil
from mflux.utils.prompt_util import PromptUtil


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Qwen Image Edit with image conditioning.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_argument("--image-paths", type=Path, nargs="+", required=True, help="Local paths to one or more init images. For single image editing, provide one path. For multiple image editing, provide multiple paths.")  # fmt: off
    parser.add_mask_path_argument(
        help_text=(
            "Optional mask image path for localized Qwen edits. White pixels are repainted and black pixels are "
            "preserved."
        ),
    )
    parser.add_argument(
        "--reframe-padding",
        default=None,
        help=(
            "Generative reframe request: expand one source image by CSS-style "
            "top,right,bottom,left padding before edit generation."
        ),
    )
    parser.add_argument(
        "--outpaint-padding",
        "--image-outpaint-padding",
        dest="outpaint_padding",
        default=None,
        help=(
            "Expand one source image by CSS-style top,right,bottom,left padding and use an adaptive "
            "source blend when the generated source window still matches the original image."
        ),
    )
    parser.add_output_arguments()
    args = parser.parse_args()
    source_image_paths = [str(p) for p in args.image_paths]
    _validate_canvas_args(parser=parser, args=args, source_image_paths=source_image_paths)

    # 1. Load the model
    try:
        model_config = ModelConfig.from_name(args.model or "qwen-image-edit", base_model=args.base_model)
    except ModelConfigError:
        if args.model_path is None:
            raise
        model_config = ModelConfig.from_name(args.base_model or "qwen-image-edit")
    if len(source_image_paths) > 1 and not _QwenImageEditImplementation._is_edit_plus_model_config(
        model_config=model_config,
        image_paths=source_image_paths,
    ):
        parser.error(
            "Multiple Qwen edit reference images require an Edit-Plus model, such as "
            "qwen-image-edit-2509 or qwen-image-edit-2511."
        )
    if not _option_was_provided(sys.argv[1:], "--scheduler"):
        args.scheduler = "flow_match_euler_discrete"
    if args.guidance is None:
        if _QwenImageEditImplementation._is_edit_plus_model_config(
            model_config=model_config, image_paths=source_image_paths
        ):
            args.guidance = 4.0
        else:
            args.guidance = 4.0
    if not _option_was_provided(sys.argv[1:], "--steps") and _QwenImageEditImplementation._is_edit_plus_model_config(
        model_config=model_config,
        image_paths=source_image_paths,
    ):
        args.steps = 40

    CallbackManager.apply_runtime_memory_options(args)

    qwen = QwenImageEdit(
        quantize=args.quantize,
        model_config=model_config,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=qwen,
        latent_creator=QwenLatentCreator,
    )

    try:
        with TemporaryDirectory(prefix="mlxgen-outpaint-") as temporary_directory:
            try:
                image_paths, outpaint_canvas, reframe_canvas = _resolve_image_paths(
                    args=args,
                    source_image_paths=source_image_paths,
                    temporary_directory=Path(temporary_directory),
                )
            except ValueError as exc:
                parser.error(str(exc))

            try:
                for seed in args.seed:
                    events = CliRuntimeEventStream(
                        enabled=bool(args.json_events),
                        command="mlxgen generate",
                        model=model_config.model_name,
                        seed=seed,
                    )
                    # 4. Generate an image for each seed value
                    output_path = resolve_output_path(args.output, overwrite=args.replace, seed=seed)
                    events.set_output_path(output_path)
                    unsubscribe = events.subscribe_model(qwen, map_complete_to_generated=True)
                    try:
                        image = qwen.generate_image(
                            seed=seed,
                            prompt=PromptUtil.read_prompt(args),
                            negative_prompt=_read_negative_prompt(args),
                            width=args.width,
                            height=args.height,
                            guidance=args.guidance,
                            image_path=source_image_paths[0],  # Use original source for metadata
                            image_paths=image_paths,
                            mask_path=args.mask_path,
                            num_inference_steps=args.steps,
                            scheduler=args.scheduler,
                            canvas_policy=args.canvas_policy,
                        )
                        if outpaint_canvas is not None:
                            image.image = OutpaintUtil.composite_source_region(
                                generated_image=image.image,
                                canvas=outpaint_canvas,
                            )
                            image.image_paths = source_image_paths
                            OutpaintUtil.attach_metadata(
                                generated_image=image,
                                canvas=outpaint_canvas,
                                padding_value=args.outpaint_padding,
                            )
                        if reframe_canvas is not None:
                            image.image_paths = source_image_paths
                            OutpaintUtil.attach_reframe_metadata(
                                generated_image=image,
                                canvas=reframe_canvas,
                                padding_value=args.reframe_padding,
                            )

                        events.emit_save()
                        image.save(
                            path=output_path,
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

            except (StopImageGenerationException, PromptFileReadError) as exc:
                cli_print(str(exc), json_events=bool(args.json_events))
    finally:
        if memory_saver:
            cli_print(memory_saver.memory_stats(), json_events=bool(args.json_events))


def _resolve_image_paths(
    *,
    args,
    source_image_paths: list[str],
    temporary_directory: Path,
) -> tuple[list[str], OutpaintCanvas | None, OutpaintCanvas | None]:
    if args.outpaint_padding is None and args.reframe_padding is None:
        return source_image_paths, None, None
    padding_value = args.outpaint_padding or args.reframe_padding
    option_name = "--outpaint-padding" if args.outpaint_padding is not None else "--reframe-padding"
    canvas_name = "outpaint_canvas.png" if args.outpaint_padding is not None else "reframe_canvas.png"

    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_image_paths[0],
        padding_value=padding_value,
        output_path=temporary_directory / canvas_name,
        option_name=option_name,
    )
    args.width = canvas.target_width
    args.height = canvas.target_height
    args.canvas_policy = CANVAS_POLICY_EXACT_RESIZE
    if args.outpaint_padding is not None:
        return [str(canvas.canvas_path)], canvas, None
    return [str(canvas.canvas_path)], None, canvas


def _validate_canvas_args(*, parser: CommandLineParser, args, source_image_paths: list[str]) -> None:
    if args.mask_path is not None:
        if len(source_image_paths) != 1:
            parser.error("--mask-path requires exactly one --image-paths value.")
        if args.outpaint_padding is not None or args.reframe_padding is not None:
            parser.error("--mask-path cannot be combined with --reframe-padding or --outpaint-padding.")
    if args.outpaint_padding is None and args.reframe_padding is None:
        return
    if args.outpaint_padding is not None and args.reframe_padding is not None:
        parser.error("--reframe-padding and --outpaint-padding are different workflows and cannot be used together.")
    option_name = "--outpaint-padding" if args.outpaint_padding is not None else "--reframe-padding"
    if len(source_image_paths) != 1:
        parser.error(f"{option_name} requires exactly one --image-paths value.")
    if _any_option_was_provided(sys.argv[1:], ("--width", "--height")):
        parser.error(f"{option_name} computes --width and --height from the source image; do not pass either option.")
    if _option_was_provided(sys.argv[1:], "--canvas-policy"):
        parser.error(f"{option_name} uses --canvas-policy exact-resize; do not pass --canvas-policy.")


def _read_negative_prompt(args) -> str | None:
    if _any_option_was_provided(sys.argv[1:], ("--negative-prompt", "--negative")):
        return PromptUtil.read_negative_prompt(args)
    return None


def _any_option_was_provided(argv: list[str], option_names: tuple[str, ...]) -> bool:
    return any(_option_was_provided(argv, option_name) for option_name in option_names)


def _option_was_provided(argv: list[str], option_name: str) -> bool:
    for token in argv:
        if token == option_name or token.startswith(f"{option_name}="):
            return True
    return False


if __name__ == "__main__":
    main()
