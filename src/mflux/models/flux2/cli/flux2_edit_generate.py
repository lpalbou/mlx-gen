import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.output_paths import resolve_output_path
from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.variants import Flux2KleinEdit
from mflux.models.flux2.variants.edit.flux2_klein_inpaint import Flux2KleinInpaint
from mflux.models.flux2.variants.edit.flux2_klein_outpaint import Flux2KleinOutpaint
from mflux.utils.dimension_resolver import CANVAS_POLICY_EXACT_RESIZE
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.outpaint_util import OutpaintCanvas, OutpaintUtil
from mflux.utils.prompt_util import PromptUtil

LEGACY_NOTICE = (
    "Warning: mflux-generate-flux2-edit is a legacy compatibility command. "
    "Use `mlxgen generate --model <model> --image <path> ...` for new integrations."
)

FLUX2_GREEN_BORDER_OUTPAINT_LORA_MARKERS = (
    "fal/flux-2-klein-4b-outpaint-lora",
    "ming3d/flux-2-klein-4b-outpaint-lora",
    "flux-outpaint-lora.safetensors",
)


def _print_legacy_notice() -> None:
    print(LEGACY_NOTICE, file=sys.stderr)


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(
        description=(
            "Legacy compatibility command for FLUX.2 Klein image conditioning and edit workflows. "
            "Prefer `mlxgen generate --model <model> --image <path> ...` for new integrations."
        ),
        epilog=("Preferred migration target: mlxgen generate --model <flux2-model> --image <path> ..."),
    )
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_argument("--image-paths", type=Path, nargs="+", required=True, help="Local paths to one or more init images. For single image editing, provide one path. For multiple image editing, provide multiple paths.")  # fmt: off
    parser.add_mask_path_argument(
        help_text=(
            "Optional mask image path for localized FLUX.2 Klein masked edit. White pixels are repainted "
            "and black pixels are preserved."
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
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_output_arguments()
    args = parser.parse_args()
    _print_legacy_notice()

    if getattr(args, "negative_prompt", ""):
        parser.error(
            "--negative-prompt is not supported for FLUX.2. Omit it for FLUX.2 routes. "
            "For new integrations, call `mlxgen generate --model <flux2-model> --image <path> ...` "
            "instead of `mflux-generate-flux2-edit`."
        )
    source_image_paths = [Path(p) for p in args.image_paths]
    _validate_canvas_args(parser=parser, args=args, source_image_paths=source_image_paths)

    model_name = args.model or "flux2-klein-4b"
    model_config = ModelConfig.from_name(model_name=model_name, base_model=args.base_model)

    is_base_model = _is_flux2_base_model(model_config)
    uses_masked_edit = args.mask_path is not None
    if args.guidance is None:
        uses_source_locked_denoise = args.outpaint_padding is not None or uses_masked_edit
        args.guidance = 4.0 if uses_source_locked_denoise and is_base_model else 1.0
    model_name_lower = model_config.model_name.lower()
    base_model_lower = (model_config.base_model or "").lower()
    is_flux2 = any(
        identifier in model_name_lower or identifier in base_model_lower for identifier in ("flux.2", "flux2")
    )
    if args.reframe_padding is not None and is_base_model:
        parser.error("--reframe-padding requires a validated non-base FLUX.2 Klein model.")
    if args.outpaint_padding is not None and not is_base_model:
        parser.error(
            "--outpaint-padding requires a FLUX.2 Klein base model because strict outpaint "
            "needs source-locked denoising. Use a base model such as "
            "black-forest-labs/FLUX.2-klein-base-9B or "
            "AbstractFramework/flux.2-klein-base-9b-8bit."
        )
    if args.guidance != 1.0 and not is_flux2:
        parser.error("--guidance is only supported for FLUX.2 models. Use --guidance 1.0.")
    if args.guidance != 1.0 and not is_base_model and args.outpaint_padding is None:
        parser.error("--guidance is only supported for FLUX.2 base models. Use --guidance 1.0.")

    CallbackManager.apply_runtime_memory_options(args)

    uses_strict_outpaint = args.outpaint_padding is not None and is_base_model
    model_kwargs = {
        "model_config": model_config,
        "quantize": args.quantize,
        "model_path": args.model_path,
        "lora_paths": args.lora_paths,
        "lora_scales": args.lora_scales,
    }
    if uses_masked_edit:
        model = Flux2KleinInpaint(**model_kwargs)
    elif uses_strict_outpaint:
        model = Flux2KleinOutpaint(**model_kwargs)
    else:
        model = Flux2KleinEdit(**model_kwargs)

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Flux2LatentCreator,
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
                    output_path = resolve_output_path(args.output, overwrite=args.replace, seed=seed)
                    events.set_output_path(output_path)
                    unsubscribe = events.subscribe_model(model, map_complete_to_generated=True)
                    try:
                        if uses_masked_edit:
                            image = model.generate_image(
                                seed=seed,
                                prompt=PromptUtil.read_prompt(args),
                                image_path=image_paths[0],
                                mask_path=args.mask_path,
                                reference_image_paths=image_paths[1:] or None,
                                width=args.width,
                                height=args.height,
                                guidance=args.guidance,
                                num_inference_steps=args.steps,
                                scheduler="flow_match_euler_discrete",
                                canvas_policy=args.canvas_policy,
                            )
                        elif uses_strict_outpaint:
                            image = model.generate_image(
                                seed=seed,
                                prompt=PromptUtil.read_prompt(args),
                                canvas=outpaint_canvas,
                                guidance=args.guidance,
                                num_inference_steps=args.steps,
                                scheduler="flow_match_euler_discrete",
                            )
                        else:
                            image = model.generate_image(
                                seed=seed,
                                prompt=PromptUtil.read_prompt(args),
                                width=args.width,
                                height=args.height,
                                guidance=args.guidance,
                                image_paths=image_paths,
                                num_inference_steps=args.steps,
                                scheduler="flow_match_euler_discrete",
                                canvas_policy=args.canvas_policy,
                            )
                        if outpaint_canvas is not None:
                            image.image = OutpaintUtil.composite_source_region(
                                generated_image=image.image,
                                canvas=outpaint_canvas,
                                feather_px=None,
                                restore_threshold=-1.0 if uses_strict_outpaint else 12.0,
                            )
                            image.image_path = source_image_paths[0]
                            image.image_paths = source_image_paths
                            OutpaintUtil.attach_metadata(
                                generated_image=image,
                                canvas=outpaint_canvas,
                                padding_value=args.outpaint_padding,
                                preservation=(
                                    "latent-locked-transition-band-no-postblend"
                                    if uses_strict_outpaint
                                    else "adaptive-content-aware-source-blend"
                                ),
                            )
                        if reframe_canvas is not None:
                            image.image_path = source_image_paths[0]
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
    source_image_paths: list[Path],
    temporary_directory: Path,
) -> tuple[list[Path], OutpaintCanvas | None, OutpaintCanvas | None]:
    if args.outpaint_padding is None and args.reframe_padding is None:
        return source_image_paths, None, None
    padding_value = args.outpaint_padding or args.reframe_padding
    option_name = "--outpaint-padding" if args.outpaint_padding is not None else "--reframe-padding"
    canvas_name = "outpaint_canvas.png" if args.outpaint_padding is not None else "reframe_canvas.png"
    if len(source_image_paths) != 1:
        raise ValueError(f"{option_name} requires exactly one --image-paths value.")

    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_image_paths[0],
        padding_value=padding_value,
        output_path=temporary_directory / canvas_name,
        option_name=option_name,
        fill_mode=_outpaint_fill_mode(args=args),
        fill_color=_outpaint_fill_color(args=args),
    )
    args.width = canvas.target_width
    args.height = canvas.target_height
    args.canvas_policy = CANVAS_POLICY_EXACT_RESIZE
    if args.outpaint_padding is not None:
        return [canvas.canvas_path], canvas, None
    return [canvas.canvas_path], None, canvas


def _outpaint_fill_mode(*, args) -> str:
    if args.outpaint_padding is not None and _uses_green_border_outpaint_lora(args.lora_paths):
        return "solid"
    return "edge"


def _outpaint_fill_color(*, args) -> tuple[int, int, int]:
    if args.outpaint_padding is not None and _uses_green_border_outpaint_lora(args.lora_paths):
        return (0, 255, 0)
    return (255, 255, 255)


def _uses_green_border_outpaint_lora(lora_paths: list[str] | None) -> bool:
    if not lora_paths:
        return False
    for path in lora_paths:
        normalized = path.lower()
        if any(marker in normalized for marker in FLUX2_GREEN_BORDER_OUTPAINT_LORA_MARKERS):
            return True
    return False


def _validate_canvas_args(*, parser: CommandLineParser, args, source_image_paths: list[Path]) -> None:
    if args.mask_path is not None:
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


def _any_option_was_provided(argv: list[str], option_names: tuple[str, ...]) -> bool:
    return any(_option_was_provided(argv, option_name) for option_name in option_names)


def _option_was_provided(argv: list[str], option_name: str) -> bool:
    for token in argv:
        if token == option_name or token.startswith(f"{option_name}="):
            return True
    return False


def _is_flux2_base_model(model_config: ModelConfig) -> bool:
    model_name_lower = model_config.model_name.lower()
    base_model_lower = (model_config.base_model or "").lower()
    return "klein-base" in model_name_lower or "klein-base" in base_model_lower


if __name__ == "__main__":
    main()
