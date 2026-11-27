import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.outpaint_cli import (
    any_option_was_provided,
    emit_canvas_notices,
    option_was_provided,
    prepare_canvas_session,
)
from mflux.cli.output_paths import resolve_output_path
from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print
from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.variants import Flux2KleinEdit
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
from mflux.models.flux2.variants.edit.flux2_klein_inpaint import Flux2KleinInpaint
from mflux.models.flux2.variants.edit.flux2_klein_outpaint import Flux2KleinOutpaint
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

LEGACY_NOTICE = (
    "Warning: mflux-generate-flux2-edit is a legacy compatibility command. "
    "Use `mlxgen generate --model <model> --image <path> ...` for new integrations."
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
            "Strict outpaint: expand one source image by CSS-style top,right,bottom,left padding, "
            "lock the source region in latent space behind a narrow transition band while the "
            "added area is denoised, then paste the original crop back where the generated window "
            "still matches it. Guidance defaults to 4.0 on base Klein and 1.0 on distilled."
        ),
    )
    parser.add_outpaint_fill_arguments()
    parser.add_outpaint_pass_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_output_arguments()
    args = parser.parse_args()
    _print_legacy_notice()

    source_image_paths = [Path(p) for p in args.image_paths]
    _validate_canvas_args(parser=parser, args=args, source_image_paths=source_image_paths)

    model_name = args.model or "flux2-klein-4b"
    model_config = ModelConfig.from_name(model_name=model_name, base_model=args.base_model)

    is_base_model = _is_flux2_base_model(model_config)
    uses_masked_edit = args.mask_path is not None
    negative_prompt = PromptUtil.read_negative_prompt(args)
    if args.guidance is None:
        # Source-locked denoising (strict outpaint, masked edit) is the one place a FLUX.2 route
        # wants the model's own guidance default instead of a flat 1.0: base weights run true CFG,
        # and distilled weights are step-distilled and stay at 1.0. A negative prompt on base
        # weights asks for that guidance branch too, so it takes the base default as well.
        uses_source_locked_denoise = args.outpaint_padding is not None or uses_masked_edit
        wants_guidance_branch = uses_source_locked_denoise or (is_base_model and bool(negative_prompt))
        args.guidance = _Flux2KleinEditHelpers.default_guidance(model_config) if wants_guidance_branch else 1.0
    try:
        # Base Klein runs true CFG and takes a negative prompt with guidance above 1.0; distilled
        # Klein is step-distilled and has no guidance branch, so the option is refused there
        # before any weight loads.
        _Flux2KleinEditHelpers.validate_negative_prompt(
            model_config=model_config, guidance=args.guidance, negative_prompt=negative_prompt
        )
    except ValueError as exc:
        parser.error(
            f"{exc} For new integrations, call `mlxgen generate --model <flux2-model> --image <path> ...` "
            "instead of `mflux-generate-flux2-edit`."
        )
    model_name_lower = ConfigResolution.identifying_name(model_config.model_name).lower()
    base_model_lower = (model_config.base_model or "").lower()
    is_flux2 = any(
        identifier in model_name_lower or identifier in base_model_lower for identifier in ("flux.2", "flux2")
    )
    if args.reframe_padding is not None and is_base_model:
        parser.error("--reframe-padding requires a validated non-base FLUX.2 Klein model.")
    if args.guidance != 1.0 and not is_flux2:
        parser.error("--guidance is only supported for FLUX.2 models. Use --guidance 1.0.")
    if args.guidance != 1.0 and not is_base_model:
        # Distilled Klein outpaints through the same latent-locked route as base Klein, but it is
        # step-distilled: CFG above 1.0 is out of distribution on those weights on every route.
        parser.error("--guidance is only supported for FLUX.2 base models. Use --guidance 1.0.")

    CallbackManager.apply_runtime_memory_options(args)

    uses_strict_outpaint = args.outpaint_padding is not None
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
            # The conditioning canvas, the fill policy, the guard and the metadata are the shared
            # outpaint layer's; this command only supplies the parsed request and the model.
            try:
                canvas_session = prepare_canvas_session(
                    args=args,
                    source_image_paths=source_image_paths,
                    workspace=temporary_directory,
                    model_config=model_config,
                )
            except ValueError as exc:
                parser.error(str(exc))
            emit_canvas_notices(canvas_session)
            image_paths = canvas_session.conditioning_image_paths if canvas_session is not None else source_image_paths

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
                                negative_prompt=negative_prompt,
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
                        elif canvas_session is not None:
                            image = canvas_session.generate(
                                model,
                                seed=seed,
                                prompt=PromptUtil.read_prompt(args),
                                negative_prompt=negative_prompt,
                                guidance=args.guidance,
                                num_inference_steps=args.steps,
                                scheduler="flow_match_euler_discrete",
                            )
                        else:
                            image = model.generate_image(
                                seed=seed,
                                prompt=PromptUtil.read_prompt(args),
                                negative_prompt=negative_prompt,
                                width=args.width,
                                height=args.height,
                                guidance=args.guidance,
                                image_paths=image_paths,
                                num_inference_steps=args.steps,
                                scheduler="flow_match_euler_discrete",
                                canvas_policy=args.canvas_policy,
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


def _validate_canvas_args(*, parser: CommandLineParser, args, source_image_paths: list[Path]) -> None:
    argv = sys.argv[1:]
    fill_options_provided = any_option_was_provided(argv, ("--outpaint-fill", "--outpaint-fill-color"))
    if fill_options_provided and args.outpaint_padding is None:
        parser.error(
            "--outpaint-fill and --outpaint-fill-color configure the --outpaint-padding conditioning "
            "canvas. Pass --outpaint-padding, or drop these options."
        )
    if option_was_provided(argv, "--outpaint-passes") and args.outpaint_padding is None:
        parser.error("--outpaint-passes configures the --outpaint-padding run. Pass --outpaint-padding, or drop it.")
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
    if any_option_was_provided(argv, ("--width", "--height")):
        parser.error(f"{option_name} computes --width and --height from the source image; do not pass either option.")
    if option_was_provided(argv, "--canvas-policy"):
        parser.error(f"{option_name} uses --canvas-policy exact-resize; do not pass --canvas-policy.")


def _is_flux2_base_model(model_config: ModelConfig) -> bool:
    return _Flux2KleinEditHelpers.is_base_model(model_config)


if __name__ == "__main__":
    main()
