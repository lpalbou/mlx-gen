import sys

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.output_paths import resolve_output_path
from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print
from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.variants import Flux2Klein
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

LEGACY_NOTICE = (
    "Warning: mflux-generate-flux2 is a legacy compatibility command. "
    "Use `mlxgen generate --model <model> ...` for new integrations."
)


def _print_legacy_notice() -> None:
    print(LEGACY_NOTICE, file=sys.stderr)


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(
        description=(
            "Legacy compatibility command for FLUX.2 Klein image generation. "
            "Prefer `mlxgen generate --model <model> ...` for new integrations."
        ),
        epilog="Preferred migration target: mlxgen generate --model <flux2-model> --prompt ...",
    )
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_resize_mode_argument()
    parser.add_output_arguments()
    args = parser.parse_args()
    _print_legacy_notice()

    model_name = args.model or "flux2-klein-4b"
    model_config = ModelConfig.from_name(model_name=model_name, base_model=args.base_model)

    is_distilled = "base" not in ConfigResolution.identifying_name(model_config.model_name).lower()
    negative_prompt = PromptUtil.read_negative_prompt(args)
    if args.guidance is None:
        # A negative prompt on base weights asks for the guidance branch, which only runs above
        # 1.0; without one every FLUX.2 route defaults to 1.0.
        wants_guidance_branch = bool(negative_prompt) and not is_distilled
        args.guidance = _Flux2KleinEditHelpers.default_guidance(model_config) if wants_guidance_branch else 1.0
    if args.guidance != 1.0 and is_distilled:
        parser.error("--guidance is only supported for FLUX.2 base models. Use --guidance 1.0.")
    try:
        _Flux2KleinEditHelpers.validate_negative_prompt(
            model_config=model_config, guidance=args.guidance, negative_prompt=negative_prompt
        )
    except ValueError as exc:
        parser.error(
            f"{exc} For new integrations, call `mlxgen generate --model <flux2-model> ...` instead of "
            "`mflux-generate-flux2`."
        )

    CallbackManager.apply_runtime_memory_options(args)

    model = Flux2Klein(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Flux2LatentCreator,
    )

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
                image = model.generate_image(
                    seed=seed,
                    prompt=PromptUtil.read_prompt(args),
                    negative_prompt=negative_prompt,
                    width=args.width,
                    height=args.height,
                    guidance=args.guidance,
                    image_path=args.image_path,
                    num_inference_steps=args.steps,
                    image_strength=args.image_strength,
                    scheduler="flow_match_euler_discrete",
                    canvas_policy=args.canvas_policy,
                    resize_mode=args.resize_mode,
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


if __name__ == "__main__":
    main()
