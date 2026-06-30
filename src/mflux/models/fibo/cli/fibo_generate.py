from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.output_paths import resolve_output_path
from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.fibo.variants.txt2img.util import FiboUtil
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def _resolve_fibo_model_config(parser: CommandLineParser, args) -> ModelConfig:
    try:
        model_config = ModelConfig.from_name(model_name=args.model, base_model=args.base_model)
    except ModelConfigError:
        model_config = ModelConfig.fibo()

    compatible_aliases = {"fibo", "fibo-lite"}
    if not compatible_aliases.intersection(model_config.aliases):
        parser.error(
            "mflux-generate-fibo requires a base FIBO or FIBO Lite model; "
            "metadata or --model resolved to an incompatible model."
        )
    return model_config


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using FIBO model.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.set_defaults(model="fibo")
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    args = parser.parse_args()

    if args.image_path is not None:
        parser.error(
            "Base FIBO does not expose a validated latent image-to-image path. "
            "Use mflux-generate-fibo-edit with a FIBO Edit model for image-conditioned editing."
        )

    model_config = _resolve_fibo_model_config(parser, args)

    # 0. Set default guidance value if not provided by user
    if args.guidance is None:
        if "fibo-lite" in model_config.aliases:
            args.guidance = 1.0  # distilled, no CFG
        elif "fibo" in model_config.aliases:
            args.guidance = 5.0  # base FIBO typical
        else:
            args.guidance = ui_defaults.GUIDANCE_SCALE

    json_prompt = FiboUtil.get_json_prompt(args, quantize=args.quantize)

    CallbackManager.apply_runtime_memory_options(args)

    # 1. Load the FIBO model
    fibo = FIBO(
        quantize=args.quantize,
        model_path=args.model_path,
        model_config=model_config,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=fibo,
        latent_creator=FiboLatentCreator,
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
            unsubscribe = events.subscribe_model(fibo, map_complete_to_generated=True)
            try:
                image = fibo.generate_image(
                    seed=seed,
                    prompt=json_prompt,
                    width=args.width,
                    height=args.height,
                    guidance=args.guidance,
                    image_path=args.image_path,
                    num_inference_steps=args.steps,
                    image_strength=args.image_strength,
                    scheduler="flow_match_euler_discrete",
                    negative_prompt=PromptUtil.read_negative_prompt(args),
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


if __name__ == "__main__":
    main()
