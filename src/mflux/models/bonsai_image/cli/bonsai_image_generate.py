from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.output_paths import resolve_output_path
from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print
from mflux.models.bonsai_image import BonsaiImage
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def _parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Bonsai Image.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_output_arguments()
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()

    if getattr(args, "negative_prompt", ""):
        parser.error("--negative-prompt is not supported for Bonsai Image. Focus on describing what you want.")
    if args.quantize is not None:
        parser.error("Bonsai checkpoints are already packed low-bit MLX artifacts. Omit --quantize/-q.")
    if args.guidance is None:
        args.guidance = 1.0
    if args.guidance != 1.0:
        parser.error("Bonsai Image is distilled and supports only --guidance 1.0.")

    CallbackManager.apply_runtime_memory_options(args)

    model_name = args.model or "bonsai-image-ternary"
    try:
        model = BonsaiImage(
            model_config=ModelConfig.from_name(model_name=model_name, base_model=args.base_model),
            model_path=args.model_path,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Flux2LatentCreator,
    )

    try:
        width, height = DimensionResolver.resolve(width=args.width, height=args.height)
        for seed in args.seed:
            events = CliRuntimeEventStream(
                enabled=bool(args.json_events),
                command="mlxgen generate",
                model=model.model_config.model_name,
                seed=seed,
            )
            output_path = resolve_output_path(args.output, overwrite=args.replace, seed=seed)
            events.set_output_path(output_path)
            unsubscribe = events.subscribe_model(model, map_complete_to_generated=True)
            try:
                image = model.generate_image(
                    seed=seed,
                    prompt=PromptUtil.read_prompt(args),
                    width=width,
                    height=height,
                    guidance=args.guidance,
                    num_inference_steps=args.steps,
                    scheduler="flow_match_euler_discrete",
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
