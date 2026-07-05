from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.output_paths import resolve_output_path
from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.ernie_image.latent_creator import ErnieImageLatentCreator
from mflux.models.ernie_image.variants import ErnieImageTurbo
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def _parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using ERNIE-Image-Turbo.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_lora_arguments()
    parser.add_argument(
        "--use-prompt-enhancer",
        "--use-pe",
        action="store_true",
        help="Use ERNIE's Prompt Enhancer. Requires a full ERNIE Image Turbo snapshot with pe/ files.",
    )
    parser.add_argument(
        "--prompt-enhancer-system-prompt",
        default=None,
        help="Optional system prompt passed to ERNIE's Prompt Enhancer.",
    )
    parser.add_argument(
        "--prompt-enhancer-temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for ERNIE's Prompt Enhancer. Default is 0.6.",
    )
    parser.add_argument(
        "--prompt-enhancer-top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p for ERNIE's Prompt Enhancer. Default is 0.95.",
    )
    parser.add_argument(
        "--prompt-enhancer-max-new-tokens",
        type=int,
        default=None,
        help="Maximum Prompt Enhancer tokens. Default is the PE tokenizer model_max_length.",
    )
    parser.add_output_arguments()
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()

    if args.guidance is None:
        args.guidance = 1.0

    model_config = ModelConfig.from_name(args.model or "ernie-image-turbo", base_model=args.base_model)

    CallbackManager.apply_runtime_memory_options(args)

    model = ErnieImageTurbo(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=ErnieImageLatentCreator,
    )

    try:
        if all(isinstance(value, int) for value in (args.width, args.height)) and min(args.width, args.height) < 384:
            cli_print(
                "Warning: ERNIE-Image-Turbo is validated for practical generation at 384px and above. "
                "Very small outputs can crop or truncate subjects.",
                json_events=bool(args.json_events),
            )
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
                    width=args.width,
                    height=args.height,
                    guidance=args.guidance,
                    image_path=args.image_path,
                    image_strength=args.image_strength,
                    num_inference_steps=args.steps,
                    negative_prompt=args.negative_prompt,
                    use_pe=args.use_prompt_enhancer,
                    pe_system_prompt=args.prompt_enhancer_system_prompt,
                    pe_temperature=args.prompt_enhancer_temperature,
                    pe_top_p=args.prompt_enhancer_top_p,
                    pe_max_new_tokens=args.prompt_enhancer_max_new_tokens,
                    canvas_policy=args.canvas_policy,
                    lora_paths=getattr(model, "lora_paths", None),
                    lora_scales=getattr(model, "lora_scales", None),
                    extra_metadata=LoRALoader.extra_metadata_for_model(model),
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
