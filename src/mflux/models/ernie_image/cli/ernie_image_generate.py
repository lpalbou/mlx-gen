from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.ernie_image.latent_creator import ErnieImageLatentCreator
from mflux.models.ernie_image.variants import ErnieImageTurbo
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    parser = CommandLineParser(description="Generate an image using ERNIE-Image-Turbo.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
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
    args = parser.parse_args()

    if args.guidance is None:
        args.guidance = 1.0

    model = ErnieImageTurbo(
        model_config=ModelConfig.ernie_image_turbo(),
        quantize=args.quantize,
        model_path=args.model_path,
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=ErnieImageLatentCreator,
    )

    try:
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )
        if min(width, height) < 384:
            print(
                "Warning: ERNIE-Image-Turbo is validated for practical generation at 384px and above. "
                "Very small outputs can crop or truncate subjects."
            )
        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=width,
                height=height,
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
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata, overwrite=args.replace)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
