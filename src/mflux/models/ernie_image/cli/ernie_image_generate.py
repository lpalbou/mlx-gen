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
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_output_arguments()
    args = parser.parse_args()

    if args.quantize is not None:
        parser.error("ERNIE quantized generation is not enabled yet. Use the BF16 source or prepared BF16 weights.")

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
            reference_image_path=None,
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
                num_inference_steps=args.steps,
                negative_prompt=args.negative_prompt,
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
