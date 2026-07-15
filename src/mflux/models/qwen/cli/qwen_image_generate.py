import sys

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.output_paths import resolve_output_path
from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.variants.controlnet.qwen_image_controlnet import QwenImageControlNet
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.task_inference import TaskInferenceError, resolve_generation_plan
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Qwen Image model.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_mask_path_argument(
        help_text=(
            "Optional mask image path for base-Qwen masked edit. White pixels are repainted and black pixels are "
            "preserved. The exact validated control-inpaint row uses its ControlNet sidecar; other base rows use "
            "native masked edit."
        ),
    )
    parser.add_controlnet_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # 0. Set model-specific defaults if not provided by user
    if "--scheduler" not in sys.argv:
        args.scheduler = "flow_match_euler_discrete"
    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE
    if args.controlnet_model is not None and args.controlnet_image_path is None and args.mask_path is None:
        parser.error("--controlnet-model requires --controlnet-image-path or --mask-path.")

    # 1. Load the model
    model_config = ModelConfig.from_name(model_name=args.model or "qwen-image", base_model=args.base_model)
    CallbackManager.apply_runtime_memory_options(args)
    uses_native_inpaint = False
    if args.mask_path is not None:
        if args.image_path is None:
            parser.error("--mask-path requires --image-path.")
        if args.image_strength is not None:
            parser.error(
                "--image-strength cannot be combined with --mask-path; masked edit is a separate route "
                "from latent image-to-image."
            )
        if args.controlnet_image_path is not None:
            parser.error("--mask-path cannot be combined with --controlnet-image-path on base-Qwen masked routes.")
        try:
            plan = resolve_generation_plan(
                model=args.model,
                model_config=model_config,
                base_model=args.base_model,
                image_count=1,
                has_mask=True,
            )
        except TaskInferenceError as exc:
            parser.error(str(exc))
        uses_native_inpaint = plan.control_model is None
        if uses_native_inpaint:
            if args.controlnet_model is not None:
                parser.error(
                    "--controlnet-model is not supported on the native base-Qwen masked edit route. "
                    "The validated control-inpaint sidecar row is AbstractFramework/qwen-image-8bit."
                )
            if _option_was_provided(sys.argv[1:], "--controlnet-strength"):
                parser.error("--controlnet-strength is not supported on the native base-Qwen masked edit route.")
            qwen = QwenImage(
                model_config=model_config,
                quantize=args.quantize,
                model_path=args.model_path,
                lora_paths=args.lora_paths,
                lora_scales=args.lora_scales,
            )
        else:
            if args.controlnet_model is not None and args.controlnet_model != plan.control_model:
                parser.error(
                    "--controlnet-model conflicts with the exact base-Qwen control-inpaint row. "
                    "Use the documented route, or call a different backend explicitly if you need another ControlNet package."
                )
            qwen = QwenImageControlNet(
                controlnet_model=args.controlnet_model or plan.control_model,
                model_config=model_config,
                quantize=args.quantize,
                model_path=args.model_path,
                lora_paths=args.lora_paths,
                lora_scales=args.lora_scales,
            )
    elif args.controlnet_image_path is not None:
        if args.image_path is not None:
            parser.error("--controlnet-image-path cannot be combined with --image-path or latent image-to-image mode.")
        try:
            plan = resolve_generation_plan(
                model=args.model,
                model_config=model_config,
                base_model=args.base_model,
                image_count=0,
                has_control_image=True,
            )
        except TaskInferenceError as exc:
            parser.error(str(exc))
        if args.controlnet_model is not None and args.controlnet_model != plan.control_model:
            parser.error(
                "--controlnet-model conflicts with the exact structured-control row selected by this backend. "
                "Use the documented route, or call a different backend explicitly if you need another ControlNet package."
            )
        qwen = QwenImageControlNet(
            controlnet_model=args.controlnet_model or plan.control_model,
            model_config=model_config,
            quantize=args.quantize,
            model_path=args.model_path,
            lora_paths=args.lora_paths,
            lora_scales=args.lora_scales,
        )
    else:
        qwen = QwenImage(
            model_config=model_config,
            quantize=args.quantize,
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
        for seed in args.seed:
            events = CliRuntimeEventStream(
                enabled=bool(args.json_events),
                command="mlxgen generate",
                model=model_config.model_name,
                seed=seed,
            )
            output_path = resolve_output_path(args.output, overwrite=args.replace, seed=seed)
            events.set_output_path(output_path)
            unsubscribe = events.subscribe_model(qwen, map_complete_to_generated=True)
            try:
                if uses_native_inpaint:
                    image = qwen.generate_image(
                        seed=seed,
                        prompt=PromptUtil.read_prompt(args),
                        negative_prompt=_read_negative_prompt(args),
                        width=args.width,
                        height=args.height,
                        guidance=args.guidance,
                        scheduler=args.scheduler,
                        num_inference_steps=args.steps,
                        image_path=args.image_path,
                        mask_path=args.mask_path,
                        canvas_policy=args.canvas_policy,
                    )
                elif args.controlnet_image_path is not None or args.mask_path is not None:
                    image = qwen.generate_image(
                        seed=seed,
                        prompt=PromptUtil.read_prompt(args),
                        negative_prompt=_read_negative_prompt(args),
                        width=args.width,
                        height=args.height,
                        guidance=args.guidance,
                        scheduler=args.scheduler,
                        controlnet_image_path=args.controlnet_image_path,
                        controlnet_strength=args.controlnet_strength,
                        num_inference_steps=args.steps,
                        image_path=args.image_path,
                        mask_path=args.mask_path,
                        canvas_policy=args.canvas_policy,
                    )
                else:
                    image = qwen.generate_image(
                        seed=seed,
                        prompt=PromptUtil.read_prompt(args),
                        negative_prompt=_read_negative_prompt(args),
                        width=args.width,
                        height=args.height,
                        guidance=args.guidance,
                        scheduler=args.scheduler,
                        image_path=args.image_path,
                        num_inference_steps=args.steps,
                        image_strength=args.image_strength,
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
