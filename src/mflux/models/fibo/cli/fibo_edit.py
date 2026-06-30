from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.output_paths import normalize_output_template, resolve_output_path
from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator
from mflux.models.fibo.variants.edit.fibo_edit import FIBOEdit
from mflux.models.fibo.variants.edit.util import FIBO_EDIT_RMBG_DEFAULT_JSON_PROMPT, FiboEditUtil
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

FIBO_EDIT_GUIDANCE_DEFAULT = 5.0
FIBO_EDIT_RMBG_GUIDANCE_DEFAULT = 1.0


def _resolve_fibo_edit_model_config(parser: CommandLineParser, args) -> ModelConfig:
    try:
        model_config = ModelConfig.from_name(model_name=args.model, base_model=args.base_model)
    except ModelConfigError as exc:
        parser.error(str(exc))

    compatible_edit_aliases = {"fibo-edit", "fibo-edit-rmbg"}
    if not compatible_edit_aliases.intersection(model_config.aliases):
        parser.error(
            "mflux-generate-fibo-edit requires a FIBO Edit model; metadata or --model resolved to an incompatible model."
        )

    return model_config


def _is_rmbg(model_config: ModelConfig) -> bool:
    return "fibo-edit-rmbg" in model_config.aliases


def _validate_matte_output(parser: CommandLineParser, args, model_config: ModelConfig) -> None:
    if args.matte_output is not None and not _is_rmbg(model_config):
        parser.error("--matte-output is only supported with --model fibo-edit-rmbg.")


def _apply_default_guidance(args, model_config: ModelConfig) -> None:
    if args.guidance is None:
        args.guidance = FIBO_EDIT_RMBG_GUIDANCE_DEFAULT if _is_rmbg(model_config) else FIBO_EDIT_GUIDANCE_DEFAULT


def _json_prompt_for_edit(args, model_config: ModelConfig) -> str:
    default_if_missing = FIBO_EDIT_RMBG_DEFAULT_JSON_PROMPT if _is_rmbg(model_config) else None
    return FiboEditUtil.get_json_prompt_for_edit(
        args,
        quantize=args.quantize,
        default_json_prompt_if_missing=default_if_missing,
    )


def _save_edit_result(
    image: GeneratedImage,
    args,
    model_config: ModelConfig,
    seed: int,
    output_path: str | Path,
) -> None:
    if _is_rmbg(model_config):
        rgba_pil = FiboEditUtil.build_rgba_composite_image(args.image_path, image.image)
        image.save_with_image(
            path=output_path,
            pixel_image=rgba_pil,
            export_json_metadata=args.metadata,
            overwrite=True,
            embed_metadata=args.embed_metadata,
        )
        if args.matte_output is not None:
            matte_output_path = resolve_output_path(
                args.matte_output,
                overwrite=args.replace,
                seed=seed,
            )
            if Path(matte_output_path) == Path(output_path):
                raise ValueError(
                    "--matte-output resolved to the same path as --output. Choose a different matte path "
                    "or pass --replace false so MLX-Gen can suffix one of the files safely."
                )
            image.save(
                path=matte_output_path,
                export_json_metadata=False,
                overwrite=True,
                embed_metadata=args.embed_metadata,
            )
    else:
        image.save(
            path=output_path,
            export_json_metadata=args.metadata,
            overwrite=True,
            embed_metadata=args.embed_metadata,
        )


def main():
    parser = CommandLineParser(description="Generate an edited image using Bria FIBO Edit.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.set_defaults(model="fibo-edit")
    parser.add_image_generator_arguments(
        supports_metadata_config=True,
        require_prompt=False,
        supports_dimension_scale_factor=True,
    )
    parser.add_argument("--image-path", type=Path, required=False, help="Local path to source image for editing.")
    parser.add_argument(
        "--mask-path",
        "--masked-image-path",
        dest="mask_path",
        type=Path,
        default=None,
        help="Optional mask image path for localized edits.",
    )
    parser.add_argument("--matte-output", type=str, default=None, help="fibo-edit-rmbg only: also save the raw grayscale matte. Supports {seed} like --output.")  # fmt: skip
    parser.add_output_arguments()
    args = parser.parse_args()
    if args.matte_output is not None and len(args.seed) > 1:
        args.matte_output = normalize_output_template(args.matte_output, include_seed=True)

    if args.image_path is None:
        parser.error("--image-path is required, or 'image_path' must be specified in the config file.")

    model_config = _resolve_fibo_edit_model_config(parser, args)
    _validate_matte_output(parser, args, model_config)
    _apply_default_guidance(args, model_config)
    try:
        json_prompt = _json_prompt_for_edit(args, model_config)
    except (PromptFileReadError, ValueError) as exc:
        parser.error(str(exc))

    CallbackManager.apply_runtime_memory_options(args)

    fibo_edit = FIBOEdit(
        quantize=args.quantize,
        model_path=args.model_path,
        model_config=model_config,
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=fibo_edit,
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
            unsubscribe = events.subscribe_model(fibo_edit, map_complete_to_generated=True)
            try:
                image = fibo_edit.generate_image(
                    seed=seed,
                    prompt=json_prompt,
                    image_path=args.image_path,
                    mask_path=args.mask_path,
                    width=args.width,
                    height=args.height,
                    guidance=args.guidance,
                    num_inference_steps=args.steps,
                    scheduler="flow_match_euler_discrete",
                    negative_prompt=PromptUtil.read_negative_prompt(args),
                    canvas_policy=args.canvas_policy,
                )
                events.emit_save()
                _save_edit_result(image, args, model_config, seed, output_path)
                events.emit_complete()
            except Exception as exc:
                events.emit_failed(error=exc)
                raise
            finally:
                if unsubscribe is not None:
                    unsubscribe()
    except (StopImageGenerationException, PromptFileReadError, ValueError) as exc:
        cli_print(str(exc), json_events=bool(args.json_events), error=True)
        raise SystemExit(1) from exc
    finally:
        if memory_saver:
            cli_print(memory_saver.memory_stats(), json_events=bool(args.json_events))


if __name__ == "__main__":
    main()
