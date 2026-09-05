from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.common.download_policy import allow_downloads
from mflux.models.ernie_image.variants import ErnieImageTurbo
from mflux.models.fibo.variants.edit.fibo_edit import FIBOEdit
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.seedvr2 import SeedVR2
from mflux.models.wan import Wan2_2_TI2V
from mflux.models.z_image import ZImage, ZImageTurbo
from mflux.utils.exceptions import ModelConfigError


def _model_class_for_config(model_config: ModelConfig):
    family_tokens = {
        token.lower()
        for token in (
            *model_config.aliases,
            model_config.model_name,
            model_config.base_model,
        )
        if token is not None
    }

    def has(value: str) -> bool:
        return any(value in token for token in family_tokens)

    if has("bonsai"):
        return None
    if has("qwen-image-edit") or has("qwen-edit"):
        return QwenImageEdit
    if has("qwen"):
        return QwenImage
    if has("fibo-edit") or has("fiboedit"):
        return FIBOEdit
    if has("fibo"):
        return FIBO
    if has("z-image-turbo") or has("zimage-turbo"):
        return ZImageTurbo
    if has("z-image") or has("zimage"):
        return ZImage
    if has("ernie"):
        return ErnieImageTurbo
    if has("seedvr2"):
        return SeedVR2
    if has("minimax-h3"):
        from mflux.models.minimax_h3.variants.minimax_h3 import MiniMaxH3

        return MiniMaxH3
    if has("wan"):
        return Wan2_2_TI2V
    if has("flux2") or has("flux.2"):
        return Flux2Klein
    if has("flux.1") or has("flux"):
        return Flux1
    raise ValueError(
        "Cannot infer prepare backend for "
        f"{model_config.model_name!r}. Add --base-model with a supported family such as "
        "dev, schnell, krea-dev, qwen-image, qwen-image-edit, fibo, fibo-edit, "
        "z-image, z-image-turbo, ernie-image-turbo, seedvr2, wan2.2-ti2v-5b, or flux2-klein-4b."
    )


def main():
    with allow_downloads():
        # 0. Parse command line arguments
        parser = CommandLineParser(
            description=(
                "Prepare a reusable local MLX-Gen model folder, optionally quantized, "
                "and write a Hugging Face model card."
            )
        )
        parser.add_model_arguments(path_type="save", require_model_arg=True)
        parser.add_lora_arguments()
        args = parser.parse_args()

        # 1. Resolve config once and select the prepare backend from its family
        try:
            model_config = ModelConfig.from_name(args.model, base_model=args.base_model)
            if model_config.transformer_overrides.get("supports_bernini_renderer"):
                parser.error(
                    "Bernini-R currently supports only its pinned factored BF16 source route. "
                    "Use `mlxgen download --model bernini-r-1.3b`; `mlxgen prepare` and "
                    "q4/q8 Bernini packages are not supported."
                )
            if "swiftvr" in {alias.lower() for alias in model_config.aliases} or (
                "swiftvr" in model_config.model_name.lower()
            ):
                parser.error(
                    "SwiftVR currently supports only its bf16 source route. Use "
                    "`mlxgen download --model H-oliday/SwiftVR`; `mlxgen prepare` and quantized "
                    "SwiftVR packages are not supported yet."
                )
            model_class = _model_class_for_config(model_config)
        except ModelConfigError as exc:
            parser.error(
                f"{exc}. For local or custom model sources, add --base-model with the closest supported family."
            )
        except ValueError as exc:
            parser.error(str(exc))
        if model_class is None:
            parser.error(
                "Bonsai checkpoints are already MLX-packed low-bit artifacts. "
                "Use `mlxgen download --model ...` to cache Bonsai locally; `mlxgen prepare` is not needed."
            )

        # 2. Load, quantize and save the model
        model_kwargs = {
            "quantize": args.quantize,
            "model_config": model_config,
        }
        if args.lora_paths or args.lora_scales:
            parser.error(
                "mlxgen prepare --lora-paths/--lora-scales is gated until MLX-Gen proves deterministic "
                "save/reload LoRA bake behavior for the selected family and quantization mode. "
                "Use LoRA at runtime through mlxgen generate for now."
            )
        model = model_class(**model_kwargs)
        model.save_model(args.path)


if __name__ == "__main__":
    main()
