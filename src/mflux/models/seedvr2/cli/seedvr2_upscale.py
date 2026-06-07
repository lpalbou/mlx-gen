from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.download_policy import DownloadRequiredError, is_huggingface_repo_id
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.seedvr2.latent_creator.seedvr2_latent_creator import SeedVR2LatentCreator
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.utils.exceptions import StopImageGenerationException

SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _resolve_seedvr2_model(model_arg: str | None, model_path: str | None) -> tuple[ModelConfig, str | None]:
    if model_arg is None:
        return ModelConfig.seedvr2_3b(), model_path

    normalized = model_arg.lower()
    if normalized in {"seedvr2", "seedvr2-3b"}:
        return ModelConfig.seedvr2_3b(), None
    if normalized == "seedvr2-7b":
        return ModelConfig.seedvr2_7b(), None
    if normalized in {"bytedance-seed/seedvr2-3b", "bytedance-seed/seedvr2_3b"}:
        return ModelConfig.seedvr2_3b(), model_arg
    if normalized in {"bytedance-seed/seedvr2-7b", "bytedance-seed/seedvr2_7b"}:
        return ModelConfig.seedvr2_7b(), model_arg
    if normalized == "numz/seedvr2_comfyui":
        return ModelConfig.seedvr2_3b(), model_arg
    if normalized.startswith("abstractframework/seedvr2-3b-"):
        return ModelConfig.seedvr2_3b(), model_arg
    if normalized.startswith("abstractframework/seedvr2-7b-"):
        return ModelConfig.seedvr2_7b(), model_arg

    requested_model_path = model_path
    if requested_model_path is None and Path(model_arg).expanduser().exists():
        requested_model_path = model_arg

    if requested_model_path is not None:
        path = Path(requested_model_path).expanduser()
        if path.is_dir():
            has_3b = (path / "seedvr2_ema_3b_fp16.safetensors").exists()
            has_official_3b = (path / "seedvr2_ema_3b.pth").exists()
            has_7b = (path / "seedvr2_ema_7b_fp16.safetensors").exists()
            has_official_7b = (path / "seedvr2_ema_7b.pth").exists()
            if (has_7b or has_official_7b) and not (has_3b or has_official_3b):
                return ModelConfig.seedvr2_7b(), requested_model_path
            if (has_3b or has_official_3b) and not (has_7b or has_official_7b):
                return ModelConfig.seedvr2_3b(), requested_model_path
            if (path / "transformer" / "model.safetensors.index.json").exists():
                if "seedvr2-7b" in normalized or "7b" in path.name.lower():
                    return ModelConfig.seedvr2_7b(), requested_model_path
                return ModelConfig.seedvr2_3b(), requested_model_path

    if is_huggingface_repo_id(model_arg):
        raise ValueError(
            "Unsupported SeedVR2 model handle "
            f"{model_arg!r}. Use seedvr2, seedvr2-3b, seedvr2-7b, "
            "ByteDance-Seed/SeedVR2-3B, ByteDance-Seed/SeedVR2-7B, "
            "AbstractFramework/seedvr2-3b-8bit, AbstractFramework/seedvr2-3b-4bit, "
            "AbstractFramework/seedvr2-7b-8bit, AbstractFramework/seedvr2-7b-4bit, "
            "or an explicit local SeedVR2 path."
        )

    source = (requested_model_path or model_arg).lower()
    if "seedvr2_ema_7b" in source or "seedvr2-7b" in source:
        return ModelConfig.seedvr2_7b(), requested_model_path
    return ModelConfig.seedvr2_3b(), requested_model_path


def _expand_image_paths(image_paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for image_path in image_paths:
        if image_path.is_dir():
            dir_images = sorted(
                [path for path in image_path.iterdir() if _is_image_file(path)],
                key=lambda path: path.name.lower(),
            )
            if not dir_images:
                print(f"No images found in directory: {image_path}")
            expanded.extend(dir_images)
        else:
            expanded.append(image_path)
    return expanded


def main():
    # 1. Parse command line arguments
    parser = CommandLineParser(description="Upscale an image using SeedVR2 diffusion-based super-resolution.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_seedvr2_upscale_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    image_paths = _expand_image_paths(args.image_path)
    if not image_paths:
        print("No images to upscale.")
        return

    try:
        model_config, resolved_model_path = _resolve_seedvr2_model(args.model, args.model_path)
    except ValueError as exc:
        parser.error(str(exc))

    # 3. Load the SeedVR2 model
    try:
        model = SeedVR2(
            quantize=args.quantize,
            model_path=resolved_model_path,
            model_config=model_config,
        )
    except DownloadRequiredError as exc:
        parser.error(str(exc))
    if args.vae_tiling:
        model.tiling_config = TilingConfig()

    # 4. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=SeedVR2LatentCreator,
    )

    try:
        # 5. Upscale the image for each seed
        for image_path in image_paths:
            for seed in args.seed:
                result = model.generate_image(
                    seed=seed,
                    image_path=image_path,
                    resolution=args.resolution,
                    softness=args.softness,
                )

                # 6. Save result
                result.save(
                    args.output.format(seed=seed, image_name=image_path.stem),
                    export_json_metadata=args.metadata,
                    overwrite=args.replace,
                )
    except StopImageGenerationException as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
