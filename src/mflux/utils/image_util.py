import json
import logging
from pathlib import Path

import mlx.core as mx
import numpy as np
import PIL.Image
from PIL._typing import StrOrBytesPath

from mflux.cli.output_paths import resolve_collision_free_path
from mflux.models.common.config.config import Config
from mflux.models.flux.variants.concept_attention.attention_data import ConceptHeatmap
from mflux.utils.box_values import AbsoluteBoxValues, BoxValues
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.metadata_builder import MetadataBuilder
from mflux.utils.tensor_health import TensorHealth

# No module-scope PIL.Image.init(): PIL registers format plugins itself on the
# first open()/save(), and eagerly initializing all plugins costs ~0.2 s on
# every import of this module (0088). PIL.ImageDraw/ImageOps/piexif are
# imported inside the few methods that use them for the same reason.

log = logging.getLogger(__name__)


class ImageUtil:
    @staticmethod
    def resolve_output_path(path: str | Path, overwrite: bool = True) -> Path:
        # Public surface kept: the pure path logic lives in cli/output_paths so
        # path-only callers do not import the image stack (0088).
        return resolve_collision_free_path(path=path, overwrite=overwrite)

    @staticmethod
    def to_image(
        decoded_latents: mx.array,
        config: Config,
        seed: int,
        prompt: str,
        quantization: int,
        generation_time: float,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        controlnet_image_path: str | Path | None = None,
        image_path: str | Path | None = None,
        image_paths: list[str] | list[Path] | None = None,
        redux_image_paths: list[str] | list[Path] | None = None,
        redux_image_strengths: list[float] | None = None,
        image_strength: float | None = None,
        masked_image_path: str | Path | None = None,
        depth_image_path: str | Path | None = None,
        concept_heatmap: ConceptHeatmap | None = None,
        negative_prompt: str | None = None,
        init_metadata: dict | None = None,
        extra_metadata: dict | None = None,
    ) -> GeneratedImage:
        TensorHealth.ensure_finite(decoded_latents, name="decoded_image", phase="image-conversion")
        normalized = ImageUtil._denormalize(decoded_latents)
        normalized_numpy = ImageUtil._to_numpy(normalized)
        image = ImageUtil._numpy_to_pil(normalized_numpy)
        return GeneratedImage(
            image=image,
            model_config=config.model_config,
            seed=seed,
            steps=config.num_inference_steps,
            prompt=prompt,
            guidance=config.guidance,
            precision=config.precision,
            quantization=quantization,
            generation_time=generation_time,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            height=config.height,
            width=config.width,
            image_path=image_path,
            image_paths=image_paths,
            image_strength=image_strength,
            controlnet_image_path=controlnet_image_path,
            controlnet_strength=config.controlnet_strength,
            masked_image_path=masked_image_path,
            depth_image_path=depth_image_path,
            redux_image_paths=redux_image_paths,
            redux_image_strengths=redux_image_strengths,
            concept_heatmap=concept_heatmap,
            negative_prompt=negative_prompt,
            init_metadata=init_metadata,
            canvas_policy=config.canvas_policy,
            requested_width=config.requested_width,
            requested_height=config.requested_height,
            source_image_width=config.source_image_width,
            source_image_height=config.source_image_height,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def to_composite_image(generated_images: list[GeneratedImage]) -> PIL.Image.Image:
        return ImageUtil.to_composite_pil_images([gen_img.image for gen_img in generated_images])

    @staticmethod
    def to_composite_pil_images(images: list[PIL.Image.Image]) -> PIL.Image.Image:
        total_width = sum(image.width for image in images)
        max_height = max(image.height for image in images)
        composite_img = PIL.Image.new("RGB", (total_width, max_height))
        current_x = 0
        for image in images:
            composite_img.paste(image, (current_x, 0))
            current_x += image.width
        return composite_img

    @staticmethod
    def _denormalize(images: mx.array) -> mx.array:
        return mx.clip((images / 2 + 0.5), 0, 1)

    @staticmethod
    def _normalize(images: mx.array) -> mx.array:
        return 2.0 * images - 1.0

    @staticmethod
    def _binarize(image: mx.array) -> mx.array:
        return mx.where(image < 0.5, mx.zeros_like(image), mx.ones_like(image))

    @staticmethod
    def _to_numpy(images: mx.array) -> np.ndarray:
        if len(images.shape) == 5:
            images = mx.squeeze(images, axis=2)
        images = mx.transpose(images, (0, 2, 3, 1))
        images = mx.array.astype(images, mx.float32)
        images = np.array(images)
        return images

    @staticmethod
    def _numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
        TensorHealth.ensure_finite(images, name="normalized_image", phase="image-conversion")
        images = (images * 255).round().astype("uint8")
        pil_images = [PIL.Image.fromarray(image) for image in images]
        return pil_images[0]

    @staticmethod
    def _pil_to_numpy(image: PIL.Image.Image) -> np.ndarray:
        image = np.array(image).astype(np.float32) / 255.0
        images = np.stack([image], axis=0)
        return images

    @staticmethod
    def to_array(image: PIL.Image.Image, is_mask: bool = False) -> mx.array:
        image = ImageUtil._pil_to_numpy(image)
        array = mx.array(image)
        array = mx.transpose(array, (0, 3, 1, 2))
        if is_mask:
            array = ImageUtil._binarize(array)
        else:
            array = ImageUtil._normalize(array)
        return array

    @staticmethod
    def load_image(image_or_path: PIL.Image.Image | StrOrBytesPath) -> PIL.Image.Image:
        if isinstance(image_or_path, PIL.Image.Image):
            return image_or_path.convert("RGB")
        else:
            return PIL.Image.open(image_or_path).convert("RGB")

    @staticmethod
    def expand_image(
        image: PIL.Image.Image,
        box_values: AbsoluteBoxValues | None = None,
        top: int | str = 0,
        right: int | str = 0,
        bottom: int | str = 0,
        left: int | str = 0,
        fill_color: tuple = (255, 255, 255),
    ) -> PIL.Image.Image:
        if box_values is None:
            box_values = BoxValues(top=top, right=right, bottom=bottom, left=left).normalize_to_dimensions(
                width=image.width,
                height=image.height,
            )

        new_width = image.width + box_values.left + box_values.right
        new_height = image.height + box_values.top + box_values.bottom
        expanded_image = PIL.Image.new(image.mode, (new_width, new_height), fill_color)
        expanded_image.paste(image, (box_values.left, box_values.top))
        return expanded_image

    @staticmethod
    def create_outpaint_mask_image(orig_width: int, orig_height: int, **create_bordered_image_kwargs):
        return ImageUtil.create_bordered_image(
            orig_width,
            orig_height,
            border_color=(255, 255, 255),
            content_color=(0, 0, 0),
            **create_bordered_image_kwargs,
        )

    @staticmethod
    def create_bordered_image(
        orig_width: int,
        orig_height: int,
        border_color: tuple,
        content_color: tuple,
        box_values: AbsoluteBoxValues | None = None,
        top: int | str = 0,
        right: int | str = 0,
        bottom: int | str = 0,
        left: int | str = 0,
    ) -> PIL.Image.Image:
        if box_values is None:
            box_values = BoxValues(top=top, right=right, bottom=bottom, left=left).normalize_to_dimensions(
                orig_width, orig_height
            )

        # Create a new white image
        new_width = orig_width + box_values.right + box_values.left
        new_height = orig_height + box_values.top + box_values.bottom

        # `from PIL import ...` avoids rebinding the module-scope `PIL` name locally.
        from PIL import ImageDraw

        result = PIL.Image.new("RGB", (new_width, new_height), border_color)
        draw = ImageDraw.Draw(result)

        # Draw black rectangle in the center
        draw.rectangle(
            [(box_values.left, box_values.top), (box_values.left + orig_width, box_values.top + orig_height)],
            fill=content_color,
        )

        return result

    @staticmethod
    def scale_to_dimensions(
        image: PIL.Image.Image,
        target_width: int,
        target_height: int,
        resize_mode: str = "resize",
    ) -> PIL.Image.Image:
        if (image.width, image.height) == (target_width, target_height):
            return image
        if resize_mode == "resize":
            return image.resize((target_width, target_height), PIL.Image.LANCZOS)
        if resize_mode == "crop":
            # `from PIL import ...` avoids rebinding the module-scope `PIL` name locally.
            from PIL import ImageOps

            return ImageOps.fit(
                image,
                (target_width, target_height),
                method=PIL.Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
        raise ValueError("resize_mode must be 'resize' or 'crop'.")

    @staticmethod
    def save_image(
        image: PIL.Image.Image,
        path: str | Path,
        metadata: dict | None = None,
        export_json_metadata: bool = False,
        overwrite: bool = True,
        embed_metadata: bool = False,
    ) -> Path:
        file_path = ImageUtil.resolve_output_path(path=path, overwrite=overwrite)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        exif_bytes = None
        pnginfo = None
        if metadata is not None and embed_metadata:
            exif_bytes = ImageUtil._build_exif_bytes(metadata)
            if file_path.suffix.lower() == ".png":
                pnginfo = MetadataBuilder.build_pnginfo(metadata)

        saved_with_external_encoder = ImageUtil._save_image_with_ffmpeg(image=image, path=file_path)
        if not saved_with_external_encoder:
            image_format = ImageUtil._image_format_for_path(file_path)
            save_kwargs = {}
            if image_format is not None:
                save_kwargs["format"] = image_format
            if exif_bytes is not None:
                save_kwargs["exif"] = exif_bytes
            if pnginfo is not None:
                save_kwargs["pnginfo"] = pnginfo
            if image_format is None:
                image.save(file_path, **save_kwargs)
            else:
                image.save(file_path, **save_kwargs)
        log.info(f"Image saved successfully at: {file_path}")

        # Export metadata to a dedicated sidecar path so it never
        # collides with model-specific JSON artifacts like FIBO prompts.
        if export_json_metadata:
            metadata_path = file_path.with_suffix(".metadata.json")
            with open(metadata_path, "w") as json_file:
                json.dump(metadata, json_file, indent=4)
        return file_path

    @staticmethod
    def _save_image_with_ffmpeg(*, image: PIL.Image.Image, path: Path) -> bool:
        del image, path
        return False

    @staticmethod
    def _image_format_for_path(path: Path) -> str | None:
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            return "JPEG"
        if suffix == ".png":
            return "PNG"
        if suffix == ".webp":
            return "WEBP"
        if suffix in {".tif", ".tiff"}:
            return "TIFF"
        if suffix == ".bmp":
            return "BMP"
        return None

    @staticmethod
    def _build_exif_bytes(metadata: dict) -> bytes:
        import piexif

        try:
            # Convert metadata dictionary to a string
            metadata_str = json.dumps(metadata)

            # Convert the string to bytes (using UTF-8 encoding)
            # Add the ASCII character code prefix required by EXIF spec
            user_comment_bytes = b"ASCII\x00\x00\x00" + metadata_str.encode("utf-8")

            # Define the UserComment tag ID
            USER_COMMENT_TAG_ID = 0x9286

            # Create a piexif-compatible dictionary structure
            exif_piexif_dict = {"Exif": {USER_COMMENT_TAG_ID: user_comment_bytes}}

            exif_bytes = piexif.dump(exif_piexif_dict)
            return exif_bytes

        except Exception as e:  # noqa: BLE001
            log.error(f"Error embedding EXIF metadata: {e}")
            raise

    @staticmethod
    def preprocess_for_model(
        image: PIL.Image.Image,
        target_size: tuple = (384, 384),
        mean: list = [0.5, 0.5, 0.5],
        std: list = [0.5, 0.5, 0.5],
        resample: int = PIL.Image.LANCZOS,
    ) -> mx.array:
        # Resize the image to target size
        image = image.resize(target_size, resample=resample)

        # Convert PIL image to numpy array and normalize to [0, 1]
        image_np = np.array(image).astype(np.float32) / 255.0

        # Normalize using specified mean and std
        mean_np = np.array(mean)
        std_np = np.array(std)
        image_np = (image_np - mean_np) / std_np

        # Convert from HWC to CHW format
        image_np = image_np.transpose(2, 0, 1)

        # Convert to MLX array and add batch dimension
        image_mx = mx.array(image_np)
        image_mx = mx.expand_dims(image_mx, axis=0)

        return image_mx

    @staticmethod
    def preprocess_for_depth_pro(
        image: PIL.Image.Image,
        target_size: tuple = (384, 384),
        mean: list = [0.5, 0.5, 0.5],
        std: list = [0.5, 0.5, 0.5],
        resample: int = PIL.Image.LANCZOS,
    ) -> mx.array:
        # Convert PIL image to numpy array and normalize to [0, 1]
        image_np = np.array(image).astype(np.float32) / 255.0

        # Convert from HWC to CHW format
        image_np = image_np.transpose(2, 0, 1)

        # Normalize using specified mean and std
        mean_np = np.array(mean).reshape(-1, 1, 1)
        std_np = np.array(std).reshape(-1, 1, 1)
        image_np = (image_np - mean_np) / std_np

        # Convert to MLX array
        return mx.array(image_np)
