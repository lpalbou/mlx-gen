from pathlib import Path

import numpy as np
from PIL import Image


class MaskUtil:
    # Policy: surfaces ported from an upstream reference keep that reference's mask resampling
    # (Qwen edit/control inpaint and Z-Image use NEAREST to match diffusers); in-house surfaces
    # (Wan masked video-to-video) default to BOX + 0.5 binarization.
    @staticmethod
    def load_binary_mask(
        mask_path: Path | str,
        *,
        target_width: int,
        target_height: int,
        resampling: Image.Resampling,
        alpha_warning_context: str | None = None,
        resize_mode: str = "resize",
    ) -> np.ndarray:
        with Image.open(mask_path) as image:
            if alpha_warning_context is not None and "A" in image.getbands():
                print(f"⚠️  {alpha_warning_context} has an alpha channel; alpha is ignored and luminance is used.")
            mask_image = MaskUtil.map_mask_to_canvas(
                image.convert("L"),
                target_width=target_width,
                target_height=target_height,
                resampling=resampling,
                resize_mode=resize_mode,
            )
        mask_values = np.asarray(mask_image, dtype=np.float32) / 255.0
        # White (>= 0.5) marks the editable region; black is preserved.
        return (mask_values >= 0.5).astype(np.float32)

    @staticmethod
    def map_mask_to_canvas(
        mask_image: Image.Image,
        *,
        target_width: int,
        target_height: int,
        resampling: Image.Resampling,
        resize_mode: str = "resize",
    ) -> Image.Image:
        # Masks are authored against the SOURCE content, so they must undergo the
        # exact source-to-canvas mapping the pixels get (same crop centering, same
        # letterbox math) or inpaint/vace alignment breaks. Padded borders are 0
        # (preserved): synthetic bars must never become a silent outpaint claim.
        if resize_mode == "resize":
            return mask_image.resize((target_width, target_height), resampling)
        if resize_mode == "crop":
            from PIL import ImageOps

            return ImageOps.fit(
                mask_image,
                (target_width, target_height),
                method=resampling,
                centering=(0.5, 0.5),
            )
        if resize_mode == "pad":
            from mflux.utils.image_util import ImageUtil

            scaled_width, scaled_height, left, top = ImageUtil.letterbox_geometry(
                source_width=mask_image.width,
                source_height=mask_image.height,
                target_width=target_width,
                target_height=target_height,
            )
            resized = mask_image.resize((scaled_width, scaled_height), resampling)
            canvas = Image.new("L", (target_width, target_height), 0)
            canvas.paste(resized, (left, top))
            return canvas
        raise ValueError("resize_mode must be 'resize', 'crop' or 'pad'.")

    @staticmethod
    def interpolate_bilinear(values: np.ndarray, *, target_height: int, target_width: int) -> np.ndarray:
        # Matches torch.nn.functional.interpolate(mode="bilinear", align_corners=False), which
        # ported inpaint surfaces use to shrink pixel masks onto the latent grid. Unlike PIL's
        # area-scaled BILINEAR filter, this samples a fixed 2x2 source neighborhood, so binarized
        # masks stay hard except where a boundary falls between the two sampled source pixels.
        source_height, source_width = values.shape
        y0, y1, weight_y = MaskUtil._source_axis_indices(source_height, target_height)
        x0, x1, weight_x = MaskUtil._source_axis_indices(source_width, target_width)
        top = values[y0][:, x0] * (1.0 - weight_x)[None, :] + values[y0][:, x1] * weight_x[None, :]
        bottom = values[y1][:, x0] * (1.0 - weight_x)[None, :] + values[y1][:, x1] * weight_x[None, :]
        return (top * (1.0 - weight_y)[:, None] + bottom * weight_y[:, None]).astype(np.float32)

    @staticmethod
    def _source_axis_indices(source_size: int, target_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scale = source_size / target_size
        source_coords = np.maximum(0.0, (np.arange(target_size, dtype=np.float64) + 0.5) * scale - 0.5)
        lower_indices = np.floor(source_coords).astype(np.int64)
        weights = (source_coords - lower_indices).astype(np.float32)
        upper_indices = np.minimum(lower_indices + 1, source_size - 1)
        return lower_indices, upper_indices, weights
