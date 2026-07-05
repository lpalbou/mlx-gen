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
    ) -> np.ndarray:
        with Image.open(mask_path) as image:
            if alpha_warning_context is not None and "A" in image.getbands():
                print(f"⚠️  {alpha_warning_context} has an alpha channel; alpha is ignored and luminance is used.")
            mask_image = image.convert("L").resize((target_width, target_height), resampling)
        mask_values = np.asarray(mask_image, dtype=np.float32) / 255.0
        # White (>= 0.5) marks the editable region; black is preserved.
        return (mask_values >= 0.5).astype(np.float32)
