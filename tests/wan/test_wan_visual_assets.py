from pathlib import Path

import numpy as np
from PIL import Image

ASSET_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets" / "generation"
T2V_CONTACT_SHEET = ASSET_DIR / "wan2.2-ti2v-5b-t2v-256-17f-12steps-contact-sheet.png"
I2V_CONTACT_SHEET = ASSET_DIR / "wan2.2-ti2v-5b-i2v-bateau-128-5f-2steps-contact-sheet.png"


def test_wan_text_to_video_contact_sheet_is_nonblank_and_varied():
    WanVisualAssetAssertions.assert_contact_sheet(
        T2V_CONTACT_SHEET,
        min_width=1024,
        min_height=256,
        min_std=32.0,
        min_adjacent_delta=16.0,
    )


def test_wan_image_to_video_contact_sheet_is_nonblank_and_varied():
    WanVisualAssetAssertions.assert_contact_sheet(
        I2V_CONTACT_SHEET,
        min_width=384,
        min_height=128,
        min_std=32.0,
        min_adjacent_delta=16.0,
    )


class WanVisualAssetAssertions:
    @staticmethod
    def assert_contact_sheet(
        path: Path,
        min_width: int,
        min_height: int,
        min_std: float,
        min_adjacent_delta: float,
    ) -> None:
        assert path.exists(), f"missing committed Wan validation asset: {path}"

        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            assert width >= min_width
            assert height >= min_height
            pixels = np.asarray(image, dtype=np.float32)

        std = float(pixels.std())
        assert std >= min_std, f"{path.name} looks blank: std={std:.3f}"

        content = pixels[height // 10 :]
        crops = np.array_split(content, 4, axis=1)
        adjacent_deltas = [
            WanVisualAssetAssertions._mean_absolute_delta(left, right) for left, right in zip(crops, crops[1:])
        ]
        max_delta = max(adjacent_deltas)
        assert max_delta >= min_adjacent_delta, f"{path.name} looks static: max_delta={max_delta:.3f}"

    @staticmethod
    def _mean_absolute_delta(left: np.ndarray, right: np.ndarray) -> float:
        height = min(left.shape[0], right.shape[0])
        width = min(left.shape[1], right.shape[1])
        return float(np.abs(left[:height, :width] - right[:height, :width]).mean())
