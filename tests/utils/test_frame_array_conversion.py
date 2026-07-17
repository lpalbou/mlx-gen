import numpy as np
from PIL import Image

from mflux.utils.video_health import VideoHealth
from mflux.utils.video_util import VideoUtil


def _reference_getdata_array(image: Image.Image, dtype) -> np.ndarray:
    # The pre-buffer-protocol conversion this module replaced; kept as the parity oracle.
    rgb = image.convert("RGB")
    return np.asarray(rgb.getdata(), dtype=dtype).reshape(rgb.height, rgb.width, 3)


def _test_image() -> Image.Image:
    rng = np.random.default_rng(3)
    return Image.fromarray(rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8), "RGB")


def test_pil_rgb_to_array_matches_getdata_reference():
    image = _test_image()
    fast = VideoUtil._pil_rgb_to_array(image)
    assert fast.dtype == np.uint8
    assert np.array_equal(fast, _reference_getdata_array(image, np.uint8))


def test_pil_rgb_to_array_converts_non_rgb_modes():
    rgba = _test_image().convert("RGBA")
    fast = VideoUtil._pil_rgb_to_array(rgba)
    assert fast.shape == (48, 64, 3)
    assert np.array_equal(fast, _reference_getdata_array(rgba, np.uint8))


def test_luma_array_matches_getdata_reference():
    image = _test_image()
    fast = VideoHealth._luma_array(image)
    reference = VideoHealth._luma_rgb_array(_reference_getdata_array(image, np.float32))
    assert fast.dtype == np.float32
    assert np.allclose(fast, reference)
