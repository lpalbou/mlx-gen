import numpy as np
import pytest
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.image_util import ImageUtil
from mflux.utils.mask_util import MaskUtil


def test_normalize_resize_mode_accepts_choices_and_rejects_unknown():
    assert DimensionResolver.normalize_resize_mode(None) == "resize"
    assert DimensionResolver.normalize_resize_mode("resize") == "resize"
    assert DimensionResolver.normalize_resize_mode("crop") == "crop"
    assert DimensionResolver.normalize_resize_mode("pad") == "pad"
    with pytest.raises(ValueError, match="Unsupported resize mode"):
        DimensionResolver.normalize_resize_mode("stretch")


def test_letterbox_geometry_centers_scaled_source():
    # 100x50 into 64x64: scale 0.64 -> 64x32 content, vertically centered.
    assert ImageUtil.letterbox_geometry(
        source_width=100, source_height=50, target_width=64, target_height=64
    ) == (64, 32, 0, 16)
    # Tall source pillarboxes horizontally.
    assert ImageUtil.letterbox_geometry(
        source_width=50, source_height=100, target_width=64, target_height=64
    ) == (32, 64, 16, 0)
    # Matching aspect fills the canvas exactly.
    assert ImageUtil.letterbox_geometry(
        source_width=200, source_height=100, target_width=64, target_height=32
    ) == (64, 32, 0, 0)
    with pytest.raises(ValueError, match="positive"):
        ImageUtil.letterbox_geometry(source_width=0, source_height=10, target_width=8, target_height=8)


def test_scale_to_dimensions_pad_letterboxes_with_black_default():
    image = Image.new("RGB", (100, 50), (255, 255, 255))

    padded = ImageUtil.scale_to_dimensions(image, target_width=64, target_height=64, resize_mode="pad")

    assert padded.size == (64, 64)
    pixels = np.asarray(padded)
    # Bars above and below are the black fill; the letterboxed content is white.
    assert pixels[:16].max() == 0
    assert pixels[48:].max() == 0
    assert pixels[16:48].min() == 255


def test_scale_to_dimensions_pad_honors_custom_fill_color():
    image = Image.new("RGB", (100, 50), (0, 0, 0))

    padded = ImageUtil.scale_to_dimensions(
        image, target_width=64, target_height=64, resize_mode="pad", fill_color=(255, 255, 255)
    )

    pixels = np.asarray(padded)
    assert pixels[:16].min() == 255
    assert pixels[48:].min() == 255
    assert pixels[16:48].max() == 0


def test_scale_to_dimensions_rejects_unknown_mode():
    image = Image.new("RGB", (10, 10))
    with pytest.raises(ValueError, match="resize_mode"):
        ImageUtil.scale_to_dimensions(image, target_width=8, target_height=8, resize_mode="letterbox")


def test_mask_pad_geometry_matches_image_pad_geometry_exactly():
    # A full-white source: after pad mapping, the mask's editable region must be
    # EXACTLY the image's content region (shared letterbox math), and the pad
    # borders must binarize to 0 (preserved).
    source_image = Image.new("RGB", (100, 50), (255, 255, 255))
    padded_image = ImageUtil.scale_to_dimensions(source_image, target_width=64, target_height=64, resize_mode="pad")
    image_content = (np.asarray(padded_image)[..., 0] > 127).astype(np.float32)

    mask = MaskUtil.map_mask_to_canvas(
        Image.new("L", (100, 50), 255),
        target_width=64,
        target_height=64,
        resampling=Image.Resampling.BOX,
        resize_mode="pad",
    )
    mask_values = (np.asarray(mask, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)

    np.testing.assert_array_equal(mask_values, image_content)


@pytest.mark.parametrize("resize_mode", ["resize", "crop", "pad"])
def test_mask_and_image_content_centroids_agree_across_modes(tmp_path, resize_mode):
    # A distinctive off-center marker must land on the same target pixels whether it
    # travels through the image mapping (LANCZOS) or the mask mapping (BOX + 0.5).
    # The marker sits inside the central 60x60 of the 120x60 source so the crop
    # mode's center square keeps it visible in all three modes.
    marker_box = (40, 10, 60, 25)  # left, top, right, bottom in a 120x60 source
    image = Image.new("RGB", (120, 60), (0, 0, 0))
    image.paste(Image.new("RGB", (marker_box[2] - marker_box[0], marker_box[3] - marker_box[1]), (255, 255, 255)),
                (marker_box[0], marker_box[1]))
    mask_path = tmp_path / "mask.png"
    image.convert("L").save(mask_path)

    mapped_image = ImageUtil.scale_to_dimensions(image, target_width=64, target_height=64, resize_mode=resize_mode)
    image_values = (np.asarray(mapped_image)[..., 0] > 127).astype(np.float32)
    mask_values = MaskUtil.load_binary_mask(
        mask_path,
        target_width=64,
        target_height=64,
        resampling=Image.Resampling.BOX,
        resize_mode=resize_mode,
    )

    assert image_values.sum() > 0
    assert mask_values.sum() > 0
    image_centroid = np.array(np.nonzero(image_values)).mean(axis=1)
    mask_centroid = np.array(np.nonzero(mask_values)).mean(axis=1)
    np.testing.assert_allclose(mask_centroid, image_centroid, atol=1.0)


def test_config_resize_mode_defaults_validates_and_is_orthogonal_to_canvas(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (432, 240), "white").save(image_path)
    model_config = ModelConfig.qwen_image()

    default_config = Config(model_config=model_config, num_inference_steps=1, height=256, width=256)
    assert default_config.resize_mode == "resize"

    with pytest.raises(ValueError, match="Unsupported resize mode"):
        Config(model_config=model_config, num_inference_steps=1, height=256, width=256, resize_mode="fit")

    # The canvas is resolved by canvas_policy alone: sweeping resize_mode never
    # changes the resolved dimensions, for either policy.
    for canvas_policy in ("source-aspect", "exact-resize"):
        resolved = {
            resize_mode: Config(
                model_config=model_config,
                num_inference_steps=1,
                height=256,
                width=256,
                image_path=image_path,
                image_strength=0.5,
                canvas_policy=canvas_policy,
                resize_mode=resize_mode,
            )
            for resize_mode in ("resize", "crop", "pad")
        }
        dims = {(config.width, config.height) for config in resolved.values()}
        assert len(dims) == 1
        assert all(config.resize_mode == mode for mode, config in resolved.items())
