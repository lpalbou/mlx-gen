import numpy as np
import pytest
from PIL import Image

from mflux.utils.mask_util import MaskUtil


def _write_mask(path, *, size=(8, 8), color=0, mode="L"):
    Image.new(mode, size, color).save(path)
    return path


def test_load_binary_mask_binarizes_at_half(tmp_path):
    mask_path = tmp_path / "mask.png"
    image = Image.new("L", (8, 8), 0)
    for x in range(4, 8):
        for y in range(8):
            image.putpixel((x, y), 255)
    image.save(mask_path)

    mask = MaskUtil.load_binary_mask(
        mask_path,
        target_width=8,
        target_height=8,
        resampling=Image.Resampling.NEAREST,
    )

    assert mask.dtype == np.float32
    assert mask.shape == (8, 8)
    assert set(np.unique(mask)) == {0.0, 1.0}
    assert mask[:, :4].sum() == 0
    assert mask[:, 4:].sum() == 32


def test_load_binary_mask_downsamples_to_target_grid(tmp_path):
    mask_path = _write_mask(tmp_path / "mask.png", size=(64, 64), color=255)

    mask = MaskUtil.load_binary_mask(
        mask_path,
        target_width=8,
        target_height=8,
        resampling=Image.Resampling.BOX,
    )

    assert mask.shape == (8, 8)
    assert mask.sum() == 64


def test_load_binary_mask_warns_on_alpha_channel(tmp_path, capsys):
    mask_path = _write_mask(tmp_path / "mask.png", mode="RGBA", color=(255, 255, 255, 128))

    MaskUtil.load_binary_mask(
        mask_path,
        target_width=8,
        target_height=8,
        resampling=Image.Resampling.NEAREST,
        alpha_warning_context="Test mask",
    )

    output = capsys.readouterr().out
    assert "Test mask has an alpha channel" in output


def test_load_binary_mask_silent_without_context(tmp_path, capsys):
    mask_path = _write_mask(tmp_path / "mask.png", mode="RGBA", color=(255, 255, 255, 128))

    MaskUtil.load_binary_mask(
        mask_path,
        target_width=8,
        target_height=8,
        resampling=Image.Resampling.NEAREST,
    )

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("loader", "expected_resampling"),
    [
        ("qwen_edit", Image.Resampling.NEAREST),
        ("z_image", Image.Resampling.NEAREST),
        ("wan", Image.Resampling.BOX),
    ],
)
def test_surface_resampling_policies_are_pinned(tmp_path, monkeypatch, loader, expected_resampling):
    # Reference-faithful policy: diffusers-ported surfaces keep NEAREST; in-house Wan uses BOX.
    observed = {}
    real_load = MaskUtil.load_binary_mask

    def spy(mask_path, **kwargs):
        observed["resampling"] = kwargs["resampling"]
        return real_load(mask_path, **kwargs)

    monkeypatch.setattr(MaskUtil, "load_binary_mask", staticmethod(spy))
    mask_path = _write_mask(tmp_path / "mask.png", size=(64, 64), color=255)

    if loader == "qwen_edit":
        from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil

        QwenEditUtil.create_inpaint_mask_latents(str(mask_path), height=64, width=64)
    elif loader == "z_image":
        from mflux.models.z_image.variants.z_image import ZImage

        ZImage._create_inpaint_mask_latents(mask_path=mask_path, height=64, width=64)
    else:
        from types import SimpleNamespace

        from mflux.models.wan.variants.wan2_2_ti2v import Wan2_2_TI2V

        model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
        model.vae = SimpleNamespace(spatial_scale=8)
        model._prepare_video_mask(mask_path, height=64, width=64)

    assert observed["resampling"] == expected_resampling
