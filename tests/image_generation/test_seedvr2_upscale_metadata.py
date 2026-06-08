import logging

import mlx.core as mx
import pytest
from PIL import Image

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.seedvr2.seedvr2_initializer import SeedVR2Initializer
from mflux.models.seedvr2.variants.upscale import seedvr2 as seedvr2_module
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2


@pytest.mark.fast
def test_seedvr2_upscale_metadata_records_final_output_dimensions(monkeypatch, tmp_path, caplog):
    source = tmp_path / "source.png"
    Image.new("RGB", (320, 192), (120, 90, 60)).save(source)

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = lambda txt, vid, timestep: mx.zeros_like(vid[:, :16])

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "encode",
        staticmethod(lambda vae, image, tiling_config: mx.zeros((1, 16, 16, 27), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "decode",
        staticmethod(lambda vae, latent, tiling_config: mx.zeros((1, 3, 256, 426), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2TextEmbeddings,
        "load_positive",
        staticmethod(lambda: mx.zeros((1, 1, 5120), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "apply_color_correction",
        staticmethod(lambda content, style: content),
    )

    caplog.set_level(logging.WARNING, logger="mflux.models.common.config.config")

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    result = model.generate_image(seed=42, image_path=source, resolution=256)
    metadata = result._get_metadata()

    assert result.image.size == (426, 256)
    assert metadata["width"] == 426
    assert metadata["height"] == 256
    assert metadata["image_path"] == str(source)
    assert metadata["source_image_width"] == 320
    assert metadata["source_image_height"] == 192
    assert "Rounding down" not in caplog.text


@pytest.mark.fast
def test_seedvr2_initializer_defaults_to_untiled_small_outputs_with_large_output_auto_decode():
    class DummyModel:
        pass

    model = DummyModel()
    SeedVR2Initializer._init_config(model, ModelConfig.seedvr2_3b())

    assert model.tiling_config.vae_encode_tiled is False
    assert model.tiling_config.vae_decode_tiles_per_dim == 0
    assert model.tiling_config.vae_decode_auto_tile_min_pixels == 1024 * 1024


@pytest.mark.fast
def test_seedvr2_large_outputs_auto_enable_tiled_decode(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (1024, 1024), (120, 90, 60)).save(source)
    seen: dict[str, object] = {}

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig(vae_encode_tiled=False, vae_decode_tiles_per_dim=0)
        model.bits = quantize
        model.vae = object()
        model.transformer = lambda txt, vid, timestep: mx.zeros_like(vid[:, :16])

    def fake_decode(vae, latent, tiling_config):
        seen["decode_tiling_config"] = tiling_config
        return mx.zeros((1, 3, 4, 4), dtype=mx.float32)

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "preprocess_image",
        staticmethod(lambda image_path, resolution, softness: (mx.zeros((1, 3, 4, 4), dtype=mx.float32), 2048, 2048)),
    )
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "encode",
        staticmethod(lambda vae, image, tiling_config: mx.zeros((1, 16, 4, 4), dtype=mx.float32)),
    )
    monkeypatch.setattr(seedvr2_module.VAEUtil, "decode", staticmethod(fake_decode))
    monkeypatch.setattr(
        seedvr2_module.SeedVR2TextEmbeddings,
        "load_positive",
        staticmethod(lambda: mx.zeros((1, 1, 5120), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "apply_color_correction",
        staticmethod(lambda content, style: content),
    )

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    model.generate_image(seed=42, image_path=source, resolution="2x")

    tiling_config = seen["decode_tiling_config"]
    assert tiling_config.vae_encode_tiled is False
    assert tiling_config.vae_decode_tiles_per_dim == 8
