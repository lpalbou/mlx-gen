import os

import mlx.core as mx
from PIL import Image

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.variants.edit import (
    flux2_klein_edit as flux2_klein_edit_module,
    flux2_klein_inpaint as flux2_klein_inpaint_module,
)
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
from mflux.models.flux2.variants.edit.flux2_klein_inpaint import Flux2KleinInpaint
from mflux.models.qwen.qwen_initializer import QwenImageInitializer
from mflux.models.qwen.variants.controlnet import (
    qwen_controlnet_util as qwen_controlnet_util_module,
    qwen_image_controlnet as qwen_image_controlnet_module,
)
from mflux.models.qwen.variants.controlnet.qwen_controlnet_util import QwenControlNetUtil
from mflux.models.qwen.variants.controlnet.qwen_image_controlnet import QwenImageControlNet
from mflux.models.z_image.variants import z_image as z_image_module
from mflux.models.z_image.variants.z_image import ZImage


def test_qwen_control_inpaint_condition_packs_masked_latents(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (32, 32), color=(40, 80, 120)).save(source_path)
    mask_image = Image.new("L", (32, 32), color=0)
    for x in range(16, 32):
        for y in range(32):
            mask_image.putpixel((x, y), 255)
    mask_image.save(mask_path)

    captured = {}

    def fake_encode(vae, image, tiling_config):
        captured["encoded_input_shape"] = image.shape
        return mx.zeros((1, 16, 4, 4), dtype=mx.float32)

    def fake_pack(latents, height, width, num_channels_latents):
        captured["pack_shape"] = latents.shape
        captured["pack_height"] = height
        captured["pack_width"] = width
        captured["num_channels_latents"] = num_channels_latents
        captured["mask_channel"] = latents[:, 16:17]
        return latents

    monkeypatch.setattr(qwen_controlnet_util_module.VAEUtil, "encode", fake_encode)
    monkeypatch.setattr(qwen_controlnet_util_module.QwenLatentCreator, "pack_latents", fake_pack)

    packed = QwenControlNetUtil.create_inpaint_controlnet_condition(
        vae=object(),
        image_path=str(source_path),
        mask_path=str(mask_path),
        height=32,
        width=32,
        tiling_config=None,
    )

    assert packed.shape == (1, 17, 4, 4)
    assert captured["encoded_input_shape"] == (1, 3, 32, 32)
    assert captured["pack_shape"] == (1, 17, 4, 4)
    assert captured["pack_height"] == 32
    assert captured["pack_width"] == 32
    assert captured["num_channels_latents"] == 17
    assert float(mx.min(captured["mask_channel"][:, :, :, :2]).item()) == 1.0
    assert float(mx.max(captured["mask_channel"][:, :, :, :2]).item()) == 1.0
    assert float(mx.min(captured["mask_channel"][:, :, :, 2:]).item()) == 0.0
    assert float(mx.max(captured["mask_channel"][:, :, :, 2:]).item()) == 0.0


def test_qwen_control_inpaint_skips_cfg_negative_branch_when_guidance_is_one(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (32, 32), color=(40, 80, 120)).save(source_path)
    Image.new("L", (32, 32), color=255).save(mask_path)

    captured = {"transformer_calls": 0, "negative_prompt": "sentinel", "condition_calls": 0}

    def fake_init_controlnet(
        model,
        *,
        controlnet_model,
        model_config,
        quantize,
        model_path=None,
        lora_paths=None,
        lora_scales=None,
    ):
        model.model_config = model_config
        model.prompt_cache = {}
        model.controlnet_condition_cache = {}
        model.tokenizers = {"qwen": object()}
        model.text_encoder = object()
        model.vae = object()
        model.bits = quantize
        model.lora_paths = lora_paths
        model.lora_scales = lora_scales
        model.controlnet_model = controlnet_model
        model.tiling_config = None
        model.callbacks = CallbackRegistry()
        model.transformer_controlnet = lambda **kwargs: object()

        def fake_transformer(**kwargs):
            captured["transformer_calls"] += 1
            return mx.zeros_like(kwargs["hidden_states"])

        model.transformer = fake_transformer

    def fake_encode_prompt(
        *,
        prompt,
        negative_prompt,
        prompt_cache,
        qwen_tokenizer,
        qwen_text_encoder,
    ):
        captured["negative_prompt"] = negative_prompt
        embeddings = mx.zeros((1, 4, 8), dtype=mx.float32)
        mask = mx.ones((1, 4), dtype=mx.float32)
        return embeddings, mask, embeddings, mask

    def fake_encode_positive_prompt(
        *,
        prompt,
        prompt_cache,
        qwen_tokenizer,
        qwen_text_encoder,
    ):
        embeddings = mx.zeros((1, 4, 8), dtype=mx.float32)
        mask = mx.ones((1, 4), dtype=mx.float32)
        return embeddings, mask

    monkeypatch.setattr(QwenImageInitializer, "init_controlnet", fake_init_controlnet)
    monkeypatch.setattr(qwen_image_controlnet_module.QwenPromptEncoder, "encode_prompt", fake_encode_prompt)
    monkeypatch.setattr(
        qwen_image_controlnet_module.QwenPromptEncoder,
        "encode_positive_prompt",
        fake_encode_positive_prompt,
    )
    monkeypatch.setattr(
        qwen_image_controlnet_module.QwenControlNetUtil,
        "create_inpaint_controlnet_condition",
        staticmethod(
            lambda **kwargs: (
                captured.__setitem__("condition_calls", captured["condition_calls"] + 1)
                or mx.zeros((1, 4, 68), dtype=mx.float32)
            )
        ),
    )
    monkeypatch.setattr(
        qwen_image_controlnet_module.VAEUtil,
        "decode",
        staticmethod(lambda vae, latent, tiling_config=None: mx.zeros((1, 3, 32, 32), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        qwen_image_controlnet_module.ImageUtil,
        "to_image",
        staticmethod(lambda **kwargs: kwargs),
    )
    monkeypatch.setattr(
        qwen_image_controlnet_module.LoRALoader,
        "extra_metadata_for_model",
        staticmethod(lambda model: {}),
    )

    qwen = QwenImageControlNet(
        controlnet_model="dummy/controlnet.safetensors",
        quantize=8,
        model_config=ModelConfig.qwen_image(),
    )

    first = qwen.generate_image(
        seed=123,
        prompt="repair the masked area only",
        image_path=source_path,
        mask_path=mask_path,
        width=32,
        height=32,
        num_inference_steps=2,
        guidance=1.0,
        negative_prompt="should be ignored at guidance one",
    )
    qwen.generate_image(
        seed=123,
        prompt="repair the masked area only",
        image_path=source_path,
        mask_path=mask_path,
        width=32,
        height=32,
        num_inference_steps=2,
        guidance=1.0,
        negative_prompt="should still be ignored at guidance one",
    )
    stat = mask_path.stat()
    os.utime(mask_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    qwen.generate_image(
        seed=123,
        prompt="repair the masked area only",
        image_path=source_path,
        mask_path=mask_path,
        width=32,
        height=32,
        num_inference_steps=2,
        guidance=1.0,
        negative_prompt="should still be ignored after cache invalidation",
    )

    assert captured["negative_prompt"] == "sentinel"
    assert first["negative_prompt"] == "should be ignored at guidance one"
    assert captured["transformer_calls"] == 6
    assert captured["condition_calls"] == 2


def test_z_image_inpaint_mask_latents_and_blend_preserve_unmasked_region(tmp_path):
    mask_path = tmp_path / "mask.png"
    mask_image = Image.new("L", (32, 32), color=0)
    for x in range(16, 32):
        for y in range(32):
            mask_image.putpixel((x, y), 255)
    mask_image.save(mask_path)

    mask_latents = ZImage._create_inpaint_mask_latents(mask_path=mask_path, height=32, width=32)
    assert mask_latents.shape == (1, 1, 4, 4)
    assert float(mx.min(mask_latents[:, :, :, :2]).item()) == 0.0
    assert float(mx.max(mask_latents[:, :, :, :2]).item()) == 0.0
    assert float(mx.min(mask_latents[:, :, :, 2:]).item()) == 1.0
    assert float(mx.max(mask_latents[:, :, :, 2:]).item()) == 1.0

    latents = mx.ones((16, 1, 4, 4), dtype=mx.float32) * 9
    image_latents = mx.zeros((16, 1, 4, 4), dtype=mx.float32)
    noise = mx.ones((16, 1, 4, 4), dtype=mx.float32) * 2

    blended = ZImage._blend_inpaint_latents(
        latents=latents,
        image_latents=image_latents,
        initial_noise=noise,
        mask_latents=mask_latents,
        sigma=0.5,
    )

    assert bool(mx.allclose(blended[:, :, :, :2], mx.ones((16, 1, 4, 2), dtype=mx.float32)).item())
    assert bool(mx.allclose(blended[:, :, :, 2:], mx.ones((16, 1, 4, 2), dtype=mx.float32) * 9).item())


def test_z_image_inpaint_latents_reuse_run_noise(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (32, 32), color=(40, 80, 120)).save(source_path)
    Image.new("L", (32, 32), color=255).save(mask_path)

    model = ZImage.__new__(ZImage)
    model.vae = object()
    model.tiling_config = None
    model.inpaint_condition_cache = {}
    captured = {"encode_calls": 0, "mask_calls": 0}

    monkeypatch.setattr(
        z_image_module.LatentCreator,
        "encode_image",
        staticmethod(
            lambda **kwargs: (
                captured.__setitem__("encode_calls", captured["encode_calls"] + 1)
                or mx.zeros((1, 16, 4, 4), dtype=mx.float32)
            )
        ),
    )
    monkeypatch.setattr(
        z_image_module.ZImage,
        "_create_inpaint_mask_latents",
        staticmethod(
            lambda **kwargs: (
                captured.__setitem__("mask_calls", captured["mask_calls"] + 1)
                or mx.ones((1, 1, 4, 4), dtype=mx.float32)
            )
        ),
    )

    def fail_create_noise(*args, **kwargs):
        raise AssertionError("native inpaint should reuse the run noise instead of allocating a new seed-0 tensor")

    monkeypatch.setattr(
        z_image_module.ZImageLatentCreator,
        "create_noise",
        staticmethod(fail_create_noise),
    )

    initial_noise = mx.ones((16, 1, 4, 4), dtype=mx.float32) * 7
    inpaint_latents = model._create_inpaint_latents(
        image_path=source_path,
        mask_path=mask_path,
        height=32,
        width=32,
        initial_noise=initial_noise,
    )
    second_noise = mx.ones((16, 1, 4, 4), dtype=mx.float32) * 3
    second_inpaint_latents = model._create_inpaint_latents(
        image_path=source_path,
        mask_path=mask_path,
        height=32,
        width=32,
        initial_noise=second_noise,
    )
    stat = mask_path.stat()
    os.utime(mask_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    third_noise = mx.ones((16, 1, 4, 4), dtype=mx.float32) * 5
    third_inpaint_latents = model._create_inpaint_latents(
        image_path=source_path,
        mask_path=mask_path,
        height=32,
        width=32,
        initial_noise=third_noise,
    )

    assert bool(mx.allclose(inpaint_latents["noise"], initial_noise).item())
    assert bool(mx.allclose(second_inpaint_latents["noise"], second_noise).item())
    assert bool(mx.allclose(third_inpaint_latents["noise"], third_noise).item())
    assert captured["encode_calls"] == 2
    assert captured["mask_calls"] == 2


def test_qwen_control_inpaint_progress_uses_image_to_image_task(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (32, 32), color=(40, 80, 120)).save(source_path)
    Image.new("L", (32, 32), color=255).save(mask_path)

    def fake_init_controlnet(
        model,
        *,
        controlnet_model,
        model_config,
        quantize,
        model_path=None,
        lora_paths=None,
        lora_scales=None,
    ):
        model.model_config = model_config
        model.prompt_cache = {}
        model.controlnet_condition_cache = {}
        model.tokenizers = {"qwen": object()}
        model.text_encoder = object()
        model.vae = object()
        model.bits = quantize
        model.lora_paths = lora_paths
        model.lora_scales = lora_scales
        model.controlnet_model = controlnet_model
        model.tiling_config = None
        model.callbacks = CallbackRegistry()
        model.transformer_controlnet = lambda **kwargs: object()
        model.transformer = lambda **kwargs: mx.zeros_like(kwargs["hidden_states"])

    monkeypatch.setattr(QwenImageInitializer, "init_controlnet", fake_init_controlnet)
    monkeypatch.setattr(
        qwen_image_controlnet_module.QwenPromptEncoder,
        "encode_positive_prompt",
        staticmethod(
            lambda **kwargs: (
                mx.zeros((1, 4, 8), dtype=mx.float32),
                mx.ones((1, 4), dtype=mx.float32),
            )
        ),
    )
    monkeypatch.setattr(
        qwen_image_controlnet_module.QwenControlNetUtil,
        "create_inpaint_controlnet_condition",
        staticmethod(lambda **kwargs: mx.zeros((1, 4, 68), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        qwen_image_controlnet_module.VAEUtil,
        "decode",
        staticmethod(lambda vae, latent, tiling_config=None: mx.zeros((1, 3, 32, 32), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        qwen_image_controlnet_module.ImageUtil,
        "to_image",
        staticmethod(lambda **kwargs: kwargs),
    )
    monkeypatch.setattr(
        qwen_image_controlnet_module.LoRALoader,
        "extra_metadata_for_model",
        staticmethod(lambda model: None),
    )

    qwen = QwenImageControlNet(
        controlnet_model="dummy/controlnet.safetensors",
        quantize=8,
        model_config=ModelConfig.qwen_image(),
    )
    all_events = []
    img2img_events = []
    text_events = []
    qwen.callbacks.subscribe_progress(all_events.append)
    qwen.callbacks.subscribe_progress(img2img_events.append, task="image-to-image")
    qwen.callbacks.subscribe_progress(text_events.append, task="text-to-image")

    result = qwen.generate_image(
        seed=123,
        prompt="repair the masked area only",
        image_path=source_path,
        mask_path=mask_path,
        width=32,
        height=32,
        num_inference_steps=2,
        guidance=1.0,
    )

    assert [event.phase for event in all_events] == ["start", "denoise", "denoise", "complete"]
    assert [event.task for event in all_events] == ["image-to-image"] * 4
    assert img2img_events == all_events
    assert text_events == []
    assert result["extra_metadata"] == {"controlnet_model": "dummy/controlnet.safetensors"}


def test_qwen_control_route_none_negative_uses_blank_cfg_prompt_and_preserves_explicit_empty(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (32, 32), color=(40, 80, 120)).save(source_path)

    captured = {"negatives": []}

    def fake_init_controlnet(
        model,
        *,
        controlnet_model,
        model_config,
        quantize,
        model_path=None,
        lora_paths=None,
        lora_scales=None,
    ):
        model.model_config = model_config
        model.prompt_cache = {}
        model.controlnet_condition_cache = {}
        model.tokenizers = {"qwen": object()}
        model.text_encoder = object()
        model.vae = object()
        model.bits = quantize
        model.lora_paths = lora_paths
        model.lora_scales = lora_scales
        model.controlnet_model = controlnet_model
        model.tiling_config = None
        model.callbacks = CallbackRegistry()
        model.transformer_controlnet = lambda **kwargs: object()
        model.transformer = lambda **kwargs: mx.zeros_like(kwargs["hidden_states"])

    def fake_encode_prompt(
        *,
        prompt,
        negative_prompt,
        prompt_cache,
        qwen_tokenizer,
        qwen_text_encoder,
    ):
        captured["negatives"].append(negative_prompt)
        embeddings = mx.zeros((1, 4, 8), dtype=mx.float32)
        mask = mx.ones((1, 4), dtype=mx.float32)
        return embeddings, mask, embeddings, mask

    monkeypatch.setattr(QwenImageInitializer, "init_controlnet", fake_init_controlnet)
    monkeypatch.setattr(qwen_image_controlnet_module.QwenPromptEncoder, "encode_prompt", fake_encode_prompt)
    monkeypatch.setattr(
        qwen_image_controlnet_module.QwenControlNetUtil,
        "create_controlnet_condition",
        staticmethod(lambda **kwargs: mx.zeros((1, 4, 68), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        qwen_image_controlnet_module.VAEUtil,
        "decode",
        staticmethod(lambda vae, latent, tiling_config=None: mx.zeros((1, 3, 32, 32), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        qwen_image_controlnet_module.ImageUtil,
        "to_image",
        staticmethod(lambda **kwargs: kwargs),
    )
    monkeypatch.setattr(
        qwen_image_controlnet_module.LoRALoader,
        "extra_metadata_for_model",
        staticmethod(lambda model: {}),
    )

    qwen = QwenImageControlNet(
        controlnet_model="dummy/controlnet.safetensors",
        quantize=8,
        model_config=ModelConfig.qwen_image(),
    )

    qwen.generate_image(
        seed=123,
        prompt="follow the control image",
        controlnet_image_path=str(source_path),
        width=32,
        height=32,
        num_inference_steps=2,
        guidance=4.0,
        negative_prompt=None,
    )
    qwen.generate_image(
        seed=123,
        prompt="follow the control image",
        controlnet_image_path=str(source_path),
        width=32,
        height=32,
        num_inference_steps=2,
        guidance=4.0,
        negative_prompt="",
    )

    assert captured["negatives"] == [" ", ""]


def test_z_image_inpaint_progress_uses_image_to_image_task(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (32, 32), color=(40, 80, 120)).save(source_path)
    Image.new("L", (32, 32), color=255).save(mask_path)

    def fake_init(model, *, quantize, model_path=None, lora_paths=None, lora_scales=None, model_config):
        model.model_config = model_config
        model.tokenizers = {"z_image": object()}
        model.text_encoder = object()
        model.vae = object()
        model.transformer = lambda **kwargs: mx.zeros_like(kwargs["x"])
        model.prompt_cache = {}
        model.tiling_config = None
        model.bits = quantize
        model.lora_paths = lora_paths
        model.lora_scales = lora_scales
        model.callbacks = CallbackRegistry()

    monkeypatch.setattr(z_image_module.ZImageInitializer, "init", fake_init)
    monkeypatch.setattr(
        z_image_module.ZImageLatentCreator,
        "create_noise",
        staticmethod(lambda seed, height, width: mx.zeros((16, 1, 4, 4), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        z_image_module.ZImage,
        "_create_inpaint_latents",
        lambda self, **kwargs: {
            "image": mx.zeros((16, 1, 4, 4), dtype=mx.float32),
            "noise": mx.zeros((16, 1, 4, 4), dtype=mx.float32),
            "mask": mx.ones((1, 1, 4, 4), dtype=mx.float32),
        },
    )
    monkeypatch.setattr(
        z_image_module.ZImage,
        "_encode_prompts",
        lambda self, **kwargs: (
            mx.zeros((1, 4, 8), dtype=mx.float32),
            None,
        ),
    )
    monkeypatch.setattr(
        z_image_module.ZImage,
        "_decode_latents",
        lambda self, **kwargs: mx.zeros((1, 3, 32, 32), dtype=mx.float32),
    )
    monkeypatch.setattr(
        z_image_module.ImageUtil,
        "to_image",
        staticmethod(lambda **kwargs: kwargs),
    )
    monkeypatch.setattr(
        z_image_module.LoRALoader,
        "extra_metadata_for_model",
        staticmethod(lambda model: {}),
    )

    model = ZImage(quantize=8, model_config=ModelConfig.z_image_turbo())
    all_events = []
    img2img_events = []
    text_events = []
    model.callbacks.subscribe_progress(all_events.append)
    model.callbacks.subscribe_progress(img2img_events.append, task="image-to-image")
    model.callbacks.subscribe_progress(text_events.append, task="text-to-image")

    model.generate_image(
        seed=321,
        prompt="repair the masked area",
        image_path=source_path,
        mask_path=mask_path,
        width=32,
        height=32,
        num_inference_steps=2,
        guidance=1.0,
    )

    assert [event.phase for event in all_events] == ["start", "denoise", "denoise", "complete"]
    assert [event.task for event in all_events] == ["image-to-image"] * 4
    assert img2img_events == all_events
    assert text_events == []


def test_flux2_klein_inpaint_mask_binarizes_then_soft_downsamples_to_packed_grid(tmp_path):
    mask_path = tmp_path / "mask.png"
    mask_image = Image.new("L", (32, 32), color=0)
    for x in range(16, 32):
        for y in range(32):
            mask_image.putpixel((x, y), 255)
    # Off-binary gray pixels must binarize at pixel resolution before latent downsampling.
    for y in range(32):
        mask_image.putpixel((0, y), 96)
    mask_image.save(mask_path)

    mask = _Flux2KleinEditHelpers.prepare_inpaint_mask(mask_path=mask_path, height=32, width=32)

    assert mask.shape == (1, 4, 1)
    grid = mask.reshape(2, 2)
    assert float(mx.max(grid[:, 0]).item()) == 0.0
    assert float(mx.min(grid[:, 1]).item()) == 1.0


def test_flux2_klein_inpaint_mask_keeps_soft_boundary_values(tmp_path):
    mask_path = tmp_path / "mask.png"
    mask_image = Image.new("L", (32, 32), color=0)
    for x in range(8, 32):
        for y in range(32):
            mask_image.putpixel((x, y), 255)
    mask_image.save(mask_path)

    mask = _Flux2KleinEditHelpers.prepare_inpaint_mask(mask_path=mask_path, height=32, width=32)

    grid = mask.reshape(2, 2)
    # The left latent column covers pixels 0-15, half of which are white: soft 0.5 boundary.
    assert float(mx.max(mx.abs(grid[:, 0] - 0.5)).item()) < 1e-6
    assert float(mx.min(grid[:, 1]).item()) == 1.0


def test_flux2_klein_inpaint_blend_preserves_unmasked_region():
    clean = mx.zeros((1, 4, 8), dtype=mx.float32)
    noise = mx.ones((1, 4, 8), dtype=mx.float32) * 2
    denoised = mx.ones((1, 4, 8), dtype=mx.float32) * 9
    mask = mx.array([0.0, 0.0, 1.0, 1.0], dtype=mx.float32).reshape(1, 4, 1)
    sigmas = mx.array([1.0, 0.5, 0.0], dtype=mx.float32)

    preserved_mid = _Flux2KleinEditHelpers.preserved_source_latents(
        clean_latents=clean,
        noise_latents=noise,
        sigmas=sigmas,
        timestep=0,
    )
    blended_mid = (1.0 - mask) * preserved_mid + mask * denoised
    assert bool(mx.allclose(blended_mid[:, :2], mx.ones((1, 2, 8)) * 1.0).item())
    assert bool(mx.allclose(blended_mid[:, 2:], mx.ones((1, 2, 8)) * 9.0).item())

    preserved_last = _Flux2KleinEditHelpers.preserved_source_latents(
        clean_latents=clean,
        noise_latents=noise,
        sigmas=sigmas,
        timestep=1,
    )
    blended_last = (1.0 - mask) * preserved_last + mask * denoised
    assert bool(mx.allclose(blended_last[:, :2], mx.zeros((1, 2, 8))).item())
    assert bool(mx.allclose(blended_last[:, 2:], mx.ones((1, 2, 8)) * 9.0).item())


def test_flux2_klein_inpaint_source_conditioning_uses_reserved_t_coordinates():
    packed_source = mx.zeros((1, 4, 128), dtype=mx.float32)

    source_latents, source_ids = _Flux2KleinEditHelpers.prepare_inpaint_source_conditioning(
        packed_source_latents=packed_source,
        height=32,
        width=32,
        batch_size=1,
    )

    assert source_latents.shape == (1, 4, 128)
    assert source_ids.shape == (1, 4, 4)
    assert set(source_ids[:, :, 0].reshape(-1).tolist()) == {10}
    assert source_ids[0, :, 1].tolist() == [0, 0, 1, 1]
    assert source_ids[0, :, 2].tolist() == [0, 1, 0, 1]


def test_flux2_klein_inpaint_reference_conditioning_avoids_source_t_coordinate(tmp_path, monkeypatch):
    reference_a = tmp_path / "reference_a.png"
    reference_b = tmp_path / "reference_b.png"
    Image.new("RGB", (32, 32), color=(200, 30, 30)).save(reference_a)
    Image.new("RGB", (32, 32), color=(30, 200, 30)).save(reference_b)

    from mflux.models.flux2.variants.edit import flux2_klein_edit_helpers as helpers_module

    monkeypatch.setattr(
        helpers_module.LatentCreator,
        "encode_image",
        staticmethod(lambda **kwargs: mx.zeros((1, 32, 4, 4), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        helpers_module._Flux2KleinEditHelpers,
        "bn_normalize_vae_encoded_latents",
        staticmethod(lambda encoded, *, vae: encoded),
    )

    reference_latents, reference_ids = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
        vae=object(),
        tiling_config=None,
        image_paths=[reference_a, reference_b],
        height=32,
        width=32,
        batch_size=1,
        t_coord_start=20,
    )

    reference_t_coords = sorted(set(reference_ids[:, :, 0].reshape(-1).tolist()))
    assert reference_t_coords == [20, 30]

    source_latents, source_ids = _Flux2KleinEditHelpers.prepare_inpaint_source_conditioning(
        packed_source_latents=mx.zeros((1, 4, 128), dtype=mx.float32),
        height=32,
        width=32,
        batch_size=1,
    )
    combined_t_coords = sorted(
        set(source_ids[:, :, 0].reshape(-1).tolist()) | set(reference_ids[:, :, 0].reshape(-1).tolist())
    )
    assert combined_t_coords == [10, 20, 30]
    assert source_latents.shape[2] == reference_latents.shape[2]


def test_mask_util_interpolate_bilinear_matches_torch_align_corners_false_semantics():
    import numpy as np

    from mflux.utils.mask_util import MaskUtil

    # 48 -> 3 downsample: output pixel centers map to source coords 7.5, 23.5, 39.5,
    # so a ramp must produce exactly those interpolated values (align_corners=False).
    ramp = np.tile(np.arange(48, dtype=np.float32), (48, 1))
    result = MaskUtil.interpolate_bilinear(ramp, target_height=3, target_width=3)

    assert result.shape == (3, 3)
    expected_columns = np.array([7.5, 23.5, 39.5], dtype=np.float32)
    assert np.allclose(result, np.tile(expected_columns, (3, 1)), atol=1e-6)

    # Binary vertical split at 24: cells stay hard unless their sampled 2x2 neighborhood
    # straddles the boundary. The middle center (23.5) samples columns 23 and 24 -> 0.5.
    binary = np.zeros((48, 48), dtype=np.float32)
    binary[:, 24:] = 1.0
    binary_result = MaskUtil.interpolate_bilinear(binary, target_height=3, target_width=3)
    assert binary_result[0].tolist() == [0.0, 0.5, 1.0]


def test_flux2_klein_inpaint_progress_and_masked_loop(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (32, 32), color=(40, 80, 120)).save(source_path)
    mask_image = Image.new("L", (32, 32), color=0)
    for x in range(16, 32):
        for y in range(32):
            mask_image.putpixel((x, y), 255)
    mask_image.save(mask_path)

    observed = {"transformer_calls": 0, "hidden_seq_lens": []}

    def fake_init(model, *, quantize, model_path=None, lora_paths=None, lora_scales=None, model_config):
        model.model_config = model_config
        model.tokenizers = {"qwen3": object()}
        model.text_encoder = object()
        model.bits = quantize
        model.lora_paths = lora_paths
        model.lora_scales = lora_scales
        model.tiling_config = None
        model.callbacks = CallbackRegistry()

        class FakeVAE:
            def decode_packed_latents(self, packed_latents):
                observed["decoded_shape"] = packed_latents.shape
                return mx.zeros((1, 3, 32, 32), dtype=mx.float32)

        model.vae = FakeVAE()

        def fake_transformer(**kwargs):
            observed["transformer_calls"] += 1
            observed["hidden_seq_lens"].append(kwargs["hidden_states"].shape[1])
            observed["img_ids_t_coords"] = sorted(set(kwargs["img_ids"][:, :, 0].reshape(-1).tolist()))
            return mx.zeros_like(kwargs["hidden_states"])

        model.transformer = fake_transformer

    monkeypatch.setattr(flux2_klein_inpaint_module.Flux2Initializer, "init", fake_init)
    monkeypatch.setattr(
        flux2_klein_edit_module.AppleSiliconUtil,
        "is_m1_or_m2",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(
        flux2_klein_inpaint_module._Flux2KleinEditHelpers,
        "encode_reference_image_to_packed_latents",
        staticmethod(lambda **kwargs: mx.zeros((1, 4, 128), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        flux2_klein_inpaint_module.Flux2KleinEdit,
        "_encode_prompt_pair",
        lambda self, *, prompt, negative_prompt, guidance: (
            mx.zeros((1, 4, 8), dtype=mx.float32),
            mx.zeros((1, 4, 4), dtype=mx.float32),
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        flux2_klein_inpaint_module.ImageUtil,
        "to_image",
        staticmethod(lambda **kwargs: kwargs),
    )
    monkeypatch.setattr(
        flux2_klein_inpaint_module.LoRALoader,
        "extra_metadata_for_model",
        staticmethod(lambda model: {}),
    )

    model = Flux2KleinInpaint(quantize=8, model_config=ModelConfig.flux2_klein_4b())
    all_events = []
    img2img_events = []
    text_events = []
    model.callbacks.subscribe_progress(all_events.append)
    model.callbacks.subscribe_progress(img2img_events.append, task="image-to-image")
    model.callbacks.subscribe_progress(text_events.append, task="text-to-image")

    result = model.generate_image(
        seed=42,
        prompt="repair the masked area",
        image_path=source_path,
        mask_path=mask_path,
        width=32,
        height=32,
        num_inference_steps=2,
        guidance=1.0,
    )

    assert [event.phase for event in all_events] == ["start", "denoise", "denoise", "complete"]
    assert [event.task for event in all_events] == ["image-to-image"] * 4
    assert img2img_events == all_events
    assert text_events == []
    # Sequence: 4 generation tokens + 4 clean-source conditioning tokens.
    assert observed["transformer_calls"] == 2
    assert observed["hidden_seq_lens"] == [8, 8]
    assert observed["img_ids_t_coords"] == [0, 10]
    assert observed["decoded_shape"] == (1, 128, 2, 2)
    assert result["masked_image_path"] == mask_path


def test_flux2_klein_inpaint_rejects_image_strength(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (32, 32), color=(40, 80, 120)).save(source_path)
    Image.new("L", (32, 32), color=255).save(mask_path)

    def fake_init(model, *, quantize, model_path=None, lora_paths=None, lora_scales=None, model_config):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()

    monkeypatch.setattr(flux2_klein_inpaint_module.Flux2Initializer, "init", fake_init)

    model = Flux2KleinInpaint(quantize=8, model_config=ModelConfig.flux2_klein_4b())

    try:
        model.generate_image(
            seed=42,
            prompt="repair the masked area",
            image_path=source_path,
            mask_path=mask_path,
            image_strength=0.5,
            num_inference_steps=2,
        )
        raise AssertionError("image_strength must be rejected for the masked edit route")
    except ValueError as exc:
        assert "image_strength cannot be combined with mask_path" in str(exc)


def test_z_image_guidance_uses_standard_cfg_formula(monkeypatch):
    monkeypatch.setattr(z_image_module.AppleSiliconUtil, "is_m1_or_m2", staticmethod(lambda: True))

    def fake_transformer(*, cap_feats, **kwargs):
        return cap_feats

    predict = ZImage._predict(fake_transformer)
    out = predict(
        latents=mx.zeros((1, 1, 1), dtype=mx.float32),
        timestep=mx.array([0.0], dtype=mx.float32),
        sigmas=mx.array([0.0], dtype=mx.float32),
        text_encodings=mx.array([[[1.0]]], dtype=mx.float32),
        negative_encodings=mx.array([[[0.0]]], dtype=mx.float32),
        guidance=2.0,
    )

    assert float(out.item()) == 2.0
