import os

import mlx.core as mx
from PIL import Image

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.model_config import ModelConfig
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
