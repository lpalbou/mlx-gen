import json
import os
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.wan.model.wan_transformer import WanTransformer
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.variants import Wan2_2_TI2V
from mflux.models.wan.variants.wan2_2_ti2v import _GUIDANCE_2_UNSET
from mflux.models.wan.wan_initializer import WanInitializer
from mflux.models.wan.weights import WanWeightDefinition
from mflux.models.wan.weights.wan_weight_mapping import WanWeightMapping


def test_wan_a14b_t2v_config_resolves_exact_repo():
    config = ModelConfig.from_name("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

    assert config is ModelConfig.wan2_2_t2v_a14b()
    assert config.transformer_overrides["has_transformer_2"] is True
    assert config.transformer_overrides["expand_timesteps"] is False
    assert config.transformer_overrides["boundary_ratio"] == 0.875
    assert config.transformer_overrides["flow_shift"] == 3.0
    assert config.transformer_overrides["vae_variant"] == "wan21"
    assert config.transformer_overrides["default_guidance"] == 4.0
    assert config.transformer_overrides["default_guidance_2"] == 3.0
    assert "低质量" in config.transformer_overrides["default_negative_prompt"]


def test_wan_ti2v_5b_config_matches_official_generation_defaults():
    config = ModelConfig.from_name("Wan-AI/Wan2.2-TI2V-5B-Diffusers")

    assert config is ModelConfig.wan2_2_ti2v_5b()
    assert config.transformer_overrides["flow_shift"] == 5.0
    assert config.transformer_overrides["default_guidance"] == 5.0
    assert config.transformer_overrides["default_steps"] == 50
    assert config.transformer_overrides["default_fps"] == 24
    assert config.transformer_overrides["default_frames"] == 121
    assert "低质量" in config.transformer_overrides["default_negative_prompt"]
    assert config.transformer_overrides.get("default_guidance_2") is None


def test_wan_a14b_i2v_config_resolves_local_path_by_alias():
    config = ModelConfig.from_name("models/wan2.2-i2v-a14b-8bit")

    assert config.base_model == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    assert config.transformer_overrides["in_channels"] == 36
    assert config.transformer_overrides["out_channels"] == 16
    assert config.transformer_overrides["supports_image_to_video"] is True
    assert config.transformer_overrides["default_guidance"] == 3.5
    assert config.transformer_overrides["default_guidance_2"] == 3.5


def test_wan_a14b_weight_definition_includes_second_transformer():
    definition = WanWeightDefinition.for_config(ModelConfig.wan2_2_t2v_a14b())
    components = definition.get_components()

    assert [component.name for component in components] == ["transformer", "transformer_2", "vae"]
    assert components[0].num_layers == 40
    assert components[1].num_layers == 40
    assert "transformer_2/*.safetensors" in definition.get_download_patterns()


def test_wan_initializer_streams_component_weight_application(monkeypatch, tmp_path):
    calls = []
    model = SimpleNamespace(
        transformer=SimpleNamespace(name="transformer"),
        transformer_2=SimpleNamespace(name="transformer_2"),
        vae=SimpleNamespace(name="vae"),
        bits=None,
    )
    definition = WanWeightDefinition.for_config(ModelConfig.wan2_2_t2v_a14b())

    def fake_load_component(root_path, component, raw_weights_cache=None):
        assert root_path == tmp_path
        assert raw_weights_cache is None
        calls.append(("load", component.name))
        return {"marker": component.name}, 8, "test-version"

    def fake_apply_and_quantize_single(weights, model, component, quantize_arg, quantization_predicate=None):
        assert quantize_arg == 8
        assert quantization_predicate is definition.quantization_predicate
        assert weights.components == {component.name: {"marker": component.name}}
        assert weights.meta_data.quantization_level == 8
        calls.append(("apply", component.name, model.name))
        return 8

    monkeypatch.setattr(WeightLoader, "_load_component", fake_load_component)
    monkeypatch.setattr(WeightApplier, "apply_and_quantize_single", fake_apply_and_quantize_single)
    monkeypatch.setattr("mflux.models.wan.wan_initializer.gc.collect", lambda: calls.append(("gc",)))
    monkeypatch.setattr("mflux.models.wan.wan_initializer.mx.synchronize", lambda: calls.append(("sync",)))
    monkeypatch.setattr("mflux.models.wan.wan_initializer.mx.clear_cache", lambda: calls.append(("clear",)))

    WanInitializer._load_and_apply_weights(
        model=model,
        root_path=tmp_path,
        quantize=8,
        weight_definition=definition,
    )

    assert model.bits == 8
    assert calls == [
        ("load", "transformer"),
        ("apply", "transformer", "transformer"),
        ("gc",),
        ("sync",),
        ("clear",),
        ("load", "transformer_2"),
        ("apply", "transformer_2", "transformer_2"),
        ("gc",),
        ("sync",),
        ("clear",),
        ("load", "vae"),
        ("apply", "vae", "vae"),
        ("gc",),
        ("sync",),
        ("clear",),
    ]


def test_wan_initializer_rejects_component_quantization_mismatch(monkeypatch, tmp_path):
    model = SimpleNamespace(
        transformer=SimpleNamespace(name="transformer"),
        transformer_2=SimpleNamespace(name="transformer_2"),
        vae=SimpleNamespace(name="vae"),
        bits=None,
    )
    definition = WanWeightDefinition.for_config(ModelConfig.wan2_2_t2v_a14b())
    resolved_bits = {"transformer": 8, "transformer_2": 4, "vae": 8}
    cleanup_calls = []

    def fake_load_component(root_path, component, raw_weights_cache=None):
        return {"marker": component.name}, resolved_bits[component.name], "test-version"

    def fake_apply_and_quantize_single(weights, model, component, quantize_arg, quantization_predicate=None):
        return weights.meta_data.quantization_level

    monkeypatch.setattr(WeightLoader, "_load_component", fake_load_component)
    monkeypatch.setattr(WeightApplier, "apply_and_quantize_single", fake_apply_and_quantize_single)
    monkeypatch.setattr("mflux.models.wan.wan_initializer.gc.collect", lambda: cleanup_calls.append("gc"))
    monkeypatch.setattr("mflux.models.wan.wan_initializer.mx.synchronize", lambda: cleanup_calls.append("sync"))
    monkeypatch.setattr("mflux.models.wan.wan_initializer.mx.clear_cache", lambda: cleanup_calls.append("clear"))

    with pytest.raises(ValueError, match="Wan component quantization mismatch"):
        WanInitializer._load_and_apply_weights(
            model=model,
            root_path=tmp_path,
            quantize=None,
            weight_definition=definition,
        )
    assert cleanup_calls == ["gc", "sync", "clear", "gc", "sync", "clear"]


def test_wan_initializer_allows_skipped_vae_without_quantization_metadata(monkeypatch, tmp_path):
    model = SimpleNamespace(
        transformer=SimpleNamespace(name="transformer"),
        transformer_2=SimpleNamespace(name="transformer_2"),
        vae=SimpleNamespace(name="vae"),
        bits=None,
    )
    definition = WanWeightDefinition.for_config(ModelConfig.wan2_2_t2v_a14b())
    resolved_bits = {"transformer": 8, "transformer_2": 8, "vae": None}

    def fake_load_component(root_path, component, raw_weights_cache=None):
        return {"marker": component.name}, resolved_bits[component.name], "test-version"

    def fake_apply_and_quantize_single(weights, model, component, quantize_arg, quantization_predicate=None):
        return weights.meta_data.quantization_level

    monkeypatch.setattr(WeightLoader, "_load_component", fake_load_component)
    monkeypatch.setattr(WeightApplier, "apply_and_quantize_single", fake_apply_and_quantize_single)
    monkeypatch.setattr("mflux.models.wan.wan_initializer.gc.collect", lambda: None)
    monkeypatch.setattr("mflux.models.wan.wan_initializer.mx.synchronize", lambda: None)
    monkeypatch.setattr("mflux.models.wan.wan_initializer.mx.clear_cache", lambda: None)

    WanInitializer._load_and_apply_weights(
        model=model,
        root_path=tmp_path,
        quantize=None,
        weight_definition=definition,
    )

    assert model.bits == 8


def test_wan_a14b_vae_mapping_uses_wan21_flat_encoder_and_plural_upsamplers():
    targets = {target.to_pattern for target in WanWeightMapping.get_vae_mapping(variant="wan21")}

    assert "encoder.down_blocks.0.conv1.conv3d.weight" in targets
    assert "encoder.down_blocks.2.resample_conv.weight" in targets
    assert "decoder.up_blocks.0.upsamplers.0.resample_conv.weight" in targets


def test_wan_a14b_vae_config_uses_16_channel_latents_and_scale_8():
    vae = Wan2_2_VAE(**ModelConfig.wan2_2_t2v_a14b().transformer_overrides["vae_config"])

    assert vae.z_dim == 16
    assert vae.spatial_scale == 8
    assert vae.temporal_scale == 4
    assert vae.patch_size == 1
    assert vae.latents_mean.shape == (16,)


def test_wan_a14b_boundary_selects_low_noise_transformer_below_boundary():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_t2v_a14b()
    model.transformer = SimpleNamespace(name="high")
    model.transformer_2 = SimpleNamespace(name="low")

    high, high_guidance = model._select_transformer_and_guidance(
        timestep=900,
        boundary_timestep=875,
        guidance=4.0,
        guidance_2=3.0,
    )
    low, low_guidance = model._select_transformer_and_guidance(
        timestep=800,
        boundary_timestep=875,
        guidance=4.0,
        guidance_2=3.0,
    )

    assert high.name == "high"
    assert high_guidance == 4.0
    assert low.name == "low"
    assert low_guidance == 3.0


def test_wan_a14b_can_release_high_noise_transformer_after_boundary(monkeypatch):
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    high = SimpleNamespace(name="high")
    low = SimpleNamespace(name="low")
    model.transformer = high
    model.transformer_2 = low
    calls = []
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.gc.collect", lambda: calls.append("gc"))
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.mx.synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.mx.clear_cache", lambda: calls.append("clear"))

    released = model._maybe_release_high_noise_denoiser(
        timestep=900,
        boundary_timestep=900,
        release_inactive_denoiser=True,
        already_released=False,
    )
    assert released is False
    assert model.transformer is high

    released = model._maybe_release_high_noise_denoiser(
        timestep=875,
        boundary_timestep=900,
        release_inactive_denoiser=True,
        already_released=False,
    )

    assert released is True
    assert model.transformer is None
    assert model.transformer_2 is low
    assert calls == ["gc", "sync", "clear"]

    released = model._maybe_release_high_noise_denoiser(
        timestep=800,
        boundary_timestep=900,
        release_inactive_denoiser=True,
        already_released=True,
    )

    assert released is True
    assert calls == ["gc", "sync", "clear"]


def test_wan_generate_releases_high_noise_transformer_before_low_noise_call(monkeypatch):
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.bits = None
    model.vae = SimpleNamespace(
        z_dim=16,
        temporal_scale=4,
        spatial_scale=8,
        decode_normalized_latents=lambda latents, **kwargs: mx.zeros((1, 3, 1, 8, 8), dtype=mx.float32),
    )
    calls = []

    class FakeTransformer:
        patch_size = (1, 2, 2)
        in_channels = 36
        out_channels = 16

        def __init__(self, name):
            self.name = name

        def __call__(self, **kwargs):
            calls.append((self.name, model.transformer is None))
            return mx.zeros((1, 16, 1, 8, 8), dtype=mx.float32)

    class FakeScheduler:
        num_train_timesteps = 1000

        def __init__(self, flow_shift):
            self.flow_shift = flow_shift
            self.timesteps = mx.array([], dtype=mx.int64)

        def set_timesteps(self, num_inference_steps):
            self.timesteps = mx.array([900, 875], dtype=mx.int64)

        def step(self, model_output, timestep, sample, return_dict):
            return (sample,)

    monkeypatch.setattr(
        "mflux.models.wan.variants.wan2_2_ti2v.WanUniPCMultistepScheduler",
        FakeScheduler,
    )
    monkeypatch.setattr(
        "mflux.models.wan.variants.wan2_2_ti2v.WanEulerScheduler",
        FakeScheduler,
    )
    monkeypatch.setattr(
        "mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(model, "encode_prompt", lambda **kwargs: (mx.zeros((1, 1, 4096)), None))
    monkeypatch.setattr(
        model,
        "prepare_latents",
        lambda **kwargs: mx.zeros((1, 16, 1, 8, 8), dtype=mx.float32),
    )
    monkeypatch.setattr(
        model,
        "_encode_video_condition",
        lambda **kwargs: mx.zeros((1, 20, 1, 8, 8), dtype=mx.float32),
    )
    monkeypatch.setattr(
        model,
        "_resolve_video_spatial_size",
        lambda **kwargs: (kwargs["height"], kwargs["width"], {}),
    )
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.gc.collect", lambda: None)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.mx.synchronize", lambda: None)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.mx.clear_cache", lambda: None)
    model.transformer = FakeTransformer("high")
    model.transformer_2 = FakeTransformer("low")

    model.generate_video(
        seed=1,
        prompt="a slow wave",
        width=64,
        height=64,
        num_frames=1,
        num_inference_steps=2,
        guidance=1,
        guidance_2=1,
        image_path="input.png",
        release_inactive_denoiser=True,
    )

    assert calls == [("high", False), ("low", True)]


def test_wan_generate_fails_on_non_finite_noise_prediction(monkeypatch):
    model = _fake_t2v_a14b_model()
    model.transformer = _FakeWanTransformer(mx.array([[[[[np.nan]]]]], dtype=mx.float32))
    model.transformer_2 = _FakeWanTransformer(mx.zeros((1, 16, 1, 8, 8), dtype=mx.float32))
    events = []
    calls = _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError) as exc_info:
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=1,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            progress_callback=events.append,
            tensor_health_check_interval=1,
        )

    message = str(exc_info.value)
    assert "wan-conditional-denoise-prediction" in message
    assert "tensor=noise_pred" in message
    assert "step=1/2" in message
    assert "timestep=900" in message
    assert "denoiser=high" in message
    assert [event.phase for event in events] == ["start"]
    assert calls == {"scheduler_steps": 0, "to_video": 0, "scheduler_flow_shift": 3.0}


def test_wan_generate_fails_on_non_finite_scheduler_latents(monkeypatch):
    model = _fake_t2v_a14b_model()
    events = []
    calls = _patch_fake_wan_generation(
        monkeypatch,
        model,
        scheduler_output=mx.array([[[[[np.inf]]]]], dtype=mx.float32),
    )

    with pytest.raises(ValueError) as exc_info:
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=1,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            progress_callback=events.append,
            tensor_health_check_interval=1,
        )

    message = str(exc_info.value)
    assert "wan-scheduler-step" in message
    assert "tensor=latents" in message
    assert "step=1/2" in message
    assert "timestep=900" in message
    assert "denoiser=high" in message
    assert [event.phase for event in events] == ["start"]
    assert calls == {"scheduler_steps": 1, "to_video": 0, "scheduler_flow_shift": 3.0}


def test_wan_generate_fails_on_non_finite_vae_decode(monkeypatch):
    model = _fake_t2v_a14b_model()
    decoded = np.zeros((1, 3, 1, 8, 8), dtype=np.float32)
    decoded[0, 0, 0, 0, 0] = np.nan
    model.vae.decode_normalized_latents = lambda latents, **kwargs: mx.array(decoded)
    events = []
    calls = _patch_fake_wan_generation(monkeypatch, model, patch_to_video=False)

    with pytest.raises(ValueError) as exc_info:
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=1,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            progress_callback=events.append,
        )

    message = str(exc_info.value)
    assert "wan-vae-decode" in message
    assert "tensor=decoded" in message
    assert [event.phase for event in events] == ["start", "denoise", "denoise", "decode"]
    assert calls == {"scheduler_steps": 2, "to_video": 0, "scheduler_flow_shift": 3.0}


def test_wan_generate_fails_on_non_finite_i2v_condition(monkeypatch):
    model = _fake_t2v_a14b_model()
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer.in_channels = 36
    model.transformer_2.in_channels = 36
    events = []
    calls = _patch_fake_wan_generation(monkeypatch, model)
    monkeypatch.setattr(
        model,
        "_encode_video_condition",
        lambda **kwargs: mx.array([[[[[np.nan]]]]], dtype=mx.float32),
    )

    with pytest.raises(ValueError) as exc_info:
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=1,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            image_path="input.png",
            progress_callback=events.append,
        )

    message = str(exc_info.value)
    assert "wan-image-conditioning" in message
    assert "tensor=condition" in message
    assert [event.phase for event in events] == ["start"]
    assert calls == {"scheduler_steps": 0, "to_video": 0, "scheduler_flow_shift": 3.0}


def test_wan_generate_rejects_reuse_after_denoiser_release():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer = None
    model.transformer_2 = SimpleNamespace()

    with pytest.raises(ValueError, match="denoisers have been released"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=1,
            num_inference_steps=2,
            guidance=1,
            image_path="input.png",
        )


def test_wan_generate_rejects_non_finite_latents_during_denoise(monkeypatch):
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_t2v_a14b()
    model.bits = 8
    model.vae = SimpleNamespace(
        z_dim=16,
        temporal_scale=4,
        spatial_scale=8,
        decode_normalized_latents=lambda latents, **kwargs: mx.zeros((1, 3, 1, 8, 8), dtype=mx.float32),
    )

    class FakeTransformer:
        patch_size = (1, 2, 2)
        in_channels = 16
        out_channels = 16

        def __call__(self, **kwargs):
            return mx.zeros((1, 16, 1, 8, 8), dtype=mx.float32)

    class FakeScheduler:
        num_train_timesteps = 1000

        def __init__(self, flow_shift):
            self.flow_shift = flow_shift
            self.timesteps = mx.array([], dtype=mx.int64)

        def set_timesteps(self, num_inference_steps):
            self.timesteps = mx.array([900, 800], dtype=mx.int64)

        def step(self, model_output, timestep, sample, return_dict):
            del model_output, timestep, return_dict
            failed = np.zeros(sample.shape, dtype=np.float32)
            failed[0, 0, 0, 0, 0] = np.nan
            return (mx.array(failed),)

    def fail_video_conversion(**kwargs):
        raise AssertionError("video conversion should not run after non-finite denoise latents")

    monkeypatch.setattr(
        "mflux.models.wan.variants.wan2_2_ti2v.WanUniPCMultistepScheduler",
        FakeScheduler,
    )
    monkeypatch.setattr(
        "mflux.models.wan.variants.wan2_2_ti2v.WanEulerScheduler",
        FakeScheduler,
    )
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video", fail_video_conversion)
    monkeypatch.setattr(model, "encode_prompt", lambda **kwargs: (mx.zeros((1, 1, 4096)), None))
    monkeypatch.setattr(
        model,
        "_resolve_video_spatial_size",
        lambda **kwargs: (kwargs["height"], kwargs["width"], {}),
    )
    monkeypatch.setattr(
        model,
        "prepare_latents",
        lambda **kwargs: mx.zeros((1, 16, 1, 8, 8), dtype=mx.float32),
    )
    model.transformer = FakeTransformer()
    model.transformer_2 = FakeTransformer()

    with pytest.raises(ValueError, match="phase=wan-scheduler-step.*step=1/2"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=1,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            tensor_health_check_interval=1,
        )


def test_wan_a14b_default_guidance_pair_uses_model_defaults():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_t2v_a14b()

    guidance, guidance_2 = model._resolve_guidance_pair(guidance=None, guidance_2=_GUIDANCE_2_UNSET)

    assert guidance == 4.0
    assert guidance_2 == 3.0


def test_wan_a14b_explicit_guidance_without_guidance_2_follows_guidance():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_t2v_a14b()

    guidance, guidance_2 = model._resolve_guidance_pair(guidance=4.5, guidance_2=_GUIDANCE_2_UNSET)

    assert guidance == 4.5
    assert guidance_2 == 4.5


def test_wan_ti2v_5b_default_guidance_has_no_low_noise_stage():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_ti2v_5b()

    guidance, guidance_2 = model._resolve_guidance_pair(guidance=None, guidance_2=_GUIDANCE_2_UNSET)

    assert guidance == 5.0
    assert guidance_2 is None


def test_wan_tensor_health_check_interval_validation():
    assert Wan2_2_TI2V._validate_tensor_health_check_interval(1) == 1
    assert Wan2_2_TI2V._validate_tensor_health_check_interval(None) is None
    with pytest.raises(ValueError, match="tensor_health_check_interval"):
        Wan2_2_TI2V._validate_tensor_health_check_interval(0)
    with pytest.raises(ValueError, match="tensor_health_check_interval"):
        Wan2_2_TI2V._validate_tensor_health_check_interval(-1)


def test_wan_generate_can_request_transformer_block_cache_clearing(monkeypatch):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)

    model.generate_video(
        seed=1,
        prompt="a slow wave",
        width=64,
        height=64,
        num_frames=1,
        num_inference_steps=2,
        guidance=1,
        guidance_2=1,
        clear_cache_each_transformer_block=True,
    )

    calls = model.transformer.calls + model.transformer_2.calls
    assert calls
    assert all(call["clear_cache_each_block"] is True for call in calls)


def test_wan_supports_video_to_video_is_config_gated():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_ti2v_5b()

    assert model._supports_video_to_video() is False

    model.model_config = ModelConfig.wan2_2_t2v_a14b()

    assert model._supports_video_to_video() is True


def test_wan_video_to_video_uses_scalar_timesteps_when_route_enabled(monkeypatch):
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_ti2v_5b()
    model.bits = None
    model.vae = SimpleNamespace(
        z_dim=48,
        temporal_scale=4,
        spatial_scale=8,
        decode_normalized_latents=lambda latents, **kwargs: mx.zeros((1, 3, 1, 8, 8), dtype=mx.float32),
    )
    model.transformer = _FakeWanTransformer(output=mx.zeros((1, 48, 1, 8, 8), dtype=mx.float32))
    model.transformer.in_channels = 48
    model.transformer.out_channels = 48
    model.transformer_2 = None

    class FakeScheduler:
        num_train_timesteps = 1000

        def __init__(self, flow_shift):
            self.flow_shift = flow_shift
            self.timesteps = mx.array([], dtype=mx.int64)

        def set_timesteps(self, num_inference_steps):
            del num_inference_steps
            self.timesteps = mx.array([900, 875], dtype=mx.int64)

        def set_begin_index(self, begin_index=0):
            self.begin_index = begin_index

        def step(self, model_output, timestep, sample, return_dict):
            del model_output, timestep, return_dict
            return (sample,)

    monkeypatch.setattr(
        "mflux.models.wan.variants.wan2_2_ti2v.WanUniPCMultistepScheduler",
        FakeScheduler,
    )
    monkeypatch.setattr(
        "mflux.models.wan.variants.wan2_2_ti2v.WanEulerScheduler",
        FakeScheduler,
    )
    monkeypatch.setattr(
        "mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(model, "encode_prompt", lambda **kwargs: (mx.zeros((1, 1, 4096)), None))
    monkeypatch.setattr(
        model,
        "_prepare_video_to_video_latents",
        lambda **kwargs: (
            mx.zeros((1, 48, 1, 8, 8), dtype=mx.float32),
            mx.zeros((1, 48, 1, 8, 8), dtype=mx.float32),
            mx.zeros((1, 48, 1, 8, 8), dtype=mx.float32),
            {},
        ),
    )
    monkeypatch.setattr(
        model,
        "_resolve_video_spatial_size",
        lambda **kwargs: (kwargs["height"], kwargs["width"], {}),
    )
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.gc.collect", lambda: None)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.mx.synchronize", lambda: None)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.mx.clear_cache", lambda: None)
    monkeypatch.setattr(model, "_supports_video_to_video", lambda: True)

    model.generate_video(
        seed=1,
        prompt="replace the ship hull",
        width=64,
        height=64,
        num_frames=1,
        num_inference_steps=2,
        guidance=1,
        video_path="input.mp4",
        video_strength=0.8,
    )

    assert model.transformer.calls
    assert model.transformer.calls[0]["timestep"].shape == (1,)
    assert np.array(model.transformer.calls[0]["timestep"]).tolist() == [875.0]


def test_wan_v2v_metadata_records_requested_and_effective_steps(monkeypatch):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model, patch_to_video=False)
    observed = {}

    def to_video(**kwargs):
        observed["to_video"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video", to_video)
    monkeypatch.setattr(
        model,
        "_prepare_video_to_video_latents",
        lambda **kwargs: (
            mx.zeros((1, 16, 1, 8, 8), dtype=mx.float32),
            mx.zeros((1, 16, 1, 8, 8), dtype=mx.float32),
            mx.zeros((1, 16, 1, 8, 8), dtype=mx.float32),
            {"source_video_frame_count": 30, "source_video_duration_seconds": 3.0, "source_video_fps": 10.0},
        ),
    )

    model.generate_video(
        seed=1,
        prompt="replace the ship hull",
        width=64,
        height=64,
        num_frames=1,
        num_inference_steps=2,
        guidance=1,
        guidance_2=1,
        video_path="input.mp4",
        video_strength=0.5,
    )

    metadata = observed["to_video"]
    # `steps` must stay the requested count so --config-from-metadata replays the same schedule.
    assert metadata["steps"] == 2
    assert metadata["extra_metadata"]["effective_steps"] == 1
    assert metadata["extra_metadata"]["video_strength"] == 0.5
    assert metadata["extra_metadata"]["high_noise_stage_skipped"] is False
    assert metadata["extra_metadata"]["source_video_frame_count"] == 30
    assert metadata["extra_metadata"]["source_video_fps"] == 10.0


def _run_masked_v2v_with_real_scheduler(monkeypatch, tmp_path, *, mask_color: int | None, seed: int = 1):
    model = _fake_t2v_a14b_model()
    source_latents = mx.arange(16 * 8 * 8, dtype=mx.float32).reshape(1, 16, 1, 8, 8) / 100.0
    noise = mx.ones((1, 16, 1, 8, 8), dtype=mx.float32) * 0.5
    warm_start = 0.9 * noise + 0.1 * source_latents
    observed = {}

    def decode(latents, **kwargs):
        observed["decode_latents"] = latents
        return mx.zeros((1, 3, 1, 8, 8), dtype=mx.float32)

    model.vae = SimpleNamespace(
        z_dim=16,
        temporal_scale=4,
        spatial_scale=8,
        decode_normalized_latents=decode,
    )
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(model, "encode_prompt", lambda **kwargs: (mx.zeros((1, 1, 4096)), None))
    monkeypatch.setattr(
        model,
        "_prepare_video_to_video_latents",
        lambda **kwargs: (warm_start, source_latents, noise, {}),
    )
    monkeypatch.setattr(
        model,
        "_resolve_video_spatial_size",
        lambda **kwargs: (kwargs["height"], kwargs["width"], {}),
    )
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.gc.collect", lambda: None)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.mx.synchronize", lambda: None)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.mx.clear_cache", lambda: None)

    mask_path = None
    if mask_color is not None:
        mask_file = tmp_path / f"mask_{mask_color}.png"
        Image.new("L", (64, 64), mask_color).save(mask_file)
        mask_path = str(mask_file)

    model.generate_video(
        seed=seed,
        prompt="replace the speaker",
        width=64,
        height=64,
        num_frames=1,
        num_inference_steps=4,
        guidance=1,
        guidance_2=1,
        video_path="input.mp4",
        video_strength=0.75,
        video_mask_path=mask_path,
    )
    return observed["decode_latents"], source_latents


def test_wan_masked_v2v_all_black_mask_returns_exact_source_roundtrip(monkeypatch, tmp_path, capsys):
    decode_latents, source_latents = _run_masked_v2v_with_real_scheduler(monkeypatch, tmp_path, mask_color=0)

    expected = source_latents.astype(ModelConfig.precision).astype(mx.float32)
    assert np.array_equal(np.array(decode_latents.astype(mx.float32)), np.array(expected))
    assert "no editable region" in capsys.readouterr().out


def test_wan_masked_v2v_all_white_mask_matches_plain_v2v(monkeypatch, tmp_path, capsys):
    masked, _ = _run_masked_v2v_with_real_scheduler(monkeypatch, tmp_path, mask_color=255)
    plain, _ = _run_masked_v2v_with_real_scheduler(monkeypatch, tmp_path, mask_color=None)

    masked_np = np.array(masked.astype(mx.float32))
    plain_np = np.array(plain.astype(mx.float32))
    assert np.array_equal(masked_np, plain_np)
    assert not np.array_equal(masked_np, np.zeros_like(masked_np))
    assert "equivalent to plain video-to-video" in capsys.readouterr().out


def test_wan_masked_v2v_partial_mask_locks_background_and_edits_foreground(monkeypatch, tmp_path):
    mask_file = tmp_path / "half.png"
    half = Image.new("L", (64, 64), 0)
    for x in range(32, 64):
        for y in range(64):
            half.putpixel((x, y), 255)
    half.save(mask_file)

    model = _fake_t2v_a14b_model()
    source_latents = mx.arange(16 * 8 * 8, dtype=mx.float32).reshape(1, 16, 1, 8, 8) / 100.0
    noise = mx.ones((1, 16, 1, 8, 8), dtype=mx.float32) * 0.5
    observed = {}

    def decode(latents, **kwargs):
        observed["decode_latents"] = latents
        return mx.zeros((1, 3, 1, 8, 8), dtype=mx.float32)

    model.vae = SimpleNamespace(z_dim=16, temporal_scale=4, spatial_scale=8, decode_normalized_latents=decode)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(model, "encode_prompt", lambda **kwargs: (mx.zeros((1, 1, 4096)), None))
    monkeypatch.setattr(
        model,
        "_prepare_video_to_video_latents",
        lambda **kwargs: (0.9 * noise + 0.1 * source_latents, source_latents, noise, {}),
    )
    monkeypatch.setattr(model, "_resolve_video_spatial_size", lambda **kwargs: (kwargs["height"], kwargs["width"], {}))
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.gc.collect", lambda: None)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.mx.synchronize", lambda: None)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.mx.clear_cache", lambda: None)

    model.generate_video(
        seed=1,
        prompt="replace the right half",
        width=64,
        height=64,
        num_frames=1,
        num_inference_steps=4,
        guidance=1,
        guidance_2=1,
        video_path="input.mp4",
        video_strength=0.75,
        video_mask_path=str(mask_file),
    )

    result = np.array(observed["decode_latents"].astype(mx.float32))
    expected_background = np.array(source_latents.astype(ModelConfig.precision).astype(mx.float32))
    # Preserved (black, left) columns are exactly the source; edited (white, right) columns differ.
    assert np.array_equal(result[..., :4], expected_background[..., :4])
    assert not np.array_equal(result[..., 4:], expected_background[..., 4:])


def test_wan_prepare_video_mask_polarity_shape_and_binarization(tmp_path):
    mask_image = Image.new("L", (64, 64), 0)
    for x in range(32, 64):
        for y in range(64):
            mask_image.putpixel((x, y), 255)
    mask_file = tmp_path / "half_mask.png"
    mask_image.save(mask_file)

    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.vae = SimpleNamespace(spatial_scale=8)

    mask = model._prepare_video_mask(mask_file, height=64, width=64)

    assert mask.shape == (1, 1, 1, 8, 8)
    values = np.array(mask)[0, 0, 0]
    assert np.array_equal(values[:, :4], np.zeros((8, 4), dtype=np.float32))
    assert np.array_equal(values[:, 4:], np.ones((8, 4), dtype=np.float32))


def test_wan_composite_masked_video_state_locks_scheduler_state():
    sigmas = mx.array([0.9, 0.7, 0.4, 0.0], dtype=mx.float32)
    source = mx.ones((1, 2, 1, 2, 2), dtype=mx.float32) * 3.0
    noise = mx.ones((1, 2, 1, 2, 2), dtype=mx.float32) * -1.0
    latents = mx.zeros((1, 2, 1, 2, 2), dtype=mx.float32)
    last_sample = mx.zeros((1, 2, 1, 2, 2), dtype=mx.float32)
    model_outputs = [None, mx.zeros((1, 2, 1, 2, 2), dtype=mx.float32)]
    scheduler = SimpleNamespace(sigmas=sigmas, step_index=2, last_sample=last_sample, model_outputs=model_outputs)
    mask = mx.zeros((1, 1, 1, 2, 2), dtype=mx.float32)  # preserve everywhere

    result = Wan2_2_TI2V._composite_masked_video_state(
        scheduler=scheduler,
        latents=latents,
        video_mask=mask,
        source_latents=source,
        noise=noise,
    )

    # Returned latents locked at the current level: sigma=0.4 -> 0.4*(-1) + 0.6*3 = 1.4
    np.testing.assert_allclose(np.array(result), np.full((1, 2, 1, 2, 2), 1.4, dtype=np.float32), rtol=1e-6)
    # Corrector anchor locked one level back: sigma=0.7 -> 0.7*(-1) + 0.3*3 = 0.2
    np.testing.assert_allclose(
        np.array(scheduler.last_sample), np.full((1, 2, 1, 2, 2), 0.2, dtype=np.float32), rtol=1e-6, atol=1e-6
    )
    # x0 history locked to the clean source.
    np.testing.assert_allclose(
        np.array(scheduler.model_outputs[-1]), np.full((1, 2, 1, 2, 2), 3.0, dtype=np.float32), rtol=1e-6
    )


def test_wan_video_mask_without_video_path_raises(monkeypatch, tmp_path):
    mask_file = tmp_path / "mask.png"
    Image.new("L", (64, 64), 255).save(mask_file)
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="video_mask_path requires video_path"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=1,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            video_mask_path=str(mask_file),
        )


def test_wan_video_strength_without_video_path_raises(monkeypatch):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="video_strength requires video_path"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=1,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            video_strength=0.5,
        )


def test_wan_cache_path_key_tracks_file_identity(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"first")
    first_key = Wan2_2_TI2V._cache_path_key(source)

    source.write_bytes(b"rewritten-content")
    os.utime(source, ns=(1, 1))
    second_key = Wan2_2_TI2V._cache_path_key(source)

    assert first_key != second_key
    assert first_key[0] == second_key[0]
    assert Wan2_2_TI2V._cache_path_key(None) is None
    assert Wan2_2_TI2V._cache_path_key(tmp_path / "missing.mp4") == (str(tmp_path / "missing.mp4"), 0, 0)


def test_wan_v2v_warns_on_source_aspect_stretch_resampling_and_truncation(capsys):
    source_metadata = {
        "source_width": 320,
        "source_height": 240,
        "source_video_frame_count": 90,
        "source_video_duration_seconds": 3.0,
        "source_video_fps": 30.0,
        "source_video_audio_present": True,
        "source_video_resampled": True,
    }

    Wan2_2_TI2V._warn_video_to_video_source_handling(
        source_metadata=source_metadata, num_frames=17, height=256, width=448, fps=16
    )

    output = capsys.readouterr().out
    assert "stretches source frames" in output
    assert "resamples the source from 30 fps to 16 fps" in output
    assert "keeps real-time speed" in output
    assert "uses the first 1.06s of the 3.00s source" in output

    upsampled_source = {
        "source_width": 448,
        "source_height": 256,
        "source_video_frame_count": 11,
        "source_video_duration_seconds": 1.1,
        "source_video_fps": 10.0,
        "source_video_audio_present": False,
        "source_video_resampled": True,
    }
    Wan2_2_TI2V._warn_video_to_video_source_handling(
        source_metadata=upsampled_source, num_frames=17, height=256, width=448, fps=24
    )
    upsample_output = capsys.readouterr().out
    assert "up to 24 fps by duplicating frames" in upsample_output

    matching = {
        "source_width": 448,
        "source_height": 256,
        "source_video_frame_count": 17,
        "source_video_duration_seconds": 1.7,
        "source_video_fps": 10.0,
        "source_video_audio_present": False,
        "source_video_resampled": False,
    }
    Wan2_2_TI2V._warn_video_to_video_source_handling(
        source_metadata=matching, num_frames=17, height=256, width=448, fps=10
    )
    assert capsys.readouterr().out == ""


def test_wan_video_to_video_rejects_euler_solver(monkeypatch):
    model = _fake_t2v_a14b_model()
    monkeypatch.setattr(
        model,
        "_resolve_video_spatial_size",
        lambda **kwargs: (kwargs["height"], kwargs["width"], {}),
    )

    with pytest.raises(ValueError, match="requires solver='unipc'"):
        model.generate_video(
            seed=1,
            prompt="replace the ship hull",
            width=64,
            height=64,
            num_frames=1,
            num_inference_steps=2,
            guidance=1,
            solver="euler",
            video_path="input.mp4",
            video_strength=0.8,
        )


def test_wan_generate_materializes_cfg_predictions_without_tensor_health(monkeypatch):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)
    model.encode_prompt = lambda **kwargs: (
        mx.zeros((1, 1, 4096), dtype=mx.float32),
        mx.zeros((1, 1, 4096), dtype=mx.float32),
    )
    materialized = []
    model._materialize_denoise_prediction = lambda prediction, clear_cache: materialized.append(clear_cache)

    model.generate_video(
        seed=1,
        prompt="a slow wave",
        width=64,
        height=64,
        num_frames=1,
        num_inference_steps=2,
        guidance=5,
        clear_cache_each_transformer_block=True,
        tensor_health_check_interval=None,
    )

    assert materialized == [True, True, True, True, True, True]


def test_wan_generate_releases_denoisers_before_decode(monkeypatch):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)
    observed = {}

    def decode(latents, *, clear_cache_each_slice=False):
        observed["transformer"] = model.transformer
        observed["transformer_2"] = model.transformer_2
        observed["clear_cache_each_slice"] = clear_cache_each_slice
        yield mx.zeros((1, 3, 1, 8, 8), dtype=mx.float32)

    model.vae.iter_decode_normalized_latent_slices = decode

    video = model.generate_video(
        seed=1,
        prompt="a slow wave",
        width=64,
        height=64,
        num_frames=1,
        num_inference_steps=2,
        guidance=1,
        release_denoisers_before_decode=True,
    )
    video.first_frame()

    assert observed == {
        "transformer": None,
        "transformer_2": None,
        "clear_cache_each_slice": True,
    }


@pytest.mark.parametrize("guidance", [np.nan, np.inf, -np.inf])
def test_wan_generate_rejects_non_finite_guidance(monkeypatch, guidance):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="Wan guidance must be finite"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=1,
            num_inference_steps=2,
            guidance=guidance,
            guidance_2=1,
        )


@pytest.mark.parametrize("guidance_2", [np.nan, np.inf, -np.inf])
def test_wan_generate_rejects_non_finite_guidance_2(monkeypatch, guidance_2):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="Wan guidance_2 must be finite"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=1,
            num_inference_steps=2,
            guidance=1,
            guidance_2=guidance_2,
        )


class _FakeWanTransformer:
    patch_size = (1, 2, 2)
    in_channels = 16
    out_channels = 16

    def __init__(self, output=None):
        self._output = output
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self._output is not None:
            return self._output
        return mx.zeros((1, 16, 1, 8, 8), dtype=mx.float32)


def _fake_t2v_a14b_model():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_t2v_a14b()
    model.bits = None
    model.vae = SimpleNamespace(
        z_dim=16,
        temporal_scale=4,
        spatial_scale=8,
        decode_normalized_latents=lambda latents, **kwargs: mx.zeros((1, 3, 1, 8, 8), dtype=mx.float32),
    )
    model.transformer = _FakeWanTransformer()
    model.transformer_2 = _FakeWanTransformer()
    return model


def _patch_fake_wan_generation(monkeypatch, model, scheduler_output=None, patch_to_video=True):
    calls = {"scheduler_steps": 0, "to_video": 0}

    class FakeScheduler:
        num_train_timesteps = 1000

        def __init__(self, flow_shift):
            calls["scheduler_flow_shift"] = flow_shift
            self.flow_shift = flow_shift
            self.timesteps = mx.array([], dtype=mx.int64)

        def set_timesteps(self, num_inference_steps):
            self.timesteps = mx.array([900, 875], dtype=mx.int64)

        def step(self, model_output, timestep, sample, return_dict):
            calls["scheduler_steps"] += 1
            return (scheduler_output if scheduler_output is not None else sample,)

    def fake_to_video(**kwargs):
        calls["to_video"] += 1
        return SimpleNamespace()

    monkeypatch.setattr(
        "mflux.models.wan.variants.wan2_2_ti2v.WanUniPCMultistepScheduler",
        FakeScheduler,
    )
    monkeypatch.setattr(
        "mflux.models.wan.variants.wan2_2_ti2v.WanEulerScheduler",
        FakeScheduler,
    )
    if patch_to_video:
        monkeypatch.setattr(
            "mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video",
            fake_to_video,
        )
    monkeypatch.setattr(model, "encode_prompt", lambda **kwargs: (mx.zeros((1, 1, 4096)), None))
    monkeypatch.setattr(
        model,
        "prepare_latents",
        lambda **kwargs: mx.zeros((1, 16, 1, 8, 8), dtype=mx.float32),
    )
    monkeypatch.setattr(
        model,
        "_resolve_video_spatial_size",
        lambda **kwargs: (kwargs["height"], kwargs["width"], {}),
    )
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.gc.collect", lambda: None)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.mx.synchronize", lambda: None)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.mx.clear_cache", lambda: None)
    return calls


def test_wan_guidance_2_requires_two_transformer_boundary():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_ti2v_5b()

    with pytest.raises(ValueError, match="guidance_2 is only supported"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=128,
            height=128,
            num_frames=5,
            num_inference_steps=1,
            guidance_2=2.0,
        )


def test_wan_python_api_infers_a14b_config_from_model_path(monkeypatch):
    observed = {}

    def fake_init(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(WanInitializer, "init", fake_init)

    Wan2_2_TI2V(model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers")

    assert observed["model_config"] is ModelConfig.wan2_2_t2v_a14b()
    assert observed["model_path"] == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


def test_wan_runtime_contract_rejects_mismatched_t2v_channels():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_t2v_a14b()
    model.transformer = SimpleNamespace(in_channels=16, out_channels=16)
    model.transformer_2 = SimpleNamespace(in_channels=16, out_channels=16)
    model.vae = SimpleNamespace(z_dim=48)

    with pytest.raises(ValueError, match="Wan runtime config mismatch"):
        model._validate_runtime_contract(is_image_to_video=False)


def test_wan_runtime_contract_rejects_consistent_ti2v_modules_under_a14b_config():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_t2v_a14b()
    model.transformer = SimpleNamespace(in_channels=48, out_channels=48)
    model.transformer_2 = None
    model.vae = SimpleNamespace(z_dim=48)

    with pytest.raises(ValueError, match="transformer.in_channels"):
        model._validate_runtime_contract(is_image_to_video=False)


def test_wan_source_config_rejects_a14b_checkpoint_with_ti2v_runtime(tmp_path):
    _write_wan_source_configs(
        tmp_path,
        has_transformer_2=True,
        boundary_ratio=0.875,
        transformer_in_channels=16,
        transformer_out_channels=16,
        transformer_layers=40,
        transformer_heads=40,
        transformer_ffn_dim=13824,
        vae_z_dim=16,
        vae_base_dim=96,
    )

    with pytest.raises(ValueError, match="Wan source/config mismatch"):
        WanInitializer._validate_source_config(tmp_path, ModelConfig.wan2_2_ti2v_5b())


def test_wan_source_config_accepts_matching_a14b_checkpoint(tmp_path):
    _write_wan_source_configs(
        tmp_path,
        has_transformer_2=True,
        boundary_ratio=0.875,
        transformer_in_channels=16,
        transformer_out_channels=16,
        transformer_layers=40,
        transformer_heads=40,
        transformer_ffn_dim=13824,
        vae_z_dim=16,
        vae_base_dim=96,
    )

    WanInitializer._validate_source_config(tmp_path, ModelConfig.wan2_2_t2v_a14b())


def test_wan_transformer_rejects_wrong_input_channels_before_conv():
    transformer = WanTransformer(
        in_channels=16,
        out_channels=16,
        num_layers=0,
        num_attention_heads=1,
        attention_head_dim=8,
        text_dim=16,
        ffn_dim=16,
    )

    with pytest.raises(ValueError, match="input channel mismatch"):
        transformer(
            hidden_states=mx.zeros((1, 48, 1, 4, 4)),
            timestep=mx.array([1], dtype=mx.float32),
            encoder_hidden_states=mx.zeros((1, 1, 16)),
        )


def test_wan_runtime_contract_accepts_a14b_t2v_channels():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_t2v_a14b()
    model.transformer = SimpleNamespace(in_channels=16, out_channels=16)
    model.transformer_2 = SimpleNamespace(in_channels=16, out_channels=16)
    model.vae = SimpleNamespace(z_dim=16)

    model._validate_runtime_contract(is_image_to_video=False)


def test_wan_runtime_contract_requires_image_for_i2v_model():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer = SimpleNamespace(in_channels=36)
    model.vae = SimpleNamespace(z_dim=16)

    with pytest.raises(ValueError, match="requires image-to-video input"):
        model._validate_runtime_contract(is_image_to_video=False)


def test_wan_a14b_prepare_latents_uses_vae_channels_not_i2v_input_channels():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.transformer = SimpleNamespace(in_channels=36)
    model.vae = SimpleNamespace(z_dim=16, temporal_scale=4, spatial_scale=8)

    latents = model.prepare_latents(seed=1, batch_size=1, height=64, width=80, num_frames=9)
    mx.eval(latents)

    assert latents.shape == (1, 16, 3, 8, 10)


def test_wan_spatial_size_rounds_up_to_patch_multiple():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.transformer = SimpleNamespace(patch_size=(1, 4, 4))
    model.vae = SimpleNamespace(spatial_scale=8)

    assert model._validated_spatial_size(height=240, width=432) == (256, 448)
    assert model._validated_spatial_size(height=256, width=448) == (256, 448)


def test_wan_i2v_spatial_size_preserves_source_ratio_for_ti2v_multiple():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.transformer = SimpleNamespace(patch_size=(1, 4, 4))
    model.vae = SimpleNamespace(spatial_scale=8)

    height, width = model._validated_source_aspect_spatial_size(
        height=240,
        width=432,
        source_height=240,
        source_width=320,
        source_label="image",
    )

    assert (height, width) == (288, 384)
    assert width / height == pytest.approx(320 / 240)


def test_wan_i2v_spatial_size_preserves_source_ratio_for_a14b_multiple():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.transformer = SimpleNamespace(patch_size=(1, 2, 2))
    model.vae = SimpleNamespace(spatial_scale=8)

    height, width = model._validated_source_aspect_spatial_size(
        height=288,
        width=512,
        source_height=240,
        source_width=320,
        source_label="image",
    )

    assert (height, width) == (336, 448)
    assert width / height == pytest.approx(320 / 240)


def test_wan_i2v_resolved_spatial_size_reads_source_image(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (320, 240), "white").save(image_path)
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.transformer = SimpleNamespace(patch_size=(1, 4, 4))
    model.vae = SimpleNamespace(spatial_scale=8)

    assert model._resolved_spatial_size(height=240, width=432, image_path=image_path) == (288, 384)


def test_wan_v2v_resolved_spatial_size_uses_requested_canvas(monkeypatch):
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.transformer = SimpleNamespace(patch_size=(1, 4, 4))
    model.vae = SimpleNamespace(spatial_scale=8)

    monkeypatch.setattr(
        "mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.inspect_video",
        lambda path: SimpleNamespace(source_width=320, source_height=240),
    )

    assert model._resolve_video_spatial_size(height=240, width=432, image_path=None, video_path="input.mp4") == (
        256,
        448,
        {"source_width": 320, "source_height": 240, "requested_width": 432, "requested_height": 240},
    )


def test_wan_generate_i2v_uses_resolved_source_ratio_size_for_a14b_condition(monkeypatch):
    model = _fake_t2v_a14b_model()
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer.in_channels = 36
    model.transformer_2.in_channels = 36
    _patch_fake_wan_generation(monkeypatch, model, patch_to_video=False)
    observed = {}

    def resolved_spatial_size(**kwargs):
        observed["resolve"] = kwargs
        return 336, 448, {"source_width": 320, "source_height": 240, "requested_width": 512, "requested_height": 288}

    def prepare_latents(**kwargs):
        observed["prepare"] = kwargs
        return mx.zeros((1, 16, 1, 8, 8), dtype=mx.float32)

    def encode_video_condition(**kwargs):
        observed["condition"] = kwargs
        return mx.zeros((1, 20, 1, 8, 8), dtype=mx.float32)

    def to_video(**kwargs):
        observed["to_video"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(model, "_resolve_video_spatial_size", resolved_spatial_size)
    monkeypatch.setattr(model, "prepare_latents", prepare_latents)
    monkeypatch.setattr(model, "_encode_video_condition", encode_video_condition)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video", to_video)

    model.generate_video(
        seed=1,
        prompt="a slow wave",
        width=512,
        height=288,
        num_frames=1,
        num_inference_steps=2,
        guidance=1,
        guidance_2=1,
        image_path="input.png",
    )

    assert observed["resolve"] == {"height": 288, "width": 512, "image_path": "input.png", "video_path": None}
    assert observed["prepare"]["height"] == 336
    assert observed["prepare"]["width"] == 448
    assert observed["condition"]["height"] == 336
    assert observed["condition"]["width"] == 448
    assert observed["to_video"]["task"] == "image-to-video"
    assert observed["to_video"]["source_width"] == 320
    assert observed["to_video"]["source_height"] == 240
    assert observed["to_video"]["requested_width"] == 512
    assert observed["to_video"]["requested_height"] == 288


def test_wan_generate_i2v_uses_resolved_source_ratio_size_for_ti2v_condition(monkeypatch):
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_ti2v_5b()
    model.bits = None
    model.vae = SimpleNamespace(
        z_dim=48,
        temporal_scale=4,
        spatial_scale=8,
        decode_normalized_latents=lambda latents, **kwargs: mx.zeros((1, 3, 1, 8, 8), dtype=mx.float32),
    )
    model.transformer = _FakeWanTransformer(output=mx.zeros((1, 48, 1, 8, 8), dtype=mx.float32))
    model.transformer.in_channels = 48
    model.transformer.out_channels = 48
    model.transformer_2 = None
    _patch_fake_wan_generation(monkeypatch, model, patch_to_video=False)
    observed = {}

    def resolved_spatial_size(**kwargs):
        observed["resolve"] = kwargs
        return 288, 384, {"source_width": 320, "source_height": 240, "requested_width": 432, "requested_height": 240}

    def prepare_latents(**kwargs):
        observed["prepare"] = kwargs
        return mx.zeros((1, 48, 1, 8, 8), dtype=mx.float32)

    def encode_first_frame_condition(**kwargs):
        observed["condition"] = kwargs
        return mx.zeros((1, 48, 1, 8, 8), dtype=mx.float32)

    def to_video(**kwargs):
        observed["to_video"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(model, "_resolve_video_spatial_size", resolved_spatial_size)
    monkeypatch.setattr(model, "prepare_latents", prepare_latents)
    monkeypatch.setattr(model, "_encode_first_frame_condition", encode_first_frame_condition)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video", to_video)

    model.generate_video(
        seed=1,
        prompt="a slow wave",
        width=432,
        height=240,
        num_frames=1,
        num_inference_steps=2,
        guidance=1,
        image_path="input.png",
    )

    assert observed["resolve"] == {"height": 240, "width": 432, "image_path": "input.png", "video_path": None}
    assert observed["prepare"]["height"] == 288
    assert observed["prepare"]["width"] == 384
    assert observed["condition"]["height"] == 288
    assert observed["condition"]["width"] == 384
    assert observed["to_video"]["task"] == "image-to-video"
    assert observed["to_video"]["source_width"] == 320
    assert observed["to_video"]["source_height"] == 240
    assert observed["to_video"]["requested_width"] == 432
    assert observed["to_video"]["requested_height"] == 240


def test_wan_video_to_video_source_conditioning_uses_float32(monkeypatch):
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    observed = {}

    def encode_normalized(video):
        observed["dtype"] = video.dtype
        return mx.zeros((1, 16, 1, 8, 8), dtype=mx.float32)

    model.vae = SimpleNamespace(encode_normalized=encode_normalized)
    clip = SimpleNamespace(
        clip_frame_count=1,
        frames=[Image.new("RGB", (64, 64), "white")],
        source_width=64,
        source_height=64,
        source_frame_count=1,
        source_duration_seconds=0.1,
        fps=10.0,
        audio_present=False,
        sampled_fps=None,
    )
    monkeypatch.setattr(
        "mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.read_video_clip",
        lambda video_path, max_frames, target_fps=None: clip,
    )

    latents, metadata = model._load_video_to_video_latents(
        video_path="input.mp4", height=64, width=64, num_frames=1, fps=10
    )

    assert observed["dtype"] == mx.float32
    assert latents.dtype == mx.float32
    assert metadata["source_width"] == 64
    assert metadata["source_video_resampled"] is False


def test_wan_explicit_empty_negative_prompt_disables_default():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_ti2v_5b()

    assert "低质量" in model._resolve_negative_prompt(None)
    assert model._resolve_negative_prompt("") == ""
    assert model._resolve_negative_prompt("no blur") == "no blur"


def test_wan_generate_passes_explicit_flow_shift_to_scheduler_and_metadata(monkeypatch):
    model = _fake_t2v_a14b_model()
    calls = _patch_fake_wan_generation(monkeypatch, model)
    observed = {}

    def to_video(**kwargs):
        observed["to_video"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video", to_video)

    model.generate_video(
        seed=1,
        prompt="a slow misty lake",
        width=64,
        height=64,
        num_frames=1,
        num_inference_steps=2,
        guidance=1,
        flow_shift=2.5,
    )

    assert calls["scheduler_flow_shift"] == 2.5
    assert observed["to_video"]["flow_shift"] == 2.5
    assert observed["to_video"]["solver"] == "unipc"


def test_wan_generate_uses_model_default_flow_shift(monkeypatch):
    model = _fake_t2v_a14b_model()
    calls = _patch_fake_wan_generation(monkeypatch, model)

    model.generate_video(
        seed=1,
        prompt="a slow misty lake",
        width=64,
        height=64,
        num_frames=1,
        num_inference_steps=2,
        guidance=1,
    )

    assert calls["scheduler_flow_shift"] == 3.0


def test_wan_generate_can_select_euler_solver(monkeypatch):
    model = _fake_t2v_a14b_model()
    calls = _patch_fake_wan_generation(monkeypatch, model)
    observed = {}

    def to_video(**kwargs):
        observed["to_video"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video", to_video)

    model.generate_video(
        seed=1,
        prompt="a fast lift-off",
        width=64,
        height=64,
        num_frames=1,
        num_inference_steps=2,
        guidance=1,
        solver="euler",
    )

    assert calls["scheduler_flow_shift"] == 3.0
    assert observed["to_video"]["solver"] == "euler"


def test_wan_generate_rejects_invalid_flow_shift():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_ti2v_5b()

    with pytest.raises(ValueError, match="flow_shift"):
        model._resolve_flow_shift(0)


def test_wan_a14b_i2v_condition_uses_20_condition_channels(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (80, 64), "white").save(image_path)
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.vae = Wan2_2_VAE(**ModelConfig.wan2_2_i2v_a14b().transformer_overrides["vae_config"])

    condition = model._encode_video_condition(
        image_path=image_path,
        height=64,
        width=80,
        num_frames=9,
        batch_size=1,
    )
    mx.eval(condition)

    assert condition.shape == (1, 20, 3, 8, 10)


def _write_wan_source_configs(
    path,
    *,
    has_transformer_2: bool,
    boundary_ratio: float | None,
    transformer_in_channels: int,
    transformer_out_channels: int,
    transformer_layers: int,
    transformer_heads: int,
    transformer_ffn_dim: int,
    vae_z_dim: int,
    vae_base_dim: int,
) -> None:
    (path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "WanPipeline",
                "boundary_ratio": boundary_ratio,
                "transformer_2": ["diffusers", "WanTransformer3DModel"] if has_transformer_2 else [None, None],
            }
        )
    )
    transformer_config = {
        "_class_name": "WanTransformer3DModel",
        "in_channels": transformer_in_channels,
        "out_channels": transformer_out_channels,
        "num_layers": transformer_layers,
        "num_attention_heads": transformer_heads,
        "ffn_dim": transformer_ffn_dim,
        "patch_size": [1, 2, 2],
    }
    for component in ("transformer", "transformer_2"):
        if component == "transformer_2" and not has_transformer_2:
            continue
        component_path = path / component
        component_path.mkdir()
        (component_path / "config.json").write_text(json.dumps(transformer_config))
    vae_path = path / "vae"
    vae_path.mkdir()
    (vae_path / "config.json").write_text(
        json.dumps({"_class_name": "AutoencoderKLWan", "z_dim": vae_z_dim, "base_dim": vae_base_dim})
    )
