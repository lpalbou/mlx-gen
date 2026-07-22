from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.wan.model.wan_transformer import WanTransformer
from mflux.models.wan.wan_initializer import WanInitializer
from mflux.models.wan.weights.wan_lora_mapping import WanLoRAMapping
from mflux.models.wan.weights.wan_weight_definition import WanWeightDefinition


class _TinyWanAttention(nn.Module):
    def __init__(self, with_image_proj: bool = False):
        super().__init__()
        self.to_q = nn.Linear(4, 3, bias=False)
        self.to_k = nn.Linear(4, 3, bias=False)
        self.to_v = nn.Linear(4, 3, bias=False)
        self.to_out = [nn.Linear(3, 4, bias=False)]
        self.add_k_proj = nn.Linear(4, 3, bias=False) if with_image_proj else None
        self.add_v_proj = nn.Linear(4, 3, bias=False) if with_image_proj else None


class _TinyWanFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = [nn.Linear(4, 6, bias=False), nn.Linear(6, 4, bias=False)]


class _TinyWanBlock(nn.Module):
    def __init__(self, with_image_proj: bool = False):
        super().__init__()
        self.attn1 = _TinyWanAttention()
        self.attn2 = _TinyWanAttention(with_image_proj=with_image_proj)
        self.ffn = _TinyWanFFN()


class _TinyWanTransformer(nn.Module):
    def __init__(self, with_image_proj: bool = False):
        super().__init__()
        self.blocks = [_TinyWanBlock(with_image_proj=with_image_proj)]


@pytest.mark.fast
def test_wan_non_diffusers_lora_keys_apply_to_attention_and_ffn(tmp_path: Path):
    lora_path = tmp_path / "wan-ti2v-lora.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight": mx.zeros((2, 4)),
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight": mx.zeros((3, 2)),
            "diffusion_model.blocks.0.cross_attn.k.lora_A.weight": mx.zeros((2, 4)),
            "diffusion_model.blocks.0.cross_attn.k.lora_B.weight": mx.zeros((3, 2)),
            "diffusion_model.blocks.0.ffn.0.lora_A.weight": mx.zeros((2, 4)),
            "diffusion_model.blocks.0.ffn.0.lora_B.weight": mx.zeros((6, 2)),
            "diffusion_model.blocks.0.ffn.2.lora_A.weight": mx.zeros((2, 6)),
            "diffusion_model.blocks.0.ffn.2.lora_B.weight": mx.zeros((4, 2)),
        },
    )
    transformer = _TinyWanTransformer()

    result = LoRALoader.load_and_apply_lora_detailed(
        lora_mapping=WanLoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(lora_path)],
        lora_scales=[1.0],
        role="transformer",
    )

    assert isinstance(transformer.blocks[0].attn1.to_q, LoRALinear)
    assert isinstance(transformer.blocks[0].attn2.to_k, LoRALinear)
    assert isinstance(transformer.blocks[0].ffn.net[0], LoRALinear)
    assert isinstance(transformer.blocks[0].ffn.net[1], LoRALinear)
    assert result.reports[0].matched_key_count == 8
    assert result.reports[0].unmatched_key_count == 0


@pytest.mark.fast
def test_wan_musubi_lora_keys_apply_to_attention_and_ffn(tmp_path: Path):
    lora_path = tmp_path / "wan-musubi-lora.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "lora_unet_blocks_0_self_attn_q.lora_down.weight": mx.zeros((2, 4)),
            "lora_unet_blocks_0_self_attn_q.lora_up.weight": mx.zeros((3, 2)),
            "lora_unet_blocks_0_cross_attn_o.lora_down.weight": mx.zeros((2, 3)),
            "lora_unet_blocks_0_cross_attn_o.lora_up.weight": mx.zeros((4, 2)),
            "lora_unet_blocks_0_ffn_0.lora_down.weight": mx.zeros((2, 4)),
            "lora_unet_blocks_0_ffn_0.lora_up.weight": mx.zeros((6, 2)),
            "lora_unet_blocks_0_ffn_2.lora_down.weight": mx.zeros((2, 6)),
            "lora_unet_blocks_0_ffn_2.lora_up.weight": mx.zeros((4, 2)),
        },
    )
    transformer = _TinyWanTransformer()

    result = LoRALoader.load_and_apply_lora_detailed(
        lora_mapping=WanLoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(lora_path)],
        lora_scales=[0.8],
        role="transformer",
    )

    assert isinstance(transformer.blocks[0].attn1.to_q, LoRALinear)
    assert isinstance(transformer.blocks[0].attn2.to_out[0], LoRALinear)
    assert isinstance(transformer.blocks[0].ffn.net[0], LoRALinear)
    assert isinstance(transformer.blocks[0].ffn.net[1], LoRALinear)
    assert result.reports[0].matched_key_count == 8
    assert result.reports[0].unmatched_key_count == 0


@pytest.mark.fast
def test_wan_lora_refusion_on_fresh_weights_is_deterministic(tmp_path: Path):
    # Primary risk of the per-item A14B reload (0089 e4): re-applying the same
    # LoRA stack (including FusedLoRALinear layering and alpha scaling) on a
    # fresh module with identical base weights must reproduce the original
    # fused behavior exactly. Two files stack on the same layer to cover the
    # fusion-order path.
    mx.random.seed(11)
    first_lora = tmp_path / "first.safetensors"
    mx.save_safetensors(
        str(first_lora),
        {
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight": mx.random.normal((2, 4)),
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight": mx.random.normal((3, 2)),
            "diffusion_model.blocks.0.self_attn.q.alpha": mx.array([4.0]),
            "diffusion_model.blocks.0.ffn.0.lora_A.weight": mx.random.normal((2, 4)),
            "diffusion_model.blocks.0.ffn.0.lora_B.weight": mx.random.normal((6, 2)),
        },
    )
    second_lora = tmp_path / "second.safetensors"
    mx.save_safetensors(
        str(second_lora),
        {
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight": mx.random.normal((3, 4)),
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight": mx.random.normal((3, 3)),
        },
    )

    def fuse(transformer):
        for lora_path, scale in ((first_lora, 0.8), (second_lora, 0.5)):
            LoRALoader.load_and_apply_lora_detailed(
                lora_mapping=WanLoRAMapping.get_mapping(),
                transformer=transformer,
                lora_paths=[str(lora_path)],
                lora_scales=[scale],
                role="high_noise_transformer",
            )

    original = _TinyWanTransformer()
    base_params = original.parameters()  # snapshot BEFORE fusion replaces layers
    refused = _TinyWanTransformer()
    refused.update(base_params)  # identical base weights, fresh module
    base_only = _TinyWanTransformer()
    base_only.update(base_params)
    fuse(original)
    fuse(refused)

    probe = mx.random.normal((2, 4))
    for layer in (lambda t: t.blocks[0].attn1.to_q, lambda t: t.blocks[0].ffn.net[0]):
        original_output = np.array(layer(original)(probe))
        refused_output = np.array(layer(refused)(probe))
        base_output = np.array(layer(base_only)(probe))
        assert np.array_equal(original_output, refused_output)
        assert not np.array_equal(original_output, base_output)  # the LoRA does something


@pytest.mark.fast
def test_wan_high_noise_reload_reproduces_fused_weights_bitwise(tmp_path: Path, monkeypatch):
    # End-to-end reload path (0089 e4): WanInitializer.reload_high_noise_transformer
    # rebuilds a REAL (tiny) WanTransformer from the captured spec and re-fuses the
    # high-noise-role LoRAs. Its output must equal the originally fused expert
    # bitwise; the low-noise-role file must NOT be applied to it.
    overrides = {
        "in_channels": 16,
        "out_channels": 16,
        "num_layers": 1,
        "num_attention_heads": 1,
        "attention_head_dim": 8,
        "text_dim": 16,
        "ffn_dim": 16,
        "has_transformer_2": True,
    }
    model_config = SimpleNamespace(transformer_overrides=overrides)
    transformer_kwargs = WanInitializer._transformer_kwargs(model_config)
    mx.random.seed(7)
    base_params = WanTransformer(**transformer_kwargs).parameters()

    high_lora = tmp_path / "high-noise.safetensors"
    mx.random.seed(21)
    mx.save_safetensors(
        str(high_lora),
        {
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight": mx.random.normal((2, 8)),
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight": mx.random.normal((8, 2)),
            "diffusion_model.blocks.0.self_attn.q.alpha": mx.array([4.0]),
        },
    )
    low_lora = tmp_path / "low-noise.safetensors"
    mx.save_safetensors(
        str(low_lora),
        {
            "diffusion_model.blocks.0.self_attn.v.lora_A.weight": mx.random.normal((2, 8)),
            "diffusion_model.blocks.0.self_attn.v.lora_B.weight": mx.random.normal((8, 2)),
        },
    )

    def load_component(root_path=None, component=None, raw_weights_cache=None):
        del root_path, raw_weights_cache
        assert component.name == "transformer"  # the reload touches only the high expert
        return base_params, None, None

    monkeypatch.setattr(WeightLoader, "_load_component", load_component)

    # The originally fused expert, built exactly like WanInitializer.init does.
    original = WanTransformer(**transformer_kwargs)
    original.update(base_params, strict=False)
    LoRALoader.load_and_apply_lora_detailed(
        lora_mapping=WanLoRAMapping.get_mapping(),
        transformer=original,
        lora_paths=[str(high_lora)],
        lora_scales=[0.8],
        role="high_noise_transformer",
        state_dict_transform=WanInitializer._transform_wan_lora_state_dict,
    )

    model = SimpleNamespace(
        model_config=model_config,
        weight_definition=WanWeightDefinition.for_config(model_config),
        root_path=tmp_path,
        quantize_arg=None,
        bits=None,
        transformer=None,
        transformer_2=SimpleNamespace(),
        lora_paths=[str(high_lora), str(low_lora)],
        lora_scales=[0.8, 1.0],
        lora_target_roles=["high_noise_transformer", "low_noise_transformer"],
    )
    WanInitializer.reload_high_noise_transformer(model)

    probe = {
        "hidden_states": mx.random.normal((1, 16, 1, 4, 4)),
        "timestep": mx.array([500], dtype=mx.float32),
        "encoder_hidden_states": mx.random.normal((1, 2, 16)),
    }
    original_output = np.array(original(**probe))
    reloaded_output = np.array(model.transformer(**probe))
    base_only = WanTransformer(**transformer_kwargs)
    base_only.update(base_params, strict=False)
    base_output = np.array(base_only(**probe))

    assert np.array_equal(original_output, reloaded_output)
    assert not np.array_equal(base_output, reloaded_output)  # re-fusion actually applied
    assert isinstance(model.transformer.blocks[0].attn1.to_q, LoRALinear)
    assert not isinstance(model.transformer.blocks[0].attn1.to_v, LoRALinear)  # low-role file skipped


@pytest.mark.fast
def test_wan_high_noise_reload_rejects_quantization_drift(tmp_path: Path, monkeypatch):
    # If the on-disk checkpoint changes quantization between init and reload,
    # the reload must fail loudly instead of silently mixing precisions.
    overrides = {
        "in_channels": 16,
        "out_channels": 16,
        "num_layers": 0,
        "num_attention_heads": 1,
        "attention_head_dim": 8,
        "text_dim": 16,
        "ffn_dim": 16,
        "has_transformer_2": True,
    }
    model_config = SimpleNamespace(transformer_overrides=overrides)
    base_params = WanTransformer(**WanInitializer._transformer_kwargs(model_config)).parameters()

    def load_component(root_path=None, component=None, raw_weights_cache=None):
        del root_path, component, raw_weights_cache
        return base_params, None, None  # resolves to bits=None on reload

    monkeypatch.setattr(WeightLoader, "_load_component", load_component)
    model = SimpleNamespace(
        model_config=model_config,
        weight_definition=WanWeightDefinition.for_config(model_config),
        root_path=tmp_path,
        quantize_arg=None,
        bits=8,  # the process loaded a prequantized package
        transformer=None,
        transformer_2=SimpleNamespace(),
        lora_paths=[],
        lora_scales=[],
        lora_target_roles=[],
    )

    with pytest.raises(ValueError, match="reload resolved quantization"):
        WanInitializer.reload_high_noise_transformer(model)


@pytest.mark.fast
def test_wan_i2v_expands_t2v_lora_to_image_projection_layers(tmp_path: Path):
    lora_path = tmp_path / "wan-t2v-only-lora.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "diffusion_model.blocks.0.cross_attn.k.lora_down.weight": mx.zeros((2, 4)),
            "diffusion_model.blocks.0.cross_attn.k.lora_up.weight": mx.zeros((3, 2)),
        },
    )
    transformer = _TinyWanTransformer(with_image_proj=True)

    result = LoRALoader.load_and_apply_lora_detailed(
        lora_mapping=WanLoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(lora_path)],
        lora_scales=[1.0],
        role="transformer",
        state_dict_transform=WanInitializer._transform_wan_lora_state_dict,
    )

    assert isinstance(transformer.blocks[0].attn2.to_k, LoRALinear)
    assert isinstance(transformer.blocks[0].attn2.add_k_proj, LoRALinear)
    assert isinstance(transformer.blocks[0].attn2.add_v_proj, LoRALinear)
    assert result.reports[0].matched_key_count == 6
    assert result.reports[0].unmatched_key_count == 0
