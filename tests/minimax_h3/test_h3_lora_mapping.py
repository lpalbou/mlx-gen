"""Community MiniMax-H3 adapters over the original checkpoint's module names load with the reference semantics."""

from pathlib import Path

import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.lora.mapping.lora_loader import LoRAApplicationError, LoRALoader
from mflux.models.minimax_h3.minimax_h3_initializer import MiniMaxH3Initializer
from mflux.models.minimax_h3.model.h3_transformer.h3_transformer import MiniMaxH3Transformer
from mflux.models.minimax_h3.weights.h3_lora_mapping import MiniMaxH3LoRAMapping
from tests.minimax_h3.test_h3_pipeline_tiny import _tiny_model

SKELETOR = Path.home() / ".cache/huggingface/hub/H3_Skeletor_1.4.safetensors"


def _tiny_transformer() -> MiniMaxH3Transformer:
    transformer = MiniMaxH3Transformer(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=64,
        text_dim=16,
        freq_dim=32,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    mx.eval(transformer.parameters())
    return transformer


def _module(root: nn.Module, path: str):
    node = root
    for part in path.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node


def _weight(root: nn.Module, path: str) -> mx.array:
    node = _module(root, path)
    return node.linear.weight if hasattr(node, "linear") else node.weight


def original_layout_lora(
    transformer: MiniMaxH3Transformer, rank: int, prefix: str, seed: int = 0, std: float = 0.1
) -> dict:
    """A random adapter over every original-checkpoint module the transformer has, in PEFT naming."""
    key = mx.random.key(seed)
    weights: dict[str, mx.array] = {}

    def pair(name: str, out_features: int, in_features: int) -> None:
        nonlocal key
        key, k_a, k_b = mx.random.split(key, 3)
        weights[f"{prefix}{name}.lora_A.weight"] = mx.random.normal((rank, in_features), key=k_a) * std
        weights[f"{prefix}{name}.lora_B.weight"] = mx.random.normal((out_features, rank), key=k_b) * std

    block_sets = (
        ("blocks", transformer.transformer_blocks, True),
        ("token_refiner.blocks", transformer.token_refiner.refiner_blocks, False),
    )
    for name, blocks, has_adaln in block_sets:
        for index, block in enumerate(blocks):
            inner, hidden = block.attn.to_q.weight.shape
            two_ffn, _ = block.ff.net[0].proj.weight.shape
            pair(f"{name}.{index}.attn.qkv_proj", 3 * inner, hidden)
            pair(f"{name}.{index}.attn.out_proj", hidden, inner)
            pair(f"{name}.{index}.mlp.fc1", two_ffn, hidden)
            pair(f"{name}.{index}.mlp.fc2", hidden, two_ffn // 2)
            if has_adaln:
                pair(f"{name}.{index}.adaln_proj.linear", *block.adaln_proj.linear.weight.shape)
    for source, target in MiniMaxH3LoRAMapping.ORIGINAL_STANDALONE_MODULES:
        pair(source, *_weight(transformer, target).shape)
    return weights


def _apply(transformer: MiniMaxH3Transformer, path: Path, scale: float = 1.0):
    return LoRALoader.load_and_apply_lora_detailed(
        lora_mapping=MiniMaxH3LoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(path)],
        lora_scales=[scale],
        state_dict_transform=MiniMaxH3LoRAMapping.normalize_state_dict,
    )


def _close(a: mx.array, b: mx.array, rel: float = 5e-3) -> bool:
    """Equal up to TF32 rounding: fp32 GEMMs run in TF32 by default on M5 (`h3_precision`), and the
    two sides sum the same products in different orders, so the bound is relative to the output's
    scale. A wrong row order or a wrong adapter scale misses by the size of the output itself."""
    return (mx.abs(a - b).max() <= rel * mx.abs(b).max()).item()


def _matches(key: str) -> set[str]:
    mappings = LoRALoader._build_pattern_mappings(MiniMaxH3LoRAMapping.get_mapping())
    targets = set()
    for mapping in mappings:
        block = LoRALoader._match_pattern(key, mapping.source_pattern)
        if block is not None:
            targets.add(mapping.target_path.replace("{block}", str(block)))
    return targets


@pytest.mark.fast
@pytest.mark.parametrize("prefix", ["", "diffusion_model."])
@pytest.mark.parametrize("suffixes", [("lora_A.weight", "lora_B.weight"), ("lora_down.weight", "lora_up.weight")])
def test_original_layout_keys_map_onto_the_diffusers_module_names(prefix, suffixes):
    down, up = suffixes
    expectations = {
        "blocks.7.attn.qkv_proj": {f"transformer_blocks.7.attn.{p}" for p in ("to_q", "to_k", "to_v")},
        "blocks.7.attn.out_proj": {"transformer_blocks.7.attn.to_out.0"},
        "blocks.7.mlp.fc1": {"transformer_blocks.7.ff.net.0.proj"},
        "blocks.7.mlp.fc2": {"transformer_blocks.7.ff.net.2"},
        "blocks.7.adaln_proj.linear": {"transformer_blocks.7.adaln_proj.linear"},
        "token_refiner.blocks.1.attn.qkv_proj": {
            f"token_refiner.refiner_blocks.1.attn.{p}" for p in ("to_q", "to_k", "to_v")
        },
        "token_refiner.blocks.1.mlp.fc1": {"token_refiner.refiner_blocks.1.ff.net.0.proj"},
        "video_patch_proj": {"proj_in"},
        "condition_proj": {"context_embedder"},
        "time_embedder.proj_out": {"time_embedder.linear_2"},
        "final_layer.adaln_proj.linear": {"norm_out.linear"},
        "final_layer.audio_out": {"audio_proj_out"},
    }
    for source, targets in expectations.items():
        assert _matches(f"{prefix}{source}.{down}") == targets, source
        assert _matches(f"{prefix}{source}.{up}") == targets, source
        assert _matches(f"{prefix}{source}.alpha") == targets, source
    # Refiner blocks carry no AdaLN projection, and the original names never collide with the diffusers ones.
    assert _matches(f"{prefix}token_refiner.blocks.1.adaln_proj.linear.{down}") == set()
    assert _matches("transformer_blocks.7.attn.to_q.lora_A.weight") == {"transformer_blocks.7.attn.to_q"}
    assert _matches("diffusion_model.transformer_blocks.7.ff.net.0.proj.lora_B.weight") == {
        "transformer_blocks.7.ff.net.0.proj"
    }


@pytest.mark.fast
def test_original_layout_reproduces_the_reference_fused_projections(tmp_path):
    """Fused QKV shares `lora_A` and splits `lora_B` in thirds; `fc1` swaps its `[gate; value]` halves."""
    transformer = _tiny_transformer()
    rank = 4
    # Deltas larger than the base weights, so a wrong row order is an O(1) miss; the tolerance is TF32's
    # (fp32 GEMMs run in TF32 by default on M5, see `h3_precision`), far below that.
    adapter = original_layout_lora(transformer, rank=rank, prefix="diffusion_model.", std=0.5)
    path = tmp_path / "original.safetensors"
    mx.save_safetensors(str(path), adapter)

    block = transformer.transformer_blocks[1]
    base = {
        name: mx.array(_weight(transformer, f"transformer_blocks.1.{name}"))
        for name in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
    }
    proj_in_weight, proj_in_bias = mx.array(transformer.proj_in.weight), mx.array(transformer.proj_in.bias)
    adaln_weight = mx.array(block.adaln_proj.linear.weight)

    scale = 0.7
    result = _apply(transformer, path, scale=scale)
    (report,) = result.reports
    assert report.unmatched_key_count == 0 and report.matched_key_count == len(adapter)
    assert report.applied_target_count == 2 * 7 + 1 * 6 + len(MiniMaxH3LoRAMapping.ORIGINAL_STANDALONE_MODULES)

    def delta(name: str) -> mx.array:
        return adapter[f"diffusion_model.{name}.lora_B.weight"] @ adapter[f"diffusion_model.{name}.lora_A.weight"]

    close = _close

    hidden = base["attn.to_q"].shape[1]
    inner = base["attn.to_q"].shape[0]
    x = mx.random.normal((1, 5, hidden), key=mx.random.key(9))

    # Reference attention: `qkv_proj(x).chunk(3)` over `[q_all; k_all; v_all]` rows.
    fused = mx.concatenate([base["attn.to_q"], base["attn.to_k"], base["attn.to_v"]], axis=0)
    fused = fused + scale * delta("blocks.1.attn.qkv_proj")
    reference_q, reference_k, reference_v = mx.split(x @ fused.T, 3, axis=-1)
    assert reference_q.shape[-1] == inner
    assert close(block.attn.to_q(x), reference_q) and not close(block.attn.to_q(x), reference_k)
    assert close(block.attn.to_k(x), reference_k)
    assert close(block.attn.to_v(x), reference_v)
    attention = mx.random.normal((1, 5, inner), key=mx.random.key(10))
    reference_out = attention @ (base["attn.to_out.0"] + scale * delta("blocks.1.attn.out_proj")).T
    assert close(block.attn.to_out[0](attention), reference_out)

    # Reference MLP: `fc1` packs `[gate; value]`, `fc2(silu(gate) * value)`. Ours packs `[value; gate]`.
    value_rows, gate_rows = mx.split(base["ff.net.0.proj"], 2, axis=0)
    reference_fc1 = mx.concatenate([gate_rows, value_rows], axis=0) + scale * delta("blocks.1.mlp.fc1")
    gate, value = mx.split(x @ reference_fc1.T, 2, axis=-1)
    reference_hidden = nn.silu(gate) * value
    assert close(block.ff.net[0](x), reference_hidden)
    # Applying the `[gate; value]` rows unswapped onto our `[value; gate]` projection is a different function.
    unswapped = base["ff.net.0.proj"] + scale * delta("blocks.1.mlp.fc1")
    unswapped_value, unswapped_gate = mx.split(x @ unswapped.T, 2, axis=-1)
    assert not close(block.ff.net[0](x), unswapped_value * nn.silu(unswapped_gate))
    reference_ff = reference_hidden @ (base["ff.net.2"] + scale * delta("blocks.1.mlp.fc2")).T
    assert close(block.ff(x), reference_ff)

    # Standalone and AdaLN projections are plain renames (bias untouched).
    patch = mx.random.normal((1, 3, proj_in_weight.shape[1]), key=mx.random.key(11))
    reference_patch = patch @ (proj_in_weight + scale * delta("video_patch_proj")).T + proj_in_bias
    assert close(transformer.proj_in(patch), reference_patch)
    temb = mx.random.normal((2, adaln_weight.shape[1]), key=mx.random.key(12))
    reference_adaln = nn.silu(temb) @ (adaln_weight + scale * delta("blocks.1.adaln_proj.linear")).T
    reference_adaln = reference_adaln + block.adaln_proj.linear.linear.bias
    assert close(mx.concatenate(block.adaln_proj(temb), axis=-1).reshape(2, -1), reference_adaln)


@pytest.mark.fast
def test_generate_video_runs_with_an_original_layout_adapter_on_every_target(tmp_path):
    """The initializer path (scale, layout guard, key normalisation) and a forward through every wrapped linear."""
    model = _tiny_model()
    adapter = original_layout_lora(model.transformer, rank=2, prefix="diffusion_model.")
    path = tmp_path / "H3_tiny.safetensors"
    mx.save_safetensors(str(path), adapter)
    reference = model.generate_video(seed=1, prompt="A fox", num_frames=124, width=96, height=64, num_inference_steps=1)

    MiniMaxH3Initializer._apply_lora(model, lora_paths=[str(path)], lora_scales=[1.0])
    assert model.lora_paths == [str(path)] and model.lora_scales == [1.0]
    (report,) = model.lora_application_result.reports
    assert report.unmatched_key_count == 0 and report.scale == 1.0  # no alpha anywhere: alpha == rank

    video = model.generate_video(seed=1, prompt="A fox", num_frames=124, width=96, height=64, num_inference_steps=1)
    assert video.num_frames == 124 and video.frames[0].size == (96, 64)
    assert video.lora_paths == [str(path)] and video.lora_scales == [1.0]
    assert video.extra_metadata["lora_applied_target_count"] == report.applied_target_count
    assert video.frames[0].tobytes() != reference.frames[0].tobytes()


@pytest.mark.fast
def test_musubi_flattened_keys_recover_their_dotted_module_paths():
    flattened = {
        "lora_unet_blocks_0_attn_qkv_proj.lora_down.weight": mx.zeros((1, 1)),
        "lora_unet_blocks_0_attn_qkv_proj.lora_up.weight": mx.zeros((1, 1)),
        "lora_unet_blocks_0_attn_qkv_proj.alpha": mx.array(4.0),
        "lora_unet_blocks_12_mlp_fc2.lora_down.weight": mx.zeros((1, 1)),
        "lora_unet_token_refiner_blocks_1_mlp_fc1.lora_up.weight": mx.zeros((1, 1)),
        "lora_unet_final_layer_video_out.lora_up.weight": mx.zeros((1, 1)),
        "lora_unet_time_embedder_proj_in.lora_down.weight": mx.zeros((1, 1)),
        "lora_unet_condition_proj.lora_down.weight": mx.zeros((1, 1)),
    }
    normalized = MiniMaxH3LoRAMapping.normalize_state_dict(flattened)
    assert set(normalized) == {
        "blocks.0.attn.qkv_proj.lora_down.weight",
        "blocks.0.attn.qkv_proj.lora_up.weight",
        "blocks.0.attn.qkv_proj.alpha",
        "blocks.12.mlp.fc2.lora_down.weight",
        "token_refiner.blocks.1.mlp.fc1.lora_up.weight",
        "final_layer.video_out.lora_up.weight",
        "time_embedder.proj_in.lora_down.weight",
        "condition_proj.lora_down.weight",
    }
    assert all(_matches(key) for key in normalized)
    untouched = {"diffusion_model.blocks.0.mlp.fc1.lora_A.weight": mx.zeros((1, 1))}
    assert MiniMaxH3LoRAMapping.normalize_state_dict(untouched) is untouched
    with pytest.raises(LoRAApplicationError, match="lora_unet_"):
        MiniMaxH3LoRAMapping.normalize_state_dict({"lora_unet_blocks_0_attn_norm.lora_up.weight": mx.zeros((1, 1))})


@pytest.mark.fast
def test_adapter_scale_follows_the_file_and_rejects_the_interleaved_layout(tmp_path):
    def save(name: str, weights: dict, metadata: dict | None = None) -> str:
        path = tmp_path / name
        mx.save_safetensors(str(path), weights, metadata)
        return str(path)

    pair = {
        "diffusion_model.blocks.0.attn.out_proj.lora_A.weight": mx.zeros((8, 16)),
        "diffusion_model.blocks.0.attn.out_proj.lora_B.weight": mx.zeros((16, 8)),
    }
    # PEFT metadata alpha (the Turbo adapters): alpha / rank.
    assert MiniMaxH3Initializer.adapter_scale(save("peft.safetensors", pair, {"alpha": "2"})) == pytest.approx(2 / 8)
    # No alpha anywhere (ai-toolkit exports): alpha == rank.
    assert MiniMaxH3Initializer.adapter_scale(save("aitk.safetensors", pair)) == 1.0
    # Kohya `.alpha` tensors are folded per target by the loader, so the file-level factor stays 1.
    kohya = {
        "blocks.0.attn.out_proj.lora_down.weight": mx.zeros((8, 16)),
        "blocks.0.attn.out_proj.lora_up.weight": mx.zeros((16, 8)),
        "blocks.0.attn.out_proj.alpha": mx.array(4.0),
    }
    assert MiniMaxH3Initializer.adapter_scale(save("kohya.safetensors", kohya, {"alpha": "2"})) == 1.0
    diffsynth = {
        "blocks.0.attn.qkv_proj.lora_A.default.weight": mx.zeros((8, 16)),
        "blocks.0.attn.qkv_proj.lora_B.default.weight": mx.zeros((48, 8)),
    }
    with pytest.raises(LoRAApplicationError, match="DiffSynth"):
        MiniMaxH3Initializer.adapter_scale(save("diffsynth.safetensors", diffsynth))


@pytest.mark.fast
def test_kohya_alpha_tensors_scale_each_target(tmp_path):
    transformer = _tiny_transformer()
    block = transformer.transformer_blocks[0]
    base = mx.array(block.attn.to_out[0].weight)
    hidden, inner = base.shape
    down = mx.random.normal((4, inner), key=mx.random.key(1))
    up = mx.random.normal((hidden, 4), key=mx.random.key(2))
    path = tmp_path / "kohya.safetensors"
    mx.save_safetensors(
        str(path),
        {
            "blocks.0.attn.out_proj.lora_down.weight": down,
            "blocks.0.attn.out_proj.lora_up.weight": up,
            "blocks.0.attn.out_proj.alpha": mx.array(2.0),
        },
    )
    _apply(transformer, path, scale=1.0)
    x = mx.random.normal((1, 3, inner), key=mx.random.key(3))
    expected = x @ (base + (2.0 / 4) * (up @ down)).T
    assert _close(block.attn.to_out[0](x), expected)


@pytest.mark.skipif(not SKELETOR.exists(), reason="local civitai adapter not present")
@pytest.mark.fast
def test_civitai_ai_toolkit_adapter_matches_every_key():
    """`H3_Skeletor_1.4.safetensors` (ai-toolkit 0.12.18, rank 8): 52 blocks x 4 fused modules, no alpha."""
    weights, metadata = mx.load(str(SKELETOR), return_metadata=True)
    assert len(weights) == 416 and "alpha" not in (metadata or {})
    normalized = LoRALoader._normalize_peft_adapter_infix(
        {LoRALoader._normalize_state_dict_key(key): value for key, value in weights.items()}
    )
    normalized = MiniMaxH3LoRAMapping.normalize_state_dict(normalized)
    unmatched = [key for key in normalized if not _matches(key)]
    assert unmatched == []
    assert MiniMaxH3Initializer.adapter_scale(str(SKELETOR)) == 1.0
