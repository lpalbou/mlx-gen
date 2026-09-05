"""Catalog, capability, weight-definition and LoRA-mapping contracts of the MiniMax-H3 family."""

import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.minimax_h3.minimax_h3_initializer import MiniMaxH3Initializer
from mflux.models.minimax_h3.weights.h3_lora_mapping import MiniMaxH3LoRAMapping
from mflux.models.minimax_h3.weights.h3_weight_definition import MiniMaxH3WeightDefinition
from mflux.models.minimax_h3.weights.h3_weight_mapping import MiniMaxH3WeightMapping
from mflux.task_inference import get_model_capabilities


@pytest.mark.fast
def test_catalog_entries_and_capabilities():
    base = ModelConfig.from_name("minimax-h3")
    turbo = ModelConfig.from_name("minimax-h3-turbo")
    assert base.model_name == turbo.model_name == "MiniMaxAI/MiniMax-H3"
    assert base.transformer_overrides["default_steps"] == 50 and turbo.transformer_overrides["default_steps"] == 8
    assert turbo.transformer_overrides["default_video_shift"] == 6.0 and "turbo_lora" in turbo.transformer_overrides
    assert ModelConfig.from_name("MiniMaxAI/MiniMax-H3").aliases[0] == "minimax-h3"
    assert ModelConfig.minimax_h3_turbo_544p().transformer_overrides["default_width"] == 960
    assert ModelConfig.minimax_h3_turbo().transformer_overrides["default_video_shift"] == 6.0
    capabilities = get_model_capabilities(model="minimax-h3-turbo", model_config=turbo).to_dict()
    assert capabilities["family"] == "minimax-h3" and capabilities["label"] == "MiniMax-H3 Turbo"
    (row,) = capabilities["capabilities"]
    assert row["handler_id"] == "minimax-h3.generate" and row["public_task"] == "text-to-video"
    assert row["supports_negative_prompt"] is False and row["supports_lora"] is True and row["dimension_multiple"] == 32


@pytest.mark.fast
def test_weight_definition_components_and_quantization_predicate():
    names = [component.name for component in MiniMaxH3WeightDefinition.get_components()]
    assert names == ["text_encoder", "transformer", "vae", "audio_vae"]
    predicate = MiniMaxH3WeightDefinition.quantization_predicate
    assert predicate("transformer_blocks.0.attn.to_q", nn.Linear(5376, 7168, bias=False)) is True
    assert predicate("proj_in", nn.Linear(96, 5376)) is False  # fp32 keep-set
    assert predicate("time_embedder.linear_1", nn.Linear(256, 5376)) is False
    assert (
        predicate("proj_out", nn.Linear(5376, 96)) is False
        and predicate("audio_proj_out", nn.Linear(5376, 32)) is False
    )
    assert predicate("layers.0.self_attn.q_norm", nn.RMSNorm(128)) is False
    assert predicate("transformer_blocks.7.adaln_proj.linear", nn.Linear(2688, 6 * 5376 * 3)) is False
    assert predicate("norm_out.linear", nn.Linear(2688, 2 * 5376)) is False
    assert predicate("transformer_blocks.7.ff.net.2", nn.Linear(14336, 5376, bias=False)) is True
    assert MiniMaxH3WeightDefinition.transformer_precision_override("audio_proj_out.weight") == mx.float32
    assert MiniMaxH3WeightDefinition.transformer_precision_override("transformer_blocks.3.ff.net.2.weight") is None
    (tokenizer,) = MiniMaxH3WeightDefinition.get_tokenizers()
    assert tokenizer.add_special_tokens is False and tokenizer.padding == "longest"


@pytest.mark.fast
def test_text_encoder_key_selection_and_vae_transforms():
    needed = MiniMaxH3WeightMapping.text_encoder_key_is_needed
    assert needed("model.language_model.embed_tokens.weight")
    assert needed("model.language_model.layers.49.mlp.up_proj.weight")
    assert not needed("model.language_model.layers.50.mlp.up_proj.weight")
    assert (
        not needed("model.language_model.norm.weight")
        and not needed("model.visual.blocks.0.attn.qkv.weight")
        and not needed("lm_head.weight")
    )
    assert MiniMaxH3WeightMapping.video_vae_transform(mx.zeros((8, 3, 3, 3, 3))).shape == (8, 3, 3, 3, 3)
    assert MiniMaxH3WeightMapping.video_vae_transform(mx.zeros((8, 4, 3, 3, 3))).shape == (8, 3, 3, 3, 4)
    g, v = mx.random.normal((6, 1, 1)), mx.random.normal((6, 4, 7))
    folded = MiniMaxH3WeightMapping.fold_weight_norm(g, v)
    norms = mx.sqrt(mx.sum(mx.square(folded), axis=(1, 2)))
    assert mx.allclose(norms, mx.abs(g[:, 0, 0]), atol=1e-5).item()
    state = MiniMaxH3WeightMapping.convert_audio_vae_state(
        {"encoder.block.0.weight_g": g, "encoder.block.0.weight_v": v, "decoder.ups.0.0.weight": mx.zeros((4, 8, 9))}
    )
    assert state["encoder.block.0.weight"].shape == (6, 7, 4) and state["decoder.ups.0.0.weight"].shape == (8, 9, 4)


@pytest.mark.fast
def test_turbo_lora_keys_map_onto_every_target_and_alpha_scale_is_read(tmp_path):
    mappings = LoRALoader._build_pattern_mappings(MiniMaxH3LoRAMapping.get_mapping())
    keys = {
        f"{prefix}.{module}.lora_{matrix}.default.weight"
        for prefix in ("transformer_blocks.12", "token_refiner.refiner_blocks.1")
        for module in MiniMaxH3LoRAMapping.TARGET_MODULES
        for matrix in "AB"
    }
    normalized = LoRALoader._normalize_peft_adapter_infix({key: mx.zeros((1, 1)) for key in keys})
    for key in normalized:
        matched = [
            (m, LoRALoader._match_pattern(key, m.source_pattern))
            for m in mappings
            if LoRALoader._match_pattern(key, m.source_pattern) is not None
        ]
        assert matched, key
        assert {m.target_path.replace("{block}", str(block)) for m, block in matched} == {key.rsplit(".lora_", 1)[0]}
    path = tmp_path / "turbo.safetensors"
    mx.save_safetensors(
        str(path),
        {
            "transformer_blocks.0.attn.to_q.lora_A.default.weight": mx.zeros((128, 5376)),
            "transformer_blocks.0.attn.to_q.lora_B.default.weight": mx.zeros((7168, 128)),
        },
        {"alpha": "8"},
    )
    assert MiniMaxH3Initializer._peft_alpha_scale(str(path)) == pytest.approx(8 / 128)
