from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.qwen.weights.qwen_lora_mapping import QwenLoRAMapping


class _TinyQwenAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 3, bias=False)


class _TinyQwenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _TinyQwenAttention()
        self.img_mod_linear = nn.Linear(4, 6, bias=False)


class _TinyQwenTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = [_TinyQwenBlock()]


@pytest.mark.fast
def test_qwen_2511_diffusers_lora_keys_apply_to_attention_and_modulation(tmp_path: Path):
    lora_path = tmp_path / "qwen-2511-lora.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": mx.zeros((2, 4)),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": mx.zeros((3, 2)),
            "transformer.transformer_blocks.0.img_mod.1.lora_A.weight": mx.zeros((2, 4)),
            "transformer.transformer_blocks.0.img_mod.1.lora_B.weight": mx.zeros((6, 2)),
        },
    )
    transformer = _TinyQwenTransformer()

    LoRALoader.load_and_apply_lora(
        lora_mapping=QwenLoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(lora_path)],
        lora_scales=[0.9],
    )

    assert isinstance(transformer.transformer_blocks[0].attn.to_q, LoRALinear)
    assert isinstance(transformer.transformer_blocks[0].img_mod_linear, LoRALinear)
