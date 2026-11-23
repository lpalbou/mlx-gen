"""Workarounds for MLX kernel behaviour the MiniMax-H3 runtime depends on."""

import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.minimax_h3.model.h3_precision import QUANTIZED_MATMUL_MAX_ROWS, rowwise


def _bf16_scale_q8_linear(in_dim: int = 64, out_dim: int = 8) -> nn.QuantizedLinear:
    linear = nn.Linear(in_dim, out_dim, bias=False)
    linear.weight = (mx.random.normal((out_dim, in_dim)) * 0.1).astype(mx.bfloat16)
    quantized = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=8)
    mx.eval(quantized.parameters())
    assert quantized.scales.dtype == mx.bfloat16  # the layout every q8 package and bf16 checkpoint produces
    return quantized


def _reference(layer: nn.QuantizedLinear, x: mx.array) -> mx.array:
    weight = mx.dequantize(layer.weight, layer.scales, layer.biases, group_size=layer.group_size, bits=layer.bits)
    return x.astype(mx.float32) @ weight.astype(mx.float32).T


@pytest.mark.fast
def test_rowwise_quantized_linear_is_exact_past_32768_rows():
    layer = _bf16_scale_q8_linear()
    x = mx.random.normal((1, QUANTIZED_MATMUL_MAX_ROWS + 2, 64)).astype(mx.bfloat16)
    mx.eval(x)
    reference = _reference(layer, x)
    out = rowwise(layer, x)
    mx.eval(out, reference)
    rel = float(mx.abs(out.astype(mx.float32) - reference).max() / mx.abs(reference).max())
    assert out.dtype == mx.bfloat16 and out.shape == reference.shape
    assert rel < 2e-2, rel  # bf16 output rounding only


@pytest.mark.fast
def test_rowwise_matches_plain_call_below_the_limit():
    layer = _bf16_scale_q8_linear()
    x = mx.random.normal((1, 300, 64)).astype(mx.bfloat16)
    mx.eval(x)
    assert mx.array_equal(rowwise(layer, x), layer(x)).item()


@pytest.mark.fast
def test_quantized_entry_projections_keep_floating_inputs():
    """q8 packs `weight` as uint32; the entry projections must not cast their activations to it."""
    from mflux.models.minimax_h3.model.h3_precision import linear_input_dtype
    from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_vision_model import Qwen3VLVisionPatchEmbed

    embed = Qwen3VLVisionPatchEmbed(3, 4, 2, 64)
    mx.eval(embed.parameters())
    patches = mx.random.normal((10, 3 * 2 * 4 * 4))
    reference = embed(patches)
    nn.quantize(embed, group_size=32, bits=8)
    assert linear_input_dtype(embed.proj) == mx.float32 and embed.proj.weight.dtype == mx.uint32
    quantized = embed(patches)
    rel = float(mx.abs(quantized - reference).max() / mx.abs(reference).max())
    assert rel < 5e-2, rel


@pytest.mark.fast
def test_quantized_adaln_projection_keeps_its_modulation_tables():
    """A quantized AdaLN projection must feed floating activations, not the packed uint32 weight dtype.

    Casting to `weight.dtype` truncated `silu(temb)` to integers on a quantized projection, which is what
    made 8-bit AdaLN look unusable; the tables themselves survive 8-bit to well under a percent.
    """
    from mflux.models.minimax_h3.model.h3_transformer.h3_embedding import H3AdaLayerNormModulation

    modulation = H3AdaLayerNormModulation(time_embed_dim=128, hidden_size=64)
    mx.eval(modulation.parameters())
    temb = mx.random.normal((2, 128))
    reference = mx.concatenate(modulation(temb), axis=-1)
    nn.quantize(modulation, bits=8)
    assert modulation.linear.weight.dtype == mx.uint32
    quantized = mx.concatenate(modulation(temb), axis=-1)
    relative = float(mx.sqrt(mx.sum((quantized - reference) ** 2)) / mx.sqrt(mx.sum(reference**2)))
    assert relative < 2e-2, relative


@pytest.mark.fast
@pytest.mark.parametrize("bits", [3, 4, 5, 6, 8])
def test_lora_wraps_a_quantized_linear_of_any_width(bits):
    """Packed columns hold `inputs * bits / 32`, so the input width is not `columns * (32 // bits)` at 5 or 6 bits."""
    from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear

    quantized = nn.QuantizedLinear.from_linear(nn.Linear(192, 16, bias=False), group_size=64, bits=bits)
    wrapped = LoRALinear.from_linear(quantized, r=4)
    assert wrapped.lora_A.shape == (192, 4) and wrapped.lora_B.shape == (4, 16)
    assert wrapped(mx.random.normal((2, 192))).shape == (2, 16)


@pytest.mark.fast
@pytest.mark.parametrize("bits", [4, 6, 8])
def test_adapter_shapes_are_checked_against_the_real_input_width(bits):
    from mflux.models.common.lora.mapping.lora_loader import LoRAApplicationError, LoRALoader

    quantized = nn.QuantizedLinear.from_linear(nn.Linear(192, 16, bias=False), group_size=64, bits=bits)
    LoRALoader._validate_lora_matrix_shapes(quantized, mx.zeros((192, 4)), mx.zeros((4, 16)), "blocks.0.to_q")
    with pytest.raises(LoRAApplicationError, match=r"expected lora_A \(192, rank\)"):
        LoRALoader._validate_lora_matrix_shapes(quantized, mx.zeros((180, 4)), mx.zeros((4, 16)), "blocks.0.to_q")
