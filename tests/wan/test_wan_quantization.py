import pytest

from mflux.models.wan.wan_initializer import WanInitializer
from mflux.models.wan.weights.wan_weight_definition import WanWeightDefinition


class QuantizableModule:
    def to_quantized(self):
        return self


def test_wan_q8_keeps_conditioning_and_output_projection_bf16():
    module = QuantizableModule()

    assert not WanWeightDefinition.quantization_predicate("condition_embedder.time_proj", module, 8)
    assert not WanWeightDefinition.quantization_predicate("condition_embedder.text_embedder.linear_1", module, 8)
    assert not WanWeightDefinition.quantization_predicate("proj_out", module, 8)
    assert WanWeightDefinition.quantization_predicate("blocks.0.attn1.to_q", module, 8)
    assert WanWeightDefinition.quantization_predicate("blocks.0.ffn.net.0", module, 8)


def test_wan_q4_uses_existing_full_quantizable_module_policy():
    module = QuantizableModule()

    assert WanWeightDefinition.quantization_predicate("condition_embedder.time_proj", module, 4)
    assert WanWeightDefinition.quantization_predicate("proj_out", module, 4)


def test_wan_quantization_skips_non_quantizable_modules():
    assert not WanWeightDefinition.quantization_predicate("blocks.0.attn1.norm_q", object(), 8)


def test_wan_vae_component_is_not_quantized():
    components = {component.name: component for component in WanWeightDefinition().get_components()}

    assert components["vae"].skip_quantization


def test_wan_initializer_rejects_old_q8_sensitive_transformer_layout():
    weights = {
        "condition_embedder": {
            "time_embedder": {
                "linear_1": {
                    "weight": object(),
                    "scales": object(),
                    "biases": object(),
                }
            }
        }
    }

    with pytest.raises(ValueError, match="incompatible older quantization layout"):
        WanInitializer._validate_component_quantization_layout("transformer", weights, 8)


def test_wan_initializer_accepts_current_q8_sensitive_bf16_layout():
    weights = {
        "condition_embedder": {
            "time_embedder": {
                "linear_1": {
                    "weight": object(),
                }
            }
        },
        "blocks": {
            "0": {
                "attn1": {
                    "to_q": {
                        "weight": object(),
                        "scales": object(),
                        "biases": object(),
                    }
                }
            }
        },
    }

    WanInitializer._validate_component_quantization_layout("transformer", weights, 8)
