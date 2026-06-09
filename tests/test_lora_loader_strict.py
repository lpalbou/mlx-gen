from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRAApplicationError, LoRALoader
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget


class _TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.target = nn.Linear(4, 3, bias=False)


def _save_lora(path: Path, weights: dict[str, mx.array]) -> None:
    mx.save_safetensors(str(path), weights)


def _mapping() -> list[LoRATarget]:
    return [
        LoRATarget(
            model_path="target",
            possible_up_patterns=["target.lora_B.weight"],
            possible_down_patterns=["target.lora_A.weight"],
        )
    ]


@pytest.mark.fast
def test_lora_loader_applies_compatible_target(tmp_path):
    lora_path = tmp_path / "compatible.safetensors"
    _save_lora(
        lora_path,
        {
            "target.lora_A.weight": mx.zeros((2, 4)),
            "target.lora_B.weight": mx.zeros((3, 2)),
        },
    )
    transformer = _TinyTransformer()

    paths, scales = LoRALoader.load_and_apply_lora(
        lora_mapping=_mapping(),
        transformer=transformer,
        lora_paths=[str(lora_path)],
        lora_scales=[0.9],
    )

    assert paths == [str(lora_path)]
    assert scales == [0.9]
    assert isinstance(transformer.target, LoRALinear)


@pytest.mark.fast
def test_lora_loader_rejects_zero_match_adapter(tmp_path):
    lora_path = tmp_path / "zero-match.safetensors"
    _save_lora(lora_path, {"other.weight": mx.zeros((1, 1))})

    with pytest.raises(LoRAApplicationError, match="did not match"):
        LoRALoader.load_and_apply_lora(
            lora_mapping=_mapping(),
            transformer=_TinyTransformer(),
            lora_paths=[str(lora_path)],
            lora_scales=[1.0],
        )


@pytest.mark.fast
def test_lora_loader_rejects_incompatible_matrix_shapes(tmp_path):
    lora_path = tmp_path / "flux2-dev-width-on-klein.safetensors"
    _save_lora(
        lora_path,
        {
            "target.lora_A.weight": mx.zeros((16, 6144)),
            "target.lora_B.weight": mx.zeros((6144, 16)),
        },
    )

    with pytest.raises(LoRAApplicationError, match="incompatible with the selected model"):
        LoRALoader.load_and_apply_lora(
            lora_mapping=_mapping(),
            transformer=_TinyTransformer(),
            lora_paths=[str(lora_path)],
            lora_scales=[1.0],
        )


@pytest.mark.fast
def test_lora_loader_rejects_scales_without_paths():
    with pytest.raises(LoRAApplicationError, match="requires --lora-paths"):
        LoRALoader.load_and_apply_lora(
            lora_mapping=_mapping(),
            transformer=_TinyTransformer(),
            lora_paths=None,
            lora_scales=[1.0],
        )
