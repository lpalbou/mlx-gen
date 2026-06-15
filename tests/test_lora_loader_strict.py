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

    transformer = _TinyTransformer()
    result = LoRALoader.load_and_apply_lora_detailed(
        lora_mapping=_mapping(),
        transformer=transformer,
        lora_paths=[str(lora_path)],
        lora_scales=[0.9],
        role="transformer",
    )

    assert result.resolved_paths == [str(lora_path)]
    assert result.resolved_scales == [0.9]
    assert len(result.reports) == 1
    assert result.reports[0].matched_key_count == 2
    assert result.reports[0].unmatched_key_count == 0
    assert result.reports[0].applied_target_count == 1
    assert result.extra_metadata()["lora_applied_target_count"] == 1


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


@pytest.mark.fast
def test_lora_loader_summarizes_stacked_lora_logging_without_per_layer_paths(tmp_path, capsys):
    first_lora = tmp_path / "first.safetensors"
    second_lora = tmp_path / "second.safetensors"
    third_lora = tmp_path / "third.safetensors"

    for path in (first_lora, second_lora, third_lora):
        _save_lora(
            path,
            {
                "target.lora_A.weight": mx.zeros((2, 4)),
                "target.lora_B.weight": mx.zeros((3, 2)),
            },
        )

    transformer = _TinyTransformer()
    try:
        LoRALoader.set_debug_enabled(False)
        LoRALoader.load_and_apply_lora(
            lora_mapping=_mapping(),
            transformer=transformer,
            lora_paths=[str(first_lora), str(second_lora), str(third_lora)],
            lora_scales=[1.0, 1.0, 1.0],
        )

        output = capsys.readouterr().out
        assert "Fusing with existing LoRA at target" not in output
        assert "Adding to existing fusion at target" not in output
        assert "Applied to 1 layers" in output
    finally:
        LoRALoader.set_debug_enabled(False)


@pytest.mark.fast
def test_lora_loader_emits_fusion_target_logs_in_debug_mode(tmp_path, capsys):
    first_lora = tmp_path / "first.safetensors"
    second_lora = tmp_path / "second.safetensors"
    third_lora = tmp_path / "third.safetensors"

    for path in (first_lora, second_lora, third_lora):
        _save_lora(
            path,
            {
                "target.lora_A.weight": mx.zeros((2, 4)),
                "target.lora_B.weight": mx.zeros((3, 2)),
            },
        )

    transformer = _TinyTransformer()
    try:
        LoRALoader.set_debug_enabled(True)
        LoRALoader.load_and_apply_lora(
            lora_mapping=_mapping(),
            transformer=transformer,
            lora_paths=[str(first_lora), str(second_lora), str(third_lora)],
            lora_scales=[1.0, 1.0, 1.0],
        )

        output = capsys.readouterr().out
        assert "Fusing with existing LoRA at target" in output
        assert "Adding to existing fusion at target" in output
    finally:
        LoRALoader.set_debug_enabled(False)
