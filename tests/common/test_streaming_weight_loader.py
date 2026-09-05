"""The shard-streaming loader must reproduce whole-component loading, quantized or not."""

import json

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.weights.loading.streaming_weight_loader import StreamingWeightLoader
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50, 64)
        self.layers = [nn.Linear(64, 64) for _ in range(3)]
        self.head = nn.Linear(64, 8)
        self.norm = nn.RMSNorm(64)


def _write_shards(path, weights: dict, prefix: str = ""):
    path.mkdir(parents=True, exist_ok=True)
    keys = sorted(weights)
    shards = [keys[::2], keys[1::2]]
    index = {}
    for i, shard in enumerate(shards):
        name = f"model-{i}.safetensors"
        mx.save_safetensors(str(path / name), {prefix + k: weights[k] for k in shard})
        index.update({prefix + k: name for k in shard})
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))


def _predicate(path, module, bits=None):
    return hasattr(module, "to_quantized") and module.weight.shape[-1] % 64 == 0


@pytest.mark.fast
@pytest.mark.parametrize("quantize", [None, 8])
def test_streaming_matches_whole_component_loading(tmp_path, quantize):
    reference = _Tiny()
    mx.eval(reference.parameters())
    weights = {k: v.astype(mx.bfloat16) for k, v in tree_flatten(reference.parameters())}
    _write_shards(tmp_path / "comp", weights)
    expected = _Tiny()
    expected.update(tree_unflatten(list(weights.items())))
    if quantize:
        nn.quantize(expected, class_predicate=_predicate, bits=quantize)
    mx.eval(expected.parameters())

    model = _Tiny()
    component = ComponentDefinition(name="comp", hf_subdir="comp", loading_mode="multi_glob", precision=mx.bfloat16)
    bits = StreamingWeightLoader.load_into(
        model, tmp_path, component, quantize_arg=quantize, quantization_predicate=_predicate
    )
    assert bits == quantize
    got, want = dict(tree_flatten(model.parameters())), dict(tree_flatten(expected.parameters()))
    assert sorted(got) == sorted(want)
    assert all(mx.array_equal(got[k], want[k]).item() for k in want)


@pytest.mark.fast
def test_streaming_applies_mapping_prefix_filter_and_layer_limit(tmp_path):
    source = _Tiny()
    mx.eval(source.parameters())
    weights = {k: v for k, v in tree_flatten(source.parameters())}
    weights["visual.weight"] = mx.zeros((4, 4))
    _write_shards(tmp_path / "te", weights, prefix="model.language_model.")

    class _Truncated(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(50, 64)
            self.layers = [nn.Linear(64, 64) for _ in range(2)]

    mapping = [
        WeightTarget(to_pattern="embed.weight", from_pattern=["model.language_model.embed.weight"]),
        WeightTarget(to_pattern="layers.{layer}.weight", from_pattern=["model.language_model.layers.{layer}.weight"]),
        WeightTarget(to_pattern="layers.{layer}.bias", from_pattern=["model.language_model.layers.{layer}.bias"]),
    ]
    component = ComponentDefinition(
        name="te",
        hf_subdir="te",
        loading_mode="multi_glob",
        precision=mx.float32,
        num_layers=2,
        mapping_getter=lambda: mapping,
        weight_prefix_filters=["model.language_model."],
    )
    model = _Truncated()
    StreamingWeightLoader.load_into(model, tmp_path, component, quantize_arg=None)
    for i in range(2):
        assert mx.array_equal(model.layers[i].weight, source.layers[i].weight).item()
    assert mx.array_equal(model.embed.weight, source.embed.weight).item()


@pytest.mark.fast
def test_streaming_loads_prepared_packages_shard_by_shard(tmp_path):
    """A prepared (already quantized) package restores the same parameters through the streaming loader."""
    import json as _json

    source = _Tiny()
    mx.eval(source.parameters())
    nn.quantize(source, class_predicate=_predicate, bits=8)
    mx.eval(source.parameters())
    flat = dict(tree_flatten(source.parameters()))
    keys = sorted(flat)
    path = tmp_path / "comp"
    path.mkdir()
    metadata = {"quantization_level": "8", "mflux_version": "test"}
    index = {}
    for i, chunk in enumerate((keys[: len(keys) // 2], keys[len(keys) // 2 :])):
        name = f"{i}.safetensors"
        mx.save_safetensors(str(path / name), {k: flat[k] for k in chunk}, metadata)
        index.update({k: name for k in chunk})
    (path / "model.safetensors.index.json").write_text(_json.dumps({"metadata": metadata, "weight_map": index}))

    model = _Tiny()
    component = ComponentDefinition(name="comp", hf_subdir="comp", loading_mode="multi_glob", precision=mx.bfloat16)
    bits = StreamingWeightLoader.load_into(
        model, tmp_path, component, quantize_arg=None, quantization_predicate=_predicate
    )
    got = dict(tree_flatten(model.parameters()))
    assert bits == 8 and sorted(got) == keys
    assert all(mx.array_equal(got[k], flat[k]).item() for k in keys)


@pytest.mark.fast
def test_streaming_load_applies_default_cache_limit_once(tmp_path, monkeypatch):
    """Python-API hosts never run the CLI memory setup, so the streaming loader applies the process default."""
    from mflux.utils.runtime_memory import RuntimeMemory

    calls: list[int] = []
    monkeypatch.setattr(RuntimeMemory, "_cache_limit_state", "unset")
    monkeypatch.setattr(mx, "set_cache_limit", lambda value: calls.append(value))
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: None)
    model = _Tiny()
    component = ComponentDefinition(name="comp", hf_subdir="missing", loading_mode="multi_glob", precision=mx.bfloat16)
    with pytest.raises(Exception):
        StreamingWeightLoader.load_into(model, tmp_path, component, quantize_arg=None)
    assert len(calls) == 1 and calls[0] == RuntimeMemory.resolve_cache_limit_bytes(None)
    assert RuntimeMemory._cache_limit_state == "default"
