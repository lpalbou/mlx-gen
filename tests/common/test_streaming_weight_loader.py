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


def _write_prepared(path, model: nn.Module, bits: int) -> dict:
    flat = dict(tree_flatten(model.parameters()))
    keys = sorted(flat)
    path.mkdir()
    metadata = {"quantization_level": str(bits), "mflux_version": "test"}
    # Alternating keys put a module's packed weight and its scales in different shards.
    for i, chunk in enumerate((keys[::2], keys[1::2])):
        mx.save_safetensors(str(path / f"{i}.safetensors"), {k: flat[k] for k in chunk}, metadata)
    return flat


@pytest.mark.fast
def test_prepared_package_loads_each_module_at_its_stored_width(tmp_path):
    """A mixed q4/q8 package is rebuilt from its own tensors, not from whatever the current policy says."""

    def mixed(path, module):
        if not _predicate(path, module):
            return False
        return {"bits": 8 if path in ("layers.1", "embed") else 4}

    source = _Tiny()
    mx.eval(source.parameters())
    nn.quantize(source, class_predicate=mixed)
    mx.eval(source.parameters())
    flat = _write_prepared(tmp_path / "comp", source, bits=4)

    model = _Tiny()
    component = ComponentDefinition(name="comp", hf_subdir="comp", loading_mode="multi_glob", precision=mx.bfloat16)
    # A policy that would quantize everything at the package width must not decide the layout.
    bits = StreamingWeightLoader.load_into(
        model, tmp_path, component, quantize_arg=None, quantization_predicate=_predicate
    )
    assert bits == 4
    assert (model.layers[0].bits, model.layers[1].bits, model.layers[2].bits) == (4, 8, 4)
    assert model.embed.bits == 8 and isinstance(model.norm, nn.RMSNorm)
    got = dict(tree_flatten(model.parameters()))
    assert sorted(got) == sorted(flat)
    assert all(mx.array_equal(got[k], flat[k]).item() for k in flat)
    probe = mx.random.normal((2, 64))
    assert mx.array_equal(model.layers[1](probe), source.layers[1](probe)).item()
    assert mx.array_equal(model.layers[0](probe), source.layers[0](probe)).item()


@pytest.mark.fast
def test_prepared_package_for_another_layout_is_refused(tmp_path):
    class _Wider(_Tiny):
        def __init__(self):
            super().__init__()
            self.extra = nn.Linear(64, 64)

    source = _Wider()
    mx.eval(source.parameters())
    nn.quantize(source, class_predicate=_predicate, bits=8)
    mx.eval(source.parameters())
    _write_prepared(tmp_path / "comp", source, bits=8)

    component = ComponentDefinition(name="comp", hf_subdir="comp", loading_mode="multi_glob", precision=mx.bfloat16)
    with pytest.raises(ValueError, match="does not have"):
        StreamingWeightLoader.load_into(_Tiny(), tmp_path, component, quantize_arg=None)


def _prepared_from(tmp_path, source: nn.Module, bits: int, drop: tuple[str, ...] = (), replace: dict | None = None):
    mx.eval(source.parameters())
    flat = {k: v for k, v in tree_flatten(source.parameters()) if not k.startswith(drop)} if drop else None
    path = tmp_path / "comp"
    path.mkdir()
    flat = flat if flat is not None else dict(tree_flatten(source.parameters()))
    flat.update(replace or {})
    mx.save_safetensors(str(path / "0.safetensors"), flat, {"quantization_level": str(bits), "mflux_version": "test"})
    return ComponentDefinition(name="comp", hf_subdir="comp", loading_mode="multi_glob", precision=mx.bfloat16)


@pytest.mark.fast
@pytest.mark.parametrize("bits", [3, 5, 6])
def test_prepared_package_round_trips_widths_that_do_not_divide_32(tmp_path, bits):
    source = _Tiny()
    nn.quantize(source, class_predicate=_predicate, bits=bits)
    component = _prepared_from(tmp_path, source, bits)
    model = _Tiny()
    assert StreamingWeightLoader.load_into(model, tmp_path, component, quantize_arg=None) == bits
    assert model.layers[0].bits == bits and model.embed.bits == bits
    probe = mx.random.normal((2, 64))
    assert mx.array_equal(model.layers[2](probe), source.layers[2](probe)).item()


@pytest.mark.fast
def test_prepared_package_missing_a_module_is_refused(tmp_path):
    """`update(strict=False)` would leave the missing layer at its random initial value."""
    source = _Tiny()
    nn.quantize(source, class_predicate=_predicate, bits=8)
    component = _prepared_from(tmp_path, source, 8, drop=("layers.1.",))
    with pytest.raises(ValueError, match=r"missing 2 tensors.*layers\.1\."):
        StreamingWeightLoader.load_into(_Tiny(), tmp_path, component, quantize_arg=None)


@pytest.mark.fast
def test_prepared_package_with_a_tensor_of_another_shape_is_refused(tmp_path):
    source = _Tiny()
    nn.quantize(source, class_predicate=_predicate, bits=8)
    component = _prepared_from(tmp_path, source, 8, replace={"norm.weight": mx.ones((32,))})
    with pytest.raises(ValueError, match=r"'norm\.weight' is stored as \(32,\) and this model expects \(64,\)"):
        StreamingWeightLoader.load_into(_Tiny(), tmp_path, component, quantize_arg=None)


@pytest.mark.fast
def test_prepared_package_at_another_group_size_is_refused(tmp_path):
    """Group 32 at 8-bit packs like group 64 at 4-bit; read as the latter it would fail at the first forward."""
    source = _Tiny()
    nn.quantize(source, class_predicate=_predicate, group_size=32, bits=8)
    component = _prepared_from(tmp_path, source, 8)
    with pytest.raises(ValueError, match="do not fit its 64 inputs at a group size of 64"):
        StreamingWeightLoader.load_into(_Tiny(), tmp_path, component, quantize_arg=None)
