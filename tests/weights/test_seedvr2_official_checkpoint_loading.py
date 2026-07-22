import mlx.core as mx
import pytest
import torch
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.download_policy import DownloadRequiredError
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.seedvr2.model.seedvr2_text_encoder.text_embeddings import SeedVR2TextEmbeddings
from mflux.models.seedvr2.model.seedvr2_transformer.transformer import SeedVR2Transformer
from mflux.models.seedvr2.seedvr2_initializer import SeedVR2Initializer
from mflux.models.seedvr2.weights.seedvr2_weight_definition import (
    SeedVR2WeightDefinition,
    SeedVR2WeightDefinition3BOfficial,
    SeedVR2WeightDefinition3BPrepared,
    SeedVR2WeightDefinition7BOfficial,
    SeedVR2WeightDefinition7BOfficialSharp,
    SeedVR2WeightDefinition7BPrepared,
)


@pytest.mark.fast
def test_torch_checkpoint_loader_uses_weights_only(tmp_path, monkeypatch):
    observed = {}

    def fake_load(file_path, *, map_location, weights_only):
        observed["file_path"] = file_path
        observed["map_location"] = map_location
        observed["weights_only"] = weights_only
        return {"state_dict": {"linear.weight": torch.ones((1,), dtype=torch.float32)}}

    monkeypatch.setattr(torch, "load", fake_load)

    weights = WeightLoader._load_torch_checkpoint(tmp_path / "model.pth")

    assert observed == {
        "file_path": tmp_path / "model.pth",
        "map_location": "cpu",
        "weights_only": True,
    }
    assert weights["linear.weight"].shape == (1,)


@pytest.mark.fast
def test_weight_loader_loads_torch_checkpoint_directory(tmp_path):
    torch.save(
        {
            "state_dict": {
                "linear.weight": torch.ones((2, 3), dtype=torch.float32),
                "linear.bias": torch.ones((2,), dtype=torch.bfloat16),
            }
        },
        tmp_path / "model.pth",
    )

    weights = WeightLoader._load_safetensors(tmp_path, "torch_checkpoint", ["model.pth"])

    assert weights["linear.weight"].shape == (2, 3)
    assert weights["linear.bias"].dtype == mx.bfloat16


@pytest.mark.fast
def test_weight_loader_loads_torch_tensor_directory(tmp_path):
    torch.save(torch.ones((58, 5120), dtype=torch.bfloat16), tmp_path / "pos_emb.pt")

    weights = WeightLoader._load_safetensors(tmp_path, "torch_tensor", ["pos_emb.pt"])

    assert set(weights) == {"embedding"}
    assert weights["embedding"].shape == (58, 5120)
    assert weights["embedding"].dtype == mx.bfloat16


@pytest.mark.fast
def test_torch_checkpoint_loader_rejects_empty_tensor_map(tmp_path, monkeypatch):
    def fake_load(file_path, *, map_location, weights_only):
        return {"state_dict": {"epoch": 1}}

    monkeypatch.setattr(torch, "load", fake_load)

    with pytest.raises(ValueError, match="Unsupported Torch checkpoint payload"):
        WeightLoader._load_torch_checkpoint(tmp_path / "model.pth")


@pytest.mark.fast
def test_torch_checkpoint_loader_rejects_mixed_tensor_and_metadata_dict(tmp_path, monkeypatch):
    def fake_load(file_path, *, map_location, weights_only):
        return {"state_dict": {"linear.weight": torch.ones((1,), dtype=torch.float32), "epoch": 1}}

    monkeypatch.setattr(torch, "load", fake_load)

    with pytest.raises(ValueError, match="Unsupported Torch checkpoint payload"):
        WeightLoader._load_torch_checkpoint(tmp_path / "model.pth")


@pytest.mark.fast
def test_seedvr2_weight_definition_selects_official_layout_from_source_handle():
    patterns = SeedVR2WeightDefinition.get_download_patterns_for_source(
        ModelConfig.seedvr2_3b(),
        "seedvr2",
    )

    assert patterns == ["seedvr2_ema_3b.pth", "ema_vae.pth", "pos_emb.pt"]


@pytest.mark.fast
def test_seedvr2_weight_definition_selects_official_7b_layout_from_source_handle():
    patterns = SeedVR2WeightDefinition.get_download_patterns_for_source(
        ModelConfig.seedvr2_7b(),
        "ByteDance-Seed/SeedVR2-7B",
    )

    assert patterns == ["seedvr2_ema_7b.pth", "ema_vae.pth"]


@pytest.mark.fast
def test_seedvr2_weight_definition_selects_official_7b_sharp_layout_from_source_handle():
    patterns = SeedVR2WeightDefinition.get_download_patterns_for_source(
        ModelConfig.seedvr2_7b_sharp(),
        "ByteDance-Seed/SeedVR2-7B",
    )

    assert patterns == ["seedvr2_ema_7b_sharp.pth", "ema_vae.pth"]


@pytest.mark.fast
def test_seedvr2_weight_definition_selects_official_layout_from_local_files(tmp_path):
    (tmp_path / "seedvr2_ema_3b.pth").touch()

    resolved = SeedVR2WeightDefinition.resolve(ModelConfig.seedvr2_3b(), root_path=tmp_path)
    components = {component.name: component for component in resolved.get_components()}

    assert resolved is SeedVR2WeightDefinition3BOfficial
    assert components["transformer"].loading_mode == "torch_checkpoint"
    assert components["vae"].loading_mode == "torch_checkpoint"
    assert components["text_embedding"].loading_mode == "torch_tensor"


@pytest.mark.fast
def test_seedvr2_weight_definition_selects_official_7b_layout_from_local_files(tmp_path):
    (tmp_path / "seedvr2_ema_7b.pth").touch()

    resolved = SeedVR2WeightDefinition.resolve(ModelConfig.seedvr2_7b(), root_path=tmp_path)
    components = {component.name: component for component in resolved.get_components()}

    assert resolved is SeedVR2WeightDefinition7BOfficial
    assert components["transformer"].num_blocks == 36
    assert components["transformer"].loading_mode == "torch_checkpoint"
    assert components["vae"].loading_mode == "torch_checkpoint"
    assert "text_embedding" not in components


@pytest.mark.fast
def test_seedvr2_weight_definition_selects_official_7b_sharp_layout_from_local_files(tmp_path):
    (tmp_path / "seedvr2_ema_7b_sharp.pth").touch()

    resolved = SeedVR2WeightDefinition.resolve(ModelConfig.seedvr2_7b_sharp(), root_path=tmp_path)
    components = {component.name: component for component in resolved.get_components()}

    assert resolved is SeedVR2WeightDefinition7BOfficialSharp
    assert components["transformer"].weight_files == ["seedvr2_ema_7b_sharp.pth"]
    assert components["vae"].loading_mode == "torch_checkpoint"


@pytest.mark.fast
def test_seedvr2_weight_definition_uses_official_layout_by_default():
    resolved = SeedVR2WeightDefinition.resolve(ModelConfig.seedvr2_3b())

    assert resolved is SeedVR2WeightDefinition3BOfficial


@pytest.mark.fast
def test_seedvr2_weight_definition_uses_official_7b_layout_by_default():
    resolved = SeedVR2WeightDefinition.resolve(ModelConfig.seedvr2_7b())

    assert resolved is SeedVR2WeightDefinition7BOfficial


@pytest.mark.fast
def test_seedvr2_weight_definition_uses_official_7b_sharp_layout_by_default():
    resolved = SeedVR2WeightDefinition.resolve(ModelConfig.seedvr2_7b_sharp())

    assert resolved is SeedVR2WeightDefinition7BOfficialSharp


@pytest.mark.fast
def test_seedvr2_weight_definition_keeps_safetensors_layout_for_explicit_numz_source():
    model_config = SeedVR2Initializer._model_config_for_source(ModelConfig.seedvr2_3b(), "numz/SeedVR2_comfyUI")
    resolved = SeedVR2WeightDefinition.resolve(model_config)
    components = {component.name: component for component in resolved.get_components()}

    assert components["transformer"].weight_files == ["seedvr2_ema_3b_fp16.safetensors"]
    assert components["vae"].weight_files == ["ema_vae_fp16.safetensors"]
    assert "text_embedding" not in components


@pytest.mark.fast
def test_seedvr2_weight_definition_keeps_3b_patterns_for_explicit_numz_source():
    patterns = SeedVR2WeightDefinition.get_download_patterns_for_source(
        ModelConfig.seedvr2_3b(),
        "numz/SeedVR2_comfyUI",
    )

    assert patterns == ["seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors"]


@pytest.mark.fast
def test_seedvr2_weight_definition_selects_prepared_layout_from_local_package(tmp_path):
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    (transformer / "model.safetensors.index.json").write_text("{}")

    resolved = SeedVR2WeightDefinition.resolve(ModelConfig.seedvr2_3b(), root_path=tmp_path)
    patterns = resolved.get_download_patterns()

    assert resolved is SeedVR2WeightDefinition3BPrepared
    assert "transformer/model.safetensors.index.json" in patterns
    assert "vae/model.safetensors.index.json" in patterns


@pytest.mark.fast
def test_seedvr2_weight_definition_selects_prepared_layout_for_abstractframework_package():
    patterns = SeedVR2WeightDefinition.get_download_patterns_for_source(
        ModelConfig.seedvr2_3b(),
        "AbstractFramework/seedvr2-3b-8bit",
    )

    assert "transformer/model.safetensors.index.json" in patterns
    assert "vae/model.safetensors.index.json" in patterns


@pytest.mark.fast
def test_seedvr2_weight_definition_selects_prepared_7b_layout_for_abstractframework_package():
    patterns = SeedVR2WeightDefinition.get_download_patterns_for_source(
        ModelConfig.seedvr2_7b(),
        "AbstractFramework/seedvr2-7b-8bit",
    )
    resolved = SeedVR2WeightDefinition.resolve(ModelConfig.seedvr2_7b(), root_path=None)

    assert "transformer/model.safetensors.index.json" in patterns
    assert "vae/model.safetensors.index.json" in patterns
    assert SeedVR2WeightDefinition.for_saving(ModelConfig.seedvr2_7b()) is SeedVR2WeightDefinition7BPrepared
    assert resolved is SeedVR2WeightDefinition7BOfficial


@pytest.mark.fast
def test_seedvr2_positive_embedding_prepares_official_tensor_shape():
    prepared = SeedVR2TextEmbeddings.prepare_positive(mx.zeros((58, 5120), dtype=mx.float16))

    assert prepared.shape == (1, 58, 5120)


@pytest.mark.fast
def test_seedvr2_runtime_config_records_requested_official_source_without_mutating_default():
    default_config = ModelConfig.seedvr2_3b()

    runtime_config = SeedVR2Initializer._model_config_for_source(default_config, "ByteDance-Seed/SeedVR2-3B")

    assert runtime_config.model_name == "ByteDance-Seed/SeedVR2-3B"
    assert default_config.model_name == "ByteDance-Seed/SeedVR2-3B"


@pytest.mark.fast
def test_seedvr2_weight_coverage_audit_rejects_missing_transformer_key():
    transformer = SeedVR2Transformer(**(ModelConfig.seedvr2_7b().transformer_overrides or {}))
    flat = tree_flatten(transformer.parameters())
    incomplete = tree_unflatten(flat[1:])

    dummy_model = type("DummyModel", (), {"transformer": transformer, "vae": object()})()
    weights = LoadedWeights(
        components={"transformer": incomplete},
        meta_data=MetaData(),
    )

    with pytest.raises(ValueError, match="SeedVR2 transformer weight coverage mismatch"):
        SeedVR2Initializer._assert_weight_coverage(dummy_model, weights)


@pytest.mark.fast
def test_seedvr2_weight_coverage_ignores_quantized_auxiliary_keys():
    transformer = SeedVR2Transformer(**(ModelConfig.seedvr2_3b().transformer_overrides or {}))
    flat = dict(tree_flatten(transformer.parameters()))
    quantized = dict(flat)
    quantized["txt_in.scales"] = mx.ones((1,), dtype=mx.float16)
    quantized["txt_in.biases"] = mx.ones((1,), dtype=mx.float16)

    dummy_model = type("DummyModel", (), {"transformer": transformer, "vae": object()})()
    weights = LoadedWeights(
        components={"transformer": quantized},
        meta_data=MetaData(quantization_level=8),
    )

    SeedVR2Initializer._assert_weight_coverage(dummy_model, weights)


@pytest.mark.fast
def test_seedvr2_estimate_resident_weight_bytes_sums_official_component_files(tmp_path):
    (tmp_path / "seedvr2_ema_3b.pth").write_bytes(b"a" * 11)
    (tmp_path / "ema_vae.pth").write_bytes(b"b" * 7)
    (tmp_path / "pos_emb.pt").write_bytes(b"c" * 5)

    estimated = SeedVR2WeightDefinition.estimate_resident_weight_bytes(
        root_path=tmp_path,
        weight_definition=SeedVR2WeightDefinition3BOfficial,
    )

    assert estimated == 23


@pytest.mark.fast
def test_seedvr2_estimate_resident_weight_bytes_sums_prepared_safetensors(tmp_path):
    transformer = tmp_path / "transformer"
    vae = tmp_path / "vae"
    transformer.mkdir()
    vae.mkdir()
    (transformer / "a.safetensors").write_bytes(b"a" * 13)
    (transformer / "b.safetensors").write_bytes(b"b" * 17)
    (vae / "c.safetensors").write_bytes(b"c" * 19)

    estimated = SeedVR2WeightDefinition.estimate_resident_weight_bytes(
        root_path=tmp_path,
        weight_definition=SeedVR2WeightDefinition3BPrepared,
    )

    assert estimated == 49


@pytest.mark.fast
def test_seedvr2_missing_abstractframework_package_does_not_fallback_to_source(monkeypatch, tmp_path):
    repo_id = "AbstractFramework/seedvr2-7b-8bit"
    model_config = ModelConfig.seedvr2_7b()
    runtime_config = SeedVR2Initializer._model_config_for_source(model_config, repo_id)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hf-cache"))

    with pytest.raises(DownloadRequiredError) as exc_info:
        SeedVR2Initializer._resolve_weight_root(repo_id, runtime_config)

    assert exc_info.value.repo_id == repo_id
    assert "mlxgen download --model AbstractFramework/seedvr2-7b-8bit" in str(exc_info.value)
