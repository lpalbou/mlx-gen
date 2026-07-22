from pathlib import Path

import pytest
from huggingface_hub.utils import LocalEntryNotFoundError

from mflux.models.common.download_policy import allow_downloads
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader


@pytest.mark.fast
def test_load_single_requires_explicit_download_when_not_cached(monkeypatch):
    def fake_snapshot_download(**_kwargs):
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    with pytest.raises(FileNotFoundError) as exc_info:
        WeightLoader.load_single(ComponentDefinition(name="transformer", hf_subdir="transformer"), "org/model")

    error = str(exc_info.value)
    assert "MLX-Gen will not download model files during generation" in error
    assert "mlxgen download --model org/model" in error
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in error


@pytest.mark.fast
def test_load_single_downloads_when_explicitly_enabled(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("not cached")
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(WeightLoader, "_load_component", lambda root, component: ({}, None, None))

    with allow_downloads():
        result = WeightLoader.load_single(ComponentDefinition(name="transformer", hf_subdir="transformer"), "org/model")

    assert Path(calls[1]["repo_id"]).as_posix() == "org/model"
    assert calls[0]["local_files_only"] is True
    assert "local_files_only" not in calls[1]
    assert result.components == {"transformer": {}}


@pytest.mark.fast
def test_download_url_component_requires_explicit_download(monkeypatch, tmp_path):
    monkeypatch.setattr("mflux.models.common.weights.loading.weight_loader.MFLUX_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        "mflux.models.common.weights.loading.weight_loader.urllib.request.urlretrieve", lambda *_args: None
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        WeightLoader._download_from_url("https://example.com/depth_pro.pt", "depth_pro")

    error = str(exc_info.value)
    assert "MLX-Gen will not download Depth Pro weights during generation" in error
    assert "mlxgen download --model depth-pro" in error


@pytest.mark.fast
def test_download_url_component_downloads_when_explicitly_enabled(monkeypatch, tmp_path):
    calls = []

    def fake_urlretrieve(url, file_path):
        calls.append((url, file_path))
        Path(file_path).touch()

    monkeypatch.setattr("mflux.models.common.weights.loading.weight_loader.MFLUX_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        "mflux.models.common.weights.loading.weight_loader.urllib.request.urlretrieve", fake_urlretrieve
    )

    with allow_downloads():
        path = WeightLoader._download_from_url("https://example.com/depth_pro.pt", "depth_pro")

    assert calls == [("https://example.com/depth_pro.pt", tmp_path / "depth_pro" / "depth_pro.pt")]
    assert path == tmp_path / "depth_pro" / "depth_pro.pt"
