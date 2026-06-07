import pytest

from mflux.models.common.download_policy import DownloadRequiredError, explicit_download_hint


@pytest.mark.fast
def test_seedvr2_download_hint_suggests_prepare():
    hint = explicit_download_hint("ByteDance-Seed/SeedVR2-3B")

    assert "mlxgen download --model ByteDance-Seed/SeedVR2-3B" in hint
    assert "mlxgen prepare --model ByteDance-Seed/SeedVR2-3B" in hint


@pytest.mark.fast
def test_seedvr2_download_required_has_prepare_command():
    error = DownloadRequiredError("ByteDance-Seed/SeedVR2-3B")

    assert error.prepare_command == "mlxgen prepare --model ByteDance-Seed/SeedVR2-3B --path ./models/seedvr2-3b -q 8"
