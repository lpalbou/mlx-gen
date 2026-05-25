from mflux.utils.version_util import VersionUtil


def test_installed_version_checks_mlx_gen_distribution_first(monkeypatch):
    calls = []

    def fake_version(distribution_name):
        calls.append(distribution_name)
        if distribution_name == "mlx-gen":
            return "1.2.3"
        raise AssertionError("unexpected fallback")

    monkeypatch.setattr("mflux.utils.version_util.importlib.metadata.version", fake_version)

    assert VersionUtil._get_installed_version() == "1.2.3"
    assert calls == ["mlx-gen"]
