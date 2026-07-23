import mlx.core as mx
import pytest

from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.loading.weight_prefetcher import WeightPrefetcher


@pytest.fixture
def weight_files(tmp_path):
    files = []
    for index in range(3):
        file = tmp_path / f"shard_{index}.bin"
        file.write_bytes(bytes([index % 256]) * (256 * 1024 + index))
        files.append(file)
    return files


def test_prefetch_reads_every_file_fully_and_in_order(weight_files, monkeypatch):
    monkeypatch.delenv(WeightPrefetcher.DISABLE_ENV_VAR, raising=False)
    # Force the cold path: freshly written tmp files are page-cache resident.
    monkeypatch.setattr(WeightPrefetcher, "_resident_fraction", staticmethod(lambda path: 0.0))
    read_order = []
    original_read = WeightPrefetcher._read_sequential

    def recording_read(path):
        read_order.append(path)
        return original_read(path)

    monkeypatch.setattr(WeightPrefetcher, "_read_sequential", staticmethod(recording_read))

    read_bytes = WeightPrefetcher.prefetch(weight_files)

    assert read_order == weight_files
    assert read_bytes == sum(file.stat().st_size for file in weight_files)


def test_prefetch_skip_env_var_is_honored(weight_files, monkeypatch):
    monkeypatch.setenv(WeightPrefetcher.DISABLE_ENV_VAR, "1")

    def fail_read(path):
        raise AssertionError("prefetch must not read when disabled")

    monkeypatch.setattr(WeightPrefetcher, "_read_sequential", staticmethod(fail_read))

    assert WeightPrefetcher.prefetch(weight_files) == 0


def test_prefetch_skips_when_files_exceed_ram_headroom(weight_files, monkeypatch):
    monkeypatch.delenv(WeightPrefetcher.DISABLE_ENV_VAR, raising=False)
    total_bytes = sum(file.stat().st_size for file in weight_files)
    # Pretend the machine is small enough that these files blow the headroom gate.
    monkeypatch.setattr(WeightPrefetcher, "total_ram_bytes", staticmethod(lambda: total_bytes))

    assert WeightPrefetcher.prefetch(weight_files) == 0


def test_prefetch_skips_when_ram_size_is_unknown(weight_files, monkeypatch):
    monkeypatch.delenv(WeightPrefetcher.DISABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(WeightPrefetcher, "total_ram_bytes", staticmethod(lambda: 0))

    assert WeightPrefetcher.prefetch(weight_files) == 0


def test_prefetch_skips_resident_files(weight_files, monkeypatch):
    monkeypatch.delenv(WeightPrefetcher.DISABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(WeightPrefetcher, "_resident_fraction", staticmethod(lambda path: 1.0))

    def fail_read(path):
        raise AssertionError("resident files must not be re-read")

    monkeypatch.setattr(WeightPrefetcher, "_read_sequential", staticmethod(fail_read))

    assert WeightPrefetcher.prefetch(weight_files) == 0


def test_prefetch_missing_file_returns_zero_and_lets_loader_fail(tmp_path, monkeypatch):
    monkeypatch.delenv(WeightPrefetcher.DISABLE_ENV_VAR, raising=False)
    assert WeightPrefetcher.prefetch([tmp_path / "missing.safetensors"]) == 0


def test_resident_fraction_reports_fresh_file_as_resident(tmp_path):
    # A file just written through the page cache is resident; the probe must see
    # that so warm reloads skip the re-read. Best-effort kernel behavior, but
    # deterministic in practice for a fresh small file.
    file = tmp_path / "fresh.bin"
    file.write_bytes(b"\xab" * (1024 * 1024))
    file.read_bytes()

    assert WeightPrefetcher._resident_fraction(file) > 0.5


def test_loading_after_prefetch_is_bitwise_identical(tmp_path, monkeypatch):
    monkeypatch.delenv(WeightPrefetcher.DISABLE_ENV_VAR, raising=False)
    tensors = {
        "a.weight": mx.arange(64, dtype=mx.float32).reshape(8, 8),
        "b.bias": mx.ones((16,), dtype=mx.bfloat16),
    }
    file = tmp_path / "model.safetensors"
    mx.save_safetensors(str(file), tensors)

    baseline = mx.load(str(file))
    read_bytes = WeightPrefetcher.prefetch([file])
    prefetched = mx.load(str(file))

    assert read_bytes == file.stat().st_size or read_bytes == 0  # 0 when already resident
    for key, value in baseline.items():
        assert mx.array_equal(value, prefetched[key])
        assert value.dtype == prefetched[key].dtype


def test_weight_loader_skips_prefetch_for_prepared_packages(tmp_path, monkeypatch):
    # Prepared packages (ModelSaver output) are laid out in module-tree order, so
    # they materialize near-sequentially without help; the prefetch measurably
    # REGRESSED cold loads there (0093 cycle-2 ruling: +1.7-2.1 s on Klein 9B q8).
    shard = tmp_path / "0.safetensors"
    mx.save_safetensors(
        str(shard),
        {"layer.weight": mx.zeros((2, 2))},
        {"quantization_level": "8", "mflux_version": "0.0.0-test"},
    )

    def fail_prefetch(paths):
        raise AssertionError("prepared-package loads must not prefetch")

    monkeypatch.setattr(WeightPrefetcher, "prefetch", staticmethod(fail_prefetch))

    weights, quantization_level, version = WeightLoader._try_load_mflux_format(tmp_path)

    assert quantization_level == 8
    assert version == "0.0.0-test"
    assert weights is not None


def test_weight_loader_mlx_native_prefetches_resolved_shards(tmp_path, monkeypatch):
    shard_a = tmp_path / "model-00001.safetensors"
    shard_b = tmp_path / "model-00002.safetensors"
    mx.save_safetensors(str(shard_a), {"w1": mx.zeros((2, 2))})
    mx.save_safetensors(str(shard_b), {"w2": mx.ones((2, 2))})
    prefetched: list = []
    monkeypatch.setattr(
        WeightPrefetcher,
        "prefetch",
        staticmethod(lambda paths: prefetched.extend(paths) or 0),
    )

    weights = WeightLoader._load_mlx_native(tmp_path)

    assert prefetched == [shard_a, shard_b]
    assert set(weights.keys()) == {"w1", "w2"}
