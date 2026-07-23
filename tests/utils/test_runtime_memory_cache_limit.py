import argparse

import pytest

from mflux.cli.parser.parsers import cache_limit_gb_value
from mflux.utils.runtime_memory import RuntimeMemory

GIB = 1024**3


@pytest.fixture(autouse=True)
def clean_cache_limit_state(monkeypatch):
    monkeypatch.delenv(RuntimeMemory.CACHE_LIMIT_ENV_VAR, raising=False)
    original_state = RuntimeMemory._cache_limit_state
    RuntimeMemory._cache_limit_state = "unset"
    yield
    RuntimeMemory._cache_limit_state = original_state


class TestDefaultCacheLimitLadder:
    def test_ram_divided_by_eight_between_floor_and_ceiling(self):
        assert RuntimeMemory.default_cache_limit_bytes(total_ram_bytes=16 * GIB) == 2 * GIB
        assert RuntimeMemory.default_cache_limit_bytes(total_ram_bytes=32 * GIB) == 4 * GIB
        assert RuntimeMemory.default_cache_limit_bytes(total_ram_bytes=64 * GIB) == 8 * GIB

    def test_ceiling_matches_blackpixel_precedent_on_big_machines(self):
        assert RuntimeMemory.default_cache_limit_bytes(total_ram_bytes=128 * GIB) == 8 * GIB
        assert RuntimeMemory.default_cache_limit_bytes(total_ram_bytes=512 * GIB) == 8 * GIB

    def test_floor_on_small_machines(self):
        assert RuntimeMemory.default_cache_limit_bytes(total_ram_bytes=4 * GIB) == 1 * GIB

    def test_unknown_ram_size_uses_conservative_floor(self):
        assert RuntimeMemory.default_cache_limit_bytes(total_ram_bytes=0) == 1 * GIB
        assert RuntimeMemory.default_cache_limit_bytes(total_ram_bytes=-5) == 1 * GIB


class TestResolveCacheLimitBytes:
    def test_explicit_argument_wins(self, monkeypatch):
        monkeypatch.setenv(RuntimeMemory.CACHE_LIMIT_ENV_VAR, "2")
        assert RuntimeMemory.resolve_cache_limit_bytes(4.0) == 4 * 1000**3

    def test_env_var_wins_over_low_ram_and_ladder(self, monkeypatch):
        monkeypatch.setenv(RuntimeMemory.CACHE_LIMIT_ENV_VAR, "2")
        assert RuntimeMemory.resolve_cache_limit_bytes(None, low_ram=True) == 2 * 1000**3
        assert RuntimeMemory.resolve_cache_limit_bytes(None) == 2 * 1000**3

    def test_low_ram_default_preserved(self):
        assert (
            RuntimeMemory.resolve_cache_limit_bytes(None, low_ram=True)
            == RuntimeMemory.DEFAULT_LOW_RAM_CACHE_LIMIT_BYTES
        )

    def test_unset_falls_back_to_machine_ladder(self):
        assert RuntimeMemory.resolve_cache_limit_bytes(None, total_ram_bytes=32 * GIB) == 4 * GIB

    def test_negative_argument_means_unlimited(self):
        assert RuntimeMemory.resolve_cache_limit_bytes(-1.0) is None

    def test_negative_env_means_unlimited(self, monkeypatch):
        monkeypatch.setenv(RuntimeMemory.CACHE_LIMIT_ENV_VAR, "-1")
        assert RuntimeMemory.resolve_cache_limit_bytes(None) is None

    def test_invalid_env_fails_loud(self, monkeypatch):
        monkeypatch.setenv(RuntimeMemory.CACHE_LIMIT_ENV_VAR, "lots")
        with pytest.raises(ValueError, match="MFLUX_MLX_CACHE_LIMIT_GB"):
            RuntimeMemory.resolve_cache_limit_bytes(None)


class TestApplyDefaultCacheLimitOnce:
    @pytest.fixture
    def applied_limits(self, monkeypatch):
        applied: list[int] = []
        monkeypatch.setattr("mflux.utils.runtime_memory.mx.set_cache_limit", applied.append)
        monkeypatch.setattr("mflux.utils.runtime_memory.mx.clear_cache", lambda: None)
        monkeypatch.setattr("mflux.utils.runtime_memory.mx.reset_peak_memory", lambda: None)
        return applied

    def test_applies_machine_default_exactly_once(self, applied_limits):
        first = RuntimeMemory.apply_default_cache_limit_once()
        second = RuntimeMemory.apply_default_cache_limit_once()

        assert first == RuntimeMemory.default_cache_limit_bytes()
        assert second is None
        assert applied_limits == [first]
        assert RuntimeMemory._cache_limit_state == "default"

    def test_noop_after_explicit_limit(self, applied_limits):
        explicit = RuntimeMemory.apply_mlx_cache_limit(2.0)

        assert RuntimeMemory.apply_default_cache_limit_once() is None
        assert applied_limits == [explicit]
        assert RuntimeMemory._cache_limit_state == "explicit"

    def test_noop_after_explicit_unlimited_opt_out(self, applied_limits):
        assert RuntimeMemory.apply_mlx_cache_limit(-1.0) is None
        assert RuntimeMemory.apply_default_cache_limit_once() is None
        assert applied_limits == []
        assert RuntimeMemory._cache_limit_state == "explicit"

    def test_low_ram_counts_as_explicit(self, applied_limits):
        RuntimeMemory.apply_mlx_cache_limit(None, low_ram=True)

        assert RuntimeMemory.apply_default_cache_limit_once() is None
        assert applied_limits == [RuntimeMemory.DEFAULT_LOW_RAM_CACHE_LIMIT_BYTES]

    def test_default_application_prints_visible_notice(self, applied_limits, capsys):
        RuntimeMemory.apply_mlx_cache_limit(None)

        captured = capsys.readouterr()
        assert "Applying default MLX cache limit" in captured.err
        assert "--mlx-cache-limit-gb" in captured.err

    def test_explicit_application_stays_silent(self, applied_limits, capsys):
        RuntimeMemory.apply_mlx_cache_limit(2.0)

        captured = capsys.readouterr()
        assert captured.err == ""


class _FakeMlxCacheLimit:
    # Mirrors real MLX semantics: set_cache_limit stores the new value and
    # returns the previous one (MLX exposes no getter).
    def __init__(self, initial_bytes: int):
        self.current = initial_bytes
        self.calls: list[int] = []

    def __call__(self, value: int) -> int:
        previous = self.current
        self.current = value
        self.calls.append(value)
        return previous


class TestDefaultRespectsHostManagedLimit:
    # BlackPixel's WorkerCachePolicy (0.24.0) calls mx.set_cache_limit directly
    # before the first model load; the load-time default must not stomp it (0094
    # cycle-2 fix, demonstrated live: a 2 GiB host cap was silently raised to the
    # 8 GiB ladder value before this fix).
    HOST_CAP = 1 * GIB

    @pytest.fixture
    def fake_limit(self, monkeypatch):
        fake = _FakeMlxCacheLimit(initial_bytes=self.HOST_CAP)
        monkeypatch.setattr("mflux.utils.runtime_memory.mx.set_cache_limit", fake)
        monkeypatch.setattr("mflux.utils.runtime_memory.mx.clear_cache", lambda: None)
        monkeypatch.setattr("mflux.utils.runtime_memory.mx.reset_peak_memory", lambda: None)
        return fake

    def test_load_time_default_preserves_pre_existing_host_cap(self, fake_limit, capsys):
        applied = RuntimeMemory.apply_default_cache_limit_once()

        assert applied is None
        assert fake_limit.current == self.HOST_CAP  # probe restored the host value
        assert RuntimeMemory._cache_limit_state == "explicit"
        assert "Respecting pre-existing MLX cache limit" in capsys.readouterr().err

    def test_host_cap_detection_latches_no_further_defaults(self, fake_limit):
        RuntimeMemory.apply_default_cache_limit_once()
        calls_after_first = list(fake_limit.calls)

        assert RuntimeMemory.apply_default_cache_limit_once() is None
        assert fake_limit.calls == calls_after_first

    def test_untouched_mlx_default_still_gets_the_ladder(self, monkeypatch, capsys):
        # An untouched MLX process default is RAM-scale (~0.95x total RAM measured
        # on 0.31.0) — far above the half-RAM deliberate-cap threshold.
        fake = _FakeMlxCacheLimit(initial_bytes=RuntimeMemory.total_physical_memory_bytes())
        monkeypatch.setattr("mflux.utils.runtime_memory.mx.set_cache_limit", fake)
        monkeypatch.setattr("mflux.utils.runtime_memory.mx.clear_cache", lambda: None)
        monkeypatch.setattr("mflux.utils.runtime_memory.mx.reset_peak_memory", lambda: None)

        applied = RuntimeMemory.apply_default_cache_limit_once()

        assert applied == RuntimeMemory.default_cache_limit_bytes()
        assert fake.current == applied
        assert "Applying default MLX cache limit" in capsys.readouterr().err

    def test_explicit_flag_still_overrides_host_cap(self, fake_limit):
        applied = RuntimeMemory.apply_mlx_cache_limit(4.0)

        assert applied == 4 * 1000**3
        assert fake_limit.current == applied  # explicit callers always win
        assert RuntimeMemory._cache_limit_state == "explicit"


class TestCacheLimitCliValue:
    def test_positive_values_accepted(self):
        assert cache_limit_gb_value("8") == 8.0
        assert cache_limit_gb_value("2.5") == 2.5

    def test_minus_one_is_the_unlimited_opt_out(self):
        assert cache_limit_gb_value("-1") == -1.0

    def test_zero_and_other_negatives_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            cache_limit_gb_value("0")
        with pytest.raises(argparse.ArgumentTypeError):
            cache_limit_gb_value("-2")

    def test_non_numeric_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            cache_limit_gb_value("unlimited")
