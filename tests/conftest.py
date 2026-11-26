"""Shared test configuration.

`MFLUX_FAKE_TOTAL_RAM_GIB` forces the physical-memory figure the runtime reads, so RAM-derived
behaviour can be exercised at a size other than the developer's machine. The MLX cache ladder and the
MiniMax-H3 memory preflight both scale with total memory, so a test that hard-codes a byte figure
passes on a large machine and fails on a smaller CI runner. Run the band under it before pushing:

    MFLUX_FAKE_TOTAL_RAM_GIB=7 uv run pytest -m "not slow and not high_memory_requirement" -q
"""

import os

import pytest


def pytest_configure(config):
    requested = os.environ.get("MFLUX_FAKE_TOTAL_RAM_GIB", "").strip()
    if not requested:
        return
    from mflux.utils.runtime_memory import RuntimeMemory

    total_bytes = int(float(requested) * 1024**3)
    RuntimeMemory.total_physical_memory_bytes = staticmethod(lambda: total_bytes)


@pytest.fixture(scope="session", autouse=True)
def _video_run_lock_of_the_test_session(tmp_path_factory):
    # The SeedVR2/SwiftVR video lock lives under the user's cache folder. Tests use a lock of their
    # own, so they neither block on a real run nor write into the user's cache.
    from mflux.cli.defaults import defaults as ui_defaults

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(ui_defaults, "MFLUX_CACHE_DIR", tmp_path_factory.mktemp("mflux-cache"))
        yield
