"""Shared test configuration.

`MFLUX_FAKE_TOTAL_RAM_GIB` forces the physical-memory figure the runtime reads, so RAM-derived
behaviour can be exercised at a size other than the developer's machine. The MLX cache ladder and the
MiniMax-H3 memory preflight both scale with total memory, so a test that hard-codes a byte figure
passes on a large machine and fails on a smaller CI runner. Run the band under it before pushing:

    MFLUX_FAKE_TOTAL_RAM_GIB=7 uv run pytest -m "not slow and not high_memory_requirement" -q
"""

import os


def pytest_configure(config):
    requested = os.environ.get("MFLUX_FAKE_TOTAL_RAM_GIB", "").strip()
    if not requested:
        return
    from mflux.utils.runtime_memory import RuntimeMemory

    total_bytes = int(float(requested) * 1024**3)
    RuntimeMemory.total_physical_memory_bytes = staticmethod(lambda: total_bytes)
