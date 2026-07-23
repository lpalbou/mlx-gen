import ctypes
import logging
import mmap
import os
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


class WeightPrefetcher:
    # mx.load maps safetensors lazily, so page-cold weights fault in at random-access
    # speed (measured 120-320 MB/s) during first compute instead of the SSD's
    # multi-GB/s sequential rate. One bounded sequential read per weight file moves
    # those bytes into the page cache up front, where the later faults hit RAM (0093).
    DISABLE_ENV_VAR = "MFLUX_NO_WEIGHT_PREFETCH"
    # Never read more than half of physical RAM: beyond that the read-ahead starts
    # evicting pages it just warmed (or other processes' working set), which is the
    # exact failure mode low-RAM machines need protection from.
    RAM_HEADROOM_FRACTION = 0.5
    # Files that are already (almost) resident gain nothing from a re-read; skipping
    # them keeps warm reloads at ~zero overhead.
    RESIDENT_SKIP_FRACTION = 0.97
    CHUNK_BYTES = 64 * 1024 * 1024

    @staticmethod
    def prefetch(paths: Sequence[Path | str]) -> int:
        if WeightPrefetcher.is_disabled():
            return 0
        files = [Path(path) for path in paths]
        try:
            total_bytes = sum(file.stat().st_size for file in files)
        except OSError:
            # Missing/unreadable file: let the actual loader raise its own error.
            return 0
        if total_bytes <= 0:
            return 0
        total_ram = WeightPrefetcher.total_ram_bytes()
        if total_ram <= 0 or total_bytes > total_ram * WeightPrefetcher.RAM_HEADROOM_FRACTION:
            logger.debug(
                "Skipping weight prefetch: %d bytes exceed the RAM headroom gate (total RAM %d bytes).",
                total_bytes,
                total_ram,
            )
            return 0
        read_bytes = 0
        for file in files:
            read_bytes += WeightPrefetcher._prefetch_file(file)
        return read_bytes

    @staticmethod
    def is_disabled() -> bool:
        return os.environ.get(WeightPrefetcher.DISABLE_ENV_VAR, "").strip().lower() not in ("", "0", "false")

    @staticmethod
    def total_ram_bytes() -> int:
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError, AttributeError):
            # Unknown machine size: the conservative answer is to skip prefetching.
            return 0

    @staticmethod
    def _prefetch_file(path: Path) -> int:
        if WeightPrefetcher._resident_fraction(path) >= WeightPrefetcher.RESIDENT_SKIP_FRACTION:
            return 0
        return WeightPrefetcher._read_sequential(path)

    @staticmethod
    def _read_sequential(path: Path) -> int:
        buffer = bytearray(WeightPrefetcher.CHUNK_BYTES)
        view = memoryview(buffer)
        read_total = 0
        # buffering=0 gives raw sequential reads straight into our reusable buffer;
        # the bytes land in the kernel page cache, which is the whole point.
        with path.open("rb", buffering=0) as handle:
            while True:
                read_count = handle.readinto(view)
                if not read_count:
                    return read_total
                read_total += read_count

    @staticmethod
    def _resident_fraction(path: Path) -> float:
        # Residency probe via mincore(2): 0.0 ("cold") on any failure so the
        # prefetch proceeds — a wasted page-cache-speed re-read is the safe fallback.
        try:
            size = path.stat().st_size
            if size <= 0:
                return 1.0
            with path.open("rb") as handle:
                return WeightPrefetcher._mincore_resident_fraction(handle.fileno(), size)
        except (OSError, ValueError, AttributeError, ctypes.ArgumentError):
            return 0.0

    @staticmethod
    def _mincore_resident_fraction(fileno: int, size: int) -> float:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
        ]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

        address = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fileno, 0)
        if address is None or address == ctypes.c_void_p(-1).value:
            return 0.0
        try:
            page_size = mmap.PAGESIZE
            page_count = (size + page_size - 1) // page_size
            residency = (ctypes.c_ubyte * page_count)()
            if libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(size), residency) != 0:
                return 0.0
            resident_pages = sum(1 for entry in residency if entry & 1)
            return resident_pages / page_count
        finally:
            libc.munmap(ctypes.c_void_p(address), ctypes.c_size_t(size))
