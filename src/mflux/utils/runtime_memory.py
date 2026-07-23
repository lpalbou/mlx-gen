from __future__ import annotations

import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import mlx.core as mx


@dataclass(frozen=True)
class RuntimeMemorySnapshot:
    phase: str
    timestamp: float
    pid: int
    platform: str
    synchronized: bool
    mlx_active_memory_bytes: int | None
    mlx_peak_memory_bytes: int | None
    mlx_cache_memory_bytes: int | None
    process_rss_bytes: int | None
    process_peak_rss_bytes: int | None
    darwin_physical_footprint_bytes: int | None
    errors: dict[str, str]
    darwin_peak_physical_footprint_bytes: int | None = None

    def to_metadata(self) -> dict:
        return asdict(self)


class RuntimeMemory:
    DEFAULT_LOW_RAM_CACHE_LIMIT_BYTES = 1000**3
    # Machine-derived default cap when the caller sets nothing (0094): total RAM / 8,
    # clamped to [1 GiB, 8 GiB]. The 8 GiB ceiling matches the cap BlackPixel's
    # WorkerCachePolicy validated on >=64 GB machines (sized to cover Wan A14B
    # transient buffers); the 1 GiB floor matches the long-standing low-RAM default.
    # Measured disease: an uncapped 2-generation Klein q8 session grew the MLX free
    # cache to 32.9 GB, evicting page cache system-wide (2026-07-23 audit).
    DEFAULT_CACHE_LIMIT_RAM_DIVISOR = 8
    DEFAULT_CACHE_LIMIT_FLOOR_BYTES = 1 * 1024**3
    DEFAULT_CACHE_LIMIT_CEILING_BYTES = 8 * 1024**3
    # Explicit overrides: CLI flag > env var > low-RAM default > machine ladder.
    # A negative value (CLI -1 or env) means "explicitly unlimited" (ADR 0002 opt-out).
    CACHE_LIMIT_ENV_VAR = "MFLUX_MLX_CACHE_LIMIT_GB"
    _cache_limit_state: str = "unset"  # "unset" | "default" | "explicit"

    @staticmethod
    def materialize_tensors(*tensors: mx.array) -> tuple[mx.array, ...]:
        if RuntimeMemory._internal_benchmark_flag_enabled("disable_prompt_materialization"):
            return tensors
        materialized = tuple(mx.stop_gradient(tensor) for tensor in tensors)
        if materialized:
            mx.eval(*materialized)
        return materialized

    @staticmethod
    def materialize_inference_tree(value):
        if RuntimeMemory._internal_benchmark_flag_enabled("disable_prompt_materialization"):
            return value
        materialized = RuntimeMemory._stop_gradient_tree(value)
        arrays = RuntimeMemory._array_leaves(materialized)
        if arrays:
            mx.eval(*arrays)
        return materialized

    @staticmethod
    def _stop_gradient_tree(value):
        if isinstance(value, mx.array):
            return mx.stop_gradient(value)
        if isinstance(value, tuple):
            return tuple(RuntimeMemory._stop_gradient_tree(item) for item in value)
        if isinstance(value, list):
            return [RuntimeMemory._stop_gradient_tree(item) for item in value]
        if isinstance(value, dict):
            return {key: RuntimeMemory._stop_gradient_tree(item) for key, item in value.items()}
        return value

    @staticmethod
    def _array_leaves(value) -> list[mx.array]:
        if isinstance(value, mx.array):
            return [value]
        if isinstance(value, tuple | list):
            return [array for item in value for array in RuntimeMemory._array_leaves(item)]
        if isinstance(value, dict):
            return [array for item in value.values() for array in RuntimeMemory._array_leaves(item)]
        return []

    @staticmethod
    def _internal_benchmark_flag_enabled(flag: str) -> bool:
        if os.environ.get("MFLUX_INTERNAL_MEMORY_BENCHMARK_MODE") != "1":
            return False
        flags = {
            item.strip()
            for item in os.environ.get("MFLUX_INTERNAL_MEMORY_BENCHMARK_FLAGS", "").split(",")
            if item.strip()
        }
        return flag in flags

    @staticmethod
    def apply_mlx_cache_limit(mlx_cache_limit_gb: float | None, *, low_ram: bool = False) -> int | None:
        explicit = mlx_cache_limit_gb is not None or RuntimeMemory._env_cache_limit_gb() is not None or low_ram
        cache_limit_bytes = RuntimeMemory.resolve_cache_limit_bytes(mlx_cache_limit_gb, low_ram=low_ram)
        if cache_limit_bytes is None:
            if explicit:
                # Explicit unlimited opt-out: remember it so the load-time default
                # hook does not re-apply a cap behind the caller's back.
                RuntimeMemory._cache_limit_state = "explicit"
            return None
        if explicit:
            mx.set_cache_limit(cache_limit_bytes)
            mx.clear_cache()
            mx.reset_peak_memory()
            RuntimeMemory._cache_limit_state = "explicit"
            return cache_limit_bytes
        # Default path: a host embedding this process may have already capped the
        # cache directly through mx.set_cache_limit (BlackPixel's WorkerCachePolicy
        # does on 0.24.0). MLX exposes no getter, but set_cache_limit returns the
        # previous limit, so setting the candidate doubles as the probe (0094
        # cycle-2 fix: the default must never stomp a deliberate host cap).
        pre_existing = mx.set_cache_limit(cache_limit_bytes)
        host_limit = RuntimeMemory._deliberate_host_limit(pre_existing)
        if host_limit is not None:
            mx.set_cache_limit(host_limit)
            RuntimeMemory._cache_limit_state = "explicit"
            print(
                f"Respecting pre-existing MLX cache limit: {host_limit / 1024**3:.1f} GiB "
                "(host-managed); no machine-derived default applied.",
                file=sys.stderr,
            )
            return None
        mx.clear_cache()
        mx.reset_peak_memory()
        RuntimeMemory._cache_limit_state = "default"
        # ADR 0002: a new default must be visible, not silent. stderr keeps the
        # --json-events stdout stream machine-readable.
        print(
            f"Applying default MLX cache limit: {cache_limit_bytes / 1024**3:.1f} GiB "
            f"(total RAM / {RuntimeMemory.DEFAULT_CACHE_LIMIT_RAM_DIVISOR}, clamped to "
            f"[{RuntimeMemory.DEFAULT_CACHE_LIMIT_FLOOR_BYTES // 1024**3}, "
            f"{RuntimeMemory.DEFAULT_CACHE_LIMIT_CEILING_BYTES // 1024**3}] GiB). "
            f"Override with --mlx-cache-limit-gb or {RuntimeMemory.CACHE_LIMIT_ENV_VAR}; -1 disables the cap.",
            file=sys.stderr,
        )
        return cache_limit_bytes

    @staticmethod
    def _deliberate_host_limit(pre_existing_limit_bytes: object) -> int | None:
        # MLX's own untouched process default is RAM-scale (measured ~0.95x total
        # RAM on 0.31.0), while deliberate host caps are single-digit GiB, so
        # "at or below half of physical RAM" separates the two. Non-int values
        # (older MLX, test stubs) leave the probe inconclusive and the default
        # proceeds — the pre-fix behavior.
        if not isinstance(pre_existing_limit_bytes, int):
            return None
        total_ram = RuntimeMemory.total_physical_memory_bytes()
        if total_ram > 0:
            threshold = total_ram // 2
        else:
            threshold = 2 * RuntimeMemory.DEFAULT_CACHE_LIMIT_CEILING_BYTES
        if 0 <= pre_existing_limit_bytes <= threshold:
            return pre_existing_limit_bytes
        return None

    @staticmethod
    def apply_default_cache_limit_once() -> int | None:
        # Idempotent hook for bare Python-API entry points (weight loading): applies
        # the machine-derived default exactly once per process and never overrides a
        # limit already applied through apply_mlx_cache_limit (CLI/env/low-RAM).
        if RuntimeMemory._cache_limit_state != "unset":
            return None
        return RuntimeMemory.apply_mlx_cache_limit(None)

    @staticmethod
    def resolve_cache_limit_bytes(
        mlx_cache_limit_gb: float | None,
        *,
        low_ram: bool = False,
        total_ram_bytes: int | None = None,
    ) -> int | None:
        if mlx_cache_limit_gb is None:
            mlx_cache_limit_gb = RuntimeMemory._env_cache_limit_gb()
        if mlx_cache_limit_gb is not None:
            if mlx_cache_limit_gb < 0:
                return None  # explicitly unlimited
            return int(mlx_cache_limit_gb * (1000**3))
        if low_ram:
            return RuntimeMemory.DEFAULT_LOW_RAM_CACHE_LIMIT_BYTES
        return RuntimeMemory.default_cache_limit_bytes(total_ram_bytes=total_ram_bytes)

    @staticmethod
    def default_cache_limit_bytes(total_ram_bytes: int | None = None) -> int:
        if total_ram_bytes is None:
            total_ram_bytes = RuntimeMemory.total_physical_memory_bytes()
        if total_ram_bytes <= 0:
            # Unknown machine size: the conservative floor, mirroring the
            # "unknown counts as small" convention of the BlackPixel precedent.
            return RuntimeMemory.DEFAULT_CACHE_LIMIT_FLOOR_BYTES
        fraction = total_ram_bytes // RuntimeMemory.DEFAULT_CACHE_LIMIT_RAM_DIVISOR
        return max(
            RuntimeMemory.DEFAULT_CACHE_LIMIT_FLOOR_BYTES,
            min(RuntimeMemory.DEFAULT_CACHE_LIMIT_CEILING_BYTES, fraction),
        )

    @staticmethod
    def total_physical_memory_bytes() -> int:
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError, AttributeError):
            return 0

    @staticmethod
    def _env_cache_limit_gb() -> float | None:
        raw = os.environ.get(RuntimeMemory.CACHE_LIMIT_ENV_VAR, "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{RuntimeMemory.CACHE_LIMIT_ENV_VAR} must be a number (GB) or -1 for unlimited, got {raw!r}."
            ) from exc

    @staticmethod
    def snapshot(
        phase: str,
        *,
        tensors: tuple[mx.array, ...] = (),
        synchronize: bool = False,
    ) -> RuntimeMemorySnapshot:
        if os.environ.get("MFLUX_RUNTIME_MEMORY_TELEMETRY") == "0":
            return RuntimeMemorySnapshot(
                phase=phase,
                timestamp=time.time(),
                pid=os.getpid(),
                platform=platform.platform(),
                synchronized=False,
                mlx_active_memory_bytes=None,
                mlx_peak_memory_bytes=None,
                mlx_cache_memory_bytes=None,
                process_rss_bytes=None,
                process_peak_rss_bytes=None,
                darwin_physical_footprint_bytes=None,
                errors={"runtime_memory": "disabled"},
            )
        errors: dict[str, str] = {}
        if tensors:
            try:
                mx.eval(*tensors)
            except Exception as exc:  # noqa: BLE001
                errors["mlx_eval"] = f"{exc.__class__.__name__}: {exc}"
        if synchronize:
            try:
                mx.synchronize()
            except Exception as exc:  # noqa: BLE001
                errors["mlx_synchronize"] = f"{exc.__class__.__name__}: {exc}"

        footprint = RuntimeMemory._darwin_footprint_values(errors)
        return RuntimeMemorySnapshot(
            phase=phase,
            timestamp=time.time(),
            pid=os.getpid(),
            platform=platform.platform(),
            synchronized=synchronize,
            mlx_active_memory_bytes=RuntimeMemory._mlx_memory("get_active_memory", errors),
            mlx_peak_memory_bytes=RuntimeMemory._mlx_memory("get_peak_memory", errors),
            mlx_cache_memory_bytes=RuntimeMemory._mlx_memory("get_cache_memory", errors),
            process_rss_bytes=RuntimeMemory._process_rss_bytes(errors),
            process_peak_rss_bytes=RuntimeMemory._process_peak_rss_bytes(errors),
            darwin_physical_footprint_bytes=footprint[0] if footprint is not None else None,
            errors=errors,
            darwin_peak_physical_footprint_bytes=footprint[1] if footprint is not None else None,
        )

    @staticmethod
    def _mlx_memory(name: str, errors: dict[str, str]) -> int | None:
        try:
            return int(getattr(mx, name)())
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            errors[f"mlx_{name}"] = f"{exc.__class__.__name__}: {exc}"
            return None

    @staticmethod
    def _process_rss_bytes(errors: dict[str, str]) -> int | None:
        proc_rss = RuntimeMemory._linux_proc_rss_bytes(errors)
        if proc_rss is not None:
            return proc_rss
        return RuntimeMemory._ps_rss_bytes(errors)

    @staticmethod
    def _linux_proc_rss_bytes(errors: dict[str, str]) -> int | None:
        statm_path = "/proc/self/statm"
        if not os.path.exists(statm_path):
            return None
        try:
            with open(statm_path) as statm:
                resident_pages = int(statm.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, TypeError, ValueError) as exc:
            errors["process_rss_proc"] = f"{exc.__class__.__name__}: {exc}"
            return None

    @staticmethod
    def _ps_rss_bytes(errors: dict[str, str]) -> int | None:
        try:
            output = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return int(output.strip()) * 1024
        except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
            errors["process_rss_ps"] = f"{exc.__class__.__name__}: {exc}"
            return None

    @staticmethod
    def _process_peak_rss_bytes(errors: dict[str, str]) -> int | None:
        try:
            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except (OSError, TypeError, ValueError) as exc:
            errors["process_peak_rss"] = f"{exc.__class__.__name__}: {exc}"
            return None
        if sys.platform == "darwin":
            return peak
        return peak * 1024

    @staticmethod
    def _darwin_footprint_values(errors: dict[str, str]) -> tuple[int, int] | None:
        if sys.platform != "darwin":
            return None
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    RuntimeMemory._DARWIN_PHYSICAL_FOOTPRINT_HELPER,
                    str(os.getpid()),
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError, TimeoutError, TypeError, ValueError) as exc:
            errors["darwin_physical_footprint"] = f"{exc.__class__.__name__}: {exc}"
            return None
        if result.returncode != 0:
            errors["darwin_physical_footprint"] = result.stderr.strip() or f"helper returned {result.returncode}"
            return None
        try:
            current_text, peak_text = result.stdout.split()
            return (int(current_text), int(peak_text))
        except ValueError as exc:
            errors["darwin_physical_footprint"] = f"{exc.__class__.__name__}: {result.stdout!r}"
            return None

    _DARWIN_PHYSICAL_FOOTPRINT_HELPER = r"""
import ctypes
import sys


class RUsageInfoV4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


info = RUsageInfoV4()
rc = ctypes.CDLL("libproc.dylib").proc_pid_rusage(int(sys.argv[1]), 4, ctypes.byref(info))
if rc != 0:
    raise SystemExit(rc)
print(int(info.ri_phys_footprint), int(info.ri_lifetime_max_phys_footprint))
"""
