import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generation_memory_benchmark import GenerationMemoryBenchmark, ProcessTreeSampler

BYTES_PER_GB = 1000**3


@dataclass(frozen=True)
class BatchingCase:
    case_id: str
    kind: str
    argv_base: list[str]
    output_suffix: str
    seeds: tuple[int, ...]

    def output_path(self, base_dir: Path, seed: int) -> Path:
        return base_dir / f"{self.case_id}_seed_{seed}{self.output_suffix}"

    def output_template(self, base_dir: Path) -> str:
        return str(base_dir / f"{self.case_id}_seed_{{seed}}{self.output_suffix}")


SOURCE_IMAGE = "docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png"
INPAINT_MASK = "validation_outputs/qwen_inpaint_2026_06_15/masks/source01_engine_mask.png"
I2V_SOURCE = "docs/assets/i2v_takeoff_source.png"

COMMON_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "MFLUX_RUNTIME_MEMORY_TELEMETRY": "1",
    "MFLUX_BENCHMARK_PARENT_PHYSICAL_SAMPLING": "1",
}

CASES = {
    "zimage_t2i": BatchingCase(
        case_id="zimage_t2i",
        kind="image",
        argv_base=[
            "uv",
            "run",
            "mlxgen",
            "generate",
            "--model",
            "AbstractFramework/z-image-turbo-8bit",
            "--prompt",
            "A red enamel kettle on a steel counter, product photo",
            "--width",
            "384",
            "--height",
            "384",
            "--steps",
            "2",
            "--guidance",
            "1",
            "--metadata",
            "--replace",
            "--no-progress",
        ],
        output_suffix=".png",
        seeds=(7301, 7302),
    ),
    "zimage_i2i_inpaint": BatchingCase(
        case_id="zimage_i2i_inpaint",
        kind="image",
        argv_base=[
            "uv",
            "run",
            "mlxgen",
            "generate",
            "--model",
            "AbstractFramework/z-image-turbo-8bit",
            "--image",
            SOURCE_IMAGE,
            "--mask-path",
            INPAINT_MASK,
            "--prompt",
            "Keep the same silver spaceship, icy canyon, and sunrise lighting. "
            "Only inside the masked engine area, intensify both blue engines into brighter plasma thrusters.",
            "--width",
            "384",
            "--height",
            "216",
            "--steps",
            "2",
            "--guidance",
            "1",
            "--metadata",
            "--replace",
            "--no-progress",
        ],
        output_suffix=".png",
        seeds=(7311, 7312),
    ),
    "wan_t2v": BatchingCase(
        case_id="wan_t2v",
        kind="video",
        argv_base=[
            "uv",
            "run",
            "mlxgen",
            "generate",
            "--model",
            "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit",
            "--prompt",
            "A tiny clockwork robot walks across a wooden desk, stable camera",
            "--width",
            "320",
            "--height",
            "192",
            "--frames",
            "9",
            "--fps",
            "8",
            "--steps",
            "1",
            "--guidance",
            "1",
            "--flow-shift",
            "3",
            "--metadata",
            "--replace",
            "--no-progress",
        ],
        output_suffix=".mp4",
        seeds=(7321, 7322),
    ),
    "wan_i2v": BatchingCase(
        case_id="wan_i2v",
        kind="video",
        argv_base=[
            "uv",
            "run",
            "mlxgen",
            "generate",
            "--model",
            "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit",
            "--image",
            I2V_SOURCE,
            "--prompt",
            "Cinematic sequence of the spacecraft lifting off from the snowy landing field, engines glowing.",
            "--width",
            "320",
            "--height",
            "192",
            "--frames",
            "9",
            "--fps",
            "8",
            "--steps",
            "1",
            "--guidance",
            "1",
            "--flow-shift",
            "3",
            "--metadata",
            "--replace",
            "--no-progress",
        ],
        output_suffix=".mp4",
        seeds=(7331, 7332),
    ),
}


class MultiOutputBatchBenchmark:
    @staticmethod
    def main() -> None:
        args = MultiOutputBatchBenchmark._parse_args()
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        cases = [CASES[case_id] for case_id in args.cases]
        report = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "schema_version": 1,
            "environment": GenerationMemoryBenchmark._environment(),
            "cases": {},
        }
        for case in cases:
            report["cases"][case.case_id] = MultiOutputBatchBenchmark._run_case(
                case=case,
                output_dir=output_dir / case.case_id,
                sample_interval_ms=args.sample_interval_ms,
            )
        report_path = output_dir / "batching_benchmark_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(MultiOutputBatchBenchmark._compact_report(report), indent=2, sort_keys=True))
        print(f"wrote {report_path}")

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("validation_outputs/batching/batch_efficiency_20260630"),
        )
        parser.add_argument("--sample-interval-ms", type=int, default=200)
        parser.add_argument("--cases", nargs="*", choices=sorted(CASES), default=sorted(CASES))
        args = parser.parse_args()
        if args.sample_interval_ms <= 0:
            raise ValueError("--sample-interval-ms must be greater than zero.")
        return args

    @staticmethod
    def _run_case(*, case: BatchingCase, output_dir: Path, sample_interval_ms: int) -> dict[str, Any]:
        serial_dir = output_dir / "serial"
        batched_dir = output_dir / "batched"
        serial_dir.mkdir(parents=True, exist_ok=True)
        batched_dir.mkdir(parents=True, exist_ok=True)
        serial_commands = []
        for seed in case.seeds:
            argv = [
                *case.argv_base,
                "--seed",
                str(seed),
                "--output",
                str(case.output_path(serial_dir, seed)),
            ]
            serial_commands.append(
                MultiOutputBatchBenchmark._run_command(
                    label=f"serial_seed_{seed}",
                    argv=argv,
                    output_dir=serial_dir / f"seed_{seed}",
                    sample_interval_ms=sample_interval_ms,
                )
            )
        batched_command = MultiOutputBatchBenchmark._run_command(
            label="batched",
            argv=[
                *case.argv_base,
                "--seed",
                *[str(seed) for seed in case.seeds],
                "--output",
                case.output_template(batched_dir),
            ],
            output_dir=batched_dir / "batched",
            sample_interval_ms=sample_interval_ms,
        )
        serial_artifacts = MultiOutputBatchBenchmark._collect_artifacts(case=case, output_dir=serial_dir)
        batched_artifacts = MultiOutputBatchBenchmark._collect_artifacts(case=case, output_dir=batched_dir)
        return {
            "case_id": case.case_id,
            "kind": case.kind,
            "serial": {
                "commands": serial_commands,
                "summary": MultiOutputBatchBenchmark._summarize_variant(serial_commands, serial_artifacts),
                "artifacts": serial_artifacts,
            },
            "batched": {
                "commands": [batched_command],
                "summary": MultiOutputBatchBenchmark._summarize_variant([batched_command], batched_artifacts),
                "artifacts": batched_artifacts,
            },
            "comparison": MultiOutputBatchBenchmark._compare_case(
                case=case,
                serial_artifacts=serial_artifacts,
                batched_artifacts=batched_artifacts,
                serial_commands=serial_commands,
                batched_command=batched_command,
            ),
        }

    @staticmethod
    def _run_command(*, label: str, argv: list[str], output_dir: Path, sample_interval_ms: int) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        env = os.environ.copy()
        env.update(COMMON_ENV)
        timed_argv = ["/usr/bin/time", "-l", *argv]
        started = time.perf_counter()
        process = subprocess.Popen(
            timed_argv,
            cwd=Path.cwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with ProcessTreeSampler(process.pid, sample_interval_ms / 1000.0) as sampler:
            stdout, stderr = process.communicate()
        wall_seconds = time.perf_counter() - started
        stdout_path.write_text(stdout)
        stderr_path.write_text(stderr)
        if process.returncode != 0:
            raise RuntimeError(f"{label} failed; see {stderr_path}")
        return {
            "label": label,
            "argv": timed_argv,
            "returncode": process.returncode,
            "wall_seconds": wall_seconds,
            "sampler": sampler.summary(),
            "samples": sampler.samples,
            "time_l": GenerationMemoryBenchmark._parse_time_l_output(stderr),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }

    @staticmethod
    def _collect_artifacts(*, case: BatchingCase, output_dir: Path) -> list[dict[str, Any]]:
        artifacts = []
        for seed in case.seeds:
            output_path = case.output_path(output_dir, seed)
            artifacts.append(
                {
                    "seed": seed,
                    "output_path": str(output_path),
                    "exists": output_path.exists(),
                    "sha256": GenerationMemoryBenchmark._sha256(output_path) if output_path.exists() else None,
                    "metadata": GenerationMemoryBenchmark._load_metadata(output_path),
                }
            )
        return artifacts

    @staticmethod
    def _summarize_variant(commands: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        generation_times = [
            ((artifact.get("metadata") or {}).get("generation_time_seconds"))
            for artifact in artifacts
            if artifact.get("metadata") is not None
        ]
        runtime_memory = [((artifact.get("metadata") or {}).get("runtime_memory") or {}) for artifact in artifacts]
        return {
            "command_count": len(commands),
            "artifact_count": len(artifacts),
            "total_wall_seconds": round(sum(command["wall_seconds"] for command in commands), 4),
            "peak_sampled_rss_bytes": MultiOutputBatchBenchmark._max(
                command["sampler"].get("peak_sampled_rss_bytes") for command in commands
            ),
            "peak_sampled_darwin_physical_footprint_bytes": MultiOutputBatchBenchmark._max(
                command["sampler"].get("peak_sampled_darwin_physical_footprint_bytes") for command in commands
            ),
            "peak_time_l_maximum_resident_set_size_bytes": MultiOutputBatchBenchmark._max(
                command["time_l"].get("maximum_resident_set_size_bytes") for command in commands
            ),
            "total_generation_time_seconds": round(
                sum(float(value) for value in generation_times if value is not None),
                4,
            ),
            "artifact_runtime_memory": [
                {
                    "seed": artifact["seed"],
                    "process_rss_bytes": memory.get("process_rss_bytes"),
                    "darwin_physical_footprint_bytes": memory.get("darwin_physical_footprint_bytes"),
                    "mlx_peak_memory_bytes": memory.get("mlx_peak_memory_bytes"),
                }
                for artifact, memory in zip(artifacts, runtime_memory)
            ],
        }

    @staticmethod
    def _compare_case(
        *,
        case: BatchingCase,
        serial_artifacts: list[dict[str, Any]],
        batched_artifacts: list[dict[str, Any]],
        serial_commands: list[dict[str, Any]],
        batched_command: dict[str, Any],
    ) -> dict[str, Any]:
        comparisons = {}
        for serial_artifact, batched_artifact in zip(serial_artifacts, batched_artifacts):
            seed = serial_artifact["seed"]
            serial_path = Path(serial_artifact["output_path"])
            batched_path = Path(batched_artifact["output_path"])
            quality = (
                GenerationMemoryBenchmark._quality_comparison(profile="batching", left=serial_path, right=batched_path)
                if serial_artifact["exists"] and batched_artifact["exists"]
                else {"status": "failed", "reason": "missing_output"}
            )
            exact_match = False
            if case.kind == "image":
                exact_match = serial_artifact["sha256"] == batched_artifact["sha256"]
            elif quality.get("status") == "ok":
                exact_match = (
                    quality.get("mae") == 0.0
                    and quality.get("rmse") == 0.0
                    and quality.get("max_abs") == 0.0
                )
            comparisons[str(seed)] = {
                "serial_sha256": serial_artifact["sha256"],
                "batched_sha256": batched_artifact["sha256"],
                "serial_metadata_seed": ((serial_artifact.get("metadata") or {}).get("seed")),
                "batched_metadata_seed": ((batched_artifact.get("metadata") or {}).get("seed")),
                "exact_match": exact_match,
                "quality": quality,
            }
        serial_summary = MultiOutputBatchBenchmark._summarize_variant(serial_commands, serial_artifacts)
        batched_summary = MultiOutputBatchBenchmark._summarize_variant([batched_command], batched_artifacts)
        compute_improvement = MultiOutputBatchBenchmark._percent_improvement(
            baseline=serial_summary["total_wall_seconds"],
            candidate=batched_summary["total_wall_seconds"],
        )
        generation_regression = MultiOutputBatchBenchmark._percent_regression(
            baseline=serial_summary["total_generation_time_seconds"],
            candidate=batched_summary["total_generation_time_seconds"],
        )
        peak_rss_delta = MultiOutputBatchBenchmark._percent_regression(
            baseline=serial_summary["peak_sampled_rss_bytes"],
            candidate=batched_summary["peak_sampled_rss_bytes"],
        )
        peak_physical_delta = MultiOutputBatchBenchmark._percent_regression(
            baseline=serial_summary["peak_sampled_darwin_physical_footprint_bytes"],
            candidate=batched_summary["peak_sampled_darwin_physical_footprint_bytes"],
        )
        all_quality_ok = all(
            item["exact_match"] and item["serial_metadata_seed"] == item["batched_metadata_seed"] == int(seed)
            for seed, item in comparisons.items()
        )
        return {
            "per_seed": comparisons,
            "compute_improvement_percent": compute_improvement,
            "generation_time_regression_percent": generation_regression,
            "peak_rss_regression_percent": peak_rss_delta,
            "peak_physical_regression_percent": peak_physical_delta,
            "acceptance": {
                "quality_exact_per_seed": all_quality_ok,
                "compute_improved_at_least_5_percent": compute_improvement is not None and compute_improvement >= 5.0,
                "generation_time_not_worse_than_5_percent": (
                    generation_regression is not None and generation_regression <= 5.0
                ),
                "peak_rss_not_worse_than_10_percent": peak_rss_delta is not None and peak_rss_delta <= 10.0,
                "peak_physical_not_worse_than_10_percent": (
                    peak_physical_delta is not None and peak_physical_delta <= 10.0
                ),
            },
        }

    @staticmethod
    def _max(values) -> int | None:
        filtered = [int(value) for value in values if value is not None]
        return max(filtered) if filtered else None

    @staticmethod
    def _percent_improvement(*, baseline: float | int | None, candidate: float | int | None) -> float | None:
        if baseline in (None, 0) or candidate is None:
            return None
        return round(((float(baseline) - float(candidate)) / float(baseline)) * 100.0, 4)

    @staticmethod
    def _percent_regression(*, baseline: float | int | None, candidate: float | int | None) -> float | None:
        if baseline in (None, 0) or candidate is None:
            return None
        return round(((float(candidate) - float(baseline)) / float(baseline)) * 100.0, 4)

    @staticmethod
    def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
        compact = {}
        for case_id, data in report["cases"].items():
            comparison = data["comparison"]
            compact[case_id] = {
                "compute_improvement_percent": comparison["compute_improvement_percent"],
                "peak_rss_regression_percent": comparison["peak_rss_regression_percent"],
                "peak_physical_regression_percent": comparison["peak_physical_regression_percent"],
                "quality_exact_per_seed": comparison["acceptance"]["quality_exact_per_seed"],
                "serial_wall_seconds": data["serial"]["summary"]["total_wall_seconds"],
                "batched_wall_seconds": data["batched"]["summary"]["total_wall_seconds"],
                "serial_peak_rss_gb": MultiOutputBatchBenchmark._gb(
                    data["serial"]["summary"]["peak_sampled_rss_bytes"]
                ),
                "batched_peak_rss_gb": MultiOutputBatchBenchmark._gb(
                    data["batched"]["summary"]["peak_sampled_rss_bytes"]
                ),
                "serial_peak_physical_gb": MultiOutputBatchBenchmark._gb(
                    data["serial"]["summary"]["peak_sampled_darwin_physical_footprint_bytes"]
                ),
                "batched_peak_physical_gb": MultiOutputBatchBenchmark._gb(
                    data["batched"]["summary"]["peak_sampled_darwin_physical_footprint_bytes"]
                ),
            }
        return compact

    @staticmethod
    def _gb(value: int | None) -> float | None:
        if value is None:
            return None
        return round(value / BYTES_PER_GB, 4)


if __name__ == "__main__":
    MultiOutputBatchBenchmark.main()
