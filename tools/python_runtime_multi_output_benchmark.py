import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generation_memory_benchmark import GenerationMemoryBenchmark, ProcessTreeSampler
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SOURCE_IMAGE = ROOT / "docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png"
FLUX_SKETCH = ROOT / "docs/assets/validation/flux2-klein-base-starship-2026-06-10/base9b_source_d_sketch.png"
FLUX_DUSK = ROOT / "docs/assets/validation/flux2-klein-base-starship-2026-06-10/base9b_source_b_latent_dusk.png"
I2V_SOURCE = ROOT / "docs/assets/i2v_takeoff_source.png"

COMMON_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "MFLUX_RUNTIME_MEMORY_TELEMETRY": "1",
    "MFLUX_BENCHMARK_PARENT_PHYSICAL_SAMPLING": "1",
}


@dataclass(frozen=True)
class RuntimeCase:
    case_id: str
    kind: str
    model: str
    load_kwargs: dict[str, Any]
    output_suffix: str
    output_stem: str
    seeds: tuple[int, ...]

    def output_path(self, base_dir: Path, seed: int) -> Path:
        return base_dir / f"{self.output_stem}_seed_{seed}{self.output_suffix}"

    def output_template(self, base_dir: Path) -> str:
        return str(base_dir / f"{self.output_stem}{self.output_suffix}")


CASES = {
    "qwen_masked_edit": RuntimeCase(
        case_id="qwen_masked_edit",
        kind="image",
        model="AbstractFramework/qwen-image-edit-2511-8bit",
        load_kwargs={"image_count": 1, "has_mask": True},
        output_suffix=".png",
        output_stem="qwen_engine",
        seeds=(4201, 4202),
    ),
    "flux2_multi_reference": RuntimeCase(
        case_id="flux2_multi_reference",
        kind="image",
        model="AbstractFramework/flux.2-klein-9b-8bit",
        load_kwargs={"image_count": 2, "i2i_mode": "multi-reference"},
        output_suffix=".png",
        output_stem="flux2_multiref",
        seeds=(8614, 8615),
    ),
    "wan_i2v_short": RuntimeCase(
        case_id="wan_i2v_short",
        kind="video",
        model="AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit",
        load_kwargs={"image_count": 1},
        output_suffix=".mp4",
        output_stem="wan_i2v_short",
        seeds=(6601, 6602),
    ),
    "zimage_t2i_large": RuntimeCase(
        case_id="zimage_t2i_large",
        kind="image",
        model="AbstractFramework/z-image-turbo-8bit",
        load_kwargs={},
        output_suffix=".png",
        output_stem="zimage_large",
        seeds=(7401, 7402),
    ),
    "wan_i2v_large": RuntimeCase(
        case_id="wan_i2v_large",
        kind="video",
        model="AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit",
        load_kwargs={"image_count": 1},
        output_suffix=".mp4",
        output_stem="wan_i2v_large",
        seeds=(6701, 6702),
    ),
}


class PythonRuntimeMultiOutputBenchmark:
    @staticmethod
    def main() -> None:
        args = PythonRuntimeMultiOutputBenchmark._parse_args()
        if args.child:
            PythonRuntimeMultiOutputBenchmark._run_child(
                case=CASES[args.case],
                variant=args.variant,
                output_dir=args.output_dir.resolve(),
            )
            return

        os.environ.setdefault("MFLUX_BENCHMARK_PARENT_PHYSICAL_SAMPLING", "1")
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "schema_version": 1,
            "environment": GenerationMemoryBenchmark._environment(),
            "cases": {},
        }
        for case_id in args.cases:
            case = CASES[case_id]
            report["cases"][case_id] = PythonRuntimeMultiOutputBenchmark._run_case(
                case=case,
                output_dir=output_dir / case_id,
                sample_interval_ms=args.sample_interval_ms,
            )
        report_path = output_dir / "python_runtime_multi_output_benchmark_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(PythonRuntimeMultiOutputBenchmark._compact_report(report), indent=2, sort_keys=True))
        print(f"wrote {report_path}")

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("validation_outputs/python_runtime_multi_output_20260630"),
        )
        parser.add_argument("--sample-interval-ms", type=int, default=200)
        parser.add_argument("--cases", nargs="*", choices=sorted(CASES), default=sorted(CASES))
        parser.add_argument("--child", action="store_true")
        parser.add_argument("--case", choices=sorted(CASES))
        parser.add_argument("--variant", choices=["reuse", "reload"])
        args = parser.parse_args()
        if args.sample_interval_ms <= 0:
            raise ValueError("--sample-interval-ms must be greater than zero.")
        if args.child and (args.case is None or args.variant is None):
            raise ValueError("--child requires --case and --variant.")
        return args

    @staticmethod
    def _run_case(*, case: RuntimeCase, output_dir: Path, sample_interval_ms: int) -> dict[str, Any]:
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        reuse = PythonRuntimeMultiOutputBenchmark._run_variant(
            case=case,
            variant="reuse",
            output_dir=output_dir / "reuse",
            sample_interval_ms=sample_interval_ms,
        )
        reload = PythonRuntimeMultiOutputBenchmark._run_variant(
            case=case,
            variant="reload",
            output_dir=output_dir / "reload",
            sample_interval_ms=sample_interval_ms,
        )
        return {
            "case_id": case.case_id,
            "kind": case.kind,
            "model": case.model,
            "reuse": reuse,
            "reload": reload,
            "comparison": PythonRuntimeMultiOutputBenchmark._compare_case(case=case, reuse=reuse, reload=reload),
        }

    @staticmethod
    def _run_variant(
        *,
        case: RuntimeCase,
        variant: str,
        output_dir: Path,
        sample_interval_ms: int,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        report_path = output_dir / "run_report.json"
        env = os.environ.copy()
        env.update(COMMON_ENV)
        argv = [
            "uv",
            "run",
            "python",
            "tools/python_runtime_multi_output_benchmark.py",
            "--child",
            "--case",
            case.case_id,
            "--variant",
            variant,
            "--output-dir",
            str(output_dir),
        ]
        timed_argv = ["/usr/bin/time", "-l", *argv]
        started = time.perf_counter()
        process = subprocess.Popen(
            timed_argv,
            cwd=ROOT,
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
            raise RuntimeError(f"{case.case_id} {variant} failed; see {stderr_path}")
        child_report = json.loads(report_path.read_text())
        return {
            "variant": variant,
            "wall_seconds": round(wall_seconds, 4),
            "sampler": sampler.summary(),
            "samples": sampler.samples,
            "time_l": GenerationMemoryBenchmark._parse_time_l_output(stderr),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "report_path": str(report_path),
            "child_report": child_report,
        }

    @staticmethod
    def _run_child(*, case: RuntimeCase, variant: str, output_dir: Path) -> None:
        from mlxgen import load_generation_model

        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = output_dir / "artifacts"
        work_dir.mkdir(parents=True, exist_ok=True)
        generate_kwargs = PythonRuntimeMultiOutputBenchmark._generate_kwargs(case=case, work_dir=work_dir)
        save_kwargs = PythonRuntimeMultiOutputBenchmark._save_kwargs(case)
        progress_events: list[dict[str, Any]] = []
        progress_started = time.perf_counter()

        def on_progress(event) -> None:
            progress_events.append(
                {
                    "elapsed_seconds": round(time.perf_counter() - progress_started, 4),
                    "task": event.task,
                    "phase": event.phase,
                    "seed": event.seed,
                    "item_index": event.item_index,
                    "item_count": event.item_count,
                    "step": event.step,
                    "total_steps": event.total_steps,
                    "frame": event.frame,
                    "total_frames": event.total_frames,
                    "output_path": event.output_path,
                    "input_name": event.input_name,
                    "rss_bytes": PythonRuntimeMultiOutputBenchmark._current_rss_bytes(),
                    "darwin_physical_footprint_bytes": ProcessTreeSampler._darwin_physical_footprint_bytes(os.getpid()),
                }
            )

        started = time.perf_counter()
        if variant == "reuse":
            loaded = load_generation_model(model=case.model, **case.load_kwargs)
            results = loaded.generate_outputs(
                seeds=case.seeds,
                output=case.output_template(work_dir),
                overwrite=True,
                progress_callback=on_progress,
                save_kwargs=save_kwargs,
                **generate_kwargs,
            )
            runtime_id = loaded.runtime_id
            task = loaded.plan.task
        else:
            results = []
            runtime_id = None
            task = None
            for seed in case.seeds:
                loaded = load_generation_model(model=case.model, **case.load_kwargs)
                runtime_id = loaded.runtime_id
                task = loaded.plan.task
                results.append(
                    loaded.generate_output(
                        seed=seed,
                        output=case.output_path(work_dir, seed),
                        overwrite=True,
                        progress_callback=on_progress,
                        save_kwargs=save_kwargs,
                        **generate_kwargs,
                    )
                )
        wall_seconds = time.perf_counter() - started
        report = {
            "case_id": case.case_id,
            "variant": variant,
            "model": case.model,
            "runtime_id": runtime_id,
            "task": task,
            "wall_seconds": round(wall_seconds, 4),
            "results": [
                {
                    "seed": result.seed,
                    "task": result.task,
                    "item_index": result.item_index,
                    "item_count": result.item_count,
                    "output_path": str(result.output_path) if result.output_path is not None else None,
                    "saved_path": str(result.saved_path) if result.saved_path is not None else None,
                    "sha256": PythonRuntimeMultiOutputBenchmark._sha256(result.saved_path),
                    "metadata": GenerationMemoryBenchmark._load_metadata(result.saved_path),
                }
                for result in results
            ],
            "progress_events": progress_events,
            "phase_summary": PythonRuntimeMultiOutputBenchmark._phase_summary(progress_events),
        }
        report_path = output_dir / "run_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps({"case_id": case.case_id, "variant": variant, "report_path": str(report_path)}, indent=2))

    @staticmethod
    def _generate_kwargs(*, case: RuntimeCase, work_dir: Path) -> dict[str, Any]:
        if case.case_id == "qwen_masked_edit":
            mask_path = work_dir / "qwen_engine_mask.png"
            PythonRuntimeMultiOutputBenchmark._write_qwen_engine_mask(mask_path)
            return {
                "prompt": (
                    "Keep the same silver spaceship, icy canyon, and sunrise lighting. "
                    "Only inside the masked engine area, intensify both blue engines into brighter plasma thrusters, "
                    "add dense blue glow and snow vapor around the thrusters, and preserve the rest of the image unchanged."
                ),
                "negative_prompt": (
                    "blurry, low quality, distorted, deformed, extra ship parts, changed camera angle, "
                    "changed background, text, watermark"
                ),
                "image_path": SOURCE_IMAGE,
                "image_paths": [str(SOURCE_IMAGE)],
                "mask_path": mask_path,
                "width": 768,
                "height": 432,
                "guidance": 4.0,
                "num_inference_steps": 20,
            }
        if case.case_id == "flux2_multi_reference":
            return {
                "prompt": (
                    "Use the first image as the structural line-art reference and the second image as the lighting "
                    "and material reference. Produce one coherent close-up of the same starship scene: graphite line art "
                    "with cool aurora metallic reflections and darker polar-night shading, the same crop, same fuselage "
                    "and engines, same snow field and ice cliffs, no extra ships, no text."
                ),
                "image_paths": [str(FLUX_SKETCH), str(FLUX_DUSK)],
                "width": 432,
                "height": 240,
                "guidance": 1.0,
                "num_inference_steps": 20,
            }
        if case.case_id == "wan_i2v_short":
            return {
                "prompt": (
                    "Starting from the input image, animate the same compact sci-fi spaceship on the frozen snow planet. "
                    "The blue engines brighten, snow blows outward under the hull, and the spaceship slowly lifts off "
                    "vertically while keeping its shape and the icy cliffs stable. Cinematic wide camera, no people, no text."
                ),
                "image_path": I2V_SOURCE,
                "width": 448,
                "height": 256,
                "num_frames": 17,
                "fps": 10,
                "guidance": 4.0,
                "guidance_2": 3.0,
                "num_inference_steps": 8,
            }
        if case.case_id == "wan_i2v_large":
            return {
                "prompt": (
                    "Starting from the input image, animate the same compact sci-fi spaceship on the frozen snow planet. "
                    "The blue engines brighten, snow blows outward under the hull, and the spaceship slowly lifts off "
                    "vertically while keeping its shape and the icy cliffs stable. Cinematic wide camera, no people, no text."
                ),
                "image_path": I2V_SOURCE,
                "width": 1280,
                "height": 720,
                "num_frames": 9,
                "fps": 16,
                "guidance": 3.5,
                "guidance_2": 3.0,
                "num_inference_steps": 6,
            }
        if case.case_id == "zimage_t2i_large":
            return {
                "prompt": "A red enamel kettle on a steel counter, product photo",
                "width": 1024,
                "height": 1024,
                "guidance": 1.0,
                "num_inference_steps": 8,
            }
        raise ValueError(f"Unknown case: {case.case_id}")

    @staticmethod
    def _save_kwargs(case: RuntimeCase) -> dict[str, Any]:
        if case.kind == "image":
            return {"export_json_metadata": True, "embed_metadata": False}
        return {"export_json_metadata": True}

    @staticmethod
    def _write_qwen_engine_mask(path: Path) -> None:
        image = Image.new("L", (768, 432), color=0)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((250, 175, 520, 330), radius=32, fill=255)
        image.save(path)

    @staticmethod
    def _compare_case(*, case: RuntimeCase, reuse: dict[str, Any], reload: dict[str, Any]) -> dict[str, Any]:
        reuse_report = reuse["child_report"]
        reload_report = reload["child_report"]
        per_seed = {}
        for reuse_result, reload_result in zip(reuse_report["results"], reload_report["results"]):
            seed = str(reuse_result["seed"])
            quality = GenerationMemoryBenchmark._quality_comparison(
                profile="batching",
                left=Path(reuse_result["saved_path"]),
                right=Path(reload_result["saved_path"]),
            )
            exact_match = False
            if case.kind == "image":
                exact_match = reuse_result["sha256"] == reload_result["sha256"]
            elif quality.get("status") == "ok":
                exact_match = (
                    quality.get("mae") == 0.0
                    and quality.get("rmse") == 0.0
                    and quality.get("max_abs") == 0.0
                )
            per_seed[seed] = {
                "reuse_sha256": reuse_result["sha256"],
                "reload_sha256": reload_result["sha256"],
                "reuse_metadata_seed": ((reuse_result.get("metadata") or {}).get("seed")),
                "reload_metadata_seed": ((reload_result.get("metadata") or {}).get("seed")),
                "exact_match": exact_match,
                "quality": quality,
                "reuse_phase_summary": reuse_report["phase_summary"].get(seed),
                "reload_phase_summary": reload_report["phase_summary"].get(seed),
            }
        compute_improvement = PythonRuntimeMultiOutputBenchmark._percent_improvement(
            baseline=reload["wall_seconds"],
            candidate=reuse["wall_seconds"],
        )
        peak_rss_regression = PythonRuntimeMultiOutputBenchmark._percent_regression(
            baseline=reload["sampler"].get("peak_sampled_rss_bytes"),
            candidate=reuse["sampler"].get("peak_sampled_rss_bytes"),
        )
        peak_physical_regression = PythonRuntimeMultiOutputBenchmark._percent_regression(
            baseline=reload["sampler"].get("peak_sampled_darwin_physical_footprint_bytes"),
            candidate=reuse["sampler"].get("peak_sampled_darwin_physical_footprint_bytes"),
        )
        all_quality_ok = all(
            item["exact_match"] and item["reuse_metadata_seed"] == item["reload_metadata_seed"] == int(seed)
            for seed, item in per_seed.items()
        )
        return {
            "per_seed": per_seed,
            "compute_improvement_percent": compute_improvement,
            "peak_rss_regression_percent": peak_rss_regression,
            "peak_physical_regression_percent": peak_physical_regression,
            "acceptance": {
                "quality_exact_per_seed": all_quality_ok,
                "compute_improved_at_least_5_percent": compute_improvement is not None and compute_improvement >= 5.0,
                "peak_rss_not_worse_than_10_percent": (
                    peak_rss_regression is not None and peak_rss_regression <= 10.0
                ),
                "peak_physical_not_worse_than_10_percent": (
                    peak_physical_regression is not None and peak_physical_regression <= 10.0
                ),
            },
        }

    @staticmethod
    def _phase_summary(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        by_seed: dict[str, dict[str, Any]] = {}
        for event in events:
            seed = str(event["seed"])
            if seed == "None":
                continue
            summary = by_seed.setdefault(seed, {})
            phase = event["phase"]
            summary[f"{phase}_elapsed_seconds"] = event["elapsed_seconds"]
            summary[f"{phase}_rss_bytes"] = event["rss_bytes"]
            summary[f"{phase}_darwin_physical_footprint_bytes"] = event["darwin_physical_footprint_bytes"]
        for summary in by_seed.values():
            save_elapsed = summary.get("save_elapsed_seconds")
            complete_elapsed = summary.get("complete_elapsed_seconds")
            generated_elapsed = summary.get("generated_elapsed_seconds")
            summary["save_duration_seconds"] = (
                round(complete_elapsed - save_elapsed, 4)
                if save_elapsed is not None and complete_elapsed is not None
                else None
            )
            summary["generated_to_complete_seconds"] = (
                round(complete_elapsed - generated_elapsed, 4)
                if generated_elapsed is not None and complete_elapsed is not None
                else None
            )
        return by_seed

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
                "reuse_wall_seconds": data["reuse"]["wall_seconds"],
                "reload_wall_seconds": data["reload"]["wall_seconds"],
            }
        return compact

    @staticmethod
    def _current_rss_bytes() -> int | None:
        try:
            output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        except (OSError, subprocess.SubprocessError):
            return None
        value = output.strip()
        if not value:
            return None
        try:
            return int(value) * 1024
        except ValueError:
            return None

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
    def _sha256(path: str | Path | None) -> str | None:
        if path is None:
            return None
        path = Path(path)
        if not path.exists():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


if __name__ == "__main__":
    PythonRuntimeMultiOutputBenchmark.main()
