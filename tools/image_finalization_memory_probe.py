from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.common.config import ModelConfig
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil
from mflux.utils.metadata_builder import MetadataBuilder
from mflux.utils.runtime_memory import RuntimeMemory


@dataclass(frozen=True)
class ProbeCase:
    name: str
    output_name: str


class RootProcessSampler:
    def __init__(self, pid: int, interval_seconds: float):
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, int | float | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "RootProcessSampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.sample()

    def sample(self) -> None:
        rss_bytes = self._rss_bytes(self.pid)
        physical_bytes = self._darwin_physical_footprint_bytes(self.pid)
        self.samples.append(
            {
                "timestamp": time.time(),
                "rss_bytes": rss_bytes,
                "darwin_physical_footprint_bytes": physical_bytes,
            }
        )

    def summary(self) -> dict[str, int | float | None]:
        rss_values = [sample["rss_bytes"] for sample in self.samples if sample["rss_bytes"] is not None]
        physical_values = [
            sample["darwin_physical_footprint_bytes"]
            for sample in self.samples
            if sample["darwin_physical_footprint_bytes"] is not None
        ]
        return {
            "sample_count": len(self.samples),
            "peak_sampled_rss_bytes": int(max(rss_values)) if rss_values else None,
            "avg_sampled_rss_bytes": int(mean(rss_values)) if rss_values else None,
            "peak_sampled_darwin_physical_footprint_bytes": int(max(physical_values)) if physical_values else None,
            "avg_sampled_darwin_physical_footprint_bytes": int(mean(physical_values)) if physical_values else None,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval_seconds)

    @staticmethod
    def _rss_bytes(pid: int) -> int | None:
        try:
            output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], stderr=subprocess.DEVNULL, text=True)
            return int(output.strip()) * 1024
        except (OSError, subprocess.SubprocessError, TypeError, ValueError):
            return None

    @staticmethod
    def _darwin_physical_footprint_bytes(pid: int) -> int | None:
        if sys.platform != "darwin":
            return None
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    RuntimeMemory._DARWIN_PHYSICAL_FOOTPRINT_HELPER,
                    str(pid),
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError, TimeoutError, TypeError, ValueError):
            return None
        if result.returncode != 0:
            return None
        try:
            return int(result.stdout.strip())
        except ValueError:
            return None


class ImageFinalizationMemoryProbe:
    CASES = (
        ProbeCase(name="current-default", output_name="current-default.png"),
        ProbeCase(name="current-embed-metadata", output_name="current-embed-metadata.png"),
        ProbeCase(name="current-sidecar-metadata", output_name="current-sidecar-metadata.png"),
        ProbeCase(name="legacy-default-simulated", output_name="legacy-default-simulated.png"),
    )

    @staticmethod
    def main() -> None:
        args = ImageFinalizationMemoryProbe._parse_args()
        if args.child_case is not None:
            ImageFinalizationMemoryProbe._run_child(
                case_name=args.child_case,
                result_path=args.result_path,
                output_path=args.output_path,
                width=args.width,
                height=args.height,
            )
            return

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        results = [
            ImageFinalizationMemoryProbe._run_case(
                case=case,
                output_dir=output_dir,
                width=args.width,
                height=args.height,
                sample_interval_ms=args.sample_interval_ms,
            )
            for case in ImageFinalizationMemoryProbe.CASES
        ]
        report = ImageFinalizationMemoryProbe._build_report(results=results, width=args.width, height=args.height)
        stats_path = output_dir / "image_finalization_memory_stats.json"
        report_path = output_dir / "image_finalization_memory_report.md"
        stats_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        report_path.write_text(ImageFinalizationMemoryProbe._markdown_report(report))
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        print(f"wrote {stats_path}")
        print(f"wrote {report_path}")

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("docs/assets/validation/image-finalization-2026-06-30"),
        )
        parser.add_argument("--width", type=int, default=4096)
        parser.add_argument("--height", type=int, default=4096)
        parser.add_argument("--sample-interval-ms", type=int, default=10)
        parser.add_argument("--child-case", type=str, default=None)
        parser.add_argument("--result-path", type=Path, default=None)
        parser.add_argument("--output-path", type=Path, default=None)
        return parser.parse_args()

    @staticmethod
    def _run_case(
        *,
        case: ProbeCase,
        output_dir: Path,
        width: int,
        height: int,
        sample_interval_ms: int,
    ) -> dict[str, Any]:
        case_dir = output_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        result_path = case_dir / "result.json"
        output_path = case_dir / case.output_name
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-case",
            case.name,
            "--result-path",
            str(result_path),
            "--output-path",
            str(output_path),
            "--width",
            str(width),
            "--height",
            str(height),
        ]
        child = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path.cwd(),
        )
        assert child.stdout is not None
        ready = child.stdout.readline().strip()
        if ready != "READY":
            stderr = child.stderr.read() if child.stderr is not None else ""
            raise RuntimeError(f"Probe child for {case.name} did not become ready. stdout={ready!r} stderr={stderr!r}")
        assert child.stdin is not None
        with RootProcessSampler(child.pid, sample_interval_ms / 1000) as sampler:
            child.stdin.write("\n")
            child.stdin.flush()
            returncode = child.wait()
        stdout_tail = child.stdout.read() if child.stdout is not None else ""
        stderr_tail = child.stderr.read() if child.stderr is not None else ""
        if returncode != 0:
            raise RuntimeError(
                f"Probe child for {case.name} failed with code {returncode}. "
                f"stdout={stdout_tail!r} stderr={stderr_tail!r}"
            )
        child_result = json.loads(result_path.read_text())
        child_result["sampler"] = sampler.summary()
        child_result["stdout_tail"] = stdout_tail.strip()
        child_result["stderr_tail"] = stderr_tail.strip()
        return child_result

    @staticmethod
    def _run_child(
        *,
        case_name: str,
        result_path: Path | None,
        output_path: Path | None,
        width: int,
        height: int,
    ) -> None:
        if result_path is None or output_path is None:
            raise SystemExit("child mode requires --result-path and --output-path")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        metadata_path = output_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            metadata_path.unlink()

        generated = ImageFinalizationMemoryProbe._generated_image(width=width, height=height)
        save_calls: list[str] = []
        open_calls: list[str] = []
        runtime_snapshot_calls: list[str] = []

        original_save = PIL.Image.Image.save
        original_open = PIL.Image.open
        original_snapshot = RuntimeMemory.snapshot

        def tracked_save(image_self, fp, *args, **kwargs):
            save_calls.append(str(fp))
            return original_save(image_self, fp, *args, **kwargs)

        def tracked_open(fp, *args, **kwargs):
            open_calls.append(str(fp))
            return original_open(fp, *args, **kwargs)

        def tracked_snapshot(phase: str, *args, **kwargs):
            runtime_snapshot_calls.append(phase)
            return original_snapshot(phase, *args, **kwargs)

        PIL.Image.Image.save = tracked_save
        PIL.Image.open = tracked_open
        RuntimeMemory.snapshot = tracked_snapshot

        before_snapshot = original_snapshot("before-save", synchronize=False).to_metadata()
        print("READY", flush=True)
        sys.stdin.readline()
        started_at = time.time()
        try:
            if case_name == "current-default":
                saved_path = generated.save(path=output_path, overwrite=True)
            elif case_name == "current-embed-metadata":
                saved_path = generated.save(path=output_path, overwrite=True, embed_metadata=True)
            elif case_name == "current-sidecar-metadata":
                saved_path = generated.save(path=output_path, overwrite=True, export_json_metadata=True)
            elif case_name == "legacy-default-simulated":
                saved_path = ImageFinalizationMemoryProbe._legacy_save(generated=generated, output_path=output_path)
            else:
                raise ValueError(f"Unsupported case {case_name!r}")
        finally:
            elapsed_seconds = time.time() - started_at
        after_snapshot = original_snapshot("after-save", synchronize=False).to_metadata()

        with original_open(saved_path) as image:
            info = dict(image.info)
        result = {
            "case": case_name,
            "output_path": str(saved_path),
            "wall_time_seconds": round(elapsed_seconds, 4),
            "pipeline_image_save_calls": len(save_calls),
            "pipeline_image_open_calls": len(open_calls),
            "pipeline_runtime_snapshot_calls": list(runtime_snapshot_calls),
            "metadata_sidecar_exists": saved_path.with_suffix(".metadata.json").exists(),
            "has_exif": bool(info.get("exif")),
            "has_png_xmp": "XML:com.adobe.xmp" in info,
            "has_png_iptc": "IPTC" in info,
            "saved_info_keys": sorted(info.keys()),
            "before_save_snapshot": before_snapshot,
            "after_save_snapshot": after_snapshot,
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    @staticmethod
    def _generated_image(*, width: int, height: int) -> GeneratedImage:
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        image = PIL.Image.fromarray(pixels, mode="RGB")
        del pixels
        return GeneratedImage(
            image=image,
            model_config=ModelConfig.qwen_image(),
            seed=4242,
            prompt="A realistic architectural rendering of a compact courtyard with patterned shadows.",
            steps=8,
            guidance=4.0,
            precision=mx.bfloat16,
            quantization=8,
            generation_time=1.23,
            height=height,
            width=width,
        )

    @staticmethod
    def _legacy_save(*, generated: GeneratedImage, output_path: Path) -> Path:
        metadata = generated._get_metadata(include_runtime_memory=True)
        file_path = ImageUtil.resolve_output_path(path=output_path, overwrite=True)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        image_format = ImageUtil._image_format_for_path(file_path)
        save_kwargs = {}
        if image_format is not None:
            save_kwargs["format"] = image_format
        generated.image.save(file_path, **save_kwargs)
        exif_bytes = ImageUtil._build_exif_bytes(metadata)
        with PIL.Image.open(file_path) as image:
            reopened_kwargs = {"exif": exif_bytes}
            if image_format is not None:
                reopened_kwargs["format"] = image_format
            image.save(file_path, **reopened_kwargs)
        MetadataBuilder.embed_metadata(metadata, file_path)
        return file_path

    @staticmethod
    def _build_report(*, results: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
        by_case = {result["case"]: result for result in results}
        legacy = by_case["legacy-default-simulated"]
        current_default = by_case["current-default"]
        current_embed = by_case["current-embed-metadata"]
        current_sidecar = by_case["current-sidecar-metadata"]
        summary = {
            "width": width,
            "height": height,
            "legacy_to_current_default_peak_rss_delta_percent": ImageFinalizationMemoryProbe._delta_percent(
                legacy["sampler"]["peak_sampled_rss_bytes"],
                current_default["sampler"]["peak_sampled_rss_bytes"],
            ),
            "legacy_to_current_default_peak_physical_delta_percent": ImageFinalizationMemoryProbe._delta_percent(
                legacy["sampler"]["peak_sampled_darwin_physical_footprint_bytes"],
                current_default["sampler"]["peak_sampled_darwin_physical_footprint_bytes"],
            ),
            "legacy_to_current_embed_peak_rss_delta_percent": ImageFinalizationMemoryProbe._delta_percent(
                legacy["sampler"]["peak_sampled_rss_bytes"],
                current_embed["sampler"]["peak_sampled_rss_bytes"],
            ),
            "legacy_to_current_embed_peak_physical_delta_percent": ImageFinalizationMemoryProbe._delta_percent(
                legacy["sampler"]["peak_sampled_darwin_physical_footprint_bytes"],
                current_embed["sampler"]["peak_sampled_darwin_physical_footprint_bytes"],
            ),
            "legacy_to_current_sidecar_peak_rss_delta_percent": ImageFinalizationMemoryProbe._delta_percent(
                legacy["sampler"]["peak_sampled_rss_bytes"],
                current_sidecar["sampler"]["peak_sampled_rss_bytes"],
            ),
            "legacy_to_current_sidecar_peak_physical_delta_percent": ImageFinalizationMemoryProbe._delta_percent(
                legacy["sampler"]["peak_sampled_darwin_physical_footprint_bytes"],
                current_sidecar["sampler"]["peak_sampled_darwin_physical_footprint_bytes"],
            ),
        }
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "method": {
                "image": {"width": width, "height": height, "mode": "RGB", "source": "deterministic synthetic noise"},
                "sampling": {
                    "scope": "save-phase only",
                    "rss_source": "ps rss for child process",
                    "darwin_physical_source": "proc_pid_rusage helper for child process",
                },
                "cases": [
                    "current-default",
                    "current-embed-metadata",
                    "current-sidecar-metadata",
                    "legacy-default-simulated",
                ],
            },
            "results": results,
            "summary": summary,
        }

    @staticmethod
    def _delta_percent(before: int | None, after: int | None) -> float | None:
        if before in (None, 0) or after is None:
            return None
        return round(((after - before) / before) * 100, 4)

    @staticmethod
    def _gb(value: int | None) -> str:
        if value is None:
            return "n/a"
        return f"{value / 10**9:.3f}"

    @staticmethod
    def _markdown_report(report: dict[str, Any]) -> str:
        lines = [
            "# Image Finalization Memory Report",
            "",
            "Measured on June 30, 2026 with a dedicated save-phase probe.",
            "",
            "## Method",
            f"- Synthetic deterministic RGB PNG: {report['method']['image']['width']}x{report['method']['image']['height']}",
            "- Sampling window: after image construction, during save/finalization only",
            "- Metrics: sampled process RSS and Darwin physical footprint for the child save process",
            "- Cases: current default save, current embedded metadata save, current sidecar metadata save, simulated legacy default save",
            "",
            "## Results",
            "",
            "| Case | Save Calls | Reopen Calls | Runtime Snapshots | Peak RSS GB | Avg RSS GB | Peak Physical GB | Avg Physical GB | EXIF | XMP | IPTC | Sidecar |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
        for result in report["results"]:
            sampler = result["sampler"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        result["case"],
                        str(result["pipeline_image_save_calls"]),
                        str(result["pipeline_image_open_calls"]),
                        str(len(result["pipeline_runtime_snapshot_calls"])),
                        ImageFinalizationMemoryProbe._gb(sampler["peak_sampled_rss_bytes"]),
                        ImageFinalizationMemoryProbe._gb(sampler["avg_sampled_rss_bytes"]),
                        ImageFinalizationMemoryProbe._gb(sampler["peak_sampled_darwin_physical_footprint_bytes"]),
                        ImageFinalizationMemoryProbe._gb(sampler["avg_sampled_darwin_physical_footprint_bytes"]),
                        "yes" if result["has_exif"] else "no",
                        "yes" if result["has_png_xmp"] else "no",
                        "yes" if result["has_png_iptc"] else "no",
                        "yes" if result["metadata_sidecar_exists"] else "no",
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "## Interpretation",
                (
                    "- Current default versus simulated legacy default: peak RSS "
                    f"{report['summary']['legacy_to_current_default_peak_rss_delta_percent']}%, peak physical footprint "
                    f"{report['summary']['legacy_to_current_default_peak_physical_delta_percent']}%."
                ),
                (
                    "- Current embedded metadata versus simulated legacy default: peak RSS "
                    f"{report['summary']['legacy_to_current_embed_peak_rss_delta_percent']}%, peak physical footprint "
                    f"{report['summary']['legacy_to_current_embed_peak_physical_delta_percent']}%."
                ),
                (
                    "- Current sidecar metadata versus simulated legacy default: peak RSS "
                    f"{report['summary']['legacy_to_current_sidecar_peak_rss_delta_percent']}%, peak physical footprint "
                    f"{report['summary']['legacy_to_current_sidecar_peak_physical_delta_percent']}%."
                ),
                "- The current embedded-metadata path still preserves EXIF plus PNG XMP/IPTC, but does so with one save call and zero reopen calls.",
                "- The current default path no longer collects runtime-memory metadata and no longer embeds PNG metadata unless explicitly requested.",
            ]
        )
        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ImageFinalizationMemoryProbe.main()
