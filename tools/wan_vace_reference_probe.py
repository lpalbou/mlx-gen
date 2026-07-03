import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps
from huggingface_hub import snapshot_download

from mflux.utils.video_util import VideoStreamWriter, VideoUtil

BYTES_PER_GB = 1000**3
DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, paintings, still picture, "
    "low quality, JPEG residue, deformed face, extra fingers, messy background"
)
TOOLS_DIR = Path(__file__).resolve().parent
_GENERATION_MEMORY_BENCHMARK_PATH = TOOLS_DIR / "generation_memory_benchmark.py"
_GENERATION_MEMORY_BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "generation_memory_benchmark",
    _GENERATION_MEMORY_BENCHMARK_PATH,
)
assert _GENERATION_MEMORY_BENCHMARK_SPEC is not None
assert _GENERATION_MEMORY_BENCHMARK_SPEC.loader is not None
_GENERATION_MEMORY_BENCHMARK_MODULE = importlib.util.module_from_spec(_GENERATION_MEMORY_BENCHMARK_SPEC)
sys.modules.setdefault("generation_memory_benchmark", _GENERATION_MEMORY_BENCHMARK_MODULE)
_GENERATION_MEMORY_BENCHMARK_SPEC.loader.exec_module(_GENERATION_MEMORY_BENCHMARK_MODULE)
GenerationMemoryBenchmark = _GENERATION_MEMORY_BENCHMARK_MODULE.GenerationMemoryBenchmark
ProcessTreeSampler = _GENERATION_MEMORY_BENCHMARK_MODULE.ProcessTreeSampler


@dataclass(frozen=True)
class CasePreset:
    case_id: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    fps: int
    num_frames: int
    steps: int
    guidance: float
    flow_shift: float
    seed: int
    source_kind: str
    start_seconds: float = 0.0
    use_first_frame_reference: bool = False
    use_masked_source_video: bool = False


class WanVaceReferenceProbe:
    PORTRAIT_SOURCE_IMAGE = Path("tests/resources/unsplash_person.jpg")
    SHIP_SOURCE_VIDEO = Path("docs/assets/examples/spaceship-snow/06_i2v_a14b_spaceship_takeoff_from_source.mp4")

    @staticmethod
    def case_presets() -> dict[str, CasePreset]:
        return {
            "portrait_hair_eyes": CasePreset(
                case_id="portrait_hair_eyes",
                prompt=(
                    "A realistic portrait video of the same 25-year-old man with the same face, natural brown eyes, "
                    "a black shirt, a calm neutral expression, a blurred park background, and soft natural daylight. "
                    "Only change the hair color to clearly light silver-blonde with soft ash highlights, visibly lighter "
                    "than the original dark brown hair, while preserving the same identity and facial features."
                ),
                negative_prompt=(
                    f"{DEFAULT_NEGATIVE_PROMPT}, glowing eyes, red eyes, neon, overexposed face, washed out skin, "
                    "burned highlights, white skin, female face, child face, changed eye color, changed face, "
                    "rewritten eyebrows, altered jawline, brown hair, dark hair, unchanged hair color"
                ),
                width=320,
                height=480,
                fps=8,
                num_frames=17,
                steps=10,
                guidance=4.5,
                flow_shift=3.0,
                seed=7301,
                source_kind="portrait_still_animation",
                use_first_frame_reference=True,
                use_masked_source_video=False,
            ),
            "ship_reactor_nacelles": CasePreset(
                case_id="ship_reactor_nacelles",
                prompt=(
                    "Keep the same icy cliffs, snow haze, soft sunrise lighting, and lift-off camera motion. "
                    "Transform the ship into a bulkier smuggler-style starship with a bright circular rear reactor "
                    "and two side nacelles while preserving realistic vehicle detail."
                ),
                negative_prompt=(
                    f"{DEFAULT_NEGATIVE_PROMPT}, duplicate ships, warped hull, melted nacelles, unreadable reactor, "
                    "washed out frame, blown highlights"
                ),
                width=448,
                height=256,
                fps=10,
                num_frames=17,
                steps=16,
                guidance=3.5,
                flow_shift=3.0,
                seed=7302,
                source_kind="bounded_ship_video_excerpt",
                use_first_frame_reference=True,
                use_masked_source_video=True,
            ),
        }

    @staticmethod
    def main() -> None:
        args = WanVaceReferenceProbe._parse_args()
        if args.internal_run:
            WanVaceReferenceProbe._internal_run(args)
            return

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = WanVaceReferenceProbe._prefetch_model(
            model_id=args.model_id,
            skip_prefetch=args.skip_prefetch,
        )
        report = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "schema_version": 1,
            "kind": "upstream_reference_only",
            "model_id": args.model_id,
            "model_path": str(model_path),
            "environment": GenerationMemoryBenchmark._environment(),
            "cases": {},
        }
        for case_id in args.cases:
            case = WanVaceReferenceProbe.case_presets()[case_id]
            report["cases"][case_id] = WanVaceReferenceProbe._run_case(
                case=case,
                model_path=model_path,
                diffusers_src=args.diffusers_src.resolve() if args.diffusers_src is not None else None,
                output_dir=output_dir / case_id,
                sample_interval_ms=args.sample_interval_ms,
                device=args.device,
                portrait_source_image=(
                    args.portrait_source_image.resolve() if args.portrait_source_image is not None else None
                ),
                portrait_reference_image=(
                    args.portrait_reference_image.resolve() if args.portrait_reference_image is not None else None
                ),
            )

        report_path = output_dir / "wan_vace_reference_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        markdown_path = output_dir / "wan_vace_reference_report.md"
        markdown_path.write_text(WanVaceReferenceProbe._markdown_report(report))
        print(json.dumps(WanVaceReferenceProbe._compact_report(report), indent=2, sort_keys=True))
        print(f"wrote {report_path}")
        print(f"wrote {markdown_path}")

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=(
                "Run one-at-a-time upstream Wan VACE reference probes with saved artifacts and sampled memory metrics."
            )
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("validation_outputs/wan_vace_reference_2026_07_03"),
        )
        parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
        parser.add_argument("--diffusers-src", type=Path, default=Path("/Users/albou/projects/gh/diffusers/src"))
        parser.add_argument(
            "--device",
            choices=("auto", "cpu", "mps", "cuda"),
            default="auto",
            help="Execution device for the upstream Diffusers reference run. On macOS, auto uses CPU.",
        )
        parser.add_argument(
            "--portrait-source-image",
            type=Path,
            default=None,
            help="Optional portrait still to use for the portrait_hair_eyes case.",
        )
        parser.add_argument(
            "--portrait-reference-image",
            type=Path,
            default=None,
            help="Optional portrait reference still to use as explicit VACE reference conditioning.",
        )
        parser.add_argument(
            "--cases",
            nargs="*",
            choices=sorted(WanVaceReferenceProbe.case_presets()),
            default=sorted(WanVaceReferenceProbe.case_presets()),
        )
        parser.add_argument("--sample-interval-ms", type=int, default=250)
        parser.add_argument("--skip-prefetch", action="store_true")
        parser.add_argument("--internal-run", action="store_true")
        parser.add_argument("--manifest", type=Path)
        if parser.parse_known_args()[0].sample_interval_ms <= 0:
            raise ValueError("--sample-interval-ms must be greater than zero.")
        return parser.parse_args()

    @staticmethod
    def _prefetch_model(*, model_id: str, skip_prefetch: bool) -> Path:
        if skip_prefetch:
            return Path(snapshot_download(repo_id=model_id, local_files_only=True))
        return Path(snapshot_download(repo_id=model_id))

    @staticmethod
    def _run_case(
        *,
        case: CasePreset,
        model_path: Path,
        diffusers_src: Path | None,
        output_dir: Path,
        sample_interval_ms: int,
        device: str,
        portrait_source_image: Path | None,
        portrait_reference_image: Path | None,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = WanVaceReferenceProbe._prepare_case_inputs(
            case=case,
            output_dir=output_dir,
            portrait_source_image=portrait_source_image,
            portrait_reference_image=portrait_reference_image,
        )
        command_log_dir = output_dir / "run"
        command_log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = command_log_dir / "stdout.log"
        stderr_path = command_log_dir / "stderr.log"
        argv = [
            "/usr/bin/time",
            "-l",
            sys.executable,
            str(Path(__file__).resolve()),
            "--internal-run",
            "--manifest",
            str(manifest_path),
        ]
        env = os.environ.copy()
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "MFLUX_BENCHMARK_PARENT_PHYSICAL_SAMPLING": "1",
            }
        )
        os.environ["MFLUX_BENCHMARK_PARENT_PHYSICAL_SAMPLING"] = "1"
        if diffusers_src is not None:
            env["MFLUX_LOCAL_DIFFUSERS_SRC"] = str(diffusers_src)
        env["MFLUX_WAN_VACE_MODEL_PATH"] = str(model_path)
        env["MFLUX_WAN_VACE_DEVICE"] = device
        started = time.perf_counter()
        process = subprocess.Popen(
            argv,
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
            raise RuntimeError(f"{case.case_id} failed; see {stderr_path}")
        child_report_path = output_dir / "child_report.json"
        child_report = json.loads(child_report_path.read_text())
        return {
            "case": asdict(case),
            "kind": "upstream_reference_only",
            "manifest_path": str(manifest_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "wall_seconds": wall_seconds,
            "sampler": sampler.summary(),
            "samples": sampler.samples,
            "time_l": GenerationMemoryBenchmark._parse_time_l_output(stderr),
            "child_report": child_report,
        }

    @staticmethod
    def _prepare_case_inputs(
        *,
        case: CasePreset,
        output_dir: Path,
        portrait_source_image: Path | None = None,
        portrait_reference_image: Path | None = None,
    ) -> Path:
        source_video_path = output_dir / "source.mp4"
        source_contact_sheet_path = output_dir / "source_contact_sheet.png"
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_contact_sheet_path = output_dir / "mask_contact_sheet.png"
        if case.case_id == "portrait_hair_eyes":
            original_source_frames = WanVaceReferenceProbe._build_portrait_source_frames(
                case=case,
                source_image_path=portrait_source_image,
            )
            source_fps = case.fps
            mask_frames = [
                WanVaceReferenceProbe._build_portrait_mask(case.width, case.height) for _ in original_source_frames
            ]
        elif case.case_id == "ship_reactor_nacelles":
            original_source_frames = WanVaceReferenceProbe._read_ship_source_frames(case=case)
            source_fps = case.fps
            mask_frames = [
                WanVaceReferenceProbe._build_ship_mask_frame(
                    frame_index=index,
                    total_frames=len(original_source_frames),
                    width=case.width,
                    height=case.height,
                )
                for index in range(len(original_source_frames))
            ]
        else:
            raise ValueError(f"Unsupported case_id: {case.case_id}")

        source_frames = (
            WanVaceReferenceProbe._build_masked_source_video(
                source_frames=original_source_frames,
                mask_frames=mask_frames,
            )
            if case.use_masked_source_video
            else original_source_frames
        )

        WanVaceReferenceProbe._write_video(
            frames=source_frames,
            path=source_video_path,
            fps=source_fps,
            width=case.width,
            height=case.height,
        )
        WanVaceReferenceProbe._save_contact_sheet(
            frames=source_frames,
            output_path=source_contact_sheet_path,
            title=f"{case.case_id} conditioning source",
        )
        WanVaceReferenceProbe._write_mask_frames(mask_frames=mask_frames, mask_dir=mask_dir)
        WanVaceReferenceProbe._save_contact_sheet(
            frames=[frame.convert("RGB") for frame in mask_frames],
            output_path=mask_contact_sheet_path,
            title=f"{case.case_id} mask",
        )
        manifest = {
            "case": asdict(case),
            "kind": "upstream_reference_only",
            "note": "This probe exercises the upstream Diffusers Wan VACE reference pipeline, not native MLX-Gen support.",
            "model_path_env": "MFLUX_WAN_VACE_MODEL_PATH",
            "diffusers_src_env": "MFLUX_LOCAL_DIFFUSERS_SRC",
            "negative_prompt": case.negative_prompt,
            "portrait_source_image": str(portrait_source_image) if portrait_source_image is not None else None,
            "portrait_reference_image": str(portrait_reference_image) if portrait_reference_image is not None else None,
            "source_video_path": str(source_video_path),
            "source_contact_sheet_path": str(source_contact_sheet_path),
            "mask_dir": str(mask_dir),
            "mask_contact_sheet_path": str(mask_contact_sheet_path),
            "output_video_path": str(output_dir / "output.mp4"),
            "output_contact_sheet_path": str(output_dir / "output_contact_sheet.png"),
            "child_report_path": str(output_dir / "child_report.json"),
        }
        manifest_path = output_dir / "case_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest_path

    @staticmethod
    def _build_portrait_source_frames(
        *,
        case: CasePreset,
        source_image_path: Path | None = None,
    ) -> list[PIL.Image.Image]:
        source = PIL.Image.open(source_image_path or WanVaceReferenceProbe.PORTRAIT_SOURCE_IMAGE).convert("RGB")
        frames = []
        for index in range(case.num_frames):
            progress = index / max(case.num_frames - 1, 1)
            scale = 1.0 + 0.08 * progress
            drift_x = int(round(18 * progress))
            drift_y = int(round(-6 * progress))
            scaled = source.resize(
                (int(round(source.width * scale)), int(round(source.height * scale))),
                PIL.Image.Resampling.LANCZOS,
            )
            left = max((scaled.width - case.width) // 2 + drift_x, 0)
            headroom_bias = int(round(case.height * 0.18))
            top = max((scaled.height - case.height) // 2 - headroom_bias + drift_y, 0)
            crop = scaled.crop((left, top, left + case.width, top + case.height))
            frames.append(PIL.ImageOps.fit(crop, (case.width, case.height), method=PIL.Image.Resampling.LANCZOS))
        return frames

    @staticmethod
    def _read_ship_source_frames(*, case: CasePreset) -> list[PIL.Image.Image]:
        clip = VideoUtil.read_video_clip(
            WanVaceReferenceProbe.SHIP_SOURCE_VIDEO,
            start_seconds=case.start_seconds,
            max_frames=case.num_frames,
        )
        frames = []
        for frame in clip.frames:
            if frame.size != (case.width, case.height):
                frame = frame.resize((case.width, case.height), PIL.Image.Resampling.LANCZOS)
            frames.append(frame.convert("RGB"))
        return frames

    @staticmethod
    def _build_portrait_mask(width: int, height: int) -> PIL.Image.Image:
        mask = PIL.Image.new("L", (width, height), 0)
        draw = PIL.ImageDraw.Draw(mask)
        outer = (
            int(round(width * 0.18)),
            int(round(height * -0.02)),
            int(round(width * 0.82)),
            int(round(height * 0.26)),
        )
        band = (
            int(round(width * 0.22)),
            int(round(height * 0.10)),
            int(round(width * 0.78)),
            int(round(height * 0.20)),
        )
        inner = (
            int(round(width * 0.30)),
            int(round(height * 0.17)),
            int(round(width * 0.70)),
            int(round(height * 0.36)),
        )
        draw.ellipse(outer, fill=255)
        draw.rectangle(band, fill=255)
        draw.ellipse(inner, fill=0)
        draw.rectangle((0, int(round(height * 0.27)), width, height), fill=0)
        return mask

    @staticmethod
    def _build_ship_mask_frame(*, frame_index: int, total_frames: int, width: int, height: int) -> PIL.Image.Image:
        mask = PIL.Image.new("L", (width, height), 0)
        draw = PIL.ImageDraw.Draw(mask)
        progress = frame_index / max(total_frames - 1, 1)
        center_x = 118 + 210 * progress
        center_y = 146 - 102 * progress
        draw.ellipse(
            (
                int(center_x - 88),
                int(center_y - 44),
                int(center_x + 102),
                int(center_y + 52),
            ),
            fill=255,
        )
        draw.ellipse(
            (
                int(center_x - 34),
                int(center_y - 18),
                int(center_x + 64),
                int(center_y + 34),
            ),
            fill=255,
        )
        draw.rectangle(
            (
                int(center_x + 16),
                int(center_y - 10),
                int(min(width, center_x + 118)),
                int(min(height, center_y + 54)),
            ),
            fill=255,
        )
        return mask

    @staticmethod
    def _build_masked_source_video(
        *,
        source_frames: list[PIL.Image.Image],
        mask_frames: list[PIL.Image.Image],
    ) -> list[PIL.Image.Image]:
        masked_frames: list[PIL.Image.Image] = []
        gray_pixel = np.array([127, 127, 127], dtype=np.uint8)
        for source_frame, mask_frame in zip(source_frames, mask_frames):
            source_array = np.array(source_frame.convert("RGB"), dtype=np.uint8)
            mask_array = np.array(mask_frame.convert("L"), dtype=np.uint8)
            edit_region = mask_array > 127
            masked_array = source_array.copy()
            masked_array[edit_region] = gray_pixel
            masked_frames.append(PIL.Image.fromarray(masked_array, mode="RGB"))
        return masked_frames

    @staticmethod
    def _write_mask_frames(*, mask_frames: list[PIL.Image.Image], mask_dir: Path) -> None:
        for path in mask_dir.glob("*.png"):
            path.unlink()
        for index, frame in enumerate(mask_frames):
            frame.save(mask_dir / f"mask_{index:03d}.png")

    @staticmethod
    def _write_video(
        *,
        frames: list[PIL.Image.Image],
        path: Path,
        fps: int,
        width: int,
        height: int,
    ) -> None:
        with VideoStreamWriter(path=path, fps=fps, width=width, height=height, overwrite=True) as writer:
            writer.write_frames(frames)

    @staticmethod
    def _save_contact_sheet(*, frames: list[PIL.Image.Image], output_path: Path, title: str) -> None:
        sample_count = min(5, len(frames))
        indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int).tolist()
        selected = [frames[index].convert("RGB") for index in indices]
        thumb_width = 192
        thumb_height = max(1, int(round((selected[0].height / selected[0].width) * thumb_width)))
        header_height = 48
        padding = 12
        sheet = PIL.Image.new(
            "RGB",
            (
                padding * (sample_count + 1) + thumb_width * sample_count,
                header_height + padding * 2 + thumb_height,
            ),
            (18, 20, 26),
        )
        draw = PIL.ImageDraw.Draw(sheet)
        draw.text((padding, 12), title, fill=(240, 242, 247))
        for column, image in enumerate(selected):
            thumb = PIL.ImageOps.fit(image, (thumb_width, thumb_height), method=PIL.Image.Resampling.LANCZOS)
            x = padding + column * (thumb_width + padding)
            y = header_height
            sheet.paste(thumb, (x, y))
        sheet.save(output_path)

    @staticmethod
    def _internal_run(args: argparse.Namespace) -> None:
        if args.manifest is None:
            raise ValueError("--manifest is required for --internal-run.")
        manifest = json.loads(args.manifest.read_text())
        model_path = Path(os.environ[manifest["model_path_env"]])
        diffusers_src = os.environ.get(manifest["diffusers_src_env"])
        case = CasePreset(**manifest["case"])
        if diffusers_src and diffusers_src not in sys.path:
            sys.path.insert(0, diffusers_src)
        source_frames = VideoUtil.read_video_clip(
            manifest["source_video_path"],
            max_frames=case.num_frames,
        ).frames
        mask_frames = [
            PIL.Image.open(path).convert("L")
            for path in sorted(Path(manifest["mask_dir"]).glob("mask_*.png"))
        ]
        reference_images = WanVaceReferenceProbe._reference_images_for_case(
            case=case,
            manifest=manifest,
            source_frames=source_frames,
        )
        report = WanVaceReferenceProbe._run_reference_generation(
            case=case,
            model_path=model_path,
            source_frames=source_frames,
            mask_frames=mask_frames,
            reference_images=reference_images,
            negative_prompt=manifest["negative_prompt"],
            output_video_path=Path(manifest["output_video_path"]),
            output_contact_sheet_path=Path(manifest["output_contact_sheet_path"]),
        )
        Path(manifest["child_report_path"]).write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))

    @staticmethod
    def _reference_images_for_case(
        *,
        case: CasePreset,
        manifest: dict[str, Any],
        source_frames: list[PIL.Image.Image],
    ) -> list[PIL.Image.Image] | None:
        explicit_reference_path = manifest.get("portrait_reference_image")
        if case.case_id == "portrait_hair_eyes" and explicit_reference_path is not None:
            return [PIL.Image.open(explicit_reference_path).convert("RGB")]
        if case.use_first_frame_reference:
            return [source_frames[0].copy()]
        return None

    @staticmethod
    def _run_reference_generation(
        *,
        case: CasePreset,
        model_path: Path,
        source_frames: list[PIL.Image.Image],
        mask_frames: list[PIL.Image.Image],
        reference_images: list[PIL.Image.Image] | None,
        negative_prompt: str,
        output_video_path: Path,
        output_contact_sheet_path: Path,
    ) -> dict[str, Any]:
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        device = WanVaceReferenceProbe._reference_device(torch)
        transformer_dtype = WanVaceReferenceProbe._transformer_dtype(torch, device)
        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanVACEPipeline.from_pretrained(model_path, vae=vae, torch_dtype=transformer_dtype)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=case.flow_shift)
        pipe.to(device)
        started = time.perf_counter()
        generator = torch.Generator(device="cpu").manual_seed(case.seed)
        result = pipe(
            prompt=case.prompt,
            negative_prompt=negative_prompt,
            video=source_frames,
            mask=mask_frames,
            reference_images=reference_images,
            height=case.height,
            width=case.width,
            num_frames=case.num_frames,
            num_inference_steps=case.steps,
            guidance_scale=case.guidance,
            generator=generator,
            output_type="pil",
        )
        wall_seconds = time.perf_counter() - started
        output_frames = [frame.convert("RGB") for frame in result.frames[0]]
        WanVaceReferenceProbe._write_video(
            frames=output_frames,
            path=output_video_path,
            fps=case.fps,
            width=case.width,
            height=case.height,
        )
        WanVaceReferenceProbe._save_contact_sheet(
            frames=output_frames,
            output_path=output_contact_sheet_path,
            title=f"{case.case_id} output",
        )
        output_info = VideoUtil.inspect_video(output_video_path)
        return {
            "case": asdict(case),
            "kind": "upstream_reference_only",
            "note": "This run used the upstream Diffusers Wan VACE pipeline, not native MLX-Gen support.",
            "model_path": str(model_path),
            "device": device,
            "transformer_dtype": str(transformer_dtype),
            "output_video_path": str(output_video_path),
            "output_contact_sheet_path": str(output_contact_sheet_path),
            "output_video": {
                "fps": output_info.fps,
                "source_width": output_info.source_width,
                "source_height": output_info.source_height,
                "source_frame_count": output_info.source_frame_count,
                "source_duration_seconds": output_info.source_duration_seconds,
                "audio_present": output_info.audio_present,
            },
            "wall_seconds": wall_seconds,
        }

    @staticmethod
    def _transformer_dtype(torch_module, device: str):
        requested = os.environ.get("MFLUX_WAN_VACE_TRANSFORMER_DTYPE", "").strip().lower()
        if requested == "float32":
            return torch_module.float32
        if requested == "bfloat16":
            return torch_module.bfloat16
        if requested == "float16":
            return torch_module.float16
        return torch_module.float16 if device == "mps" else torch_module.bfloat16 if device != "cpu" else torch_module.float32

    @staticmethod
    def _reference_device(torch_module) -> str:
        requested = os.environ.get("MFLUX_WAN_VACE_DEVICE", "auto").strip().lower() or "auto"
        if requested == "auto":
            if platform.system() == "Darwin":
                return "cpu"
            if torch_module.cuda.is_available():
                return "cuda"
            if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda":
            if not torch_module.cuda.is_available():
                raise ValueError("Requested Wan VACE reference device cuda, but CUDA is not available.")
            return "cuda"
        if requested == "mps":
            if not hasattr(torch_module.backends, "mps") or not torch_module.backends.mps.is_available():
                raise ValueError("Requested Wan VACE reference device mps, but MPS is not available.")
            return "mps"
        if requested == "cpu":
            return "cpu"
        raise ValueError(f"Unsupported Wan VACE reference device {requested!r}.")

    @staticmethod
    def _markdown_report(report: dict[str, Any]) -> str:
        lines = [
            "# Wan VACE Reference Report",
            "",
            "This report covers the upstream Diffusers Wan VACE reference pipeline only. It does not mean MLX-Gen "
            "currently ships native prompt-guided video editing.",
            "On macOS, the reference tool defaults to CPU because the measured MPS float16 path produced invalid washed-out outputs on the bounded portrait case.",
            "",
            f"- Created: {report['created_at']}",
            f"- Model: `{report['model_id']}`",
            f"- Local model path: `{report['model_path']}`",
            "",
        ]
        for case_id, data in report["cases"].items():
            child = data["child_report"]
            sampler = data["sampler"]
            time_l = data["time_l"]
            lines.extend(
                [
                    f"## {case_id}",
                    "",
                    f"- Prompt: {child['case']['prompt']}",
                    f"- Size: `{child['case']['width']}x{child['case']['height']}`",
                    f"- Frames / FPS: `{child['case']['num_frames']} / {child['case']['fps']}`",
                    f"- Steps: `{child['case']['steps']}`",
                    f"- Wall time: `{round(float(data['wall_seconds']), 4)}` seconds",
                    f"- Peak sampled RSS: `{WanVaceReferenceProbe._gb(sampler.get('peak_sampled_rss_bytes'))}` GB",
                    f"- Peak sampled physical footprint: `{WanVaceReferenceProbe._gb(sampler.get('peak_sampled_darwin_physical_footprint_bytes'))}` GB",
                    f"- time -l max RSS: `{WanVaceReferenceProbe._gb(time_l.get('maximum_resident_set_size_bytes'))}` GB",
                    f"- Source contact sheet: `{json.loads(Path(data['manifest_path']).read_text())['source_contact_sheet_path']}`",
                    f"- Mask contact sheet: `{json.loads(Path(data['manifest_path']).read_text())['mask_contact_sheet_path']}`",
                    f"- Output video: `{child['output_video_path']}`",
                    f"- Output contact sheet: `{child['output_contact_sheet_path']}`",
                    "",
                ]
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
        compact = {}
        for case_id, data in report["cases"].items():
            child = data["child_report"]
            compact[case_id] = {
                "wall_seconds": round(float(data["wall_seconds"]), 4),
                "peak_sampled_rss_gb": WanVaceReferenceProbe._gb(data["sampler"].get("peak_sampled_rss_bytes")),
                "peak_sampled_physical_gb": WanVaceReferenceProbe._gb(
                    data["sampler"].get("peak_sampled_darwin_physical_footprint_bytes")
                ),
                "time_l_max_rss_gb": WanVaceReferenceProbe._gb(
                    data["time_l"].get("maximum_resident_set_size_bytes")
                ),
                "output_video_path": child["output_video_path"],
                "output_contact_sheet_path": child["output_contact_sheet_path"],
            }
        return compact

    @staticmethod
    def _gb(value: int | None) -> float | None:
        if value is None:
            return None
        return round(value / BYTES_PER_GB, 4)


if __name__ == "__main__":
    WanVaceReferenceProbe.main()
