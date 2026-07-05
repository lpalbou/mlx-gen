"""Motion-fidelity measurement for the Wan V2V strength ladder.

Reproduces the adversarial-forensics methodology: per-frame-pair motion energy (mean absolute
grayscale difference at 120 px width) split by person/background region, gesture-window
retention ratio and Pearson correlation against the source, plus warm-start sigma computed from
the real scheduler.

Usage:
  uv run python validation_outputs/motion_fidelity_ladder_2026_07_05/measure_motion.py \
      --source <source.mp4> --outputs <run1.mp4> <run2.mp4> ... \
      --person-mask <mask.png> --gesture-start-pair 8 --json <metrics.json>
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ANALYSIS_WIDTH = 120
SETTLE_PAIRS = 2


def extract_gray_frames(video_path: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video_path), f"{tmp}/f%04d.png"],
            check=True,
        )
        frames = []
        for png in sorted(Path(tmp).glob("f*.png")):
            image = Image.open(png).convert("L")
            height = round(image.height * ANALYSIS_WIDTH / image.width)
            frames.append(np.asarray(image.resize((ANALYSIS_WIDTH, height), Image.Resampling.BOX), dtype=np.float32))
        return np.stack(frames)


def load_region_mask(mask_path: Path, shape: tuple[int, int], dilate: int = 0) -> np.ndarray:
    image = Image.open(mask_path).convert("L").resize((shape[1], shape[0]), Image.Resampling.BOX)
    mask = (np.asarray(image, dtype=np.float32) / 255.0) >= 0.5
    for _ in range(dilate):
        padded = np.pad(mask, 1, mode="edge")
        mask = padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:] | mask
    return mask


def motion_series(frames: np.ndarray, region: np.ndarray | None) -> np.ndarray:
    diffs = np.abs(np.diff(frames, axis=0))
    if region is None:
        return diffs.mean(axis=(1, 2))
    weights = region.astype(np.float32)
    denom = max(weights.sum(), 1.0)
    return (diffs * weights).sum(axis=(1, 2)) / denom


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def series_metrics(source: np.ndarray, output: np.ndarray, gesture_start: int) -> dict:
    n = min(len(source), len(output))
    source, output = source[:n], output[:n]
    gesture_source = source[gesture_start:]
    gesture_output = output[gesture_start:]
    return {
        "series_source": [round(float(v), 3) for v in source],
        "series_output": [round(float(v), 3) for v in output],
        "mean_ratio": round(float(output.mean() / source.mean()), 3),
        "gesture_window_ratio": round(float(gesture_output.mean() / gesture_source.mean()), 3),
        "pearson_r_after_settle": round(pearson(source[SETTLE_PAIRS:], output[SETTLE_PAIRS:]), 3),
        "gesture_window_pearson_r": round(pearson(gesture_source, gesture_output), 3),
    }


def warm_start_sigma(num_steps: int, flow_shift: float, strength: float, boundary_timestep: float = 875.0) -> dict:
    # Mirrors Wan2_2_TI2V._video_to_video_timesteps exactly (incl. the max(...,1) floor).
    from mflux.models.wan.scheduler import WanUniPCMultistepScheduler

    scheduler = WanUniPCMultistepScheduler(flow_shift=flow_shift)
    scheduler.set_timesteps(num_steps)
    init_timestep = min(max(int(num_steps * strength), 1), num_steps)
    t_start = max(num_steps - init_timestep, 0)
    begin_index = t_start * getattr(scheduler, "order", 1)
    sigma = float(scheduler.sigmas[begin_index])
    timesteps = scheduler.timesteps.tolist()[begin_index:]
    return {
        "steps": num_steps,
        "flow_shift": flow_shift,
        "strength": strength,
        "effective_steps": len(timesteps),
        # A14B routes steps at/above the boundary (boundary_ratio 0.875 x 1000) to the
        # high-noise expert; strength therefore co-varies sigma AND expert routing.
        "high_noise_steps": sum(1 for t in timesteps if t >= boundary_timestep),
        "warm_start_sigma": round(sigma, 4),
        "source_signal_amplitude_percent": round((1.0 - sigma) * 100.0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--outputs", required=True, nargs="+", type=Path)
    parser.add_argument("--person-mask", required=True, type=Path)
    parser.add_argument("--gesture-start-pair", type=int, default=8)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    source_frames = extract_gray_frames(args.source)
    person = load_region_mask(args.person_mask, source_frames.shape[1:])
    # Dilated variant guards against gesture pixels escaping the mask edge (sanity check only).
    person_dilated = load_region_mask(args.person_mask, source_frames.shape[1:], dilate=8)
    background = ~person

    source_person = motion_series(source_frames, person)
    source_person_dilated = motion_series(source_frames, person_dilated)
    source_background = motion_series(source_frames, background)
    source_whole = motion_series(source_frames, None)

    results = {
        "analysis_width": ANALYSIS_WIDTH,
        "gesture_start_pair": args.gesture_start_pair,
        "significance_note": "n=22 pairs after settle: |r| below ~0.42 is indistinguishable from zero at p=0.05",
        "runs": {},
    }
    for output_path in args.outputs:
        output_frames = extract_gray_frames(output_path)
        results["runs"][output_path.name] = {
            "person": series_metrics(source_person, motion_series(output_frames, person), args.gesture_start_pair),
            "person_dilated": series_metrics(
                source_person_dilated, motion_series(output_frames, person_dilated), args.gesture_start_pair
            ),
            "background": series_metrics(
                source_background, motion_series(output_frames, background), args.gesture_start_pair
            ),
            "whole": series_metrics(source_whole, motion_series(output_frames, None), args.gesture_start_pair),
        }

    results["sigma_table"] = {
        "cfg_20_steps_shift3": [warm_start_sigma(20, 3.0, s) for s in (0.5, 0.6, 0.7, 0.8, 0.9)],
        "lightning_4_steps_shift5": [warm_start_sigma(4, 5.0, s) for s in (0.7, 0.75, 0.8)],
    }

    text = json.dumps(results, indent=2)
    if args.json is not None:
        args.json.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
