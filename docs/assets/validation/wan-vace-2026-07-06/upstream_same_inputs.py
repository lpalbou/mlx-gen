"""A/B check: run the upstream diffusers WanVACEPipeline with EXACTLY the MLX case-1 inputs
(static union mask, 17 frames, first-frame reference, short negative prompt, seed 7302).

If upstream is also a weak edit -> the case design is the problem, not the port.
If upstream edits strongly -> the port has a behavioral divergence parity missed.

Usage:
  PYTHONPATH=/Users/albou/projects/gh/diffusers/src uv run --with accelerate --with ftfy \
      python validation_outputs/wan_vace_mlx_2026_07_06/upstream_same_inputs.py
"""

import sys
import time
from pathlib import Path

import PIL.Image

sys.path.insert(0, "src")

from mflux.utils.video_util import VideoUtil  # noqa: E402

ROOT = Path("validation_outputs/wan_vace_mlx_2026_07_06")
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B-diffusers/snapshots/ec4d2cb062b548996b179d493fdd05340de702a1"
)
PROMPT = (
    "Keep the same icy cliffs, snow haze, soft sunrise lighting, and lift-off camera motion. "
    "Transform the ship into a bulkier smuggler-style starship with a bright circular rear reactor "
    "and two side nacelles while preserving realistic vehicle detail."
)
NEGATIVE = "duplicate ships, warped hull, melted nacelles, unreadable reactor, washed out frame, blown highlights"


def main() -> None:
    import torch
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    frames = VideoUtil.read_video_clip(
        "docs/assets/examples/spaceship-snow/06_i2v_a14b_spaceship_takeoff_from_source.mp4",
        max_frames=17,
        target_fps=10.0,
    ).frames[:17]
    frames = [f.resize((448, 256), PIL.Image.Resampling.LANCZOS) for f in frames]
    static_mask = PIL.Image.open(ROOT / "ship_union_mask.png").convert("L")
    masks = [static_mask.copy() for _ in range(17)]
    reference = [frames[0].copy()]

    vae = AutoencoderKLWan.from_pretrained(SNAPSHOT, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(SNAPSHOT, vae=vae, torch_dtype=torch.float32)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.to("cpu")

    started = time.perf_counter()
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        video=frames,
        mask=masks,
        reference_images=reference,
        height=256,
        width=448,
        num_frames=17,
        num_inference_steps=16,
        guidance_scale=3.5,
        generator=torch.Generator(device="cpu").manual_seed(7302),
    ).frames[0]
    print(f"upstream same-inputs run: {time.perf_counter() - started:.1f}s")
    export_to_video(result, str(ROOT / "upstream_same_inputs_static_mask.mp4"), fps=10)
    print("saved", ROOT / "upstream_same_inputs_static_mask.mp4")


if __name__ == "__main__":
    main()
