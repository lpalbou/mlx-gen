import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mflux.models.wan.variants import Wan2_2_TI2V
from mflux.utils.video_util import VideoUtil

RUN_A14B_I2V_ENV = "MFLUX_RUN_LOCAL_WAN_A14B_I2V"
IMAGE_ENV = "MFLUX_WAN_A14B_I2V_IMAGE"
OUTPUT_DIR_ENV = "MFLUX_WAN_A14B_I2V_OUTPUT_DIR"

DEFAULT_IMAGE = "docs/assets/i2v_takeoff_source.png"
DEFAULT_OUTPUT_DIR = "validation_outputs/wan"
DEFAULT_OUTPUT_NAME = "wan_a14b_i2v_takeoff_pytest_512x288_9f_8steps.mp4"

pytestmark = pytest.mark.skipif(
    os.getenv(RUN_A14B_I2V_ENV) != "1",
    reason=f"set {RUN_A14B_I2V_ENV}=1 to run local Wan A14B I2V generation tests",
)


def test_wan_a14b_i2v_generates_decodable_motion_video():
    image_path = Path(os.getenv(IMAGE_ENV, DEFAULT_IMAGE))
    if not image_path.exists():
        pytest.skip(f"Wan A14B I2V source image does not exist: {image_path}")

    output_dir = Path(os.getenv(OUTPUT_DIR_ENV, DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / DEFAULT_OUTPUT_NAME

    model = Wan2_2_TI2V(model_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    video = model.generate_video(
        seed=1701,
        prompt=(
            "The spacecraft lifts off from a snowy landing field, engines firing underneath, "
            "snow and ice blowing outward, cinematic science fiction motion"
        ),
        negative_prompt="static still image, frozen ship, no motion, blurry, low quality",
        width=512,
        height=288,
        num_frames=9,
        num_inference_steps=8,
        fps=8,
        guidance=4,
        guidance_2=3,
        image_path=image_path,
    )
    video.save(output_path, export_json_metadata=True, overwrite=True)

    assert output_path.exists()
    assert _video_codec_name(output_path) in (None, "h264")

    frame_0 = VideoUtil.extract_frame(output_path, 0)
    frame_8 = VideoUtil.extract_frame(output_path, 8)
    assert frame_0.size == (512, 288)
    assert frame_8.size == (512, 288)

    first = np.array(frame_0).astype(np.float32)
    last = np.array(frame_8).astype(np.float32)
    assert _mean_abs_delta(first, last) > 1.0
    assert not _looks_like_green_decoder_failure(first)
    assert not _looks_like_green_decoder_failure(last)


def _video_codec_name(path: Path) -> str | None:
    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def _mean_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


def _looks_like_green_decoder_failure(frame: np.ndarray) -> bool:
    red, green, blue = frame.mean(axis=(0, 1))
    return green > red * 2.0 and green > blue * 2.0
