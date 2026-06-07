import os
from pathlib import Path

import platformdirs

BATTERY_PERCENTAGE_STOP_LIMIT = 5
CONTROLNET_STRENGTH = 0.4
DEFAULT_DEV_FILL_GUIDANCE = 30
DEFAULT_DEPTH_GUIDANCE = 10
DIMENSION_STEP_PIXELS = 16
GUIDANCE_SCALE = 3.5
GUIDANCE_SCALE_KONTEXT = 2.5
HEIGHT, WIDTH = 1024, 1024
IMAGE_STRENGTH = 0.4
MODEL_CHOICES = [
    "dev",
    "schnell",
    "krea-dev",
    "dev-krea",
    "qwen",
    "qwen-image",
    "qwen-image-edit",
    "qwen-image-edit-2509",
    "qwen-image-edit-2511",
    "qwen-edit",
    "qwen-edit-plus",
    "qwen-edit-2509",
    "qwen-edit-2511",
    "fibo",
    "fibo-lite",
    "fibo-edit",
    "fibo-edit-rmbg",
    "z-image",
    "z-image-turbo",
    "ernie-image-turbo",
    "seedvr2",
    "seedvr2-3b",
    "seedvr2-7b",
    "wan2.2-ti2v-5b",
    "bonsai-image-ternary",
    "bonsai-image-binary",
    "flux2-klein-4b",
    "flux2-klein-9b",
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
]
MODEL_INFERENCE_STEPS = {
    "dev": 25,
    "schnell": 4,
    "krea-dev": 25,
    "qwen": 20,
    "qwen-image": 20,
    "qwen-image-edit": 20,
    "qwen-image-edit-2509": 40,
    "qwen-image-edit-2511": 40,
    "qwen-edit": 20,
    "qwen-edit-plus": 40,
    "qwen-edit-2509": 40,
    "qwen-edit-2511": 40,
    "fibo": 50,
    "fibo-lite": 8,
    "fibo-edit": 50,
    "fibo-edit-rmbg": 10,
    "z-image": 50,
    "z-image-turbo": 9,
    "ernie-image-turbo": 8,
    "wan2.2-ti2v-5b": 50,
    "bonsai-image-ternary": 4,
    "bonsai-image-binary": 4,
    "flux2-klein-4b": 4,
    "flux2-klein-9b": 4,
    "flux2-klein-base-4b": 50,
    "flux2-klein-base-9b": 50,
}
QUANTIZE_CHOICES = [3, 5, 4, 6, 8]

if os.environ.get("MFLUX_CACHE_DIR"):
    MFLUX_CACHE_DIR = Path(os.environ["MFLUX_CACHE_DIR"]).resolve()
else:
    MFLUX_CACHE_DIR = Path(platformdirs.user_cache_dir(appname="mflux"))

MFLUX_LORA_CACHE_DIR = MFLUX_CACHE_DIR / "loras"
