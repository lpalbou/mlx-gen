import re
import subprocess
import sys
from pathlib import Path

import mlx.core as mx

from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil
from mflux.models.z_image.variants.z_image import ZImage


def test_add_noise_by_interpolation_preserves_latent_dtype():
    # Scheduler sigmas are 0-d float32 arrays; MLX promotion must not leak f32 into the loop.
    clean = mx.zeros((1, 64, 16), dtype=mx.bfloat16)
    noise = mx.ones((1, 64, 16), dtype=mx.bfloat16)
    f32_sigma = mx.array([0.85], dtype=mx.float32)[0]
    for sigma in (f32_sigma, 0.85):
        result = LatentCreator.add_noise_by_interpolation(clean=clean, noise=noise, sigma=sigma)
        assert result.dtype == mx.bfloat16, f"sigma={sigma!r} promoted latents to {result.dtype}"
    f32_clean = mx.zeros((1, 64, 16), dtype=mx.float32)
    result = LatentCreator.add_noise_by_interpolation(clean=f32_clean, noise=noise, sigma=f32_sigma)
    assert result.dtype == mx.float32


def test_qwen_blend_inpaint_latents_preserves_latent_dtype():
    latents = mx.zeros((1, 64, 16), dtype=mx.bfloat16)
    image_latents = mx.zeros((1, 64, 16), dtype=mx.bfloat16)
    initial_noise = mx.ones((1, 64, 16), dtype=mx.bfloat16)
    f32_mask = mx.ones((1, 64, 16), dtype=mx.float32)
    result = QwenEditUtil.blend_inpaint_latents(
        latents=latents,
        image_latents=image_latents,
        initial_noise=initial_noise,
        mask_latents=f32_mask,
        sigma=mx.array([0.5], dtype=mx.float32)[0],
    )
    assert result.dtype == mx.bfloat16


def test_z_image_blend_inpaint_latents_preserves_latent_dtype():
    latents = mx.zeros((16, 1, 8, 8), dtype=mx.bfloat16)
    image_latents = mx.zeros((16, 1, 8, 8), dtype=mx.bfloat16)
    initial_noise = mx.ones((16, 1, 8, 8), dtype=mx.bfloat16)
    f32_mask = mx.ones((1, 1, 8, 8), dtype=mx.float32)
    result = ZImage._blend_inpaint_latents(
        latents=latents,
        image_latents=image_latents,
        initial_noise=initial_noise,
        mask_latents=f32_mask,
        sigma=mx.array([0.5], dtype=mx.float32)[0],
    )
    assert result.dtype == mx.bfloat16


def test_cli_entry_points_do_not_import_transformers_at_module_scope():
    # transformers pulls in torch (~0.6 s); it must stay off the CLI startup path of every
    # console script and only load when a tokenizer is actually constructed.
    # tomllib needs Python 3.11+; the entry-point table format is stable enough for a regex.
    pyproject_text = (Path(__file__).parents[2] / "pyproject.toml").read_text()
    scripts_section = re.search(r"\[project\.scripts\](.*?)(?:\n\[|\Z)", pyproject_text, re.DOTALL).group(1)
    modules = sorted({m.group(1) for m in re.finditer(r'=\s*"([\w.]+):[\w.]+"', scripts_section)})
    assert len(modules) >= 20, f"expected the full entry-point set, found {modules}"
    imports = "; ".join(f"import {module}" for module in modules)
    code = (
        f"import sys; {imports}; "
        "leaked = [m for m in ('transformers', 'torch') if m in sys.modules]; "
        "print(','.join(leaked)); "
        "sys.exit(1 if leaked else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"transformers/torch leaked into a CLI entry-point import path: "
        f"{result.stdout.strip() or result.stderr}"
    )
