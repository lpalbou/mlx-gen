# Completed: SeedVR2 upscale smoke, metadata, and quality defaults

## Metadata

- Created: 2026-06-07
- Status: Completed
- Completed: 2026-06-07

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
- ADR impact: None. This confirms an existing route and fixes metadata drift; it does not add a new
  package-wide policy.

## Context

MLX-Gen already includes SeedVR2 as a dedicated diffusion upscaler through
`mflux-upscale-seedvr2`. The broader model roadmap still tracks whether SeedVR2 should be wired
into unified `mlxgen prepare`, but the dedicated command should remain usable and accurately
describe generated artifacts.

## Current code reality

- `pyproject.toml` exposes `mflux-upscale-seedvr2`.
- `src/mflux/models/seedvr2/cli/seedvr2_upscale.py` routes the CLI to
  `SeedVR2.generate_image(...)`.
- `src/mflux/models/seedvr2/variants/upscale/seedvr2.py` pads working tensors to multiples of 16
  but crops the saved image back to the exact even output size.
- SeedVR2 now defaults to untiled VAE encode/decode for image quality. `--vae-tiling` opts into
  the tiled memory-saving path when needed for very large upscales.
- The local Hugging Face cache contains `numz/SeedVR2_comfyUI` 3B, 7B, and VAE weights.

## Problem

The upscaler worked, but the first smoke used `--resolution 256`, which means "make the shortest
edge 256px" rather than "scale by 2x". That demonstrated restoration/denoising more clearly than
true scale-factor upscaling. Metadata for non-16-multiple target sizes could also report the
rounded-down generic `Config` dimensions instead of the actual saved image dimensions. The source
image path and source dimensions were omitted from saved metadata, and the SeedVR2 CLI exposed
`--metadata` without passing it to `GeneratedImage.save(...)`.

After the first real-image pass, upper smooth regions still preserved visible source grain when
the input was visibly noisy. The current command surface now keeps VAE tiling off by default for
quality and documents `--softness 0.25` to `0.5` as the first control for noisy low-resolution
sources.

## What changed

- SeedVR2 now creates its generation `Config` with `dimension_multiple=2`, matching its final
  even-size output contract.
- SeedVR2 now passes the source image path to `ImageUtil.to_image(...)`, so saved metadata includes
  image-derived fields.
- `mflux-upscale-seedvr2 --metadata` now writes the JSON metadata sidecar.
- SeedVR2 now defaults to untiled VAE encode/decode; `mflux-upscale-seedvr2 --vae-tiling` restores
  the tiled path as an explicit memory-saving choice.
- The SeedVR2 docs now explain shortest-edge sizing, scale-factor sizing, `--softness`, and the
  tiling tradeoff.
- Fast regression tests cover the exact non-16-multiple output case and the CLI metadata save path
  without loading real weights.

## Validation

- `uv run pytest tests/image_generation/test_seedvr2_upscale_metadata.py tests/arg_parser/test_seedvr2_upscale_argparser.py tests/metadata/test_generated_image.py -q`
- Real model-backed target-short-edge smoke:

```sh
/usr/bin/time -l uv run mflux-upscale-seedvr2 \
  --image-path tests/resources/low_res.jpg \
  --resolution 256 \
  --seed 42 \
  --quantize 8 \
  --output validation_outputs/seedvr2_upscale_2026_06_07/low_res_seedvr2_3b_q8_256_fixed.png \
  --replace
```

The target-short-edge output is `426x256`, preserves the source aspect ratio from the `320x192` input, and its
metadata records `width=426`, `height=256`, `image_path=tests/resources/low_res.jpg`,
`source_image_width=320`, and `source_image_height=192`.

- Real model-backed scale-factor smokes:

```sh
uv run mflux-upscale-seedvr2 \
  --image-path tests/resources/low_res.jpg \
  --resolution 2x \
  --seed 42 \
  --quantize 8 \
  --metadata \
  --output validation_outputs/seedvr2_upscale_2026_06_07/low_res_seedvr2_3b_q8_2x_metadata.png \
  --replace

uv run mflux-upscale-seedvr2 \
  --image-path tests/resources/low_res.jpg \
  --resolution 3x \
  --seed 42 \
  --quantize 8 \
  --output validation_outputs/seedvr2_upscale_2026_06_07/low_res_seedvr2_3b_q8_3x.png \
  --replace
```

The scale-factor outputs were `640x384` for `2x` and `960x576` for `3x` from the same `320x192`
source. The `2x` metadata sidecar records `width=640`, `height=384`, `requested_width=640`,
`requested_height=384`, `source_image_width=320`, and `source_image_height=192`.

- Real model-backed quality-default checks:

```sh
uv run mflux-upscale-seedvr2 \
  --image-path tests/resources/low_res.jpg \
  --resolution 2x \
  --quantize 8 \
  --seed 42 \
  --metadata \
  --replace \
  --output validation_outputs/seedvr2_upscale_2026_06_07/low_res_seedvr2_3b_q8_2x_cli_default_no_tiling.png

uv run mflux-upscale-seedvr2 \
  --image-path tests/resources/low_res.jpg \
  --resolution 2x \
  --quantize 8 \
  --seed 42 \
  --vae-tiling \
  --metadata \
  --replace \
  --output validation_outputs/seedvr2_upscale_2026_06_07/low_res_seedvr2_3b_q8_2x_cli_vae_tiling.png
```

Both commands produced `640x384` outputs and metadata sidecars. The comparison artifact
`validation_outputs/seedvr2_upscale_2026_06_07/seedvr2_upper_region_after_fix.jpg` shows source,
old tiled behavior, new default no-tiling behavior, explicit tiling, and the recommended
`--softness 0.5` noisy-source profile.

## Expected outcomes

- `mflux-upscale-seedvr2` remains usable with the cached SeedVR2 3B model.
- `--resolution 2x` and `--resolution 3x` produce true scale-factor outputs.
- Saved metadata matches the actual image dimensions for SeedVR2 outputs.
- The source image relationship is visible in metadata for upscaled images.
- The default command path prioritizes image quality by avoiding VAE tiling unless explicitly
  requested.

## Follow-ups

- The dedicated SeedVR2 command works, and completed item 0031 adds official 3B source loading plus
  q8/q4 `mlxgen prepare` package support. Future SeedVR2 work should focus on broader quality
  validation or separate official 7B/sharp support.
