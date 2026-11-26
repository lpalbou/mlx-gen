# Restoration And Upscaling (`mlxgen upscale`)

Open this file to restore or upscale an image or a video. `mlxgen upscale` hosts two promptless
families, selected with `--model`. Their routes are the `restoration` array of
`mlxgen capabilities --model <id>` ([capabilities-schema.md](capabilities-schema.md)).

| | SeedVR2 | SwiftVR |
| --- | --- | --- |
| Ids | `seedvr2-3b` (alias `seedvr2`, the default), `seedvr2-7b`, `seedvr2-7b-sharp`, `ByteDance-Seed/SeedVR2-3B`, `ByteDance-Seed/SeedVR2-7B`, `AbstractFramework/seedvr2-{3b,7b}-{8bit,4bit}`, a local package | `swiftvr`, `swiftvr-5b` |
| Images | yes | no |
| Video | yes | yes |
| Scaling | yes | source resolution only (`1x`) |
| Quantized packages | 8-bit and 4-bit | none (BF16) |
| Character | Diffusion restoration: recovers texture, slow on video | One step per chunk: far faster, smooths texture |

Choose SwiftVR for long material, previews, or throughput-bound batches; SeedVR2 7B for archival
or grain-sensitive material where fidelity matters most; SeedVR2 for every still. SeedVR2 works
best on degraded, noisy, low-resolution, or compressed sources and can over-smooth clean
high-resolution footage.

## Images (SeedVR2)

```sh
mlxgen download --model "AbstractFramework/seedvr2-3b-8bit"
mlxgen upscale \
  --model "AbstractFramework/seedvr2-3b-8bit" \
  --image-path "<INPUT.png>" \
  --resolution 2x \
  --softness 0.25 \
  --metadata \
  --output "<UPSCALED.png>"
```

- `--resolution`: an integer sets the shorter output edge (`1024`); `2x`, `3x`, `5x` multiply the
  source. Default for images: `384`. The aspect ratio is always preserved; a `320x192` source
  becomes `640x384` at `2x` and `1706x1024` at `1024`. `1x` returns the input dimensions exactly.
- `--softness 0.0` to `1.0` (default `0.0`): smooths the source before reconstruction. Use `0.25`
  to `0.5` when the source has visible grain or JPEG texture in smooth areas; higher softens detail.
- `--steps`: automatic by default (the official single step, plus a 4-step refinement only when
  the output shows one-step noise texture on flat or dark content). `1` to `4` forces a count.
  The sidecar records `steps_mode` and `one_step_residue_pct`.
- `--color-correction wavelet|lab|off` (default `wavelet`).
- Large outputs use tiled VAE decode automatically; `--vae-tiling` forces tiling for image runs.
- `--image-path` accepts several files or directories. With several sources the source stem is
  appended to the output name, or use `{input_name}` in `--output`.
- Source models can take `--quantize 8`; published packages need no flag.
- Measured: 5x from a 133x113 source peaks at 4.73 GiB (3B 8-bit) and 8.90 GiB (7B 8-bit).

## Video (SeedVR2)

Confirm with the user first: SeedVR2 video is slow and memory-hungry. Validate on about five
seconds of the real clip before a long run. The plan admits output frames up to about 1.4
megapixels on a 48 GB budget (1280x720 is admitted, 1920x1080 is refused before any load with the
largest size that fits); see [SeedVR2 video memory](https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/seedvr2-video-memory.md).

```sh
mlxgen upscale \
  --model "ByteDance-Seed/SeedVR2-3B" \
  --video-path "<INPUT.mp4>" \
  --start-seconds 0 \
  --max-frames 149 \
  --resolution 1x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 29 \
  --temporal-chunk-overlap 8 \
  --low-ram \
  --mlx-cache-limit-gb 8 \
  --metadata \
  --output "<RESTORED.mp4>"
```

The safe video profile, applied by the CLI:

- `--resolution` defaults to `1x`; enlarged video output is refused unless you pass
  `--force-unsafe-video-memory`. Run `2x` only after the same slice looks right at `1x`.
- `--low-ram` is enabled automatically. `--mlx-cache-limit-gb 8` limits the MLX cache, not total
  process memory.
- Video is restored in sequential temporal chunks. Defaults: `--temporal-chunk-size 49`,
  `--temporal-chunk-overlap 16`. A multi-chunk run needs chunks of at least `29` frames and an
  overlap of at least `8`; smaller profiles are rejected.
- `--vae-tiling` is rejected for video. Video always uses the one-step path (`--steps` is for images).
- Source FPS is preserved. Frames are center-cropped to multiples of 16.
- Audio: the matching source segment is kept by default, muxed with `ffmpeg`. If `ffmpeg` is not
  on `PATH`, or the copy cannot be proven aligned, the run fails instead of writing a silent file.
  Pass `--drop-audio` when a silent MP4 is intended. The sidecar records `audio_present`,
  `audio_copied`, `audio_copy_mode`, and `audio_copy_reason`.
- `--start-seconds` and `--max-frames` bound the decoded window.

Measured on an Apple M5 Max with 128 GB, 149 frames of a 320x240 source, chunk 29, overlap 8:

| Profile | Wall time | Peak MLX | Max RSS |
| --- | ---: | ---: | ---: |
| 3B `1x` | 74.5 s | 14.55 GB | 27.40 GB |
| 7B `1x` | 112.9 s | 24.54 GB | 66.18 GB |
| 3B `2x` | 542.3 s | 34.40 GB | 27.40 GB |
| 7B `2x` | 460.6 s | 44.27 GB | 66.18 GB |

Known limit: SeedVR2 video is not validated for 480x360 sources (with 0.40.1, runs forced past the
memory budget there produced intermittent corrupted frames; 0.40.2 fits the chunk in the budget,
and the geometry has not been re-validated). 320x240 and 384x288 are clean; use SwiftVR for
480x360.

## Video (SwiftVR)

```sh
mlxgen download --model swiftvr
mlxgen upscale --model swiftvr --video-path "<INPUT.mp4>" --resolution 1x --output "<RESTORED.mp4>"
```

- Source resolution only: any `--resolution` other than `1x` is refused with a pointer to SeedVR2.
  For a larger SwiftVR result, resize the input yourself first, then restore at `1x`.
- BF16 only, no quantized package. `--color-correction` accepts `off` only.
- The clip is trimmed to `4a + 1` frames; the CLI reports the trim. Audio is kept on the same
  terms as SeedVR2.
- No still images: restore stills with `--model seedvr2-3b`.
- Measured on an Apple M4 Max, 121 frames at 384x288, `1x`: SwiftVR 11.5 s and 11.1 GB peak MLX;
  SeedVR2 3B 471.0 s and 57.9 GB; SeedVR2 7B 458.8 s and 67.8 GB. At 1080p SwiftVR runs at about
  1.1 frames per second: an offline batch tool, not real time.

## Seeds, outputs, events

- `--seed` takes one or more seeds and `--auto-seeds N` generates them; each seed writes one file
  with `_seed_<seed>` appended, or `{seed}` filled in `--output`. Two sources with the same
  basename are rejected when replacing; pass `--no-replace` or rename them.
- `--metadata` writes the sidecar with source size, requested resolution, final size, seed, and
  model. `--json-events` works as on `mlxgen generate`; the video `complete` event means the MP4,
  metadata, and audio copy all succeeded.
- A missing model raises the same `DownloadRequiredError` as generation.

Python: `SeedVR2` exposes `generate_image(...)` (in memory), `restore_image_to_path(...)`, and
`restore_video_to_path(...)`; see [python-api.md](python-api.md).

Source docs: <https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/upscaling.md>,
<https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/api.md>.
