# Completed: SeedVR2 video restoration and upscaling

## Metadata

- Created: 2026-06-07
- Status: Completed
- Completed: 2026-06-21

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md), [ADR 0004](../../adr/0004_seedvr2_video_host_safety_and_proof_boundaries.md), [ADR 0005](../../adr/0005_seedvr2_video_quality_proof_requires_five_second_reader_first_clips.md)
- ADR impact: This item now has explicit SeedVR2 host-safety and reader-first proof governance.

## Outcome

Completed as a SeedVR2 video restore v1 under `mlxgen upscale`.

Update 2026-06-21: the accepted public quality proof is the five-second Eiffel archival slice from
`70s` to `75s` at source-size `320x240`, preserved under
`docs/assets/validation/seedvr2-video-2026-06-21/`. The accepted public profiles are
`1x 29/8 wavelet` for route-correctness proof and `2x 29/8 wavelet` for the stronger bounded
`3B` versus `7B` quality comparison.

MLX-Gen now supports:

- `--video-path` on the existing SeedVR2 upscale route;
- bounded clip restore with `--start-seconds` and `--max-frames`;
- full-video restore through sequential temporal chunking for longer clips;
- official temporal input handling instead of silent framewise fallback;
- source-FPS preservation by default;
- `GeneratedVideo` metadata for source clip geometry, timing, padding, and audio state;
- public proof for:
  - `ByteDance-Seed/SeedVR2-3B`
  - `ByteDance-Seed/SeedVR2-7B`
  - `seedvr2-7b-sharp`
  - five-second Eiffel source-video proof on the official `3B` and `7B` routes
  - bounded q8 smoke on `AbstractFramework/seedvr2-3b-8bit`

Current explicit limitation:

- the host-safe public CLI profile is source-size-oriented and keeps enlarged video output behind
  explicit unsafe override.

## Related backlog items

- [0014 Shared progress callbacks for image and video pipelines](../completed/0014_shared_progress_callbacks.md)
- [0016 Wan video integrity release gate](../completed/0016_wan_video_integrity_release_gate.md)
- [0030 SeedVR2 upscale smoke, metadata, and quality defaults](../completed/0030_seedvr2_upscale_smoke_and_metadata.md)
- [0031 SeedVR2 official ByteDance checkpoint support](../completed/0031_seedvr2_official_bytedance_checkpoint_support.md)

## Context

SeedVR2 is officially a video restoration model. The ByteDance-Seed Hugging Face collection lists
`ByteDance-Seed/SeedVR2-3B` and `ByteDance-Seed/SeedVR2-7B` as video-to-video models, and the
SeedVR2 paper describes one-step video restoration. MLX-Gen 0.18.13 added direct official 3B/7B
checkpoint loading and reusable q8/q4 package preparation, but only for single-image restoration
and upscaling.

Update 2026-06-15: the published `ByteDance-Seed/SeedVR2-3B` route type is explicitly
`Video-to-Video`, so MLX-Gen's current image-only upscale path should now be treated as a useful
subset rather than an approximate match for the official model purpose.

Sources checked:

- ByteDance-Seed SeedVR collection: https://huggingface.co/collections/ByteDance-Seed/seedvr
- `ByteDance-Seed/SeedVR2-3B`: https://huggingface.co/ByteDance-Seed/SeedVR2-3B
- `ByteDance-Seed/SeedVR2-7B`: https://huggingface.co/ByteDance-Seed/SeedVR2-7B
- Official SeedVR repository: https://github.com/ByteDance-Seed/SeedVR
- SeedVR2 paper: https://arxiv.org/abs/2506.05301

## Closing code reality

- `mlxgen upscale` and `mflux-upscale-seedvr2` route to
  `src/mflux/models/seedvr2/cli/seedvr2_upscale.py`.
- `SeedVR2.generate_image(...)` still handles the original image-only path.
- `SeedVR2.generate_video(...)` still exists as a bounded internal path and returns
  `GeneratedVideo`, but the public CLI safe profile routes video restore through
  `SeedVR2.restore_video_to_path(...)`.
- `SeedVR2.restore_video_to_path(...)` now handles longer clips through sequential temporal
  chunks, overlap-aware context windows, and streamed MP4 output.
- `src/mflux/utils/video_util.py` now provides source inspection, bounded clip decode, frame-window
  decode, sequential overlapping window iteration, and writer cleanup helpers, all with source FPS,
  source dimensions, duration, frame count, and audio-presence metadata.
- Video save goes through the shared `GeneratedVideo`/MP4 path and existing video-health
  validation, even for chunked full-video restore.
- The public restore contract is explicit: preserve source FPS, trim temporary temporal padding
  back to the requested clip length, and record copied-vs-skipped audio state through
  `audio_present`, `audio_copied`, `audio_copy_mode`, and `audio_copy_reason`.
- Existing q4/q7B SeedVR2 package work remains image-only evidence. This item closes on bounded
  official `3B` and `7B` source proof plus the earlier q8 3B package smoke.

## Problem

Video restoration/upscaling would be valuable for AbstractVision and direct CLI users: low-quality
or compressed videos could be corrected, denoised, sharpened, and upscaled without converting each
frame manually. Silently applying the image upscaler frame-by-frame would have been the wrong
default because it can introduce temporal flicker and does not prove the official video model
contract. The implementation had to follow the official temporal path closely enough to earn a real
video-support claim.

## What changed

- `mlxgen upscale` now accepts exactly one of `--image-path` or `--video-path`.
- SeedVR2 video restore gained bounded clip controls:
  - `--start-seconds`
  - `--max-frames`
- Longer video restore gained explicit chunk controls:
  - `--temporal-chunk-size`
  - `--temporal-chunk-overlap`
  - `--color-correction`
- The public video path now:
  - decodes a bounded source clip with original FPS;
  - preprocesses video frames as a clip, not as independent images;
  - pads the temporal length to the official `T == 1` or `(T - 1) % 4 == 0` rule for `sp_size=1`;
  - creates video-shaped condition tensors and noise latents;
  - trims decoded output back to the requested frame count;
  - writes `GeneratedVideo` metadata including source clip and audio state;
  - streams longer clips through sequential frame windows instead of keeping the full decoded
    source in memory at once;
  - cleans up partial output files on failure.
- SeedVR2 CLI now treats source audio as part of the default restore contract, with explicit
  `drop_audio=True` / `--drop-audio` opt-out handled later by completed item 0046.
- Video restore rejects `--vae-tiling`; the public low-memory path for long clips is
  `--low-ram --mlx-cache-limit-gb 8` plus temporal chunking, with enlarged video output kept
  behind explicit unsafe override.
- Public docs now cover image and full-video restore together and include model-backed video
  commands plus proof artifacts.

## Acceptance criteria

1. `mlxgen upscale` accepts exactly one of `--image-path` or `--video-path`.
2. Default video restore uses the official temporal path; any framewise mode is explicit.
3. Source frame count and FPS are preserved by default unless the user explicitly asks otherwise.
4. Video save goes through the existing `GeneratedVideo`/MP4 path and passes video-health
   validation.
5. Failures preserve enough context to debug model, seed, dimensions, FPS, frame count, and error
   phase.
6. Metadata records source video path, source/output dimensions, requested/final resolution, FPS,
   frame counts, duration, model identity, quantization, softness, temporal padding/chunking, and
   audio-copied state.
7. Audio behavior is explicit and never silently dropped.
8. One tiny official-source MP4 smoke is required before docs claim support.
9. If q8/q4 video support is claimed, preserve model-backed tiny video proof artifacts for those
   package rows.
10. Do not expand `mlxgen generate`, `mlxgen capabilities`, or the public generation-task resolver
    in this item.
11. Do not reuse existing image-only SeedVR2 evidence as video-readiness proof.

All criteria are met for the full-video v1 scope. Audio copy-through is now closed by completed
item 0046 instead of remaining as a follow-up blocker.

## Validation

- Focused tests:

```sh
uv run pytest \
  tests/arg_parser/test_seedvr2_upscale_argparser.py \
  tests/metadata/test_generated_video.py \
  tests/image_generation/test_seedvr2_upscale_metadata.py \
  -q
```

- CLI router checks:

```sh
uv run pytest \
  tests/cli/test_mlx_gen_router.py::test_upscale_subcommand_routes_to_seedvr2_command \
  tests/cli/test_mlx_gen_router.py::test_upscale_subcommand_does_not_enable_downloads \
  -q
```

- Fast suite:

```sh
make test-fast
```

- Accepted 3B five-second `1x` public proof on the exact current route:

```sh
uv run mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-3B \
  --video-path 'eiffel_tower_1900_edison.mp4' \
  --start-seconds 70 \
  --max-frames 149 \
  --resolution 1x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 29 \
  --temporal-chunk-overlap 8 \
  --low-ram \
  --mlx-cache-limit-gb 8 \
  --metadata \
  --output validation_outputs/seedvr2_video_2026_06_21_pass2/eiffel_70s_149f_3b_chunk29_overlap8_wavelet_1x_after_causal_slicing.mp4 \
  --replace
```

- Accepted 7B five-second `1x` public proof on the exact current route:

```sh
uv run mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-7B \
  --video-path 'eiffel_tower_1900_edison.mp4' \
  --start-seconds 70 \
  --max-frames 149 \
  --resolution 1x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 29 \
  --temporal-chunk-overlap 8 \
  --low-ram \
  --mlx-cache-limit-gb 8 \
  --metadata \
  --output validation_outputs/seedvr2_video_2026_06_21_pass2/eiffel_70s_149f_7b_chunk29_overlap8_wavelet_1x_after_causal_slicing.mp4 \
  --replace
```

- Accepted 3B five-second `2x` public proof on the exact current route:

```sh
uv run mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-3B \
  --video-path 'eiffel_tower_1900_edison.mp4' \
  --start-seconds 70 \
  --max-frames 149 \
  --resolution 2x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 29 \
  --temporal-chunk-overlap 8 \
  --low-ram \
  --mlx-cache-limit-gb 8 \
  --force-unsafe-video-memory \
  --metadata \
  --output validation_outputs/seedvr2_video_2026_06_21_pass1/eiffel_70s_149f_3b_chunk29_overlap8_wavelet_2x_after_causal_slicing.mp4 \
  --replace
```

- Accepted 7B five-second `2x` public proof on the exact current route:

```sh
uv run mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-7B \
  --video-path 'eiffel_tower_1900_edison.mp4' \
  --start-seconds 70 \
  --max-frames 149 \
  --resolution 2x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 29 \
  --temporal-chunk-overlap 8 \
  --low-ram \
  --mlx-cache-limit-gb 8 \
  --force-unsafe-video-memory \
  --metadata \
  --output validation_outputs/seedvr2_video_2026_06_21_pass1/eiffel_70s_149f_7b_chunk29_overlap8_wavelet_2x_after_causal_slicing.mp4 \
  --replace
```

Published proof assets:

- `docs/assets/validation/seedvr2-video-2026-06-21/eiffel_70s_149f_3b_chunk29_overlap8_wavelet_1x_after_causal_slicing.mp4`
- `docs/assets/validation/seedvr2-video-2026-06-21/eiffel_70s_149f_7b_chunk29_overlap8_wavelet_1x_after_causal_slicing.mp4`
- `docs/assets/validation/seedvr2-video-2026-06-21/eiffel_70s_149f_3b_vs_7b_wavelet_1x_after_causal_slicing_comparison.mp4`
- `docs/assets/validation/seedvr2-video-2026-06-21/eiffel_70s_149f_3b_vs_7b_wavelet_1x_after_causal_slicing_contact_sheet.jpg`
- `docs/assets/validation/seedvr2-video-2026-06-21/motion_strip_1x_after_causal_slicing_frames_64_72.jpg`
- `docs/assets/validation/seedvr2-video-2026-06-21/motion_strip_1x_after_causal_slicing_frames_141_148.jpg`
- `docs/assets/validation/seedvr2-video-2026-06-21/eiffel_70s_149f_3b_chunk29_overlap8_wavelet_2x_after_causal_slicing.mp4`
- `docs/assets/validation/seedvr2-video-2026-06-21/eiffel_70s_149f_7b_chunk29_overlap8_wavelet_2x_after_causal_slicing.mp4`
- `docs/assets/validation/seedvr2-video-2026-06-21/eiffel_70s_149f_3b_vs_7b_wavelet_2x_after_causal_slicing_comparison.mp4`
- `docs/assets/validation/seedvr2-video-2026-06-21/eiffel_70s_149f_3b_vs_7b_wavelet_2x_after_causal_slicing_contact_sheet.jpg`
- `docs/assets/validation/seedvr2-video-2026-06-21/motion_strip_2x_after_causal_slicing_frames_64_72.jpg`
- `docs/assets/validation/seedvr2-video-2026-06-21/motion_strip_2x_after_causal_slicing_frames_141_149.jpg`
- `docs/assets/validation/seedvr2-video-2026-06-21/detail_crop_crowd_2x_after_causal_slicing_64_72.jpg`
- `docs/assets/validation/seedvr2-video-2026-06-21/eiffel_70s_149f_validation_report.md`

Observed timings on `M5 Max` for the accepted five-second proof:

- official `ByteDance-Seed/SeedVR2-3B` at `1x 29/8`: `71.33s` generation, `74.54s` wall time,
  peak MLX memory `14.55 GB`, max RSS `27.40 GB`
- official `ByteDance-Seed/SeedVR2-7B` at `1x 29/8`: `107.44s` generation, `112.93s` wall time,
  peak MLX memory `24.54 GB`, max RSS `66.18 GB`
- official `ByteDance-Seed/SeedVR2-3B` at `2x 29/8`: `539.03s` generation, `542.30s` wall time,
  peak MLX memory `34.40 GB`, max RSS `27.40 GB`
- official `ByteDance-Seed/SeedVR2-7B` at `2x 29/8`: `454.46s` generation, `460.61s` wall time,
  peak MLX memory `44.27 GB`, max RSS `66.18 GB`

All accepted visual-quality proof clips preserved source FPS. The shipped audio contract is now
validated separately by completed item 0046 on the published Air France `25s–35s` proof bundle.

Supporting sampled metrics on the accepted public proofs, after downscaling each candidate back to
the original `320x240` source resolution before scoring:

- `3B 1x 29/8`: `sharpness_gain 1.5240`, `contrast_gain 1.1046`, `temporal_ratio 1.5097`,
  `drift_mae 0.056926`, `heuristic_score 58.65`
- `7B 1x 29/8`: `sharpness_gain 1.2258`, `contrast_gain 1.0682`, `temporal_ratio 1.2874`,
  `drift_mae 0.041031`, `heuristic_score 60.72`
- `3B 2x 29/8`: `sharpness_gain 1.4096`, `contrast_gain 1.0504`, `temporal_ratio 1.3493`,
  `drift_mae 0.035856`, `heuristic_score 61.63`
- `7B 2x 29/8`: `sharpness_gain 1.4176`, `contrast_gain 1.0523`, `temporal_ratio 1.3563`,
  `drift_mae 0.033749`, `heuristic_score 62.57`

These metrics are supporting route-health evidence only. The current public conclusion is visual:
`1x 29/8` is the accepted native-resolution proof surface, while `2x 29/8` is the stronger bounded
quality comparison for this clip and shows `7B` slightly ahead of `3B`.

Local limitation check:

- a bounded q8 run on `SF.mp4` completed successfully but over-smoothed the
  already-clean modern source. The public docs now describe this as a fit limitation instead of a
  proof row.

## Why it mattered

This makes MLX-Gen useful for bounded local video cleanup and enhancement workflows, not only image
upscaling and generation. It also finally matches the official SeedVR2 model purpose with a real
video-backed support claim instead of an image-only approximation.

## Docs impact

Completed:

- `README.md`
- `docs/README.md`
- `docs/getting-started.md`
- `docs/api.md`
- `docs/upscaling.md`
- `docs/faq.md`
- `docs/architecture.md`
- `CHANGELOG.md`
- `llms.txt`
- `llms-full.txt`

## Non-goals that remain

- Do not treat this as video generation.
- Do not silently process video as independent images.
- Do not claim q4 or 7B video support from this item.
- Do not add automatic downloads during restoration.
- Do not expand `mlxgen generate`, `mlxgen capabilities`, or the public generation-task resolver
  in this item.

## Follow-ups

- [0046 SeedVR2 video audio copy-through](0046_seedvr2_video_audio_copythrough.md)
- Broader q4 / 7B / longer-clip public proof should stay separate until those rows are actually
  generated and reviewed.
