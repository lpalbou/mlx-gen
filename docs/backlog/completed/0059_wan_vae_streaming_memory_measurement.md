# Completed: Wan VAE streaming and memory measurement

## Metadata
- Created: 2026-06-27
- Status: Completed
- Completed: 2026-06-27
- Reopened: 2026-06-27

## ADR status
- Governing ADRs: None
- ADR impact: None unless this changes the public `GeneratedVideo` API contract.

## Context
Item 0056 added a Wan lazy-save path that avoids retaining a full PIL frame list while preserving
the existing `GeneratedVideo` API. The 2026-06-27 memory audit confirmed that Wan still decodes and
retains the full decoded video tensor before frame conversion.

## Current code reality
- `Wan2_2_VAE.decode()` appends decoded temporal slices and concatenates them before returning:
  `src/mflux/models/wan/model/wan_vae/wan_2_2_vae.py`.
- `Wan2_2_TI2V.generate_video()` calls `decode_normalized_latents()`, evaluates the full decoded
  tensor, and then passes it to `VideoUtil.to_video()`.
- `VideoUtil.to_video(..., materialize_frames=False)` avoids a full PIL frame list, but the batch
  factory still closes over the full decoded tensor.

## Problem
Longer Wan runs can peak during VAE decode or decoded-video retention even after lazy frame saving.
This makes `--low-ram` materially incomplete for single-seed video runs.

## What we want to do
Add a streamed Wan decode/save path that keeps latent residency and decoded frame residency bounded,
while preserving saved-video quality, metadata, and existing `GeneratedVideo` behavior where
possible.

## Why
Wan A14B and longer TI2V/I2V runs are memory-sensitive on Apple Silicon. Reducing decoded-video
residency directly decreases unified-memory pressure without changing denoise math.

## Requirements
- Decode temporal slices or small batches without concatenating the full decoded video.
- Preserve causal VAE decode order and feature-cache behavior.
- Preserve metadata, video-health validation, and failure diagnostics.
- Keep `GeneratedVideo.frames` compatibility unless an ADR accepts an API break.
- Record memory mode in output metadata.

## Suggested implementation
Add a VAE slice iterator and a `VideoUtil` batch-factory constructor that converts decoded slices
to frame batches during save. Use it for Wan low-RAM single-seed runs after denoisers are released.

## Scope
- Wan TI2V/I2V video generation.
- Low-RAM single-seed decode/save path first.
- Unit tests for batch-factory behavior and metadata shape.

## Non-goals
- Do not stream partial MP4s directly from denoise latents.
- Do not run long full-quality A14B generations without approval.
- Do not change frame pixels versus the existing VAE decode path except for unavoidable floating
  point evaluation ordering.

## Dependencies and related tasks
- [0060 runtime memory telemetry](0060_runtime_memory_telemetry_and_manifests.md)
- Completed [0056 Wan, VLM, and BF16 performance hardening](0056_wan_vlm_bf16_performance_hardening.md)

## Expected outcomes
- Wan low-RAM video save no longer retains a full decoded video tensor.
- Saved MP4 metadata identifies streamed decode mode.
- Tests prove lazy frame factories can save without materializing all frames.

## Validation
- Focused unit tests for VAE slice conversion helpers and `GeneratedVideo.save()`.
- Existing video metadata tests continue to pass.
- Manual memory comparison remains required for public claims.

## Progress checklist
- [x] Add streamed VAE decode iterator.
- [x] Add `VideoUtil` factory for decoded slice batches.
- [x] Route Wan low-RAM single-seed runs through streamed decode.
- [x] Add metadata and tests.

## Implementation notes
Preserve Wan generation quality first. If causal decode streaming cannot preserve output semantics,
stop and record the limitation rather than faking streaming.

## Completion report

- Date: 2026-06-27
- Original path: `docs/backlog/planned/memory/0059_wan_vae_streaming_memory_measurement.md`
- Final path: `docs/backlog/completed/0059_wan_vae_streaming_memory_measurement.md`
- Summary: Added a Wan low-RAM streamed VAE decode/save path so the denoiser-release route no
  longer needs to retain a full decoded video tensor or a full PIL frame list before saving.
- Implementation: `Wan2_2_VAE` now exposes decoded latent slice iterators; `Wan2_2_TI2V` uses the
  streamed path after denoiser release; `GeneratedVideo` and `VideoUtil` can save from frame-batch
  factories; video I/O and health checks prefer ffmpeg/ffprobe subprocesses for smaller and more
  stable local validation.
- Behavior changes: Low-RAM Wan metadata records streamed decode mode and pre-save timing scope.
  The eager path remains available when denoisers are retained for reuse.
- Validation: Focused Wan release tests passed for denoiser release before streamed decode,
  reuse rejection after release, and high-noise transformer release before low-noise routing.
- Residual risk: Quantitative public memory claims still require physical-process measurements on
  approved full-size Wan runs.

## Reopen report

- Date: 2026-06-27
- Reason: The item implemented a plausible streamed decode/save path but did not produce
  quantitative before/after memory statistics from real generation or representative decode/save
  profiles.
- Required evidence before closure: process-isolated baseline-versus-candidate runs with peak
  physical footprint/RSS, MLX peak/cache, wall time, saved-video frame count, dimensions, health,
  and frame-level quality comparison where the math is expected to be unchanged.

## Quantitative closure report

- Date: 2026-06-27
- Evidence artifact: `validation_outputs/memory/real_generation_20260627_wan_r2/generation_memory_benchmark.json`
- Real profile: `mflux-generate-wan`, `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit`,
  320x192, 9 frames, 1 step, seed 4217, eager versus `--low-ram` streamed decode.
- Process isolation: two real external CLI runs per variant, parent RSS sampling every 200 ms,
  output metadata memory snapshots, `/usr/bin/time -l`, and saved-video comparison.
- Memory result: median MLX peak fell from 14.582 GB to 10.340 GB, a 29.1% reduction. Median
  metadata Darwin physical footprint fell from 16.106 GB to 5.014 GB, a 68.9% reduction at the
  video-metadata/save boundary. Median metadata process RSS fell from 6.422 GB to 5.161 GB, a
  19.6% reduction at the same boundary.
- Peak caveat resolved into follow-up: median sampled process-tree RSS stayed effectively flat
  at 14.408 GB eager versus 14.431 GB streamed because the model-load/startup peak dominates this
  small profile. Startup peak reduction remains tracked by planned item 0063 and proposed item
  0065.
- Quality/performance result: saved MP4 comparison was exact for this profile
  (`frames=9`, `width=320`, `height=192`, `mae=0`, `rmse=0`, `max_abs=0`). Median wall time was
  58.84 s eager versus 57.87 s streamed.
- Closure decision: Completed for Wan streamed decode/save memory residency. Do not use this item
  to claim a whole-process startup peak reduction.
