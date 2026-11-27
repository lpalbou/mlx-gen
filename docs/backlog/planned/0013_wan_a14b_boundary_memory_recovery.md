# Planned: Wan A14B boundary memory recovery and full-size validation

## Metadata

- Created: 2026-06-02
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None. This is runtime lifecycle, observability, and validation work for an existing
  model route.

## Context

A Wan2.2 I2V-A14B run at 704x1280, 81 frames, 20 steps, guidance 5, and seed 321 appeared to stop
at `Generating video: 30%| 24/81 ... denoise step 6/20`. That progress point is not frame 24 being
rendered. It is denoise step 6 mapped onto an 81-frame progress bar. For the A14B I2V scheduler,
step 6 is timestep 900 and step 7 is the first timestep below the 0.9 boundary, so the run is about
to switch from `transformer` to `transformer_2`.

## Current code reality

- `src/mflux/models/wan/variants/wan2_2_ti2v.py` denoises the full latent video tensor, then
  decodes all frames and returns a `GeneratedVideo`. `VideoUtil` now converts decoded video to
  PIL frames one frame at a time, but the decoded tensor and final frame list are still whole-video
  objects.
- `src/mflux/models/wan/cli/wan_generate.py` saves the MP4 only after `generate_video` returns.
  Interrupted denoise runs therefore produce no final MP4.
- A14B configs construct both `transformer` and `transformer_2` up front. Version 0.18.9 added a
  single-seed CLI path that can release the inactive high-noise denoiser before low-noise
  denoising. The current low-RAM path clears MLX cache between transformer blocks and denoise steps,
  then releases denoisers before decode and clears between VAE temporal decode slices.
- CFG denoise predictions are explicitly materialized even when tensor-health checks are disabled
  or sampled sparsely, so low-RAM behavior no longer depends on diagnostic checks to force MLX
  graph evaluation.
- The CLI progress bar now uses denoise-step totals while preserving frame count as context.
- The pasted shell command included `--frames 81 \ ` with a trailing space after the backslash. In
  zsh this does not continue the command; it passes a literal space argument and starts the next
  line as a separate command. That is a user-invocation risk, but it does not explain a live
  progress line unless the actual run differed.

## Problem

Full-size A14B I2V runs can still hit peak memory near the high-noise to low-noise boundary or fail
late during decode/save. The runtime and progress fixes have shipped, but the original full-size
I2V retry has not yet produced memory, exit-code, metadata, and MP4 health evidence.

## What we want to do

Keep the shipped A14B memory/progress improvements visible and complete the missing full-size I2V
retry evidence before declaring the original boundary-memory concern closed.

## Why

Wan runs are multi-hour workloads. Users need clear progress semantics and the runtime should not
keep tens of gigabytes of no-longer-needed weights alive in a single-output run.

## Requirements

- Single-seed CLI A14B runs may release the high-noise denoiser before the first low-noise step.
- Multi-seed runs must not silently destroy the model instance before later seeds can run.
- Low-RAM runs should clear transient denoise arrays and MLX cache between transformer blocks and
  denoise steps.
- CLI progress should show denoise steps, not a fake frame counter.
- Full-size validation should capture exit code, output path, metadata, memory, and whether the
  run crosses step 7.
- Full-size validation must also inspect the saved MP4 for non-finite conversion warnings,
  all-black/static sampled frames, expected frame count, and nonzero temporal change.

## Suggested implementation

1. Keep the shipped runtime option that releases the inactive high-noise denoiser after the A14B
   boundary.
2. Keep the single-seed CLI default that enables boundary release only when the run will not reuse
   the same model instance for later seeds.
3. Keep low-RAM cleanup explicit and opt-in, including transformer-block cache boundaries.
4. Keep CLI progress step-based.
5. Retry the user's command as a single shell line with a unique output path and memory capture.

## Scope

- Wan2.2 A14B I2V full-size retry validation and the shipped denoise lifecycle/progress context
  needed to interpret it.

## Non-goals

- Do not add resumable video generation in this item.
- Do not stream partial MP4 files from denoise. Wan generates a full latent video, not independent
  frames.
- Do not change public Python API defaults to mutate reusable model instances.
- Do not claim the full-size OOM is fixed until the full retry has evidence.

## Dependencies and related tasks

- [Wan quantization and motion parity](0002_wan_quantization_motion_parity.md)
- [Wan q8 performance investigation](0005_wan_q8_performance_investigation.md)
- `src/mflux/models/wan/variants/wan2_2_ti2v.py`
- `src/mflux/models/wan/cli/wan_generate.py`
- `tests/wan/test_wan_a14b_config.py`
- `tests/wan/test_wan_progress.py`
- `tests/cli/test_mlx_gen_router.py`

## Expected outcomes

- The CLI no longer shows `24/81` as denoise progress for an 81-frame video.
- A single-seed A14B CLI run can free the high-noise denoiser before low-noise denoising.
- Low-RAM mode has transformer-block and denoise-loop cleanup paths instead of only decode-time
  cleanup.
- The item records a precise full-size validation command and what evidence remains
  missing.
- The item can close only after the full-size I2V retry captures memory, exit-code, metadata, and
  MP4 health evidence.

## Validation

- `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run pytest tests/wan/test_wan_progress.py tests/wan/test_wan_a14b_config.py tests/cli/test_mlx_gen_router.py -q`
- `git diff --check`
- Full-size retry:
  `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run mlxgen generate --model Wan-AI/Wan2.2-I2V-A14B-Diffusers --image docs/assets/i2v_takeoff_source.png --prompt "Cinematic sequence of the spacecraft lifting off from the snowy landing field, engines glowing as the camera tracks the ascent." --width 1280 --height 704 --frames 81 --steps 20 --guidance 5 --fps 16 --seed 321 --metadata --output validation_outputs/wan_a14b_i2v_takeoff_retry.mp4`
- Postflight validation must reject black/static MP4 output even when `ffprobe` reports the expected
  stream metadata.

## Progress checklist

- [x] Confirm `denoise step 6/20` maps to progress frame 24 for 81 frames.
- [x] Confirm step 7 is the first low-noise transformer step for A14B I2V at 20 steps.
- [x] Add boundary denoiser release for single-seed CLI runs.
- [x] Change CLI progress to denoise-step progress.
- [x] Extend low-RAM mode to request transformer-block cache boundaries.
- [x] Materialize CFG predictions independently from tensor-health checks.
- [x] Extend low-RAM decode cleanup to VAE temporal slices.
- [x] Add focused tests.
- [ ] Retry full-size A14B I2V with memory/exit-code capture.

## Implementation notes

Treat a full-size retry as expensive. Use a unique output path, avoid trailing spaces after shell
continuation backslashes, capture stdout/stderr and exit code, and inspect the output with
`ffprobe` before declaring the issue fixed.
