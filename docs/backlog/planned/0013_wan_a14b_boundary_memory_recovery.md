# Planned: Wan A14B boundary memory recovery

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
- A14B configs construct both `transformer` and `transformer_2` up front. Before this item, both
  remained resident for the entire denoise loop even though the high-noise transformer is never
  needed again after the boundary.
- The CLI progress bar previously used frame totals even though progress events are emitted per
  denoise step. Item 0014 changed the CLI display to denoise-step units while preserving frame
  count as context.
- The pasted shell command included `--frames 81 \ ` with a trailing space after the backslash. In
  zsh this does not continue the command; it passes a literal space argument and starts the next
  line as a separate command. That is a user-invocation risk, but it does not explain a live
  progress line unless the actual run differed.

## Problem

Full-size A14B I2V runs can hit peak memory exactly at the boundary where the low-noise transformer
starts, because the high-noise transformer remains resident. Progress reporting also makes this
failure mode harder to understand, and no recoverable output exists until the run finishes.

## What we want to do

Reduce unnecessary A14B memory pressure at the denoiser boundary, make CLI progress report denoise
steps honestly, and capture validation evidence for full-size A14B retry behavior.

## Why

Wan runs are multi-hour workloads. Users need clear progress semantics and the runtime should not
keep tens of gigabytes of no-longer-needed weights alive in a single-output run.

## Requirements

- Single-seed CLI A14B runs may release the high-noise denoiser before the first low-noise step.
- Multi-seed runs must not silently destroy the model instance before later seeds can run.
- Low-RAM runs should clear transient denoise arrays and MLX cache between steps.
- CLI progress should show denoise steps, not a fake frame counter.
- Full-size validation should capture exit code, output path, metadata, memory, and whether the
  run crosses step 7.
- Full-size validation must also inspect the saved MP4 for non-finite conversion warnings,
  all-black/static sampled frames, expected frame count, and nonzero temporal change.

## Suggested implementation

1. Add a runtime option that releases the inactive high-noise denoiser after the A14B boundary.
2. Enable that option automatically for single-seed CLI runs with `has_transformer_2`.
3. Keep the existing decode-time denoiser release for `--low-ram`, but add per-step cleanup only
   when `--low-ram` is requested.
4. Wan CLI progress now uses step totals and keeps frame count only as context.
5. Retry the user's command as a single shell line with a unique output path and memory capture.

## Scope

- Wan2.2 A14B T2V/I2V denoise lifecycle, CLI progress, and validation commands.
- Focused unit tests for routing, lifecycle flags, progress semantics, and helper behavior.

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
- Low-RAM mode has a denoise-loop cleanup path instead of only decode-time cleanup.
- A future agent has a precise full-size validation command and knows what evidence remains
  missing.

## Validation

- `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run pytest tests/wan/test_wan_progress.py tests/wan/test_wan_a14b_config.py tests/cli/test_mlx_gen_router.py -q`
- `git diff --check`
- Full-size retry:
  `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run mlxgen generate --model Wan-AI/Wan2.2-I2V-A14B-Diffusers --task image-to-video --image docs/assets/i2v_takeoff_source.png --prompt "Cinematic sequence of the spacecraft lifting off from the snowy landing field, engines glowing as the camera tracks the ascent." --width 1280 --height 704 --frames 81 --steps 20 --guidance 5 --fps 16 --seed 321 --metadata --output validation_outputs/wan_a14b_i2v_takeoff_retry.mp4`
- Postflight validation must reject black/static MP4 output even when `ffprobe` reports the expected
  stream metadata.

## Progress checklist

- [x] Confirm `denoise step 6/20` maps to progress frame 24 for 81 frames.
- [x] Confirm step 7 is the first low-noise transformer step for A14B I2V at 20 steps.
- [x] Add boundary denoiser release for single-seed CLI runs.
- [x] Change CLI progress to denoise-step progress.
- [x] Add focused tests.
- [ ] Retry full-size A14B I2V with memory/exit-code capture.

## Guidance for the implementing agent

Treat a full-size retry as expensive. Use a unique output path, avoid trailing spaces after shell
continuation backslashes, capture stdout/stderr and exit code, and inspect the output with
`ffprobe` before declaring the issue fixed.
