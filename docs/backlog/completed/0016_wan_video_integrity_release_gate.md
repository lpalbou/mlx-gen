# Completed: Wan video integrity release gate

## Metadata

- Created: 2026-06-03
- Status: Completed
- Completed: 2026-06-06

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None for the immediate gate. Escalate to ADR only if resumable generation or mandatory
  artifact retention becomes a package-wide public contract.

## Context

A T2V-A14B mixed q8/BF16 run using `models/wan2.2-t2v-a14b-diffusers-8bit` at 1280x720, 81 frames,
40 steps, guidance 4, guidance-2 3, fps 16, and seed 42 ran for about 13h15m. It then emitted:

```text
RuntimeWarning: invalid value encountered in cast
```

The saved `video.mp4` is a valid 81-frame H.264 MP4, but sampled frames are pure black with zero
temporal change. No metadata sidecar, latent checkpoint, or temp recovery artifact exists.

## Current code reality

- `src/mflux/utils/video_util.py` previously let `np.clip(...).astype("uint8")` convert NaN decoded
  frame values into black pixels.
- Version 0.18.9 ships fail-closed finite checks before uint8 conversion and phase-aware Wan
  tensor-health checks for prompt embeddings, conditioning, denoise predictions, scheduler latents,
  pre-decode latents, and VAE decode.
- `src/mflux/models/wan/cli/wan_generate.py` now keeps the progress lifecycle open through save and
  reserves CLI `complete` for a saved MP4 that passes video-health validation.
- `src/mflux/utils/video_health.py` adds reusable frame and MP4 read-back health checks that reject
  effectively black/white collapse and validate dimensions, frame count, and fps.
- Wan CLI failures now write a compact `<output>.failure.json` manifest with the error, tensor-health
  report when available, prompt, seed, shape, guidance, fps, and memory-related runtime flags.
- The 0.18.11 working tree also records saved-video health metadata, exposes `--failure-diagnostics`
  for compact runtime diagnostics, and adds Wan block/attention health probes for localizing
  full-size q8 numerical failures.
- Wan q8/BF16 model cards now keep full-size claims tied to validation evidence instead of treating
  the failed 1280x720, 81-frame command as proven.
- Wan VAE q8 metadata was misleading for prepared q8 folders; 0.18.9 marks Wan VAE as
  `skip_quantization` and ignores stale q metadata for skipped components.
- The 0.18.10 public docs and model-card wording keep Wan mixed q8/BF16 claims tied to measured
  validation profiles and included lower-cost examples. They do not claim full-size A14B q8
  production readiness.
- Release-artifact capture now includes saved-video health metadata, MP4 read-back checks, and
  compact contact-sheet evidence for the targeted release profile.
- 2026-06-06 regression triage found that the published
  `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` package still works through the normal q8
  runtime path on a 432x240, 41-frame, 10-step T2V-A14B profile.
- The failed full-size diagnostic attempts were not sufficient evidence that the published model
  needed republishing. They used a dirty 0.18.11 working tree that forced denoise-prediction
  materialization by default and temporarily changed q8 cross-attention runtime precision. Those
  experiments were reverted from the compatibility path.
- `--tensor-health-check-interval` is now diagnostic opt-in for Wan denoising internals; default
  generation preserves MLX lazy execution and relies on final decoded-frame and MP4 health checks
  to prevent invalid black videos from being saved as successful outputs.
- Future exact full-size production validation remains tracked by the broader Wan follow-up items;
  it is not required for the 0.18.11 release gate.

## Problem

MLX-Gen allowed an invalid numerical video result to become a successful black MP4 after a multi-hour
run, and the public card guidance overstated what the q8 package had proven.

## What we want to do

Keep video generation fail-closed when tensors become non-finite, preserve enough failure evidence
to avoid blind reruns, and require release artifacts plus a video-health gate before Wan q8 model
cards or docs claim a validated profile.

## Why

Wan A14B runs are expensive enough that a late silent failure costs many hours. Users need clear
failure messages, recoverable diagnostics where practical, and model cards that do not recommend
unvalidated generation settings.

## Requirements

- Reject NaN/Inf decoded frames before MP4 encoding.
- Add phase-aware Wan numeric checks that can report denoise step, timestep, denoiser branch, and
  tensor health when non-finite values appear.
- Add an opt-in or failure-triggered final-latent diagnostic artifact for long Wan video runs.
- Add a postflight video-health check for release validation: expected frames/dimensions/fps, black
  or white collapse, near-static warnings, and no RuntimeWarnings.
- Reserve CLI `complete` semantics for after save and validation.
- Keep Wan A14B q8 model cards tied to validated profiles.

## Suggested implementation

1. Keep the decoded-frame finite guard and unit test.
2. Add lightweight Wan tensor-health helpers for prompt/conditioning tensors, denoise predictions,
   scheduler latents, and pre-decode latents. Avoid full decoded-video reductions; validate decoded
   output frame by frame before uint8 conversion.
3. Add CLI/debug flags for latent diagnostics without enabling heavy artifacts by default.
4. Add a reusable video-health inspector for validation outputs and model-card publication checks.
5. Run targeted handle validation before release and keep longer production profiles as separate
   follow-up work.

## Scope

- Wan video numerical integrity, save/progress semantics, validation harnesses, and q8 card gating.
- T2V-A14B first, then I2V-A14B if the same failure class appears.

## Non-goals

- Do not require a full-size A14B generation for the 0.18.11 compatibility release gate.
- Do not claim a generation profile as validated unless that profile has saved output and metadata
  evidence.
- Do not delete or overwrite `video.mp4`; keep it as failure evidence until the user authorizes cleanup.

## Dependencies and related tasks

- [Wan quantization and motion parity](../planned/0002_wan_quantization_motion_parity.md)
- [Wan q8 performance investigation](../planned/0005_wan_q8_performance_investigation.md)
- [Wan A14B boundary memory recovery](../planned/0013_wan_a14b_boundary_memory_recovery.md)
- `src/mflux/utils/video_util.py`
- `src/mflux/models/wan/variants/wan2_2_ti2v.py`
- `src/mflux/models/wan/cli/wan_generate.py`
- `src/mflux/models/common/weights/saving/model_card_saver.py`

## Expected outcomes

- Non-finite video tensors never produce a successful black MP4.
- Future failed Wan runs report where numerical invalidity first appeared.
- Release/model-card validation catches black/static outputs instead of trusting exit code and ffprobe.
- Wan A14B q8 publication claims are limited to settings that have passed integrity and quality checks.

## Validation

- `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run pytest tests/metadata/test_generated_video.py tests/wan/test_wan_quantization.py tests/model_saving/test_model_card_saver.py -q`
- `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run pytest tests/wan/test_wan_progress.py tests/wan/test_wan_a14b_config.py -q`
- `uv run pytest tests/wan/test_wan_quantization.py tests/cli/test_mlx_gen_router.py::test_wan_cli_writes_failure_manifest tests/cli/test_mlx_gen_router.py::test_routes_wan_failure_diagnostics_to_backend tests/metadata/test_generated_video.py -q`
- `git diff --check`
- Manual postflight against `video.mp4`: ffprobe should see 81 frames, but sampled-frame stats should
  flag all-black/static output as invalid.
- Future longer-profile validation should preserve stdout/stderr, metadata, memory metrics, sampled
  frames, and health report.

## Progress checklist

- [x] Confirm the failed `video.mp4` is valid but all black.
- [x] Add decoded-frame finite guard before uint8 conversion.
- [x] Downgrade the local q8 A14B card usage example to validation-sized settings.
- [x] Fix Wan VAE quantization metadata handling for skipped components.
- [x] Add Wan denoise/pre-decode tensor-health checks with a configurable per-step interval.
- [x] Add reusable video-health inspector and default generated-video save gate.
- [x] Add default CLI failure manifests next to the intended Wan video output path.
- [x] Keep 0.18.10 release docs/cards limited to measured validation profiles instead of full-size
  q8 readiness claims.
- [x] Add saved-video health metadata for successful generated videos.
- [x] Add opt-in Wan CLI `--failure-diagnostics` runtime diagnostics to failure manifests.
- [x] Add Wan block/attention diagnostic probes for opt-in q8 failure localization.
- [x] Keep Wan denoise tensor-health diagnostics opt-in so default generation remains compatible
  with the published q8 runtime path.
- [x] Validate the published T2V-A14B q8 package at 432x240, 41 frames, 10 steps, seed 4242, with a
  healthy saved MP4 and metadata health report.
- [x] Validate the published T2V-A14B q8 handle at 480x240, 41 frames, 15 steps, seed 4242, with
  saved MP4, metadata health report, and contact sheet.
- [x] Validate the published I2V-A14B q8 handle at a 480x240 target, 41 frames, 15 steps, seed 4243,
  with source-aspect output resolution, saved MP4, metadata health report, and contact sheet.
- [x] Copy targeted release artifacts into `docs/assets/quantization/wan-a14b-q8-release/`.

## Guidance for the implementing agent

Treat future Wan q8 claims as profile-specific. Prefer targeted unit tests and small model-backed
probes before longer production profiles.
