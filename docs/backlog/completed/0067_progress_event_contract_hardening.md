# Completed: Progress event contract hardening

## Metadata
- Created: 2026-06-29
- Status: Completed
- Completed: 2026-06-29

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The 2026-06-29 serial adversarial validation pass found multiple shipped progress-contract failures.
Masked image routes emitted the wrong public task, and a real Qwen control-inpaint run emitted a
`complete` progress event before the generation later failed during image finalization.

## Historical pre-fix reality
- `src/mflux/callbacks/generation_context.py` emits `complete` inside `after_loop()` before decode,
  save, and metadata work finish.
- `GenerationContext._resolved_task()` only infers `image-to-image` when `image_path` is present
  and `image_strength > 0`, so masked routes and video restore routes that do not use
  `image_strength` fall back to `text-to-image`.
- `src/mflux/models/qwen/variants/controlnet/qwen_image_controlnet.py` starts callbacks without an
  explicit task even when `mask_path` is active.
- `src/mflux/models/z_image/variants/z_image.py` starts callbacks without an explicit task even
  when native inpaint is active.
- `src/mflux/models/seedvr2/variants/upscale/seedvr2.py` starts callbacks without an explicit task
  for both image upscale and streamed video restore even though those routes are source-conditioned
  and the later video metadata/output paths use `video-to-video`.
- Historical false-complete evidence exists in
  `validation_outputs/serial_validation_qwen_control_progress.json` and
  `validation_outputs/serial_validation_zimage_inpaint_progress.json`.
- Refreshed post-fix proofs are being recorded under
  `validation_outputs/runtime_contracts_2026_06_29/`.

## Problem
Progress subscribers cannot rely on task labels or completion semantics across backends. That makes
the public callback contract brittle and can trigger false success behavior in downstream tools.

## What we want to do
Make progress events truthful and consistent across image and video routes while preserving the
existing lightweight callback payload shape.

## Why
MLX-Gen is supposed to be a clean abstraction across T2I, I2I, T2V, and I2V. Incorrect task labels
and premature completion break that abstraction at the public API boundary.

## Requirements
- Masked image routes must emit `image-to-image`.
- SeedVR2 image upscale must emit `image-to-image`.
- Streamed SeedVR2 video restore must emit `video-to-video`.
- `complete` must mean artifact-ready success, not merely denoise-loop completion.
- A non-interrupt failure after `start` must not leave subscribers waiting on a false success.
- Terminal progress states must be mutually exclusive and emitted at most once.
- Failures after denoise must not emit false success.
- Keep progress events lightweight and free of raw MLX tensors.

## Suggested implementation
Pass explicit tasks from backend entrypoints that bypass generic task inference, stop emitting
public success from `after_loop()`, and move terminal success/failure onto shared callback-context
methods that enforce terminal exclusivity after artifact materialization succeeds or fails.

## Scope
- `GenerationContext` event lifecycle semantics.
- Qwen masked control-inpaint callback task labeling.
- Z-Image native inpaint callback task labeling.
- SeedVR2 image-upscale callback task labeling.
- SeedVR2 streamed video restore callback task labeling.
- Focused progress regression tests and one-at-a-time real proofs.

## Non-goals
- Do not add heavy progress payloads, latent snapshots, or backend-specific event shapes.
- Do not broaden this item into callback cancellation or streaming-output redesign.
- Do not change model math for the sake of progress semantics.

## Dependencies and related tasks
- Completed [0014 shared progress callbacks](0014_shared_progress_callbacks.md)
- Completed [0018 taskless generation routing](0018_taskless_generation_routing.md)
- Completed [0020 generation capability contract](0020_generation_capability_contract.md)
- Completed [0068 Qwen control route hardening](0068_qwen_control_route_hardening.md)
- Completed [0069 Z-Image CFG and inpaint repair](0069_zimage_cfg_and_inpaint_repair.md)

## Expected outcomes
- Backend-specific masked and video routes publish the correct public task.
- Subscribers no longer observe `complete` before a later exception on the same run.
- Real callback proofs show correct task routing to filtered subscribers.

## Validation
- Focused callback/unit tests that assert explicit task propagation, terminal exclusivity, and final
  completion/failure semantics.
- One-at-a-time runtime proofs for Qwen control-inpaint, Z-Image native inpaint, SeedVR2 image
  upscale, and SeedVR2 streamed video restore.
- Manual inspection of emitted event sequences stored under
  `validation_outputs/runtime_contracts_2026_06_29/`.

## Progress checklist
- [x] Define the target progress lifecycle semantics.
- [x] Patch backend callback entrypoints to pass explicit tasks where inference is insufficient.
- [x] Patch shared terminal emission so `complete` only means artifact-ready success and `failed`
  remains tensor-free.
- [x] Add focused tests for masked image, video restore, and failure-after-loop cases.
- [x] Regenerate serial event proofs one run at a time.

## Guidance for the implementing agent
Preserve backwards compatibility where possible, but treat false-complete behavior as a contract
bug rather than a subscriber convenience.

## Completion report

- Date: 2026-06-29
- Original path: `docs/backlog/planned/runtime_contracts/0067_progress_event_contract_hardening.md`
- Final path: `docs/backlog/completed/0067_progress_event_contract_hardening.md`
- Summary: Progress semantics are now explicit and truthful across image and video routes. Masked
  image routes and SeedVR2 routes publish the correct public task, `after_loop()` is no longer a
  public terminal success, and shared terminal emission is enforced through `complete()` /
  `failed()`.
- Implementation: `GenerationContext` now separates loop completion from terminal public progress,
  hardens `after_loop()` failure handling, and enforces one terminal phase. Backend entrypoints in
  Qwen, Z-Image, SeedVR2, Flux/Flux2 concept-attention, FIBO, ERNIE, and related routes now own
  terminal success/failure after artifact materialization.
- Validation: `tests/callbacks/test_progress_callbacks.py`, `tests/image_generation/test_masked_generation_routes.py`,
  `tests/seedvr2/test_seedvr2_progress.py`, and the post-write failure checks in
  `tests/seedvr2/test_seedvr2_video_chunking.py` passed. Real proofs were preserved for Qwen
  control-inpaint, Z-Image native inpaint, SeedVR2 image upscale, and SeedVR2 streamed restore.
- Evidence report: [runtime_contracts_report.md](../../assets/validation/runtime-contracts-2026-06-29/runtime_contracts_report.md)
- Residual risk: Progress callback exceptions still propagate by design. The public docs now state
  that handlers must stay small and defensive.
