# Completed: Shared progress callbacks for image and video pipelines

## Metadata

- Created: 2026-06-03
- Status: Completed
- Completed: 2026-06-03
- Original path: N/A; created retrospectively as a completed item.

## ADR status

- Governing ADRs: None
- ADR impact: None. This item defines a narrow public Python event contract and subscription
  surface that is documented in the API guides. It does not establish a broader architecture
  policy requiring a new ADR.

## Context

Wan video generation already had progress reporting, but the CLI presented progress against frame
counts even though the denoising loop advances by step. Image generation used the older callback
registry for loop hooks but did not expose a shared public progress subscription API for
text-to-image and image-to-image callers.

Applications such as AbstractVision need one small event surface that can feed UI progress,
background job status, and workflow monitoring without depending on model-specific progress
classes or internal latent tensors.

## Current code reality

- `src/mflux/callbacks/progress.py` exposes a frozen keyword-only `ProgressEvent` dataclass and
  `ProgressCallback` type alias.
- `src/mflux/callbacks/__init__.py` exports `ProgressEvent` and `ProgressCallback` as the public
  callback symbols.
- `CallbackRegistry.subscribe_progress(callback, task=None)` registers progress subscribers and
  returns an unsubscribe function.
- `GenerationContext` emits image progress phases through the shared event type and only forces an
  MLX evaluation before progress emission when a progress listener exists.
- `Wan2_2_TI2V.generate_video()` emits the same `ProgressEvent` type, supports registry
  subscribers, and still accepts a direct `progress_callback` for one-shot video callers.
- The Wan CLI progress bar advances by denoising step and keeps frame count only as video context.

## Problem

The earlier progress surface was inconsistent across task families. Video had a model-specific
event shape and a confusing frame-count display, while image generation had no clear public
progress event that applications could subscribe to. That made progress integration harder to
reuse and increased the chance that callers would interpret a long Wan run as stopping mid-frame
instead of reaching a denoiser boundary.

## Completed work

Added a shared lightweight progress abstraction and wired it into both image and video generation:

1. Created `ProgressEvent` with `phase`, `task`, `step`, `total_steps`, optional `timestep`, and
   optional video `frame` / `total_frames` context.
2. Made `ProgressEvent.progress` an alias for denoise-step progress and exposed
   `frame_progress` separately for video context.
3. Added `CallbackRegistry.subscribe_progress(...)` with optional task filtering and a returned
   unsubscribe function.
4. Emitted image progress events from `GenerationContext` for `start`, `denoise`, `complete`, and
   handled interruption.
5. Replaced the Wan-specific progress event with the shared event type while preserving the direct
   `generate_video(progress_callback=...)` path.
6. Changed the Wan CLI progress display to denoising-step units.
7. Updated API, Python integration, architecture, and LLM documentation to describe the shared
   event contract.

## Why

Long image and video generation jobs need application-visible progress that is simple, stable, and
cheap to subscribe to. A single shared event type avoids model-specific branching in embedding
applications and gives future orchestration layers a clean status feed without exposing internal
arrays or requiring CLI parsing.

## Requirements

- Provide a public Python API that is easy to reuse and subscribe to.
- Use one progress event shape for text-to-image, image-to-image, text-to-video, and
  image-to-video.
- Report progress by denoising step, not by inferred output frame.
- Keep video frame counts available as context without making them the primary progress unit.
- Keep the image progress path lightweight when no subscribers exist.
- Do not preserve legacy event aliases that would keep the API confusing.

## Implementation summary

- `src/mflux/callbacks/progress.py`: introduced the shared event and callback type.
- `src/mflux/callbacks/callback_registry.py`: added progress subscriptions, task filtering,
  unsubscribe handling, and progress emission.
- `src/mflux/callbacks/generation_context.py`: emitted image lifecycle progress and avoided
  extra evaluation work when no progress subscriber is present.
- `src/mflux/models/wan/variants/wan2_2_ti2v.py`: switched Wan video progress to the shared event
  and preserved direct callback support.
- `src/mflux/models/wan/cli/wan_generate.py`: made CLI progress step-based.
- `tests/callbacks/test_progress_callbacks.py`, `tests/wan/test_wan_progress.py`, and
  `tests/cli/test_mlx_gen_router.py`: covered the public export, subscription lifecycle,
  image task filtering, image evaluation behavior, Wan event semantics, and CLI progress units.

## Scope

- Shared Python progress event and subscription API.
- Image progress events for existing image generation contexts.
- Wan video progress event cleanup and CLI denoise-step display.
- Focused unit tests and public documentation updates.

## Non-goals

- No cancellation API in this item.
- No new image CLI progress option in this item.
- No full model generation run for this cleanup pass.
- No broad refactor of the older before/in/after/interrupt callback hooks.

## Dependencies and related tasks

- [Model integration roadmap](../planned/0001_model_integration_roadmap.md)
- [Wan quantization and motion parity](../planned/0002_wan_quantization_motion_parity.md)
- [Wan A14B boundary memory recovery](../planned/0013_wan_a14b_boundary_memory_recovery.md)
- [Python Integration](../../python-integration.md)
- [API And CLI](../../api.md)

## Outcomes

- Image applications can subscribe through `model.callbacks.subscribe_progress(...)`.
- Wan applications can use the same subscription path or pass a direct
  `progress_callback` to `generate_video()`.
- `ProgressEvent.progress` is denoise-step progress for every task family.
- Video-specific frame context remains available through `frame`, `total_frames`, and
  `frame_progress`.
- The Wan CLI no longer displays frame count as the progress total.

## Validation evidence

- `uv run ruff check src/mflux/callbacks/progress.py src/mflux/callbacks/__init__.py
  src/mflux/callbacks/callback_registry.py src/mflux/callbacks/generation_context.py
  src/mflux/models/wan/variants/wan2_2_ti2v.py src/mflux/models/wan/variants/__init__.py
  src/mflux/models/wan/cli/wan_generate.py tests/callbacks/test_progress_callbacks.py
  tests/wan/test_wan_progress.py tests/cli/test_mlx_gen_router.py`
- `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run pytest tests/callbacks/test_progress_callbacks.py
  tests/wan/test_wan_progress.py
  tests/cli/test_mlx_gen_router.py::test_wan_cli_progress_advances_by_denoise_steps -q`
- Focused test result: `11 passed in 1.18s`.
- Stale symbol checks found no remaining `WanProgressEvent`, legacy progress registration API, or
  frame-oriented CLI progress wording in the implementation.

## Progress checklist

- [x] Confirm current image and video callback behavior.
- [x] Define one shared public progress event.
- [x] Add subscription/unsubscribe support to the callback registry.
- [x] Emit image progress events with task filtering.
- [x] Convert Wan video progress to the shared event.
- [x] Fix Wan CLI progress units to denoising steps.
- [x] Remove legacy progress aliases.
- [x] Add focused tests.
- [x] Update coredoc and LLM documentation.

## Follow-up guidance

Treat cancellation as separate work if it becomes a committed requirement. Keep future progress
extensions additive and event-based; do not expose latents, scheduler internals, or model tensors
through the public progress callback contract.
