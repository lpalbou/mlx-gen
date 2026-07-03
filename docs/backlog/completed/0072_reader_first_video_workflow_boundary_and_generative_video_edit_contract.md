# Completed: Reader-First Video Workflow Boundary And Generative Video-Edit Contract

## Metadata
- Created: 2026-07-03
- Status: Completed
- Completed: 2026-07-03

## ADR status
- Governing ADRs: [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md), [ADR 0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md), [ADR 0006](../../adr/0006_generative_video_editing_task_boundary.md)
- ADR impact: None after ADR 0006 lands.

## Context

MLX-Gen already has a meaningful user-facing split between `mlxgen generate` and `mlxgen upscale`,
but the front door still has drift:

- top-level CLI help undersold video restoration and over-broadened video editing language;
- `mlxgen generate --help` described `--image` mainly as image-to-image input even though Wan uses
  one image for first-frame image-to-video;
- `mlxgen validation --model ...` defaulted to the historical I2I profile even for Wan video
  models, which hid exact video proof rows unless the caller already knew the right profile id;
- future Wan VACE work was still stored as one broad proposal instead of a frozen workflow
  boundary plus narrower follow-up items.

## Problem

Current code and docs contain the right primitives, but a new or returning user can still misread
the boundary between:

- generate a new video;
- animate from one image;
- restore an existing video;
- future prompt-guided edit of an existing video.

That ambiguity makes future video-edit work brittle before any runtime code is written.

## What changed

- Accepted [ADR 0006](../../adr/0006_generative_video_editing_task_boundary.md) as the durable
  public workflow boundary for future prompt-guided source-video editing.
- Updated `mlxgen --help` so `generate` and `upscale` describe the current Wan-vs-SeedVR2 video
  split truthfully.
- Updated `mlxgen generate --help` so one input image is clearly documented as Wan first-frame
  image-to-video, and existing source videos are routed to `mlxgen upscale` today.
- Added model-aware default validation profile selection through
  `default_validation_profile_id_for_model(...)`.
- Added reader-first workflow tables near the top of `docs/getting-started.md` and `docs/api.md`.
- Tightened proposal 0039 and the active backlog path so the first public runtime milestone remains
  plain `video-to-video`, not implicit VACE conditioning.

## Dependencies and related tasks

- [0039 Wan VACE video editing and control](../proposed/0039_wan_vace_video_editing_and_control.md)
- [0073 Wan VACE reference validation harness and bounded source cases](0073_wan_vace_reference_validation_harness_and_bounded_source_cases.md)
- [0074 Wan plain generative video-to-video route](0074_wan_plain_generative_video_to_video_route.md)

## Validation

- Focused CLI tests:
  - `tests/cli/test_mlx_gen_router.py`
- Manual doc review:
  - `docs/getting-started.md`
  - `docs/api.md`
  - `docs/wan-video.md`
  - `docs/adr/0006_generative_video_editing_task_boundary.md`

## Completion report

- Date: 2026-07-03
- Summary: The current CLI and doc front door now teaches the correct video workflow split before
  any new runtime work starts: Wan `generate` for new video creation or first-frame animation,
  SeedVR2 `upscale` for existing-video restoration, and future prompt-guided source-video editing
  reserved for a later explicit `video-to-video` runtime.
- Implementation:
  - `src/mflux/cli/mlx_gen.py`
  - `src/mflux/release/validation_registry.py`
  - `tests/cli/test_mlx_gen_router.py`
  - `docs/getting-started.md`
  - `docs/api.md`
  - `docs/wan-video.md`
  - `docs/adr/0006_generative_video_editing_task_boundary.md`
  - `docs/backlog/proposed/0039_wan_vace_video_editing_and_control.md`
  - `docs/adr/README.md`
- Validation:
  - `uv run pytest tests/cli/test_mlx_gen_router.py -q -k 'top_level_help_reports_version_and_release_date or generate_help_calls_out_first_frame_image_to_video or upscale_help_calls_out_video_restore_and_seedvr2_only or validation_command_reports_model_specific_status or validation_command_defaults_to_first_matching_model_profile or validation_command_lists_profiles or validation_command_reports_reframe_outpaint_profile'`
- Residual risk: the current validation default picks the first matching evidence row for a model,
  not a hand-curated “best canonical proof” row. That is acceptable for discovery, but it is still
  a pragmatic default rather than a curated proof-ranking system.
