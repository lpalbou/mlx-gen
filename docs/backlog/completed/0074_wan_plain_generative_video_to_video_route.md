# Completed: Wan Plain Generative Video-To-Video Route

## Metadata
- Created: 2026-07-03
- Status: Completed
- Completed: 2026-07-03

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md), [ADR 0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md), [ADR 0006](../../adr/0006_generative_video_editing_task_boundary.md)
- ADR impact: None. The shipped route stays inside the accepted plain `video-to-video` boundary.

## Context

After the boundary hardening and the bounded upstream VACE reference pass, the next honest video-edit
milestone was plain prompt-guided source-video editing: one source video, one prompt, one output
video, no masks or richer conditioning. The repo already had most of the planner and runtime
plumbing; what was missing was one truthful public Wan route, fail-closed boundaries, and one real
saved-output proof.

## What changed

- Enabled plain `video-to-video` on the exact `Wan2.2-T2V-A14B` config and its prepared-package
  derivatives.
- Kept non-V2V Wan routes fail-closed for source-video editing.
- Reused the existing unified planner, CLI router, Python runtime wrapper, MP4 save flow, metadata,
  and progress events instead of introducing a second public runtime abstraction.
- Hardened public V2V to require `solver=unipc`.
- Aligned source-video latent preparation with the upstream warm-start contract by keeping
  source-video conditioning in `float32`.
- Kept the current public route plain: one source video, one prompt, optional `video_strength`,
  no masks, no reference-image conditioning, no VACE taxonomy.

## Why

This completes the first public source-video editing milestone without pretending that richer VACE
conditioning is already proven or already belongs in the runtime contract.

## Scope completed

- Unified `mlxgen generate` planning and routing for public `video-to-video`.
- Exact Wan config exposure for the first public route.
- CLI help and Python documentation updates for the shipped capability.
- Focused route, CLI, runtime, geometry, and source-conditioning tests.
- One model-backed bounded ship-edit proof with preserved metrics and output.

## Validation

- Focused tests passed on 2026-07-03:
  - `uv run pytest tests/test_task_inference.py tests/test_python_runtime.py tests/cli/test_mlx_gen_router.py tests/wan/test_wan_a14b_config.py -k 'video_to_video or video-to-video or wan_a14b_video_to_video or solver or supports_video_to_video or resolved_spatial_size or source_conditioning_uses_float32'`
- Preserved bounded proof run on the shipped route:
  - model: `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit`
  - source: `docs/assets/examples/spaceship-snow/06_i2v_a14b_spaceship_takeoff_from_source.mp4`
  - output: `validation_outputs/v2v_native_a14b_q8_patched_2026_07_03/ship_a14b_q8_native.mp4`
  - contact sheet: `validation_outputs/v2v_native_a14b_q8_patched_2026_07_03/ship_a14b_q8_native_contact_sheet.png`
  - settings: `448x256`, `17` frames, `5` requested steps, `guidance 4`, `guidance_2 3`,
    `video_strength 0.7`, `solver unipc`, `seed 4242`
  - measured wall time: `85.62s`
  - generation time in metadata: `79.88s`
  - peak RSS from `/usr/bin/time -l`: `14.81 GB`
  - recorded MLX peak memory in metadata: `30.12 GB`

## Outcome

MLX-Gen now ships a bounded public Wan plain `video-to-video` route. The current supported surface
is intentionally narrow, but it is real, tested, and backed by a preserved model run. Richer
conditioning remains separate follow-up work in [0075 Wan VACE conditioning expansion after plain
video-to-video](../proposed/0075_wan_vace_conditioning_expansion_after_plain_video_to_video.md).
