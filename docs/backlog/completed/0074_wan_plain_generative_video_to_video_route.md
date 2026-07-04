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

## Post-completion hardening (2026-07-04)

Three independent adversarial audits (upstream-parity, contract truthfulness, production readiness)
confirmed the scheduler/latent math matches the Diffusers reference and surfaced these defects,
which are now fixed:

- The `mlxgen generate` router consumed `--video-strength` without forwarding it to the Wan
  backend, so public V2V runs silently used the `0.8` default. The router now re-emits the flag
  and rejects out-of-range values.
- V2V metadata recorded the strength-truncated step count as `steps`, breaking
  `--config-from-metadata` replay. `steps` now records the requested count and
  `effective_steps`, `high_noise_stage_skipped`, and `source_video_*` fields are recorded
  alongside `video_strength`.
- Unreadable or too-short source videos now fail before the A14B weight load.
- Conditioning cache keys now include source file mtime and size to prevent stale latents.
- The runtime warns when low `video_strength` skips the A14B high-noise stage (making
  `--guidance` inactive) and when source frames are stretched to a mismatched canvas.
- Six stale test call sites shipped broken in the original commit were repaired.

## Validation

- Focused suites passed on 2026-07-04 after the hardening pass:
  - `uv run pytest tests/wan/test_wan_a14b_config.py tests/wan/test_wan_scheduler_and_timesteps.py tests/test_task_inference.py tests/test_python_runtime.py` (128 passed)
  - `uv run pytest tests/cli/test_mlx_gen_router.py` (181 passed)
- Reproducible public-CLI proof run (exact command in `docs/wan-video.md`), artifacts included in
  the repo under `docs/assets/examples/spaceship-v2v/`:
  - model: `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit`
  - source: `docs/assets/examples/spaceship-snow/06_i2v_a14b_spaceship_takeoff_from_source.mp4`
  - output: `docs/assets/examples/spaceship-v2v/starship_v2v_a14b.mp4`
  - metadata: `docs/assets/examples/spaceship-v2v/starship_v2v_a14b.metadata.json`
  - contact sheets: `starship_v2v_source_contact_sheet.png`, `starship_v2v_output_contact_sheet.png`
  - settings: `448x256`, `17` frames, `5` requested steps (`3` effective), `guidance 4`,
    `guidance_2 3`, `video_strength 0.7`, `solver unipc`, `seed 4242`, `--metadata`
  - measured wall time: `234.89s` (cold load), generation time in metadata: `221.3s`
  - peak RSS from `/usr/bin/time -l`: `14.78 GB`
  - recorded MLX peak memory in metadata: `29.89 GB`
  - byte-identical output to the preserved direct-class proof run
    (`validation_outputs/v2v_native_a14b_q8_patched_2026_07_03/ship_a14b_q8_native.mp4`),
    proving the public CLI route now reproduces the internal API path exactly.

## Outcome

MLX-Gen now ships a bounded public Wan plain `video-to-video` route. The current supported surface
is intentionally narrow, but it is real, tested, and backed by an in-repo reproducible proof run.
Richer conditioning remains separate follow-up work in [0075 Wan VACE conditioning expansion after
plain video-to-video](../proposed/0075_wan_vace_conditioning_expansion_after_plain_video_to_video.md).
