# Completed: Wan Masked Video-To-Video via Latent Compositing

## Metadata
- Created: 2026-07-04
- Status: Completed
- Completed: 2026-07-04

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md), [ADR 0006](../../adr/0006_generative_video_editing_task_boundary.md)
- ADR impact: ADR 0006 amended in the same change (the "dedicated handler/runtime" enforcement
  clause is relaxed to match the shipped config-gated implementation; masked V2V is a typed
  conditioning role under the existing `video-to-video` task, not a new task name).

## Context

Plain video-to-video re-synthesizes every latent token, so background high-frequency content
(banner text, logos, posters) drifts even when the prompt demands preservation. This was measured
and adversarially confirmed as structural, not a bug. The fix is spatial selectivity: lock
everything outside a user mask to the source video.

## Design (four adversarial reviews, decisions recorded)

- Reviewer A (math): input/output compositing equivalent up to loop rotation; binarize the mask
  at 0.5 (float feathers re-inject unhealable ghost bands); explicit sigma indexing.
- Reviewer B (contract): `--video-mask-path` (new flag, not `--mask-path` reuse), white = model
  may change, strength unchanged and applies inside the mask, no mechanism label in metadata,
  all-black masks error pre-load, 18-point integration checklist.
- Reviewer C (implementation): the UniPC corrector rebuilds each step from `last_sample` and the
  x0 history and uses the incoming sample only for dtype, so scheduler state must be composited
  too (`last_sample` at the previous sigma, `model_outputs[-1]` with clean source latents);
  `z_src` already cached; no sigma off-by-one; use `vae.spatial_scale`, not a literal 8.
- Reviewer D (proof): pass bars defined before the run, ceiling measured through the real
  pipeline including H.264; the two existing 2-tuple `_prepare_video_to_video_latents`
  monkeypatch tests must be updated in the same commit.

## What changed

- `Wan2_2_TI2V.generate_video` accepts `video_mask_path`; per-step post-`scheduler.step`
  composite of latents plus UniPC state; final clean composite makes preserved regions exactly
  the source VAE latents (`src/mflux/models/wan/variants/wan2_2_ti2v.py`).
- Mask prep: PIL BOX downsample to the latent grid, binarized at 0.5, white = change; alpha
  channels warned; all-zero/all-one masks warned at runtime.
- Wan CLI: `--video-mask-path` with existence check, pre-model-load decodability + all-black
  probe, metadata replay, failure-manifest field.
- Router: parse + re-emit (the `--video-strength` regression class) + `has_video_mask`.
- Planner: `supports_video_mask` capability field on `wan.video-video`, typed constraints,
  capabilities schema version 3 -> 4.
- Python runtime: `has_video_mask` plan-resolution passthrough.
- Docs: `docs/wan-video.md` masked section with in-repo proof artifacts, `docs/api.md` option
  row, FAQ entry, getting-started pointer, README, llms files, CHANGELOG.

## Validation

- Full suite: 1244 passed, lint clean.
- Math pins (fake transformer + real UniPC scheduler): all-black mask -> output latents exactly
  equal source latents; all-white mask -> byte-identical to plain V2V; partial mask -> preserved
  columns exact, edited columns changed; scheduler-state composite verified numerically.
- Model-backed proof (`validation_outputs/masked_v2v_proof_2026_07_04/`, promoted to
  `docs/assets/examples/conference-masked-v2v/`):
  - conference gender swap, 480x832, 25 frames, steps 20, strength 0.8, seed 8602, q8 package
  - preserved-region drift 1.73 vs plain-V2V 14.92, at the measured H.264 floor of 1.92 (PASS)
  - edited-region delta 23.35 (edit went through, PASS)
  - temporal delta 2.15 vs source 2.44 (PASS); no visible boundary halo in the zoom crops
  - wall 29.3 min, peak RSS 14.83 GB

## Outcome

Masked video-to-video is shipped on the existing public route with exact background
preservation, fail-closed contracts, replayable metadata, and an in-repo reproducible proof.
VACE-style learned conditioning (reference images, control inputs) remains follow-up work in
[0075](../proposed/0075_wan_vace_conditioning_expansion_after_plain_video_to_video.md).
