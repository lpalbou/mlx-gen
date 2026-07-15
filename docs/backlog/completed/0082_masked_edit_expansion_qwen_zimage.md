# 0082 - Masked Edit Expansion: Native Base-Qwen And Z-Image Non-Turbo

- Status: completed (2026-07-15)
- Scope: close the two actionable gaps from the 2026-07-15 masked-edit surface audit -
  native base-Qwen masked edit (`qwen.base-inpaint`, diffusers `QwenImageInpaintPipeline`
  port) and the Z-Image non-turbo inpaint ungate - plus the consolidated
  `docs/masked-editing.md` page. The audit's other two items were resolved by decision:
  control-inpaint row broadening was superseded (see below), and Z-Image ControlNet-inpaint
  stays proposed (0045, refreshed with sizing facts).
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
  (model-backed proofs below), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
  (one masked route per row, fail-closed gating, plan-time maskless rejection)
- Validation: one Fable 5 adversarial subagent ran two rounds (pre-implementation design
  attack, post-implementation code audit); focused suites green (task inference, router 266,
  masked routes, arg parser; full fast band 633), lint clean; visual-smoke proof bundle in
  `docs/assets/validation/masked-edit-2026-07-15/`

## What shipped

1. Runtime: `QwenImage.generate_image` gains `mask_path` - encode the clean source once,
   reuse `QwenEditUtil.create_inpaint_mask_latents`/`blend_inpaint_latents` per step, warm
   start from the re-noised source at the internal upstream-example strength 0.85
   (`MASKED_EDIT_STRENGTH`), record `effective_steps` + the applied warm-start strength in
   metadata (key shipped as `masked_warm_start_strength`, renamed to `mask_strength` the same
   day when the strength became the tunable `--mask-strength` option) in
   metadata (Wan v2v runtime-truth precedent), label progress `image-to-image`, and reject
   `image_strength`/missing-image combinations. The public `--image-strength`-with-mask
   rejection is unchanged repo-wide.
2. Capability layer: new `qwen.base-inpaint` on trusted base rows, mutually exclusive with
   `qwen.control-inpaint` by construction (`_supports_qwen_base_native_inpaint` short-circuits
   on the control gate); untrusted inferred identities stay fail-closed except exact proven
   rows (`QWEN_BASE_NATIVE_INPAINT_EXACT_ROWS`, currently the 2512-8bit row). `qwen.latent`
   is now `default_for_task` so maskless routing keeps the actionable strength error, and any
   masked capability selected without a mask fails at plan time. Z-Image inpaint extends to
   all trusted identities (turbo and non-turbo), same 0070-hardening trust gate.
3. CLI: `mflux-generate-qwen` dispatches masked requests by plan (`control_model is None` ->
   plain `QwenImage`; sidecar row unchanged) and rejects `--controlnet-model`/explicit
   `--controlnet-strength` on the native route; `mflux-generate-z-image` gains `--mask-path`
   with the same validations as the turbo command; completions for qwen/z-image/z-image-turbo
   re-mirrored (two were already stale at HEAD).
4. Docs: new canonical `docs/masked-editing.md` (contract, model matrix with proof grades,
   per-family behavior including the 0.85 warm-start divergence from the pin-1.0 routes and
   the non-turbo explicit-guidance note); `api.md`, `faq.md`, `image-edit-modes.md`,
   `qwen-route-matrix.md` (public-boundary decision rewritten), `qwen-localized-editing.md`,
   `getting-started.md`, `docs/README.md`, root `README.md`, and both llms files now
   summarize and link there.
5. Tests: trusted/exact/untrusted gating matrix, one-masked-route-per-row invariant,
   native-vs-sidecar CLI dispatch, `-C` replay without sidecar injection, warm-start schedule
   pin (20 steps -> init index 3, 17 executed), progress/task labeling with metadata truth,
   Z-Image non-turbo routing/backend forwarding, plan-time maskless rejection.

## Decisions of record

- Item 2 of the audit ("broaden `qwen.control-inpaint` to 4bit/source") was superseded rather
  than implemented: broadening the shared `_supports_qwen_base_control` helper would also
  unlock unvalidated structured control on those rows, and two masked routes on one row would
  make `--mask-path` selection ambiguous (both are `edit-reference`). Native `qwen.base-inpaint`
  now covers those rows without any sidecar download. Consequence, stated in the docs: the
  identical masked request runs the sidecar route on the exact 8bit row and the native
  warm-start route everywhere else.
- The 0.85 warm start is an internal constant, not a public flag: exposing a strength on one
  masked route would fracture the repo-wide "no `--image-strength` with `--mask-path`"
  contract. Empirical basis (2512-8bit row, same seed/case): pure-noise start paints
  unrelated content into the mask, upstream signature default 0.6 barely repaints, upstream
  docstring example 0.85 produces the correct masked object insertion.
- `Qwen/Qwen-Image` source and prepared bf16 rows initially shared the native-inpaint wiring
  without a dedicated run (bf16-scale checkpoints did not fit local disk at the time). Closed
  2026-07-15 after disk was freed: the source bf16 row ran the four scored matrix cases
  (PARTIAL aggregate, same recolor signature as the prepared rows; source-row addendum in the
  matrix bundle). No prepared bf16 package is published, so the source row is the only bf16
  surface.

## Proof

`docs/assets/validation/masked-edit-2026-07-15/` - shared object-insertion case
(`tests/resources/glasses.jpg`, seed 42): `AbstractFramework/qwen-image-2512-8bit` (native,
17 executed steps, 55 s), `AbstractFramework/qwen-image-4bit` (native, 71 s),
`AbstractFramework/z-image-8bit` (non-turbo, 30 steps, explicit guidance 4, 98 s). Outside-mask
mean abs pixel diff 0.46-0.56/255 (16 px boundary band excluded) vs 29-137/255 inside the mask;
contact sheet plus runtime-truth metadata sidecar included. Grade: visual smoke, single case.

## Follow-ups

- Promote the smoke rows to validated visual-QA rows (multi-case matrix) before any
  release-notes "validated" claim. (Done 2026-07-15 same day: the masked-edit 5x5 matrix in
  `docs/assets/validation/masked-edit-matrix-2026-07-15/` and registry profile
  `masked_edit_matrix_5x5_2026_07_15` grade all four new-route rows - Klein 4B q8 rows PASS,
  qwen.base-inpaint rows PARTIAL with the documented warm-start recolor limitation, Z-Image
  non-turbo mixed with a seed-reproducible arm-retexture geometry failure.)
- Z-Image ControlNet-inpaint remains [0045](../proposed/0045_zimage_controlnet_followup.md),
  refreshed 2026-07-15 with upstream sizing facts and a dedicated-session execution plan.
- Post-matrix consequences (2026-07-15, same day): the non-turbo Z-Image route this item
  opened was withdrawn after the matrix ARM failure reproduced across seeds and CFG settings
  (`z-image.inpaint` is Turbo-only again; evidence retained in the matrix bundle and
  registry), and `qwen.base-inpaint` gained the tunable `--mask-strength` warm start (default
  0.85; measured 0.95 fixes the PARTIAL recolor cells).
- Re-check trigger for the withdrawn non-turbo Z-Image route: revisit only when upstream
  ships a non-turbo inpaint reference or demonstration (diffusers `ZImageInpaintPipeline`
  currently documents Turbo exclusively), or when a new non-turbo checkpoint revision lands.
  Reproduce the matrix ARM case first; do not re-open the capability on settings tweaks
  alone (guidance and seed variations were already measured as non-fixes).
