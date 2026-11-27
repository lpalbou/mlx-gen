# Completed: Wan step-distill scheduler mode (`denoising_step_list` parity)

## Metadata

- Created: 2026-07-25
- Status: Completed (scheduler mode + CLI + parity audit; the grid-honest
  Lightning-vs-base chain A/B remains an open follow-up recorded below)
- Completed: 2026-07-25 — released in 0.25.0 (tag `v0.25.0` from `2452f0c`,
  workflow 30162410505 green; PyPI + GitHub Release verified); release
  checklist and gate evidence in [0101](0101_release_0_25_0.md).
- Effort: S-M (scheduler mode + CLI surface + parity audit)

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md) (an explicit
  step-list request must never silently fall back to the UniPC grid).
- ADR impact: None until shipped.

## Context (investigation provenance)

Item 0041's audit already established the gap: with `steps=4`,
`flow_shift=5.0`, `WanUniPCMultistepScheduler` lands on timesteps
`[999, 937, 833, 625]`, while the upstream LightX2V distill contract is the
explicit list `[1000, 750, 500, 250]` (`Wan22StepDistillScheduler`,
`denoising_step_list`, `boundary_step_index`, `enable_cfg=false`). This repo
cannot express an explicit step grid today.

The 2026-07-25 storyboard consistency investigation adds a consumer-side
reason to promote just this slice, independent of the native distilled-model
loader question (which stays in 0041):

- Chained storyboard scenes run the Lightning 4-step CFG-off recipe per hop;
  part of the observed per-hop stylization drift ("cartoonish" tone ramp,
  Hasler-Susstrunk colorfulness 32 → 41.5 across the real film) is plausibly
  the distilled CFG-off look compounding, but that attribution CANNOT be
  tested honestly while the sampler cannot reproduce the distill-native grid:
  every Lightning-vs-base comparison currently confounds recipe with grid.
- Prior v2v measurements in this repo's investigation history show CFG-on
  halves frame-to-frame drift (mean per-frame delta 17 vs 34 at matched
  strength) — sizing the true Lightning-vs-base trade for i2v chains needs
  the grid-exact distill mode first.

## What we want to do

1. Add an explicit step-grid mode to the Wan scheduler surface
   (`denoising_step_list` + `boundary_step_index`), fail-closed when combined
   with options it cannot honor (ADR 0002).
2. Parity audit: reproduce the upstream distill timesteps exactly (0041's
   validation idea, scoped to the scheduler only).
3. Re-run the Lightning 4/8-step vs base-recipe comparison on the preserved
   storyboard chain case with the grid held honest, and record the
   drift-vs-cost verdict (the investigation left this as the one un-run A/B —
   a 20-step CFG-on scene is ~10x a Lightning scene on the same host, which
   exceeded the investigation's GPU budget).

## Non-goals

- Native `lightx2v/Wan2.2-Distill-Models` checkpoint loading (stays in 0041;
  state-dict naming and package layout are a separate, larger problem).
- Changing the shipped Lightning LoRA profile defaults.

## Dependencies and related tasks

- [0041](0041_lightx2v_wan_distilled_model_loader_support.md) (parent audit;
  keep it for the loader/checkpoint question).
- Host application companion track `proposed/storyboard_consistency_2026_07/`
  (consumes the drift-vs-cost verdict for its per-scene quality dial).

## Validation

- Scheduler unit parity vs the upstream distill timestep list.
- Fixed-seed chain A/B (Lightning grid-exact vs current UniPC approximation
  vs base 20-step CFG-on) with named metrics: Laplacian-variance sharpness,
  Hasler-Susstrunk colorfulness ramp, upper-band edge energy.

## Progress checklist

- [x] Explicit step-grid scheduler mode, fail-closed combinations
- [x] Upstream timestep parity audit
- [ ] Grid-honest Lightning vs base chain A/B with recorded verdict
  (deliberately left open: a 20-step CFG-on scene is ~10x a Lightning scene;
  this wave's GPU budget was spent on the 0097 quality gate)

## Implementation record (2026-07-25, pending release)

- Both schedulers (`WanUniPCMultistepScheduler`, `WanEulerScheduler`) accept
  `set_timesteps(denoising_step_list=[...])` as the exact-grid alternative to
  the count path; `WanTimestepGrid` (new
  `scheduler/wan_timestep_grid.py`) owns validation (non-empty, integer
  entries in [1, 1000], strictly decreasing) and the sigma identity.
- Sigma/timestep derivation (the parity audit): grid entries are FINAL,
  already-shifted timesteps — the transformer sees exactly the requested
  values ([1000, 750, 500, 250] reproduces the upstream
  `Wan22StepDistillScheduler` list verbatim, vs the count path's
  [999, 937, 833, 625] at steps=4/shift=5). Sigma follows the flow-matching
  identity `t / num_train_timesteps` (the exact inverse of the count path's
  `t = int(sigma * 1000)` map). The UniPC grid keeps the count path's own
  leading `1e-6` guard (sigma == 1 makes `log(1 - sigma)` degenerate in the
  order-2 corrector); the euler grid does not clamp, matching euler's own
  count path which starts at sigma 1.0 exactly.
- flow_shift interaction (documented + enforced): the shift already happened
  when a distill list was designed, so grid mode never consults flow_shift.
  An explicit `flow_shift` alongside `denoising_step_list` is a hard error
  (model + CLI); metadata records `flow_shift: null` on grid runs so
  `--config-from-metadata` replay cannot conflict.
- Fail-closed combinations (ADR 0002): mutually exclusive with
  `num_inference_steps`/`--steps`; rejected for video-to-video
  (`video_strength` truncation would silently drop grid points); rejected on
  Wan VACE. `generate_video(num_inference_steps=...)` default moved from a
  literal 50 to `None`-resolves-to-50 so the exclusion is exact — no behavior
  change without the new argument (pinned by test).
- Metadata + replay: `denoising_step_list` recorded; `steps` records the grid
  length; explicit `--steps` on the command line beats a recorded grid
  (metadata-defaults convention).
- Tests (no GPU): scheduler contract tests in
  `tests/wan/test_wan_scheduler_and_timesteps.py` (distill-list parity on
  both solvers, flow-shift independence, count-path sigma coincidence bounds,
  step-math mode-agnosticism, malformed-grid rejection, exactly-one-source
  validation) and generate-level wiring in
  `tests/wan/test_wan_step_grid_generation.py` (scheduler receives the grid,
  loop consumes exact grid timesteps, boundary routing on grid values,
  metadata/replay, all rejection combos); CLI bind-contract + replay tests in
  `tests/cli/test_mlx_gen_router.py`.
- NOT implemented (scope): `boundary_step_index` — the runtime's
  `boundary_ratio` comparison already routes experts per grid timestep value
  (grid point 1000 -> high expert, 750/500/250 -> low at the t2v 0.875
  boundary; test-pinned), so a separate index knob would duplicate authority.
  The native distilled-checkpoint loader stays in 0041.

## Cycle-2 verification (2026-07-25)

- Sigma-skew bound verified numerically: feeding each count schedule's int
  timesteps back as a grid, max |sigma_count - sigma_grid| = 9.9897e-4
  (< 1/1000) across flow_shift {1, 3, 5, 12} x steps {4, 8, 20, 50}; the
  maximum occurs at the clamped leading point (count sigma 1 - 1e-6 vs grid
  t=999 -> 0.999). Euler at int-cast coinciding points: 5.0e-4. No count
  schedule in that sweep produces duplicate int timesteps; grid validation's
  strictly-decreasing rule makes UniPC `_index_for_timestep`'s
  duplicate-match branch unreachable in grid mode (count-mode duplicates
  keep it).
- Boundary routing at the sharp edge now test-pinned: a grid value exactly ON
  the boundary (875 at the t2v 0.875 ratio) routes HIGH per the diffusers
  `t >= boundary` convention ([1000, 875, 500, 250] -> 2 high / 2 low).
- Replaced a vacuous assertion in the t2v grid sanity test with a real
  scale_noise-never-called check.
- Replay matrix hardened: a hand-edited sidecar carrying BOTH a grid and a
  numeric flow_shift (no real run records that combination) replays the grid
  and forwards flow_shift=None — pinned by a CLI test; explicit `--steps` on
  argv still beats a recorded grid (already pinned).
- Host application compat confirmed READ-ONLY: the worker forwards video kwargs by
  name (config-derived step defaults, no signature-default introspection),
  its signature filter applies to `generate_image` only, and the runtime
  wrapper passes `generate_video` kwargs through unmodified — the
  `num_inference_steps: int | None` migration and the two new keyword args
  are invisible at the current 0.24.0 pin.
