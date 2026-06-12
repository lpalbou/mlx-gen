# Completed: LightX2V Wan 4-step acceleration profiles

## Metadata

- Created: 2026-06-12
- Status: Completed
- Completed: 2026-06-12

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None if MLX-Gen keeps LightX2V acceleration as an explicit Wan LoRA profile with
  exact adapter files, exact target roles, and exact inference settings instead of a silent
  alternate default.

## Outcome

Completed with the bounded `lightx2v/Wan2.2-Lightning` path on current Wan A14B q8 public routes.
MLX-Gen now has curated 4-step LightX2V proof rows for:

- `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` on `wan.text-video`
- `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` on `wan.first-frame`

The accepted proof uses:

- `steps=4`
- `flow_shift=5.0`
- `guidance=1.0`
- `guidance_2=1.0`
- exact paired `high_noise_model.safetensors` / `low_noise_model.safetensors` files from
  `lightx2v/Wan2.2-Lightning`

Published proof assets:

- `docs/assets/validation/lightx2v-wan-4step-2026-06-12/a14b_t2v_lightx2v_4step_ab_contact_sheet.jpg`
- `docs/assets/validation/lightx2v-wan-4step-2026-06-12/a14b_i2v_lightx2v_4step_ab_contact_sheet.jpg`
- `docs/assets/validation/lightx2v-wan-4step-2026-06-12/a14b_t2v_lightx2v_81f_speed_comparison.jpg`
- `docs/assets/validation/lightx2v-wan-4step-2026-06-12/a14b_i2v_lightx2v_81f_speed_comparison.jpg`
- `docs/assets/validation/lightx2v-wan-4step-2026-06-12/a14b_t2v_lightx2v_81f_step_sweep_m5max.jpg`
- `docs/assets/validation/lightx2v-wan-4step-2026-06-12/a14b_t2v_lightx2v_480p_probe_m5max.jpg`
- `docs/assets/validation/lightx2v-wan-4step-2026-06-12/long_run_speed_stats.json`

The no-LoRA 4-step baselines were materially worse on both routes, which is the point of this
profile. No general runtime default was changed. The fast path stays explicit through paired LoRAs,
exact target roles, and exact settings.

Longer-run timing evidence also now exists against the current practical original A14B profiles at
`81` frames and `20` fps:

- T2V-A14B q8 original practical profile: `20` steps, `guidance=4`, `guidance_2=3`,
  `flow_shift=3` -> `1356.45s`
- T2V-A14B q8 LightX2V Lightning profile: `4` steps, `guidance=1`, `guidance_2=1`,
  `flow_shift=5` -> `163.94s`
- measured T2V speedup: `8.27x`
- I2V-A14B q8 original practical profile: `20` steps, `guidance=3.5`, `guidance_2=3.5`,
  `flow_shift=3` -> `887.40s`
- I2V-A14B q8 LightX2V Lightning profile: `4` steps, `guidance=1`, `guidance_2=1`,
  `flow_shift=5` -> `144.70s`
- measured I2V speedup: `6.13x`

These numbers are important because they show what the item actually delivered: a much faster
explicit recipe on top of the current Wan runtime, not a claim that the 4-step profile is
universally higher quality than the original longer Wan profile.

Additional T2V-only quality investigation on the same `M5 Max` also showed:

- raising the Lightning quick profile from `4` to `6` or `8` steps at `480x240` recovers some
  detail, but it still does not match the original `20`-step practical profile
- a `832x480`, `41`-frame, `4`-step Lightning probe is visibly stronger than the `240p` quick
  profile
- the first bad-frame issue seen later on q8 `1280x720` T2V was a real runtime precision bug, not
  a LightX2V warm-up artifact; the decisive fix was to keep the Wan FFN LoRA target family
  (`ffn.net.0` and `ffn.net.1`) at BF16 runtime precision alongside the already protected
  attention-family paths, after which the q8 `720p` run started clean from frame `0` and tracked
  the BF16 reference

That pushes the conclusion in one clear direction: the T2V weakness is mainly a quality-envelope
tradeoff for the ultra-fast `240p` recipe, not evidence of a loader or scheduler bug in the
current MLX-Gen LightX2V path once the q8 FFN runtime fix is present.

## Context

LightX2V has published Wan2.2 acceleration work that reduces generation to four denoising steps.
That is strategically interesting because MLX-Gen already has working Wan routing, Wan LoRA
support, and explicit high-noise / low-noise role selection for A14B. If the LightX2V 4-step path
works cleanly inside the current Wan runtime, it could provide the fastest useful local video path
in MLX-Gen without introducing a second video architecture.

The important upstream split is:

- `lightx2v/Wan2.2-Distill-Loras`: LoRA files extracted from the distilled models and intended to
  run on top of the base Wan2.2 models.
- `lightx2v/Wan2.2-Lightning`: paired high-noise / low-noise 4-step LoRAs and related packaging.
- `lightx2v/Wan2.2-Distill-Models`: full distilled checkpoints, which are a separate loader
  problem and should not be conflated with the LoRA-based acceleration path.

Update 2026-06-12: the native LightX2V distill configs use a dedicated step-distill scheduler with
an explicit `denoising_step_list` and `boundary_step_index`, not the ordinary Wan linear-step
schedule that MLX-Gen currently exposes. That means the first exact current-runtime target should
prefer `Wan2.2-Lightning`, while `Wan2.2-Distill-Loras` should be treated as a second-stage target
that depends on the native distill scheduler investigation in proposed item 0041.

## Current code reality

- MLX-Gen already supports Wan LoRA loading through `mlxgen generate` and the dedicated Wan CLI.
- `src/mflux/models/wan/wan_initializer.py` already supports explicit `lora_target_roles`:
  - `transformer` for TI2V-5B;
  - `high_noise_transformer` / `low_noise_transformer` for A14B.
- Completed item [0033](../completed/0033_video_lora_for_t2v_i2v.md) already proved Wan q8 LoRA
  routing across all current public Wan directions.
- `src/mflux/models/wan/cli/wan_generate.py` already exposes the key knobs that LightX2V relies
  on: `--steps`, `--guidance`, `--guidance-2`, and `--flow-shift`.
- The current MLX-Gen defaults are not LightX2V defaults. MLX-Gen generally defaults to long Wan
  runs, while the LightX2V A14B examples use `4` steps, `flow_shift=5.0`, and guidance values of
  `1.0`.
- The local audit of LightX2V assets found practical A14B paired LoRAs but not a clean public
  4-step TI2V-5B proof file set yet. The LightX2V README lineage mentions TI2V-5B, but the
  readily usable adapter files currently center on A14B.
- The `Wan2.2-Distill-Loras` configs do not just set `steps=4`; they also rely on explicit
  distill-timestep lists such as `[1000, 750, 500, 250]`. MLX-Gen's current Wan scheduler does
  not expose that schedule directly.
- MLX-Gen now surfaces exact route-level LightX2V 4-step validation profiles for the current A14B
  q8 public routes. The route capability rows now point at the accepted 2026-06-12 LightX2V proof
  profiles rather than the earlier effect-specific A14B adapter examples.

## Problem

MLX-Gen can already load Wan LoRAs, but that is not the same as exposing an officially guided
4-step acceleration profile. Users currently have to guess:

- which LightX2V repo to use;
- which paired files belong to T2V-A14B versus I2V-A14B;
- which `--lora-target-roles` are correct;
- whether `guidance` should stay at the normal Wan defaults or drop to `1.0`;
- whether `flow_shift` should stay at MLX-Gen defaults or move to `5.0`.

Without a bounded implementation pass, users can easily get weak outputs, misapply only one A14B
LoRA file, or assume that native distilled-model support already exists when it does not.

## What we want to do

Add an explicit LightX2V 4-step acceleration track on top of the current Wan runtime:

1. Audit the exact public adapter files to bless as the MLX-Gen proof set.
2. Reproduce the official 4-step A14B settings faithfully inside MLX-Gen where the current Wan
   runtime can do so exactly.
3. Generate model-backed A/B MP4 proofs and contact sheets for:
   - `Wan2.2-T2V-A14B`
   - `Wan2.2-I2V-A14B`
4. Publish clear CLI examples, validation profiles, and capability notes so AbstractVision can
   surface the fast path honestly.
5. Decide whether TI2V-5B belongs in this item or should stay deferred until an exact public
   4-step file set is audited.

## Why

This is a high-value speed feature that fits the current architecture. It is more tractable than
native distilled-model loading and more directly useful than another round of vague video speed
claims with no official upstream alignment.

## Requirements

- Keep the LightX2V path explicit. No automatic step reduction, guidance changes, or role
  assignment just because a LightX2V adapter path was passed.
- Treat T2V-A14B and I2V-A14B as separate validated rows with separate proof artifacts.
- Record exact adapter files, exact target roles, and exact settings in metadata and docs.
- Fail closed if only one of the paired A14B files is supplied for a profile that expects both.
- Do not claim TI2V-5B 4-step support unless exact public TI2V-5B LightX2V files are audited and
  visually proven.

## Suggested implementation

1. Choose one primary public file family:
   - preferred first target: `lightx2v/Wan2.2-Lightning`
   - secondary exactness target, after 0041 reduces scheduler ambiguity:
     `lightx2v/Wan2.2-Distill-Loras`
2. Add validation profiles or equivalent documented recipes for:
   - `A14B T2V 4-step`
   - `A14B I2V 4-step`
3. Keep the first pass recipe-driven rather than inventing a new public `--distilled` switch.
4. Decide after proof generation whether MLX-Gen needs stricter helper validation for paired
   A14B files, for example a profile-aware check that both high-noise and low-noise files were
   supplied.
5. Keep native LightX2V distilled checkpoints out of scope for this item; track them separately.

## Public proof targets

### Preferred A14B T2V files

- `lightx2v/Wan2.2-Lightning`
  - `Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors`
  - `Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors`

### Preferred A14B I2V files

- `lightx2v/Wan2.2-Lightning`
  - `Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors`
  - `Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors`

### Secondary compatibility files

- `lightx2v/Wan2.2-Distill-Loras`
  - `wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors`
  - `wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors`
  - `wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors`
  - `wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors`

## Suggested baseline settings to verify

- For the first exact current-runtime pass, use the Lightning-style baseline:
  - `steps=4`
  - `flow_shift=5.0`
  - `guidance=1.0`
  - `guidance_2=1.0` for A14B
- same-seed A/B with and without the LoRAs

These settings come from the current `Wan2.2-Lightning` A14B configs and should be treated as
upstream proof targets, not MLX-Gen defaults. The `Wan2.2-Distill-Loras` family carries a
different distill-scheduler contract and should not be treated as an exact Lightning-equivalent
baseline until proposed item 0041 settles the scheduler question.

## Scope

- Official LightX2V 4-step LoRA-based acceleration for current Wan A14B routes, starting with the
  Lightning family.
- Contact sheets, MP4 proof assets, validation profiles, and docs.
- Optional small helper validation around paired A14B file usage if needed.

## Non-goals

- Do not add native `lightx2v/Wan2.2-Distill-Models` support here.
- Do not claim exact `Wan2.2-Distill-Loras` parity on the current scheduler until item 0041
  resolves the distill-timestep contract.
- Do not silently override normal Wan defaults outside the explicit LightX2V profile path.
- Do not claim TI2V-5B 4-step support from README mentions alone.
- Do not broaden this item into Wan VACE, SeedVR2 restoration, or second-family video work.

## Dependencies and related tasks

- [0007 LoRA capability matrix and strict application](0007_lora_capability_matrix_and_strict_application.md)
- [0015 Wan prompt adherence parity validation](0015_wan_prompt_adherence_parity_validation.md)
- [0033 Video LoRA support for T2V and I2V](../completed/0033_video_lora_for_t2v_i2v.md)
- [0035 Wan2.2 TI2V-5B math and behavior parity](0035_wan_ti2v5b_math_and_behavior_parity.md)
- [0041 LightX2V Wan distilled-model loader support](../proposed/0041_lightx2v_wan_distilled_model_loader_support.md)
- `src/mflux/models/wan/`

## Expected outcomes

- One validated 4-step A14B T2V profile.
- One validated 4-step A14B I2V profile.
- Proof assets and docs that distinguish “LoRA-based 4-step acceleration” from “native distilled
  model support”.
- A clear answer on whether TI2V-5B belongs in a later follow-up or has a usable public 4-step
  path now.

## Validation

- Same prompt or source image, seed, dimensions, frames, and fps across no-LoRA vs 4-step-LoRA
  runs.
- MP4 outputs plus frame strips or contact sheets for T2V-A14B and I2V-A14B.
- Saved metadata proving the exact adapter file names, target roles, scales, step count,
  `flow_shift`, `guidance`, and `guidance_2`.
- Focused tests for any new validation-profile or paired-file helper logic.

## Progress checklist

- [x] Confirm the exact LightX2V proof files to support first.
- [x] Validate A14B T2V 4-step on the current Wan runtime.
- [x] Validate A14B I2V 4-step on the current Wan runtime.
- [x] Decide whether a profile-aware paired-file guard is needed.
- [x] Document the fast-path examples and promote only the exact rows that passed.

## Guidance for the implementing agent

Treat this as a bounded A14B acceleration task, not a general Wan redesign. Start from the exact
LightX2V settings, keep the role assignment explicit, and produce proof assets before discussing
speed claims in release docs.

## Completion notes

- The paired `Wan2.2-Lightning` A14B files matched `1200/1200` keys on both the high-noise and
  low-noise transformers and applied `400` targets per file.
- The accepted T2V proof used a starship takeoff prompt at `480x240`, `41` frames, `20` fps,
  requested through the q8 prepared package.
- The accepted I2V proof used the canonical spaceship-snow source image at requested `480x240`,
  which resolved to `448x256` because Wan I2V preserves source aspect ratio.
- A general runtime “both files required” guard was not added, because ordinary Wan LoRA routing
  remains intentionally flexible. The exact fast path is instead fail-closed by recipe, validation
  profile, and explicit documentation.

## Sources checked

- `src/mflux/models/wan/wan_initializer.py`
- `src/mflux/models/wan/cli/wan_generate.py`
- [0033 Video LoRA support for T2V and I2V](../completed/0033_video_lora_for_t2v_i2v.md)
- https://huggingface.co/lightx2v/Wan2.2-Distill-Loras
- https://huggingface.co/lightx2v/Wan2.2-Lightning
- https://huggingface.co/lightx2v/Wan2.2-Distill-Models
- https://github.com/ModelTC/Wan2.2-Lightning
