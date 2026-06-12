# Proposed: LightX2V Wan distilled-model loader support

## Metadata

- Created: 2026-06-12
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: May require an ADR if MLX-Gen adds a new public model-family surface for native
  distilled Wan checkpoints rather than treating them as explicit alternate Wan packages.

## Context

LightX2V publishes two different kinds of Wan acceleration assets:

- LoRA-based 4-step acceleration, which fits the current MLX-Gen Wan runtime reasonably well.
- Full distilled-model checkpoints, published under `lightx2v/Wan2.2-Distill-Models`.

The second path could be attractive because it may offer a simpler user experience than paired
LoRAs and may eventually include BF16, FP8, and INT8 variants. The 2026-06-12 audit shows that
this is not only a file-layout question: the native LightX2V distill path also uses a dedicated
step-distill scheduler contract and a distinct transformer state-dict schema.

## Current code reality

- `ModelConfig` currently knows the official Wan-AI model identities and MLX-Gen prepared-package
  identities, not `lightx2v/Wan2.2-Distill-Models`.
- `WanWeightDefinition` and the Wan initializer currently expect the standard component-oriented
  Wan package layout used by official Wan-AI models and current prepared packages.
- MLX-Gen's current Wan scheduler is `WanUniPCMultistepScheduler`. With `steps=4` and
  `flow_shift=5.0`, it currently generates timesteps equivalent to `[999, 937, 833, 625]`, not
  the LightX2V distill list `[1000, 750, 500, 250]`.
- The LightX2V distilled-model repos expose alternate packaging:
  - paired high-noise / low-noise distilled checkpoints;
  - BF16, FP8, and INT8 variants;
  - both monolithic and split-file forms.
- The downloaded `config.json` still identifies the class as `WanModel`, which suggests tensor
  compatibility may be reachable, but loader-level compatibility is not proven.
- The upstream LightX2V examples do not treat the distilled repo as a full standalone model
  package. They still use the base Wan model path for shared assets and point only the
  high-noise/low-noise transformers at the distilled checkpoints.
- The upstream native distill runner uses:
  - `Wan22StepDistillScheduler`
  - `denoising_step_list`
  - `boundary_step_index`
  - `enable_cfg=false`
  for the main native distilled examples.
- The native distilled transformer state dict is not named like the official Diffusers Wan state
  dict. For example, split INT8 checkpoints use keys such as:
  - `blocks.0.self_attn.q.weight`
  - `blocks.0.cross_attn.k.weight`
  - `blocks.0.modulation`
  - `*.weight_scale`
  while official Diffusers Wan checkpoints use patterns such as:
  - `blocks.0.attn1.to_q.weight`
  - `blocks.0.attn2.to_k.weight`
  - `blocks.0.scale_shift_table`
- MLX-Gen does not currently expose any native `lightx2v/*` Wan model alias or package validation
  row.

## Problem or opportunity

If the native LightX2V distilled checkpoints can load on the existing Wan transformer stack with
bounded adapter work, MLX-Gen could expose a very fast official Wan path without requiring users to
manage paired LoRAs manually. If they cannot, the project should say so clearly and keep the
faster path recipe-based through LoRAs instead of pretending both integration paths are equivalent.

The main architectural ambiguity is now explicit:

- direct “model handle” support for `lightx2v/Wan2.2-Distill-Models`;
- or a hybrid Wan variant that keeps the base Wan package for tokenizer/text encoder/VAE while
  overriding only the high-noise and low-noise transformers plus the scheduler mode.

## Proposed direction

Investigate native distilled-model loading as a separate follow-up after LoRA-based 4-step
acceleration is proven:

1. Audit the exact file layout of `lightx2v/Wan2.2-Distill-Models`.
2. Determine the smallest honest first milestone. Based on current evidence, that is likely:
   - BF16 A14B only;
   - explicit high-noise and low-noise distilled transformer inputs;
   - base Wan package retained for tokenizer, text encoder, and VAE;
   - a dedicated distill scheduler mode with explicit timestep lists.
3. Decide whether that first milestone should be modeled as:
   - a hybrid override path on top of base Wan;
   - or a packaged MLX-Gen prepared variant that normalizes the component layout.
4. Keep raw direct support for FP8 / INT8 / split checkpoints out of the first milestone until BF16
   is proven first.
5. Only promote this item if the loader and scheduler delta is bounded and native checkpoints
   provide a clearer user value than the already-usable LoRA path.

## Why it might matter

This could become the cleanest user-facing “fast Wan” experience in MLX-Gen, but only if the
native checkpoint layout and distill scheduler contract are supportable without destabilizing the
existing Wan route.

## Promotion criteria

- Completed item [0040](../completed/0040_lightx2v_wan_4step_acceleration_profiles.md) proves that
  the LightX2V 4-step LoRA path is genuinely useful and worth preserving separately from native
  distilled-model work.
- One exact `lightx2v/Wan2.2-Distill-Models` file set is audited completely enough to know whether
  it matches current Wan tensor expectations.
- The first target is reduced to one concrete route, preferably BF16 `T2V-A14B` or `I2V-A14B`,
  instead of “support every distilled variant”.
- The first milestone is explicit about the scheduler contract:
  - `denoising_step_list`
  - `boundary_step_index`
  - whether CFG stays disabled
- A fail-closed package identity contract is clear: no silent fallback from a native distilled
  model request to a base-Wan-plus-LoRA path.

## Validation ideas

- Small BF16 smoke on one exact LightX2V distilled A14B route.
- Loader audit with tensor-name or component-name comparison against the official Wan layout.
- Scheduler audit proving whether MLX-Gen reproduces the upstream distill timesteps exactly.
- Same prompt/source seed-controlled comparison between:
  - base Wan;
  - base Wan + LightX2V LoRAs;
  - native LightX2V distilled checkpoint.
- Saved metadata proving the exact model identity used.

## Non-goals

- Do not merge this item into the LoRA-based acceleration backlog.
- Do not start with FP8 or INT8 unless BF16 loader support is already proven.
- Do not assume `steps=4` plus `flow_shift=5.0` is sufficient for native parity; the explicit
  distill scheduler contract must be evaluated.
- Do not assume raw `lightx2v/Wan2.2-Distill-Models` can be treated as a full standalone model
  package without the base Wan assets.
- Do not broaden this into Wan VACE or generic second-family video acceleration work.

## Guidance for future agents

Treat this as a scheduler-plus-loader audit, not just a filename audit. The existence of a
`WanModel` config and public weights is encouraging, but current evidence already shows three
distinct deltas:

- raw package layout differs from official Wan Diffusers repos;
- distilled transformer state-dict naming differs from current Wan mapping expectations;
- the upstream distill scheduler uses an explicit timestep list and boundary index that MLX-Gen
  does not currently expose.

The most plausible first implementation is a BF16 A14B hybrid path on top of a base Wan package,
not raw direct support for every file in `Wan2.2-Distill-Models`.

## Sources checked

- `src/mflux/models/wan/wan_initializer.py`
- `src/mflux/models/wan/scheduler/wan_unipc_multistep_scheduler.py`
- `src/mflux/models/wan/variants/wan2_2_ti2v.py`
- `src/mflux/models/common/config/model_config.py`
- https://huggingface.co/lightx2v/Wan2.2-Distill-Models
- https://huggingface.co/lightx2v/Wan2.2-Distill-Loras
- https://huggingface.co/lightx2v/Wan2.2-Lightning
- https://github.com/ModelTC/Wan2.2-Lightning
- `/tmp/LightX2V/lightx2v/models/runners/wan/wan_distill_runner.py`
- `/tmp/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py`
- `/tmp/LightX2V/examples/wan/wan_i2v_distilled.py`
