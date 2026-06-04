# Planned: Wan prompt adherence parity validation

## Metadata

- Created: 2026-06-03
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None for validation. Revisit ADR status only if this work changes public Wan
  prompt, negative-prompt, guidance, routing, or fallback semantics.

## Context

MLX-Gen has source-level parity checks for Wan prompt embeddings and denoising mechanics, and it
now supports Wan TI2V-5B plus A14B T2V/I2V routes. The remaining gap is generation-level prompt
adherence evidence: we do not yet have same-settings Diffusers-vs-MLX contact sheets and metrics
showing that motion prompts influence Wan T2V and I2V similarly.

The latest source audit clarified an important default distinction:

- Official Wan code defines the shared Chinese `sample_neg_prompt` and uses it when no negative
  prompt is passed.
- Official Wan A14B configs express guidance as `(low noise, high noise)`: T2V uses `(3.0, 4.0)`
  and I2V uses `(3.5, 3.5)`.
- MLX-Gen stores the same semantic values as `guidance` for the high-noise denoiser and
  `guidance_2` for the low-noise denoiser.
- When a caller explicitly passes one scalar guidance value and omits the low-noise value, official
  Wan applies that scalar to both denoisers; MLX-Gen follows that behavior.
- Raw Diffusers pipelines default `negative_prompt` to an empty string and set
  `guidance_scale_2 = guidance_scale` when omitted, so comparison runs must pass the Wan
  recommended values explicitly.

## Current code reality

- `src/mflux/models/common/config/model_config.py` defines `WAN_DEFAULT_NEGATIVE_PROMPT` and the
  Wan default guidance values: TI2V-5B `5.0`, T2V-A14B `4.0/3.0`, I2V-A14B `3.5/3.5`.
- `src/mflux/models/wan/cli/wan_generate.py` applies model defaults for omitted CLI
  `--negative-prompt`, `--guidance`, and `--guidance-2`.
- `src/mflux/models/wan/variants/wan2_2_ti2v.py` resolves omitted runtime guidance, injects the
  default negative prompt, and routes `guidance` to the high-noise transformer and `guidance_2` to
  the low-noise transformer.
- `/Users/albou/projects/gh/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py` and
  `/Users/albou/projects/gh/diffusers/src/diffusers/pipelines/wan/pipeline_wan_i2v.py` use empty
  negative prompts and tie `guidance_scale_2` to `guidance_scale` when callers omit those values.
- Existing planned item [0002](0002_wan_quantization_motion_parity.md) tracks broader Wan
  quantization and motion parity. Existing proposed item
  [0006](../proposed/0006_wan_i2v_prompt_motion_validation.md) remains a narrower I2V follow-up
  if this validation exposes an I2V-specific motion weakness.
- A long full-size Wan generation was active when this item was created, so no model loading or
  generation validation was run during item creation.

## Problem

We cannot yet make a strong claim that Wan prompts influence MLX-Gen T2V and I2V outputs the same
way they influence Diffusers outputs under identical settings. Without that evidence, poor motion
or weak prompt response could be misattributed to user prompts, quantization, defaults, scheduler
behavior, or I2V first-frame conditioning.

## What we want to do

Create a focused validation pass that compares Diffusers and MLX-Gen Wan output under explicitly
matched settings, then records contact sheets, lightweight motion metrics, and conclusions for
T2V and I2V prompt adherence.

## Why

Wan A14B and mixed q8/BF16 packages are expensive to run and publish. Before treating the current
routes as quality-ready, we need evidence that prompt and motion conditioning survive the MLX
implementation and any prepared-model layout.

## Requirements

- Do not run heavy validation while another long generation is active.
- Compare with identical model family, task, seed, prompt, negative prompt, dimensions, frames,
  steps, fps, scheduler shift, `guidance`, and `guidance_2`.
- Pass the official Wan negative prompt and explicit A14B guidance pair into Diffusers; do not rely
  on raw omitted-argument Diffusers defaults.
- For explicit scalar-guidance comparisons such as `--guidance 5`, compare `5/5` unless the user
  deliberately requests a separate low-noise guidance value.
- Cover T2V and I2V. Include A14B routes; include TI2V-5B when it is useful as the cheaper
  baseline.
- Include at least one motion-sensitive prompt and one neutral/static control prompt per task.
- For I2V, use the same source image and aspect-ratio handling in both implementations.
- Record whether each run uses upstream source, prepared BF16, or mixed q8/BF16 weights.
- Preserve generated artifacts and metadata under `validation_outputs/wan/prompt_adherence/`.

## Suggested implementation

1. Build a small validation harness or documented command set that emits each run's effective
   prompt, negative prompt, guidance pair, seed, dimensions, frames, steps, fps, scheduler, and
   model path.
2. Run same-settings Diffusers and MLX-Gen cases after the machine is free for generation work.
3. Extract comparable frames and contact sheets from each MP4.
4. Compute adjacent-frame difference or optical-flow-style summary metrics. Keep metrics cheap and
   reproducible; they are supporting evidence, not the only quality signal.
5. Update model cards, Wan docs, and related backlog conclusions with the exact result.

## Scope

- Validation commands or harness code if needed.
- Contact sheets, metrics, and metadata artifacts.
- Documentation and backlog updates that explain the effective default semantics and observed
  prompt adherence.

## Non-goals

- Do not change Wan prompt, negative-prompt, guidance, scheduler, or I2V conditioning behavior as
  part of this validation item.
- Do not add prompt rewriting or prompt extension.
- Do not publish or republish models from this item alone.
- Do not treat a tiny smoke run as quality evidence for full-size A14B behavior.

## Dependencies and related tasks

- [0002 Wan quantization and motion parity](0002_wan_quantization_motion_parity.md)
- [0005 Wan q8 performance investigation](0005_wan_q8_performance_investigation.md)
- [0013 Wan A14B boundary memory recovery](0013_wan_a14b_boundary_memory_recovery.md)
- [0006 Wan I2V prompt motion validation](../proposed/0006_wan_i2v_prompt_motion_validation.md)
- [0012 Wan2.2 A14B T2V/I2V support](../completed/0012_wan_a14b_t2v_i2v_support.md)

## Expected outcomes

- A clear statement that MLX-Gen prompt adherence matches Diffusers for the tested Wan T2V/I2V
  cases, or a precise mismatch report.
- If a mismatch exists, a narrowed suspect area: prompt embeddings, negative-prompt defaults,
  guidance routing, scheduler/timestep handling, I2V conditioning, VAE decode, or quantized
  weights.
- Updated user-facing docs/model cards that distinguish official Wan recommended defaults from raw
  Diffusers omitted-argument defaults.
- A decision on whether proposed item 0006 should be promoted, revised, or closed as covered by
  this validation.

## Validation

- Use preserved run outputs and metadata, not screenshots alone.
- Store contact sheets and metric JSON under `validation_outputs/wan/prompt_adherence/`.
- Include commands in the completion report with enough detail to reproduce each run.
- Verify generated videos can be opened and frame extraction succeeds.
- Do not run validation until the active long generation has finished or the user provides a safe
  compute window.

## Progress checklist

- [x] Confirm official Wan negative-prompt and guidance defaults from source text.
- [x] Record the raw Diffusers omitted-default distinction.
- [ ] Select prompts, source image, dimensions, frame count, steps, and seeds.
- [ ] Run same-settings Diffusers T2V and MLX-Gen T2V validation.
- [ ] Run same-settings Diffusers I2V and MLX-Gen I2V validation.
- [ ] Extract frames, contact sheets, metrics, and metadata.
- [ ] Update docs/model cards/backlog conclusions.

## Guidance for the implementing agent

Re-check current code before running anything. Keep the first validation batch small enough to be
practical but large enough to show motion and prompt response. If current code or docs disagree
with this item, patch the backlog before spending generation time.
