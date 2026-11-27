# Planned: Prompt materialization for low-RAM release

## Metadata
- Created: 2026-06-27
- Status: Completed
- Completed: 2026-06-28
- Reopened: 2026-06-27

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
Generic image low-RAM mode deletes text encoders after prompt encoding. MLX lazy evaluation can keep
encoder graphs alive unless prompt outputs are detached and evaluated before encoder release.

## Current code reality
- `MemorySaver.call_before_loop()` deletes encoder attributes after prompt encoding.
- Qwen prompt encoding already uses `mx.stop_gradient()` and `mx.eval()`.
- Flux, Flux2, and ERNIE prompt paths do not consistently materialize prompt outputs before the
  low-RAM callback releases encoders.
- Z-Image already materializes its prompt encodings.

## Problem
Low-RAM mode can still carry prompt-encoder memory into denoising for families whose prompt outputs
are lazy graph nodes.

## What we want to do
Materialize prompt tensors at the end of prompt encoding for affected families, without changing the
embedding values used by generation.

## Why
This reduces peak memory at the transition from prompt encoding to denoising while preserving model
quality and scheduler behavior.

## Requirements
- Use `mx.stop_gradient()` for inference prompt tensors.
- Force `mx.eval()` before caching prompt outputs.
- Preserve prompt cache semantics.
- Do not remove per-step `mx.eval(latents)`.

## Suggested implementation
Add or reuse a shared materialization helper and update Flux, Flux2, and ERNIE prompt encoding. For
Flux2 and ERNIE, also avoid storing unnecessary hidden-state history where the output only needs a
small subset.

## Scope
- Prompt encoders for Flux, Flux2, ERNIE, and any directly equivalent path found during work.
- Focused tests for prompt cache materialization and shape parity.

## Non-goals
- Do not alter tokenization, prompt templates, negative-prompt handling, or generated prompt text.
- Do not rewrite FIBO prompt-layer behavior; it intentionally feeds many hidden layers.

## Dependencies and related tasks
- [0060 runtime memory telemetry](0060_runtime_memory_telemetry_and_manifests.md)
- Completed [0055 low-RAM repeat-generation safety](0055_low_ram_repeat_generation_safety.md)

## Expected outcomes
- Low-RAM encoder release is not defeated by lazy prompt graphs.
- Prompt caches hold detached/evaluated tensors.
- Prompt output shapes and values remain compatible with current generation code.

## Validation
- Focused prompt-encoder unit tests.
- Existing image-generation fast tests continue to pass.

## Progress checklist
- [x] Add shared materialization helper.
- [x] Update Flux prompt encoder.
- [x] Update Flux2 prompt encoder/text encoder hidden-state collection.
- [x] Update ERNIE prompt path.
- [x] Add focused tests.

## Implementation notes
This should be a no-quality-change optimization. If a proposed materialization changes output
values, revert the materialization point and document the family-specific reason.

## Completion report

- Date: 2026-06-27
- Original path: `docs/backlog/planned/memory/0061_prompt_materialization_for_low_ram_release.md`
- Final path: `docs/backlog/completed/0061_prompt_materialization_for_low_ram_release.md`
- Summary: Materialized inference prompt outputs before low-RAM encoder release so lazy MLX graphs
  do not accidentally keep prompt encoders alive into denoising.
- Implementation: Added shared inference-tree materialization in `RuntimeMemory` and applied it to
  Flux, Flux2, FIBO, Qwen/VLM, Qwen edit, and ERNIE prompt outputs. Flux2 Qwen3 and ERNIE text
  encoders now retain only the hidden states they actually return.
- Behavior changes: Prompt values and shapes are intended to stay unchanged; the change only forces
  inference arrays to be evaluated/detached at the family prompt boundary.
- Validation: Compile and ruff checks passed; focused CLI, metadata, SeedVR2, and Wan tests passed.
- Residual risk: Direct model-backed pixel parity was not run for every image family because the
  requested scope was memory hardening and the repo guidance avoids long or costly generation
  without approval.

## Reopen report

- Date: 2026-06-27
- Reason: The item changed prompt/conditioning materialization but did not quantify prompt-to-denoise
  peak memory reduction on real or representative model-family runs.
- Required evidence before closure: baseline-versus-candidate runs for at least one affected family
  with peak process memory, MLX peak, wall time, and output parity or bounded drift.

## Quantitative validation update

- Date: 2026-06-27
- Status: Still planned. The real benchmark harness exists, but no clean prompt-materialization
  baseline-versus-candidate profile has been run yet.
- Required next run: add a focused affected-family profile, preferably Flux2 or ERNIE, that isolates
  low-RAM prompt materialization from unrelated stepwise or cache-policy changes and compares peak
  process RSS, MLX peak/cache, wall time, and image parity for a fixed seed.

## Quantitative completion report

- Date: 2026-06-28
- Evidence artifact:
  `validation_outputs/memory/real_generation_20260628_0061_prompt_materialization_r3/generation_memory_benchmark.json`.
- Real profiles: `mflux-generate-flux2` with `AbstractFramework/flux2-klein-4bit` and
  `mflux-generate-ernie-image` with `AbstractFramework/ernie-image-turbo-8bit`, both low-RAM,
  fixed prompts, fixed seeds, three fresh CLI runs per variant. The benchmark compared current
  materialized prompt outputs against an internal legacy lazy-materialization mode.
- ERNIE result: exact image parity (`mae=0`, `rmse=0`, `max_abs=0`). Median sampled RSS fell from
  9.9793 GB legacy to 9.2561 GB materialized (-0.7232 GB, -7.25% when expressed as legacy to
  materialized). Median MLX peak stayed effectively flat at 9.9201 GB legacy versus 9.9188 GB
  materialized. Median wall time moved from 3.4928 s legacy to 3.5758 s materialized.
- Flux2 result: exact image parity (`mae=0`, `rmse=0`, `max_abs=0`). Median sampled RSS fell from
  4.9203 GB legacy to 4.8135 GB materialized (-0.1068 GB, -2.17% when expressed as legacy to
  materialized). Median MLX peak stayed flat at 6.2741 GB legacy versus 6.2740 GB materialized.
  Median wall time moved from 2.9103 s legacy to 2.9982 s materialized.
- Interpretation: The proof supports a peak RSS reduction claim for the prompt-materialization
  boundary on ERNIE and Flux2 with exact output parity. Metadata physical-footprint snapshots are
  mixed at the metadata boundary, so public wording must not claim a universal physical-footprint
  reduction from this item alone.
- Implementation hardening: legacy benchmark behavior is now gated behind
  `MFLUX_INTERNAL_MEMORY_BENCHMARK_MODE=1` plus explicit internal flags instead of standalone
  production-visible `MFLUX_BENCHMARK_*` toggles.
- Final status: Completed. The low-RAM prompt materialization change has real process-isolated
  memory statistics and quality parity on affected model families.
