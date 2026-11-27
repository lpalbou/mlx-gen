# Planned: Generation retention cleanup

## Metadata
- Created: 2026-06-27
- Status: Planned
- Completed: N/A
- Reopened: 2026-06-27

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The audit found several smaller but real retention issues that do not require changing model math:
duplicate conditioning concatenation, unnecessary hidden-state history, unbounded caches, and
stepwise output retaining every generated image.

## Current code reality
- Qwen edit builds the same latent/static-image concatenation twice per denoise step.
- Flux2 Qwen3 prompt encoding stores every hidden state before selecting three layers.
- ERNIE text encoding stores all hidden states while returning only the second-last.
- Prompt/control caches retain MLX arrays for the model lifetime.
- `StepwiseHandler` stores every `GeneratedImage` and rebuilds a composite every step.

## Problem
These patterns increase peak or retained memory without improving generation quality.

## What we want to do
Apply low-risk retention fixes now and preserve larger cache-policy changes as explicit follow-ups
if they need UX decisions.

## Why
These changes reduce memory pressure while keeping performance and output quality mostly unchanged.

## Requirements
- Preserve generated pixels for equivalent seeds where math is unchanged.
- Avoid mutating transformer inputs.
- Keep stepwise output behavior unless a safer disk-backed alternative is added.
- Do not silently drop caches that users rely on for repeat generation.

## Suggested implementation
Reuse duplicate per-step tensors, collect only needed hidden states, detach/evaluate cached prompt
outputs, and avoid retaining stepwise `GeneratedImage` objects when a PIL image is enough for the
composite.

## Scope
- Qwen edit denoise loop.
- Flux2 and ERNIE hidden-state collection.
- Stepwise handler host-retention cleanup.
- Proposed cache LRU only if it can be done without changing public behavior.

## Non-goals
- Do not disable prompt caches globally.
- Do not remove debug stepwise output.
- Do not change image/video quality for memory savings.

## Dependencies and related tasks
- Completed [0061 prompt materialization](../../completed/0061_prompt_materialization_for_low_ram_release.md)

## Expected outcomes
- Fewer duplicate live tensors in Qwen edit CFG.
- Prompt encoders retain fewer hidden layers.
- Stepwise output retains less host-side object graph.

## Validation
- Focused unit tests where possible plus existing fast test suite.
- Manual memory profiling remains required for quantitative claims.

## Progress checklist
- [x] Reuse Qwen edit hidden-state concat.
- [x] Reduce Flux2/ERNIE hidden-state retention.
- [x] Reduce StepwiseHandler retained objects.
- [x] Add focused tests.

## Implementation notes
Prefer surgical changes. Broad cache eviction policy should be proposed separately if it creates
repeat-generation UX tradeoffs.

## Completion report

- Date: 2026-06-27
- Original path: `docs/backlog/planned/memory/0064_generation_retention_cleanup.md`
- Final path: `docs/backlog/completed/0064_generation_retention_cleanup.md`
- Summary: Removed several avoidable retention points that increased live tensors or host object
  graphs without contributing to generation quality.
- Implementation: Qwen edit no longer builds the duplicate negative hidden-state concat; Flux2
  Qwen3 and ERNIE text encoders retain only required hidden states; stepwise output keeps copied
  PIL images instead of full `GeneratedImage` objects; image/video helpers avoid some native-array
  conversion paths that were unstable in local focused tests.
- Behavior changes: Stepwise visual output remains available, but the retained object graph is
  smaller. Prompt hidden-state values are still produced at the same semantic layers.
- Validation: Compile and ruff checks passed; focused metadata image/video tests and route tests
  passed with pytest cache/warnings disabled where needed for local native stability.
- Residual risk: Broader cache eviction policy remains intentionally unimplemented because it has
  repeat-generation UX tradeoffs and needs a separate design decision.

## Reopen report

- Date: 2026-06-27
- Reason: Retention cleanup changed host/object residency, but closure requires quantitative memory
  statistics on stepwise/debug and hidden-state-retention paths plus quality/performance checks.
- Required evidence before closure: baseline-versus-candidate runs reporting peak process memory,
  MLX peak, wall time, retained-object mode, and output equivalence for math-preserving changes.

## Quantitative validation update

- Date: 2026-06-27
- Status: Still planned. No clean real benchmark isolates stepwise retention or hidden-state
  retention yet.
- Required next run: add a focused profile that compares stepwise output before/after retained
  object cleanup and, separately, an affected prompt hidden-state path. The profile must report
  peak process RSS, MLX peak/cache, wall time, output equality for math-preserving changes, and
  whether retained objects are PIL-only or full `GeneratedImage` instances.

## Quantitative status update

- Date: 2026-06-28
- Evidence artifact:
  `validation_outputs/memory/real_generation_20260628_0064_retention_r2/generation_memory_benchmark.json`.
- Real profiles: `flux2-stepwise-retention` compared PIL-only stepwise retention against legacy
  full `GeneratedImage` retention; `zimage-hidden-state-retention` compared previous-hidden-state
  retention against legacy all-hidden-state retention. Both profiles used real image generation,
  three fresh CLI runs per variant, and exact output comparison.
- Result: both comparisons preserved exact pixels (`mae=0`, `rmse=0`, `max_abs=0`), but neither
  proved a memory reduction. Flux2 stepwise PIL retention measured 8.2089 GB sampled RSS versus
  8.1957 GB for the legacy retained-object path, with identical 10.4327 GB MLX peak and faster
  wall time. Z-Image previous-hidden-state retention measured 11.5245 GB sampled RSS versus
  11.5123 GB legacy, with identical 12.6816 GB MLX peak and essentially flat wall time.
- Production hardening: the hidden-state and stepwise legacy modes used for benchmarks are now
  gated behind explicit internal benchmark mode rather than standalone production-visible
  `MFLUX_BENCHMARK_*` toggles.
- Status: Still planned. Keep the code cleanup because it reduces object graph risk and preserves
  output, but do not claim completed memory improvement for this item until a profile shows a
  quantitative reduction or the item is explicitly reframed away from memory-reduction closure.
