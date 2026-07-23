# Proposed: Step-loop performance experiments (batched CFG, Wan compile, step caching)

## Metadata

- Created: 2026-07-22
- Status: Partially implemented (Wan compile shipped behind a flag 2026-07-22;
  batched CFG and step caching stay deferred)
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: None while experimental; a default change would need CHANGELOG +
  model-card updates.

## Context

2026-07-22 audit of 0.23.1 step loops:
- Wan and Qwen run sequential CFG — two full transformer forwards per step
  (`wan2_2_ti2v.py` ~289-336; `qwen_image.py` ~167-181) — while Flux2 and Z-Image
  fold CFG into one compiled predict (`flux2_klein.py` ~280-303, `z_image.py` ~269).
- No `mx.compile` anywhere under `models/wan/` (the 40-block loop pays per-step
  Python + kernel-launch overhead).
- No step-output caching (TeaCache-style skip/reuse) exists for the 40-50 step
  non-distilled paths.

## Problem or opportunity

Potential 10-25% (batched CFG, at 2x activation memory), mid-single-digit to ~15%
(compile), and larger-but-quality-risky (step caching) wins for non-Lightning runs.
All irrelevant for 4-step distilled runs — measure before committing.

## Proposed direction

Three isolated, measured experiments behind flags:
1. Batched cond+uncond CFG for Wan/Qwen (opt-in; auto-off under memory budget).
2. `mx.compile` the Wan per-block or per-step forward (shape-stable within a run).
3. Prototype step caching for >=30-step runs with a strict quality gate (SSIM/LPIPS
   against uncached goldens) before any default consideration.

## Implementation record (2026-07-22): Wan compile behind a flag (review item d12)

Experiment 2 shipped as OPT-IN only: `--compile-transformer` (CLI) /
`compile_transformer: bool = False` (`generate_video` kwarg). Honest expected
gain ~2-6%/step; output is NON-bitwise vs eager (compiled kernel fusion differs
by ~5e-4 on real weights), so it must NEVER become a silent default.

Why Wan was not compiled before: the forward threads `clear_cache_each_block`
(mx.eval+clear mid-graph) and a per-step `WanBlockHealthContext` (frozen
dataclass with a changing step field -> hashable non-array arg -> retrace every
step), and env-gated block-health `.item()` checks cannot trace. Shapes are
constant within a run (probe-verified: no retrace across changing timestep
VALUES).

Shipped design: eligibility inside `generate_video` = flag AND
`tensor_health_check_interval is None` AND
`not clear_cache_each_transformer_block` AND block-health env disabled;
otherwise the run stays eager and prints ONE notice naming the blockers
(documented mode choice, not a silent fallback). When eligible, one compiled
callable per expert (two for A14B) closes over `block_health_context=None`,
built once per `generate_video` call; the compiled entry for the high-noise
expert is dropped before `release_inactive_denoiser` frees it so the closure
does not retain the weights. Metadata extra records `compile_transformer: true`
on compiled runs. Parity, eligibility-notice, and CLI wiring are test-pinned.

Real-checkpoint validation (owner): fixed-seed A/B wall-time on TI2V-5B and
A14B per the validation ideas below - the ~2-6% figure comes from the review's
probe, not from a full-quality run in this change.

Experiments 1 (batched CFG, review d11) and 3 (step caching, review d14) stay
deferred; their rationale above is unchanged.

## Implementation record (2026-07-23): RoPE per-shape cache (review item F7)

`WanRotaryPosEmbed.__call__` rebuilt the rotary embeds on every forward (~33 ms
at A14B-121f token counts, ~11 ms TI2V, x2 with CFG) although the token grid is
constant within a run. The embeds now cache per `(frames, height, width)` token
grid on the embed instance in `_freqs_cache` (underscore prefix keeps the dict
out of nn.Module parameter/quantization traversal, pinned via `tree_flatten`).
Entries are `mx.eval`-materialized so hits return ready tensors; MLX arrays are
immutable and consumers only slice (`wan_attention` `0::2`/`1::2`), so sharing
is seed-safe. Cached-vs-fresh embeds are bitwise identical (test-pinned), and a
compiled-fn probe confirmed eager/compiled parity across shape changes (the
cache populates at trace time; replays never re-enter Python).

Memory accounting (cycle-3 adversarial review): one grid pair is ~114 MB f32 at
A14B 121f@1280x720 (111,600 tokens x 128 dims x cos+sin), and A14B holds one
embed instance PER expert, so the per-run working set is ~229 MB that was
previously transient. The eviction bound was reduced from 4 to 2 grids in that
review: one run uses one grid, two cover alternating host presets, and a third
resolution merely recomputes (~33 ms), so idle retention stays ~457 MB worst
case across both experts instead of ~914 MB. Eviction is FIFO by insertion
(hits do not refresh) - adequate at this size and access pattern.

## Why it might matter

The quality-mode (non-Lightning) paths are where users still wait minutes; these are
the standard levers.

## Promotion criteria

Promote each experiment separately once a reproducible benchmark harness exists and
an embedding host or CLI user demonstrably runs quality-mode workloads.

## Validation ideas

Per-experiment: fixed-seed A/B wall-time + peak-RSS + golden-image comparison, on
both TI2V-5B and A14B, 480p and 720p.

## Non-goals

Changing any default without measurement; touching distilled/Lightning paths.

## Guidance for future agents

Do not stack experiments in one measurement run; attribute wins individually.

## Deferral blueprints (2026-07-22 adversarial prioritization — preserved so
## a future implementer starts from evidence, not scratch)

**Batched CFG (d11), deferred.** Probe-verified on a tiny random-weight
WanTransformer: batch=2 [cond;uncond] vs two sequential forwards diverges
max ~9e-4 (scalar-timestep A14B convention; 0.0 on expanded 5B) — works
mechanically, NOT bitwise, must never be a silent default. RoPE broadcasts
fine; Wan has no attention masks; the A14B expert boundary is per-timestep,
so batching within one step never crosses experts. Adversarial arithmetic at
storyboard sizes (A14B, 832x480x49f, 20,280 tokens): per-linear activation
traffic ~208 MB vs ~26 MB weight read — batching halves only the weight term
=> ~3-8%, diluted further by attention FLOPs; Qwen-Image (~4k tokens) could
see 10-20%. Zero value on the Lightning guidance-1.0 default (no CFG runs at
all). Extra peak ~0.8-1.5 GB at that size. VACE needs control_hidden_states
duplication (out of scope v1); Qwen needs cond/uncond padding + per-row
txt_seq_lens handling. Test blueprint: extend the tiny-transformer parity
test (max|batched-sequential| < 5e-3, both timestep conventions) + a fake-
model harness test asserting one batch-2 transformer call per step.
Un-defer evidence: a reproducible >=10% end-to-end benchmark on a real
guidance>1 workload someone actually runs.

**Step caching / TeaCache-style (d14), deferred.** Two model-specific
hazards beyond the generic quality risk: (a) A14B needs PER-EXPERT caches
with a hard reset at the boundary (boundary_ratio 0.875/0.9) — the
near-boundary steps are where the output distribution shifts fastest, the
WORST place to reuse; (b) the default solver is UniPC order-2 whose
corrector consumes model_outputs history — a reused stale output enters the
corrector twice, compounding error in ways Euler-based TeaCache results do
not cover. Blueprint if pursued: cache the residual (output-input) keyed on
relative L1 change of the modulated timestep embedding, per expert,
euler-solver-first, >=30 steps only. Un-defer evidence: a demonstrated
non-distilled 40+ step workload AND a golden SSIM/LPIPS benchmark harness.
