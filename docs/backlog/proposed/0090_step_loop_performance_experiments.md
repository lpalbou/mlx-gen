# Proposed: Step-loop performance experiments (batched CFG, Wan compile, step caching)

## Metadata

- Created: 2026-07-22
- Status: Proposed
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
