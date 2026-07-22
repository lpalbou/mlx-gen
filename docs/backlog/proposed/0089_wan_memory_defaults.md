# Proposed: Wan memory defaults — streamed decode and per-item transformer release

## Metadata

- Created: 2026-07-22
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: None (defaults change with identical outputs; record in CHANGELOG)

## Context

2026-07-22 audit findings in 0.23.1:
- The default (non low-ram) Wan decode materializes the full video tensor in one
  array (`wan2_2_ti2v.py` ~486-492) — ~650 MB bf16 for 121f@1280x704 plus latents —
  while the streamed slice decoder (`wan_2_2_vae.py` ~312-347) is gated behind
  `--low-ram` AND single-seed (`wan_generate.py` ~66). PIL conversion is already
  per-8-frame batch (`video_util.py` ~1226-1235), so streaming by default should
  cost nothing.
- Multi-seed A14B batches keep BOTH 14 GB transformers resident for the whole run:
  `release_inactive_denoiser = single_seed and has_transformer_2`
  (`wan_generate.py` ~62-65) — the high-noise transformer is never released after
  its phase when `--seed a b c` is used.

## Problem or opportunity

~1 GB avoidable peak on every default Wan run; ~14 GB avoidable residency on batched
A14B runs — the difference between fitting and swapping on 36-48 GB machines.

## Proposed direction

- Make streamed VAE decode the default whenever the writer consumes frames in
  batches; keep the all-at-once path only where a consumer truly needs the full
  tensor.
- Release the inactive A14B transformer per batch item and reload from the OS page
  cache (measure the reload cost — page-cache hits should make it cheap); or key the
  behavior on a memory budget.

## Why it might matter

Same outputs, lower peaks, especially for the machines most likely to OOM.

## Promotion criteria

Promote when Wan A14B batch usage is real (embedding hosts expose multi-video
batches today) or when a smaller-RAM user reports pressure.

## Validation ideas

Peak-RSS measurement harness before/after (whole-process, per AGENTS.md); bitwise or
threshold-identical outputs; reload-cost timing for the per-item release.

## Non-goals

Changing low-ram semantics; touching image-family decode paths.

## Guidance for future agents

Follow AGENTS.md: physical process memory for user-facing claims; distinguish
storage vs runtime improvements.
