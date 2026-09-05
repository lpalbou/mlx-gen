# Proposed: MiniMax-H3 AdaLN precompute-and-drop and long-sequence attention performance

## Metadata

- Created: 2026-09-04
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: none.
- ADR impact: none.

## Context

The q8 MiniMax-H3 transformer keeps its AdaLN table projections in bf16 (26 GB of the ~40 GB
resident transformer) because q8 there destroys the output. The tables depend only on the timestep
embedding, so for a known schedule every table can be precomputed (50 blocks x 9 steps x a few
hundred KB) and the projection weights dropped from memory. Separately, a 768p clip packs 37.8k
rows and the fused attention kernel falls from ~48 TFLOPS at 15.5k rows to ~17 TFLOPS there,
making a 768p step ~5 minutes (8-step Turbo ~40 minutes).

## Proposed direction

1. Precompute the AdaLN tables per generation and release the projection weights (reload from the
   shards for a schedule change); measure the memory win and whether a bf16 transformer then fits
   with the q8 conditioner on 128 GB (bf16 GEMMs were ~1.4x faster than q8 at this size).
2. Measure head-grouped attention (~15% faster at 37.8k rows in a microbenchmark) and fp16
   attention inputs inside the block.
3. Apply the per-segment modulation (text, audio, video rows are contiguous) instead of the per-row
   table gather.

## Validation

Same-seed clips before and after each change with contact sheets; per-step timing at `960x544`
and `1344x768`; peak memory.
