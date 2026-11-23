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

## Update - 2026-11-22

Since 0.39.0 the AdaLN projections are quantized with the other linears
([item 0128](../completed/0128_minimax_h3_quantized_packages.md)): 12.9 GiB at 8-bit and 9.8 GiB in
the 4-bit package, instead of the 24 GiB of BF16 described above. Precomputing the tables would free
that smaller amount; it would matter most for the 4-bit package on 64 GB machines, where the load
peak is the limit.

## Proposed direction

Direction 1 shipped on 2026-11-23 as item
[0130](../completed/0130_minimax_h3_staged_loading_and_reindexed_8bit.md): the tables are computed
per schedule from disk and the projections are never held; the 16-bit transformer fits (54.4 GiB
peak at `960x544`), and at that canvas 16-bit and 8-bit steps take the same time. Directions 2 and
3 remain.

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
