# Planned: Wan q8 performance investigation

## Metadata

- Created: 2026-05-27
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: None. This is a performance investigation unless it results in a durable
  quantization/runtime policy.

## Context

Wan2.2 TI2V q8 prepare and reload now work, but the first same-settings smoke comparison was
slower than the source BF16 snapshot. The comparison used 704x384, 25 frames, 12 denoising steps,
24 fps, guidance 5.0, seed 321, and the same prompt.

## Current code reality

- The source BF16 run took 95.48 seconds.
- The prepared q8 run took 217.4 seconds.
- Runtime peak memory was not captured in metadata.
- The prepared q8 folder is about 17 GiB: transformer 5.0 GiB, text encoder 11 GiB, VAE 1.3 GiB,
  tokenizer 16 MiB.
- The source Hugging Face snapshot is about 32 GiB when following symlinks: transformer 19 GiB,
  text encoder 11 GiB, VAE 2.6 GiB.
- Wan q8 currently proves smaller disk/model footprint, not speed improvement.
- T2V-A14B mixed q8 validation on 2026-06-02 first captured peak RSS with `/usr/bin/time -l`, but
  that metric was incomplete for MLX/Metal unified memory. A follow-up package-level benchmark now
  records MLX allocator peak plus Darwin physical footprint from model init through video save.
  In the same 384x224, 17-frame, 12-step benchmark:
  - upstream source with inactive-denoiser release: 32.99 GiB MLX peak, 48.90 GiB physical
    footprint, 108.31s;
  - BF16 prepared with inactive-denoiser release: 32.98 GiB MLX peak, 45.12 GiB physical
    footprint, 114.39s, byte-identical to source;
  - mixed q8/BF16 prepared with inactive-denoiser release: 20.84 GiB MLX peak, 31.75 GiB physical
    footprint, 110.34s;
  - mixed q8/BF16 prepared with low-RAM release: 15.48 GiB MLX peak, 20.74 GiB physical footprint,
    108.70s.
- The prepared q8 folder reduces storage from a 118 GiB source snapshot to 40 GiB and does reduce
  package-level usage memory. It does not prove a speed improvement; measured times remain within
  short-run variance around BF16.
- The same A14B run proved full q8 is a quality failure rather than a performance candidate:
  full q8 generated in 111.44s at 13.72 GiB peak RSS but produced near-black/static output. The
  current Wan q8 policy keeps `condition_embedder.*` and `proj_out` BF16.
- A 2026-06-02 component-streaming Wan load/apply experiment made the prepared mixed-q8 output
  byte-identical to the previous prepared mixed-q8 MP4, but did not materially lower peak RSS:
  103.84s at 19.58 GiB versus the earlier 110.50s at 19.51 GiB. Treat the time delta as
  run-to-run/cache variance unless repeated benchmarks prove otherwise.
- Video frame conversion now builds PIL frames one frame at a time instead of materializing
  full-video float32 and uint8 NumPy arrays. The 17-frame A14B q8 sample remained byte-identical
  and measured 103.84s at 19.59 GiB peak RSS. The expected benefit scales with output resolution
  and frame count, so the tiny 384x224 sample is not large enough to show a meaningful RSS change.
- On 2026-06-03, the full-size 1280x720, 81-frame, 40-step T2V-A14B mixed q8/BF16 run took about
  13h15m and produced an invalid all-black MP4 from non-finite decoded values. Treat that as a
  numerical-integrity blocker before interpreting full-size q8 speed or memory results.

## Problem

q8 being substantially slower than BF16 for Wan is suspicious. It may be caused by unsupported or
slow quantized kernels for the Wan tensor shapes, repeated dequantization, compile/cache behavior,
text encoder/VAE effects dominating the short run, memory pressure, or an avoidable MLX-Gen
loading/runtime issue.

## What we want to do

Identify why Wan q8 is slower, decide whether it can be optimized, and make future Wan benchmark
claims capture both speed and peak memory.

## Why

Wan generations are expensive. If q8 only reduces disk/RAM footprint, users and model cards should
say that. If the slowdown is a bug or avoidable runtime path, fixing it would materially improve
local video generation.

## Requirements

- Measure peak memory for source BF16 and q8 runs using either internal MLX peak memory or
  `/usr/bin/time -l`.
- Split timing into text encoding, denoising transformer loop, VAE decode, and MP4 encoding where
  feasible.
- Benchmark at least one tiny smoke setting and one meaningful short setting with the same prompt,
  seed, width, height, frames, steps, fps, and guidance.
- Inspect which Wan modules are quantized and whether the hot path uses `mx.quantized_matmul` or
  expensive fallback/dequantization.
- Compare q8 performance with and without cache limits or low-RAM options if relevant.

## Suggested implementation

1. Add a benchmark harness or validation script under `validation_outputs/wan/` or `scripts/`
   that wraps `mlxgen generate` with timing and peak-memory capture.
2. Add optional internal timing hooks around Wan prompt encoding, transformer denoise, VAE decode,
   and video save.
3. Inspect MLX profiler or targeted microbenchmarks for the largest Wan linear layers in BF16
   versus q8.
4. If q8 kernels are intrinsically slower for these shapes, document q8 as memory-footprint
   focused.
5. If MLX-Gen is doing avoidable work, patch the runtime and update the Wan validation results.

## Scope

- Wan2.2 TI2V 5B and A14B q8 runtime performance and memory behavior.
- Benchmarking and possible low-risk runtime optimizations.

## Non-goals

- Do not publish Wan q4/q8 speed claims until this is resolved.
- Do not delete source or prepared model folders to make room without explicit user approval.
- Do not change global quantization policy based on Wan alone.

## Dependencies and related tasks

- [Wan quantization and motion parity](0002_wan_quantization_motion_parity.md)
- `src/mflux/models/wan/`
- `src/mflux/models/common/weights/loading/weight_applier.py`
- `src/mflux/models/wan/weights/wan_weight_definition.py`
- `validation_outputs/wan/user_video_analysis/`
- `validation_outputs/wan/a14b_q8_t2v/t2v_a14b_mixed_q8_validation_report.json`

## Expected outcomes

- A clear explanation for why q8 is slower or a patch that makes q8 performance reasonable.
- Future Wan validation reports include peak memory and phase timings, not only generation time.
- Full-size q8 timing is only meaningful when the output passes video-health validation.
- Model cards and docs accurately describe q8 as speed-improving, memory-improving, or both.

## Validation

- Re-run source BF16 and prepared q8 at the same settings with peak-memory capture.
- Record phase timings or a profiler summary.
- Preserve the commands and results under `validation_outputs/wan/` or docs.

## Progress checklist

- [ ] Add a repeatable Wan benchmark command/harness with memory capture.
- [x] Measure BF16/source and q8 prepared folder at identical A14B T2V short settings.
- [x] Measure source, BF16 prepared, and mixed q8/BF16 prepared with MLX peak allocator memory.
- [ ] Split or profile timing by generation phase.
- [x] Inspect prepared-folder load/runtime memory behavior enough to rule out component bundling
  as the source of the 19.5 GiB prepared-q8 peak.
- [ ] Inspect quantized Wan hot-path kernels.
- [x] Update docs/model-card guidance based on A14B T2V quality and timing evidence.

## Guidance for the implementing agent

Start with measurement. Do not assume q8 should be faster until the hot path is profiled, but treat
the current 2.3x slowdown as suspicious enough to investigate seriously.
