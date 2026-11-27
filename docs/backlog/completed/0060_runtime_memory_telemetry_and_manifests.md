# Planned: Runtime memory telemetry and manifests

## Metadata
- Created: 2026-06-27
- Status: Completed
- Completed: 2026-06-28
- Reopened: 2026-06-27

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The memory audit found that MLX-Gen mostly reports MLX allocator counters, not whole-process memory.
MLX arrays live in unified memory and MLX active memory does not include cached buffers, so allocator
metrics alone are not a user-facing memory-consumption report.

## Current code reality
- `MemorySaver.memory_stats()` prints only `mx.get_peak_memory()`.
- Wan and SeedVR2 failure diagnostics capture only `mx.get_active_memory`,
  `mx.get_peak_memory`, and `mx.get_cache_memory`.
- `GeneratedImage` and `GeneratedVideo` metadata record parameters and timing, but not memory
  snapshots or phase evidence.

## Problem
The project cannot prove memory reductions, compare low-RAM modes, or guard release notes against
incorrect memory claims without one shared telemetry contract.

## What we want to do
Add a shared memory snapshot utility and attach memory evidence to runtime diagnostics and generated
metadata without making model families surrender their memory policy decisions.

## Why
This is the proof layer for every optimization in this track. It also prevents confusing storage
size, MLX allocator memory, RSS, and macOS physical footprint.

## Requirements
- Capture MLX active, peak, and cache memory when available.
- Capture process RSS and peak RSS without adding a required runtime dependency.
- Capture macOS physical footprint when available, falling back cleanly elsewhere.
- Provide explicit phase labels.
- Avoid heavy sampling in normal generation; use snapshots at natural phase boundaries.

## Suggested implementation
Add `mflux.utils.memory` with `RuntimeMemory` and `MemorySnapshot`. Reuse it from callbacks, Wan,
SeedVR2, and generated metadata helpers.

## Scope
- Shared utility and unit tests.
- Enriched `MemorySaver.memory_stats()`.
- Runtime diagnostics for Wan and SeedVR2.
- Generated image/video metadata memory snapshot at metadata-build time.

## Non-goals
- Do not add `psutil` as a required dependency unless the standard-library approach proves
  insufficient.
- Do not claim memory reductions from schema tests alone.
- Do not synchronize every denoise step by default.

## Dependencies and related tasks
- Completed [0059 Wan VAE streaming](0059_wan_vae_streaming_memory_measurement.md)
- [0062 SeedVR2 chunk-bounded noise](../planned/memory/0062_seedvr2_chunk_bounded_noise.md)

## Expected outcomes
- Runtime metadata contains comparable MLX and process memory fields.
- Failure manifests use the same schema as successful metadata where practical.
- Tests cover serialization and fallback behavior without loading models.

## Validation
- Unit tests for `RuntimeMemory.snapshot()`, metadata serialization, and diagnostics merge points.
- `uv run pytest tests/utils/test_runtime_memory.py tests/metadata/test_generated_image.py tests/metadata/test_generated_video.py`

## Progress checklist
- [x] Add shared memory snapshot utility.
- [x] Wire snapshots into metadata and diagnostics.
- [x] Update memory saver output.
- [x] Add focused tests.

## Implementation notes
Keep MLX metrics and process metrics separate. `mx.clear_cache()` does not free live tensors, and
RSS/physical footprint are the user-facing memory pressure signals.

## Completion report

- Date: 2026-06-27
- Original path: `docs/backlog/planned/memory/0060_runtime_memory_telemetry_and_manifests.md`
- Final path: `docs/backlog/completed/0060_runtime_memory_telemetry_and_manifests.md`
- Summary: Added shared runtime memory snapshots for MLX allocator counters and process-level
  memory evidence, then wired those snapshots into generated metadata, low-RAM callback summaries,
  and video failure manifests.
- Implementation: Added `mflux.utils.runtime_memory.RuntimeMemory` and `RuntimeMemorySnapshot`;
  generated image/video metadata now records memory snapshots; Wan and SeedVR2 failure paths use
  the same schema; `MemorySaver` records before/after-loop snapshots; cache-limit parsing and
  application are centralized.
- Behavior changes: Metadata now separates MLX active/peak/cache memory from process RSS, peak
  RSS, and Darwin physical footprint when those fields are available.
- Validation: Compile and ruff checks passed; focused metadata image/video tests passed with
  pytest cache disabled to avoid local native test-runner crashes.
- Residual risk: Snapshot-at-boundary telemetry is proof scaffolding, not continuous sampling. Long
  production runs still need explicit process monitoring when publishing memory numbers.

## Reopen report

- Date: 2026-06-27
- Reason: Point-in-time snapshots and metadata fields are not enough to prove peak memory
  reductions. The telemetry layer needs a quantitative validation artifact that compares runtime
  snapshots with externally sampled process peaks.
- Required evidence before closure: a process-isolated memory benchmark artifact showing peak RSS
  and Darwin physical-footprint sampling, MLX peak/cache values, agreement limits, schema fields,
  and overhead from telemetry collection.

## Quantitative validation update

- Date: 2026-06-27
- Evidence artifacts:
  `validation_outputs/memory/real_generation_20260627_wan_r2/generation_memory_benchmark.json`,
  `validation_outputs/memory/real_generation_20260627_zimage_r3/generation_memory_benchmark.json`,
  `validation_outputs/memory/real_generation_20260627_seedvr2_r3/generation_memory_benchmark.json`,
  `validation_outputs/memory/real_generation_20260627_seedvr2_image_1280_r2/generation_memory_benchmark.json`,
  and
  `validation_outputs/memory/real_generation_20260627_seedvr2_image_1280_tiling_r2/generation_memory_benchmark.json`.
- Result: The benchmark harness now runs each variant in a fresh CLI process, samples process-tree
  RSS externally, parses `/usr/bin/time -l`, loads generated metadata, and records MLX
  active/peak/cache, process RSS/peak RSS, and Darwin physical footprint from runtime snapshots.
- Status: Still planned. The artifacts validate schema and sampling usefulness, but this item still
  needs an explicit telemetry-overhead comparison and documented agreement limits before closure.
  The SeedVR2 1280px image reports also show why both MLX peak and process metrics must remain in
  the schema: process RSS was nearly flat while MLX peak changed by up to 46.24% under explicit
  VAE tiling.

## Quantitative completion report

- Date: 2026-06-28
- Evidence artifact:
  `validation_outputs/memory/real_generation_20260628_0060_zimage_runtime_telemetry_physical/generation_memory_benchmark.json`.
- Real profile: `mflux-generate-z-image-turbo`, `AbstractFramework/z-image-turbo-8bit`, 384x384,
  2 steps, seed 5151, seven fresh CLI runs per variant. Parent process-tree RSS and Darwin
  physical-footprint sampling were enabled with
  `MFLUX_BENCHMARK_PARENT_PHYSICAL_SAMPLING=1`.
- Result: enabling runtime memory telemetry preserved exact pixels (`mae=0`, `rmse=0`,
  `max_abs=0`) while median sampled RSS moved from 11.4996 GB to 11.4998 GB (+0.0019%),
  median sampled Darwin physical footprint moved from 17.9410 GB to 18.0014 GB (+0.3370%),
  and median wall time moved from 3.0331 s to 3.0659 s (+1.0814%).
- Agreement evidence: telemetry-enabled metadata reported median Darwin physical footprint
  17.9856 GB against parent sampled median 18.0014 GB, and median process RSS 11.4567 GB against
  parent sampled process-tree RSS 11.4998 GB. The disabled variant retained the same metadata
  schema with telemetry fields intentionally null and exact output parity.
- Implementation hardening: the benchmark parent sampler now queries only descendant PIDs and uses
  an out-of-process Darwin `proc_pid_rusage` helper, avoiding the previous native instability from
  direct in-process physical-footprint sampling.
- Final status: Completed. The telemetry layer now has process-isolated overhead statistics,
  schema evidence, parent physical-footprint sampling, and agreement evidence.
