# Planned: Wan, VLM, and BF16 performance hardening

## Metadata
- Created: 2026-06-27
- Status: Completed
- Completed: 2026-06-27

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The audit found concrete performance and memory issues in video/VLM-heavy paths, plus a precision
policy question for Torch-backed BF16 weights.

## Current code reality
- Wan video generation decodes the full latent video and converts it into a full frame list before
  saving.
- `VideoUtil` already has streaming writer utilities used by SeedVR2 video restore, but Wan does
  not use a streaming save path.
- Qwen3/FIBO VLM sampling converts probabilities to NumPy and sorts on CPU each token.
- Qwen3 VLM image-token handling uses scalar host logic and Python loops over MLX arrays.
- Torch-backed BF16 loading currently converts through FP16 before creating MLX arrays.

## Problem
Large video and VLM workflows can incur avoidable memory spikes and host synchronization overhead.
BF16 narrowing may be correct for some paths, but today it is silent and not validated per
component.

## What we want to do
Stream or batch Wan video output where possible, keep VLM sampling/token replacement on MLX, and
make BF16 precision policy explicit.

## Why
These paths are the likely pressure points for Apple Silicon memory and latency. Silent precision
changes also make parity debugging harder.

## Requirements
- Add a Wan save path that does not require retaining all PIL frames when possible.
- Benchmark or at least micro-test VLM sampling changes before claiming speedups.
- Preserve BF16 into MLX where supported, or explicitly document and validate FP16 narrowing for
  affected components.
- Keep existing video-health and metadata behavior.

## Suggested implementation
Split the work into separate patches: Wan streaming/batched save, VLM sampling vectorization, VLM
image-token vectorization, and BF16 loader policy. Start with the lowest-risk isolated path.

## Scope
- Wan video output memory behavior.
- Qwen3/FIBO VLM utility hot paths.
- Torch-backed BF16 conversion policy.

## Non-goals
- Do not rewrite the Wan denoising loop in this item.
- Do not claim measured performance improvements without local timing evidence.

## Dependencies and related tasks
- Related: `0005_wan_q8_performance_investigation.md`
- Related: `0051_safe_checkpoint_loading.md`

## Expected outcomes
- Wan save memory use is bounded better than the current full-frame-list path.
- VLM token loops avoid unnecessary host synchronization.
- BF16 conversion is either preserved or explicitly justified.

## Validation
- Focused unit tests for video save metadata and frame counts.
- Focused VLM utility tests for sampling/token placement equivalence.
- A recorded local timing or memory note before any release claim.

## Progress update: 2026-06-27
- Added a lazy `GeneratedVideo` mode and `VideoUtil.to_video(..., materialize_frames=False)` so
  Wan can save decoded latents through frame batches without retaining the full PIL frame list.
- Switched Wan video generation to the lazy video path.
- Moved Qwen3 VLM top-p sampling from NumPy sorting/probability sampling to MLX array operations
  and MLX categorical sampling.
- Vectorized Qwen3 VLM and Qwen image-token embedding replacement, removing per-token Python list
  stacking in those hot paths.
- Preserved Torch BF16 tensors as MLX BF16 for checkpoint and Torch-backed safetensors loading.
- Added focused unit coverage for lazy video save routing, VLM sampling/token placement, and BF16
  dtype preservation.
- No peak-memory claim is made here: Wan still decodes the full latent tensor before the save path.
  Full VAE decode streaming and measured process-memory evidence remain follow-up performance work.

## Progress checklist
- [x] Add Wan streaming or batched save path.
- [x] Keep VLM top-p sampling on MLX.
- [x] Vectorize VLM image-token replacement.
- [x] Make BF16 loader policy explicit.

## Implementation notes
Do not bundle all optimizations into one risky patch. Preserve existing output metadata and health
gates while reducing memory pressure.
