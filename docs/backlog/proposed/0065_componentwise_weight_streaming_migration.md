# Proposed: Component-wise weight streaming migration

## Metadata
- Created: 2026-06-27
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: May need an ADR if the migration establishes a cross-family loader contract or
  rollback policy.

## Context
Planned item [0063](../planned/memory/0063_componentwise_model_loading_memory_policy.md) moved the
safe runtime memory policy earlier by applying MLX cache limits before model construction across
CLIs, but it is not complete until quantitative startup-memory evidence exists. It did not migrate
generic image and restoration initializers to Wan-style component-wise load/apply/clear.

Wan already streams component loading and clears each component after application. Most other
families still load all components into one `LoadedWeights` object before model construction,
weight application, quantization, and LoRA handling.

## Current code reality
- `WeightLoader.load()` still returns all components together, while Wan uses
  `WeightLoader._load_component(...)` one component at a time and clears after each apply.
- The 2026-06-29 prepared-package size audit found sharply different overlap ceilings by family:
  SeedVR2 prepared packages can only remove about `0.5 GB` of raw overlap, while cached Z-Image,
  FLUX.2, and Qwen prepared packages can remove about `4-8 GB` if startup peaks are actually
  dominated by concurrent raw component residency.
- Phase-isolated startup proof is still missing. The existing 0063 cache-policy evidence shows
  retained-cache improvement but not a clear launch-to-first-step reduction.

## Problem
Startup peaks can still include raw loaded component weights, initialized modules, quantization
transients, and adapter work at the same time. This is likely the next meaningful startup-memory
reduction after the pre-construction cache policy fix.

## What we want to do
Design and prove an incremental component-wise migration path for non-Wan families without
changing quantization semantics, validation checks, prepared package layout, or generated quality.

## Why
The memory audit found this as a real issue, but the adversarial pass also found that a naive broad
helper is risky: Qwen, Flux, FIBO, ERNIE, Z-Image, and SeedVR2 each have family-specific
construction, validation, and quantization assumptions.

## Requirements
- Preserve existing `WeightLoader.load()` and `WeightApplier.apply_and_quantize()` paths until a
  family migration is proven.
- Reuse Wan's load/apply/delete/cache-clear pattern where it fits.
- Keep family initializers responsible for component order, validation, model factories, and
  quantization predicates.
- Add rollback or feature-gate guidance if the first migration touches release-critical routes.
- Measure process memory, not only MLX allocator memory, before making public claims.

## Promotion criteria
Promote when profiling shows startup peak remains a practical blocker after the remaining planned
memory items 0060-0064, or when one specific family with a material overlap ceiling needs the
memory reduction for a supported profile.

Do not promote this item solely from the SeedVR2 1280px image measurements in
`validation_outputs/memory/real_generation_20260627_seedvr2_image_1280_tiling_r2/`. Those runs
show VAE spatial encode tiling reduces MLX peak, while cache-control and startup policy do not.
This migration should remain tied to measured startup or first-step weight-loading overlap.

## Suggested first target
Prefer one large prepared image family with simple component ownership and lower validation risk
for the first migration. Z-Image and FLUX.2 currently look like the best first targets because the
measured prepared-package overlap ceiling is materially larger than SeedVR2 and the initializer
shape is simpler than Qwen's mixed-policy paths. Treat official eager source-checkpoint streaming
as a separate follow-up if those routes remain important after phase profiling.

## Non-goals
- Do not rewrite every initializer in one pass.
- Do not change prepared checkpoint layout or model-card semantics.
- Do not hide family-specific validation behind a generic abstraction.
- Do not accept output-quality drift to reduce startup memory.

## Validation ideas
- Fake-component tests proving load/apply/delete/cache-clear order.
- Family-specific initializer tests for the first migrated backend.
- Prepared-package q8/q4 smoke for the migrated family.
- Physical-process memory comparison for model construction through first denoise/upscale step.
- Phase-separated comparison that includes `decode`, `save`, and `health` so the migration does not
  simply move the peak later in the run.
