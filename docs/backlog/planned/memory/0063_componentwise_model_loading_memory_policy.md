# Planned: Component-wise model loading memory policy

## Metadata
- Created: 2026-06-27
- Status: Planned
- Completed: N/A
- Reopened: 2026-06-27

## ADR status
- Governing ADRs: None
- ADR impact: May revise existing ADR if model profile ownership becomes a durable policy.

## Context
Wan loads, applies, and clears weights component by component. Most generic image families load all
components into one `LoadedWeights` object before applying and quantizing.

## Current code reality
- `WeightLoader.load()` loads every component before returning.
- Flux, Qwen, Z-Image, Flux2, FIBO, and ERNIE initializers generally load all weights, initialize
  modules, apply all weights, and then apply LoRA.
- Wan's initializer has a component-wise path that clears component weights after each apply.
- Generic image CLIs now apply the shared cache-limit hook before model construction, but that
  earlier policy still has not been proven to reduce launch-to-first-step peak memory.

## Problem
Startup peaks can include raw loaded weights, quantized model parameters, and LoRA work at the same
time. `--low-ram` cannot affect that peak when it is applied after construction.

## What we wanted to do
Introduce a reusable component-wise loading/applying primitive and move memory/cache policy earlier
where it is safe.

## Why
Startup load peak matters for large image families and prepared packages, not only Wan.

## Requirements
- Preserve quantization resolution semantics.
- Preserve family-specific validation such as Qwen corrupt-weight checks and FIBO compatibility
  checks.
- Clear component weights after applying and synchronizing.
- Avoid import-cycle regressions.

## Suggested implementation
First add a reusable primitive that can apply one component and clear it. Migrate one lower-risk
family or add a guarded helper path before broad migration.

## Scope
- Common weight-loading primitives.
- Cache-limit helper reuse.
- Focused initializer tests using fake components.

## Non-goals
- Do not rewrite every initializer blindly in one pass.
- Do not change quantized checkpoint layout or model-card semantics.
- Do not change generated image/video quality.

## Dependencies and related tasks
- Proposed [0058 model profile registry authority](../../proposed/0058_model_profile_registry_authority.md)
- Completed [0060 runtime memory telemetry](../../completed/0060_runtime_memory_telemetry_and_manifests.md)

## Expected outcomes
- There is a tested path for component-wise apply/clear outside Wan.
- Future family migrations can be incremental.
- Low-RAM/cache policy has a clear pre-construction hook.

## Validation
- Unit tests for component-wise helper behavior.
- Existing weight/prepare tests continue to pass.

## Progress checklist
- [ ] Add common component-wise apply helper. Split to proposed item [0065](../../proposed/0065_componentwise_weight_streaming_migration.md).
- [x] Factor cache-limit helper.
- [ ] Migrate at least one safe family or leave a tested primitive with explicit follow-up. Split
  to proposed item [0065](../../proposed/0065_componentwise_weight_streaming_migration.md).

## Guidance for the implementing agent
Do not trade a startup-memory win for silent quantization or validation drift. This item is
architecture-sensitive and can be completed incrementally only if the primitive is real and tested.

## Completion report

- Date: 2026-06-27
- Original path: `docs/backlog/planned/memory/0063_componentwise_model_loading_memory_policy.md`
- Final path: `docs/backlog/completed/0063_componentwise_model_loading_memory_policy.md`
- Summary: Moved the safe actionable startup-memory policy into a shared pre-construction
  cache-limit hook and applied it across CLI entrypoints. The broader component-wise loader
  migration was not completed and is now tracked as proposed item
  [0065](../../proposed/0065_componentwise_weight_streaming_migration.md).
- Implementation: `CallbackManager.apply_runtime_memory_options(args)` now applies MLX cache
  limits before model construction. Flux, Flux2, FIBO, Qwen, Z-Image, ERNIE, Bonsai, SeedVR2, and
  Wan CLI paths call the hook before constructing large model objects.
- Behavior changes: `--mlx-cache-limit-gb` and low-RAM cache policy now affect startup loading
  peaks where possible instead of only the denoising loop.
- Validation: Compile and ruff checks passed; focused CLI tests for Wan low-RAM cache behavior
  passed.
- Residual risk: Generic component-wise load/apply was not migrated across image/restoration
  initializers in this pass because each family has validation and quantization semantics that need
  model-backed profiling before broad refactor. The completed scope is the pre-load cache policy
  fix that was safe across entrypoints.

## Reopen report

- Date: 2026-06-27
- Reason: The safe pre-construction cache hook landed, but closure requires quantitative startup
  memory statistics proving that the policy reduces model-construction or first-step peak memory.
- Required evidence before closure: process-isolated runs from process launch through first
  denoise/upscale step, with and without pre-construction cache policy, reporting peak physical
  footprint/RSS, MLX peak/cache, wall time, and whether startup remains dominated by raw
  weight-loading overlap covered by proposed item 0065.

## Quantitative validation update

- Date: 2026-06-27
- Evidence artifact:
  `validation_outputs/memory/real_generation_20260627_zimage_r3/generation_memory_benchmark.json`.
- Real profile: `mflux-generate-z-image-turbo`, `AbstractFramework/z-image-turbo-8bit`, 384x384,
  2 steps, seed 5151, default cache policy versus `--mlx-cache-limit-gb 1`, three fresh CLI runs
  per variant.
- Result: median MLX cache fell from 12.628 GB to 1.001 GB and median metadata physical footprint
  fell from 17.630 GB to 6.134 GB at the image-metadata boundary. Median sampled process-tree RSS
  only fell from 11.498 GB to 11.392 GB, and median MLX peak stayed flat at about 12.682 GB.
- Quality/performance result: generated image comparison was exact (`mae=0`, `rmse=0`,
  `max_abs=0`). Median wall time was 3.72 s default versus 3.56 s cache-limited.
- Status: Still planned. The pre-load cache policy reduces retained cache and post-generation
  footprint, but it does not solve startup/first-step peak. Proposed item 0065 remains the likely
  next high-impact startup-memory task.

## SeedVR2 1280px image cache-policy update

- Date: 2026-06-27
- Evidence artifacts:
  `validation_outputs/memory/real_generation_20260627_seedvr2_image_1280_r2/generation_memory_benchmark.json`
  and
  `validation_outputs/memory/real_generation_20260627_seedvr2_image_1280_tiling_r2/generation_memory_benchmark.json`.
- Result: SeedVR2 image `--low-ram` is now a stable pre-generation cache-control path for 1280px
  3B and 7B restores, and it preserves exact pixels. It reduces metadata-time physical footprint
  but does not reduce MLX peak.
- Interpretation: The measured 1280px SeedVR2 image peak is dominated by VAE spatial encode
  residency, not by startup weight-loading overlap. Tuned explicit `--vae-tiling` reduces the MLX
  peak by 46.24% for 3B and 24.16% for 7B, but changes pixels. Do not promote proposed item
  [0065](../../proposed/0065_componentwise_weight_streaming_migration.md) solely to solve this
  1280px image peak; promote it only for a measured startup/first-step problem.

## Quantitative status update

- Date: 2026-06-28
- Status: Still planned. No new startup or first-step profile has proven that current
  pre-construction cache policy reduces the launch-to-first-step peak. The 0060 telemetry work
  now provides a stronger parent physical-footprint sampler, and the 0061 report plus reopened
  0062 normal 1:1 evidence provide context for other parts of the memory track, but they do not
  isolate startup model-loading overlap.
- Current interpretation: keep proposed
  [0065 component-wise weight streaming migration](../../proposed/0065_componentwise_weight_streaming_migration.md)
  as the architecture follow-up. Promote it only after a real startup profile identifies one model
  family whose peak remains dominated by raw weight loading plus live initialized parameters.
- Required next proof: a process-isolated launch-to-first-denoise or launch-to-first-upscale profile
  with and without pre-construction cache policy, including parent sampled RSS/physical footprint,
  MLX peak/cache, wall time, and a clear phase marker for model construction versus first compute.

## Research update

- Date: 2026-06-29
- Local package-size ceilings from cached prepared packages show that the likely upside varies
  sharply by family. The practical upper bound for component-wise overlap removal is roughly
  `total_loaded_component_bytes - largest_component_bytes` before quantization, eval, and LoRA
  transients:
  - `seedvr2-3b-4bit`: total `2.732 GB`, largest component `2.234 GB`, overlap ceiling `0.498 GB`
  - `seedvr2-7b-4bit`: total `5.138 GB`, largest component `4.639 GB`, overlap ceiling `0.498 GB`
  - `z-image-turbo-8bit`: total `10.981 GB`, largest component `6.541 GB`, overlap ceiling
    `4.440 GB`
  - `flux.2-klein-4b-8bit`: total `8.558 GB`, largest component `4.274 GB`, overlap ceiling
    `4.284 GB`
  - `flux.2-klein-9b-8bit`: total `17.854 GB`, largest component `9.646 GB`, overlap ceiling
    `8.208 GB`
  - `qwen-image-8bit`: total `29.479 GB`, largest component `21.713 GB`, overlap ceiling
    `7.767 GB`
- Interpretation: the highest-value prepared-package streaming targets are the larger image
  families, not SeedVR2 prepared packages. SeedVR2 still matters for safety-sensitive video work,
  but its prepared-package overlap ceiling is modest compared with Z-Image, FLUX.2, and Qwen.
- External evidence: Safetensors supports partial tensor access and slice-oriented loading, while
  Diffusers documents sharded checkpoint loading as a load-time memory reducer. MLX unified memory
  and lazy evaluation explain why cache-limit policy can improve retained cache without proving a
  launch-to-first-step peak win. See:
  `https://huggingface.co/docs/safetensors/en/index`,
  `https://huggingface.co/docs/diffusers/optimization/memory`,
  `https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html`,
  `https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html`.
- Revised direction: keep item 0063 focused on two deliverables:
  `1.` phase-isolated proof of where startup and end-of-run peaks actually occur;
  `2.` one family-scoped component-wise or shard-wise migration with exact output parity.
  Prefer a simpler prepared image family such as Z-Image or FLUX.2 for the first migration.
  Defer Qwen until mixed-policy and corrupt-weight checks are preserved, and treat eager
  source-checkpoint streaming as a separate follow-up if official `.pth` routes remain important.
- Required next proof before closure: add benchmark phases for `model_init`,
  `load_component.<name>`, `apply_component.<name>`, `first_eval`, `decode`, `save`, and `health`,
  then compare launch-to-first-step and full-run peaks before claiming that startup memory is fixed.
