# Proposed: Krea 2 Turbo integration

## Metadata

- Created: 2026-06-25
- Status: Proposed

## ADR status

- Governing ADRs:
  - [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
  - [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: May need an ADR only if MLX-Gen decides to support custom source-available model
  families with materially restrictive commercial and redistribution terms alongside the current
  permissive/open model mix.

## Context

`krea/Krea-2-Turbo` is a recent public text-to-image release with a clear local-product shape:

- open weights on Hugging Face;
- an official upstream repo;
- an eight-step distilled inference path;
- Qwen3-VL text conditioning plus a custom Krea transformer;
- strong public positioning around fast, aesthetic text-to-image output.

That makes it technically interesting for MLX-Gen. But the license is the opposite of a routine
Apache/MIT/BSD-style integration:

- custom `Krea 2 Community License Agreement v.1`;
- commercial use capped below `$1M` annual revenue without a separate enterprise license;
- revocable, non-transferable, and non-sublicensable;
- derivative distribution requirements, including `Krea` model-name prefixing;
- mandatory content-filter expectations and acceptable-use policy coupling;
- termination rights that can require stopping use and destroying copies.

This is materially more restrictive than the current permissive/open-weight families in MLX-Gen.

## Current code reality

- MLX-Gen already supports Qwen Image/Edit, FLUX.2 Klein, Z-Image/Z-Image-Turbo, ERNIE Image
  Turbo, Wan2.2, SeedVR2, and adjacent validated LoRA routes.
- MLX-Gen already has reusable Qwen3-VL components under
  `src/mflux/models/common_models/qwen3_vl/`.
- MLX-Gen already has a reusable flow-match scheduler family under
  `src/mflux/models/common/schedulers/flow_match_euler_discrete_scheduler.py`.
- MLX-Gen does not have any `Krea2Pipeline`, `Krea2TurboPipeline`, or Krea-specific transformer
  route today.
- Proposed item [0011](0011_next_generation_image_edit_watchlist.md) preserves the broader
  next-generation watchlist; this item is the narrower Krea-specific follow-up.
- Current planned work still favors Wan parity, first-class image-edit UX, and FLUX/Qwen/Z-Image
  validation over another new image family.

## Problem or opportunity

Krea 2 Turbo looks feasible enough that it should not be forgotten, but it is not a good default
next implementation target:

- the runtime shape is credible for MLX because parts of the stack overlap with existing Qwen3-VL
  and flow-match support;
- the likely first slice is narrow and useful: `text-to-image` on `Krea-2-Turbo`;
- but the port cost is still medium-high because the custom Krea transformer and exact pipeline
  semantics would need a new parity-first implementation;
- and the license is restrictive enough that even a technically successful integration may not fit
  MLX-Gen's support posture cleanly.

Without a focused backlog item, Krea will either be reconsidered repeatedly from scratch or
promoted prematurely because the images look interesting.

## Proposed direction

Keep Krea as a low-priority proposed item with a narrow future spike shape:

1. Treat `krea/Krea-2-Turbo` as the first candidate, not `Krea-2-Raw`.
2. Treat the first possible implementation as `text-to-image` only.
3. Match the official upstream contract exactly before any cleanup:
   - `8` inference steps,
   - `guidance=0`,
   - official `mu` handling,
   - official tokenizer, Qwen3-VL encoder, VAE, scheduler, and transformer semantics.
4. Re-check whether the license is acceptable for MLX-Gen before promotion from proposed to
   planned.
5. Promote only if Krea clearly adds user value beyond the current FLUX.2, Qwen, and Z-Image
   surfaces and the licensing tradeoff is still judged worth carrying.

## Why it might matter

Krea 2 Turbo is one of the more credible recent text-to-image families for Apple Silicon follow-up
because it combines:

- a bounded fast-path inference recipe;
- a modern multimodal text-conditioning stack rather than a legacy encoder path;
- a relatively clear official implementation reference;
- and real user interest in high-quality fast text-to-image generation.

That makes it more interesting than a random checkpoint clone, even if it stays low priority.

## Promotion criteria

- An explicit repo-level decision accepts support for this restrictive license class.
- A bounded upstream smoke confirms that Krea 2 Turbo is worth porting on output quality, speed,
  or user demand rather than curiosity alone.
- The Krea transformer and pipeline mapping are clear enough to estimate parity scope honestly.
- Apple Silicon memory and prepared-package strategy look credible enough that the port would be
  usable in practice, not just technically possible.

## Validation ideas

- One upstream smoke against the official Krea repo on the published Turbo recipe.
- Exact component inventory and checkpoint-size audit for `Krea-2-Turbo`.
- A small same-prompt comparison panel against one current fast local route:
  - FLUX.2 Klein,
  - Qwen Image,
  - or Z-Image-Turbo.
- If promotion ever happens, publish docs that state the license constraints plainly instead of
  implying a permissive open-source model story.

## Non-goals

- Do not treat the Krea license as Apache/MIT/BSD-adjacent.
- Do not start a Krea implementation from this proposal alone.
- Do not start from `Krea-2-Raw`, LoRA training, or hosted Krea API features.
- Do not let this item outrank current Wan, Qwen, FLUX.2, or Z-Image work unless the license and
  product-value picture changes materially.

## Implementation notes

Re-read the current Krea license before doing anything beyond watchlist triage. If the license
terms remain revocable, revenue-capped, and redistribution-restrictive, keep the item proposed
unless there is a deliberate product decision to carry that cost.

## Sources checked

- Krea 2 Turbo model card: https://huggingface.co/krea/Krea-2-Turbo
- Krea 2 Raw model card: https://huggingface.co/krea/Krea-2-Raw
- Official Krea 2 repo: https://github.com/krea-ai/krea-2
- Krea 2 licensing page: https://www.krea.ai/krea-2-licensing
- Krea 2 overview docs: https://docs.krea.ai/developers/krea-2/overview
- Krea 2 Turbo announcement: https://www.krea.ai/blog/krea-2-turbo
