# Proposed: Boogu Image family support

## Metadata

- Created: 2026-06-21
- Status: Proposed

## ADR status

- Governing ADRs:
  - [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
  - [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: May need a backend-shape ADR if MLX-Gen decides to support another custom
  Qwen3-VL-conditioned image family rather than only extending the current Qwen/FLUX/Z-Image
  routes.

## Context

Boogu released a compact public image family on 2026-06-16:

- `Boogu-Image-0.1-Base`
- `Boogu-Image-0.1-Turbo`
- `Boogu-Image-0.1-Edit`
- matching `-fp8` variants

The official positioning is attractive for MLX-Gen's product shape:

- unified text-to-image plus image editing;
- strong Chinese/English text rendering;
- a fast four-step Turbo variant;
- Apache-2.0 model cards and open Hugging Face access.

But the strongest current quality evidence is still mostly Boogu's own reporting. Independent
leaderboards and reproducible local Apple Silicon evidence are not yet strong enough to justify a
full MLX port ahead of current Qwen, Z-Image, and Wan work.

## Current code reality

- MLX-Gen already supports Qwen Image/Edit, FLUX.2 Klein, Z-Image/Z-Image-Turbo, ERNIE Image
  Turbo, Wan2.2, and SeedVR2.
- MLX-Gen already has reusable Qwen3-VL building blocks under
  `src/mflux/models/common_models/qwen3_vl/`.
- MLX-Gen already has a reusable flow-match scheduler family under
  `src/mflux/models/common/schedulers/flow_match_euler_discrete_scheduler.py`.
- MLX-Gen does not have a `BooguImagePipeline`, `BooguImageTurboPipeline`, or
  `BooguImageTransformer2DModel` route.
- The current public image backlog already prefers finishing:
  - completed item [0008](../completed/0008_qwen_edit_parity_expansion.md),
  - completed item [0043](../completed/0043_zimage_native_inpaint.md), and
  - completed item [0007](../completed/0007_lora_capability_matrix_and_strict_application.md)
    before adding another image family.
- Proposed item [0011](0011_next_generation_image_edit_watchlist.md) preserves the broader
  next-generation image/edit watchlist; this item is the focused Boogu-specific follow-up.

## Problem or opportunity

Boogu is interesting enough to preserve as a concrete candidate, but not yet proven enough to
promote into active implementation:

- it may offer better bilingual dense-text rendering and prompt adherence than current defaults;
- it may offer a useful unified generation/editing family;
- but the port cost is medium-high because the hard part is a custom Boogu transformer and
  pipeline, not just another alias or weight mapping.

Without a focused backlog item, Boogu will either be forgotten or repeatedly reconsidered from
scratch.

## Proposed direction

Keep Boogu as a focused proposed item with a narrow future spike shape:

1. Start from the original non-fp8 `Base`, `Turbo`, and `Edit` checkpoints rather than the
   `-fp8` repos.
2. Audit the exact upstream component inventory and confirm whether the custom transformer can be
   mapped cleanly onto MLX-Gen's existing Qwen3-VL plus flow-match scaffolding.
3. Compare Boogu against current MLX-Gen routes on the tasks where it claims differentiated value:
   - bilingual poster or document-style text rendering,
   - prompt adherence on layout-heavy prompts,
   - one-image instruction editing against Qwen Image Edit 2511 and FLUX.2 Klein.
4. Promote only if Boogu clearly adds value rather than duplicating current support.

## Why it might matter

This is one of the few recent public image families that combines:

- open model cards,
- a fast Turbo route,
- a dedicated Edit route,
- Qwen3-VL-style multimodal conditioning,
- and explicit Chinese/English text-rendering ambition.

That makes it more interesting than a generic checkpoint clone, even though it is not yet a
priority.

## Promotion criteria

- At least one independent benchmark or repeatable local comparison shows a meaningful win over
  Qwen Image/Edit, Z-Image, or FLUX.2 Klein on a task MLX-Gen actually cares about.
- The original non-fp8 checkpoints prove locally runnable enough that an MLX port is credible on
  Apple Silicon.
- The architecture mapping from Boogu's custom transformer and pipeline into MLX-Gen is clear
  enough to estimate implementation scope honestly.
- License and provenance remain acceptable for MLX-Gen's support story and any later
  AbstractFramework packaging decisions.

## Validation ideas

- Upstream PyTorch or Diffusers smoke with one text-heavy prompt and one edit prompt.
- A small comparison panel against:
  - `Qwen/Qwen-Image-2512`,
  - `Qwen/Qwen-Image-Edit-2511`,
  - `Tongyi-MAI/Z-Image-Turbo`,
  - and one current FLUX.2 Klein edit route.
- Record source checkpoint size, component sizes, expected VRAM/RAM envelope, and exact pipeline
  dependencies.
- Confirm whether the Turbo route still looks meaningfully better than existing fast local image
  paths once prompt adherence and typography are judged directly.

## Non-goals

- Do not start a Boogu MLX port from this proposal alone.
- Do not treat the fp8 checkpoints as the primary MLX target.
- Do not mirror or publish Boogu derivatives from MLX-Gen without a separate packaging and
  license/provenance review.
- Do not let this item jump ahead of the active Qwen, Z-Image, and Wan backlog unless the
  evidence changes materially.

## Implementation notes

Re-check current public evidence before promotion. If Boogu is still mostly supported by vendor
benchmarks and anecdotal community reviews, leave it proposed. If it starts winning on independent
typography/editing evaluations, promote it as a focused image-family spike rather than burying it
inside the generic watchlist.

## Sources checked

- Boogu model org page: https://huggingface.co/Boogu/models
- Boogu Turbo model card: https://huggingface.co/Boogu/Boogu-Image-0.1-Turbo-fp8
- Boogu Edit model card: https://huggingface.co/Boogu/Boogu-Image-0.1-Edit-fp8
- Official Boogu repo: https://github.com/boogu-project/Boogu-Image
- Official project page: https://boogu.org/
- Qwen-Image-Bench repo: https://github.com/QwenLM/Qwen-Image-Bench
- Public text-to-image leaderboard: https://artificialanalysis.ai/image/leaderboard/text-to-image
- Public image-editing leaderboard: https://artificialanalysis.ai/image/leaderboard/editing
