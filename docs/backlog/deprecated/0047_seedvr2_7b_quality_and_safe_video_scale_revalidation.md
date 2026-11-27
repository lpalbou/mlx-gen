# Deprecated: SeedVR2 7B quality and safe video scale revalidation

## Metadata

- Created: 2026-06-20
- Status: Deprecated
- Completed: N/A
- Deprecated: 2026-06-21

## ADR status

- Governing ADRs:
  - [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
  - [ADR 0004](../../adr/0004_seedvr2_video_host_safety_and_proof_boundaries.md)
- ADR impact: None if this remains a validation-and-proof item rather than a broad runtime policy
  change.

## Context

Completed item [0032](../completed/0032_seedvr2_video_restoration_upscaling.md) established
SeedVR2 video restoration as a real MLX-Gen route and preserved official full-source `3B` and `7B`
proof videos. The runtime now also has a conservative host-safe video contract under ADR 0004.

That closes route proof, not family-quality ranking.

Two questions remain open:

1. under the current MLX-Gen port, what is the fairest bounded visual comparison between official
   `3B`, regular `7B`, and `7B-sharp` on archival source-size restoration; and
2. whether any enlarged SeedVR2 video profile can be promoted from unsafe experiment to a validated
   safe public recipe.

## Problem or opportunity

Public route proof is now honest about host safety, but SeedVR2 family-quality guidance is still
too soft. Users need a clear answer on what `3B`, `7B`, and `7B-sharp` are each good at before
MLX-Gen should recommend one as the default video family choice.

## Proposed direction

1. Keep the current full-source `1x` Eiffel videos as route proof only.
2. Build one bounded, like-for-like family comparison on a source-size archival segment:
   - `3B`
   - `7B`
   - `7B-sharp`
3. Require direct visual review plus bounded metrics before making any public quality claim.
4. Keep enlarged video output behind `--force-unsafe-video-memory` until one profile proves both:
   - host-safe memory behavior; and
   - better restored output than the source-size route on a preserved proof set.

## Why it might matter

This is the smallest honest follow-up that can turn SeedVR2 video from “working and host-safe” into
“well-explained and well-ranked.”

## Promotion criteria

Promote after:

- the ADR 0004 host-safe CLI contract is merged and documented;
- the invalid heuristic-only family ranking language is removed from public docs; and
- one bounded source-size archival comparison set is selected for direct visual review.

## Deprecation report

- Date: 2026-06-21
- Reason: the original scope is no longer the right follow-up boundary.

Release `0.18.20` completed the bounded five-second public comparison work that this item was
meant to drive:

- the accepted June 21 public SeedVR2 proof bundle now lives under
  `docs/assets/validation/seedvr2-video-2026-06-21/`;
- completed item [0032](../completed/0032_seedvr2_video_restoration_upscaling.md) now records the
  accepted `1x 29/8` route-correctness proof and the bounded `2x 29/8` quality comparison;
- the current docs and release assets already make the bounded `3B` versus `7B` conclusions
  explicit.

That closes the "7B quality revalidation" half of the item. The only remaining question is
whether any enlarged SeedVR2 video profile should ever graduate from explicit unsafe override to a
documented safe public profile. That narrower follow-up is now tracked separately in proposed item
[0048](../proposed/0048_seedvr2_enlarged_video_safe_profile_certification.md).

This item is deprecated instead of completed because its original mixed scope no longer matches the
remaining work.

## Implementation notes

Do not reopen this item. Use completed item 0032 for the accepted bounded proof history, use
completed item 0046 for the shipped audio copy-through follow-up, and use proposed item 0048 for
any future safe-enlarged-profile certification work.
