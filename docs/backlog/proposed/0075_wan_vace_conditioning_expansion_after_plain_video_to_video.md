# Proposed: Wan VACE Conditioning Expansion After Plain Video-To-Video

## Metadata
- Created: 2026-07-03
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md), [ADR 0006](../../adr/0006_generative_video_editing_task_boundary.md)
- ADR impact: May revise ADR 0006 only if real implementation evidence shows the first public task
  boundary is insufficient.

## Context

Upstream Wan VACE supports richer conditioning than plain video-to-video:

- source-video masking;
- reference-image-guided edits;
- control conditioning and layer-wise conditioning scale.

These are valuable, but they require more than the first plain source-video route: a better input
role contract, stricter validation, and more memory-sensitive proof.

## Current code reality

- Current MLX-Gen does not ship plain generative source-video editing yet.
- Existing planner contracts are still image-centric.
- Local reference evidence is still being established under the bounded-proof work.

## Problem or opportunity

If plain `video-to-video` lands cleanly, the next useful expansion is Wan VACE-style conditioning.
That should be preserved as its own follow-up instead of being silently smuggled into the first
runtime milestone.

## Proposed direction

- Keep public task naming at `video-to-video`.
- Add optional conditioning roles later only after plain V2V is implemented and validated:
  - video mask;
  - reference images;
  - control inputs;
  - conditioning scale.
- Decide later whether any of those inputs deserve a public CLI noun beyond `video-to-video`, or
  whether they remain route-specific options under the same task.

## Why it might matter

This is where localized video edits and stronger structure-preserving changes become practical, but
only if the base route and host-memory profile are already trustworthy.

## Promotion criteria

- [0074 Wan plain generative video-to-video route](../completed/0074_wan_plain_generative_video_to_video_route.md)
  is complete with one accepted model-backed proof.
- The bounded reference harness proves at least one simple VACE-style case is plausible on this
  host.
- Planner ownership for typed conditioning roles is clear enough to add without overloading the
  old image-count contract.

## Validation ideas

- Localized portrait edit with repeated video mask and preserved output metrics.
- Structural ship edit with either mask or reference-image conditioning and preserved memory stats.
- Failure-manifest and progress-event checks for conditioned video runs.

## Non-goals

- Do not authorize VACE implementation before plain V2V lands.
- Do not add public `video-edit`, `control-to-video`, or other new task names without explicit ADR
  evidence.
- Do not treat reference-pipeline success as automatic native MLX-Gen readiness.

## Guidance for future agents

Treat this as a distinct second milestone. If a proposed conditioning role cannot be named clearly
for users and tested clearly for machines, it is not ready for promotion.
