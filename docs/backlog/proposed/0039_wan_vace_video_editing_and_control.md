# Proposed: Wan VACE video editing and control

## Metadata

- Created: 2026-06-11
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md),
  [ADR 0006](../../adr/0006_generative_video_editing_task_boundary.md)
- ADR impact: None. ADR 0006 already fixes the public workflow boundary.

## Context

Wan is no longer only a text-to-video/image-to-video family upstream. The local Diffusers checkout
already includes:

- `pipeline_wan_video2video.py`
- `pipeline_wan_vace.py`
- `pipeline_wan_animate.py`

The upstream Diffusers Wan docs describe Wan VACE as a controllable video-generation/editing path,
and official Hugging Face weights exist for Wan VACE. This is strategically important because it
extends a family MLX-Gen already supports rather than introducing a completely different video
architecture.

Update 2026-06-15: the official Wan family now also includes `Wan2.2-Animate-14B` and
`Wan2.2-S2V-14B`. That reinforces Wan as a growing video platform, but this item should stay
strictly scoped to VACE, video-to-video, masking, and control surfaces closest to the current
MLX-Gen Wan routes.

## Current code reality

- MLX-Gen currently exposes Wan text-to-video and first-frame image-to-video routes only.
- Wan LoRA support is now implemented and validated for the current q8 public Wan rows.
- MLX-Gen does not expose first-class video-to-video, masked video editing, reference-image-guided
  video editing, or structured video control.
- The local Diffusers Wan stack already has explicit APIs for video conditioning, masks, and
  reference images in the VACE path.
- Proposed item 0009 tracks *second-family* video selection, but Wan VACE is not a second family.
  It is a higher-level extension of the Wan family already in MLX-Gen.

## Problem or opportunity

AbstractVision and direct MLX-Gen users will eventually want video editing, not only fresh
generation. The strongest near-family gap is Wan VACE-style control and editing:

- video-to-video restyle or motion-preserving edit;
- masked video editing;
- reference-image-guided video edits;
- structured control-to-video.

That opportunity is currently under-tracked because it is buried inside the broader second-family
selection proposal.

## Decision boundary

This item is no longer the place to decide the first public generative video-edit route.

That boundary is already fixed:

- public generative source-video editing belongs to `mlxgen generate`, not `mlxgen upscale`;
- the first public implementation must be plain `video-to-video`;
- richer Wan conditioning such as masks, reference images, or control belongs later.

Those decisions are already owned by [ADR 0006](../../adr/0006_generative_video_editing_task_boundary.md),
[0074 Wan plain generative video-to-video route](../completed/0074_wan_plain_generative_video_to_video_route.md),
and [0075 Wan VACE conditioning expansion after plain video-to-video](0075_wan_vace_conditioning_expansion_after_plain_video_to_video.md).

## Proposed direction

Keep this item as a bounded umbrella pointer for the Wan-family opportunity only:

1. Preserve Wan VACE as the strongest already-adjacent path for future richer video editing and
   control inside a family MLX-Gen already supports.
2. Keep plain `video-to-video` runtime work separate and earlier in execution order.
3. Keep VACE-specific mask, reference-image, and conditioning-scale work separate and later in
   execution order.
4. Reuse MLX-Gen's existing video save, metadata, and progress surfaces when later follow-up work
   proves one specific VACE capability is worth shipping.

## Why it might matter

Wan VACE is probably the highest-value video editing/control opportunity that stays inside an
already-supported architecture. It is more strategically aligned than starting a whole new video
family just to get video editing primitives.

## Promotion criteria

- Planned items 0035 and 0015 settle current Wan TI2V prompt/motion confidence sufficiently that a
  new Wan mode will not hide unresolved base-family issues.
- The plain public `video-to-video` route is either shipped or rejected with exact evidence.
- One exact Wan VACE-conditioned use case is proven locally strongly enough to justify richer
  conditioning beyond plain `video-to-video`.

## Validation ideas

- Preserve one exact upstream reference proof for a conditioned Wan VACE case with source video,
  prompt, seed, dimensions, frames, steps, and output artifacts.
- Preserve contact sheets comparing source, mask, and edited output frames.
- Keep any public-runtime smoke proof on the separate plain `video-to-video` item, not here.

## Non-goals

- Do not treat Wan VACE as a second video family; it is a Wan-family extension.
- Do not silently repurpose current Wan I2V/T2V flags for video-to-video or masked editing.
- Do not broaden into audio-video, full animation pipelines, or a new general video provider
  abstraction from this proposal alone.
- Do not silently absorb `Wan2.2-Animate-14B` or `Wan2.2-S2V-14B` into this item.

## Guidance for future agents

Do not use this item to reopen the first-route decision. The execution path is already split:

- [0074 Wan plain generative video-to-video route](../completed/0074_wan_plain_generative_video_to_video_route.md)
  owns the first public runtime milestone.
- [0075 Wan VACE conditioning expansion after plain video-to-video](0075_wan_vace_conditioning_expansion_after_plain_video_to_video.md)
  owns the later richer-conditioning follow-up.

## Sources checked

- Local Diffusers Wan pipelines under `/Users/albou/projects/gh/diffusers/src/diffusers/pipelines/wan/`
- Diffusers Wan docs: https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/wan.md
- Wan VACE official weights: https://huggingface.co/Wan-AI/Wan2.1-VACE-14B
- Ali-ViLab VACE collection: https://huggingface.co/collections/ali-vilab/vace
- Wan2.2-Animate-14B: https://huggingface.co/Wan-AI/Wan2.2-Animate-14B
- Wan2.2-S2V-14B: https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
