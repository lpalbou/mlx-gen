# Proposed: Wan VACE video editing and control

## Metadata

- Created: 2026-06-11
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: May need a small video-task ADR if MLX-Gen adds first-class video-to-video,
  masked video editing, or control-to-video taxonomy beyond the current text-to-video and
  image-to-video split.

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

## Proposed direction

Track Wan VACE as its own future Wan extension:

1. Audit the upstream Wan VACE and Wan video-to-video routes against current MLX-Gen Wan code.
2. Decide the smallest useful first route:
   - plain video-to-video;
   - masked video edit;
   - reference-image-guided video edit; or
   - structured control-to-video.
3. Reuse MLX-Gen's existing Wan scheduler, transformer-role, metadata, and MP4 save surface where
   possible.
4. Keep first implementation bounded and fail-closed. Do not infer VACE behavior from current T2V
   or I2V routes.

## Why it might matter

Wan VACE is probably the highest-value video editing/control opportunity that stays inside an
already-supported architecture. It is more strategically aligned than starting a whole new video
family just to get video editing primitives.

## Promotion criteria

- Planned items 0035 and 0015 settle current Wan TI2V prompt/motion confidence sufficiently that a
  new Wan mode will not hide unresolved base-family issues.
- At least one official Wan VACE or Wan video-to-video model path is runnable locally with bounded
  settings.
- The public MLX-Gen/AbstractVision contract for video-to-video or masked video editing is clear
  enough to expose without guesswork.
- One exact upstream weight set is selected as the first proof target.

## Validation ideas

- Small MP4 smoke for the chosen first route with source video, prompt, seed, dimensions, frames,
  steps, and output metadata.
- Contact sheet or frame strip comparing source frames and edited frames.
- If masked editing is chosen, include the mask visualization and command surface in proof assets.
- If structured control is chosen, include the control input in the proof artifact set.

## Non-goals

- Do not treat Wan VACE as a second video family; it is a Wan-family extension.
- Do not silently repurpose current Wan I2V/T2V flags for video-to-video or masked editing.
- Do not broaden into audio-video, full animation pipelines, or a new general video provider
  abstraction from this proposal alone.

## Guidance for future agents

Start with the smallest credible Wan editing route and preserve exact evidence. If a first-class
video-to-video or masked-video task needs a new public taxonomy, create the ADR before claiming
support in docs or capabilities.

## Sources checked

- Local Diffusers Wan pipelines under `/Users/albou/projects/gh/diffusers/src/diffusers/pipelines/wan/`
- Diffusers Wan docs: https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/wan.md
- Wan VACE official weights: https://huggingface.co/Wan-AI/Wan2.1-VACE-14B
- Ali-ViLab VACE collection: https://huggingface.co/collections/ali-vilab/vace
