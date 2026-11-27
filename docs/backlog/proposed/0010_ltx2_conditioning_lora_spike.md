# Proposed: LTX family conditioning and LoRA spike

## Metadata

- Created: 2026-05-28
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: Likely needs a video/audio backend ADR before implementation because the LTX family
  is not only a simple T2V image model.

## Context

LTX remains one of the most interesting non-Wan video families, but the first realistic MLX-Gen
target is not "all of LTX." The public family now includes:

- `LTX-Video` and `LTX-Video-0.9.8-13B-distilled`, which are the most practical first local
  T2V/I2V/V2V candidates;
- `LTX-2`, which broadens conditioning and modality support;
- `LTX-2.3`, which pushes toward a larger audio-video platform with distilled checkpoints, LoRAs,
  and latent upscalers.

This proposal is separate from the broader second-family selection item because the LTX family has
its own conditioning, LoRA, latent-upscaling, and potential audio-video surface.

## Current code reality

- MLX-Gen has no LTX or LTX-2 model family today.
- The local Diffusers checkout includes:
  - `pipeline_ltx.py`
  - `pipeline_ltx2.py`
  - `pipeline_ltx2_image2video.py`
  - `pipeline_ltx2_condition.py`
  - `pipeline_ltx2_ic_lora.py`
  - `pipeline_ltx2_hdr_lora.py`
  - `pipeline_ltx2_latent_upsample.py`
  - LTX-2 export utilities, image processing, connectors, and vocoder support.
- MLX-Gen's current LoRA implementation is image-transformer oriented and does not yet define
  video LoRA capability metadata.
- MLX-Gen's current video output path is Wan MP4 output, not a general audio-video container
  pipeline.

## Problem or opportunity

If MLX-Gen wants meaningful video LoRA support beyond Wan, the LTX family may be the cleanest
target to study because Diffusers already separates IC LoRA and HDR LoRA pipelines. But leading
with the full LTX-2.3 platform would be risky: audio latents, extra conditioning surfaces,
temporal upscaling, and LoRA adapters all pull the public API beyond today's Wan scope.

## Proposed direction

Keep this as a spike that can be promoted only after item 0009 selects the LTX family or after a
concrete consumer workflow needs it. If promoted, start with `LTX-Video` first:

1. Run the upstream Diffusers `LTX-Video` pipelines first and record exact model files, memory,
   runtime, output shape, fps, and license constraints.
2. Decide whether MLX-Gen should initially port only `LTX-Video` / distilled I2V without audio,
   or whether `LTX-2` adds enough value to justify a broader first pass.
3. Map LTX-family LoRA semantics separately from image LoRA semantics.
4. Prototype only the smallest deterministic path first: `LTX-Video` distilled I2V without audio.
5. Keep `LTX-2.3` audio-video and larger multi-stage workflows as explicit follow-up scope.
6. Add IC LoRA only after the base LTX path has parity fixtures.

## Why it might matter

The LTX family is one of the few open-weight candidates that could move MLX-Gen beyond
"generate a video" into reusable video direction: reference conditioning, camera/control LoRAs,
video-to-video, and latent upscaling. That is valuable, but only if the package can support it
cleanly without inheriting an audio-video contract too early.

## Promotion criteria

- Item 0009 selects the LTX family as the next video family, or a concrete consumer workflow needs
  LTX-specific conditioning/LoRA.
- License review confirms the intended use and redistribution terms are acceptable.
- Upstream Diffusers generation works locally with a practical low-cost validation clip.
- A scoped first implementation can avoid broad audio-video API churn.

## Validation ideas

- Upstream Diffusers baseline for distilled I2V at a documented low-cost size.
- MLX prompt-embedding and latent-pack parity tests before full video generation.
- Short MLX output clip compared against Diffusers for the same prompt, seed, dimensions, frames,
  steps, and guidance.
- Separate LoRA validation showing a visible effect from a known IC LoRA while base output remains
  stable without it.

## Non-goals

- Do not implement the LTX family before Wan quality and q8 behavior are understood.
- Do not mix audio-video, IC LoRA, HDR LoRA, latent upscaling, and base T2V in one first pass.
- Do not represent the LTX family as Apache unless the current model card/license says so; it uses
  its own LTX license terms.

## Implementation notes

Treat LTX as a video platform, not a single model id. Start with upstream Diffusers behavior and
write a small ADR if the port requires new public concepts such as audio latents, temporal
upscalers, video LoRA adapters, or multi-stage generation.

## Sources checked

- Local Diffusers checkout LTX pipelines under `diffusers/src/diffusers/pipelines/ltx/` and
  `diffusers/src/diffusers/pipelines/ltx2/`
- LTX-Video model card: https://huggingface.co/Lightricks/LTX-Video
- LTX-Video-0.9.8-13B-distilled model card: https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled
- LTX-2 model card: https://huggingface.co/Lightricks/LTX-2
- LTX-2.3 model card: https://huggingface.co/Lightricks/LTX-2.3
