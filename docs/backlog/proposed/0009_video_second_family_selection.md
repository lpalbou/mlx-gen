# Proposed: Video second-family selection

## Metadata

- Created: 2026-05-28
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: May need a video-backend ADR if MLX-Gen adds a second major video architecture,
  audio-video support, or provider-style runtime boundary.

## Context

Wan2.2 TI2V 5B is already the right first native video target because it is Apache 2.0, supports
text-to-video and image-to-video, and is already partially ported. Planned items 0002 and 0005
track Wan quantization, motion parity, and q8 performance. Proposed item 0006 tracks a focused
I2V prompt-motion concern.

The next decision should not be "port everything." It should be a selection pass that compares the
next video family after Wan against MLX-Gen's constraints: Apple Silicon memory, local cache-only
generation, reuse of the shared progress callback API, clear cancellation behavior, quantization
quality, licensing, and AbstractVision reuse.

## Current code reality

- MLX-Gen has initial Wan2.2 TI2V text-to-video and first-frame image-to-video support under
  `src/mflux/models/wan/`.
- The local Diffusers checkout includes several candidate video stacks:
  - Wan T2V/I2V/V2V/VACE/Animate under `pipelines/wan/`.
  - LTX and LTX-2 under `pipelines/ltx/` and `pipelines/ltx2/`.
  - HunyuanVideo 1.5 T2V/I2V under `pipelines/hunyuan_video1_5/`.
  - CogVideoX T2V/I2V/V2V/Fun-Control under `pipelines/cogvideo/`.
  - Mochi T2V under `pipelines/mochi/`.
  - SkyReels and Hunyuan FramePack-related pipelines in the broader Diffusers tree.
- MLX-Gen also has SeedVR2 code, but that is video restoration/upscale rather than a new T2V/I2V
  generator family.

## Problem or opportunity

Video ports are expensive and slow to validate. Starting LTX-2, HunyuanVideo, CogVideoX, Mochi,
and Wan extensions simultaneously would fragment the package and leave no backend fully reliable.

The second video family should be selected by measured value, not hype.

## Proposed direction

Create a short selection report before starting a second large video port. Compare:

| Candidate | Why it is interesting | Concern |
| --- | --- | --- |
| LTX-2.3 | Broad T2V/I2V/V2V/audio-video feature set, distilled checkpoints, latent upscalers, IC/HDR LoRA pipelines. | 22B model family, custom license, audio-video complexity. |
| HunyuanVideo 1.5 | 8.3B, T2V/I2V, Diffusers-supported, current video ecosystem interest, LoRA accelerator ecosystem. | Custom Tencent license and another major architecture. |
| CogVideoX-2B | Apache 2.0, smaller and older, local Diffusers stack available. | Lower current strategic value than Wan/LTX/Hunyuan. |
| Mochi 1 Preview | Apache 2.0, strong open T2V history, Diffusers pipeline exists. | Older T2V-only target with less AbstractVision differentiation. |
| Wan VACE / Wan Animate | Same family as current port; may reuse Wan infrastructure after A14B. | Much larger models or additional modes; should wait for Wan 5B/A14B parity. |

Selection criteria:

- permissive or clearly manageable license;
- fits Apple Silicon memory after q8 or mixed q4/q8;
- generates usable motion at acceptable settings;
- has an upstream Diffusers/Transformers implementation to port line-by-line;
- supports a modality gap AbstractVision actually needs;
- can produce small deterministic validation clips within a reasonable time budget.

## Why it might matter

The second video backend will shape MLX-Gen's video API. Choosing too early could lock the package
into audio/video abstractions or memory behavior that the current Wan work has not proven yet.

## Promotion criteria

- Planned Wan item 0002 has a clear quality/quantization status.
- Planned Wan item 0005 has explained or fixed q8 slowness.
- A candidate produces materially better or different value than Wan on the same Apple Silicon
  machine.
- The selected candidate has an acceptable license and a concrete local source snapshot.

## Validation ideas

- One fixed prompt across candidates: 3-5 seconds, 24 fps where supported, recommended resolution,
  and documented low-cost fallback.
- One fixed I2V prompt and source image where supported.
- Wall time, peak memory, output frame count, accepted resolutions, and qualitative motion score.
- Verify Diffusers output first, then port MLX only after the upstream behavior is known.

## Non-goals

- This proposal does not authorize starting LTX-2.3 or HunyuanVideo before Wan is stable.
- This proposal does not duplicate Wan motion-parity work in items 0002 and 0006.
- Wan2.2 A14B T2V/I2V moved out of this proposal into planned item 0012 on 2026-05-30.
- This proposal does not include closed/API-only models except as AbstractVision provider ideas.

## Guidance for future agents

Do the selection pass as documentation and measurements first. If a candidate wins, create a
separate planned item for the actual port with current code reality, model size, license, expected
CLI/API, progress events, quantization plan, and validation clips.

## Sources checked

- `src/mflux/models/wan/`
- Local Diffusers video pipelines under `/Users/albou/projects/gh/diffusers/src/diffusers/pipelines/`
- Wan2.2 TI2V 5B model card: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers
- LTX-2.3 model card: https://huggingface.co/Lightricks/LTX-2.3
- HunyuanVideo-1.5 model card: https://huggingface.co/tencent/HunyuanVideo-1.5
- Mochi 1 Preview model card: https://huggingface.co/genmo/mochi-1-preview
