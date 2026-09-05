# Deprecated: MiniMax H3 joint audio-video license-blocked watch

## Metadata

- Created: 2026-08-03
- Status: Deprecated
- Deprecated: 2026-09-04

## Deprecation note - 2026-09-04

Superseded by the owner's decision to port MiniMax-H3 despite the license territory exclusion
recorded below; the runtime shipped as completed item
[0117](../completed/0117_minimax_h3_text_to_video_audio_runtime.md), first-frame conditioning is
planned item [0118](../planned/0118_minimax_h3_first_frame_conditioning_vision_tower.md), and
performance follow-ups are proposed item
[0119](../proposed/0119_minimax_h3_adaln_precompute_and_long_sequence_performance.md). The license
analysis in this item remains the reference for the policy decision that is still owed (an ADR or
an explicit organizational license). The original text is preserved unchanged below.

## ADR status

- Governing ADRs:
  - [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
  - [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
  - [ADR 0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md)
  - [ADR 0006](../../adr/0006_generative_video_editing_task_boundary.md)
- ADR impact: Promotion would require a narrow ADR for generated joint audio-video task,
  result, save, metadata, and capability semantics. The current `GeneratedVideo` contract owns
  frames and optional source-audio copy-through; it does not own generated audio.

## Context

MiniMax H3 is an open-weight 768p video family that jointly generates video and synchronized
stereo audio. The public release includes separate FL2VA and Ref2VA 33B dense transformers and
advertises four-to-fifteen-second, 24 fps output. That is a genuinely new capability relative to
MLX-Gen's current silent video-generation routes.

The public license is a hard blocker for work in this checkout's normal France/EU development
scope. Its grant applies worldwide except the European Union, United Kingdom, Republic of Korea,
and United States, and its use restriction applies to the model, derivatives, and outputs outside
the permitted territory. MiniMax's own license Q&A confirms that these are restricted regions and
offers organizations a separate-license application path. Public Hugging Face access does not
override those terms.

This item records an interesting future candidate; it does not authorize downloading or executing
the weights. It is an engineering interpretation of the published terms, not legal advice.

## Current code reality

- MLX-Gen's public video routes are Wan-family T2V, I2V, and bounded V2V/VACE paths.
- `GeneratedVideo` saves frame sequences and may copy source audio for source-video workflows; it
  has no generated-audio tensor, codec, synchronization, mux, or provenance contract.
- H3 FL2VA and Ref2VA are distinct 33B dense-transformer routes. Each route also depends on a
  roughly 66.7 GB BF16 Qwen3-VL-32B text encoder, roughly 10.4 GB visual VAE, and a smaller audio
  VAE; a naive per-route BF16 component total is about 144 GB before activations and caches.
- The Diffusers work inspected on 2026-08-03 was still an unmerged branch/PR rather than a stable
  released pipeline.
- Same-week community MLX ports prove that key conversion and execution experiments have begun,
  but their reported speed and memory numbers are not MLX-Gen validation evidence. Quantization
  can improve storage and residency without solving the dense attention/runtime cost.
- H3 Context-IR and H3-Regenerate-2K are API products, not part of the public local-weight release;
  a future local port must not imply those capabilities.

## Problem or opportunity

H3 is strategically interesting because local joint audio-video generation would differentiate
MLX-Gen from a suite centered on silent Wan clips. It is currently unsuitable for implementation
because:

- the published license excludes the development territory;
- the upstream Diffusers contract has not stabilized;
- a useful Apple Silicon profile has not been demonstrated independently;
- generated audio expands MLX-Gen's public task, result, artifact, metadata, and validation
  contracts;
- the two large route-specific transformers make a complete port materially larger than a normal
  Wan variant.

## Proposed direction

Keep H3 as a license-blocked watch item:

1. Do not download, cache, execute, convert, redistribute, or publish H3 weights or outputs under
   the current public terms from the normal France/EU development environment.
2. Monitor the formal license and MiniMax Q&A for an amendment. A written organizational license
   that explicitly covers EU development is an acceptable alternative gate.
3. Monitor stable official Diffusers support and whether block-sparse attention becomes available
   with reference semantics that can be reproduced in MLX.
4. If and only if the license gate clears, start with a non-public five-second FL2VA/T2VA
   architecture spike using exact exported reference latents and the smallest defensible frame
   geometry.
5. Keep Ref2VA as a separate follow-up and keep the closed Context-IR and 2K API products out of
   local scope.
6. Decide the generated joint audio-video public contract by ADR before exposing a CLI task or
   Python result type. Hosted H3 API integration, if desired, belongs in a higher-level provider
   layer under ADR 0003 rather than in MLX-Gen's local runtime.

## Estimated work after the license gate clears

- Architecture and parity spike: about one to two engineering weeks.
- First non-public FL2VA/T2VA route: roughly six to ten engineering weeks if upstream semantics
  are stable.
- Production-quality FL2VA with audio artifact contracts, validation, quantized packaging, and
  documentation: roughly twelve to twenty engineering weeks total.
- Adding Ref2VA: roughly another six to ten engineering weeks, for a combined
  eighteen-to-thirty-week family scope.

These ranges exclude unavailable Context-IR and 2K API-only capabilities and may increase if
sparse-attention kernels or a new streaming decode path are required.

## Why it might matter

- Jointly generated speech, ambience, effects, and video are a new user workflow, not another
  quality variant of an existing T2V route.
- A validated quantized path could make a capability currently aimed at large CUDA systems
  available to high-memory Apple Silicon machines.
- The model would force MLX-Gen to define reusable generated-media provenance and synchronization
  contracts that other future audio-video families could share.

## Promotion criteria

- The formal public license permits the intended EU development, execution, output, and
  distribution scope, or the owner has obtained equivalent written permission.
- Stable official upstream code or merged Diffusers support fixes component names, schedulers,
  prompt processing, audio/video latent timing, and decode/mux semantics.
- A target-Mac profile defines an acceptable full-process physical memory ceiling and per-step or
  per-clip latency target before a full port begins.
- A generated joint audio-video ADR is accepted before public task, result, capability, metadata,
  or save behavior is added.
- One exact-source reference smoke can satisfy ADR 0001, and FL2VA versus Ref2VA model identity
  can fail closed under ADR 0002.
- Concrete user demand justifies the engineering cost over smaller differentiated candidates.

## Validation ideas after authorization

- Record the exact license revision, retrieved text hash, and written authorization scope before
  fetching weights.
- Export initial noise, text embeddings, visual latents, audio latents, timesteps, and one-step
  outputs from the reference implementation; do not use seed equality as the parity claim.
- Work backwards from artifacts: validate visual decode and audio decode independently, then mux
  duration and synchronization, then transformer/scheduler parity, then text/reference encoders.
- Run a five-second clip first and record prompt, seed, dimensions, frames, fps, steps, guidance,
  checkpoint revision, wall time, on-disk component sizes, and whole-process physical memory.
- Validate video integrity, stereo audio presence, finite waveforms, duration alignment, clipping,
  silence ratio, A/V synchronization, metadata, cancellation, and partial-output cleanup.
- Test FL2VA and Ref2VA routing as separate exact capabilities with intentional unsupported-state
  failures for closed Context-IR and 2K requests.

## Non-goals

- Do not treat this proposal as permission to fetch, run, convert, or publish H3 artifacts.
- Do not infer licensing rights from a public, ungated Hugging Face repository or its region tag.
- Do not claim local H3 Context-IR, H3-Regenerate-2K, or generic 2K support.
- Do not design a generic audio-video abstraction before one authorized model-backed spike proves
  the concrete contract.
- Do not hide H3 behind an existing silent T2V route or silently discard generated audio.

## Guidance for future agents

Read the then-current formal license and the official license Q&A together before doing any model
work. If the exclusion remains, update only public-source research and leave this item proposed.
If the gate clears, preserve the license evidence, create the joint audio-video ADR, and start with
one FL2VA five-second parity bundle. Re-estimate from that evidence before committing to Ref2VA or
public packaging.

## Sources checked

- MiniMax H3 model card: https://huggingface.co/MiniMaxAI/MiniMax-H3
- MiniMax H3 formal license: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
- MiniMax H3 license Q&A: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md
- MiniMax H3 official repository: https://github.com/MiniMax-AI/MiniMax-H3
- Diffusers H3 development documentation: https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md
- Community MLX 8-bit experiment: https://huggingface.co/pipenetwork/MiniMax-H3-MLX-8bit
