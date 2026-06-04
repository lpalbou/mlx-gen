# Proposed: Next-generation image/edit watchlist

## Metadata

- Created: 2026-05-28
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: May need a provider/backend ADR if MLX-Gen supports pixel-space or VLM-native image
  models that do not fit the current VAE plus diffusion-transformer pattern.

## Context

The May 2026 open image landscape is moving quickly. MLX-Gen should keep shipping Qwen, FLUX.2,
Z-Image, ERNIE, and Bonsai support, but it also needs a watchlist for models that could change the
best local Apple Silicon strategy.

Two candidates stand out:

- HiDream-O1-Image, because it is MIT-licensed, supports text-to-image, editing, and
  subject-driven personalization, and uses a pixel-level unified transformer rather than the usual
  separate VAE/text-encoder stack.
- Qwen-Image-2.0, because the May 2026 technical report describes a unified generation and editing
  model with high-fidelity generation and precise editing. Public weights and load paths still need
  verification before it becomes implementation work.

Z-Image-Edit and Z-Image-Omni-Base are also worth tracking: the Z-Image model card describes edit
and omni variants, but marks them as not yet released at the time checked.

## Current code reality

- MLX-Gen already supports Qwen Image/Edit, FLUX.2 Klein, Z-Image/Z-Image-Turbo, ERNIE Image
  Turbo, Bonsai ternary, and FIBO family routing.
- Local Diffusers includes `hidream_image`, `glm_image`, `flux`, `flux2`, `qwenimage`, and
  `z_image` pipelines.
- `ModelConfig` does not include HiDream-O1, Qwen-Image-2.0, GLM-Image, FLUX.1 Kontext, or
  Z-Image-Edit/Omni as first-class MLX-Gen targets.
- HiDream-O1 likely needs an MLX-VLM/pixel-space design review rather than a small weight-mapping
  addition.

## Problem or opportunity

We need to distinguish three categories:

1. Immediate engineering work on already-supported families.
2. Watchlist models that may become high value once weights, license, and upstream code stabilize.
3. Deprioritized models that are interesting but overlap existing support or introduce licensing
   constraints.

Without that split, every new model announcement can derail the current Wan/Qwen/LoRA work.

## Proposed direction

Maintain a watchlist table and promote only after concrete evidence appears:

| Candidate | Current evidence | Proposed status |
| --- | --- | --- |
| HiDream-O1-Image / Dev | MIT, 8-9B class, text-to-image, editing, subject-driven personalization, native 2048px claims, Transformers loading, local Diffusers pipeline. | Research after Qwen parity and ERNIE non-turbo validation. |
| Qwen-Image-2.0 | May 2026 technical report describes unified generation/editing and stronger VAE research. | Watch until public weights and Diffusers/Transformers loading paths are verified. |
| Z-Image-Edit / Z-Image-Omni-Base | Z-Image card describes editing/omni variants but says they are to be released. | Watch; likely high value once weights ship because Z-Image is already in MLX-Gen. |
| FLUX.1 Kontext | Strong open-weight image editing, Diffusers support, but non-commercial license and overlaps Qwen Edit. | Deprioritize for native MLX-Gen; consider only if user explicitly needs it. |
| GLM-Image | MIT and local Diffusers pipeline, but custom GLM/VLM stack. | Lower priority than ERNIE/Qwen/HiDream unless text-rendering evidence beats them locally. |

## Why it might matter

AbstractVision should expose the best practical local models, not only the models that were easiest
to port first. A watchlist lets MLX-Gen react quickly when a genuinely better permissive model
becomes available without interrupting current release-critical work.

## Promotion criteria

- Public weights are available and loadable without opaque service-only code.
- License permits the intended local and derivative-weight use.
- Upstream Diffusers/Transformers code exists or the original implementation is clear enough to
  port line-by-line to MLX.
- A source model beats current MLX-Gen defaults on a documented local benchmark, or offers a
  capability MLX-Gen lacks.
- The model fits Apple Silicon memory with BF16, q8, mixed q4/q8, or a documented low-bit format.

## Validation ideas

- Local source snapshot size and component inventory.
- One T2I contact sheet and one edit/contact or personalization sheet compared with Qwen
  Image/Edit, ERNIE Turbo, and Z-Image Turbo.
- License and redistribution audit before any AbstractFramework quant publication.
- Upstream PyTorch/Diffusers smoke before MLX port work starts.

## Non-goals

- Do not start HiDream, GLM, or FLUX.1 Kontext implementation from this proposal alone.
- Do not treat Qwen-Image-2.0 as available until actual model weights and loading code are
  verified.
- Do not publish derivatives of gated or non-commercial models without matching upstream terms.

## Guidance for future agents

Re-run the online and local-cache check before promotion. If a model needs a fundamentally
different runtime shape, create an ADR or a dedicated planned item rather than forcing it into the
existing Qwen/FLUX-style backend.

## Sources checked

- Local Diffusers pipelines in `/Users/albou/projects/gh/diffusers/src/diffusers/pipelines/`
- HiDream-O1-Image-Dev model card: https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev
- Qwen-Image-2.0 technical report: https://arxiv.org/abs/2605.10730
- Z-Image-Turbo model card and model zoo: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- FLUX.1 Kontext announcement: https://bfl.ai/blog/flux-1-kontext-dev
