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

The June 2026 open image landscape is moving quickly. MLX-Gen should keep shipping Qwen, FLUX.2,
Z-Image, ERNIE, and Bonsai support, but it also needs a watchlist for models that could change the
best local Apple Silicon strategy.

The strongest external watch candidates right now are:

- HiDream-O1-Image, because it is MIT-licensed, supports text-to-image, editing, and
  subject-driven personalization, and uses a pixel-level unified transformer rather than the usual
  separate VAE/text-encoder stack.
- Step1X-Edit-v1p2, because it is Apache 2.0, openly published, and pushes hard on reasoning-heavy
  image editing inside a Diffusers-friendly public surface.
- JoyAI-Image-Edit, because it is Apache 2.0, ships a dedicated Diffusers repo, and targets the
  same direct edit space that AbstractVision cares about.
- OmniGen2, because it is Apache 2.0, Diffusers-ready, and tries to unify text-to-image, editing,
  and in-context multimodal generation behind one public model family.
- Ideogram 4, because open-weight fp8/nf4 checkpoints, Diffusers support, strong text rendering,
  and structured JSON prompting make it strategically interesting even though the current license is
  non-commercial and gated.
- Ovis-Image 7B, because it is a compact text-to-image route optimized for strong text rendering
  under tighter compute budgets.
- PRXPixel, because it is a public pixel-space text-to-image architecture with no VAE and a
  Qwen3-VL text encoder, making it a strong "different runtime shape" watch candidate rather than
  just another diffusion-family clone.
- DreamLite, because the current public Diffusers surface already exposes both a full pipeline and
  a distilled mobile pipeline for text-to-image and editing.

Qwen-Image-2.0 remains worth tracking because the technical report describes a stronger unified
generation/editing path, but it still needs public weights and loading code before it becomes
implementation work. Z-Image-Edit and Z-Image-Omni-Base also remain worth tracking: the Z-Image
docs still describe them as not yet released.

## Current code reality

- MLX-Gen already supports Qwen Image/Edit, FLUX.2 Klein, Z-Image/Z-Image-Turbo, ERNIE Image
  Turbo, Bonsai ternary, and FIBO family routing.
- Local Diffusers includes `hidream_image`, `glm_image`, `flux`, `flux2`, `qwenimage`,
  `ideogram4`, `dreamlite`, and `z_image` pipelines.
- `ModelConfig` does not include HiDream-O1, Qwen-Image-2.0, GLM-Image, Step1X-Edit, JoyAI,
  OmniGen2, Ideogram 4, Ovis-Image, PRXPixel, DreamLite, or Z-Image-Edit/Omni as first-class
  MLX-Gen targets.
- MLX-Gen already has explicit FLUX.1 Kontext code and config surfaces, so Kontext is no longer a
  missing-implementation watch item. It is now mainly a product-positioning and license question.
- HiDream-O1 likely needs an MLX-VLM/pixel-space design review rather than a small weight-mapping
  addition.
- Ideogram 4 and PRXPixel also imply runtime-shape work beyond MLX-Gen's current VAE plus
  diffusion-transformer norm.

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
| HiDream-O1-Image / Dev | MIT, 8-9B class, text-to-image, editing, subject-driven personalization, Transformers loading, local Diffusers pipeline. | Research after Qwen parity and ERNIE non-turbo validation. |
| Step1X-Edit-v1p2 | Apache 2.0, public Diffusers usage, explicit image-edit focus, reasoning-heavy edit framing. | Watch; likely worth a focused upstream smoke before any MLX port work. |
| JoyAI-Image-Edit / Diffusers | Apache 2.0, dedicated edit family, public Diffusers repo, relevant to direct instruction editing. | Watch; compare against Qwen 2511 and Step1X before promoting. |
| OmniGen2 | Apache 2.0, Any-to-Any positioning, public Diffusers route, strong in-context multimodal claim. | Watch; only promote if MLX-Gen decides the multimodal/edit opportunity is worth a new runtime shape. |
| Qwen-Image-2.0 | May 2026 technical report describes unified generation/editing and stronger VAE research. | Watch until public weights and Diffusers/Transformers loading paths are verified. |
| Z-Image-Edit / Z-Image-Omni-Base | Z-Image docs describe editing/omni variants but say they are not yet released. | Watch; likely high value once weights ship because Z-Image is already in MLX-Gen. |
| FLUX.1 Kontext | Already present locally in MLX-Gen, but gated/non-commercial and partly overlaps Qwen edit goals. | Keep as a positioning/licensing watch item, not a missing-core-model item. |
| GLM-Image | MIT and local Diffusers pipeline, but custom GLM/VLM stack. | Lower priority than ERNIE/Qwen/HiDream unless text-rendering evidence beats them locally. |
| Ideogram 4 | Open weights, Diffusers support, strong text/layout control, structured JSON prompting. | Watch only; current Ideogram Non-Commercial gate and a different prompt contract make it a product-positioning question before implementation. |
| Ovis-Image 7B | Compact 7B text-to-image, text-rendering focus, public weights. | Watch; potentially strong local text-rendering route, but needs real quality comparison against Qwen/Z-Image/Ideogram before promotion. |
| PRXPixel | Apache 2.0, public pixel-space no-VAE text-to-image pipeline, different runtime shape. | Watch; likely needs an ADR because it is not just another VAE plus transformer port. |
| DreamLite | Public text-to-image plus editing surface, including a distilled mobile route. | Watch; interesting for low-latency local editing, but still lower priority than finishing Qwen and Z-Image adjacent work. |

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

- Do not start HiDream, GLM, Ideogram 4, PRXPixel, or FLUX.1 Kontext implementation from this
  proposal alone.
- Do not treat Qwen-Image-2.0 as available until actual model weights and loading code are
  verified.
- Do not publish derivatives of gated or non-commercial models without matching upstream terms.

## Implementation notes

Re-run the online and local-cache check before promotion. If a model needs a fundamentally
different runtime shape, create an ADR or a dedicated planned item rather than forcing it into the
existing Qwen/FLUX-style backend.

## Sources checked

- Local Diffusers checkout pipelines under `diffusers/src/diffusers/pipelines/`
- HiDream-O1-Image-Dev model card: https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev
- HiDream-O1-Image model card: https://huggingface.co/HiDream-ai/HiDream-O1-Image
- Step1X-Edit-v1p2 model card: https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2
- JoyAI-Image-Edit-Diffusers model card: https://huggingface.co/jdopensource/JoyAI-Image-Edit-Diffusers
- JoyAI-Image-Edit base model: https://huggingface.co/jdopensource/JoyAI-Image-Edit
- OmniGen2 model card: https://huggingface.co/OmniGen2/OmniGen2
- Qwen-Image-2.0 technical report: https://arxiv.org/abs/2605.10730
- Z-Image-Turbo model card and model zoo: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- FLUX.1 Kontext announcement: https://bfl.ai/blog/flux-1-kontext-dev
- Ideogram 4 model card: https://huggingface.co/ideogram-ai/ideogram-4-nf4
- Ideogram 4 license: https://huggingface.co/ideogram-ai/ideogram-4-fp8/blob/main/LICENSE.md
- Diffusers Ideogram 4 docs: https://huggingface.co/docs/diffusers/main/api/pipelines/ideogram4
- Ovis-Image-7B model card: https://huggingface.co/AIDC-AI/Ovis-Image-7B
- PRXPixel model card: https://huggingface.co/Photoroom/prxpixel-t2i
- Diffusers PRXPixel docs: https://huggingface.co/docs/diffusers/main/api/pipelines/prx_pixel
- Diffusers DreamLite docs: https://huggingface.co/docs/diffusers/main/api/pipelines/dreamlite
