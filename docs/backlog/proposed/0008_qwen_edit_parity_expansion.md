# Proposed: Qwen edit parity expansion

## Metadata

- Created: 2026-05-28
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: None unless Qwen edit support becomes a separate plugin boundary.

## Context

Qwen Image/Edit is currently the best strategic image lane for MLX-Gen: Apache 2.0, already ported,
already quantized with a working mixed q4/q8 policy, and directly relevant to AbstractVision image
editing. Online evidence as of 2026-05-28 makes Qwen-Image-Edit-2511 especially important: the
model card describes better consistency, lower drift, improved character preservation, industrial
design gains, geometric reasoning, and integrated LoRA capabilities.

The local Diffusers checkout has broader Qwen pipelines than MLX-Gen currently exposes.

## Current code reality

- MLX-Gen has Qwen text-to-image and image-edit variants under `src/mflux/models/qwen/`.
- `ModelConfig` resolves Qwen Image, Qwen Image Edit 2509, and Qwen Image Edit 2511.
- `mlxgen generate` routes Qwen edit when the model is an edit model, the task is `edit`, or
  multiple input images are present without explicit img2img.
- The local Diffusers checkout includes:
  - `pipeline_qwenimage.py`
  - `pipeline_qwenimage_img2img.py`
  - `pipeline_qwenimage_edit.py`
  - `pipeline_qwenimage_edit_plus.py`
  - `pipeline_qwenimage_edit_inpaint.py`
  - `pipeline_qwenimage_inpaint.py`
  - `pipeline_qwenimage_layered.py`
  - `pipeline_qwenimage_controlnet.py`
  - `pipeline_qwenimage_controlnet_inpaint.py`
- MLX-Gen docs and model cards already emphasize Qwen mixed q4/q8 because full q4 quality was not
  good enough.

## Problem or opportunity

MLX-Gen currently has the core Qwen generation/edit path, but the broader Diffusers Qwen feature
surface is richer. The highest-value missing pieces are not random new models; they are adjacent
Qwen edit modes that users naturally expect once they can edit images:

- inpainting and masked edit;
- edit-plus / multi-image behavior;
- layered composition;
- ControlNet and ControlNet-inpaint, if public Qwen ControlNet weights are stable enough;
- LoRA validation for 2511 workflows.

## Proposed direction

Promote Qwen parity as the next image-edit expansion after current Wan and LoRA-truthfulness work:

1. Create a Qwen feature matrix comparing MLX-Gen to the local Diffusers Qwen pipelines.
2. Add tests that lock current Qwen route decisions: text-to-image, img2img, single-image edit,
   multi-image edit, and explicit task override.
3. Port one adjacent feature at a time, starting with masked edit/inpaint because it is a common
   AbstractVision requirement and fits existing edit semantics.
4. Keep Qwen-Image-Edit-2511 as the quality baseline for new edit features.
5. Validate BF16/q8/mixed q4/q8 with the same prompt/image set and publish contact sheets in docs.

## Why it might matter

Qwen is permissively licensed and already works in MLX-Gen. Improving it likely gives more user
value per engineering hour than starting another large image model port. It also avoids depending
on non-commercial FLUX.1 Kontext for high-quality local editing.

## Promotion criteria

- AbstractVision needs mask/inpaint or multi-image editing on Apple Silicon.
- Users begin publishing Qwen-Image-Edit-2511 LoRA workflows that MLX-Gen cannot run or validate.
- A Diffusers Qwen pipeline becomes the de facto upstream path for a feature that MLX-Gen lacks.

## Validation ideas

- Same input image, prompt, seed, resolution, and steps across BF16, q8, and mixed q4/q8.
- Masked inpaint test with visible before/after and mask overlay.
- Multi-image edit test preserving identity and layout.
- LoRA test with a known public Qwen-Image-Edit-2511 adapter.
- Diffusers-vs-MLX focused parity tests for prompt encoding, latent packing, and one short denoise
  replay before full generation validation.

## Non-goals

- Do not port every Qwen pipeline in one pass.
- Do not make Qwen generation auto-download models or LoRAs.
- Do not promote Qwen-Image-2.0 until public weights and Diffusers/Transformers loading paths are
  verified.
- Do not use FLUX.1 Kontext license terms as a reason to weaken Qwen validation.

## Guidance for future agents

Favor narrow, reviewable Qwen parity PRs. If adding a feature requires new public CLI concepts
such as masks, layers, or multiple condition groups, document the generic `mlxgen` API first so it
does not become Qwen-specific vocabulary.

## Sources checked

- `src/mflux/models/qwen/`
- `src/mflux/cli/mlx_gen.py`
- Local Diffusers Qwen pipelines in `/Users/albou/projects/gh/diffusers/src/diffusers/pipelines/qwenimage/`
- Qwen-Image-Edit-2511 model card: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- Qwen-Image-2.0 technical report watch item: https://arxiv.org/abs/2605.10730
