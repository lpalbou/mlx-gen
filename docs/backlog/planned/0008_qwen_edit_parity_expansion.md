# Planned: Qwen edit parity expansion

## Metadata

- Created: 2026-05-28
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None if Qwen control, masked edit, and structured conditioning remain task-specific
  capability metadata under the existing generation contract.

## Context

Qwen Image/Edit is currently the best strategic image lane for MLX-Gen: Apache 2.0, already ported,
already quantized with a working mixed q4/q8 policy, and directly relevant to AbstractVision image
editing. Online evidence as of 2026-05-28 makes Qwen-Image-Edit-2511 especially important: the
model card describes better consistency, lower drift, improved character preservation, industrial
design gains, geometric reasoning, and integrated LoRA capabilities.

Update 2026-06-05: standardized local I2I sequence validation in
[completed item 0025](../completed/0025_standardized_i2i_sequence_validation.md) found that
`AbstractFramework/qwen-image-edit-2511-4bit` passed only the cinematic edit row and failed complex
crash, pencil-crash, and composition rows. `AbstractFramework/qwen-image-edit-2511-8bit` passed
cinematic and crash rows, but only partially handled pencil-crash and failed to carry crash/debris
state into multi-reference composition. That makes Qwen edit parity a concrete follow-up rather
than a vague expansion idea.

Update 2026-06-11: this item should no longer sit in `proposed/`. The local Diffusers checkout has
real Qwen control and masked-edit pipelines, the official
[`Qwen/Qwen-Image-Edit-2509`](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) card explicitly
advertises native ControlNet conditions such as depth, edge maps, keypoints, and sketches, and
public control weights now exist for current Qwen families. That makes structured Qwen control a
concrete missing capability inside an already-supported family rather than a speculative idea.

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
- The local Diffusers Qwen control pipelines already expose `control_image` inputs, multi-control
  batching, and dedicated inpaint/controlnet handling.
- Public upstream control weights now exist and are practical validation candidates:
  - `InstantX/Qwen-Image-ControlNet-Union`
  - `InstantX/Qwen-Image-ControlNet-Inpainting`
  - `alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union`
- MLX-Gen docs and model cards already emphasize Qwen mixed q4/q8 because full q4 quality was not
  good enough.
- MLX-Gen does not currently expose first-class Qwen mask inputs, control images, control types,
  or Qwen ControlNet package resolution through `mlxgen generate`.
- There is legacy mask/control plumbing in the inherited FLUX.1 command surface, but it is not
  wired into current unified Qwen routing.

## Problem

MLX-Gen currently has the core Qwen generation/edit path, but the broader Diffusers Qwen feature
surface is richer. The highest-value missing pieces are not random new models; they are adjacent
Qwen edit modes that users naturally expect once they can edit images:

- inpainting and masked edit;
- edit-plus / multi-image behavior;
- layered composition;
- structured control and ControlNet-inpaint with depth, edge, pose, sketch, and related
  condition maps;
- LoRA validation for 2511 workflows.

## What we want to do

Make Qwen parity the next serious image-edit expansion after current LoRA truthfulness work:

1. Create a Qwen feature matrix comparing MLX-Gen to the local Diffusers Qwen pipelines.
2. Add tests that lock current Qwen route decisions: text-to-image, img2img, single-image edit,
   multi-image edit, and explicit task override.
3. Port one adjacent feature family at a time, starting with masked edit or structured control on
   the Qwen route that has the cleanest public weights and strongest visible effect.
4. Keep Qwen-Image-Edit-2509 and Qwen-Image-Edit-2511 as the practical edit baselines for new
   Qwen control work, and use Qwen Image 2512 only where the public control weights clearly target
   that generation route.
5. Validate BF16/q8/mixed q4/q8 only for the exact rows that are actually claimed.

## Why

Qwen is permissively licensed and already works in MLX-Gen. Improving it likely gives more user
value per engineering hour than starting another large image model port. It also avoids depending
on non-commercial FLUX.1 Kontext for high-quality local editing.

## Requirements

- Keep Qwen structured control and masked edit fail-closed until exact model, route, and control
  weights are proven.
- Do not auto-generate control images silently. If MLX-Gen later offers helper generation for
  canny/depth/pose, it must be explicit in the CLI and metadata.
- Keep capability metadata honest: structured control, masked edit, and plain edit are not the
  same route even if they share some transformer weights.
- Prefer one public control family first, for example a union ControlNet or dedicated inpainting
  ControlNet, before adding multiple condition-specific routes.
- Preserve exact package identity. A request for a Qwen edit route with ControlNet must not fall
  back silently to latent img2img or a different Qwen family.
- Publish proof rows with contact sheets, prompts, control inputs, and command logs before
  claiming support in docs or capabilities.

## Suggested implementation

1. Write a Qwen capability matrix that separates:
   - plain text-to-image;
   - latent img2img;
   - single-image edit;
   - multi-reference edit;
   - masked edit/inpaint;
   - structured control;
   - structured control + inpaint.
2. Start with the local Diffusers control or inpaint path that has the smallest new surface area
   and the clearest public weights.
3. Reuse the existing Qwen transformer, tokenizer, scheduler, and LoRA strictness patterns before
   adding new abstractions.
4. Extend unified `mlxgen generate` only after the underlying Qwen route is proven. Capability
   output should expose the exact control/mask directions the selected row supports.
5. Add focused parity fixtures only where a math mismatch appears; do not start with large
   full-generation comparisons if the route wiring is obviously incomplete.

## Scope

- Qwen edit/control/inpaint parity for currently supported Qwen families.
- Structured-control route surfacing, mask/control inputs, and exact validation rows.
- Documentation and capability updates once routes are proven.

## Non-goals

- Do not port every Qwen pipeline in one pass.
- Do not make Qwen generation auto-download models, ControlNets, or LoRAs.
- Do not promote Qwen-Image-2.0 until public weights and Diffusers/Transformers loading paths are
  verified.
- Do not let Qwen-specific naming leak into the generic `mlxgen` contract unless the same concept
  is useful across families.

## Dependencies and related tasks

- [0007 LoRA capability matrix and strict application](0007_lora_capability_matrix_and_strict_application.md)
- [0019 First-class I2I modes and outpaint/reframe UX](0019_first_class_i2i_modes_and_outpaint_reframe.md)
- `src/mflux/models/qwen/`
- `src/mflux/task_inference.py`
- `src/mflux/cli/mlx_gen.py`
- `/Users/albou/projects/gh/diffusers/src/diffusers/pipelines/qwenimage/`

## Expected outcomes

- A clear Qwen feature matrix with implemented vs unsupported rows.
- At least one validated structured-control or masked-edit Qwen route with public proof artifacts.
- Capability output and docs that tell AbstractVision exactly when Qwen can expose control/mask
  inputs.

## Validation

- Same prompt, seed, and dimensions across no-control vs controlled runs.
- Visible contact sheets for at least one structure-driven example and one masked-edit example.
- Exact model-handle or prepared-package proof rows only for the combinations actually tested.
- Focused `uv run pytest` coverage for route resolution, control/mask argument validation, and any
  parity or loader helpers added during implementation.

## Progress checklist

- [ ] Write the Qwen feature matrix against local Diffusers pipelines.
- [ ] Decide the first public Qwen control target and its proof weights.
- [ ] Implement strict route selection and capability surfacing for that target.
- [ ] Validate source and q8 rows with visible contact sheets.
- [ ] Decide whether masked edit/inpaint should be the second Qwen expansion or stay separate.

## Guidance for the implementing agent

Favor narrow, reviewable Qwen parity PRs. Start with the strongest upstream evidence and one
public proof adapter family. If adding masks or structured controls requires new generic CLI/API
concepts, document the generic `mlxgen` contract first so it does not become Qwen-specific
vocabulary.

## Sources checked

- `src/mflux/models/qwen/`
- `src/mflux/cli/mlx_gen.py`
- Local Diffusers checkout Qwen pipelines under `diffusers/src/diffusers/pipelines/qwenimage/`
- Qwen-Image-Edit-2509 model card: https://huggingface.co/Qwen/Qwen-Image-Edit-2509
- Qwen-Image-Edit-2511 model card: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- InstantX Qwen control weights: https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union
- InstantX Qwen inpainting ControlNet: https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting
- Alibaba PAI Qwen 2512 union ControlNet: https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union
- Qwen-Image-2.0 technical report watch item: https://arxiv.org/abs/2605.10730
