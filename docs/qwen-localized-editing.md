# Qwen Localized Editing

MLX-Gen now ships four different Qwen workflows that are easy to confuse:

- masked edit / inpaint on the Qwen edit route;
- native masked edit on base Qwen rows;
- structured control on the base Qwen route;
- base-Qwen control-inpaint on the exact validated base row.

This page explains the practical difference between them.

If you need the broader Qwen route surface, including `qwen.edit`, `qwen.multi-reference`,
`qwen.reframe`, and `qwen.outpaint`, use [Qwen route matrix](qwen-route-matrix.md). For the
cross-family masked-edit matrix and contract, use [Masked editing](masked-editing.md).

## Current Status

Current exact public proof rows:

- `AbstractFramework/qwen-image-edit-2511-8bit` on `qwen.inpaint` (validated visual QA)
- `AbstractFramework/qwen-image-8bit` on `qwen.control` (validated visual QA)
- `AbstractFramework/qwen-image-8bit` on `qwen.control-inpaint` (validated visual QA)
- `AbstractFramework/qwen-image-4bit` and `AbstractFramework/qwen-image-2512-8bit` on
  `qwen.base-inpaint` (visual smoke, [proof bundle](assets/validation/masked-edit-2026-07-15/README.md))

## Route Matrix

| Workflow | Exact public proof row | Public inputs | Best use | Published proof |
| --- | --- | --- | --- | --- |
| `qwen.inpaint` | `AbstractFramework/qwen-image-edit-2511-8bit` | `--image + --mask-path + --prompt` | Straightforward localized repair on the edit checkpoint | [masked-edit sheet](assets/validation/qwen-inpaint-2026-06-15/qwen2511_q8_inpaint_lightning_contact_sheet.png) |
| `qwen.base-inpaint` | `AbstractFramework/qwen-image-4bit`, `AbstractFramework/qwen-image-2512-8bit` | `--image + --mask-path + --prompt` | Localized edits on base rows without a sidecar or edit checkpoint | [masked-edit expansion sheet](assets/validation/masked-edit-2026-07-15/masked_edit_expansion_contact_sheet.png) |
| `qwen.control` | `AbstractFramework/qwen-image-8bit` | `--controlnet-image-path + --prompt` | Layout-first generation from canny, pose, or similar structure guides | [structured-control sheet](assets/validation/qwen-control-2026-06-15/qwen_q8_control_lightning_contact_sheet.png) |
| `qwen.control-inpaint` | `AbstractFramework/qwen-image-8bit` | `--image + --mask-path + --prompt` | Harder localized repair where the edit route drifts too much | [control-inpaint sheet](assets/validation/qwen-control-inpaint-2026-06-21/qwen_control_inpaint_contact_sheet.png) |

Each base Qwen row carries exactly one masked route: the exact `AbstractFramework/qwen-image-8bit`
row keeps control-inpaint, every other trusted base row runs `qwen.base-inpaint` natively. The
native route warm-starts from the re-noised source (strength `0.85`, upstream
`QwenImageInpaintPipeline` example semantics) and records the executed `effective_steps` in
metadata; prompt it with a caption of the target scene rather than an edit instruction.

## One-Sentence Difference

- **Masked edit / inpaint**: start from an existing image and repaint only the white masked region.
- **Structured control**: generate from text, but force layout from a control image such as canny
  edges or pose.
- **Control-inpaint**: still start from an existing image and a mask, but add a dedicated
  inpainting ControlNet sidecar so the localized replacement is more disciplined.

## What “ControlNet” Means

For MLX-Gen users, the practical definition is simple:

- a **ControlNet** is an extra model package that guides the base image model;
- it is **not** a LoRA;
- it is **not** a replacement base model;
- it is loaded **alongside** the base model for one exact route.

When MLX-Gen docs say **sidecar**, they mean that extra model package.

## Masked Edit / Inpaint On Qwen Image Edit

Current shipped route:

- model family: `Qwen Image Edit`
- exact public proof row: `AbstractFramework/qwen-image-edit-2511-8bit`
- capability id: `qwen.inpaint`

User-facing inputs:

- one source image with `--image`;
- one mask image with `--mask-path`;
- one prompt;
- optional negative prompt;
- optional Lightning adapter for the fast `4`-step path.

Best use:

- straightforward local repairs;
- object-part replacement in one bounded region;
- appearance changes that should stay inside the mask.

Proof:

- [masked-edit contact sheet](assets/validation/qwen-inpaint-2026-06-15/qwen2511_q8_inpaint_lightning_contact_sheet.png)
- [masked-edit command log](assets/validation/qwen-inpaint-2026-06-15/qwen2511_q8_inpaint_lightning_command_log.md)

## Structured Control On Base Qwen

Current shipped route:

- model family: base `Qwen Image`
- exact public proof row: `AbstractFramework/qwen-image-8bit`
- capability id: `qwen.control`
- exact sidecar:
  `InstantX/Qwen-Image-ControlNet-Union:diffusion_pytorch_model.safetensors`

User-facing inputs:

- prompt;
- one control image with `--controlnet-image-path`;
- optional negative prompt;
- optional Lightning adapter for the fast `4`-step path.

Best use:

- enforce a canny/sketch/pose layout;
- generate from text while anchoring geometry;
- layout-first generation rather than source-image repair.

Proof:

- [structured-control contact sheet](assets/validation/qwen-control-2026-06-15/qwen_q8_control_lightning_contact_sheet.png)
- [structured-control command log](assets/validation/qwen-control-2026-06-15/qwen_q8_control_lightning_command_log.md)

## Base-Qwen Control-Inpaint

Current shipped route:

- model family: base `Qwen Image`
- exact public proof row: `AbstractFramework/qwen-image-8bit`
- capability id: `qwen.control-inpaint`
- exact sidecar:
  `InstantX/Qwen-Image-ControlNet-Inpainting:diffusion_pytorch_model.safetensors`

User-facing inputs stay generic:

- one source image with `--image`;
- one mask image with `--mask-path`;
- one prompt;
- optional negative prompt;
- optional `--controlnet-strength` on the exact validated route;
- optional Lightning adapter for the fast `4`-step path.

The route still owns the exact sidecar identity. If you pass `--controlnet-model`, it must match
`InstantX/Qwen-Image-ControlNet-Inpainting:diffusion_pytorch_model.safetensors`.

Best use:

- harder local repairs where the edit route drifts too much;
- stricter mask-boundary behavior on difficult replacements;
- keeping the user request shape the same while switching to a tighter backend.

Proof:

- [control-inpaint contact sheet](assets/validation/qwen-control-inpaint-2026-06-21/qwen_control_inpaint_contact_sheet.png)
- [control-inpaint report](assets/validation/qwen-control-inpaint-2026-06-21/qwen_control_inpaint_report.md)
- [control-inpaint command log](assets/validation/qwen-control-inpaint-2026-06-21/qwen_control_inpaint_command_log.md)

## When To Use Which One

Use `qwen.inpaint` when:

- the edit is straightforward;
- you want the smallest setup;
- the edit checkpoint already keeps the change local enough.

Use `qwen.control` when:

- there is no source frame to preserve;
- the main problem is layout, pose, or edge structure;
- you want text-to-image generation with a guide image.

Use `qwen.control-inpaint` when:

- the request is still “edit this one masked part of an existing image”;
- the plain masked-edit route is not disciplined enough;
- locality matters more than minimal setup.

## Related Docs

- [Qwen route matrix](qwen-route-matrix.md)
- [Image edit modes](image-edit-modes.md)
- [Image edit capabilities](edit-capabilities.md)
- [FAQ](faq.md)
