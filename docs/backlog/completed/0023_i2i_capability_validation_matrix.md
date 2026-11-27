# Planned: I2I capability validation matrix

## Metadata
- Created: 2026-06-05
- Status: Completed
- Completed: 2026-06-05

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None

## Context

MLX-Gen now exposes typed model capabilities for image-to-image modes, but recent manual validation
only proved a subset of the published image-edit surface. A partial contact sheet mixed valid
outputs with a failed Qwen edit attempt, and it did not show every cached model/mode pair.

This item turns I2I validation into an explicit release gate: every published and locally cached
image package that advertises an I2I mode must have a command, source image, prompt, output path,
and visual status. Unsupported or unavailable models must be reported as such instead of silently
omitted.

## Current code reality

- `src/mflux/task_inference.py` defines the semantic capabilities and modes:
  `latent-img2img`, `edit-reference`, `multi-reference`, and mask support.
- `mlxgen capabilities --model <model>` exposes those capabilities without loading weights.
- Current README-published image I2I packages include:
  - `AbstractFramework/flux.2-klein-4b-4bit`
  - `AbstractFramework/flux.2-klein-4b-8bit`
  - `AbstractFramework/flux.2-klein-9b-4bit`
  - `AbstractFramework/flux.2-klein-9b-8bit`
  - `AbstractFramework/qwen-image-2512-4bit`
  - `AbstractFramework/qwen-image-2512-8bit`
  - `AbstractFramework/qwen-image-edit-2511-4bit`
  - `AbstractFramework/qwen-image-edit-2511-8bit`
  - `AbstractFramework/z-image-turbo-4bit`
  - `AbstractFramework/z-image-turbo-8bit`
  - `AbstractFramework/ernie-image-turbo-4bit`
  - `AbstractFramework/ernie-image-turbo-8bit`
- Local Hugging Face cache currently has complete snapshots for those twelve AbstractFramework
  packages.
- `fibo-edit` previously exposed an edit-reference capability with mask support through unified
  `mlxgen generate`; local validation downloaded `briaai/Fibo-Edit` and found the edit path failed
  visual QA, so unified capability discovery now fails closed for FIBO Edit until a dedicated fix
  validates it.
- `validation_outputs/i2i_matrix_2026_06_04/final_after_fix/` has a partial prior matrix that
  covers some but not all model/mode pairs.
- `validation_outputs/i2i_8bit_validation_2026_06_05/qwen_edit_2511_8bit_crash.png` is a failed
  single-image Qwen edit proof: the object dissolves into a noisy mesh instead of preserving a
  solid spaceship.

## Problem

The project can currently report I2I capabilities that have not all been proven with model-backed
outputs in one complete, inspectable table. That weakens release confidence and makes it too easy
to treat one or two successful examples as coverage for every model and mode.

## What we want to do

Build a complete I2I validation matrix for the cached published image packages and any additional
supported I2I family that can be made locally available. Each row must state the model, advertised
capability, exact command inputs, prompt, result image, and manual status.

## Why

I2I modes are not interchangeable. Latent img2img uses `--image-strength`, edit-reference uses
instruction/reference conditioning, and multi-reference uses more than one image. The release gate
must prove the actual model/mode contract, not only that a route exists.

## Requirements

- Derive the candidate list from current capabilities and published package docs, not memory.
- Test every advertised I2I mode for every locally cached published image package.
- Keep generation serial; do not run heavyweight image models in parallel.
- Use one stable spaceship-in-snow source image unless a mode explicitly requires another input.
- Use mode-appropriate prompts:
  - latent: whole-image restyle or lighting variation with `--image-strength`;
  - edit-reference: instruction edit without `--image-strength`;
  - multi-reference: two image inputs and a prompt that uses both references;
  - mask-capable edit: source plus mask if the model is locally available.
- Reject outputs that dissolve the main subject, ignore the requested mode, stretch the source,
  produce blank/noisy blobs, or lose the basic spaceship-in-snow scene.
- Produce a table with model, capability, source image(s), prompt, result image, and status.
- Preserve failed attempts separately when they explain a prompt/model weakness.

## Suggested implementation

1. Generate a machine-readable capability and cache inventory.
2. Reuse prior accepted outputs only after visual inspection; regenerate missing or weak rows.
3. Retry failed rows with stronger prompts or proper default step counts before marking a model
   capability as weak or failed.
4. Build a final contact sheet from accepted outputs plus a manifest/table.
5. Run focused routing/parser tests after any code changes.

## Scope

- Cached AbstractFramework image I2I packages.
- `fibo-edit` availability and masked edit validation if the model can be downloaded without
  access failure.
- Local proof assets under `validation_outputs/`.
- Focused code fixes only if validation exposes resolver, parser, routing, dimension, or metadata
  bugs.

## Non-goals

- Do not test video models here.
- Do not claim outpainting/reframing from this historical matrix; item 0019 tracks the current
  reframe and outpaint UX.
- Do not publish validation images automatically.
- Do not run unrelated release steps.

## Dependencies and related tasks

- [0018](../completed/0018_taskless_generation_routing.md): taskless routing.
- [0020](../completed/0020_generation_capability_contract.md): capability contract and route planning.
- [0022](../completed/0022_i2i_source_aspect_ratio_policy.md): source-aspect canvas policy.
- [0019](0019_first_class_i2i_modes_and_outpaint_reframe.md): future outpaint/reframe work.

## Expected outcomes

- A complete I2I capability proof table exists for all relevant image models.
- Every accepted row has a real output image and visual inspection status.
- Failed or unavailable capabilities are explicit with reason and next action.
- The Qwen edit crash failure is either replaced by a passing prompt/settings row or recorded as a
  failed complex-object-edit attempt.

## Validation

- `mlxgen capabilities` inventory for every candidate model.
- HF cache inventory for every candidate model.
- Model-backed generation commands for every required mode.
- Manual image inspection and contact sheet.
- Focused tests:
  - parser/router/task inference tests after code changes;
  - lint on touched code and tests.

## Progress checklist
- [x] Create candidate capability/cache inventory.
- [x] Generate or inspect every model/mode proof row.
- [x] Retry weak rows with better prompts/settings.
- [x] Build final table and contact sheet.
- [x] Apply code fixes only if validation exposes implementation bugs.
- [x] Run focused tests.
- [x] Move item to completed with final evidence if all required rows are resolved.

## Completion evidence

- Passing proof table:
  `validation_outputs/i2i_full_matrix_2026_06_05/i2i_full_matrix_manifest.md`.
- Passing contact sheet:
  `validation_outputs/i2i_full_matrix_2026_06_05/i2i_full_matrix_pass_contact_sheet.jpg`.
- Failed-attempt contact sheet:
  `validation_outputs/i2i_full_matrix_2026_06_05/i2i_full_matrix_failed_attempts_contact_sheet.jpg`.
- Originally accepted rows: 22 installed release-validation model/mode rows across ERNIE Image
  Turbo q4/q8 latent I2I, FLUX.2 Klein 4B/9B q4/q8 latent/edit/multi-reference I2I, Qwen Image
  2512 q4/q8 latent I2I, Qwen Image Edit 2511 q4/q8 edit/multi-reference I2I, and Z-Image Turbo
  q4/q8 latent I2I. The correction report below supersedes this count for complex-edit
  release-readiness.
- Failed rows kept visible: the original Qwen Edit 2511 q8 crash prompt dissolved the spaceship
  into a noisy mesh. Constrained Qwen hard-landing prompts were initially accepted, but are now
  marked unverified because they do not prove the requested complex crash edit.
- FIBO outcome in this historical matrix: `briaai/Fibo-Edit` was downloaded and tested with masked
  minimal JSON, unmasked minimal JSON, full-schema JSON at 432x240, and full-schema JSON at
  640x352. All failed visual QA in this matrix. Later items 0024 and 0026 supersede the routing
  state: one-image unmasked FIBO Edit is route-available through unified `mlxgen generate`, but it
  still failed the standardized release-quality sequence in item 0026.
- Unavailable rows: `fibo-edit-rmbg` refused generation before weight load because
  `briaai/Fibo-Edit-RMBG` is not installed; historical cache-shell repos with no local model bytes
  are listed in the manifest and are not counted as passing validation.
- Focused validation passed for the then-current task inference and router FIBO behavior.

## Correction report - 2026-06-05

This completion record is historical evidence, not the final release-quality I2I proof matrix.
Manual review after closure found that the matrix overstates complex edit coverage:

- `AbstractFramework/qwen-image-edit-2511-8bit` pencil/style rows remain useful evidence, but the
  hard-landing/crash behavior is not proven. The original
  `validation_outputs/i2i_8bit_validation_2026_06_05/qwen_edit_2511_8bit_crash.png` failed visual
  QA, and the later constrained "minor hard landing" rows do not prove the requested complex
  crash edit.
- The matrix mixed generated references from different runs and did not define a single canonical
  source, cinematic reference, crash reference, pencil/crash reference, and composition sequence.
  That makes model-to-model comparison weaker than required.
- Historical routing note: this correction originally moved FIBO Edit repair into
  [0024](../planned/0024_fibo_edit_unified_i2i_validation.md). Later item
  [0026](0026_edit_model_prepared_capability_contact_sheets.md) supersedes that routing note:
  one-image unmasked FIBO Edit is route-available, but source/BF16/q8 packages did not pass the
  standardized release-quality sequence.

Follow-ups [0025](0025_standardized_i2i_sequence_validation.md) and
[0026](0026_edit_model_prepared_capability_contact_sheets.md) now supersede this item
as the release gate for comparable I2I model/mode proof. Future release notes must not cite the
0023 "22 passing rows" as complete complex-edit validation.

### Post-completion update - 2026-06-05

Planned item [0024](../planned/0024_fibo_edit_unified_i2i_validation.md) later re-enabled the
narrow plain one-image unmasked `fibo-edit` route after q8 and default-weight unified visual proof.
The historical FIBO failures in this item remain useful evidence for why masked FIBO Edit and RMBG
are still not public unified capabilities.

## Implementation notes

Do not equate "route exists" with "model works." The deliverable is the evidence table: model,
capability, source image(s), prompt, result image, and status. Keep unsuccessful attempts visible
when they reveal a real limitation.
