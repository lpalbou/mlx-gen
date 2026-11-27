# Completed: Edit model prepared-package capability contact sheets

## Metadata
- Created: 2026-06-05
- Status: Completed
- Completed: 2026-06-05

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None

## Context

Recent FIBO Edit validation was too narrow. It used the short alias `fibo-edit` and runtime
`--quantize 8` instead of the true upstream Hugging Face handle or a prepared q8 package, and it
proved only one sketch-style edit. That is not enough evidence for release-quality edit support.

Edit-capable image models expose different capabilities:

| Family | Source HF handle | Prepared q8 handle or path | Public modes to prove |
| --- | --- | --- | --- |
| FIBO Edit | `briaai/Fibo-Edit` | to prepare locally, then use the prepared model path | `edit-reference` |
| Qwen Image Edit 2509 | `Qwen/Qwen-Image-Edit-2509` | `AbstractFramework/qwen-image-edit-2509-8bit` | `edit-reference`, `multi-reference` |
| Qwen Image Edit 2511 | `Qwen/Qwen-Image-Edit-2511` | `AbstractFramework/qwen-image-edit-2511-8bit` | `edit-reference`, `multi-reference` |
| FLUX.2 Klein 4B | `black-forest-labs/FLUX.2-klein-4B` | `AbstractFramework/flux.2-klein-4b-8bit` | `latent-img2img`, `edit-reference`, `multi-reference` |
| FLUX.2 Klein 9B | `black-forest-labs/FLUX.2-klein-9B` | `AbstractFramework/flux.2-klein-9b-8bit` | `latent-img2img`, `edit-reference`, `multi-reference` |

The local Hugging Face cache now contains the source snapshots and prepared q8 packages above.
FIBO Edit BF16/q8 folders were prepared locally as `models/fibo-edit-bf16` and
`models/fibo-edit-8bit`; `models/fibo-4bit` and `models/fibo-8bit` remain base FIBO folders, not
FIBO Edit folders.

## Current code reality

- `src/mflux/task_inference.py` exposes:
  - `fibo.edit` as one-image `edit-reference`;
  - `qwen.edit` and `qwen.multi-reference` for Qwen edit checkpoints;
  - `flux2.latent`, `flux2.edit`, and `flux2.multi-reference` for FLUX.2 Klein.
- `mlxgen generate` accepts true source handles and prepared derivative handles/paths.
- `mlxgen prepare` now routes FIBO Edit/RMBG names through `FIBOEdit`.
- Prior FIBO q8 evidence used runtime `--quantize 8`; that is rejected as release evidence for
  this item because q8 validation must use a prepared q8 package.
- Prior Qwen/FLUX standardized matrices are useful historical evidence, but this item requires
  contact sheets grouped per edit model and per advertised edit capability.

## Problem

Users need to know whether a selected edit model and prepared q8 package can actually perform the
range of edits it advertises. A single sketch output does not prove instruction editing,
multi-reference composition, or latent I2I, and runtime quantization does not prove a prepared q8
package.

## What we want to do

Create model-backed contact sheets for each edit-capable family/package using true HF source
handles for source validation and prepared q8 handles or local prepared folders for quantized
validation.

## Requirements

- Use the true upstream HF handle for source runs.
- Use an already-prepared q8 model folder or HF repo for q8 runs; do not use runtime
  `--quantize 8` as q8 release evidence.
- Prepare missing FIBO Edit BF16/q8 folders before validation.
- For each model family, build a contact sheet that includes every mode it advertises:
  - `latent-img2img`: cinematic restyle with `--image-strength`;
  - `edit-reference`: at minimum pencil sketch and hard landing / crash;
  - `multi-reference`: composition using a crash/pencil reference plus a cinematic reference.
- Record source image path, reference image paths, prompt, exact command, output path, model
  handle/path, precision/package, and visual status for every row.
- Keep generation serial.
- Mark failed rows honestly instead of tuning them into a hidden success.

## Scope

- FIBO Edit, Qwen Image Edit 2509/2511, and FLUX.2 Klein 4B/9B.
- Source and q8 prepared package validation.
- Local validation outputs and contact sheets under `validation_outputs/`.
- Focused code fixes only when validation exposes a real routing, prepare, metadata, or capability
  bug.

## Non-goals

- Do not publish new prepared models automatically unless explicitly requested.
- Do not validate q4 in the initial pass; if a published q4 edit package is already present and
  claimed in docs, add it to the post-review matrix instead of implying untested coverage.
- Do not claim outpainting/reframing; item 0019 tracks first-class fill/outpaint UX.
- Do not treat FIBO mask/RMBG as unified public capabilities unless they receive separate passing
  proof and a clear contract.

## Expected outcomes

- FIBO Edit has prepared BF16/q8 folders or a documented blocker.
- Every edit-capable family has a contact sheet showing all advertised edit capabilities for the
  packages being claimed.
- The final report distinguishes source handles from prepared q8 handles/paths.
- Documentation and backlog no longer cite alias/runtime-q8 sketch-only proof as sufficient.

## Validation

- `mlxgen capabilities --model ...` inventory for every source and q8 package.
- `mlxgen prepare` commands for missing FIBO Edit packages.
- Serial `mlxgen generate` commands for every mode row.
- Manual visual inspection of contact sheets.
- Focused router/prepare tests after code changes.
- `git diff --check` and `ruff check` on touched code/docs.

## Progress checklist
- [x] Prepare missing FIBO Edit BF16/q8 packages.
- [x] Generate FIBO Edit source and prepared q8 contact-sheet rows.
- [x] Generate Qwen Image Edit 2509 source and q8 contact-sheet rows.
- [x] Generate Qwen Image Edit 2511 source and q8 contact-sheet rows.
- [x] Generate FLUX.2 Klein 4B source and q8 contact-sheet rows.
- [x] Generate FLUX.2 Klein 9B source and q8 contact-sheet rows.
- [x] Build per-model contact sheets and one summary manifest.
- [x] Review outputs and update docs/backlog with pass/fail status.

## Result

Proof assets:

- Primary 5x4 review matrix:
  `validation_outputs/edit_prepared_capability_2026_06_05/edit_capability_matrix_5x4_contact_sheet.jpg`
- Primary 5x4 manifest:
  `validation_outputs/edit_prepared_capability_2026_06_05/edit_capability_matrix_5x4_manifest.md`
- Source-handle 5x4 review matrix:
  `validation_outputs/edit_prepared_capability_2026_06_05/edit_base_model_capability_matrix_5x4_contact_sheet.jpg`
- Source-handle 5x4 manifest:
  `validation_outputs/edit_prepared_capability_2026_06_05/edit_base_model_capability_matrix_5x4_manifest.md`
- Manifest with exact commands:
  `validation_outputs/edit_prepared_capability_2026_06_05/edit_capability_manifest.md`
- Summary sheet:
  `validation_outputs/edit_prepared_capability_2026_06_05/edit_capability_summary_contact_sheet.jpg`
- Per-model sheets:
  - `validation_outputs/edit_prepared_capability_2026_06_05/fibo_edit_variant_matrix_contact_sheet.jpg`
  - `validation_outputs/edit_prepared_capability_2026_06_05/flux2_klein_4b_variant_matrix_contact_sheet.jpg`
  - `validation_outputs/edit_prepared_capability_2026_06_05/flux2_klein_9b_variant_matrix_contact_sheet.jpg`
  - `validation_outputs/edit_prepared_capability_2026_06_05/qwen_image_edit_2509_variant_matrix_contact_sheet.jpg`
  - `validation_outputs/edit_prepared_capability_2026_06_05/qwen_image_edit_2511_variant_matrix_contact_sheet.jpg`

Manual visual status:

| Family | Package | Tested capabilities | Status | Notes |
| --- | --- | --- | --- | --- |
| FIBO Edit | `briaai/Fibo-Edit` source | `edit-reference` | FAIL | Route executes, but source-handle rerun after prompt/BOT, final-bias, mask-broadcast, FP16-clipping, and NoPE/RoPE parity fixes still overexposes the pencil/crash row and does not satisfy the edit. |
| FIBO Edit | `models/fibo-edit-bf16` | `edit-reference` | STALE | Historical prepared-folder output only; the folder predates `norm_out.linear.bias` support and is rejected by current code. |
| FIBO Edit | `models/fibo-edit-8bit` | `edit-reference` | STALE | Historical prepared-folder output only; the folder predates `norm_out.linear.bias` support and is rejected by current code. |
| FLUX.2 Klein 4B | `black-forest-labs/FLUX.2-klein-4B` source | `latent-img2img`, `edit-reference`, `multi-reference` | PASS | Full B/C/D/E sequence passed. |
| FLUX.2 Klein 4B | `AbstractFramework/flux.2-klein-4b-8bit` | `latent-img2img`, `edit-reference`, `multi-reference` | PASS | Full B/C/D/E sequence passed. |
| FLUX.2 Klein 9B | `black-forest-labs/FLUX.2-klein-9B` source | `latent-img2img`, `edit-reference`, `multi-reference` | PASS | Full B/C/D/E sequence passed. |
| FLUX.2 Klein 9B | `AbstractFramework/flux.2-klein-9b-8bit` | `latent-img2img`, `edit-reference`, `multi-reference` | PASS | Full B/C/D/E sequence passed. |
| Qwen Image Edit 2509 | `Qwen/Qwen-Image-Edit-2509` source | `edit-reference`, `multi-reference` | PASS | Full B/C/D/E sequence passed. |
| Qwen Image Edit 2509 | `AbstractFramework/qwen-image-edit-2509-8bit` | `edit-reference`, `multi-reference` | PASS | Full B/C/D/E sequence passed and visually matches source behavior closely. |
| Qwen Image Edit 2509 | `AbstractFramework/qwen-image-edit-2509-4bit` | `edit-reference`, `multi-reference` | PARTIAL | B/C/D passed; E preserved the pencil crash structure but weakly applied the cinematic color reference. |
| Qwen Image Edit 2511 | `Qwen/Qwen-Image-Edit-2511` source | `edit-reference`, `multi-reference` | FAIL | Single-image edits are useful, but the tested multi-reference composition fails. |
| Qwen Image Edit 2511 | `AbstractFramework/qwen-image-edit-2511-8bit` | `edit-reference`, `multi-reference` | FAIL | Single-image edits are useful, but the tested multi-reference composition fails. |

Implementation outcome:

- Unified `mlxgen generate` keeps routing FIBO Edit through `fibo.edit`; the route was not removed.
- Release evidence now uses true source handles and prepared package handles/paths, not short
  aliases or runtime `--quantize`.
- The q8 package that passes the standardized Qwen edit sequence is
  `AbstractFramework/qwen-image-edit-2509-8bit`, not 2511.
- FIBO Edit should remain documented as route-available but not release-quality for this
  standardized edit workflow until a separate model/prompt investigation produces passing proof.

Post-review correction:

- The primary review artifact is now the 5-row by 4-column matrix above. It uses one row per model
  family and columns for the canonical source, pencil sketch, crash from source, and composition.
  Package variants appear inside each applicable cell so source/BF16/q8/q4 behavior can be
  compared without mixing purposes across rows.
- The clearer post-review artifacts separate the questions:
  - source-handle capability matrix: one row per source model, four columns;
  - variant matrices: one row per source/prepared package variant for each model family.
- The legacy summary filename
  `validation_outputs/edit_prepared_capability_2026_06_05/edit_capability_summary_contact_sheet.jpg`
  now points to the clear source-handle 5x4 matrix instead of the old representative grid.
- Completed item [0028](0028_release_validation_registry.md) adds machine-readable release
  validation metadata and `mlxgen validation` for the same package/status evidence. Use that API
  for application decisions; use these contact sheets for visual review.
- Qwen Image Edit 2509 q4 was downloaded and added after the initial matrix because it is a
  published package listed in the quantization docs. B/C/D passed; E is partial.
- FIBO Edit source/BF16/q8 failures are retained as failure evidence. A subsequent Diffusers parity
  audit found concrete MLX port issues: empty CFG rows must encode to FIBO's begin-of-text token,
  the FIBO transformer must include `norm_out.linear.bias`, single-attention masks must broadcast
  to all heads, FP16 activations must use Diffusers' finite-range clipping, and the SmolLM3 text
  encoder must respect the upstream NoPE/RoPE layer pattern. Current code contains those fixes and
  rejects old prepared FIBO folders missing the bias, but source-handle visual validation still
  fails and needs tensor-level parity work.
- Planned item [0027](../planned/0027_fibo_edit_diffusers_parity_release_quality.md) tracks
  tensor-level Diffusers parity, then re-preparing FIBO Edit BF16/q8 folders only if the corrected
  source route passes release-quality visual validation.

## Implementation notes

Do not accept alias commands, runtime quantization, or a single style-transfer proof as release
evidence. The evidence must let a reviewer inspect each advertised edit mode for each model and
package that MLX-Gen claims as working.
