# Completed: Standardized I2I sequence validation

## Metadata
- Created: 2026-06-05
- Status: Completed
- Completed: 2026-06-05

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None

## Context

Item [0023](../completed/0023_i2i_capability_validation_matrix.md) proved that many image routes
execute, but it did not prove comparable model behavior for a fixed edit workflow. It mixed source
and reference assets across runs, accepted weak Qwen hard-landing rows, and did not clearly list
the source, cinematic reference, pencil reference, and crash reference needed for model-to-model
inspection.

The required sequence is:

1. source;
2. cinematic restyle;
3. crash / hard landing;
4. pencil sketch of the crash from the source;
5. composition using the pencil/crash result and the cinematic result.

## Current code reality

- `src/mflux/task_inference.py` exposes typed I2I modes:
  - `latent-img2img` for whole-image variation through `--image-strength`;
  - `edit-reference` for instruction-conditioned single-image edits;
  - `multi-reference` for two-or-more-image reference composition.
- `mlxgen capabilities --model <model>` is the source of truth for which mode a model advertises.
- Current complete installed release packages include FLUX.2 Klein 4B/9B q4/q8, Qwen Image 2512
  q4/q8, Qwen Image Edit 2511 q4/q8, Z-Image Turbo q4/q8, and ERNIE Image Turbo q4/q8.
- FIBO Edit has a dedicated backend, but unified capabilities fail closed after local visual QA
  failures. Repair is tracked by [0024](../planned/0024_fibo_edit_unified_i2i_validation.md).
- Existing proof assets that should be treated as references, not final standardized evidence:
  - `docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png`
  - `docs/assets/examples/spaceship-snow/02_i2i_pencil_sketch.png`
  - `docs/assets/examples/spaceship-snow/03_i2i_crash_snow.png`
  - `validation_outputs/i2i_full_matrix_2026_06_05/i2i_full_matrix_manifest.md`

## Problem

The current validation table is not a fair comparison. A user cannot inspect one row per model and
answer whether the model handles the same source, the same prompts, the same references, and the
same sequence of requested transformations.

## What we want to do

Create a standardized I2I validation suite and report for every installed image model that
advertises I2I capabilities. The report must show exactly which source and reference images were
used for every row, the exact prompt, the command, the output image, and a manual visual status.

## Why

I2I quality claims are otherwise ambiguous. A pencil sketch can pass while a complex hard-landing
edit fails; a latent restyle can look good while an edit/reference model ignores an object-state
instruction. Release documentation must separate those outcomes clearly.

## Requirements

- Use one canonical source image for the sequence:
  `docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png`.
- Use one common target canvas for low-cost validation, preferably `432x240`, while preserving the
  source aspect ratio.
- For each model, run only modes that `mlxgen capabilities` advertises. Unsupported sequence steps
  must be marked `N/A: capability not advertised`, not forced through a fallback.
- Use the same prompts for the same sequence step across all models that support that step.
- Record the exact command, model, capability, input images, prompt, output path, seed, steps,
  guidance, `--image-strength` when used, and manual status.
- Do not accept an output as passing if it:
  - dissolves or smears the spaceship;
  - ignores the crash/hard-landing state;
  - loses the source scene identity;
  - uses the wrong reference image;
  - changes canvas ratio unexpectedly;
  - only proves pencil/style transfer when the tested step is object-state editing.
- Treat Qwen Edit 2511 hard-landing/crash rows as failed or unverified until a fresh standardized
  row passes visual QA.
- Keep FIBO Edit out of the standardized matrix until
  [0024](../planned/0024_fibo_edit_unified_i2i_validation.md) re-enables validated unified
  capabilities.

## Standard sequence

| Step | Name | Mode | Inputs | Prompt intent |
| --- | --- | --- | --- | --- |
| A | source | N/A | `01_t2i_spaceship_snow.png` | Canonical input; no generation. |
| B | cinematic | `latent-img2img` when supported; otherwise `edit-reference` for edit-only models | source | Make the same spaceship-in-snow scene more cinematic while preserving layout. |
| C | crash | `edit-reference` | source | Make the same spaceship hard-landed/crashed in the snow while preserving a solid, recognizable ship. |
| D | pencil crash | `edit-reference` or `multi-reference` if required by the model | source, crash result/reference | Produce a clean graphite pencil sketch of the hard-landed spaceship scene. |
| E | composition | `multi-reference` | D pencil/crash result, B cinematic result, source if the model needs a primary anchor | Compose the pencil/crash structure with the cinematic lighting/reference while preserving the same scene. |

## Candidate model matrix

| Model | Steps to run |
| --- | --- |
| `AbstractFramework/flux.2-klein-4b-4bit` | B latent, C edit, D edit, E multi-reference |
| `AbstractFramework/flux.2-klein-4b-8bit` | B latent, C edit, D edit, E multi-reference |
| `AbstractFramework/flux.2-klein-9b-4bit` | B latent, C edit, D edit, E multi-reference |
| `AbstractFramework/flux.2-klein-9b-8bit` | B latent, C edit, D edit, E multi-reference |
| `AbstractFramework/qwen-image-edit-2511-4bit` | B edit, C edit, D edit, E multi-reference |
| `AbstractFramework/qwen-image-edit-2511-8bit` | B edit, C edit, D edit, E multi-reference |
| `AbstractFramework/qwen-image-2512-4bit` | B latent only |
| `AbstractFramework/qwen-image-2512-8bit` | B latent only |
| `AbstractFramework/z-image-turbo-4bit` | B latent only |
| `AbstractFramework/z-image-turbo-8bit` | B latent only |
| `AbstractFramework/ernie-image-turbo-4bit` | B latent only |
| `AbstractFramework/ernie-image-turbo-8bit` | B latent only |
| `fibo-edit` | Historical matrix exclusion: 0024 had not passed when this matrix ran |

## Suggested implementation

1. Generate a `validation_outputs/i2i_standard_sequence_<date>/` directory.
2. Write a small serial harness that resolves `mlxgen capabilities` for each candidate model and
   emits the exact command before running it.
3. Use fixed seeds per step, not per model, unless a model requires a documented alternate seed.
4. Build one contact sheet per model and one global comparison sheet grouped by sequence step.
5. Produce a manifest table with these columns: model, step, capability, source image(s), prompt,
   command, result image, status, reviewer notes.
6. Update user-facing docs only with behaviors that pass this standardized matrix.

## Scope

- Installed AbstractFramework image I2I packages.
- Unified `mlxgen generate` routes only.
- Local proof images and manifest under `validation_outputs/`.
- Focused fixes to routing, mode validation, source/reference ordering, or docs if the harness
  exposes implementation defects.

## Non-goals

- Do not repair FIBO Edit here; use item 0024.
- Do not implement outpainting/reframing here; use item 0019.
- Do not add new model families.
- Do not publish the validation images automatically.
- Do not turn weak prompt outcomes into release claims.

## Dependencies and related tasks

- [0020](../completed/0020_generation_capability_contract.md): capability contract.
- [0022](../completed/0022_i2i_source_aspect_ratio_policy.md): source-aspect canvas policy.
- [0023](../completed/0023_i2i_capability_validation_matrix.md): historical matrix and correction.
- [0024](../planned/0024_fibo_edit_unified_i2i_validation.md): FIBO repair.
- [0019](../planned/0019_first_class_i2i_modes_and_outpaint_reframe.md): first-class outpaint/reframe UX.

## Expected outcomes

- A normalized source/reference table exists before any generated row.
- Every generated row has an exact command, prompt, source image(s), output path, and visual status.
- Qwen Edit 2511 complex crash behavior is either proven with passing outputs or marked as a model
  limitation / unresolved integration issue.
- FIBO Edit remains excluded from unified release claims unless 0024 passes.
- Release docs can truthfully explain which models support latent restyle, edit, and
  multi-reference composition.

## Validation

- `mlxgen capabilities` output saved for every candidate model.
- Serial model-backed generation for every applicable step.
- Manual visual inspection of every output.
- Contact sheets grouped by model and by sequence step.
- `uv run pytest` focused on task inference/router tests if any code changes are made.
- `uv run ruff check` on touched code if any code changes are made.

## Progress checklist
- [x] Generate candidate capability inventory.
- [x] Create canonical source/reference manifest before running models.
- [x] Run Step B for all latent-capable and edit-only models as applicable.
- [x] Run Step C for all edit-capable models.
- [x] Run Step D for all edit/multi-reference-capable models.
- [x] Run Step E for all multi-reference-capable models.
- [x] Build contact sheets and manifest.
- [x] Mark Qwen and FIBO limitations honestly.
- [x] Update docs only with validated claims.
- [x] Move to completed with evidence.

## Completion report - 2026-06-05

### Summary

Ran the standardized I2I sequence at `432x240`, 20 steps, fixed source, fixed prompts, and fixed
seeds. The harness executed 30 serial `mlxgen generate` commands covering FLUX.2 Klein 4B/9B
q4/q8, Qwen Image Edit 2511 q4/q8, Qwen Image 2512 q4/q8, Z-Image Turbo q4/q8, and ERNIE Image
Turbo q4/q8. All commands returned successfully and wrote output images plus metadata.

### Evidence

- Harness:
  `validation_outputs/i2i_standard_sequence_2026_06_05/run_standardized_i2i_sequence.py`
- Manifest:
  `validation_outputs/i2i_standard_sequence_2026_06_05/i2i_standard_sequence_manifest.md`
- Result JSONL:
  `validation_outputs/i2i_standard_sequence_2026_06_05/i2i_standard_sequence_results.jsonl`
- Full contact sheet:
  `validation_outputs/i2i_standard_sequence_2026_06_05/i2i_standard_sequence_all_contact_sheet.jpg`
- Step contact sheets:
  - `validation_outputs/i2i_standard_sequence_2026_06_05/step_b_contact_sheet.jpg`
  - `validation_outputs/i2i_standard_sequence_2026_06_05/step_c_contact_sheet.jpg`
  - `validation_outputs/i2i_standard_sequence_2026_06_05/step_d_contact_sheet.jpg`
  - `validation_outputs/i2i_standard_sequence_2026_06_05/step_e_contact_sheet.jpg`

### Results

| Model family/package | Result |
| --- | --- |
| FLUX.2 Klein 4B/9B q4/q8 | Passed all applicable B/C/D/E rows. |
| Qwen Image Edit 2511 4-bit | Passed B cinematic only; failed C crash, D pencil-crash, and E composition. |
| Qwen Image Edit 2511 8-bit | Passed B cinematic and C crash; D pencil-crash was partial; E composition failed to carry crash/debris state. |
| Qwen Image 2512 q4/q8 | Failed B latent cinematic because the spaceship became a car-like vehicle. |
| Z-Image Turbo q4/q8 | Passed B latent cinematic. |
| ERNIE Image Turbo q4/q8 | Passed B latent cinematic. |
| FIBO Edit | Excluded from this matrix at the time; later true-handle/prepared validation is recorded in item 0026, and open parity/reprepare work is tracked in item 0027. |

Aggregate visual statuses: 23 `PASS`, 6 `FAIL`, 1 `PARTIAL`.

### Behavior changes

No package code was changed for this item. The validation artifacts changed the release-readiness
state: the project now has a fair standardized matrix and should not use the historical 0023 matrix
as complex-edit proof.

### Residual risks and follow-ups

- Qwen Edit 2511 is not reliable for this complex crash/composition sequence through current
  MLX-Gen prompts/settings. A future Qwen parity item should compare MLX-Gen against upstream
  Qwen Image Edit 2511 for hard object-state edits and multi-reference composition.
- Qwen Image 2512 latent I2I at `--image-strength 0.35` is too destructive for this spaceship
  identity preservation test. Future tuning can try lower strength, but this run must remain marked
  failed for the tested settings.
- FIBO Edit remained a separate malfunction investigation in planned item 0024 at the time of this
  standardized matrix.

### Post-completion update - 2026-06-05

Planned item [0024](../planned/0024_fibo_edit_unified_i2i_validation.md) later re-enabled the
narrow plain one-image unmasked `fibo-edit` route after q8 and default-weight unified visual proof.
This completed matrix remains valid for its original candidate set and should not be retroactively
treated as FIBO masked, RMBG, or multi-reference evidence.

### Validation commands

- `uv run python validation_outputs/i2i_standard_sequence_2026_06_05/run_standardized_i2i_sequence.py`
- Manual visual inspection of the full and step contact sheets.
- `git diff --check`

## Implementation notes

Do not reuse arbitrary older outputs as passing proof unless they match the standardized source,
inputs, prompt, dimensions, seed policy, and command contract. A capability is release-proven only
when the row for that exact model and exact sequence step passes manual visual inspection.
