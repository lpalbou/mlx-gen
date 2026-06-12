# Completed: Qwen Image Edit 2511 base parity

## Metadata
- Created: 2026-06-05
- Status: Completed
- Completed: 2026-06-06

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None

## Context

Completed items [0025](../completed/0025_standardized_i2i_sequence_validation.md) and
[0026](../completed/0026_edit_model_prepared_capability_contact_sheets.md) showed that Qwen Image
Edit 2509 source/q8 can pass the standardized spaceship edit sequence, while Qwen Image Edit 2511
does not yet pass the complex multi-reference composition row. The broader Qwen parity expansion
remains in [planned item 0008](../planned/0008_qwen_edit_parity_expansion.md), but 2511 base
parity is now release-blocking work and needs its own planned item.

## Current code reality

- `ModelConfig` resolves `Qwen/Qwen-Image-Edit-2511` and sets `zero_cond_t=True` for the
  transformer path.
- Unified `mlxgen generate` routes 2511 with one image to `image-to-image` / `edit-reference` and
  with multiple images to `image-to-image` / `multi-reference`.
- Qwen edit now preserves omitted negative prompts instead of synthesizing a space prompt.
- MLX-Gen has tests for EditPlus prompt formatting, multi-reference M-RoPE positions, per-reference
  conditioning latents, and zero-condition-timestep selection.
- Current release validation reports Qwen Image Edit 2511 source/q8 as failing the standardized
  multi-reference composition row.
- The local Diffusers checkout contains `pipeline_qwenimage_edit.py`,
  `pipeline_qwenimage_edit_plus.py`, and adjacent Qwen inpaint/control/layered pipelines that must
  be treated as upstream contracts when deciding MLX-Gen capabilities.

## Problem

Qwen Image Edit 2511 is advertised upstream as the stronger edit model, but MLX-Gen currently
cannot treat its base model as passing the same edit capabilities that Qwen Edit 2509 passes
locally. This can be a math/contract mismatch, an unsupported capability claim, or a validation
prompt/settings issue; the task is to determine and fix the MLX-Gen side where feasible.

## What we want to do

Make the base `Qwen/Qwen-Image-Edit-2511` route pass the standardized edit-reference and
multi-reference validation sequence, or narrow its capability claims with precise evidence if the
upstream model itself does not support a given row.

## Why

Qwen Image Edit is the permissively licensed image-edit lane. AbstractVision and MLX-Gen users need
clear, working base-model edit behavior before quantized package quality can be evaluated.

## Requirements

- Use the true base handle `Qwen/Qwen-Image-Edit-2511` before testing q8/q4 packages.
- Compare MLX-Gen to local Diffusers and Transformers code for:
  - EditPlus prompt construction and image order;
  - single-image versus multi-reference canvas semantics;
  - Qwen2.5-VL multimodal position ids and attention masks;
  - conditioning latent packing and `img_shapes`;
  - `zero_cond_t` target/reference timestep conditioning;
  - scheduler timesteps, sigma handling, CFG gating, and negative prompt behavior.
- Check upstream Hugging Face/model-card capability claims before expanding or narrowing the
  capability matrix.
- Keep quantized packages out of scope until the base model passes.

## Suggested implementation

Start with tensor-level parity tests for short, deterministic 2511 edit inputs before running
full visual validation. Prefer small unit tests that isolate prompt encoding, image order, latent
packing, and first-denoise inputs.

## Scope

- Base `Qwen/Qwen-Image-Edit-2511` only.
- Single-image edit-reference and multi-reference edit behavior.
- Capability matrix corrections if upstream or local evidence shows a mode is unsupported.

## Non-goals

- Do not validate q8/q4 packages in this item.
- Do not port every Qwen pipeline in proposed item 0008.
- Do not claim inpaint/outpaint/layered/control support unless a separate planned item implements
  and validates that public API.

## Dependencies and related tasks

- [0020 generation capability contract](../completed/0020_generation_capability_contract.md)
- [0025 standardized I2I sequence validation](../completed/0025_standardized_i2i_sequence_validation.md)
- [0026 edit model prepared-package capability contact sheets](../completed/0026_edit_model_prepared_capability_contact_sheets.md)
- [0028 release validation registry](../completed/0028_release_validation_registry.md)
- [0008 Qwen edit parity expansion](../planned/0008_qwen_edit_parity_expansion.md)
- Local upstream: `/Users/albou/projects/gh/diffusers/`,
  `/Users/albou/projects/gh/transformers/`

## Expected outcomes

- `Qwen/Qwen-Image-Edit-2511` base either passes the standardized source/cinematic/crash/pencil/
  composition sequence or has an exact unsupported/failing capability record.
- The capability matrix does not imply that a visually failing route is release-quality.
- Any math parity fixes are covered by focused tests.
- The final report lists exact commands, prompts, source images, output artifacts, and pass/fail
  status for base 2511 rows.

## Validation

- Focused unit/parity tests for prompt encoding, image order, conditioning latent packing, and
  denoise-loop contracts.
- One serial base-model validation run for the standardized edit sequence after parity fixes.
- Manual visual review contact sheet using one row per model/package and columns for source,
  cinematic, crash, pencil crash, and multi-reference composition.

## Progress checklist
- [x] Confirm upstream 2511 capability claims from model cards and Diffusers code.
- [x] Audit current MLX-Gen Qwen edit path against Diffusers/Transformers.
- [x] Add or adjust focused parity tests.
- [x] Patch MLX-Gen math/contract mismatches.
- [x] Run base-model validation.
- [x] Update release validation registry and docs with the final base-model status.

## Completion Report

### Date

2026-06-06

### Summary

MLX-Gen's FlowMatch Euler scheduler now uses the model-config dynamic-shift values required by the
Qwen Image/Edit Diffusers pipelines. The previous implementation used an empirical step-dependent
shift for Qwen Image Edit 2511, producing a much stronger sigma shift than the upstream scheduler
contract at small validation sizes.

After the scheduler fix, `Qwen/Qwen-Image-Edit-2511`,
`AbstractFramework/qwen-image-edit-2511-8bit`, and
`AbstractFramework/qwen-image-edit-2511-4bit` all passed the focused source/pencil/crash/composition
profile at `432x240`, 40 steps, guidance `4`.

### Code And Docs

- Updated `src/mflux/models/common/schedulers/flow_match_euler_discrete_scheduler.py` so
  `set_image_seq_len()` passes `sigma_base_seq_len`, `sigma_max_seq_len`, `sigma_base_shift`,
  `sigma_max_shift`, and `sigma_shift_terminal` from `ModelConfig`.
- Added scheduler regression coverage in `tests/schedulers/test_linear_scheduler.py`.
- Updated `src/mflux/release/validation_registry.py` so Qwen Image Edit 2511 source/q8/q4 rows
  point at the 2026-06-06 proof assets and report `PASS`.
- Updated current user docs and LLM indexes to show Qwen Image Edit 2511 source/q8/q4 as validated
  for the focused profile and to keep FIBO Edit unsupported through unified `mlxgen generate`.

### Evidence

- Contact sheet:
  `docs/assets/validation/qwen-edit-2511-parity-2026-06-06/qwen-image-edit-2511-source-q8-q4-parity.jpg`
- Command log:
  `docs/assets/validation/qwen-edit-2511-parity-2026-06-06/qwen-image-edit-2511-command-log.md`
- Generated source-model outputs:
  `validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_base_b_pencil.png`,
  `validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_base_c_crash.png`,
  `validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_base_d_composition.png`
- Generated q8 outputs:
  `validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q8_b_pencil.png`,
  `validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q8_c_crash.png`,
  `validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q8_d_composition.png`
- Generated q4 outputs:
  `validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q4_b_pencil.png`,
  `validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q4_c_crash_retry_wide.png`,
  `validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q4_d_composition.png`

### Validation

```sh
uv run pytest tests/schedulers/test_linear_scheduler.py \
  tests/image_generation/test_qwen_edit_prompt_contract.py \
  tests/image_generation/test_qwen_edit_util.py \
  tests/image_generation/test_qwen_edit_dimensions.py \
  tests/test_task_inference.py::test_fibo_edit_exposes_no_public_generation_capabilities -q
```

Result: 25 passed.

The final q8/q4 validation was intentionally run after the source handle passed. The q4
hard-landing row required an explicit wide-shot prompt and crop-avoidance negative prompt to keep
the full scene framing.

### Residual Risks

- This item validates the documented pencil/crash/composition profile. It does not claim Qwen Image
  Edit 2511 inpaint, outpaint, layered, or control pipelines.
- FIBO Edit remains unsupported through unified `mlxgen generate`; future FIBO work remains tracked
  separately in item 0027.
