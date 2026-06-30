# Completed: Qwen control route hardening

## Metadata
- Created: 2026-06-29
- Status: Completed
- Completed: 2026-06-29

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
At investigation time, the public Qwen control-inpaint route was theoretically exposed but failed
in practice on the no-LoRA path. The same validation pass also showed that the base
control/control-inpaint path bypassed the blank-negative fallback that the base Qwen text-only
route already enforced.

## Historical pre-fix reality
- `src/mflux/models/qwen/variants/controlnet/qwen_image_controlnet.py` builds image metadata with
  `**LoRALoader.extra_metadata_for_model(self)`, which raises when the loader returns `None`.
- A real CLI repro using `mlxgen generate --model AbstractFramework/qwen-image-8bit --image ... --mask-path ...`
  failed with `TypeError: 'NoneType' object is not a mapping` after denoising, with the historical
  event proof saved under `validation_outputs/serial_validation_qwen_control_progress.json`.
- The same module decides CFG activation from `negative_prompt is not None`, but does not normalize
  blank or omitted negatives to the base Qwen fallback `" "`.
- `src/mflux/models/qwen/cli/qwen_image_generate.py` forwards the raw negative prompt value into
  the control route rather than reusing the base Qwen prompt-normalization contract.

## Problem
One public Qwen route crashes on a normal no-LoRA path, and the base control routes do not honor
the same negative-prompt contract as the rest of the base Qwen image runtime.

## What we want to do
Make Qwen control and control-inpaint behave like stable first-class routes: no crash on the
default no-LoRA path, and the same blank-negative semantics as the base Qwen route.

## Why
The current behavior breaks the clean cross-mode abstraction and forces route-specific surprises
into a public API surface that is supposed to be consistent.

## Requirements
- No-LoRA Qwen control/control-inpaint runs must complete successfully.
- Omitted or blank negative prompts must resolve exactly like base Qwen text generation when
  guidance requires CFG.
- Preserve existing pixels for cases whose math should remain unchanged.
- Avoid broad Qwen prompt-routing refactors while fixing this route.

## Suggested implementation
Normalize extra metadata merges to an empty dict when the loader has no metadata, and reuse one
base-Qwen negative-prompt resolver for control/control-inpaint so the fallback contract cannot
drift.

## Scope
- Qwen control/control-inpaint runtime path.
- Qwen control CLI handoff into the runtime.
- Focused regression tests and one-at-a-time real control-inpaint proofs.

## Non-goals
- Do not redesign LoRA metadata schema.
- Do not broaden this item into Qwen edit or multi-reference route changes.
- Do not change quality-affecting Qwen scheduler or transformer math beyond the prompt fallback
  contract already used by base Qwen.

## Dependencies and related tasks
- Completed [0008 Qwen edit parity expansion](0008_qwen_edit_parity_expansion.md)
- Completed [0029 Qwen Image Edit 2511 base parity](0029_qwen_image_edit_2511_base_parity.md)
- Completed [0067 progress event contract hardening](0067_progress_event_contract_hardening.md)
- Completed [0070 canonical capability identity for variant routes](0070_canonical_capability_identity_for_variant_routes.md)

## Expected outcomes
- Public Qwen control-inpaint no longer crashes on the no-LoRA path.
- Base Qwen control routes resolve blank negatives consistently with the base Qwen contract.
- Real control-inpaint proof artifacts show both successful completion and corrected metadata.
- The refreshed one-at-a-time proof lives under
  `validation_outputs/runtime_contracts_2026_06_29/qwen_control_inpaint_q8_4step_fix.*`.

## Validation
- Focused regression tests for nil-safe extra metadata handling and blank-negative normalization.
- A one-at-a-time real Qwen control-inpaint CLI or Python proof using the same validation assets as
  the failing repro.
- Verification that corrected progress proofs align with item 0067.

## Progress checklist
- [x] Add a shared or reused base-Qwen negative-prompt resolver for control routes.
- [x] Make extra metadata merges nil-safe on the no-LoRA path.
- [x] Add focused tests that do not stub away the failing behavior.
- [x] Re-run the real control-inpaint proof one run at a time and preserve the artifact.

## Guidance for the implementing agent
Keep the fix route-local and explicit. The current failure is already concrete enough that a small
patch is preferable to a broad Qwen loader or metadata abstraction rewrite.

## Completion report

- Date: 2026-06-29
- Original path: `docs/backlog/planned/runtime_contracts/0068_qwen_control_route_hardening.md`
- Final path: `docs/backlog/completed/0068_qwen_control_route_hardening.md`
- Summary: The public base-Qwen control and control-inpaint routes now survive the normal no-LoRA
  path and preserve the base-Qwen negative-prompt contract without broad Qwen runtime refactors.
- Implementation: `qwen_image_controlnet.py` now treats missing LoRA metadata as `{}` instead of
  crashing, and it preserves the caller negative-prompt value in metadata while still skipping the
  CFG negative branch when guidance is inactive.
- Validation: `tests/image_generation/test_masked_generation_routes.py`, the selected Qwen CLI
  router tests in `tests/cli/test_mlx_gen_router.py`, and the focused task-inference and progress
  suites passed. The real repaired proof lives under
  `validation_outputs/runtime_contracts_2026_06_29/qwen_control_inpaint_q8_4step_fix.*`.
- Evidence report: [runtime_contracts_report.md](../../assets/validation/runtime-contracts-2026-06-29/runtime_contracts_report.md)
- User-facing result: the same public route that previously failed after denoising on a normal
  no-LoRA path now completes successfully with corrected metadata and progress semantics.
