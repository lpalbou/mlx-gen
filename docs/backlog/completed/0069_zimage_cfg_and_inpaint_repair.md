# Completed: Z-Image CFG and inpaint repair

## Metadata
- Created: 2026-06-29
- Status: Completed
- Completed: 2026-06-29

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The 2026-06-29 validation pass confirmed that the base Z-Image CFG branch was mathematically
wrong, and the native inpaint route inherited the public progress-label bug tracked separately in
item 0067.

## Historical pre-fix reality
- `src/mflux/models/z_image/variants/z_image.py` computed guided noise as
  `noise + guidance * (noise - negative_noise)`.
- A direct `_predict()` probe using `cond=1`, `negative=0`, and `guidance=2` returned `3.0`, which
  proves the current CFG branch over-applies guidance.
- The same module already normalizes blank negative prompts to `" "` in `_encode_prompts()`, so the
  confirmed issue is the CFG formula rather than negative-prompt handling.
- A real native inpaint proof at
  `validation_outputs/serial_validation_zimage_inpaint_4step.png` completed quickly but looked
  visually weak enough that the route should be rechecked after the CFG fix rather than assumed to
  be quality-sound.

## Problem
The base Z-Image guided-denoise math is wrong, which can degrade prompt fidelity and route quality.
Because the bug changes model math, it must be repaired with explicit quality-sensitive validation.

## What we want to do
Restore correct CFG math for Z-Image and revalidate native inpaint behavior on a real sample after
the fix.

## Why
This is a model-quality correctness issue, not just a UX paper cut. MLX-Gen should not claim a
clean abstraction if one backend applies a different guidance formula than the others.

## Requirements
- The guided branch must use the mathematically correct conditional/unconditional combination.
- Unguided generation behavior must remain unchanged.
- The fix must preserve or improve prompt fidelity on the real native inpaint proof.
- Validation must separate correctness from subjective quality: prove the math, then inspect a real
  case.

## Suggested implementation
Change the Z-Image guided branch to the standard CFG combination and pair it with a small math
regression test plus one-at-a-time real native inpaint reruns using the existing validation assets.

## Scope
- Z-Image `_predict()` guidance math.
- Native inpaint regression coverage.
- Real native inpaint proof refresh after the formula change.

## Non-goals
- Do not broaden this item into ControlNet or unrelated Z-Image family expansion.
- Do not bundle speculative speed or memory tuning that changes pixels.
- Do not claim quality improvements without refreshed real-case evidence.

## Dependencies and related tasks
- Completed [0043 Z-Image native inpaint](0043_zimage_native_inpaint.md)
- Proposed [0045 Z-Image ControlNet follow-up](../proposed/0045_zimage_controlnet_followup.md)
- Completed [0067 progress event contract hardening](0067_progress_event_contract_hardening.md)

## Expected outcomes
- A focused math regression proves the CFG branch uses the correct formula.
- Native inpaint keeps working and receives refreshed quality evidence after the fix.
- Progress-label validation from item 0067 remains green on the same route.

## Validation
- Focused unit test for the guided-noise combination.
- One-at-a-time real native inpaint rerun using the existing outpaint validation assets.
- Manual before/after inspection of the saved proof image plus metadata/perf comparison.

## Progress checklist
- [x] Correct the Z-Image CFG branch.
- [x] Add a focused math regression test.
- [x] Re-run native inpaint one case at a time and preserve the proof artifact.
- [x] Compare the refreshed output and metadata against the pre-fix proof.

## Guidance for the implementing agent
Treat this as quality-sensitive math repair. The real proof matters because a mathematically correct
fix that still produces visibly weak output needs to be reported honestly rather than silently
declared solved.

## Completion report

- Date: 2026-06-29
- Original path: `docs/backlog/planned/runtime_contracts/0069_zimage_cfg_and_inpaint_repair.md`
- Final path: `docs/backlog/completed/0069_zimage_cfg_and_inpaint_repair.md`
- Summary: Z-Image now uses the standard CFG combination, the focused math regression is green,
  and the published native-inpaint engine case was rerun on the same source, mask, prompt, seed,
  and step count for an apples-to-apples correctness check.
- Implementation: `z_image.py` now computes guided noise as
  `negative_noise + guidance * (noise - negative_noise)`. The masked route also inherits the
  shared explicit task/terminal progress hardening from item 0067.
- Validation: the masked-route regression suite passed, the base guided real prepared package run
  completed under `zimage_base_q8_latent_guided_4step_fix.*`, and the same-case native-inpaint
  rerun completed under `zimage_turbo_inpaint_engine_samecase_fix.*`.
- Comparison note: the refreshed same-case native-inpaint rerun preserved the accepted proof
  surface and kept peak RSS at the same scale as the historical published engine proof
  (`18.09 GB` vs `18.11 GB`). This item closes correctness, not a generalized performance claim.
- Evidence report: [runtime_contracts_report.md](../../assets/validation/runtime-contracts-2026-06-29/runtime_contracts_report.md)
- Residual risk: the quality claim stays narrow to the published engine-mask case and the exact
  prepared `AbstractFramework/z-image-turbo-8bit` row already documented in item 0043.
