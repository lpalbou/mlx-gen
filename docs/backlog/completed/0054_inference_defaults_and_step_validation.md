# Planned: Centralize inference defaults and step validation

## Metadata
- Created: 2026-06-27
- Status: Completed
- Completed: 2026-06-27

## ADR status
- Governing ADRs: ADR 0002
- ADR impact: May revise existing ADR if a new default-source-of-truth policy is added.

## Context
The audit found two related issues: direct Python APIs can carry different default step counts from
the CLI, and common config/scheduler paths do not consistently reject invalid step counts early.

## Current code reality
- CLI model step defaults live in `src/mflux/cli/defaults/defaults.py` and are applied by
  `src/mflux/cli/parser/parsers.py`.
- Direct model APIs still hard-code defaults such as `num_inference_steps=4` on several image
  variants.
- `src/mflux/models/common/config/config.py` stores `num_inference_steps` without validating it.
- Common schedulers can divide by `num_steps` or `num_steps - 1`.
- Wan schedulers already validate positive step counts in their own scheduler layer.

## Problem
Invalid `--steps` values can fail late with scheduler exceptions, and direct API defaults can
silently produce different behavior from the CLI.

## What we want to do
Add early validation for common image config step counts first, then move model defaults toward one
source of truth.

## Why
Users should get clear errors before model loading or denoising starts. Public Python APIs should
not quietly drift away from CLI behavior.

## Requirements
- Reject `num_inference_steps < 1` in common `Config`.
- Reject single-step configurations for schedulers that require at least two steps.
- Add focused tests for invalid values.
- Plan the larger default-source consolidation without forcing a broad signature rewrite in the
  same patch.

## Suggested implementation
Start with `Config` validation and a scheduler-specific minimum for `flow_match_euler_discrete`.
Then replace hard-coded direct API defaults with `None` and resolve through model profiles in a
separate, focused pass.

## Scope
- Common image `Config` and common scheduler validation.
- Focused scheduler/config/parser tests.

## Non-goals
- Do not run long model generations just to validate parser/config errors.
- Do not change Wan scheduler semantics unless a Wan-specific bug is found.

## Dependencies and related tasks
- Related: `0053_prepare_model_class_registry_resolution.md`

## Expected outcomes
- Invalid step counts fail early with clear `ValueError` or parser errors.
- A follow-up pass has a clear route to eliminate CLI/Python default drift.

## Validation
- `uv run pytest tests/schedulers/test_linear_scheduler.py tests/arg_parser/test_cli_argparser.py -q`
- `uv run ruff check src/mflux/models/common/config/config.py src/mflux/models/common/schedulers/flow_match_euler_discrete_scheduler.py tests/schedulers/test_linear_scheduler.py`

## Progress update: 2026-06-27
- Added parser-level rejection for `--steps 0` / negative values.
- Added common `Config` rejection for non-positive inference steps.
- Added FlowMatch scheduler rejection for single-step configs, avoiding the `num_steps - 1`
  division path.
- Added a common `default_inference_steps()` resolver and made CLI defaults re-export the same
  table.
- Made common `Config` resolve model defaults when step counts are unspecified.
- Updated concrete drift points called out in review: FLUX.1, FLUX.2 Klein/base, Qwen ControlNet,
  FIBO Edit/RMBG, and Z-Image direct APIs now resolve from `model_config` when callers omit steps.

## Progress checklist
- [x] Add common step-count validation.
- [x] Add flow-match minimum validation.
- [x] Add focused tests.
- [x] Follow up on direct API default consolidation for confirmed drift points.

## Implementation notes
Split validation from broad default rewrites. Clear early errors are the urgent part; complete
default unification can land safely after the blockers.
