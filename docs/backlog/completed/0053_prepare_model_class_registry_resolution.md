# Planned: Resolve prepare model class from model config

## Metadata
- Created: 2026-06-27
- Status: Completed
- Completed: 2026-06-27

## ADR status
- Governing ADRs: ADR 0002
- ADR impact: May revise existing ADR if this becomes the first step toward a full model registry.

## Context
`mlxgen prepare` must instantiate the same backend family that `ModelConfig` resolves. Local and
custom model paths often need `--base-model` to declare that family.

## Current code reality
- `src/mflux/models/common/cli/save.py` selects the model class from raw `args.model` substring
  checks.
- The same function separately computes `ModelConfig.from_name(args.model, base_model=args.base_model)`.
- A custom path that lacks a family substring can resolve to a Qwen, Wan, FIBO, or other base model
  but still instantiate `Flux1`.

## Problem
Class selection and config resolution are split. This can load or prepare a custom model with the
wrong Python backend.

## What we want to do
Choose the prepare backend from the resolved `ModelConfig` family instead of the raw model path.

## Why
Prepared packages must preserve the source model architecture. Substring routing is brittle and
violates the existing fail-closed routing policy.

## Requirements
- Resolve `ModelConfig` once.
- Select the model class from the resolved base family or aliases.
- Keep the existing Bonsai prepacked rejection.
- Add tests for custom/local paths with explicit `--base-model`.

## Suggested implementation
Add a small private selector in `save.py`, such as `_model_class_for_config(model_config)`, that
uses `model_config.aliases` and `model_config.base_model` to identify the family. Use the returned
class for instantiation.

## Scope
- `src/mflux/models/common/cli/save.py`
- `tests/cli/test_prepare_save.py`

## Non-goals
- Do not build the full unified model registry in this item.
- Do not change generation routing behavior unless a test proves the same bug exists there.

## Dependencies and related tasks
- Related: `0054_inference_defaults_and_step_validation.md`

## Expected outcomes
- `mlxgen prepare --model ./local-qwen --base-model qwen-image` instantiates `QwenImage`.
- Existing Wan, FIBO Edit, ERNIE, Z-Image, SeedVR2, FLUX.2, and FLUX.1 prepare paths keep working.

## Validation
- `uv run pytest tests/cli/test_prepare_save.py -q`
- `uv run ruff check src/mflux/models/common/cli/save.py tests/cli/test_prepare_save.py`

## Progress update: 2026-06-27
- `mlxgen prepare` now resolves `ModelConfig` once before selecting the backend class.
- Backend class selection now uses resolved family aliases/model names instead of raw path
  substrings.
- Added custom path plus `--base-model qwen-image` coverage.
- Replaced the hidden final `Flux1` fallback with an explicit parser error for unresolved prepare
  backends.
- Added ambiguous custom-path coverage proving `--base-model` recovery guidance is shown before a
  backend is instantiated.

## Progress checklist
- [x] Resolve `ModelConfig` before class selection.
- [x] Add config-based class selector.
- [x] Cover explicit-base custom path routing.
- [x] Fail closed for ambiguous custom/local prepare paths.

## Implementation notes
Prefer a narrow selector over a broad registry refactor. Preserve current behavior where the
resolved family is unambiguous.
