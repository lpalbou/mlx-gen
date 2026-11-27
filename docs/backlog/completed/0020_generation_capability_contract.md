# Completed: Generation capability contract and route planning

## Metadata
- Created: 2026-06-04
- Status: Completed
- Completed: 2026-06-04

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None for this item; it implements the existing fail-closed routing policy and keeps any broader I2I/outpaint taxonomy work in planned item 0019.

## Context

Taskless routing made `mlxgen generate` easier to use, but it still treats `edit` as a public task
next to `text-to-image`, `image-to-image`, `text-to-video`, and `image-to-video`. That leaks a
backend implementation detail into the public API. It also prevents applications from asking a
model what it supports before starting an expensive generation.

The desired shape is:

- public tasks describe media direction only;
- internal modes describe how inputs are consumed;
- model capabilities describe which task/mode/input combinations are valid;
- routing produces a reusable generation plan for CLI and Python callers.

## Current code reality

- `src/mflux/task_inference.py` defines `EDIT = "edit"` as a task and returns only `ResolvedTask`.
- `src/mflux/cli/mlx_gen.py` hard-codes route selection from family checks and `resolved_task == "edit"`.
- `src/mflux/models/common/config/model_config.py` stores architecture, defaults, and Wan task information, but no normalized generation capability list.
- `--image-strength` is a latent img2img control implemented by `Config.init_time_step` and `LatentCreator.create_for_txt2img_or_img2img(...)`, but the unified router does not centrally reject it for edit-conditioned I2I.
- `GenerationContext` infers image progress as `image-to-image` only when `image_strength > 0`, which is wrong for edit-conditioned image-to-image.
- Existing router tests lock FLUX.2 single-image input without `--image-strength` to latent img2img, while the desired default for edit-capable FLUX.2 is edit/reference I2I.

## Problem

Users and application integrations cannot reliably distinguish:

- text-to-image versus image-to-image media direction;
- latent img2img versus instruction/reference image editing;
- single-reference edit versus multi-reference edit;
- video first-frame conditioning versus image-strength img2img.

The same confusion causes invalid flags to reach backends late. For example, `--image-strength`
should not be accepted for instruction/reference image editing.

## What we want to do

Add a public capability and generation-plan API that lets the CLI and Python callers resolve:

- public task;
- internal mode;
- model family;
- route handler;
- input image cardinality;
- whether `image_strength`, masks, outpainting, frames, or FPS are supported.

## Why

This makes the CLI easier to reason about and gives external apps a stable contract before they
start long or memory-heavy generations. It also aligns routing with ADR 0002 by rejecting unsupported
task/mode/option combinations before model load.

## Requirements

- Add typed capability descriptors and generation plans.
- Add public Python APIs:
  - `get_model_capabilities(...)`;
  - `resolve_generation_plan(...)`.
- Keep `infer_task(...)` and `resolve_task(...)` compatible, but make them return/derive public
  media-direction tasks, not `edit` as a task.
- Treat `--task edit` as a deprecated compatibility alias for `--task image-to-image --i2i-mode edit`.
- Add `--i2i-mode {auto,latent,edit,multi-reference}` to disambiguate image-to-image internals.
- Route FLUX.2 single-image input without `--image-strength` to edit/reference mode by default.
- Route FLUX.2 single-image input with `--image-strength` to latent img2img.
- Route exact edit models such as Qwen Image Edit and FIBO Edit to edit/reference I2I.
- Route multiple images to multi-reference I2I only for models that support it.
- Reject `--image-strength` for edit/reference and multi-reference modes before backend load.
- Keep Wan T2V/I2V fixed-task validation fail-closed.
- Add `mlxgen capabilities --model ...` for CLI inspection.
- Ensure progress events for image-conditioned edit can be reported as `image-to-image`.

## Suggested implementation

1. Refactor `src/mflux/task_inference.py` around `GenerationCapability`, `ModelCapabilities`, and
   `GenerationPlan`.
2. Keep capability builders near the resolver for this pass; migrate into `ModelConfig` only if
   that reduces duplication without creating a second source of truth.
3. Update `src/mflux/cli/mlx_gen.py` to route from `GenerationPlan.handler_id` and expose
   `mlxgen capabilities`.
4. Update public exports and Python integration docs.
5. Update focused resolver, router, and progress callback tests.

## Scope

- Capability descriptors for currently unified model families: Qwen, FLUX.2, FIBO, Z-Image,
  ERNIE Image, Bonsai, and Wan.
- CLI/Python plan resolution.
- Early option rejection for `--image-strength`.
- Progress task inference fix for image-conditioned edit contexts.
- Focused documentation and backlog completion evidence.

## Non-goals

- Do not implement first-class outpaint/reframe in this item; keep that in planned item 0019.
- Do not remove legacy model-specific CLIs.
- Do not remove `--task edit` immediately; keep it as a deprecated compatibility alias.
- Do not claim new model support without model-backed smoke validation.
- Do not run large model generations just to validate routing.

## Dependencies and related tasks

- [Completed item 0018](0018_taskless_generation_routing.md)
- [Planned item 0019](../planned/0019_first_class_i2i_modes_and_outpaint_reframe.md)
- `src/mflux/task_inference.py`
- `src/mflux/cli/mlx_gen.py`
- `src/mflux/callbacks/generation_context.py`
- `tests/test_task_inference.py`
- `tests/cli/test_mlx_gen_router.py`
- `tests/callbacks/test_progress_callbacks.py`
- `docs/python-integration.md`

## Expected outcomes

- `edit` is no longer returned as a public task by resolver APIs.
- Apps can inspect model capabilities without loading weights.
- Apps can resolve a concrete `GenerationPlan` without invoking the CLI.
- `mlxgen generate` routes edit-capable I2I predictably without requiring `--task edit`.
- Unsupported `--image-strength` combinations fail before backend load.
- Progress subscriptions can target `image-to-image` for edit-conditioned image workflows.

## Validation

- Unit tests for capability descriptors and generation plans across all unified families.
- Router tests for FLUX.2 default edit mode, latent mode with `--image-strength`, exact edit
  models, multi-reference mode, legacy `--task edit`, Wan fixed-task validation, and early
  `--image-strength` rejection.
- Progress callback tests showing explicit image task wins even when `image_strength` is absent.
- CLI capabilities JSON smoke test.
- Lightweight validation artifacts with source/reference image inputs and resolved modes/prompts.

## Progress checklist

- [x] Add capability descriptors and generation plan resolver.
- [x] Update unified CLI routing and capabilities command.
- [x] Update public Python exports and docs.
- [x] Update focused tests.
- [x] Generate lightweight validation artifacts.
- [x] Run tests and review.
- [x] Move this item to completed with evidence.

## Implementation notes

Keep the public task contract small and stable. Treat `image_strength` as a latent-img2img-only
control. Prefer explicit `GenerationPlan` fields over hidden family heuristics, and preserve enough
compatibility for `--task edit` to guide users toward `--i2i-mode edit`.

## Completion report

Completed 2026-06-04.

Implemented:

- Added `GenerationCapability`, `ModelCapabilities`, and `GenerationPlan` in
  `src/mflux/task_inference.py`.
- Added public Python APIs `get_model_capabilities(...)` and `resolve_generation_plan(...)`, and
  kept `infer_task(...)` / `resolve_task(...)` compatible while returning public media-direction
  tasks.
- Added `mlxgen capabilities --model ...` JSON inspection.
- Routed unified CLI generation from `GenerationPlan.handler_id`.
- Added `--i2i-mode` for latent img2img, edit/reference I2I, and multi-reference I2I.
- Kept `--task edit` as a compatibility alias for image-to-image edit/reference mode.
- Made FLUX.2 one-image default route to edit/reference I2I, with `--image-strength` or
  `--i2i-mode latent` selecting latent img2img.
- Added preflight rejection for unsupported `--image-strength`, mask, and outpaint options.
- Fixed metadata replay so a stored positive `image_strength` routes back to latent img2img.
- Added `--base-model` routing support for custom/local paths where the selected backend needs a
  concrete model config.
- Kept Wan fixed T2V/I2V contracts fail-closed before model load.
- Updated edit/fill progress starts so edit-conditioned image workflows emit `task="image-to-image"`.

Key files:

- `src/mflux/task_inference.py`
- `src/mflux/__init__.py`
- `src/mflux/cli/mlx_gen.py`
- `src/mflux/models/flux2/cli/flux2_generate.py`
- `src/mflux/models/flux2/cli/flux2_edit_generate.py`
- `src/mflux/models/z_image/cli/z_image_generate.py`
- `src/mflux/models/bonsai_image/cli/bonsai_image_generate.py`
- `src/mflux/models/flux2/variants/edit/flux2_klein_edit.py`
- `src/mflux/models/qwen/variants/edit/qwen_image_edit.py`
- `src/mflux/models/fibo/variants/edit/fibo_edit.py`
- `src/mflux/models/flux/variants/fill/flux_fill.py`
- `tests/test_task_inference.py`
- `tests/cli/test_mlx_gen_router.py`
- `tests/callbacks/test_progress_callbacks.py`
- `README.md`
- `docs/api.md`
- `docs/python-integration.md`
- `docs/getting-started.md`
- `docs/faq.md`
- `docs/troubleshooting.md`
- `docs/model-management.md`
- `docs/backlog/planned/0019_first_class_i2i_modes_and_outpaint_reframe.md`

Validation:

- `uv run pytest tests/test_task_inference.py tests/cli/test_mlx_gen_router.py tests/callbacks/test_progress_callbacks.py -q`
  passed with 91 tests.
- `uv run ruff check src/mflux/task_inference.py src/mflux/cli/mlx_gen.py src/mflux/__init__.py src/mflux/models/flux2/cli/flux2_generate.py src/mflux/models/flux2/cli/flux2_edit_generate.py src/mflux/models/z_image/cli/z_image_generate.py src/mflux/models/bonsai_image/cli/bonsai_image_generate.py src/mflux/models/flux2/variants/edit/flux2_klein_edit.py src/mflux/models/qwen/variants/edit/qwen_image_edit.py src/mflux/models/fibo/variants/edit/fibo_edit.py src/mflux/models/flux/variants/fill/flux_fill.py tests/test_task_inference.py tests/cli/test_mlx_gen_router.py tests/callbacks/test_progress_callbacks.py`
  passed.
- `uv run mlxgen capabilities --model flux2-klein-4b` emitted schema version 1 JSON with
  `text-only`, `latent-img2img`, `edit-reference`, and `multi-reference` capabilities.
- Lightweight validation artifacts were generated in
  `validation_outputs/generation_capability_contract_2026_06_04/`, including
  `capability_mode_contact_sheet.png` and `route_plan_report.json`.

No model-backed smoke was required for this item because it adds routing, capability inspection,
and option preflight behavior. It does not claim support for a new model backend. New model routes
remain governed by ADR 0001. The broader outpaint/reframe workflow remains planned in item 0019.

## Release-readiness update

Updated 2026-06-04 before the 0.18.10 release.

Current code and docs now agree on the public contract:

- `image-to-image` remains one public media-direction task.
- `latent-img2img`, `edit-reference`, and `multi-reference` are internal modes exposed through
  `mlxgen capabilities` and `resolve_generation_plan(...)`.
- `--image-strength` is a latent-img2img-only control and is rejected for edit/reference and
  multi-reference modes before model load.
- FLUX.2 and dedicated edit checkpoints route one image without `--image-strength` to
  `edit-reference`; one image with `--image-strength` routes to `latent-img2img`; two or more
  images route to `multi-reference` when supported.
- Outpainting/reframing is explicitly not first-class in unified `mlxgen generate` yet and remains
  tracked in planned item 0019.

Additional release validation:

- `uv run python - <<'PY' ... resolve_generation_plan(...) ... PY` confirmed the current planner
  behavior for FLUX.2 text-only, edit-reference, latent-img2img, multi-reference, Wan T2V-A14B, and
  Wan I2V-A14B.
- `uv run mlxgen capabilities --model flux2-klein-4b` emitted schema version 1 JSON with
  `text-only`, `latent-img2img`, `edit-reference`, and `multi-reference`.
- `uv run pytest tests/test_task_inference.py tests/cli/test_mlx_gen_router.py -q` passed with
  83 tests.
- `uv run ruff check src tests` passed.
