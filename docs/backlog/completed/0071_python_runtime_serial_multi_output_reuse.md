# Completed: Python runtime serial multi-output reuse

## Metadata
- Created: 2026-06-30
- Status: Completed
- Completed: 2026-06-30

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
Embedding applications needed MLX-Gen to own the shared Python execution contract for several
outputs on the unified `mlxgen generate` families. Before this work, apps had to resolve routes,
load model classes, loop over seeds, derive output names, and decide collision behavior outside
the package.

## Problem
The CLI already had a seed-driven multi-output contract, but the Python integration surface did
not expose the same ownership boundary. That left warm-worker integrations duplicating seed loops,
save logic, and per-seed path handling in app code.

## What we changed
- Added route-resolved runtime loading through `resolve_generation_runtime(...)` and
  `load_generation_model(...)`.
- Added `generate_output(...)` and `generate_outputs(...)` on the loaded runtime wrapper for the
  unified `mlxgen generate` families.
- Reused the public seed/output naming contract in Python, including `_seed_<seed>` suffixing and
  predictable collision handling when `overwrite=False`.
- Kept SeedVR2 outside this wrapper so image/video restore remains on direct `SeedVR2` methods.

## Requirements
- One loaded model instance can generate several outputs without reloading the model for each seed.
- Output naming and overwrite behavior must match the public CLI contract.
- Per-seed image or video outputs must stay exact relative to separate single-output calls.
- The wrapper must not imply tensor batching where the runtime still executes one seed at a time.

## Scope
- Public Python runtime planning/loading helpers.
- Loaded runtime multi-output execution helpers.
- Image and video save-path collision behavior.
- Focused tests plus one-at-a-time real reuse-vs-reload benchmarks.

## Non-goals
- Do not implement tensor batching.
- Do not broaden the wrapper to SeedVR2/upscale in this item.
- Do not accept output-quality drift to save time or memory.

## Validation
- Focused contract tests:
  - `tests/test_python_runtime.py`
- Dedicated benchmark harness:
  - `tools/python_runtime_multi_output_benchmark.py`
- Published evidence:
  - [python_runtime_multi_output_reuse_report.md](../../assets/validation/python-runtime-multi-output-2026-06-30/python_runtime_multi_output_reuse_report.md)

## Measured outcome
The published reuse-vs-reload validation covers four real routes:

- Qwen masked edit: exact image parity, reuse `0.80%` faster, peak RSS `1.35%` lower.
- FLUX.2 multi-reference edit: exact image parity, reuse `17.34%` faster, peak RSS `1.73%` lower.
- Wan A14B image-to-video: exact frame parity, reuse `35.80%` faster, peak RSS `10.22%` lower.
- Z-Image Turbo `1024x1024` generation: exact image parity, reuse `1.78%` faster, peak RSS
  `2.11%` lower.

The evidence shows that the wrapper is valuable where reload and route setup cost dominate, while
still preserving exact output quality on flatter routes.

## Completion report

- Date: 2026-06-30
- Summary: MLX-Gen now owns the shared Python multi-output execution contract for the unified
  `mlxgen generate` families through a route-resolved loaded runtime wrapper with exact per-seed
  output parity and overwrite-safe save handling.
- Implementation:
  - `src/mflux/python_runtime.py`
  - `src/mflux/__init__.py`
  - `src/mlxgen/__init__.py`
  - `tests/test_python_runtime.py`
  - `tools/python_runtime_multi_output_benchmark.py`
  - `docs/python-integration.md`
- Validation:
  - `uv run pytest tests/test_python_runtime.py -q`
  - `uv run ruff check tests/test_python_runtime.py tools/python_runtime_multi_output_benchmark.py`
  - `uv run python tools/python_runtime_multi_output_benchmark.py --report-only`
- Evidence report: [python_runtime_multi_output_reuse_report.md](../../assets/validation/python-runtime-multi-output-2026-06-30/python_runtime_multi_output_reuse_report.md)
- Residual risk: this surface is serial multi-output reuse, not tensor batching. SeedVR2 remains a
  separate Python surface by design.
