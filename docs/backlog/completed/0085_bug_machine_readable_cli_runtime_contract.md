# 0085 - Bug: embedding apps lacked a machine-readable CLI runtime contract

- Status: completed (resolved in 0.19.0; filed 2026-06-30 at the repo root, moved into the
  backlog during the 2026-07-15 hygiene pass)
- Resolution: `--json-events` shipped on `mlxgen generate` and `mlxgen upscale` (0.19.0
  changelog "Machine-readable runtime events"): JSONL events on stdout with human text on
  stderr, authoritative routed command/model identity, step-based progress, terminal
  `complete` only after the artifact is written, `diagnostics_path` on failure manifests,
  `remediation`/`download-required`/`cli-usage` kinds, documented in `docs/api.md`
  ("CLI Runtime Events") and covered by focused CLI tests. All acceptance criteria from the
  original report below are met.

## Original report (2026-06-30)

## Summary

The unpublished tree materially improves route authority and Python-side progress semantics, but it
still does not give CLI consumers a structured runtime/event contract. That means apps like
BlackPixel still have to parse human stdout, infer phases, and guess terminal state from backend
logs.

This is the main remaining ownership gap between `mlx-gen` and embedding apps.

## What is already better

- `src/mflux/task_inference.py` now keeps stronger canonical model identity and route truth.
- `src/mflux/callbacks/generation_context.py` now has explicit terminal handling through
  `complete()` and `failed()`.
- The focused runtime-contract tests now pass, including task-label and terminal-event coverage.

## Why that is still not enough

- Those improvements primarily help Python callers using `model.callbacks.subscribe_progress(...)`.
- CLI embedding apps still do not get a generic JSON or JSONL event stream for start, progress,
  artifact-ready completion, failure, interruption, or diagnostics output.
- As a result, the app layer still needs convenience heuristics for things that should be emitted
  by the authoritative runtime.

## What I need from `mlx-gen`

Add a stable machine-readable CLI event mode for routed generation and upscale commands.

## Requested direction

- Add an opt-in CLI flag such as `--json-events` or `--event-stream jsonl`.
- Emit a documented event schema for:
  - `start`
  - progress / denoise updates
  - phase transitions where relevant
  - `complete`
  - `failed`
  - `interrupted`
- Include authoritative route/task identity and artifact paths in terminal events.
- Ensure terminal success is emitted only after the saved artifact is ready for consumers.

## Acceptance criteria

- A CLI consumer can drive progress UI and terminal state without parsing human log text.
- Structured events include task identity, model identity, phase, progress counters, and output
  artifact paths.
- Failure events include the diagnostics manifest path when available.
- The schema is documented and covered by focused tests.

## Evidence

- `docs/adr/0003_runtime_truth_vs_consumer_convenience.md`
- `docs/api.md`
- `docs/python-integration.md`
- `src/mflux/callbacks/generation_context.py`
- `src/mflux/task_inference.py`

