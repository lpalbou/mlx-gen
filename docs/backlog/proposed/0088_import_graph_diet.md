# Proposed: Import-graph diet for cold CLI dispatch

## Metadata

- Created: 2026-07-22
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: None

## Context

Measured 2026-07-22 in a 0.23.1 install (Python 3.12, M5 Max): `import mflux` costs
~1.2 s CPU / ~1.5-6 s wall (up to 14 s under load, cold FS cache ~11.5 s).
`-X importtime` attributes the bulk to `mflux.task_inference` ->
`mflux.python_runtime` -> `mflux.cli.output_paths` -> `mflux.utils.image_util`,
which transitively pulls ALL PIL plugins, `huggingface_hub.hf_api`, `httpx`, and a
module-scope `mx.compile` in `flow_match_euler_discrete_scheduler.py:13` (~1.7 s
self-time in one cold trace). Every cold CLI generation, every `mlxgen capabilities`
probe, and every embedded-host subprocess pays this.

matplotlib is NOT on the import hot path (only
`models/flux/variants/concept_attention/concept_util.py:1`) but remains a HARD
install dependency (pyproject) used solely for concept-attention heatmaps —
packaging bloat for every consumer.

## Problem or opportunity

1-3+ s of avoidable latency on every cold dispatch; a heavyweight install for a
feature most users never touch.

## Proposed direction

- Lazy-import huggingface_hub/httpx inside the functions that need them; import PIL
  plugins on demand (PIL.Image without the full plugin registration where possible).
- Move the module-scope `mx.compile` into first-use.
- Demote matplotlib to an extra (`mlx-gen[concept]`) with a clear runtime error when
  missing.
- Add a CI regression gate: `python -X importtime -c "import mflux"` budget.

## Why it might matter

Directly reduces every embedded-host cold path and capability probe; smaller
install footprint.

## Promotion criteria

Promote when an embedding host measurably depends on cold-dispatch latency (the
BlackPixel warm-worker work reduces urgency; capability probes remain).

## Validation ideas

importtime budget test; capabilities CLI wall-time before/after; extras install
matrix (with/without concept extra).

## Non-goals

Rewriting the CLI router; changing public import surfaces.

## Guidance for future agents

Watch for tokenizers/transformers creeping into module scope with new model
families — the budget gate should catch it.
