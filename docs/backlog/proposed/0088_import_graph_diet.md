# Proposed: Import-graph diet for cold CLI dispatch

## Metadata

- Created: 2026-07-22
- Status: Implemented (0.25-track, 2026-07-22)
- Completed: 2026-07-22

## ADR status

- Governing ADRs: None
- ADR impact: None

## Context

Measured 2026-07-22 in a 0.23.1 install (Python 3.12, M5 Max): `import mflux` costs
~1.2 s CPU / ~1.5-6 s wall (up to 14 s under load, cold FS cache ~11.5 s).
`-X importtime` attributes the bulk to `mflux.task_inference` ->
`mflux.python_runtime` -> `mflux.cli.output_paths` -> `mflux.utils.image_util`,
which transitively pulls ALL PIL plugins, `huggingface_hub.hf_api`, `httpx`
(which pulls `rich`). Every cold CLI generation, every `mlxgen capabilities`
probe, and every embedded-host subprocess pays this.

CORRECTION (2026-07-22 re-measurement): this document originally attributed
~1.7 s of one cold trace to the module-scope `mx.compile` decorator in
`flow_match_euler_discrete_scheduler.py:13`. That claim is WRONG: the decorator
only wraps the function at import (~0.2 ms measured; compilation happens on the
first call), and the module's real import cost is `mlx.core` itself (~28 ms
warm). The 1.7 s in the cold trace was FS-cache noise. No first-use move is
needed for it.

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
- ~~Move the module-scope `mx.compile` into first-use.~~ (Dropped: claim was wrong,
  see CORRECTION above.)
- Demote matplotlib to an extra (`mlx-gen[concept]`) with a clear runtime error when
  missing.
- Add a CI regression gate: `python -X importtime -c "import mflux"` budget.

## Implementation record (2026-07-22)

- `cli/output_paths.py` now owns the pure collision-free path logic
  (`resolve_collision_free_path`); `ImageUtil.resolve_output_path` delegates to it,
  so the CLI/runtime dispatch chain no longer imports the image stack.
- `utils/image_util.py`: module-scope `PIL.Image.init()` deleted (PIL
  self-registers plugins on first open/save - verified by a multi-format save
  round-trip); `PIL.ImageDraw`/`PIL.ImageOps`/`piexif` moved into their methods.
- huggingface_hub imports made function-local at all 6 module-scope sites:
  `weight_loader.py`, `tokenizer_loader.py`, `lora_resolution.py`,
  `path_resolution.py`, `lora_compatibility.py`, `cli/mlx_gen.py`.
- `utils/dimension_resolver.py`: PIL.Image imported inside the methods that open
  a reference image (task_inference chain stays PIL-free).
- matplotlib moved to `[project.optional-dependencies] concept`; its sole
  importer `concept_util.py` fails loudly with the `pip install mlx-gen[concept]`
  hint when missing. `uv.lock` regenerated (dev extra still carries matplotlib,
  so the test env keeps it).
- CI gate: `tests/test_import_hygiene.py` asserts a fresh
  `import mflux` loads NO module starting with torch, transformers, tokenizers,
  matplotlib, httpx, huggingface_hub, cv2, av, rich, or PIL.
- Measured result (M5 Max, Python 3.10 venv, warm): `import mflux`
  237-274 ms -> 57-60 ms wall; importtime cumulative 237 ms -> 60 ms, and the
  remaining cost is `mlx.core` (~28 ms). numpy and PIL left the chain entirely
  (numpy is not gated - modules that need it may still import it eagerly).
- Public import surfaces unchanged: `mflux/__init__.py` eager re-exports and
  `mlxgen/__init__.py` eager getattrs untouched (stage-2 laziness out of scope).

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
