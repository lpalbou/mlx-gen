# Proposed: Default MLX buffer-cache limit for Python API and CLI

## Metadata

- Created: 2026-07-23
- Status: Proposed (from the 2026-07-23 BlackPixel i2i latency audit)
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
  (a silent new default must be documented and overridable, not hidden)
- ADR impact: possibly a small ADR note fixing the default-cap policy

## Context

`RuntimeMemory.resolve_cache_limit_bytes(None, low_ram=False)` returns None,
and `CallbackManager` only applies a limit when `--mlx-cache-limit-gb` is
passed or low-RAM mode picks the 1 GB default. A bare Python-API consumer
(`load_generation_model(...).generate_outputs(...)` or direct model classes)
never gets ANY cache limit. Measured 2026-07-23: after two Klein 9B q8 edit
generations at 1024x768 in one process, the MLX buffer cache had grown to
32.9 GB and stays there — memory that competes with other processes' page
cache (including the model weights themselves), recreating the cold fault-in
tax system-wide and feeding the within-batch slowdown the BlackPixel audit
measured (218 -> 310 s/image across one warm 4-seed batch at 944x1680).

Precedents already in the ecosystem: low-RAM mode defaults to 1 GB here, and
BlackPixel's worker (its backlog 0054-A) now sets 8 GiB on >=64 GB machines
and 1 GiB below — but that only covers BlackPixel's worker path at its HEAD;
bare API users, `mlxgen` CLI runs, and BlackPixel's own cold-CLI jobs
(masked/outpaint) still run uncapped.

## Proposal

Pick and document a default cache limit applied at model-load/runtime-init
time when the caller has not set one, e.g. machine-size derived (8 GB at
>=64 GB total RAM, 2-4 GB at 32-64 GB, 1 GB below — mirroring the two
existing precedents), always overridable via the existing
`--mlx-cache-limit-gb` / constructor argument, including an explicit
"unlimited" opt-out. `mx.set_cache_limit` bounds only the FREE cache
(reclaim happens at the next allocation), so resident weights and in-flight
activations are unaffected; the risk is limited to some buffer-pool rebuild
cost for workloads that alternated between very different tensor shapes.

## Expected impact

Bounds multi-generation processes to the cap instead of tens of GB
(measured 32.9 GB after two generations), protecting page cache on shared
machines. Per-generation speed is expected neutral for same-shape repeats
(the bounded pool still serves them); validate that claim with the 0090
benchmark harness before promoting.

## Promotion criteria

Promote when (a) a benchmark shows same-shape repeat generations are not
measurably slowed by the cap on at least one image and one video family,
and (b) the default is written into docs and `--help` so it is not a silent
behavior change (ADR 0002).
