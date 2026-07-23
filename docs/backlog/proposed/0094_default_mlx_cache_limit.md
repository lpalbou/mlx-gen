# Proposed: Default MLX buffer-cache limit for Python API and CLI

## Metadata

- Created: 2026-07-23
- Status: Implemented (pending release) — 2026-07-23, cycle-1 implementation wave
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

## Implementation record (2026-07-23)

### Design decision

Machine-derived ladder: `total RAM / DEFAULT_CACHE_LIMIT_RAM_DIVISOR (8)`,
clamped to `[1 GiB, 8 GiB]`, resolved by
`RuntimeMemory.default_cache_limit_bytes()` (pure, injectable RAM size for
tests). Justification of the numbers:

- 8 GiB ceiling: the BlackPixel WorkerCachePolicy number for >=64 GB
  machines, chosen there to cover "the transient buffer working set of the
  largest supported generations (Wan A14B video scenes)" — i.e. the 8 GiB
  precedent was already sized for video, not just images, and pool-rebuild
  churn below it was measured at ~0.5-2 s/scene in that work.
- 1 GiB floor: mirrors the long-standing low-RAM default (1 GB decimal;
  that constant is untouched) and the BlackPixel small-machine bound.
- The smooth RAM/8 ladder lands exactly on both precedents (128 GB -> 8 GiB
  via ceiling, 64 GB -> 8 GiB, 32 GB -> 4 GiB, 16 GB -> 2 GiB) instead of a
  two-step function, so mid-size machines get proportionate caps.
- Task-aware or size-aware defaults were considered and rejected for now:
  the load seam has no task knowledge, `mx.set_cache_limit` bounds only the
  FREE cache (resident weights/activations unaffected), and the one measured
  churn-sensitive workload (Wan A14B) is covered by the ceiling. Revisit if
  a video family measures churn at 8 GiB.

Precedence (documented and tested): explicit `--mlx-cache-limit-gb` >
`MFLUX_MLX_CACHE_LIMIT_GB` env (new; serves bare Python-API hosts) >
low-RAM 1 GB default > machine ladder. `-1` (flag or env) is the explicit
"unlimited" opt-out (`0` stays rejected at the parser to avoid ambiguity —
`mx.set_cache_limit(0)` would disable caching entirely). Application seams:

- CLI runs: unchanged call order (`apply_runtime_memory_options` before
  model construction) now resolves the ladder instead of `None`.
- Bare Python API: `RuntimeMemory.apply_default_cache_limit_once()` runs at
  `WeightLoader.load`/`load_single` — exactly once per process, and never
  after any explicit application (including an explicit unlimited opt-out).
- Visibility (ADR 0002): applying the DEFAULT prints one stderr line naming
  the cap and both override paths; stderr keeps `--json-events` stdout
  machine-readable. Explicit applications stay silent as before.
- SeedVR2 safe video mode clamps `-1` like any over-bound value, with the
  existing visible clamp notice (the certified safe profile is
  memory-bounded by contract).

Coordination note for hosts that call `mx.set_cache_limit` directly
(BlackPixel WorkerCachePolicy): mlx-gen cannot detect a limit set outside
its own seams, so the load-time default would override it (to a practically
identical value on both their ladder rungs). Such hosts should export
`MFLUX_MLX_CACHE_LIMIT_GB` (their value, or `-1`) — flagged to the
BlackPixel-side sibling working this wave.

### Cycle-2 fix (2026-07-23, adversarial review): host-managed limits are detected and preserved

The coordination note above understated the problem: the stomp was real and
demonstrated live (host sets 2 GiB via `mx.set_cache_limit`, then the first
`WeightLoader.load` silently raised it to the 8 GiB ladder value). On
128 GB machines both BlackPixel rungs coincide with the ladder (8 GiB) so
the bug was masked; on <64 GB machines BlackPixel's deliberate 1 GiB cap
would have been raised to 2-8 GiB behind the worker's back.

Fix: MLX 0.31.0 exposes no cache-limit getter, but `mx.set_cache_limit`
RETURNS the previous limit, so applying the default doubles as a probe. A
pre-existing limit at or below HALF of physical RAM is classified as a
deliberate host cap (MLX's own untouched process default measured
~0.95x total RAM — 121.6 GiB on the 128 GiB dev machine — while deliberate
caps are single-digit GiB): the probe restores it, marks the latch
"explicit", and prints one stderr line
(`Respecting pre-existing MLX cache limit: ...`). Anything above the
threshold is treated as the untouched MLX default and the ladder applies as
before. A non-int probe return (older MLX, test stubs) is inconclusive and
falls back to the pre-fix behavior. Explicit callers
(`--mlx-cache-limit-gb`, env var, low-RAM) still win over any pre-existing
value — the probe only guards the DEFAULT path. The env-var coordination
advice above remains valid but is no longer load-bearing.

Verified live both ways (2 GiB host cap respected + restored; untouched
121.6 GiB default replaced by the 8 GiB ladder). Regression tests:
`TestDefaultRespectsHostManagedLimit` (host cap preserved, latch behavior,
untouched-default ladder, explicit-override-wins).

### Measured numbers

- Disease (from tonight's audit, context above): 32.9 GB MLX free cache
  after two Klein 9B q8 generations with no cap; that resident cache is the
  memory pressure that produces the 0093 fault-in regime system-wide.
- Perf-neutrality of the cap on same-shape repeats: not re-benchmarked
  tonight (the 0090 harness run is the promotion gate); the claim rests on
  `mx.set_cache_limit` semantics (free-cache-only) plus the BlackPixel
  precedent that shipped the same 8 GiB bound for the same workloads
  without measured per-scene regression. The 512x512 q8 bench runs tonight
  executed under the new default cap (8 GiB) with second-generation times of
  3.5-3.6 s, consistent between legs.

### Files changed

- `src/mflux/utils/runtime_memory.py` (ladder, env var, precedence,
  apply-once state, stderr notice)
- `src/mflux/models/common/weights/loading/weight_loader.py` (apply-once
  hook)
- `src/mflux/cli/parser/parsers.py` (`cache_limit_gb_value` type: positive
  or exactly `-1`; updated `--mlx-cache-limit-gb` help)
- `src/mflux/models/wan/cli/wan_generate.py` (same type + help)
- `src/mflux/callbacks/callback_manager.py` (negative-value guard in the
  legacy resolve helper)
- `src/mflux/models/seedvr2/cli/seedvr2_upscale.py` (safe-mode clamp covers
  `-1`)
- `tests/utils/test_runtime_memory_cache_limit.py` (new: ladder values,
  precedence, opt-outs, invalid env fail-loud, apply-once state machine,
  visible-notice assertion, CLI value parsing)
- Docs: `docs/api.md`, `docs/python-integration.md`, `CHANGELOG.md`.

### Gates

Cycle-1: `make lint` clean; fast suite 1457 passed / 7 skipped (baseline
1413/7). Cycle-2 (after the host-cap probe fix): `make lint` clean; fast
suite 1465 passed / 7 skipped.
