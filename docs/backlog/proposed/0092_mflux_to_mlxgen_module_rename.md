# Proposed: Rename the Python module from `mflux` to `mlxgen`

## Metadata

- Created: 2026-07-23
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: None yet
- ADR impact: Needs a short ADR before execution (public import-path break,
  deprecation window policy, and metadata field naming are durable decisions)

## Context

The project renamed distribution and CLI surfaces to MLX-Gen (`mlx-gen` on
PyPI, `mlxgen` console scripts), but the import package is still `mflux`
(`src/mflux`, `import mflux`, `from mflux.python_runtime import ...`). The
0.25-track cycle-1 investigation flagged the mismatch as recurring friction:
embedding hosts mix `mlxgen` commands with `mflux` imports, error strings and
env vars straddle both names (`MFLUX_WAN_BLOCK_HEALTH`,
`MFLUX_PRESERVE_TEST_OUTPUT`), and metadata records `mflux_version`.

## Problem or opportunity

One name end to end removes a real onboarding trap (docs and searches that
find the upstream mflux project instead of MLX-Gen) and makes the fork
boundary explicit. The cost is a breaking import path for every embedding
host and script.

## Proposed direction

- Rename `src/mflux` to `src/mlxgen`; keep a thin `mflux` shim package for at
  least one minor release that re-exports and warns loudly on import.
- Sweep env vars (`MFLUX_*` -> `MLXGEN_*` with fallback reads), metadata keys
  (`mflux_version` alongside a new `mlxgen_version` for one release), cache
  directory names, and user-facing strings.
- Update console-script entry points, tests, docs, and README examples in the
  same wave; CI gates on zero remaining bare `mflux` references outside the
  shim.

## Why it might matter

Hosts embedding the Python runtime (the audit's BlackPixel host chief among
them) currently document two names for one dependency; every new integration
pays the confusion once.

## Promotion criteria

Promote when a release window can absorb a breaking import path (shim
provided), an ADR fixes the deprecation policy, and the 0.25-track wave has
shipped so the rename does not collide with in-flight feature branches.

## Non-goals

Renaming the GitHub repository or the PyPI distribution (already `mlx-gen`);
changing any generation behavior.

## Guidance for future agents

Do the rename as a mechanical, single-commit move plus shim; never mix it
with behavior changes. Grep for `mflux` in metadata readers before dropping
the legacy `mflux_version` key: `--config-from-metadata` must keep replaying
old sidecars.
