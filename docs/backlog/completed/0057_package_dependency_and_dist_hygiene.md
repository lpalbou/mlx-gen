# Planned: Package dependency and distribution hygiene

## Metadata
- Created: 2026-06-27
- Status: Completed
- Completed: 2026-06-27

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The audit found packaging hygiene issues that are lower priority than security and release
integrity, but still worth fixing before the next release cleanup pass.

## Current code reality
- `pyproject.toml` names the package `mlx-gen`.
- `Makefile` build checks still target `dist/mflux-*`.
- Runtime dependencies include release-only `twine`.
- Dev dependencies already exist under optional `dev`, so not every tooling dependency is in the
  runtime set.

## Problem
The Makefile can miss or fail against the actual distribution artifact name, and normal users may
install release tooling they do not need.

## What we want to do
Align dist globs with the actual package name and move release-only tooling out of runtime
dependencies.

## Why
Build checks should verify the real artifacts. Runtime dependency footprint should reflect what
ordinary generation users need.

## Requirements
- Replace stale `dist/mflux-*` globs with the current normalized package artifact pattern.
- Move `twine` to a development/release optional dependency group if local release scripts are not
  part of normal runtime.
- Verify `make build` or the equivalent `uv build` plus artifact inspection.

## Suggested implementation
Patch the Makefile globs to match `mlx_gen` / `mlx-gen` build artifacts, then move `twine` under an
appropriate optional dependency extra and ensure release checks still install it in dev workflows.

## Scope
- `Makefile`
- `pyproject.toml`
- Release/build verification notes.

## Non-goals
- Do not redesign GitHub Actions release publishing in this item.
- Do not remove runtime dependencies needed by normal model execution.

## Dependencies and related tasks
- Related: `0052_release_publish_fail_closed.md`

## Expected outcomes
- Build verification inspects the actual `mlx-gen` distribution artifacts.
- Normal installs no longer pull release-only upload tooling.
- Release tooling remains available through a documented development or release extra.

## Validation
- `uv run ruff check pyproject.toml` is not applicable; validate with build commands instead.
- `uv build`
- `make build` if local package dependencies are installed.

## Progress update: 2026-06-27
- Replaced stale `dist/mflux-*` cleanup/extraction with `mlx_gen-*` / `mlx-gen-*` artifact
  handling.
- Moved `twine` out of runtime dependencies and into `dev` plus `release` optional extras.
- Refreshed `uv.lock`.
- Verified `make build` builds and extracts `dist/mlx_gen-0.18.22.tar.gz` and
  `dist/mlx_gen-0.18.22-py3-none-any.whl`.

## Progress checklist
- [x] Fix dist artifact globs.
- [x] Move release-only dependency.
- [x] Verify build artifact inspection.

## Implementation notes
Keep this behind the confirmed blockers. Packaging hygiene matters most when preparing a release or
when `make build` is a gate.
