# Planned: Fail closed on PyPI publish failures

## Metadata
- Created: 2026-06-27
- Status: Completed
- Completed: 2026-06-27

## ADR status
- Governing ADRs: ADR 0002
- ADR impact: None

## Context
The local release manager publishes to PyPI before creating git and GitHub release artifacts. That
ordering is good only if publish failures stop the release.

## Current code reality
- `src/mflux/release/release_manager.py` calls `PyPIPublisher.publish_to_pypi()` before
  `GitOperations.create_and_push_tag()` and `GitHubAPI.create_github_release()`.
- `src/mflux/release/pypi_publisher.py` catches non-duplicate `TwineException`, `OSError`,
  `ValueError`, and `RuntimeError`, prints warnings, and returns.
- The release manager cannot tell the upload failed and continues to tag and release.

## Problem
A failed or partial PyPI upload can be recorded as a completed git/GitHub release. This creates a
split-brain release state for users and maintainers.

## What we want to do
Make PyPI upload failures fatal unless the version already exists or an explicit future recovery
mode is added.

## Why
Release artifacts are public coordination points. A failed package upload should not be hidden by a
successful tag or GitHub release.

## Requirements
- Re-raise non-duplicate Twine failures.
- Re-raise unexpected upload exceptions with enough context for recovery.
- Keep the existing already-exists idempotency path.
- Add focused tests for fatal and idempotent behavior.

## Suggested implementation
Change `_upload_to_pypi` to return only on successful upload or an already-existing version.
For all other caught errors, raise `RuntimeError` or re-raise with context.

## Scope
- `src/mflux/release/pypi_publisher.py`
- Focused release publisher tests.

## Non-goals
- Do not add a partial-release recovery CLI unless needed after the fail-closed fix.

## Dependencies and related tasks
- Related: recurrent backlog release hygiene item `0017_backlog_release_hygiene.md`.

## Expected outcomes
- A PyPI upload failure stops local release execution before tag/release creation.
- Duplicate existing uploads remain idempotent.

## Validation
- `uv run pytest tests/release/test_pypi_publisher.py -q`
- `uv run ruff check src/mflux/release/pypi_publisher.py tests/release/test_pypi_publisher.py`

## Progress update: 2026-06-27
- Made non-duplicate Twine upload failures fatal.
- Made unexpected upload exceptions fatal.
- Preserved the existing already-exists idempotency path.
- Added ReleaseManager-level coverage proving PyPI failure stops before tag/GitHub release
  creation.
- Reordered the GitHub Actions release workflow so PyPI publish succeeds before the GitHub Release
  is created.
- Clarified local PyPI failure wording to distinguish uncertain PyPI state from tag/release state.

## Progress checklist
- [x] Make non-duplicate upload failures fatal.
- [x] Preserve already-existing behavior.
- [x] Add focused tests.
- [x] Cover ReleaseManager fail-closed propagation.
- [x] Align GitHub Actions publish ordering with the fail-closed release graph.

## Implementation notes
Keep the release graph conservative. If the package upload state is uncertain, force explicit human
recovery instead of silently advancing release state.
