# Planned: GitHub Actions Node 24 migration

## Metadata

- Created: 2026-06-12
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: None. This is CI maintenance unless the workflow contract itself changes.

## Context

MLX-Gen release `0.18.17` completed successfully, but the GitHub Actions run emitted a time-bound
warning: the repository still uses JavaScript actions running on Node 20, and GitHub will force
Node 24 by default starting 2026-06-16. That is close enough to treat as planned maintenance, not
as a distant cleanup note.

## Current code reality

- `.github/workflows/tests.yml` uses:
  - `actions/checkout@v4`
  - `actions/setup-python@v5`
- `.github/workflows/release.yml` uses:
  - `actions/checkout@v4`
  - `actions/setup-python@v5`
  - `actions/upload-artifact@v4`
  - `actions/download-artifact@v4`
  - `softprops/action-gh-release@v2`
  - `pypa/gh-action-pypi-publish@release/v1`
- Release workflow run `27440684820` for `v0.18.17` succeeded, but GitHub emitted Node 20
  deprecation annotations for `actions/checkout@v4`, `actions/setup-python@v5`, and
  `actions/upload-artifact@v4`.
- The repository does not currently track this warning in backlog, docs, or CI comments.
- There is no explicit workflow-level opt-in to Node 24 and no recorded compatibility pass after
  the warning.

## Problem

The release and test workflows are operational today, but they are sitting on a published platform
deprecation with a near-term date. If GitHub's default switch exposes an action incompatibility,
MLX-Gen can lose CI signal or release automation even though the package code is fine.

## What we want to do

Make the GitHub Actions workflows explicitly safe for the Node 24 transition before the platform
forces the change.

## Why

This is release-path infrastructure. A working package is not enough if the test and publish
automation become flaky or fail under the next runner default.

## Requirements

- Audit every JavaScript action used by the current workflows.
- Upgrade to Node 24-compatible action versions where available.
- Keep release semantics unchanged: tag-driven GitHub release creation, trusted PyPI publishing,
  and macOS fast tests must still behave the same.
- Avoid speculative workflow redesign. This is a compatibility pass, not a CI rewrite.
- Record the result in backlog and changelog/release notes only if behavior visible to maintainers
  changes.

## Suggested implementation

1. Check the current upstream recommended versions for:
   - `actions/checkout`
   - `actions/setup-python`
   - `actions/upload-artifact`
   - `actions/download-artifact`
2. Update the workflow files narrowly.
3. If useful, add a temporary opt-in to Node 24 during validation so the repository proves the
   post-switch path before GitHub makes it default.
4. Run the tests workflow and a bounded release workflow rehearsal after the upgrade.

## Scope

- `.github/workflows/tests.yml`
- `.github/workflows/release.yml`
- Minimal workflow metadata or comments needed to explain the migration

## Non-goals

- Do not redesign the release flow.
- Do not change package versioning, tagging, or trusted publishing policy.
- Do not treat unrelated CI cleanup as part of this item.

## Dependencies and related tasks

- Release run `27440684820`
- `.github/workflows/tests.yml`
- `.github/workflows/release.yml`

## Expected outcomes

- No Node 20 deprecation warnings on the maintained workflow versions, or a clearly documented
  bounded exception if one action cannot be upgraded immediately.
- One successful post-change tests workflow run.
- One successful post-change release rehearsal or documented reason why a rehearsal was skipped.

## Validation

- GitHub Actions tests workflow succeeds after the version changes.
- Release workflow metadata/build checks succeed after the version changes.
- The new runs no longer emit the current Node 20 deprecation warnings for the updated actions.

## Progress checklist

- [ ] Audit the current upstream action versions.
- [ ] Update `tests.yml`.
- [ ] Update `release.yml`.
- [ ] Run and inspect CI.
- [ ] Close the backlog item with the exact run ids and residual risk.

## Guidance for the implementing agent

Treat this as urgent maintenance because the published GitHub deadline is close. Keep the patch
small, verify the actual workflows, and avoid folding unrelated CI opinions into the same change.
