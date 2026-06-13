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
  - `actions/checkout@v5`
  - `actions/setup-python@v6`
- `.github/workflows/release.yml` uses:
  - `actions/checkout@v5`
  - `actions/setup-python@v6`
  - `actions/upload-artifact@v6`
  - `actions/download-artifact@v7`
  - `softprops/action-gh-release@v2`
  - `pypa/gh-action-pypi-publish@release/v1`
- Release workflow run `27440684820` for `v0.18.17` originally emitted Node 20 deprecation
  annotations for `actions/checkout@v4`, `actions/setup-python@v5`, and `actions/upload-artifact@v4`.
- PR `#4` migrated those actions and the branch release rehearsal `27443742691` passed.
- Release workflow run `27454332191` for `v0.18.18` still emitted one remaining Node 20 warning
  from `softprops/action-gh-release@v2`.
- The remaining gap is confined to the GitHub Release publication step.

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
- Remove `softprops/action-gh-release@v2` from the release path.
- Keep release semantics unchanged: tag-driven GitHub release creation, trusted PyPI publishing,
  and macOS fast tests must still behave the same.
- Avoid speculative workflow redesign. This is a compatibility pass, not a CI rewrite.
- Prefer a reusable pattern that other repositories can copy without swapping one aging
  JavaScript action for another.

## Suggested implementation

1. Check the current upstream recommended versions for:
   - `actions/checkout`
   - `actions/setup-python`
   - `actions/upload-artifact`
   - `actions/download-artifact`
   - any remaining JavaScript action on the release path
2. Update the workflow files narrowly.
3. Replace `softprops/action-gh-release@v2` with a shell-driven `gh` CLI step that creates the
   release when missing and uploads assets idempotently when the release already exists.
4. If useful, add a temporary opt-in to Node 24 during validation so the repository proves the
   post-switch path before GitHub makes it default.
5. Run the tests workflow and a bounded release workflow rehearsal after the upgrade.

## Scope

- `.github/workflows/tests.yml`
- `.github/workflows/release.yml`
- Minimal workflow metadata or comments needed to explain the migration
- Backlog state files that currently mark this migration complete

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
- No Node 20 deprecation warning from the GitHub Release publish job on the next workflow run.
- One successful post-change tests workflow run.
- One successful post-change release rehearsal or documented reason why a rehearsal was skipped.

## Validation

- GitHub Actions tests workflow succeeds after the version changes.
- Release workflow metadata/build checks succeed after the version changes.
- The new runs no longer emit the current Node 20 deprecation warnings for the updated actions.
- The GitHub Release publication job succeeds with the `gh` CLI replacement and still uploads the
  built distributions.

## Progress checklist

- [x] Audit the current upstream action versions.
- [x] Update `tests.yml`.
- [ ] Replace `softprops/action-gh-release@v2` in `release.yml`.
- [ ] Run and inspect CI.
- [ ] Close the backlog item with the exact run ids and residual risk.

## Guidance for the implementing agent

Treat this as urgent maintenance because the published GitHub deadline is close. Keep the patch
small, verify the actual workflows, and avoid folding unrelated CI opinions into the same change.

## Completion report

### 2026-06-12 partial completion

#### Summary

Migrated the repository workflows off the Node 20 action runtime with a narrow version-only patch.
The implemented strategy was:

- `actions/checkout@v4` -> `actions/checkout@v5`
- `actions/setup-python@v5` -> `actions/setup-python@v6`
- `actions/upload-artifact@v4` -> `actions/upload-artifact@v6`
- `actions/download-artifact@v4` -> `actions/download-artifact@v7`

The workflow graph, runner selection, release semantics, and trusted publishing behavior were left
unchanged.

#### Files changed

- `.github/workflows/tests.yml`
- `.github/workflows/release.yml`
- `docs/backlog/overview.md`
- `docs/backlog/recurrent/0017_backlog_release_hygiene.md`

#### Validation

- PR branch: `ci/node24-migration`
- PR: `#4`
- PR CI run: `27443720109`
- Release workflow rehearsal: `27443742691`

Observed result:

- The release rehearsal succeeded end to end on the branch.
- The upgraded actions no longer emitted the Node 20 deprecation warnings seen in release run
  `27440684820`.
- The PR CI run exercised the upgraded actions successfully, but the overall workflow stayed red
  because `ruff` found pre-existing repository lint issues outside the scope of this migration:
  - `src/mflux/models/ernie_image/cli/ernie_image_generate.py`
  - `src/mflux/models/ernie_image/weights/ernie_image_lora_mapping.py`
  - `src/mflux/models/wan/wan_initializer.py`
  - `src/mflux/models/wan/weights/wan_lora_mapping.py`

#### Reusable strategy

For similar repositories, use the same bounded sequence:

1. identify the specific JavaScript actions named by the deprecation warning;
2. upgrade only those actions to the first Node 24-compatible major lines;
3. avoid redesigning the workflow graph during the migration;
4. validate both the ordinary CI path and a bounded release/publish rehearsal;
5. record unrelated baseline failures separately instead of burying them inside the migration.

### 2026-06-13 reopen note

Release `0.18.18` showed that this item had been closed too early. The workflow still emitted a
Node 20 deprecation warning from `softprops/action-gh-release@v2`, so the record was moved back to
`planned/` for one final cleanup pass. The remaining work is intentionally narrow: replace the
GitHub Release publication action with a Node-runtime-independent `gh` CLI step, rerun CI, and
capture a clean PR that other repositories can copy.
