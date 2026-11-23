# Release

MLX-Gen publishes Python distributions to PyPI and attaches the same distributions to a GitHub Release through `.github/workflows/release.yml`.

## Required GitHub And PyPI Setup

PyPI trusted publishing must be configured for:

- PyPI project: `mlx-gen`
- GitHub repository: `lpalbou/mlx-gen`
- Workflow filename: `release.yml`
- GitHub environment: `pypi`

The workflow uses OpenID Connect through `pypa/gh-action-pypi-publish`; it does not require a long-lived PyPI API token.

The workflow uses the `pypi` environment for trusted publishing, then uses the `github-release`
environment only after PyPI publication succeeds. Configure environment protection rules in GitHub
if releases should require manual approval.

## Release Checks

The release workflow:

- runs fast tests on macOS;
- validates package name, version, changelog entry, tag name, and duplicate PyPI versions;
- builds distributions with `uv build`;
- validates distributions with `twine check`;
- uploads build artifacts;
- publishes the built artifacts to PyPI through trusted publishing;
- creates a GitHub Release with the same wheel and source distribution only after PyPI succeeds.
- asks Context7 to re-index the user docs after PyPI succeeds (see [Context7](#context7)).

## Rehearsal

Run a non-publishing rehearsal from GitHub Actions:

```sh
gh workflow run release.yml \
  -f version=<version> \
  -f publish=false
```

This validates the release metadata and builds artifacts, but does not create a GitHub Release or publish to PyPI.

## Publish

The preferred release path is a version tag:

```sh
git tag v<version>
git push origin main --tags
```

Publishing can also be triggered manually when the target commit has already been pushed:

```sh
gh workflow run release.yml \
  -f version=<version> \
  -f publish=true \
  -f publish_confirmation=publish-mlx-gen-<version>
```

For manual publishing, the confirmation string must match `publish-mlx-gen-<version>`.

## Context7

[Context7](https://context7.com) serves library documentation to coding agents. After a release
reaches PyPI, the `context7-refresh` job of `release.yml` asks it to re-index this repository, so
agents read the docs of the version just published. `context7.json` at the repository root names
what is indexed: `docs/` without the backlog, ADRs and assets, plus the root `README.md`, `llms.txt`
and `llms-full.txt`.

- The job runs `tools/context7_refresh.py`. It refreshes the library `/lpalbou/mlx-gen`, and submits
  the repository the first time, when Context7 does not know it yet.
- The credential `CONTEXT7_API_KEY` lives in the GitHub environment `context7`. Without it the job
  warns and passes; it never blocks a release.
- Context7 re-indexes a project at most once every 10 days. A release inside that window ends with a
  notice and Context7 keeps the previous index until its own schedule or the next refresh. Any other
  refusal fails the job, after PyPI and the GitHub Release are already done.
- The refresh reads the default branch, not the tag, so the release commit must be on `main` when
  the tag is pushed.

For a docs-only change, run the same step by hand once the docs are final:

```sh
gh workflow run context7-refresh.yml
```

Because of the 10-day window, a manual refresh shortly before a release leaves that release
unindexed. Refresh once, with the content you want agents to read.
