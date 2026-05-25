# Contributing

MLX-Gen is an independent project forked from mflux. Contributions should preserve the current user-facing command surface while keeping compatibility with the inherited mflux runtime where practical.

## Development Setup

Use `uv` for local development:

```sh
uv sync --extra dev
```

Run commands from the repository root with `uv run`.

## Useful Checks

Before opening a pull request, run focused checks for the files you changed:

```sh
uv run ruff check src tests
MFLUX_PRESERVE_TEST_OUTPUT=1 uv run pytest -q
```

For documentation-only changes, check that links and command examples match the current CLI help.

## Documentation Expectations

User-facing behavior should be documented in the core docs:

- `README.md` for installation, quick start, and project scope.
- `docs/getting-started.md` for first-run workflows.
- `docs/api.md` for public CLI and Python integration boundaries.
- `docs/model-management.md` for download and prepare behavior.
- `docs/troubleshooting.md` for common runtime failures.

Use `mlxgen` commands in new documentation. Compatibility entry points inherited from mflux may remain available, but they should not be the primary workflow for new MLX-Gen users.

## Pull Request Scope

Keep pull requests focused. Separate compatibility fixes, model behavior changes, documentation repairs, and release automation changes when they can be reviewed independently.

When changing model loading, quantization, or routing behavior, include tests or a clear explanation of the manual validation needed for model-sized workflows.
