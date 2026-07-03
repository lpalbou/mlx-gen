# Bug: `mlxgen generate` advertises `--failure-diagnostics` as common, but only Wan implements it

## Summary

`src/mflux/cli/mlx_gen.py` advertises `--failure-diagnostics` as one of the common generation
options forwarded by `mlxgen generate`. In the current unpublished tree, that statement is too
broad: the router forwards the flag, but the option is only implemented in the Wan backend CLI.

That is a feature-contract mismatch between help text and actual backend coverage.

## Current code reality

- `src/mflux/cli/mlx_gen.py` epilog lists `--failure-diagnostics` among common generation options.
- `_resolve_invocation(...)` forwards unknown options to the selected backend, including
  `--failure-diagnostics`.
- Source search shows the actual option declaration only in
  `src/mflux/models/wan/cli/wan_generate.py`.
- Focused tests only cover Wan routing and Wan failure manifest behavior.

## Why this matters

- A user or embedding app reading `mlxgen generate --help` is told this is a common routed option.
- Non-Wan image backends can still fail when the forwarded flag reaches their backend parser.
- This makes declared feature coverage look broader than it is.

## What I need from `mlx-gen`

Choose one of these and make it explicit:

1. implement `--failure-diagnostics` consistently for all routed backends where it is meant to be
   supported; or
2. narrow the router help text so it only claims support where it actually exists.

## Acceptance criteria

- The routed help text no longer overclaims backend coverage.
- Tests cover both the supported path and an unsupported path, depending on the chosen contract.
- If the option stays Wan-only, docs and help say Wan-only.
- If the option becomes common, non-Wan routed backends accept it and emit a consistent diagnostics
  artifact contract.

## Evidence

- `src/mflux/cli/mlx_gen.py`
- `src/mflux/models/wan/cli/wan_generate.py`
- `tests/cli/test_mlx_gen_router.py`
