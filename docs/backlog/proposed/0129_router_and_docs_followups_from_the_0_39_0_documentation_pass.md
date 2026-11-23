# Proposed: Router and docs follow-ups from the 0.39.0 documentation pass

## Metadata

- Created: 2026-11-22
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [0002](../../adr/0002_no_silent_automatic_fallbacks.md) (a request is refused with
  the fix that works, not routed to something else),
  [0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md).
- ADR impact: none.

## Context

The 0.39.0 pass ([item 0128](../completed/0128_minimax_h3_quantized_packages.md)) drafted the
package agent skill and checked every statement it makes against the code. That check found seven
places where the user docs contradicted the code, which were corrected in the same pass, and the
defects below, which were left because each changes behavior or a message and deserves its own
test.

## Findings left open

1. **The "could not infer a backend" error names a fix that does not work.**
   `task_inference.py` answers an unknown model name with "Could not infer a supported backend from
   model ... Pass family='qwen', ...". Passing a family then fails with "family=... is not enough to
   configure model". The fix that works is `--base-model <alias>`. The docs now say so; the message
   should too.
2. **A local folder publishes fewer routes when `--base-model` names a base Klein entry.** A folder
   with `--base-model flux2-klein-base-4b` reports `flux2.text` and `flux2.edit`, while the alias
   and the `AbstractFramework` package report six routes. Without the hint the same folder gets the
   distilled route set with `supports_negative_prompt: false`.
3. **Wan ignores `--base-model` when loading a local folder** (`wan_generate.py`): the folder name
   must carry a Wan alias. Either honor the flag or refuse it.
4. **`mlxgen prepare` still describes Bernini as a "pinned factored BF16 source route"** and names
   only `bernini-r-1.3b` (`models/common/cli/save.py`). "Factored" also remains in
   `docs/README.md`, `docs/architecture.md` and the measurement rows of `docs/recommendations.md`.
5. **Bernini examples use the source alias while the docs recommend downloading the BF16 package.**
   `docs/bernini.md` and `docs/api.md` download `bernini-r-1.3b-bf16` and then generate with
   `--model bernini-r-1.3b`; the two aliases pin different repositories, so that sequence asks for a
   second download. One sentence now says to use the same alias; the examples should be rewritten.
6. **`docs/troubleshooting.md` says the FLUX.2 contract is to omit `--negative-prompt` entirely**,
   while base Klein accepts it (stated lower on the same page).
7. **`docs/api.md` Wan section says "at the default 24 fps"**, which holds for TI2V-5B only; A14B and
   Bernini default to 16.
8. **`docs/api.md` describes `mlxgen upscale` as SeedVR2 only**; SwiftVR is hosted there too.
9. **`--battery-percentage-stop-limit` exits 0 with no image** when it stops a run, and Wan parses
   the flag without using it.
10. **The download disk-space preflight treats optional download patterns as required** when it
    decides whether a snapshot is already cached (`mlx_gen.py`, `_preflight_download_space`). It has
    no effect today because MiniMax-H3, the one family with optional patterns, declares no expected
    size.
11. **`mlxgen capabilities` publishes `recommended_quantize: 8` for a prepared package id**, where
    `--quantize` does not apply.

## Promotion criteria

Promote the message and routing items (1 to 3, 9) together when the router is next touched; the
documentation items (4 to 8) can ride with any docs pass.
