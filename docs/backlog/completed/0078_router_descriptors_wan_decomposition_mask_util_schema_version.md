# 0078 - Router Option Descriptors, Wan Decomposition Phase 1, MaskUtil, Metadata Schema Version

- Status: completed (2026-07-05)
- Scope: `mlxgen generate` router, shell completions, Wan runtime structure, user-mask loading,
  metadata sidecar contract
- Validation: 2 adversarial design reviews + 2 adversarial code verifications (both SHIP),
  full no-weights band green (1275 passed), lint clean

## Problem

Three structural debts, all sources of shipped bugs or consumer gaps:

1. One router flag lived in six hand-synced copies; the mechanism shipped two bugs in one
   release (`--video-strength` consumed-but-dropped; `steps` replay shrinkage).
2. `wan2_2_ti2v.py` (1571 lines) duplicated its save-branch metadata in twin ~45-line blocks
   (where the `steps` bug lived) and mixed a 40-line validation head into `generate_video`.
3. Four user-mask loaders with divergent resampling and no shared code; metadata sidecars grew
   8+ ad-hoc keys with no version marker while capabilities JSON is versioned.

## What shipped

1. Router option descriptors (`src/mflux/cli/router_options.py`): one table drives both the
   `mlxgen generate` parser construction and the re-emission of consumed flags, with explicit
   `ForwardPolicy` per option (ROUTER_ONLY / REEMIT_VALUE / REEMIT_FLAG / TRANSFORMED with a
   named emitter). New tests: a completeness test (every parser action maps 1:1 to a descriptor
   with a declared fate), round-trip tests (argv and metadata sources), an exact-argv block
   test pinning emission order, and pass-through guards.
2. Live bugs found and fixed by the process:
   - `mlxgen generate --debug` was silently dropped (third instance of the class); Wan CLI
     gained `--debug` wired to LoRA debug logging so the flag is safe on every route (verified
     per-route by the adversarial verifier).
   - Metadata-sourced `video_strength` was validated only after the multi-minute weight load;
     the router now backfills + re-emits it so the backend parser rejects invalid values at
     parse time.
   - `mflux-completions` crashed on EVERY invocation at HEAD (duplicate `add_lora_arguments`
     in the upscale-controlnet branch -> `ArgumentError`); `mflux-generate-z-image` completion
     was silently empty. Both fixed; completions now exist for wan/ernie/bonsai (generated
     from the real entrypoint parsers after mechanical `_parser()` extraction), and a truth
     test asserts every `pyproject` console script has a completion or a documented exclusion
     (the subcommand-based `mlxgen` router aliases are excluded until subcommand-aware
     completion exists).
3. Wan decomposition phase 1: `WanVideoRequest.resolve` (new `wan_video_request.py`) executes
   the validation/resolution head through the model's helpers (all instance monkeypatches
   survive; scheduler creation and timestep truncation stay in the runtime - the truncation
   mutates scheduler state via `set_begin_index`); the twin metadata blocks collapsed into
   `_to_video_shared_kwargs` with per-branch `generation_time` evaluation and absent-by-default
   decode extras. Verified byte-identical metadata; `eq=False` on the request (mx.array field).
4. `MaskUtil.load_binary_mask` centralizes the four user-mask loaders. Policy corrected by the
   adversarial design review: the NEAREST/BOX difference is NOT drift - Qwen's NEAREST matches
   its diffusers reference; the shipped policy is "reference-ported surfaces keep the
   reference's resampling; in-house surfaces default to BOX + 0.5". Pixel behavior unchanged
   (verified empirically byte-identical); alpha-channel warning unified across surfaces; the
   per-surface policies are pinned by tests.
5. `metadata_schema_version: 1` (constant in `metadata_schema.py`) ships first in both image
   and video metadata, sidecar and embedded, with the additive-only policy documented in
   `docs/api.md`.

## Deliberately out of scope

Backend parsers, planner constraints, and metadata replay remain per-backend copies (stated in
the changelog); the Wan denoise-loop/strategy seam is deferred to open any VACE port; numeric
mask unification will likely never happen (reference fidelity beats uniformity).
