# Planned: Low-RAM repeat-generation safety

## Metadata
- Created: 2026-06-27
- Status: Completed
- Completed: 2026-06-27

## ADR status
- Governing ADRs: ADR 0002
- ADR impact: None

## Context
`--low-ram` deletes encoders to reduce memory pressure. Some CLIs run multiple seeds in the same
process and may reread prompt files between generations.

## Current code reality
- `src/mflux/callbacks/instances/memory_saver.py` deletes text encoders, image encoders, depth
  models, VLM encoders, and a VLM tokenizer entry.
- The code comment says repeated generation only works with the same prompt cache.
- Flux and Qwen encode prompts before the deletion callback and can survive same-prompt cache hits.
- FIBO and dynamic prompt-file runs can still need a deleted encoder on a later generation.
- CLIs loop over `args.seed` and `PromptUtil.read_prompt(args)` rereads prompt files each time.

## Problem
Valid multi-seed or dynamic prompt-file runs can fail after the first output when `--low-ram`
removes encoders that later prompt encoding needs.

## What we want to do
Make unsupported low-RAM repeat-generation combinations fail early, or keep/precompute the required
encoders before deletion.

## Why
Low-RAM mode is used when reruns are expensive. It should not fail late after producing only part
of a requested batch.

## Requirements
- Detect repeat-generation cases that need prompt/image encoders after the first run.
- Either precompute immutable embeddings/prompts before encoder deletion or reject unsafe
  combinations with a clear parser error.
- Preserve the single-generation low-RAM memory-saving path.
- Add focused tests for multi-seed prompt-file behavior.

## Suggested implementation
Start conservatively in `CallbackManager._register_memory_saver`: if `--low-ram` is combined with
multiple seeds and a prompt file or a model family without a reliable prompt cache, reject early.
Then consider a richer precompute path later.

## Scope
- Callback registration or CLI validation for low-RAM repeat-generation cases.
- Focused callback/parser tests.

## Non-goals
- Do not reload encoders mid-run in this item.
- Do not reduce Wan A14B boundary memory behavior unless affected by the same callback.

## Dependencies and related tasks
- Related: `0013_wan_a14b_boundary_memory_recovery.md`

## Expected outcomes
- Unsafe low-RAM multi-generation requests fail before generation starts.
- Safe single-generation low-RAM behavior remains unchanged.

## Validation
- `uv run pytest tests/callbacks/test_memory_saver.py tests/cli/test_mlx_gen_router.py -q`
- `uv run ruff check src/mflux/callbacks tests/callbacks`

## Progress update: 2026-06-27
- Added parser-level rejection for `--low-ram` plus multiple seeds plus `--prompt-file`.
- Added parser-level rejection for FIBO-family `--low-ram` plus multiple seeds, where the released
  text encoder would be needed again.
- Covered both guards in fast parser tests.

## Progress checklist
- [x] Define unsafe repeat-generation predicates.
- [x] Add early rejection or prompt precompute support.
- [x] Cover prompt-file and multi-seed behavior.

## Implementation notes
Prefer fail-closed validation before a complex lifecycle change. Keep the memory-saving contract
explicit for users.
