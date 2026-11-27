# Completed: warn on silent UMT5 prompt truncation (Wan routes)

## Metadata

- Created: 2026-07-25
- Status: Completed
- Completed: 2026-07-25 — released in 0.25.0 (tag `v0.25.0` from `2452f0c`,
  workflow 30162410505 green; PyPI + GitHub Release verified). The 0101
  smoke ran both 0.24.0 and 0.25.0 with `--no-prompt-cache` so the rewritten
  encode-measurement path was exercised end to end; output bitwise identical.
- Effort: S

## ADR status

- Governing ADRs: none directly; aligns with
  [ADR 0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md) (the
  runtime should tell the truth about what it actually conditioned on).
- ADR impact: None.

## Context (investigation provenance)

Verified in the 2026-07-25 storyboard consistency investigation: Wan prompt
encoding tokenizes with `truncation=True` at `max_length=512`
(`_tokenize_prompts` in `src/mflux/models/wan/variants/wan2_2_ti2v.py`) and
nothing checks whether the raw prompt exceeded the budget. A prompt beyond
512 UMT5 tokens is silently right-truncated: the user's trailing action or
style text simply never conditions the video, with no signal anywhere (CLI,
metadata, events).

This matters more now that hosts are steered toward PREPENDING board-level
world/style anchors to every scene prompt (the highest-leverage consistency
fix from the same investigation — measured anchor+longest-scene-prompt was 74
tokens on the real film, so the headroom is large, but power users stacking
anchors, subject sheets, and long action text can reach the cap).

## What we want to do

1. Detect overflow at encode time (tokenize once without the cap, or check
   `attention_mask` saturation plus an uncapped length probe) and emit ONE
   clear warning per generate call naming the dropped token count.
2. Record `prompt_tokens` / `prompt_truncated` in the metadata sidecar so
   hosts can surface it.
3. Same for the negative prompt WHEN it is actually encoded (guidance > 1.0).

## Non-goals

- Raising the 512 budget (checkpoint contract).
- Auto-summarizing or rewriting prompts.

## Dependencies and related tasks

- Host application companion track `proposed/storyboard_consistency_2026_07/`
  (anchor prepending is what consumes budget).

## Validation

- Focused test: >512-token prompt produces the warning + metadata fields;
  <=512 stays silent.

## Progress checklist

- [x] Overflow detection at encode time
- [x] CLI warning + metadata fields
- [x] Negative-prompt case under CFG
- [x] Focused tests

## Implementation record (2026-07-25, pending release)

- `encode_prompt` (shared by Wan2_2_TI2V and WanVace) now runs
  `_check_prompt_truncation` before every encode: an uncapped tokenizer probe
  on the same `_prompt_clean`ed text the capped encode sees. One stderr line
  per truncated prompt, naming the counts
  ("Wan prompt truncated: 547 -> 512 UMT5 tokens; the last 35 tokens do not
  condition the video."). The probe runs even on prompt-embed cache hits — the
  truth about what conditioned the video does not depend on cache state.
- Metadata sidecar gains `prompt_tokens` / `prompt_truncated` on every Wan
  run, plus `negative_prompt_tokens` / `negative_prompt_truncated` ONLY when
  CFG actually encodes the negative (guidance > 1.0) — recording a negative
  count under the CFG-off Lightning recipe would misstate what ran.
- Runtime-event field: deliberately NOT added. The JSONL event schema is
  phase-based progress shared across families; there is no per-run
  prompt-facts event to ride, and adding a Wan-only field to every progress
  event would violate the "lightweight shared events" contract. The metadata
  sidecar is the host-facing record (the host application reads it).
- Tests: `tests/wan/test_wan_prompt_truncation.py` (8 tests, fake tokenizer —
  the logic under test is the overflow accounting, not the HF tokenizer):
  over/under budget, exact-fit boundary, negative-prompt gating under CFG,
  cleaned-text parity, and the metadata merge in generate_video.

## Cycle-2 verification (2026-07-25, no defects found)

- Verified the uncapped probe counts the SAME `_prompt_clean`ed text the
  capped encode tokenizes, runs before (and independent of) both the
  in-memory and disk prompt-embed caches, and gates the negative-prompt pair
  on the actual CFG encode condition (`guidance > 1.0`).
- Live confirmation: both 0097 probe sidecars carry
  `prompt_tokens`/`prompt_truncated` (14/false); WanVace merges the same
  report into its extra_metadata. No changes needed.
