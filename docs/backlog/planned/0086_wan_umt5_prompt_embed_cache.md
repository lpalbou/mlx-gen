# Planned: Wan UMT5 prompt-embed residency and disk cache

## Metadata

- Created: 2026-07-22
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md) (cache
  misses must fail loud, never silently degrade prompts)
- ADR impact: None if the cache is exact (keyed on model + tokenizer output); needs a
  note only if approximate reuse is ever proposed.

## Context

A 2026-07-22 adversarial performance audit of BlackPixel (an embedding host) traced
Wan generation end-to-end. The single largest fixed cost after weight loading is
prompt encoding: `Wan2_2_TI2V._load_t5_prompt_embeds` constructs
`UMT5EncoderModel.from_pretrained(...)` in torch on CPU, runs one bf16 forward, and
deletes the encoder — per cache miss (`src/mflux/models/wan/variants/wan2_2_ti2v.py`
around lines 815-877 in the 0.23.1 release). The `text_encoder/` snapshot is ~11 GB;
loading it costs tens of seconds and a transient ~11 GB RSS spike that coexists with
the resident MLX transformers (peak-setter for A14B runs).

The in-process cache is keyed by prompt, so it only helps repeated identical prompts
within ONE process. Host apps that chain scenes (storyboard-style: new prompt per
scene, one process per scene today, warm-process tomorrow) pay the full reload every
scene.

## Current code reality

- torch import is already lazy (function-level) — good.
- The embed for a 512-token sequence at 4096 dim in bf16 is ~4 MB: trivially
  cacheable on disk.
- Qwen's text encoder is native MLX; Wan's is the only remaining torch-CPU encoder
  on a generation hot path.

## Problem

Per-new-prompt cost: tens of seconds + ~11 GB transient, repeated for every scene in
long-form video workflows, even when the process stays warm.

## What we want to do

Two layers, both exact:
1. Optional encoder residency: `keep_text_encoder_resident=True` (Python API) /
   `--keep-text-encoder` (CLI) keeps the torch module alive across generations in
   one process. Default stays load-and-release.
2. Disk-backed prompt-embed cache: key = (model id/revision, tokenizer config hash,
   prompt, max sequence length, dtype); value = safetensors file with the embeds.
   Bounded size, LRU pruning, explicit `--no-prompt-cache` opt-out.

Longer-term (separate item if pursued): port UMT5-XXL to MLX like Qwen's encoder,
removing torch from the Wan hot path entirely.

## Why

Directly serves the embedding-host storyboard workflow and every repeat-prompt CLI
user; removes the largest transient memory spike in A14B runs.

## Requirements

- Exact reuse only; cache hit produces bitwise-identical embeds for identical keys.
- Cache location under the existing platformdirs cache root; size-bounded.
- Loud failure on cache corruption (delete + re-encode, log the event).
- No behavior change when disabled.

## Suggested implementation

Factor `_load_t5_prompt_embeds` into a small PromptEmbedProvider owned by the
variant; residency and disk cache are provider strategies. Emit a progress event
phase ("encode-cached" vs "encode") so hosts can show honest progress.

## Scope

Wan TI2V/A14B/VACE prompt encoding paths.

## Non-goals

- Approximate/semantic prompt reuse.
- The MLX UMT5 port (separate item; this one keeps torch but stops repeating it).

## Dependencies and related tasks

- BlackPixel planned item 0051 (warm video worker) consumes this.
- Related: planned/memory/0063 component-wise loading policy.

## Expected outcomes

Scene N+1 with a new prompt pays ~0 encoder cost after the first run of a session
(residency) or across sessions (disk cache); A14B peak RSS drops by the transient
torch component when the cache hits.

## Validation

- Unit: cache key round-trip; corruption -> re-encode path; identical embeds
  (allclose exact) between cached and fresh.
- Timed: second run of the same prompt set skips encode (assert via progress phases
  and wall-time bound in a marked-slow test).

## Progress checklist

- [x] Store extraction (`WanPromptEmbedStore` in
      src/mflux/models/wan/prompt_embed_store.py — named Store, not Provider:
      it owns disk persistence + keys; the variant keeps the torch forward)
- [x] Residency flag (`keep_text_encoder_resident` constructor param +
      `--keep-text-encoder` CLI; reachable through the public wrapper via the
      new `load_generation_model(..., model_kwargs={...})` passthrough)
- [x] Disk cache + LRU bounds (64 entries) + `--no-prompt-cache` opt-out;
      tokenization moved to numpy so a disk hit never imports torch
- [x] Tests (tests/test_wan_prompt_embed_store.py round-trip/corruption/prune/
      collision/fingerprint; CLI constructor wiring in test_mlx_gen_router;
      wrapper passthrough in test_python_runtime)
- [ ] Progress event phases (encode-cached vs encode) — DEFERRED: public
      callback schema change; cli_print lines carry the honesty for now
- [x] docs/python-integration.md mention of model_kwargs + the two Wan options
      (added in cycle 1; cycle-2 review confirmed at python-integration.md
      ~171-183)

## Release caveat (cycle-2 review, 2026-07-22)
On exactly-64 GB machines, weight residency (28 GB A14B) plus UMT5 residency
(11 GB) plus activations approaches the Metal working-set budget. When an
embedding host enables residency by machine class, recommend disk-cache-only
(no residency) as the safer default at the 64 GB boundary; residency is
clearly safe at 96 GB+.

## Implementation note (2026-07-22)

Exactness guarantee: the disk key covers the text-encoder snapshot fingerprint
(file names+sizes+mtimes), the TOKENIZED input ids/attention mask (which
capture prompt text and tokenizer config exactly), max_sequence_length, and
precision. MLX-side transformer/LoRA state is deliberately excluded — embeds
depend only on the encoder and tokenizer. Corrupt entries are dropped loudly
and re-encoded (ADR 0002 compliant).
