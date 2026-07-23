# Proposed: Wire flux2 prompt_cache and reuse compiled predict per instance

## Metadata

- Created: 2026-07-23
- Status: Implemented (pending release) — 2026-07-23, cycle-1 implementation wave
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: None

## Context

`Flux2Initializer` writes `model.prompt_cache = {}` (and the training adapter
resets it), but NOTHING on the flux2 path ever reads it: `Flux2PromptEncoder.
encode_prompt` takes no cache argument and every `generate_image` call
re-encodes the prompt through the 8B Qwen3 text encoder. Compare flux (v1),
whose `PromptEncoder` checks `prompt_cache` before encoding, and qwen, which
passes its cache through. The flux2 attribute is dead state — verified by
grep across `src/mflux/models/flux2/` on 2026-07-23 (writes only, no reads).

Marginal cost is ~0.8-1.1 s per encode (q8, warm). It compounds in the
embedding-host pattern: BlackPixel sends num_images=4 per click, and the
runtime wrapper loops seeds through generate_image — the SAME prompt is
re-encoded 4x per click (~3-4 s wasted), on every click.

Same shape of waste, second instance: per-call `mx.compile` of the step
function (measured/estimated 1-6 s per call on this path) is not reused
across calls on one resident instance where inputs shapes are unchanged.

## Proposal

1. Pass `self.prompt_cache` into `Flux2PromptEncoder.encode_prompt` from all
   four flux2 variants (txt2img, edit, inpaint, outpaint), keyed exactly as
   flux v1 does (prompt string -> embeds tuple); negative prompt included.
   Bound the dict (e.g. small LRU) since edit hosts can stream many prompts
   through one resident worker.
2. Cache the compiled predict callable per model instance (keyed by the
   shape/dtype signature that forces re-trace) so repeat same-shape calls on
   a resident model skip re-compilation. Follow the 0090 experiment
   discipline: measure before shipping; skip if measurement shows current
   MLX already reuses traces effectively at this call pattern.

## Expected impact

~3-5 s per 4-seed BlackPixel click (measured encode cost x repeated seeds,
plus compile reuse); a few percent of a small-canvas batch, negligible at
944x1680 — this is hygiene, not the headline fix (those are 0093/0094 and
the BlackPixel-side items in that repo's backlog 0069).

## Non-goals / nuance recorded from the audit

flux2.latent LoRA for Klein 9B stays deny-listed in
`lora_validation_registry._UNSUPPORTED_RECORDS` (hard-rejected at plan time,
verified). If a validated klein-9b latent LoRA row ever lands, BlackPixel
could route LoRA global-restyles through latent i2i (shares the t2i handler
-> no worker reload on task flips, and image_strength skips steps); until
then that routing idea is blocked and lives only as context here.

## Promotion criteria

Promote (1) opportunistically with any flux2 touch — it is small, isolated,
and testable (same-prompt second call must skip the encoder; cache bounded).
Promote (2) only with a benchmark-harness measurement per 0090's rule.

## Implementation record (2026-07-23)

### Design decision

1. Prompt cache: `Flux2PromptEncoder.encode_prompt` gained an optional
   `prompt_cache` dict argument, and all four variants pass
   `self.prompt_cache` (txt2img and edit directly; inpaint/outpaint through
   the edit `_encode_prompt_pair` they already delegate to; negative prompts
   ride the same key namespace). Deviation from flux v1's raw-string key,
   on purpose: the key is `repr((prompt, num_images_per_prompt,
   max_sequence_length, text_encoder_out_layers))` so one dict safely
   serves any call pattern (and list-prompts). Bounded as an 8-entry LRU
   managed in the encoder (plain insertion-ordered dict, hit re-inserts,
   overflow pops oldest): Klein 9B embeds are ~12-13 MB/entry, so the bound
   is ~100 MB worst case for streaming edit hosts. Entries are materialized
   (`RuntimeMemory.materialize_tensors`) before caching, as flux/qwen do.
   Side benefit: flux2 `--low-ram` multi-seed runs now survive the
   MemorySaver text-encoder release for repeated prompts, same as flux v1.
2. Compile reuse: `CompiledPredictCache` (new, `src/mflux/utils/`) is owned
   by every flux2 model instance. Key = `(variant, has_negative)` — the
   python-level branch structure; argument shape/dtype changes retrace
   inside `mx.compile` itself and need no key. `weights_token` = the
   transformer instance: replacement drops all entries. mx.compile bakes
   closed-over weight arrays as constants, so IN-PLACE mutation cannot be
   detected by identity — every mutation seam clears explicitly (the wan
   0090 d12 drop-before-release discipline):
   - `MemorySaver._delete_transformer` (low-RAM release; a cached callable
     would otherwise keep the freed weights alive),
   - both flux2 training adapters before each preview (training updates
     weights between previews),
   - `Flux2BaseTrainingAdapter.load_lora_adapter` / `load_training_adapter`,
   - `Flux2Klein.save_model` (ModelSaver bakes and strips LoRA in place).
   The M1/M2 eager path caches the raw closure — harmless.
   z_image has the same per-call `mx.compile` shape (`z_image.py:273`) and
   was deliberately left out of scope; noted for a follow-up.

### Measured numbers

- Re-encode cost being skipped: ~0.8-1.1 s per identical-prompt encode on a
  resident q8 Klein 9B TE (tonight's audit measurement; not re-measured —
  the unit tests prove the skip, and the encoder path is unchanged).
- Compile reuse: per-call trace cost was estimated 1-6 s in the audit; on
  tonight's 512x512 q8 bench the second same-shape generation ran 3.5 s vs
  4.7-4.9 s for the first (some of that delta is residual fault-in), with
  the reused callable confirmed by identity in tests. Per 0090's rule the
  full benchmark-harness measurement remains open for the release gate.

### Files changed

- `src/mflux/utils/compiled_predict_cache.py` (new)
- `src/mflux/models/flux2/model/flux2_text_encoder/prompt_encoder.py`
- `src/mflux/models/flux2/flux2_initializer.py` (owns both caches)
- `src/mflux/models/flux2/variants/txt2img/flux2_klein.py`
- `src/mflux/models/flux2/variants/edit/flux2_klein_edit.py`,
  `flux2_klein_edit_helpers.py`, `flux2_klein_inpaint.py`,
  `flux2_klein_outpaint.py`
- `src/mflux/callbacks/instances/memory_saver.py`
- `src/mflux/models/flux2/training_adapter/` (base + txt2img + edit)
- `tests/image_generation/test_flux2_generation_caches.py` (new: wiring
  hit/miss on real generate loops, encoder-level LRU bound + key coverage,
  compiled-predict reuse + invalidation-on-replacement, abort-safety
  contract), `tests/utils/test_compiled_predict_cache.py` (new),
  fake-contract updates in `tests/training/test_flux2_training_adapter.py`
  and `tests/image_generation/test_masked_generation_routes.py`.

### Abort-safety note (host cooperative-cancel contract)

Verified by code reading and pinned by test
(`TestFlux2AbortSafety::test_exception_from_progress_callback_propagates_and_model_stays_reusable`):
an exception raised from a progress callback inside the denoise loop
propagates out of `generate_image` (only `KeyboardInterrupt` is translated
to `StopImageGenerationException`), and the instance stays reusable — the
retry produced bitwise-identical decoded latents to an uninterrupted run,
with the prompt cache and compiled predict intact (embeds are materialized
before the loop; per-call state lives in `Config`/`GenerationContext`).

### Cycle-2 review additions (2026-07-23, adversarial review)

Invalidation-completeness audit: every in-place transformer mutation site in
the repo was enumerated (`load_and_apply_lora*`, `apply_and_quantize*`,
`nn.quantize`, `bake_and_strip_lora`, runtime dequant, concept-attention).
Findings: initializer-time sites precede any cached predict (safe);
concept-attention is flux v1 only; runtime dequant is wan-only (no
`CompiledPredictCache`); `bake_and_strip_lora` is reachable for flux2 only
via `Flux2Klein.save_model` (covered; edit/inpaint/outpaint have no
`save_model`). One convention-only gap was hardened:
`TrainingUtil.assistant_disabled` restores assistant LoRA scales in place on
CONTEXT EXIT, leaving a preview's cached predict with `scale=0` baked while
the live model returned to the real scale — previously safe only because
the next supported call site cleared first. The flux2 adapter's
`_assistant_disabled` now clears the compiled cache on exit (test:
`test_assistant_disabled_clears_compiled_cache_on_exit`).

Prompt-cache key audit: negative prompts get their own entries (separate
`encode_prompt` call, same namespace — covered); LoRA swaps do NOT belong in
the key because flux2 LoRA mappings target only the transformer
(`Flux2LoRAMapping` + `load_and_apply_lora_detailed(transformer=...)`), so
embeds are LoRA-independent — and the cache is per-instance anyway. The
4-seed same-prompt click pattern holds one entry and hits it three times
(`test_identical_prompt_second_generation_skips_reencode` covers the
pattern).

Low-RAM interaction claim verified against the real `MemorySaver`: encode
of seed 1 fills the cache before `call_before_loop` sets
`model.text_encoder = None`; seed 2 with the same prompt hits the cache and
never touches the released encoder; a NEW prompt after release fails loud
(`AttributeError`), matching the documented same-prompt-only contract
(tests: `TestFlux2LowRamPromptCacheInteraction`).

Abort-safety mutation check: with `CallbackRegistry.emit_progress` wrapped
in a swallowing try/except, the abort-safety test fails as intended
(exception no longer propagates); patch reverted. The test is a real tripwire.

### Gates

Cycle-1: `make lint` clean; fast suite 1457 passed / 7 skipped (baseline
1413/7). Cycle-2 (after the assistant-scale hardening and low-RAM
interaction pinning): `make lint` clean; fast suite 1465 passed /
7 skipped; BlackPixel geometry passthrough suite 12 passed against this
tree via `uv run --project`.
