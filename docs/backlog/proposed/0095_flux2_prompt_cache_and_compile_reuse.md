# Proposed: Wire flux2 prompt_cache and reuse compiled predict per instance

## Metadata

- Created: 2026-07-23
- Status: Proposed (from the 2026-07-23 BlackPixel i2i latency audit)
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
