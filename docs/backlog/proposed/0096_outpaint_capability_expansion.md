# Proposed: outpaint capability expansion beyond Qwen edit and Klein Base

## Metadata

- Created: 2026-07-23
- Status: Proposed (owner-requested investigation, 2026-07-23)
- Completed: N/A
- Effort: M (investigation) + M-L per family (implementation)

## ADR status

- Governing ADRs: [ADR 0001](../adr/0001_runtime_smoke_validation_for_model_routes.md)
  (any new outpaint route needs real-checkpoint smoke evidence before its
  capability row ships), [ADR 0002](../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None yet (investigation first).

## Context (owner request, verbatim intent)

The embedding host (BlackPixel) now surfaces only outpaint-capable models in
its Outpaint selector, driven by this repo's capability registry. Today that
set is exactly two families:

- `qwen.outpaint` — Qwen edit identities (expanded canvas generation plus
  adaptive source restoration);
- `flux2.outpaint` — FLUX.2 Klein BASE models only (source-locked denoising
  with a narrow latent transition band); distilled Klein advertises
  `flux2.reframe` instead.

The owner asks: "should other models be capable to do outpaint, including if
we have to implement some low level math? or are those the only 2 models that
can (would find that surprising)?"

## Working hypothesis

The two-family limit is an EVIDENCE policy, not a technical ceiling.
Outpainting is structurally inpainting at an expanded canvas border: any
family with (a) masked/source-locked denoising machinery or (b) latent i2i
plus per-step source compositing can, in principle, outpaint. In this repo:

- **Z-Image Turbo** already ships native inpaint (`z-image.inpaint`) — the
  same masked-denoise loop at an expanded canvas with a border mask is the
  natural extension.
- **Distilled FLUX.2 Klein** ships `flux2.inpaint` (per-step source
  compositing port of the Klein inpaint pipeline); its old outpaint rows
  were withdrawn as stale history, not disproven physics — the base-model
  source-locked approach may transfer.
- **Base Qwen Image** ships `qwen.base-inpaint` (tunable mask-strength warm
  start) — border-mask outpainting is the same mechanism pointed outward.
- **Latent i2i families without masks (ERNIE, FIBO, FLUX.1)** would need the
  low-level piece: source-locked denoising (lock the source region's latents
  at every step, denoise only the border region from noise, with a
  transition band) — exactly what `flux2.outpaint` (base) already
  implements; the math generalizes, the per-family work is wiring +
  validation.

## What we want to do

1. Investigation pass: for each family above, identify the concrete
   mechanism (existing mask route vs new source-locked loop), the expected
   quality risk (edge coherence, color drift at the seam, distilled-model
   few-step behavior at borders), and the cost of a runnable prototype.
2. Prototype the cheapest promising route first (likely Z-Image Turbo
   border-mask inpaint → outpaint, since the masked loop exists end to end).
3. Real-checkpoint validation per ADR 0001 (seam quality proof rows like the
   existing starship outpaint matrices) before any capability row flips
   `supports_outpaint=True`.
4. Ship rows additively; the host's selector consumes them automatically at
   the next pin bump (see the BlackPixel companion item).

## Non-goals

- Advertising outpaint on any route without accepted visual proof (the
  registry stays evidence-based — that discipline is why the host can trust
  it).
- Wan/video outpainting (different problem; out of scope here).

## Dependencies and related tasks

- BlackPixel backlog 0070 (selector adoption when this expands).
- Existing proof tooling: docs/reframe-outpaint.md matrices, validation
  profile machinery.

## Expected outcomes

Either more families truthfully advertising outpaint, or a documented
per-family verdict of why not (quality evidence, not assumption).

## Validation

Per-family accepted proof bundles (source/q8 rows, seam close-ups), matching
the existing outpaint validation style.

## Progress checklist

- [ ] Per-family mechanism + risk survey (investigation pass)
- [ ] First prototype route (candidate: Z-Image Turbo border-mask)
- [ ] Real-checkpoint seam proofs
- [ ] Capability rows + docs + host notification
