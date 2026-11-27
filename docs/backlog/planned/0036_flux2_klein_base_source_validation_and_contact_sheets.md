# Planned: FLUX.2 Klein base source validation and contact sheets

## Metadata
- Created: 2026-06-10
- Status: Planned

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None

## Context

Strict FLUX.2 outpaint now routes only through base Klein models. The runtime contract and CLI
already enforce that split: distilled Klein keeps reframe, while base Klein exposes strict
outpaint. That fixed the earlier misconception where FLUX outpaint was treated like a reframing or
canvas-regeneration workflow.

The remaining gap is release evidence. The old `reframe_outpaint_2026_06_08` profile keeps useful
historical rows for distilled FLUX outpaint, but those rows are now stale. We needed a fresh
starship-based proof set for base source models and we still need separate proof for prepared base
q8/q4 packages.

## Current code reality

- `src/mflux/task_inference.py` advertises:
  - distilled FLUX.2 Klein `4B/9B`: `reframe` only;
  - base FLUX.2 Klein `4B/9B`: strict `outpaint` only;
  - both source and prepared base handles also expose text, latent img2img, edit-reference, and
    multi-reference.
- `src/mflux/models/flux2/cli/flux2_edit_generate.py` rejects unsupported base/distilled canvas
  combinations before generation starts.
- Local source-model evidence now exists for:
  - `black-forest-labs/FLUX.2-klein-base-9B`
  - `black-forest-labs/FLUX.2-klein-base-4B`
- Local prepared base package handles resolve and expose the same capability surface, but this item
  does not yet treat that as release-quality visual proof.

## Problem

Users need current, source-backed proof that base FLUX.2 Klein models can generate and edit images,
including strict outpaint on the detailed cropped-starship case. Without that proof, docs and
validation either understate the current route contract or overstate prepared-package coverage.

## What we want to do

Publish source-model starship contact sheets, a seam-review sheet for strict outpaint, exact
command logs, and a machine-readable validation profile for FLUX.2 Klein base `4B/9B`. Keep
prepared base q8/q4 validation as explicit follow-up work until the same starship profile is run on
those packages.

## Why

- Preserve a clean distinction between runtime support and release-quality evidence.
- Give users a concrete answer for whether base `4B/9B` can now generate and edit images.
- Avoid folding current base-only outpaint evidence into the older mixed reframe/outpaint profile.

## Requirements

- Use the cropped starship image as the canonical source.
- Run real source-model commands for:
  - text-to-image smoke;
  - latent img2img;
  - single-image edit-reference;
  - multi-reference;
  - strict outpaint.
- Review strict outpaint visually around the added border area, not only in the full image.
- Publish command logs and contact sheets under `docs/assets/validation/`.
- Keep prepared base q8/q4 claims separate until they have equivalent visual proof.

## Suggested implementation

1. Keep the old `reframe_outpaint_2026_06_08` profile as historical/current mixed evidence.
2. Add a new base-only starship profile for source-model base `4B/9B`.
3. Build:
   - one edit-capability matrix for base `4B/9B`;
   - one strict-outpaint seam-review sheet;
   - one text-to-image smoke panel.
4. Update the docs so they describe base FLUX outpaint as source-locked denoising, not adaptive
   source blending.
5. Leave prepared base q8/q4 validation as the residual work tracked by this item.

## Scope

- Base source handles only in the published proof set:
  - `black-forest-labs/FLUX.2-klein-base-4B`
  - `black-forest-labs/FLUX.2-klein-base-9B`
- Validation registry additions for the new source-only profile.
- Core docs and LLM indexes that mention FLUX reframe/outpaint behavior.

## Non-goals

- Do not claim prepared base q8/q4 packages passed the starship profile unless they were actually
  run and reviewed.
- Do not restore distilled FLUX outpaint as a current supported route.
- Do not mutate the June 8, 2026 profile into a base-model profile.

## Dependencies and related tasks

- [0019 first-class I2I modes and outpaint/reframe UX](0019_first_class_i2i_modes_and_outpaint_reframe.md)
- [0028 release validation registry](../completed/0028_release_validation_registry.md)
- [0026 edit model prepared-package capability contact sheets](../completed/0026_edit_model_prepared_capability_contact_sheets.md)

## Expected outcomes

- Base source `4B/9B` have published starship contact sheets and a source-only validation profile.
- Docs clearly state:
  - distilled FLUX reframe is current;
  - distilled FLUX outpaint rows are stale history;
  - base FLUX strict outpaint is current for source models;
  - prepared base q8/q4 proof is still pending.
- Residual package-level validation work remains visible and auditable.

## Validation

- `mlxgen capabilities --model black-forest-labs/FLUX.2-klein-base-9B`
- `mlxgen capabilities --model black-forest-labs/FLUX.2-klein-base-4B`
- `mlxgen capabilities --model AbstractFramework/flux.2-klein-base-9b-8bit`
- `mlxgen capabilities --model AbstractFramework/flux.2-klein-base-9b-4bit`
- `mlxgen capabilities --model AbstractFramework/flux.2-klein-base-4b-8bit`
- `mlxgen capabilities --model AbstractFramework/flux.2-klein-base-4b-4bit`
- Real source-model starship generations plus manual visual review.
- Focused router/validation tests.

## Progress checklist

- [x] Confirm source and prepared base route surfaces for `4B/9B`.
- [x] Run source-model starship text/edit/outpaint proof for base `9B`.
- [x] Run source-model starship text/edit/outpaint proof for base `4B`.
- [x] Build source-only base `4B/9B` contact sheets, seam review, command log, and manifest.
- [x] Add a source-only FLUX.2 Klein base validation profile.
- [ ] Run the same starship profile on prepared base q8/q4 packages before treating those package
      rows as passed.

## Implementation notes

Do not collapse source-model proof and prepared-package proof into one claim. Keep the source base
story honest now, and keep package-level release evidence pending until it exists.
