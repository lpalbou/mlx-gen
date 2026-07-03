# Bug: image finalization still causes an avoidable end-of-run memory spike

## Status on 2026-06-30

Resolved in the current tree. The default image save path is now one-pass and metadata-light, the
embedded metadata path remains opt-in and preserves EXIF/XMP/IPTC, and the dedicated save-phase
probe under `docs/assets/validation/image-finalization-2026-06-30/` measured peak sampled RSS
`-51.2316%` and peak Darwin physical footprint `-54.1496%` versus the legacy three-pass
simulation. See `docs/backlog/completed/0066_image_finalization_memory_peak_and_metadata_rewrite.md`.

## Summary

The current unpublished `mlx-gen` tree improves routing truth, progress semantics, and declared
feature coverage, but it still keeps the old image finalization path that can spike memory after
generation is already done.

This matters to embedding apps because users see the run as "finished" and then observe a large
memory jump during save/finalization. That blurs the boundary between model-runtime pressure and
save-pipeline overhead.

## Historical code reality

- `src/mflux/utils/generated_image.py` adds
  `RuntimeMemory.snapshot("image-metadata").to_metadata()` while building default image metadata.
- `src/mflux/utils/image_util.py` saves the image first, then may reopen and resave it for EXIF via
  `_embed_metadata(...)`, then reopen and resave again for PNG XMP/IPTC via
  `MetadataBuilder.embed_metadata(...)`.
- `docs/backlog/completed/0066_image_finalization_memory_peak_and_metadata_rewrite.md` now records
  the implemented fix and proof bundle.

## Why this mattered

- This path can perform multiple full-image write/rewrite passes after denoising is complete.
- Runtime-memory metadata collection happens in the same tail phase.
- The result is an end-of-run memory peak that is not model compute, but still shows up as
  `mlx-gen` memory growth.

## What I need from `mlx-gen`

1. Make the default final image save path memory-conscious and explicit.
2. Stop treating repeated reopen-and-resave metadata embedding as the default success path.
3. Separate "artifact saved successfully" from optional heavy metadata enrichment.

## Requested direction

- Prefer a single final write pass where possible.
- If full embedded metadata cannot be kept cheap, make it opt-in instead of unconditional in the
  default generated-image path.
- Treat runtime-memory metadata as a sidecar or diagnostics-mode concern unless there is strong
  proof that it belongs in the default embedded metadata contract.

## Acceptance criteria

- A representative large PNG route no longer performs the current three-step write/rewrite path in
  the default success case.
- The saved artifact is considered complete before optional metadata enrichment that can fail or
  consume extra memory.
- Validation includes phase-separated memory sampling around `save`, metadata embedding, and any
  runtime-memory snapshot step.
- Public docs state whether embedded runtime-memory metadata is default, opt-in, or sidecar-only.

## Evidence

- `src/mflux/utils/generated_image.py`
- `src/mflux/utils/image_util.py`
- `docs/backlog/completed/0066_image_finalization_memory_peak_and_metadata_rewrite.md`
