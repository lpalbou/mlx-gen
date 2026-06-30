# Completed: Image finalization memory peak and metadata rewrite

## Metadata
- Created: 2026-06-29
- Status: Completed
- Completed: 2026-06-30

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The current memory track already covers startup, prompt materialization, and retention behavior in
items [0060](../completed/0060_runtime_memory_telemetry_and_manifests.md),
[0061](../completed/0061_prompt_materialization_for_low_ram_release.md), and
[0064](../planned/memory/0064_generation_retention_cleanup.md). A separate user-observed issue
remains at the very end of image generation: process memory spikes while the generated image is
being finalized and written to disk.

This is not the denoising loop itself. The spike appears after generation has logically finished
and before control returns to the caller.

## Historical pre-fix reality
- `GeneratedImage.save()` always built the full metadata payload even for the default save path.
- The default PNG save path performed a primary write, then reopened and resaved for EXIF, then
  reopened and resaved again for PNG XMP/IPTC metadata.
- Runtime-memory sampling happened inside metadata construction, so the tail save path mixed
  metadata serialization, helper subprocess work, and full-image rewrites.

## Problem
Image finalization used to combine two independent memory-pressure sources:

1. repeated reopen-and-resave behavior for metadata embedding; and
2. runtime-memory snapshot collection during metadata construction.

That creates a plausible end-of-run RSS / physical-footprint spike even when the denoising path is
already done.

## What we changed
- `src/mflux/utils/generated_image.py` now treats metadata as opt-in work. The default image save
  path does not build metadata, and runtime-memory sampling only happens for JSON sidecar export.
- `src/mflux/utils/image_util.py` now builds EXIF and PNG metadata before one final `image.save(...)`
  call instead of reopening and rewriting the saved PNG.
- `--embed-metadata` is now an explicit opt-in CLI contract; lightweight default save remains the
  default success path.
- The focused metadata tests now assert one save call, zero reopen calls, zero runtime-memory
  snapshots by default, and preserved EXIF/XMP/IPTC on the opt-in embedded path.

## Requirements
- Default image save must be lightweight and explicit.
- Embedded metadata must remain available without reducing image quality.
- Runtime-memory metadata must not be part of the default save path.
- Measured proof must separate save-phase memory from denoise/runtime memory.

## Validation
- Focused unit tests:
  - `tests/metadata/test_generated_image.py`
  - `tests/metadata/test_metadata.py::TestMetadata::test_metadata_complete`
- Dedicated save-phase probe:
  - `tools/image_finalization_memory_probe.py`
  - [image_finalization_memory_report.md](../../assets/validation/image-finalization-2026-06-30/image_finalization_memory_report.md)
  - [image_finalization_memory_stats.json](../../assets/validation/image-finalization-2026-06-30/image_finalization_memory_stats.json)

## Measured outcome
The June 30, 2026 probe used a deterministic `4096x4096` RGB PNG and sampled only the child
process responsible for save/finalization.

- Current default: `1` save call, `0` reopen calls, `0` runtime-memory snapshots, no embedded
  metadata.
- Current embedded metadata: `1` save call, `0` reopen calls, `0` runtime-memory snapshots, EXIF
  plus PNG XMP/IPTC preserved.
- Current sidecar metadata: `1` save call, `0` reopen calls, `1` runtime-memory snapshot, sidecar
  metadata only.
- Simulated legacy default: `3` save calls, `2` reopen calls, `1` runtime-memory snapshot.

Measured peak reductions versus the simulated legacy default:

- Current default peak RSS: `0.394 GB -> 0.192 GB` (`-51.2316%`)
- Current default peak physical footprint: `0.373 GB -> 0.171 GB` (`-54.1496%`)
- Current embedded-metadata peak RSS: `0.394 GB -> 0.192 GB` (`-51.2066%`)
- Current embedded-metadata peak physical footprint: `0.373 GB -> 0.171 GB` (`-54.1364%`)

## Non-goals
- Do not broaden this into generic model-runtime memory tuning.
- Do not remove useful metadata silently without deciding the public metadata contract.
- Do not conflate this save-path issue with prompt-materialization or hidden-state-retention work
  unless measurements prove they overlap materially.

## Completion report

- Date: 2026-06-30
- Original path: `docs/backlog/proposed/0066_image_finalization_memory_peak_and_metadata_rewrite.md`
- Final path: `docs/backlog/completed/0066_image_finalization_memory_peak_and_metadata_rewrite.md`
- Summary: Image finalization is now a one-pass save contract by default, embedded metadata is
  explicit and still preserved, runtime-memory sampling moved off the default path, and a dedicated
  save-phase proof bundle shows the legacy tail spike was real and materially reduced.
- Implementation:
  - `src/mflux/utils/generated_image.py`
  - `src/mflux/utils/image_util.py`
  - `src/mflux/cli/parser/parsers.py`
  - `tests/metadata/test_generated_image.py`
  - `tests/metadata/test_metadata.py`
  - `tools/image_finalization_memory_probe.py`
- Validation:
  - `uv run pytest tests/metadata/test_generated_image.py -q`
  - `uv run pytest tests/metadata/test_metadata.py -q -k test_metadata_complete`
  - `uv run python tools/image_finalization_memory_probe.py`
- Evidence report: [image_finalization_memory_report.md](../../assets/validation/image-finalization-2026-06-30/image_finalization_memory_report.md)
- Residual risk: `MetadataReader.read_xmp_metadata(...)` is still a lightweight parser. The raw
  PNG XMP/IPTC contract is preserved and tested, but richer XMP parsing should be treated as a
  separate follow-up from save-path memory.
