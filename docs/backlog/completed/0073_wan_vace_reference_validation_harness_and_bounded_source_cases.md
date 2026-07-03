# Completed: Wan VACE Reference Validation Harness And Bounded Source Cases

## Metadata
- Created: 2026-07-03
- Status: Completed
- Completed: 2026-07-03

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md), [ADR 0006](../../adr/0006_generative_video_editing_task_boundary.md)
- ADR impact: None.

## Context

Current MLX-Gen does not ship Wan VACE or prompt-guided source-video editing. The first public
runtime milestone is still plain `video-to-video`, not richer VACE conditioning. At the same time,
the repo already has:

- local reference access to upstream Diffusers Wan video-to-video and Wan VACE pipelines;
- a 128 GB Apple Silicon host suitable for bounded reference checks;
- existing project measurement helpers for sampled RSS, Darwin physical footprint, and saved-output
  artifact validation.

What is still missing is a repeatable, local, one-test-at-a-time proof path for richer conditioned
editing that answers basic questions with real artifacts:

- which exact Wan VACE reference checkpoint is the first bounded target;
- what wall time and peak memory look like on this host;
- whether simple localized or structural edit cases are visually plausible enough to justify the
  later VACE-conditioning work after plain `video-to-video`.

## Current code reality

- `tools/generation_memory_benchmark.py` and `tools/multi_output_batch_benchmark.py` already expose
  reusable process-tree sampling helpers and report structure for one-at-a-time runs.
- `src/mflux/utils/video_util.py` already has video inspection, clip decoding, and save helpers.
- There is no checked-in tool for Wan VACE or Wan V2V reference runs, and no exact official
  checkpoint has been frozen as the first bounded validation target.
- Existing repo assets already include a ship takeoff video and image assets that can serve as a
  bounded source case. A portrait still image exists in `tests/resources/unsplash_person.jpg`.

## Problem

The VACE discussion leaned too hard on upstream existence without local proof. We needed one
repeatable way to answer two questions with actual artifacts instead of speculation:

1. does the official Wan VACE reference path run on this host at all; and
2. are the resulting edits good enough to justify promoting richer VACE conditioning toward a
   public MLX-Gen runtime.

## What changed

- Added `tools/wan_vace_reference_probe.py`, a one-at-a-time upstream reference harness that:
  - prepares bounded source clips and mask assets;
  - launches one isolated child process per case;
  - samples whole-process RSS and Darwin physical footprint;
  - preserves output MP4s, contact sheets, logs, and reports under `validation_outputs/`.
- Added focused regression coverage in `tests/test_wan_vace_reference_probe.py`.
- Downloaded the official `Wan-AI/Wan2.1-VACE-1.3B-diffusers` checkpoint once outside the measured
  runs so the real timing and memory measurements stay offline and comparable.
- Ran two bounded edit cases sequentially:
  - a portrait clip animated from the local portrait still with a localized hair-color edit mask;
  - a short source excerpt from the local ship lift-off video with a localized ship rewrite mask.
- Ran one additional higher-precision ship retry to test whether the washed-out bounded result was
  mainly a float16 issue on `mps`.

## Dependencies and related tasks

- [0072 Reader-first video workflow boundary and generative video-edit contract](0072_reader_first_video_workflow_boundary_and_generative_video_edit_contract.md)
- [0075 Wan VACE conditioning expansion after plain video-to-video](../proposed/0075_wan_vace_conditioning_expansion_after_plain_video_to_video.md)

## Validation

- Focused tool tests:
  - `uv run pytest tests/test_wan_vace_reference_probe.py -q`
- Local tool lint:
  - `uv run ruff check tools/wan_vace_reference_probe.py tests/test_wan_vace_reference_probe.py`
- Manual bounded runs:
  - `uv run --with accelerate --with ftfy python tools/wan_vace_reference_probe.py --cases portrait_hair_eyes --skip-prefetch`
  - `uv run --with accelerate --with ftfy python tools/wan_vace_reference_probe.py --cases ship_reactor_nacelles --skip-prefetch`
  - `MFLUX_WAN_VACE_TRANSFORMER_DTYPE=float32 uv run --with accelerate --with ftfy python tools/wan_vace_reference_probe.py --cases ship_reactor_nacelles --skip-prefetch`
  - `uv run python tools/wan_vace_reference_probe.py --device cpu --skip-prefetch --cases ship_reactor_nacelles --output-dir validation_outputs/wan_vace_cpu_probe_2026_07_03`
  - `uv run python tools/wan_vace_reference_probe.py --device cpu --skip-prefetch --cases portrait_hair_eyes --portrait-source-image validation_outputs/v2v_real_2026_07_03/portrait_source.png --output-dir validation_outputs/wan_vace_cpu_probe_2026_07_03_refined_portrait4`

## Measured outcome

The official `Wan2.1-VACE-1.3B` reference path is operational on this host, but the quality result
depends heavily on device and case shape.

- Bounded `mps` runs stayed invalid:
  - Portrait output path: `validation_outputs/wan_vace_reference_2026_07_03/portrait_hair_eyes/output.mp4`
  - Ship output path: `validation_outputs/wan_vace_reference_2026_07_03/ship_reactor_nacelles/output.mp4`
  - Result: the float16 `mps` path completed, but the portrait remained blown out and the ship edit
    remained washed out. That path is not trustworthy enough for user-facing promotion.
- CPU ship localized structural-edit attempt:
  - Output path: `validation_outputs/wan_vace_cpu_probe_2026_07_03/ship_reactor_nacelles/output.mp4`
  - Outer wall time: `642.1741` seconds
  - Peak sampled RSS: `57.1216 GB`
  - Peak sampled physical footprint: `49.1576 GB`
  - Result: the ship rewrite is visibly credible on the bounded source clip. The model preserved
    the snowy lift-off context while materially changing the ship structure.
- CPU portrait localized hair-edit attempt:
  - Output path: `validation_outputs/wan_vace_cpu_probe_2026_07_03_refined_portrait4/portrait_hair_eyes/output.mp4`
  - Outer wall time: `1124.1919` seconds
  - Peak sampled RSS: `59.2842 GB`
  - Peak sampled physical footprint: `51.2619 GB`
  - Result: after fixing the portrait framing and mask placement, the adult male identity and face
    stay stable, but the hair-color change remains too subtle to claim a strong localized portrait
    rewrite.

## Completion report

- Date: 2026-07-03
- Summary: MLX-Gen now has a repeatable upstream Wan VACE reference probe with preserved artifacts
  and real memory statistics, and the later CPU reruns sharpened what the tool can and cannot prove
  today. The reference path is operational; the bounded ship edit is credible on CPU; the bounded
  portrait edit is no longer destructive but still too weak to count as a strong localized
  appearance rewrite; and the original `mps` path remains visually invalid.
- Implementation:
  - `tools/wan_vace_reference_probe.py`
  - `tests/test_wan_vace_reference_probe.py`
- Evidence bundle:
  - `validation_outputs/wan_vace_reference_2026_07_03/wan_vace_reference_summary.md`
  - `validation_outputs/wan_vace_reference_2026_07_03/portrait_hair_eyes/`
  - `validation_outputs/wan_vace_reference_2026_07_03/ship_reactor_nacelles/`
  - `validation_outputs/wan_vace_cpu_probe_2026_07_03/ship_reactor_nacelles/`
  - `validation_outputs/wan_vace_cpu_probe_2026_07_03_refined_portrait4/portrait_hair_eyes/`
- Residual risk: this result is still bounded to one official `1.3B` VACE checkpoint and two small
  edit cases on one host. It proves that the reference path can produce one credible object-level
  edit on CPU, but it does not yet justify a broader localized-portrait quality claim or a public
  MLX-native runtime claim.
