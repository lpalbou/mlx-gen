# Planned: Post-save video health-check opt-out for embedded hosts

## Metadata

- Created: 2026-07-22
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None — the check stays default-ON; hosts opt out explicitly and the
  metadata records that they did.

## Context

`GeneratedVideo` runs a full-file re-decode health check in a spawned child process
after every save (`src/mflux/utils/video_health.py` ~152-216; `validate_health=True`
default in `generated_video.py` ~101-137). A 2026-07-22 audit of BlackPixel showed
the host then probes the file again for metadata/poster and (for storyboards)
extracts a last frame. Cycle-2 measurement corrected the cost taxonomy: the health
check is the one real full decode (~0.3-0.8 s at storyboard size, larger for
720p/81f clips); the host metadata/poster probe is a header scan + one frame
(~7-25 ms), and last-frame extraction seeks near the tail. The opt-out removes the
dominant term for embedded hosts that validate their own artifacts.

## Current code reality

The health check exists because encoder failures were real (truthfulness contract).
The generator also already computes in-memory frame statistics on the materialized
path — much of the signal exists before the re-decode.

## Problem

Embedded hosts that immediately probe the file themselves pay a redundant full
decode per clip; on multi-scene storyboard runs this multiplies.

## What we want to do

- Add `validate_health=False` reachable from the Python API and
  `--no-validate-health` on the CLI, default unchanged (ON).
- Record `health_check: "skipped"` in the save metadata/runtime event so downstream
  tooling can tell.
- Where cheap, prefer in-memory stats over re-decode for the checks that allow it,
  keeping the child-process re-decode for the container-level checks only.

## Why

Removes a fixed per-clip tax for hosts that do their own probe, without weakening
the default truthfulness contract.

## Scope

Video save path (Wan families, SeedVR2 video restore).

## Non-goals

- Changing the default.
- Weakening what the ON check verifies.

## Dependencies and related tasks

- BlackPixel planned item 0052 consumes this (single-decode completion path).

## Expected outcomes

Hosts opting out save one full decode per clip; `--json-events` save event carries
enough metadata (path, fps, frames, dimensions) that hosts need zero probe decodes.

## Validation

- CLI flag test: skip path emits the skipped marker and no child validator process.
- Default path unchanged (existing health tests stay green).
- Save event contains fps/frames/dimensions (add if missing) with a contract test.

## Progress checklist

- [x] API + CLI flag (`--no-validate-health` on Wan generate and SeedVR2 video
      restore; `validate_health` already existed on the save/restore APIs) +
      metadata marker (`health_check: "skipped"` in GeneratedVideo metadata
      and the save runtime event)
- [x] Save-event metadata completeness: Wan save events now carry fps, width,
      height, total_frames (None-dropped, additive — old consumers unaffected)
- [x] Tests (router: skip path asserts save kwargs + event fields; defaults
      pinned ON in the wan and seedvr2 routing tests)
- [x] docs/api.md + python-integration.md mention (api.md save-event section
      in cycle 1; python-integration.md save_kwargs paragraph in cycle 2)

## Implementation note (2026-07-22)

The embedded-metadata marker rides `GeneratedVideo._get_metadata()`'s dict, so
it lands in the JSON sidecar (--metadata) and any embedded metadata path; the
runtime-event marker is unconditional under --json-events. Image routes are
untouched (no health check exists there).
