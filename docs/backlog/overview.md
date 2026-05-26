# Backlog Overview

## Project summary

MLX-Gen is an independent Apple Silicon image and video generation package derived from
[mflux](https://github.com/filipstrand/mflux). The backlog tracks model integration work,
compatibility fixes, release-readiness work, and follow-up investigations that should survive
outside chat history.

## Counts

| State | Count |
| --- | ---: |
| Planned | 1 |
| Proposed | 0 |
| Completed | 0 |
| Deprecated | 0 |
| Recurrent | 0 |

## Next recommended work

1. Finish the [model integration roadmap](planned/0001_model_integration_roadmap.md) in priority
   order, starting with automated publication audits, supported q4/q8 validation, and
   gated-derivative hygiene.
2. Continue ERNIE-Image/Turbo after the Turbo text-to-image, Prompt Enhancer, and q4/q8
   validation work: add stronger Diffusers parity tests and non-turbo validation.
3. Continue Wan2.2 after the first text-to-video and first-frame image-to-video milestones: add
   q8/q4 validation, stronger quality/performance checks, progress/cancel APIs, and keep SeedVR2
   as the lower-risk existing-code video utility track.

## Planned ledger

| ID | Item | Area | Priority | Status |
| --- | --- | --- | --- | --- |
| 0001 | [Model integration roadmap](planned/0001_model_integration_roadmap.md) | Models, routing, quantization, UX | P0-P3 | Planned |

## Completed ledger

No completed backlog items yet.

## Deprecated ledger

No deprecated backlog items yet.

## Process

- New backlog items must use a unique four-digit global prefix.
- Planned items need current code reality, scope, non-goals, validation, and ADR status.
- When implementation completes, move the item to `docs/backlog/completed/` and append completion
  evidence instead of deleting the planning history.
- If a backlog item establishes lasting architecture policy, create or update an ADR before
  closure.

## Planning notes

- Created the backlog system on 2026-05-25 while triaging local and online model integration
  candidates for MLX-Gen and AbstractVision.
- Refined the model roadmap on 2026-05-25 after checking Hugging Face model sizes, licenses,
  local cache state, and current T2I/I2I/T2V/I2V popularity.
- Added 2026-05-25 AbstractFramework publication audit note: checked Qwen, FLUX.2 Klein/Base,
  Z-Image, and Z-Image-Turbo q4/q8 repos are complete; future work should automate this audit.
- Added 2026-05-25 collection follow-up: several q8 and non-turbo Z-Image repos still need to be
  added to the Hugging Face `AbstractFramework / mlx-gen` collection once collection write
  permission is available.
- Added 2026-05-25 ERNIE follow-up: ERNIE Image Turbo text-to-image support exists with BF16,
  q8, q4, and optional Prompt Enhancer validation; remaining work is parity coverage and
  non-turbo ERNIE-Image.
- Added 2026-05-26 Wan follow-up: initial Wan2.2 TI2V text-to-video and first-frame
  image-to-video support exists with MP4 output and focused tests; remaining work is quality
  validation, quantized validation, progress/cancel APIs, and broader Diffusers parity.
