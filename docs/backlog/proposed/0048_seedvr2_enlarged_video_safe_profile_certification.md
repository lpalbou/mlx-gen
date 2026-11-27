# Proposed: SeedVR2 enlarged-video safe-profile certification

## Metadata

- Created: 2026-06-21
- Status: Proposed

## ADR status

- Governing ADRs:
  - [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
  - [ADR 0004](../../adr/0004_seedvr2_video_host_safety_and_proof_boundaries.md)
  - [ADR 0005](../../adr/0005_seedvr2_video_quality_proof_requires_five_second_reader_first_clips.md)
- ADR impact: None if this stays a validation-and-proof item for when enlarged SeedVR2 video may
  be treated as a safe documented public profile.

## Context

Release `0.18.20` closed the bounded SeedVR2 quality revalidation loop for the current public
proof surface:

- completed item [0032](../completed/0032_seedvr2_video_restoration_upscaling.md) now records the
  accepted June 21 five-second `1x 29/8` route-correctness proof and the bounded `2x 29/8`
  `3B` versus `7B` quality comparison;
- public docs now preserve only the accepted June 21 SeedVR2 validation bundle under
  `docs/assets/validation/seedvr2-video-2026-06-21/`;
- the public CLI still treats enlarged video output as an explicit unsafe-memory path and requires
  `--force-unsafe-video-memory`.

That means the family-ranking question is closed for the current bounded proof, but the safe
public-profile question is not.

## Current code reality

- `src/mflux/models/seedvr2/cli/seedvr2_upscale.py` defaults video restore to a conservative safe
  profile and rejects enlarged video output unless `--force-unsafe-video-memory` is passed.
- `src/mflux/models/seedvr2/variants/upscale/seedvr2.py` now supports the accepted bounded `1x`
  and `2x` proof profiles with clip-global noise continuity, causal temporal VAE slicing, and
  streamed chunk windows.
- `docs/upscaling.md` treats `1x 29/8` as the public route-correctness proof and `2x 29/8` as the
  stronger bounded visual comparison regime, but still describes `2x` as an explicit unsafe-memory
  run rather than a host-safe documented default.
- Current accepted `2x` proof shows that enlarged SeedVR2 output can look better on the bounded
  archival slice, but it does not yet prove that the same recipe belongs in the safe public
  contract on a busy machine.

## Problem or opportunity

The current docs and runtime are honest, but there is still a product gap:

- users can see a good bounded enlarged-video result;
- the public route cannot recommend that recipe as safe by default;
- there is no durable backlog item scoped only to certifying, or explicitly refusing to certify,
  an enlarged SeedVR2 video public profile.

## Proposed direction

Evaluate one bounded enlarged SeedVR2 recipe as a candidate safe public profile.

The minimal honest outcome is one of:

1. certify one enlarged profile as safe and document it; or
2. explicitly conclude that enlarged SeedVR2 video should remain an unsafe opt-in path on current
   hardware/runtime assumptions.

## Why it might matter

This is the remaining SeedVR2 proof question that affects public guidance. It is narrower than the
original 7B-quality revalidation work and should stay separate from the practical audio follow-up.

## Promotion criteria

Promote after:

- the accepted `0.18.20` bounded proof bundle is in place and stable;
- the immediate practical SeedVR2 follow-up in completed item [0046](../completed/0046_seedvr2_video_audio_copythrough.md)
  is already closed and no longer blocks profile certification; and
- there is a concrete reason to revisit the safe-memory boundary rather than leaving enlarged video
  as an explicit expert-only path.

## Validation ideas

- Re-run a five-second reader-first enlarged proof clip with the exact candidate profile.
- Measure peak MLX, max RSS, and final post-run memory state on the target machine.
- Validate frame count, FPS, and tail continuity.
- Confirm that the candidate enlarged profile still looks better than the accepted source-size
  proof on preserved motion strips and detail crops.
- Decide explicitly whether the memory and stability behavior are good enough for the safe public
  contract.

## Non-goals

- Do not reopen the bounded 3B-versus-7B quality ranking question already settled by the current
  public proof bundle.
- Do not change the public safe profile silently.
- Do not broaden this into generic SeedVR2 performance tuning or full-length movie restoration.

## Implementation notes

Start from the current accepted June 21 public bundle and the live CLI guardrails, not from older
June 18/19/20 experiments. The goal here is not "find another better-looking clip"; it is
"decide whether enlarged SeedVR2 video can honestly belong in the safe documented public
contract."
