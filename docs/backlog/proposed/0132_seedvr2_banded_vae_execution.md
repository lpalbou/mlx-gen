# Proposed: banded execution of the SeedVR2 video VAE

## Metadata

- Created: 2026-11-26
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [0002](../../adr/0002_no_silent_automatic_fallbacks.md) (a frame that cannot fit is
  refused with its reason, not run into swap).
- ADR impact: none.

## Context

0.40.2 made the SeedVR2 video VAE's causal 3D convolutions sum their temporal taps one output frame
at a time, bit-identical to the single MLX call they replace; the largest frame a restoration can
plan is now limited by the VAE's remaining full-frame tensors (see
[SeedVR2 video memory](../../seedvr2-video-memory.md#limits)). A second design was built and measured
for that work and not landed.

## Candidates

1. **Run the VAE over horizontal bands of rows.** Every convolution runs on bands fed with the
   input rows its receptive field needs, GroupNorm statistics are accumulated band by band and then
   applied, the causal temporal memory keeps its rows per band, and the mid-block attention runs in
   query chunks. Measured on the convolution formulation of 0.40.1 (VAE only, 17 frames, 1 GiB
   bands, MLX peak above the weights): decode 22.3 GB and encode 12.9 GB at 1280x720, 12.1 GB
   decode at 960x528, 6.5 GB at 640x352 (61.9 GB whole-frame then); decode 39 s and encode 27 s at
   1280x720. That would lift the frame-size limit toward 1080p on a 48 GB budget.
2. **The cost is exactness.** Statistics summed band by band differ from MLX's own reduction by
   summation order: decoded frames were 61.3 / 60.6 / 56.7 dB against whole-frame execution at
   320x192 / 480x272 / 640x352 (at most 4 to 8 levels of 255), and the same residue appears with one
   band covering the frame. Outputs would be equal within rounding, not bitwise.
3. **Not measured:** banding on top of the 0.40.2 per-frame tap sums, and the full-pipeline peak
   with banding (the figures above are VAE-only). The prototype is not in the repository.

## Promotion criteria

Promote when restoring frames above the current limit (about 1.4 megapixels on a 48 GB budget) is
requested, and only if results equal within rounding rather than bitwise are acceptable for that
route.
