# Completed: Wan A14B i2v `last_image` bracket conditioning (first+last frame)

## Metadata

- Created: 2026-07-25
- Status: Completed (shipped as EXPERIMENTAL; quality probe PASSED, measured
  below — one Lightning 4-step probe pair, in-distribution target; broader
  recipes remain unverified)
- Completed: 2026-07-25 — released in 0.25.0 (tag `v0.25.0` from `2452f0c`,
  workflow 30162410505 green; PyPI + GitHub Release verified); release
  checklist and smoke evidence in [0101](0101_release_0_25_0.md).
- Effort: S (code) + A/B gate (quality evidence)

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
  (the route needs real-checkpoint proof before any capability row ships),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md) (if A14B quality
  fails the gate, reject the flag on that route — no silent single-frame fallback).
- ADR impact: None until the route ships.

## Context (investigation provenance)

The 2026-07-25 storyboard consistency investigation (host application
backlog `proposed/storyboard_consistency_2026_07/`) traced why chained
storyboard scenes lose subject identity: each scene conditions on exactly ONE
VAE-encoded frame (the predecessor's last frame), which is systematically the
most degraded frame of a drifting clip. Measured on the real film chain:
predecessor-tail Laplacian-variance sharpness collapsed 952 → 497 → 93 across
the three handoffs, and a chain rerun reproduced the same collapse class
(30 at the S3→S4 handoff).

A verified porting fact makes a second anchor cheap: the local diffusers
reference (`diffusers/pipelines/wan/pipeline_wan_i2v.py`,
`prepare_latents`) implements `last_image` bracket conditioning in EXACTLY the
tensor layout this repo already ported for single-frame i2v:

- without `last_image`: `video_condition = cat([image, zeros(num_frames-1)])`
  — identical to `_build_first_frame_video_condition` in
  `src/mflux/models/wan/variants/wan2_2_ti2v.py`;
- with `last_image`: `video_condition = cat([image, zeros(num_frames-2),
  last_image])` and the latent mask keeps BOTH frame 0 and the final frame at
  1 before the same temporal-scale expansion this repo already performs in
  `_load_video_condition`.

The delta is ~30 lines in `_load_video_condition` (accept an optional second
image, pad with `num_frames-2` zeros, set the tail mask row) plus CLI/config
plumbing for a `--last-image` style flag.

## Why it might matter

For chained storyboards, bracket conditioning lets a host pin a scene's END
state (e.g. the next scene's intended start, or a clean subject reference),
turning open-loop drift into interpolation between two anchors. It is the
strongest identity lever that stays on the full-quality A14B path (unlike the
VACE 1.3B reference route, see 0100).

## Honesty constraints (quality gate)

- Official first+last-frame training exists for Wan 2.1 (`Wan2.1-FLF2V-14B`);
  Wan 2.2 A14B first+last quality is EMPIRICAL — community reports are
  positive but this repo must gate on its own A/B before advertising it.
- Gate: fixed-seed pairs (single-frame vs bracketed) on the preserved
  storyboard chain case; verdict on subject identity at the far end plus
  mid-clip motion plausibility (bracketing can produce a "morph" if the two
  anchors are inconsistent).
- If A14B fails the bar, reject `last_image` on 2.2 A14B routes with a clear
  error naming the evidence (ADR 0002), and record the verdict here.

## What we want to do

1. Port the bracket layout in `_load_video_condition` (optional last frame,
   mask tail row), mirroring the diffusers reference exactly.
2. Expose it on the Wan A14B i2v route (`--last-image <path>`), fail-closed on
   variants whose conditioning layout differs (TI2V-5B `expand_timesteps`
   path, VACE).
3. Run the fixed-seed A/B gate on the preserved storyboard case; keep the
   clips as the proof bundle.
4. Ship the capability row + docs only on a pass verdict.

## Non-goals

- Prefix-K-frames conditioning (K>1 leading frames): expressible in the same
  function but out-of-distribution for every Wan 2.x checkpoint this repo
  ships; SkyReels-V2-class territory. Not gated here; do not fold it in.
- Wan 2.1 FLF2V checkpoint integration (separate model family decision).

## Dependencies and related tasks

- Host application companion: `proposed/storyboard_consistency_2026_07/` (handoff
  frame picker is the host-side half of the same failure).
- Related here: 0099 (step-grid scheduler parity for honest Lightning A/Bs),
  0100 (VACE identity-anchor recipe, the draft-tier alternative).

## Validation

- Layout parity: tensor-shape and mask-row assertions against the diffusers
  reference implementation on a tiny synthetic case.
- Quality: the fixed-seed A/B gate above, on real checkpoints (ADR 0001).

## Progress checklist

- [x] Bracket layout ported behind an optional argument
- [x] Route gating (A14B i2v only; fail-closed elsewhere)
- [x] Fixed-seed A/B gate on the storyboard chain case (probe below)
- [x] Capability row + docs (shipped as EXPERIMENTAL with the probe verdict)

## Implementation record (2026-07-25, pending release)

- `_load_video_condition` ports the diffusers `WanImageToVideoPipeline`
  `last_image` variant exactly: condition video
  `[first, zeros x (num_frames - 2), last]` (built via `_build_video_condition`
  with the F2/0089 precision-first discipline — the no-last path is bitwise
  unchanged, test-pinned) and the latent mask keeps BOTH endpoint frames at 1
  before the existing temporal-scale packing. The mask packing is
  test-verified against a line-for-line numpy port of the diffusers reference
  (`tests/wan/test_wan_last_image_bracket.py`).
- The last image maps through the SAME resolved canvas and `resize_mode` as
  the first frame (`_normalized_condition_frame`, one geometry for both
  anchors); the canvas itself stays derived from the FIRST image (i2v
  source-aspect rules unchanged).
- Condition-cache identity: `_encode_video_condition` keys now include the
  last-image identity `(path, mtime_ns, size)` (or None), so a bracketed
  condition can never alias the single-frame condition of the same first
  image, and overwriting either source file invalidates the entry.
- Surface: `generate_video(last_image_path=...)` on BOTH Wan variant
  signatures (WanVace accepts and rejects explicitly — the bind-contract
  lesson), `--last-image` CLI flag, metadata `last_image_path`,
  `--config-from-metadata` replay, additive `supports_last_image` capability
  field on `wan.first-frame` rows (True only for non-expand-timesteps i2v =
  A14B i2v), failure-manifest field. Runtime start event unchanged.
- Fail-closed routes (ADR 0002): text-to-video and video-to-video (no
  image_path), TI2V-5B (`expand_timesteps` — its 48-channel first-frame path
  has no last-frame slot; the 36-channel concat path this feature rides is
  `is_image_to_video and not expand_timesteps`, i.e. exactly I2V-A14B),
  Wan VACE, and `num_frames < 2`. CLI rejects before weight load.

## Quality probe (2026-07-25, ADR 0001 gate) — PASSED, shipped as EXPERIMENTAL

Fixed-seed pair on the preserved storyboard chain case (`/tmp/bpx_ab/`
armA_s2, the film's ship scene): first anchor = frame 0, target last anchor =
frame 48 (2.0 s of REAL future motion from the same continuous shot, so the
target is in-distribution but non-trivial: first-vs-target MAE 43.2/255,
gray NCC 0.475). Both runs: I2V-A14B q8, Lightning 4-step CFG-off storyboard
recipe (`steps 4, guidance 1.0/1.0, flow_shift 5.0, unipc`, Seko-V1 rank-64
LoRAs), 480x240, 33 frames, seed 4242, identical prompt. Artifacts + metrics
script preserved in `untracked/flf_probe_2026_07_25/`.

| Metric (final frame vs target) | baseline (first only) | bracket (first+last) |
| --- | --- | --- |
| MAE (0-255) | 56.1 | **4.6** (first-anchor reproduction floor is 3.2) |
| PSNR | 11.1 dB | **31.1 dB** |
| gray NCC | 0.335 | **0.9948** |
| final-frame Laplacian-variance sharpness | 55 (whiteout blur) | **711** (target: 713) |

Artifact scan (mid-clip plausibility): bracket max consecutive-frame
brightness step 5.9 gray levels (baseline 17.5), max inter-frame MAE 11.8 at
mid-motion (baseline 23.6), final hop 9.0 — BELOW the clip's own mid-motion
peaks, so no terminal snap onto the anchor; brightness arc is smooth
(112 -> 152 -> 132), no flashes or exposure jumps; mid-frame visual check
clean (no ghosting/morph, ship + mountains intact). Side effect worth
noting: the baseline run ends in the investigation's documented
drift-to-whiteout failure (ship swallowed by blur clouds), which the bracket
suppressed entirely in this pair.

Verdict: bracket adherence VALIDATED on this probe — the clip ends at
VAE/codec-floor distance from the requested last frame with no mid-clip
artifacts. Honesty bounds: ONE pair, Lightning 4-step CFG-off only, target
drawn from the same shot's real future (the favorable storyboard handoff
case; adversarial targets — different shot, mismatched lighting — were not
probed). The 20-step CFG-on pair was NOT run: ~10x Lightning cost per scene
exceeded this wave's GPU budget (~2 short runs used of 4). The flag therefore
ships as EXPERIMENTAL with these numbers quoted in docs; item 0100's
supersession decision (VACE recipe) can now be taken against a measured
result.

## Cycle-2 verification (2026-07-25, no code defects found)

- Packing re-verified line-for-line against the diffusers reference
  (`pipeline_wan_i2v.py` `prepare_latents`), including the
  repeat-interleave/view/transpose mask packing semantics and the
  bracket-endpoint rows; the port's extra `[:, :, :latent_frames]` slice is a
  no-op under the enforced `num_frames % temporal_scale == 1` contract. The
  F2 build-in-precision path is cast-commutative with the reference's
  concat-then-cast (bitwise; zeros cast exactly).
- Probe audited: both sidecars share seed 4242 and every parameter except
  `last_image_path`; the metrics script re-run reproduced every quoted
  number exactly (MAE 4.62/56.13, NCC 0.9948/0.3354, sharpness 711.2/713.3,
  brightness steps 5.94/17.53, final hop 8.98 vs mid-motion 11.83). Wording
  in docs/CHANGELOG confirmed appropriately hedged (EXPERIMENTAL + bounds).
- Replay round-trip executed against the probe bundle's real bracket sidecar:
  every recorded field restores exactly (seed, steps 4, flow_shift 5.0, both
  anchors, LoRAs, quantize 8).
- One convention fix: `supports_last_image` shipped without the additive-field
  capabilities `schema_version` bump the `supports_video_mask` precedent
  established; bumped 4 -> 5 (no host reads the version today — verified
  the host application never consumes it).
- New regression pins: condition-cache FIFO eviction order with plain +
  bracketed pairs, and resize_mode reaching the LAST anchor through the same
  geometry as the first frame (pad letterbox vs stretch discriminates).
