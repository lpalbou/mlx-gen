# Wan A14B i2v multi-frame context head conditioning (`--context-frames`)

## Metadata

- Created: 2026-07-27
- Status: Implemented (shipped as EXPERIMENTAL; zero-shot probe measured
  below — Lightning 4-step, one seed pair, small resolution; broader recipes
  unverified)
- Effort: S-M (code) + GPU probe gate (quality evidence)

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
  (real-checkpoint proof before the capability ships),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md) (non-A14B-i2v
  routes reject the parameters loudly; no silent single-frame fallback).
- ADR impact: None.

## Context (investigation provenance)

The 2026-07-27 storyboard blueprint of a host application
(its backlog `proposed/storyboard_frame_consistency_2026_07/`,
validated by its R4 forensics + probe wave) confirmed on a failed
4-scene film that chaining scenes by seeding scene N+1 from ONE still frame of
scene N resets motion at every boundary — one frame carries no velocity. The
motion-carry literature (SkyReels-V2 overlap, SVI 5-frame handover, VACE
temporal extension) conditions the next clip on the predecessor's last K
frames instead. Item 0097 (the `last_image` bracket) named prefix-K-frames
conditioning an explicit NON-goal pending its own gate; this item IS that
gate, mandated by the blueprint's Phase 1 (Layer 1, "K-frame head
conditioning on A14B i2v", verdict row 5 - PROBE).

Engine ground truth (verified at 0.25.0): the A14B i2v conditioning layout is
exactly the K-frame insertion point — condition `[first, zeros x (n-2),
last?]` plus a keep-mask whose frame-0 row is repeated `temporal_scale = 4`
times and packed 4-to-a-latent-frame (`wan2_2_ti2v.py` `_load_video_condition`
/ `_build_video_condition`), concatenated as 20 channels onto the 16 noisy
latent channels. A K-frame head changes ONLY the frame list and the mask
fill; the packing, channel contract, and denoise-loop concat are untouched.
The VAE's causal 4x temporal packing forces the conditioned head to fill
whole latent groups: head K = 1 (mod 4), i.e. K in {5, 9, 13}.

## The surface as shipped

- CLI (`mlxgen generate` / `mlxgen-generate-wan`): `--context-frames <png>...`
  = the ordered frames that FOLLOW `--image-path` in the motion being
  continued; the conditioned head is `[--image-path, *--context-frames]`.
  Pass 4, 8, or 12 context frames (heads 5, 9, 13). `--context-noise <0-1000>`
  = optional SkyReels-`addnoise_condition`-style noise on the conditioned
  head (~20 recommended when used). Both forwarded by the router untouched,
  like `--last-image`.
- Python: `generate_video(context_image_paths=[...], context_noise=...)` on
  both Wan variant signatures (WanVace accepts and rejects explicitly — the
  0097 bind-contract lesson).
- Why "start frame + tail frames" instead of one replacing list: the router
  resolves the i2v plan from image presence (`min_images=1` on
  `wan.first-frame`), hosts keep their existing `--image` staging and add
  staged tail PNGs, the K=1 fallback is literally "omit --context-frames",
  and the mod-4 count check catches the passed-all-K-frames misuse loudly.
- Constraints (fail loudly, ADR 0002): requires `image_path` (so t2v and v2v
  carrying context fail), A14B-i2v-only (TI2V-5B `expand_timesteps` and VACE
  reject; CLI rejects before weight load), context count multiple of 4, head
  <= 13, `num_frames >= head + 4` (one free latent group; +1 with
  `--last-image`), `context_noise` in [0, 1000] and only with context frames.
  Composes with `--last-image` (head + bracket tail in one mask).
- Mask semantics: `mask[:, :, :head] = 1`, zeros beyond (last-image tail slot
  unchanged); packing untouched. K=1 is bitwise identical to the shipped
  single-frame path (test-pinned).
- `context_noise` applies IN LATENT SPACE to the head's latent channels only
  (`latent = (1 - t/1000) * latent + (t/1000) * noise`), outside the condition
  cache (seed-dependent), deterministic per seed via a seed-derived key; mask
  channels stay binary.
- Timestep side: A14B keeps the scalar timestep — that is the diffusers
  convention for this conditioning mechanism (the mask channel carries the
  "pinned" signal). The TI2V-5B `WanTimestepPolicy` first-frame-mask machinery
  belongs to a different conditioning route and is not touched; per-frame
  timestep vectors are Pusa-class Phase 2 work.
- Truth surfaces: metadata sidecar `context_image_paths` + `context_noise`,
  `--config-from-metadata` replay, failure manifest fields, additive
  `supports_context_frames` capability field on `wan.first-frame`
  (capabilities `schema_version` 5 -> 6, same additive convention as
  `supports_last_image`). Condition-cache identity includes every context
  frame's `(path, mtime_ns, size)`.

## Honesty constraints (quality gate)

Zero-shot multi-frame acceptance on Wan 2.2 A14B is CONTESTED (blueprint
grade C, no solid replication; known "flash" risk at the context boundary —
the mask says "keep" over frames the model never saw as conditioning at
train time). The blueprint's Gate L1 as amended by R4:

- Calibration arm: the same flow metrics measured INSIDE a single uncut clip
  are the "no seam" reference.
- Motion carry: Farneback mean-flow direction cosine between scene 1's last
  ~5 frames and the continuation's first ~5 FREE frames (after the head)
  >= 0.7x the within-clip calibration value; magnitude ratio in [0.5, 2.0].
- Flash: per-frame luma/contrast delta spike over the first free transitions
  < 2x the within-clip max delta.
- No visual regression in stills (described honestly).

FAIL after iteration -> the surface still ships behind its experimental
framing (Phase 2's SVI LoRA reuses the same conditioning surface), with the
failure documented here in numbers.

## Probe results (2026-07-27)

Rig: this checkout (0.25.0 + this item), M3 Ultra 128 GB. Scene A =
`untracked/context_probe_2026_07_27/scene1.mp4`, t2v A14B q8 Lightning
4-step (t2v Seko-V1.1 pair, `steps 4, guidance 1.0/1.0, flow_shift 5.0`),
480x272, 49 frames, fps 16, seed 4242, prompt "a starship lifts off in a
snowstorm, camera static" — a shuttle mid-liftoff with strong coherent tail
motion (tail mean flow 1.29 px/frame). Continuations: i2v A14B q8, i2v
Seko-V1 Lightning pair, same recipe/canvas, prompt "the starship continues
rising through the snowstorm, camera static", head = scene A's last K frames
extracted as lossless PNGs (`--image` = first of the K, `--context-frames` =
rest). Metrics per the R4-amended L1 gate (`measure.py`): Farneback mean-flow
direction cosine + magnitude ratio between scene A's last 4 transitions and
the continuation's first 4 FREE transitions (after the head); flash = max
per-transition |delta mean gray| over the first 3 free transitions;
calibration = the same window pair at every interior position of the UNCUT
scene A.

Calibration (scene A, no seam): cosine median 1.000 (p10 0.999), magnitude
ratio median 1.08 [p10 0.93, p90 1.16], max within-clip luma delta 1.15.
Gates: direction cosine >= 0.70 (0.7 x median); flash < 2.30 (2 x max);
ratio band [0.5, 2.0].

| arm | K | noise | seed | dir cosine | mag ratio | flash(free3) | head MAE | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k1 (baseline) | 1 | 0 | 5151 | 0.984 | **1.90** | 0.82 | 5.1 | dir PASS, flash PASS, ratio at band edge (2x restart) |
| k5 | 5 | 0 | 5151 | 0.999 | **0.90** | 3.17 | 11.6 | dir PASS, ratio PASS, flash FAIL (2.8x calib max) |
| k9 | 9 | 0 | 5151 | 0.987 | 1.49 | 3.43 | 13.4 | dir PASS, ratio PASS, flash FAIL |
| k5 + noise 20 | 5 | 20 | 5151 | 0.999 | 0.93 | 3.58 | 12.2 | flash unchanged |
| k5 + noise 60 | 5 | 60 | 5151 | 1.000 | 0.97 | 3.08 | 13.6 | flash unchanged |
| k5 + noise 100 | 5 | 100 | 5151 | 0.998 | 1.05 | 3.64 | 15.2 | flash unchanged, head loosest |
| k1 (seed B) | 1 | 0 | 6262 | 1.000 | 1.14 | 0.50 | 4.0 | lucky restart this seed |
| k5 (seed B) | 5 | 0 | 6262 | 1.000 | **0.71** | 4.07 | 10.7 | ratio PASS, flash FAIL |

Per-transition detail (seed 5151): scene A's tail runs ~1.4 px/frame mean;
k1's very first transitions run 1.70, 3.22, 1.78, 3.12 (an immediate ~2x
speed restart), while k5's free section runs 1.0-2.0 — the momentum carry is
visible in the raw speed profile, not just the window mean. Across seeds the
single-frame restart speed is a lottery (1.90 vs 1.14); the K=5 head tracks
the tail consistently (0.90, 0.71).

Flash structure: the spike sits at/after the conditioned-to-free boundary
(k5 seed A: 3.17 at transition 5 vs in-head deltas ~0.5-0.9) and the K arms
also run livelier luma dynamics deeper into the free section (3.9-5.6 at
t10-11, engine-flare pulses). Full-size stills at the boundary
(`inspect_k5_f04/05/06.png`): structural continuity is perfect — no ghost,
no morph, no blink; the artifact is a mild 1-2 frame flare/exposure
brightening. By the 0097 bracket probe's own artifact bar ("max brightness
step 5.9 = no exposure jumps") this is small in absolute terms; it fails
HERE because scene A's within-clip reference is exceptionally smooth
(max 1.15). `context_noise` does not move it (3.1-3.6 flat across 0/20/60/
100), so the knob defaults to OFF and exists for adapter recipes.

Visual verdict (labeled review, not blinded — single probe): all arms
keep the shuttle identity and continue the climb; k1 arms read as a speed
jump at the cut, k5/k9 read continuous with a subtle flame brighten at the
boundary. No structural regression in any arm.

### Direction wave (scene Q: lateral tracking shot)

Scene Q = t2v "sideways tracking shot ... slides horizontally to the right
past a snowy mountain ridge" (seed 8484; a first attempt with "the camera
pans steadily from left to right" produced NO pan at all — a live specimen
of the blueprint's action-suppression failure; preserved as `scenep.mp4`).
Scene Q's tail is strongly lateral (u +1.4..+8.4 px/frame, bursty foreground
parallax; within-clip calibration is much looser: cosine median 0.983 but
p10 -0.534, ratio band [0.15, 4.22], max luma delta 15.94). Continuation
prompt deliberately names NO direction ("continues its steady sideways
tracking motion").

| arm | K | seed | dir cosine | mag ratio | flash(free3) | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| q_k1 | 1 | 9494 | 0.993 | 1.86 | 9.96 | direction kept, ~2x speed restart |
| q_k5 | 5 | 9494 | 0.976 | 2.10 | 12.54 | direction kept, ratio outlier (busy-scene flow noise) |
| q2_k1 | 1 | 1111 | 1.000 | 2.12 | 16.12 | direction kept, ~2x speed restart |
| q2_k5 | 5 | 1111 | 0.996 | 1.05 | 21.64 | direction kept, speed carried |
| q3_k1 | 1 | 2222 | 0.996 | 2.38 | 7.02 | direction kept, ~2x speed restart |

Honest interpretation: the single-frame DIRECTION reset did not manifest on
either probe scene — the rocket still implies its direction semantically,
and scene Q's last frame leaks the motion axis through heavy horizontal
motion blur on the foreground pine (`framesq/f48.png`), with the sign
apparently recoverable too (no flip in 3 seeds). Scenes whose stills
genuinely hide motion direction (ball at apex, pendulum mid-swing) remain
the untested worst case. What DOES replicate everywhere is the SPEED reset:
k1 restarts at ~2x tail speed in 5 of 6 runs across both scenes and 4 seeds
(1.90, 1.14, 1.86, 2.12, 2.38), while k5 tracks the tail in 3 of 4 (0.90,
0.71, 1.05; one 2.10 outlier on the busiest scene). On the busy scene the
boundary flare disappears into the clip's own dynamics (gate 31.9, spikes
12.5-21.6): flash severity is relative to source smoothness.

### Mask-feather iteration (probe-only, REJECTED)

Last free knob from the blueprint's iteration list: scale the LAST head
latent group's keep-mask (probe-only monkeypatch in `run_arms2.py
--feather`; never shipped). Scene A, seed 5151, K=5: feather 0.5 -> flash
9.57; feather 0.25 -> flash 9.30 (vs 3.17 unfeathered). A fractional keep
mask is even further out of the training distribution (masks are binary at
train time) and TRIPLES the boundary discontinuity while motion carry stays
(ratio 0.83 / 1.00). Rejected with numbers; not exposed as a parameter.

### Gate verdict (final, 18 GPU runs)

Against the R4-amended L1 gates as pre-registered:

- Direction cosine >= 0.7 x calibration: PASS on every K arm (0.965-1.000)
  — but non-discriminative on these scenes (K=1 also passed; both probe
  scenes leak direction through the still).
- Magnitude ratio in [0.5, 2.0]: PASS for K=5 in 3 of 4 runs (0.71-1.05;
  one 2.10 outlier on the busiest scene); the K=1 baseline sits at or past
  the band edge in 5 of 6 runs (1.86-2.38) — the replicated momentum win.
- Flash spike < 2 x within-clip max: **FAIL on the smooth scene** (3.08-4.07
  vs bound 2.30) for every K>1 arm, after iterating context noise (0/20/60/
  100 — flat) and mask feather (0.25/0.5 — 3x worse). PASS on the busy
  scene (12.5-21.6 vs bound 31.9). The artifact is a mild 1-2 frame
  flare/exposure step, structurally clean in stills; its gate severity is
  relative to how smooth the source clip is.
- No visual regression in stills: PASS (labeled review; blinded protocol
  needs a human).

Formal verdict: **FAIL by the letter of the flash gate on smooth sources;
the momentum objective itself measured decisively positive and replicated.**
Per the pre-registered failure path the surface ships EXPERIMENTAL (docs
framed accordingly, capability-gated), the storyboard host decides the
continue-seam default with these numbers on the table, and Phase 2's SVI
LoRA (trained FOR multi-frame conditioning; expected to remove both the
flare and the loose head reproduction) reuses this exact surface unchanged.
Recommended host default given the evidence: K=5, context_noise off — the
K=1 double-speed restart is the reported, always-visible failure;
the K=5 flare is 1-2 frames, mild, and masked by any non-static content.

Probe artifacts preserved in `untracked/context_probe_2026_07_27/`
(scenes, 15 continuation arms + sidecars, extracted heads, contact sheets,
full-size boundary stills, `measure.py`, `results_*.json`, manifests).

## Progress checklist

- [x] Head layout + mask fill generalized (`_build_video_condition`,
  `_load_video_condition`), K=1 bitwise-unchanged pin
- [x] Request validation + rejection matrix (`wan_video_request.py`), VACE
  explicit rejection
- [x] CLI flags, pre-load rejection, metadata replay, failure manifest
- [x] Capability field + schema v6; router forward tests
- [x] Zero-shot probe (scoreboard below)
- [x] Docs (api.md, wan-video.md, python-integration.md, llms parity,
  CHANGELOG)
