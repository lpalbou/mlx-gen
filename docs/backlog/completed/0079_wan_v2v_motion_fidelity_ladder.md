# 0079 - Wan V2V Motion-Fidelity Ladder And Control Run

- Status: completed (2026-07-05)
- Scope: measured strength-vs-motion-preservation contract for Wan `Wan2.2-T2V-A14B` plain
  video-to-video; documentation truth for motion claims
- Proof: `docs/assets/validation/motion-ladder-2026-07-05/` (committed mirror: README, metrics
  JSONs incl. anchors, measurement scripts, person mask, contact sheets, face crops); full mp4
  bundle preserved locally in `validation_outputs/motion_fidelity_ladder_2026_07_05/`
- Validation: 5 adversarial subagents - experiment-design attack (pre-GPU), metrics/command
  verification (pre-GPU, GO), independent results recomputation (all numbers reproduced within
  0.01; conclusions SUPPORTED), user-standpoint judge, and evidence-standard judge (both
  verdicts' fix lists fully applied)

## Problem

The 2026-07-05 motion investigation proved gesture re-synthesis at the Lightning point is
expected SDEdit behavior, and the docs were corrected qualitatively - but the strength advice
("lower toward 0.5-0.6") was uncalibrated: no measured evidence said where gesture preservation
actually lives, whether the edit still lands there, or what the Lightning fast path can reach.

## What shipped

1. Measured ladder (same seed 8602, same prompt + custom negative, 25 frames, 20 steps CFG-on,
   q8 A14B): strength 0.5 / 0.6 new runs, 0.7 reused from the 2026-07-04 proof (provenance and
   comparability argued in the bundle README), 0.8 new run, plus a zero-GPU null row (source
   re-encoded through the production writer) proving the metric floor is clean (ratio 1.00,
   r 1.00). Result: gesture-timing correlation 0.86 / 0.90 / 0.73 / 0.20 - gestures survive at
   0.5-0.6 (one band; the 0.86 vs 0.90 ordering is statistically indistinguishable), mostly at
   0.7, and are re-synthesized at the 0.8 default. The man-to-woman edit landed at every
   strength (pre-registered criterion + multi-frame independent recheck).
2. Paired control run at the Lightning point (seed 4242, one change: prompt gains "gesturing
   naturally with his hands"): gesture-window motion energy 0.49 -> 0.91 with gesturing visibly
   restored; both correlations below the n=14 significance floor, so the conclusion rests on
   energy + visuals - prompt language recovers the class of motion, never its timing. One seed;
   framed as suggestive throughout.
3. Warm-start sigma table computed from the real scheduler (mirrors `_video_to_video_timesteps`
   exactly), exposing that strength co-varies noise level AND expert routing (0.5/0.6 skip the
   high-noise expert entirely; `--guidance` inert, `--guidance-2` carries CFG).
4. Docs: `docs/wan-video.md` gains "Motion Fidelity Versus Strength" (table + guidance + a
   copy-pasteable motion-preserving restyle recipe command); the Lightning section and
   `docs/lora.md` now state the fast path cannot reach the motion-preserving band (mutual
   exclusivity, cross-referenced both ways); `docs/faq.md` and `docs/getting-started.md` cite
   the measured band; README/llms index lines updated; changelog Added + Fixed entries.
5. Measurement methodology preserved and reproducible from the committed mirror alone:
   `measure_motion.py` (person-region motion energy, gesture-window ratio + Pearson r with
   stated significance floors, dilated-mask sensitivity variant), `make_null_row.py`,
   `person_mask.png`, `metrics_anchors.json` preserving the prior-run anchor measurements the
   docs cite (masked-V2V floor, prior Lightning-point instability: person r 0.64 / -0.05 /
   0.19 across runs).

## Decisions

- NOT registered in `validation_registry.py`: the ladder characterizes a parameter; it
  validates no route, package, or adapter, and no capabilities surface consumes it (matches the
  masked-V2V and fps/audio proof precedents, which are doc-cited but unregistered).
- The reused 0.7 row's generation-path equivalence across the intervening (verified
  behavior-preserving) refactors is assumed, not re-proven; stated explicitly in the README.
- Peak memory surfaced in the README (32.2-33.1 GiB ladder rows, 36.3 GiB Lightning control,
  whole-process darwin peak physical footprint).
