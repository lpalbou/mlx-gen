# Wan V2V Motion-Fidelity Ladder (2026-07-05)

Measures how much of the source clip's subject motion (gesture timing) survives plain
prompt-guided video-to-video as `--video-strength` rises, on `Wan2.2-T2V-A14B` (q8 package,
20-step CFG-on recipe), plus one paired control run at the Lightning 4-step point testing
whether prompt gesture language changes the outcome. Backs the "Motion Fidelity Versus
Strength" section in `docs/wan-video.md`.

## Result table

Person-region gesture-window metrics vs the source (higher = closer to source motion; see
Methodology). Warm-start sigma computed from the real scheduler code (`measure_motion.py`
mirrors `_video_to_video_timesteps` exactly).

| Strength | Warm-start sigma | Source signal (amplitude) | Effective steps | High-noise steps | Edit landed (man->woman) | Gesture ratio | Gesture timing r | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null row (re-encode, no model) | - | 100% | - | - | - | 1.00 | 1.00 | codec floor is clean |
| 0.5 | 0.750 | 25.0% | 10 | 0 (skipped) | yes | 0.92 | **0.86** | tracks source motion |
| 0.6 | 0.818 | 18.2% | 12 | 0 (skipped) | yes | 0.90 | **0.90** | tracks source motion |
| 0.7 (reused prior run) | 0.875 | 12.5% | 14 | 1 | yes | 0.80 | **0.73** | mostly tracks |
| 0.8 (default) | 0.923 | 7.7% | 16 | 3 | yes | 0.63 | **0.20** | re-synthesized (r below the n=22 significance floor of ~0.42) |
| 0.9 (math only, no run) | 0.964 | 3.6% | 18 | 5 | - | - | - | - |

Reading: the hand-raise gesture survives with correct timing at strengths 0.5-0.6, mostly at
0.7, and is replaced by plausible-but-different motion at the 0.8 default. The 0.86 vs 0.90
values at 0.5/0.6 are statistically indistinguishable (single seed; treat 0.5-0.6 as one band,
not an ordering). The subject swap (man to woman, driven by the prompt plus a strong negative
prompt) landed at every strength including 0.5 - see `face_crops_frame12.png` (left to right:
source, 0.5, 0.6, 0.7, 0.8; the judgment criterion, written before measuring:
"female-presenting face and hair at frame 12"; an independent multi-frame recheck confirmed a
consistent female face at every sampled frame for 0.5 and 0.6).

Two variables change together on A14B (state this when citing the table): higher strength both
raises the warm-start noise level AND engages the high-noise expert (boundary 875): the
0.5/0.6 rows run entirely on the low-noise expert with `--guidance` inert (the CLI prints the
boundary-skip warning), while 0.7/0.8 engage both experts. The motion result is the shipped
end-to-end behavior of the strength knob, not a pure noise-level ablation.

## Control run: prompt gesture language at the Lightning point

Paired with the red-tie proof run (`validation_outputs/fps_audio_proof_2026_07_05/`): same
seed 4242, same 30 fps source, same Lightning 4-step settings (strength 0.75 -> sigma 0.938),
ONE change - the prompt adds "and gesturing naturally with his hands".

| Run | Prompt motion language | Gesture ratio | Gesture timing r (n=14 floor ~0.53) |
| --- | --- | --- | --- |
| red-tie baseline | "stands ... speaking" | 0.49 | -0.16 |
| control | "gesturing naturally with his hands" | 0.91 | 0.45 |

With identical noise, the gesture-window motion energy nearly doubled (0.49 -> 0.91, stable
across window and scale choices) - visible in `control_contact_sheet.png` (hands lift and
gesture in the second half). Both r values sit below the n=14 significance floor (~0.53) and
the baseline's r is window-sensitive, so the correlation numbers carry no timing claim in
either direction; the conclusion rests on the energy ratio and the visual check. Honest
framing: in this paired run, prompt motion language decided whether gesturing appeared at all,
but prompts recover the class of motion, not the instance or timing. One seed; suggestive, not
conclusive.

## Fixed factors (ladder rows)

Source `validation_outputs/v2v_conference_480p_2026_07_04/man_conference_480p_lightning_source.mp4`
(25 frames, 16 fps, 480x832; right-hand raise ~t=0.53-1.0s, both hands ~t=0.93-1.5s). Matching
source/requested fps means NO resampling (decode path bit-identical with the pre-resampling
release; pinned by `test_video_util_resample_skips_matching_fps`). Model
`AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit`, seed 8602, steps 20, guidance 4,
guidance-2 3, flow-shift 3, solver unipc, `--low-ram --metadata`, and the woman-swap prompt +
custom negative prompt copied verbatim from the reused 0.7 row's sidecar (the negative prompt
is part of the edit contract; all rows share it exactly).

Reused 0.7 row: `validation_outputs/v2v_conference_480p_2026_07_04/woman_conference_480p_v2v.mp4`,
generated 2026-07-04 on 0.18.24. Its sidecar predates `metadata_schema_version` and
`source_video_resampled`; its `effective_steps: 14` proves strength 0.7 applied (this matters:
0.18.24 had a router bug that silently dropped `--video-strength`, and 14 = floor(20 x 0.7)
clears it). Decode-path equivalence with today's tree is test-pinned (matching fps =
passthrough); generation-path equivalence across the intervening refactors is ASSUMED, not
re-proven - the refactors were adversarially verified behavior-preserving, but no cross-check
rerun of this row exists.

## Reproduce

Ladder row (swap `--video-strength` 0.5 / 0.6 / 0.8 and the output name):

```bash
uv run mlxgen generate --model AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit \
  --video-path validation_outputs/v2v_conference_480p_2026_07_04/man_conference_480p_lightning_source.mp4 \
  --prompt "A realistic wide shot of a woman giving a talk on a conference stage. Keep the exact same stage, wooden podium with microphones, large bright presentation screen with the same logo, conference banner, warm stage lighting, camera framing, and natural speaking gestures. She is fully visible from head to feet, has shoulder-length dark hair tied back, and wears the exact same dark blue suit with a white shirt." \
  --negative-prompt "man, male face, beard, stubble, mustache, short male haircut, different clothes, dress, skirt, changed background, moved podium, missing screen, different logo, close-up, cropped body, cut off feet, skewed face, distorted face, melted hands, extra limbs, text overlay, watermark, subtitles, blur, low quality, flicker, scene cut" \
  --width 480 --height 832 --frames 25 --fps 16 --steps 20 --guidance 4 --guidance-2 3 \
  --flow-shift 3 --solver unipc --video-strength 0.5 --seed 8602 --low-ram --metadata \
  --output validation_outputs/motion_fidelity_ladder_2026_07_05/ladder_s05.mp4
```

Control run (paired with the red-tie baseline; the baseline used the identical command with
"stands at a conference speaking to the audience," i.e. without the gesture clause):

```bash
uv run mlxgen generate --model AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit \
  --video-path validation_outputs/fps_audio_proof_2026_07_05/conference_30fps_with_audio.mp4 \
  --prompt "A man in a dark blue suit stands at a conference speaking to the audience and gesturing naturally with his hands, wearing a bright red necktie, photorealistic, stage lighting" \
  --negative-prompt "cartoon, illustration, low quality, blurry, distorted face" \
  --width 480 --height 832 --frames 17 --fps 16 --steps 4 --video-strength 0.75 \
  --guidance 1 --guidance-2 1 --flow-shift 5 --solver unipc \
  --lora-paths "lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors" "lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors" \
  --lora-target-roles high_noise_transformer low_noise_transformer \
  --seed 4242 --low-ram --metadata \
  --output validation_outputs/motion_fidelity_ladder_2026_07_05/control_gesture_prompt.mp4
```

The control's 30 fps source clip is itself derived from the ladder source (fps=30 re-encode +
sine audio; exact ffmpeg command in the fps/audio proof README). The ladder source clip was
generated with the seed-8601 Lightning T2V command documented in
`docs/assets/validation/lightning-v2v-2026-07-04/README.md`. The person mask ships in-repo at
`docs/assets/examples/conference-masked-v2v/person_mask.png` (byte-identical to the
`validation_outputs` copy used in the commands here).

Null row: `uv run python validation_outputs/motion_fidelity_ladder_2026_07_05/make_null_row.py`
(reads the source through the runtime decode path and re-writes it through the production
CRF-18 writer; measures the codec floor).

Measurement:

```bash
uv run python validation_outputs/motion_fidelity_ladder_2026_07_05/measure_motion.py \
  --source validation_outputs/v2v_conference_480p_2026_07_04/man_conference_480p_lightning_source.mp4 \
  --outputs <null> <s05> <s06> <reused 0.7> <s08> \
  --person-mask validation_outputs/v2v_conference_480p_2026_07_04/person_mask.png \
  --gesture-start-pair 8 --json metrics_ladder.json
# control (17 frames): --gesture-start-pair 7, outputs control + red-tie baseline
```

## Methodology

- Motion energy: mean |frame[i+1] - frame[i]| per consecutive pair, grayscale at 120 px
  analysis width, over the person region (`person_mask.png`, a feathered rectangle covering
  the subject head-to-feet incl. raised-hand positions, 33% of frame; drawn for the earlier
  masked-V2V proof) and the background complement. A dilated-mask variant (8 px) is reported
  in `metrics_ladder.json` as an edge-sensitivity check; conclusions are unchanged.
- Gesture window: pairs 8-23 (t = 0.50-1.50 s) for 25-frame rows; pairs 7-15 for the 17-frame
  control. "Gesture ratio" = output/source mean motion energy in the window (amplitude,
  confounded upward by re-synthesis flicker - context only). "Gesture timing r" = Pearson
  correlation of the motion-energy series after excluding 2 settle pairs (the headline
  number; n=22 -> |r| below ~0.42 is statistically indistinguishable from zero, n=14 -> ~0.53).
- The null row (source re-encoded through the same writer) measures the pipeline floor:
  ratio 1.00, r 1.00 - the metric itself adds no noise at this analysis scale.
- Sanity anchors, preserved in `metrics_anchors.json` (this bundle's own measurements of the
  prior 2026-07-04 artifacts): reused 0.7 row r = 0.73, masked-V2V run person r = 0.10 /
  background r = 0.97 (mask preserved at codec floor), and the prior Lightning-point runs
  showing run-to-run instability at strength 0.75/4-step: person r = 0.64 (v1.1 seed 9001),
  -0.05 (v2.0 seed 8602), 0.19 (v1.1 on-grid seed 8602) - all at or below the significance
  floor except one, with sign flips across runs.
- Wall times (M-series, q8, `--low-ram`): 0.5 -> 2335 s, 0.6 -> 2598 s, 0.8 -> 1533 s
  generation (system load varied; the 0.8 run had the machine to itself), control -> 217 s.
  Peak physical memory (whole process, `darwin_peak_physical_footprint_bytes` from the
  sidecars): 32.2-33.1 GiB for the 20-step ladder rows, 36.3 GiB for the Lightning control.

## Files

Included in the committed mirror (`docs/assets/validation/motion-ladder-2026-07-05/`):

- `README.md` (this file), `metrics_ladder.json`, `metrics_control.json`, `metrics_anchors.json`
- `measure_motion.py`, `make_null_row.py` (note: the sigma table requires an `mflux` checkout
  with MLX; the motion metrics only need numpy, PIL, and ffmpeg)
- `person_mask.png` (byte-identical copy of the tracked
  `docs/assets/examples/conference-masked-v2v/person_mask.png`)
- `ladder_s06.mp4` + sidecar (the recommended motion-preserving row, r 0.90),
  `ladder_s08.mp4` + sidecar (the default-strength row, r 0.20),
  `control_gesture_prompt.mp4` + sidecar (the paired prompt control; its baseline is the
  red-tie run published at `docs/assets/examples/conference-fps-audio/red_tie_fps_audio.mp4`)
- `ladder_s05_contact_sheet.png`, `ladder_s06_contact_sheet.png`, `ladder_s08_contact_sheet.png`,
  `control_contact_sheet.png`, `face_crops_frame12.png`

Preserved locally on the validation host (`validation_outputs/motion_fidelity_ladder_2026_07_05/`,
git-ignored):

- `ladder_s05.mp4` + sidecar (contact sheet is mirrored; metrics in `metrics_ladder.json`)
- `null_row_source_reencode.mp4` (regenerable with `make_null_row.py`)
