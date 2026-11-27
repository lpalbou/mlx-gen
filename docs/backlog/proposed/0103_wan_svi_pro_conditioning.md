# Wan A14B i2v SVI 2.0 Pro conditioning (`--svi-anchor-image`)

## Metadata

- Created: 2026-07-27
- Status: Implemented (shipped as EXPERIMENTAL; cycle-1 probe results
  recorded below — one scenario chain, one seed set, small resolution;
  broader recipes unverified)
- Effort: M (code) + GPU probe gate (quality evidence)

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
  (real-checkpoint proof before the capability ships),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md) (SVI mode and
  the SVI LoRA pack gate each other loudly in BOTH directions; every
  non-A14B-i2v route rejects the parameters; the strict key-match contract
  refuses partial LoRA application instead of warn-and-skip).
- ADR impact: None.

## Context (doctrine provenance)

The 2026-07-27 storyboard-consistency REDO doctrine of a host application
(its backlog `proposed/storyboard_consistency_redo_2026_07/`)
ranks SVI 2.0 Pro (Stable Video Infinity, ICLR'26 Oral; arxiv 2510.09212;
`svi_wan22` branch, Dec 2025) as mechanism 4: the engine-side
identity/anti-drift layer for chains of 3+ scenes, behind a loud-fail probe
gate (doctrine item B1/B2). The doctrine's suite must pass WITHOUT SVI;
SVI raises the ceiling on long chains.

## The recipe as verified from primary sources

Verified against `diffsynth/pipelines/wan_video_svi_pro.py` at
vita-epfl/Stable-Video-Infinity@7dac0f9 (`WanVideoUnit_ImageEmbedderVAE`),
`docs/svi/svi_2.0_pro.md`, `inference_svi_2.0_pro.py`, and the
kijai/ComfyUI-WanVideoWrapper issue #1718 author thread:

- Conditioning `y = concat([mask(4ch), anchor_latent, motion_latent?,
  zero_latents])`: the anchor image (user-given first frame) is VAE-encoded
  ALONE to 1 temporal latent at position 0; the previous clip's final
  denoised latent tensor contributes its LAST `num_motion_latent` (default 1)
  temporal entries at positions 1..count; the rest is genuinely zero-valued
  LATENT padding — explicitly NOT the stock Wan i2v convention of encoding a
  zero-padded pixel video (those padding latents are non-zero). The mask is
  the STANDARD first-frame i2v mask; the motion latent stays mask=0 and is
  read positionally by the fine-tuned model.
- First clip: `y = [anchor, zeros]` (anchor = the input image; i2v-like).
  Continuation decodes start with 1 anchor-restoration frame + 4 x count
  re-rendered predecessor frames: the authors stitch with the first FIVE
  frames removed (count=1). Motion is handed over in LATENT space only
  ("we will never use the decoded last frame from the previous clip").
- Error-Recycling LoRA pair (high/low noise experts), rank 128, alpha-free
  PEFT state dicts in `lora_A.default.weight` format; loaded at alpha=1.
  Official weights: HF `vita-video-gen/svi-model`,
  `version-2.0/SVI_Wan2.2-I2V-A14B_{high,low}_noise_lora_v2.0_pro.safetensors`,
  1,226,928,552 bytes each, sha256
  `299b33006863194d077a43bc0abf16fc52963457657d867763f2b61fd6a9bd52` (high) /
  `e8fcce153df0f5a2b49a17c2f82bd795002f0e3b35f25d6922da9cfe072b9c0b` (low);
  800 tensors per file = 40 blocks x 10 projections (self/cross q,k,v,o +
  ffn.0/ffn.2) x A/B.
- lightx2v coexistence (author ablations in kijai #1718): high-noise
  lightx2v scale 0.5-0.6 + SVI 1.0; low-noise lightx2v 1.0 + SVI 1.0.
  lightx2v at 1.0 on high noise = weaker dynamics/text-following and
  anchor snap-back.
- Operational hygiene: unique seed per clip (identical seeds accumulate
  artifacts — author statement + community replication); continue segments
  <= 65 frames (>65 showed per-window color shifts); SVI LoRAs on a stock
  workflow produce garbage (pinned upstream warning) and vice versa.

## The surface as shipped

- CLI: `--svi-anchor-image`, `--svi-motion-latent`,
  `--svi-motion-latent-count` (default 1), `--svi-lora-high`,
  `--svi-lora-low`; forwarded by the `mlxgen generate` router with the
  anchor filling the image slot for plan resolution (never emitted as
  `--image-path`). Every SVI run exports `<output>.svi_latent.safetensors`
  (full final fp32 latent + provenance metadata) for the next clip.
- Python: `generate_video(svi_anchor_image_path=..., svi_motion_latent_path=...,
  svi_motion_latent_count=..., svi_motion_latent_export_path=...)`;
  constructor `svi_lora_high_path`/`svi_lora_low_path` (fixed scale 1.0).
- Strict LoRA contract (doctrine B1): `unmatched_key_count == 0` per SVI
  file or the load aborts (the generic loader warns-and-skips; that is the
  silent-failure trap this contract closes). Achieved via a general
  collision-safe PEFT adapter-infix normalization
  (`lora_A.<adapter>.weight` -> `lora_A.weight` when the file uses exactly
  one adapter name). Verified on the official pack: 800/800 matched, 400
  layers fused per expert. The SVI high LoRA re-fuses on per-item
  high-noise expert reloads (0089 e4) under the same contract.
- Rejection matrix (ADR 0002): SVI conflicts with `image_path`,
  `last_image_path`, `context_image_paths`, `video_path`; requires the pack
  (and the pack requires SVI mode — fused SVI weights corrupt stock runs);
  TI2V-5B (`expand_timesteps`) and VACE reject; count requires the motion
  latent; the chain must keep one canvas (latent geometry validated at
  load); >65-frame continuations warn (trained-length advisory).
- Metadata: `svi_anchor_image_path`, `svi_motion_latent_path`/`_count`,
  `svi_motion_latent_export`, `svi_assembly_trim_frames`
  (= `1 + 4 x count` for continuations, 0 for first clips), per-expert
  `svi_lora_high`/`svi_lora_low` key-match reports; replay through
  `--config-from-metadata`; failure manifests carry the request fields.
- Capabilities: additive `supports_svi` on `wan.first-frame`
  (`schema_version` 6 -> 7); true only on dual-expert non-expand-timesteps
  non-VACE rows (today: Wan2.2-I2V-A14B).

## Cycle-1 probe (2026-07-27, this machine, q8 A14B)

Doctrine scenario 4 class (occlusion/state-change reveal), 3-scene chain,
480x240, 49 frames @ 16 fps, seeds 101/202/303 (unique per clip, shared
across arms), one anchor still (z-image-turbo drone over geyser field):

- Arm A — SVI + Lightning 4-step (lightx2v high 0.6 / low 1.0, SVI 1.0,
  guidance 1.0/1.0, flow_shift 5.0).
- Arm B — SVI alone, 20 steps, guidance 4.0/4.0 (reference-style CFG with
  the default Wan negative prompt), flow_shift 5.0.
- Arm C — no-SVI Lightning baseline (lightx2v 1.0/1.0), single-frame
  handoff (scene N+1 seeded from scene N's last frame).

Results (full scoreboard + stills in the host application's
`untracked/svi_probe_cycle1/PROBE_REPORT.md`): **PASS — SVI is
usable on MLX.** The SVI+Lightning arm held THE SAME drone across all 3
chained clips (anchor re-injection visibly re-normalized scale and
composition per clip) while the no-SVI baseline reinvented the subject
after its whiteout seam (a different black, gimbal-less drone — the
doctrine's convicted out-of-sight class, reproduced and fixed side by
side). Honest bounds: the anchor dampens prompted dynamics (the SVI arms
under-executed "engulfs"/"rises"); after a FULL occlusion the SVI-alone
CFG arm did not re-emerge the subject (no reinvention either — the clip
stayed a steam-scape), so the doctrine's window-QA + reference
re-grounding escalation remains load-bearing for occlusion seams; one
scenario, one seed set, 480x240. Recommended dispatch: SVI 1.0/1.0 +
lightx2v HIGH 0.6 / LOW 1.0, steps 4, guidance 1.0/1.0, flow_shift 5.0,
unique seed per clip, trim 5 frames per continuation at assembly.

## Cycle-2 tuning probe (2026-07-28, host application `untracked/svi_probe_cycle2/`)

- lightx2v HIGH 0.5 vs 0.6: visually equivalent on a prompted-action
  continuation (same seed) — 0.5 does NOT recover the anchor-dampened
  action; the SVI anchor's subject-visibility pull dominates the scale for
  occlusion-class actions (plain i2v with the same phrasing executes the
  occlusion). Defaults stay HIGH 0.6 / LOW 1.0. Occlusion beats should be
  dispatched as NON-SVI scenes.
- `--svi-motion-latent-count 2`: no visible benefit over count 1 on a
  clean seam; doubles the assembly trim (9 vs 5). Default stays 1.
- 69-frame continuation: the >65 advisory prints; no tail color shift
  measured (post-f48 luma drift -0.3/255). Advisory-not-block confirmed as
  the right shape.
- fp32-VAE decode check (doctrine B3): CLOSED by measurement — bf16 vs
  fp32 decode of identical latents differ by mean 0.000011 on [-1,1]
  (~0.001/255); the D-1-05 tone ramp lives in the latents (denoise-level).
  bf16 decode remains the default; no engine change.

## Follow-ups / open items

- Cycle-3: SVI app-wiring (dispatch integration per the cycle-1
  contract; settings now stable: 0.6/1.0, count 1, trim 5).
- Mixed chains (SVI continuation from a non-SVI predecessor latent) are
  mechanically possible (same latent space) but unprobed; the export is
  SVI-mode-only until a probe justifies widening.
- Occlusion re-emergence via the escalation path (anchor + re-grounded
  start still) remains the doctrine's answer for scenario-4-class beats;
  probed and confirmed necessary in cycles 1-2.
