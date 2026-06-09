# Planned: Wan2.2 TI2V-5B math and behavior parity

## Metadata

- Created: 2026-06-09
- Status: In progress
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None unless this work changes public Wan routing, fallback, default prompt,
  negative-prompt, guidance, scheduler, or prepared-package compatibility semantics.

## Context

MLX-Gen supports Wan2.2 TI2V-5B text-to-video and first-frame image-to-video, plus A14B T2V and
I2V. Recent local comparison clips on an Apple M5 Max showed useful practical guidance but also a
TI2V-5B quality question:

- `Wan2.2 T2V-A14B`, `480x240`, 25 steps, 101 frames, 20 fps, about 30 minutes, produced the
  preferred starship-takeoff result.
- `Wan2.2 TI2V-5B`, `832x480`, 25 steps, 101 frames, 20 fps, about 12 minutes, was faster but
  visually weaker.
- `Wan2.2 TI2V-5B`, `1280x704`, 25 steps, 101 frames, 20 fps, about 35 minutes, improved over the
  832x480 row but still did not match the A14B practical result.

The public docs now present these clips as practical guidance. They should not be treated as proof
that TI2V-5B is inherently weak until MLX-Gen behavior is checked against the official Wan
implementation and the Hugging Face Diffusers/Transformers ports.

## Current code reality

- `src/mflux/models/wan/variants/wan2_2_ti2v.py` owns the TI2V-5B runtime route, T2V/I2V task
  selection, prompt and negative-prompt resolution, dimension normalization, scheduler setup, and
  denoising loop.
- `src/mflux/models/wan/model/wan_transformer/` contains the MLX Wan transformer port.
- `src/mflux/models/wan/model/wan_vae/wan_2_2_vae.py` contains the MLX Wan VAE port.
- `src/mflux/models/wan/scheduler/wan_unipc_multistep_scheduler.py` and
  `src/mflux/models/wan/latent_creator/wan_timestep_policy.py` contain Wan scheduler and expanded
  timestep behavior.
- `src/mflux/models/wan/weights/` contains Wan weight mapping and quantization policy.
- Local Hugging Face reference checkouts are available at:
  - `/Users/albou/projects/gh/diffusers/`
  - `/Users/albou/projects/gh/transformers/`
- No obvious official Wan checkout was found under `/Users/albou/projects/gh/` by a shallow
  `*wan*` directory search during item creation. The implementing agent should locate, clone, or
  otherwise identify the official Wan2.2 source before claiming official-code parity.
- Official Wan2.2 source was checked from `https://github.com/Wan-Video/Wan2.2` at commit
  `42bf4cfaa384bc21833865abc2f9e6c0e67233dc` under `/private/tmp/Wan2.2-official`.
- Practical comparison MP4 assets are tracked under
  `docs/assets/examples/wan-video-comparison/` and documented in [Wan Video](../../wan-video.md).

## Problem

The TI2V-5B route may be producing weaker practical videos because of model limitations,
resolution/step settings, prompt fit, quantization/package behavior, or a math/conditioning mismatch
in the MLX port. Without a parity ladder, users cannot know whether to prefer A14B because it is
actually better for the prompt class or because TI2V-5B is being mishandled.

## What we want to do

Audit and validate MLX-Gen Wan2.2 TI2V-5B against official Wan and Diffusers/Transformers, working
backwards from pixels to model internals until the behavior gap is either fixed or clearly
attributed.

## Why

TI2V-5B is the smaller Wan route and should be useful for faster local T2V/I2V iteration. If the
MLX port has a subtle scheduler, timestep, RoPE, VAE, prompt-embedding, or first-frame conditioning
mismatch, users will waste long video runs and may draw the wrong conclusion about the model.

## Requirements

- Start with source/BF16 behavior before q8 or q4. Do not debug quantization before source parity.
- Use explicit prompts, negative prompts, seeds, dimensions, frames, steps, fps, scheduler settings,
  guidance, and source images.
- Compare T2V and first-frame I2V separately.
- Include the starship-takeoff prompt from `docs/wan-video.md` plus at least one simpler
  motion-control prompt.
- Use same initial latents/noise where feasible. Do not rely only on matching integer seeds across
  PyTorch and MLX.
- Match official Wan and Diffusers semantics for timestep expansion, flow shift, scheduler update,
  prompt embeddings, negative prompt, text encoder dtype, VAE scale, latent packing/unpacking,
  first-frame conditioning, and output decode.
- Keep validation batches small first. Do not run long `101`-frame or native-resolution tests until
  smaller parity checks narrow the suspect area or the user approves the compute cost.
- Preserve tensor dumps, MP4s, frame strips, metrics, and command logs under a dedicated
  `validation_outputs/wan/ti2v5b_parity/` folder.
- Keep public docs free of investigation notes. Record conclusions in backlog, validation reports,
  and release notes only after evidence is available.

## Suggested implementation

1. Locate the official Wan2.2 code used by the model authors and record the exact commit or source
   path. Use `/Users/albou/projects/gh/diffusers/` and `/Users/albou/projects/gh/transformers/` as
   the Hugging Face references.
2. Build a parity ladder:
   - text tokenizer and UMT5 prompt embeddings;
   - negative prompt embeddings;
   - scheduler timesteps, sigmas, expanded timestep masks, and update rule;
   - VAE encode/decode on fixed images and fixed latents;
   - transformer forward on exported latents, context, masks, and timesteps;
   - one denoise step with fixed latents;
   - short full denoise loop with fixed initial latent/noise.
3. Run a small T2V comparison at a practical supported size such as `448x256` or `832x480`, then
   one native-shape comparison only after the smaller run is credible.
4. Run a small I2V comparison using the same source image and aspect-ratio resolution handling.
5. If a mismatch is found, patch the narrowest code surface and add focused tests or parity
   fixtures.
6. Re-run the practical starship profile only after the math path is either fixed or confirmed
   equivalent.

## Scope

- Wan2.2 TI2V-5B source route behavior for text-to-video and first-frame image-to-video.
- MLX math, tensor shape/order, scheduler, VAE, text conditioning, first-frame conditioning, and
  output decode parity.
- Focused docs/backlog/model-card updates after evidence is available.

## Non-goals

- Do not make q4/q8 TI2V-5B quality claims before source parity is resolved.
- Do not publish or republish models from this item alone.
- Do not change A14B behavior unless the same bug is proven to affect A14B.
- Do not add prompt rewriting or hidden fallback behavior.
- Do not present investigation notes in end-user docs.

## Dependencies and related tasks

- [0002 Wan quantization and motion parity](0002_wan_quantization_motion_parity.md)
- [0015 Wan prompt adherence parity validation](0015_wan_prompt_adherence_parity_validation.md)
- [0016 Wan video integrity release gate](../completed/0016_wan_video_integrity_release_gate.md)
- [0021 Wan I2V source aspect-ratio preservation](../completed/0021_wan_i2v_source_aspect_ratio.md)
- `src/mflux/models/wan/variants/wan2_2_ti2v.py`
- `src/mflux/models/wan/model/wan_transformer/`
- `src/mflux/models/wan/model/wan_vae/`
- `src/mflux/models/wan/scheduler/wan_unipc_multistep_scheduler.py`
- `/Users/albou/projects/gh/diffusers/`
- `/Users/albou/projects/gh/transformers/`

## Expected outcomes

- A clear result: MLX-Gen TI2V-5B matches official/Diffusers behavior for the tested profiles, or
  a precise mismatch is fixed or documented.
- If fixed, focused tests or fixtures cover the failing math/conditioning surface.
- If no mismatch is found, docs can continue to recommend A14B for the tested starship prompt
  class while describing TI2V-5B as a faster route with model-dependent quality.
- Quantization work can resume from a known-good source baseline.

## 2026-06-09 Findings

- The local Hugging Face Diffusers snapshot
  `Wan-AI/Wan2.2-TI2V-5B-Diffusers` uses `UniPCMultistepScheduler` with `use_flow_sigmas=true`,
  `flow_shift=5.0`, `prediction_type=flow_prediction`, and the expected TI2V-5B transformer/VAE
  shapes.
- MLX-Gen follows the Diffusers-compatible scheduler path for this Diffusers snapshot. The official
  Wan private scheduler starts from a slightly different raw sigma grid, but the published
  Diffusers snapshot explicitly carries the Diffusers scheduler config, so no scheduler-grid patch
  was made.
- Existing opt-in local parity fixtures passed after this audit:
  prompt embeddings, full transformer forward, VAE encode/decode, scheduler replay, and a tiny CFG
  denoise loop match Diffusers-generated TI2V-5B fixtures.
- A practical missing control was found: MLX-Gen did not expose the Wan flow-shift override.
  TI2V-5B defaults to `5.0`, which is right for native 720p-class runs. Wan/Diffusers references
  use `3.0` for 480p-class runs. MLX-Gen now exposes `--flow-shift` in the Wan CLI and
  `flow_shift=` in the Python generation API, records it in MP4 metadata, and keeps the model
  default when the override is omitted.
- A source-model CLI smoke completed with `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, `448x256`, 9 frames,
  4 steps, `--flow-shift 3`, seed `3510`. The output and metadata are under
  `validation_outputs/wan/ti2v5b_parity_2026_06_09/`; metadata records `flow_shift: 3.0`, 9 decoded
  frames, `448x256`, and non-black video-health metrics. This is a route/metadata check, not a
  quality proof.
- A targeted source-model starship check completed at `832x480`, 41 frames, 15 steps, 20 fps,
  `--flow-shift 3`, seed `4244`, in 159 seconds. The output, metadata, and frame strip are under
  `validation_outputs/wan/ti2v5b_parity_2026_06_09/`. The clip is coherent and video-health passes,
  but it is a short prompt-adherence check, not a replacement for a longer 101-frame comparison.
- The recorded `832x480` starship comparison should be treated as a previous practical clip. New
  480p-class TI2V-5B visual checks should be rerun with `--flow-shift 3` before drawing a final
  model-quality conclusion at that size.
- Native `1280x704` TI2V-5B uses the correct default `flow_shift=5.0`. If it remains weaker than
  A14B for the starship prompt after the same prompt/negative/seed/settings review, that points more
  toward model/prompt behavior than an MLX math mismatch.

## Validation

- Tensor parity reports with shapes, dtypes, tolerances, and artifact paths.
- Short T2V MP4 comparison with extracted frame strips and command logs.
- Short I2V MP4 comparison with the source image, resolved output dimensions, frame strips, and
  command logs.
- `uv run pytest` for any focused tests added during implementation.
- Optional longer starship-profile rerun only after smaller parity checks pass or identify the
  issue.

## Progress checklist

- [x] Locate and record official Wan2.2 reference source and version.
- [x] Compare MLX prompt and negative-prompt embeddings against references.
- [x] Compare scheduler timesteps, expanded timesteps, flow shift, and update rule.
- [x] Compare VAE encode/decode on fixed images and fixed latents.
- [x] Compare transformer forward on exported reference tensors.
- [x] Compare one denoise step with fixed initial latents.
- [ ] Run short same-settings T2V reference-vs-MLX comparison.
- [ ] Run short same-settings I2V reference-vs-MLX comparison.
- [x] Patch and test any confirmed mismatch.
- [x] Update backlog/docs with evidence-based conclusions.

## Guidance for the implementing agent

Treat this as a porting/parity task, not prompt tuning. Keep edits small and evidence-driven.
Export tensors from the reference implementation when possible, then replay them through MLX.
Prefer exact math fixes over new abstractions unless a shared helper already exists in the codebase.
