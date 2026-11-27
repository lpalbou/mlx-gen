# Planned: Wan quantization and motion parity

## Metadata

- Created: 2026-05-27
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: Needs new ADR only if Wan cancellation or reporting beyond the existing
  `ProgressEvent` callback contract becomes a stable AbstractVision provider API contract. No ADR
  is required for the current narrow prepare, validation, and Diffusers-parity fixes.

## Context

Wan2.2 TI2V is the first MLX-Gen text-to-video and first-frame image-to-video backend. It is also
large enough that small routing bugs, bad quantization policy, or subtle scheduler drift can waste
hours of local compute. A user reported that `mlxgen prepare` could not create a q8 Wan folder and
that three 121-frame, 50-step videos generated at 1280x704 felt too static.

## Current code reality

- An earlier requested checkout path was stale; the active working repository for this backlog
  pass was the MLX-Gen repository root.
- `src/mflux/models/common/cli/save.py` previously passed `lora_paths` and `lora_scales` to every
  prepare backend. `Wan2_2_TI2V.__init__()` does not accept those kwargs, so Wan q8/q4 prepare
  failed before model loading or quantization.
- A local fix now inspects the selected model class signature and passes LoRA kwargs only to
  backends that declare `lora_paths`; `tests/cli/test_prepare_save.py` covers both Wan and Qwen.
- `WanWeightDefinition.quantization_predicate()` now keeps Wan `condition_embedder.*` and
  `proj_out` linears BF16 for q8 while quantizing the bulky transformer block linears. This was
  added after full q8 T2V-A14B validation collapsed to near-black/static output.
- The local Wan source snapshot declares `WanPipeline`, `expand_timesteps: true`,
  `UniPCMultistepScheduler`, `flow_shift: 5.0`, `prediction_type: flow_prediction`, 30 transformer
  layers, and transformer `in_channels: 48`.
- MLX-Gen already has parity tests for Wan expanded timestep masks, scheduler timesteps/sigmas,
  scheduler replay, UMT5 prompt embeddings, transformer output, VAE encode/decode, and a tiny CFG
  denoise loop against Diffusers fixtures.
- On 2026-05-27, `mlxgen prepare --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --quantize 8 --path
  models/wan2.2-ti2v-5b-diffusers-8bit` succeeded. The prepared folder is about 17 GiB:
  transformer 5.0 GiB, VAE 1.3 GiB, text encoder 11 GiB, tokenizer 16 MiB.
- Wan generated model cards now use `pipeline_tag: text-to-video`, Apache 2.0 frontmatter,
  video-generation tags, q8 Wan transformer/VAE wording, and a `video.mp4` usage example.
- A q8 prepared-folder smoke generation succeeded from that folder at 128x128, 5 frames, 2 steps,
  8 fps, guidance 1.0. This validates reload/wiring only, not quality.
- A same-settings 25-frame smoke comparison now exists for source BF16 versus the prepared q8
  folder at 704x384, 25 frames, 12 steps, 24 fps, guidance 5.0, seed 321. The source run took
  95.48 seconds; the prepared q8 run took 217.4 seconds. The q8 output stayed visually close in
  the contact sheet, but this run shows no speed win. Runtime peak memory was not captured by the
  Wan CLI/metadata path.
- Three 1280x704 reference videos were inspected at 121 frames, 24 fps, and about
  5.04 seconds each. Analysis artifacts live in `validation_outputs/wan/user_video_analysis/`.
- On 2026-06-02, `Wan-AI/Wan2.2-T2V-A14B-Diffusers` mixed q8 prepare succeeded at
  `models/wan2.2-t2v-a14b-diffusers-8bit`. The prepared folder is about 39.5 GiB: `transformer`
  14 GiB, `transformer_2` 14 GiB, text encoder 11 GiB, VAE 242 MiB, tokenizer 16 MiB. The source
  snapshot is about 117.5 GiB when following symlinks.
- A controlled T2V-A14B 384x224, 17-frame, 12-step, guidance 4/guidance-2 3, fps 8, seed 4242
  validation showed full q8 is not publishable: sampled frame MAE against BF16 was 99.95 and
  sampled temporal change collapsed to 2.03 versus 16.61 for BF16. The mixed q8 policy restored
  visual quality: prepared mixed q8 MAE against BF16 was 11.57 and sampled temporal change was
  17.39. The final contact sheet and report are under `validation_outputs/wan/a14b_q8_t2v/`.
- On 2026-06-03, a full-size T2V-A14B mixed q8/BF16 run at 1280x720, 81 frames, 40 steps,
  guidance 4/guidance-2 3, and fps 16 completed after about 13h15m but saved an all-black MP4 after
  non-finite decoded values reached `VideoUtil`. Version 0.18.9 shipped fail-closed tensor and
  video-health guards, but the full-size q8 path should not get broader quality claims until
  [item 0016](0016_wan_video_integrity_release_gate.md) captures release artifacts and exact
  settings pass.
- Wan BF16 and mixed q8/BF16 A14B packages have been published with cards that describe the measured
  storage and memory tradeoff. This item now tracks residual quantization quality and policy work,
  especially I2V-A14B validation, q4 decisions, and motion/prompt checks not covered by items 0015
  and 0016.
- On 2026-06-04, a clean TI2V-5B source/BF16/q8 validation passed at 1280x704, 17 frames,
  20 steps, guidance 5, fps 24, seed 321, with `--negative-prompt ""`. The current source run and
  prepared BF16 run were byte-identical after MP4 decode. The mixed q8/BF16 run stayed visually in
  the same family with mean frame MAE 1.66 versus source/BF16. The MP4s, contact sheet, and metrics
  are tracked under `docs/assets/quantization/wan-ti2v5b-clean/` and documented in
  `docs/quantization.md`.
- The TI2V-5B source snapshot is not already all 16-bit on disk: safetensors headers show FP32
  transformer and VAE files plus a BF16 UMT5 text encoder. MLX-Gen loads Wan transformer/VAE
  weights at BF16 runtime precision, so the prepared BF16 package reduces storage/download size
  from 31.9 GiB to 21.2 GiB but does not reduce runtime memory in the clean profile.
- TI2V-5B clean-profile memory measured on 2026-06-04: source 102.7 GiB physical peak, 13.7 GiB
  max RSS, 58.5 GiB MLX peak, 216.2s; prepared BF16 102.6 GiB physical peak, 14.5 GiB max RSS,
  58.5 GiB MLX peak, 261.6s; mixed q8/BF16 103.7 GiB physical peak, 13.8 GiB max RSS, 54.2 GiB
  MLX peak, 243.4s. q8 reduces storage, Wan MLX model bytes from 10.6 to 6.3 GiB, active MLX bytes
  after generation from 10.3 to 6.1 GiB, and MLX peak. This was a 1280x704 normal-cache profile,
  not a low-RAM profile and not comparable to the A14B 384px-class low-RAM rows as model-size
  evidence. It did not reduce full-process physical peak in this profile because transient
  prompt/denoise/decode/save allocations dominated the sampled high-water mark.
- The same prompt with the official default Wan negative prompt produced a visibly noisy textured
  background. `Wan2_2_TI2V.generate_video()` now distinguishes an omitted negative prompt from an
  explicit empty string: omitted uses the model default, while `negative_prompt=""` disables the
  default. This is covered by `test_wan_explicit_empty_negative_prompt_disables_default`.
- Wan spatial size normalization now rounds up to the required VAE/patch multiple instead of
  rounding down, so TI2V-5B `432x240` becomes `448x256` rather than `416x224`. This is covered by
  `test_wan_spatial_size_rounds_up_to_patch_multiple`.

## Problem

Wan q8 prepare had a concrete CLI/backend argument bug. Beyond that, MLX-Gen needs evidence that
its Wan implementation is not accidentally suppressing motion, that q8 keeps acceptable quality,
and that q4 should either remain unsupported, use full q4, or get a model-specific mixed q4/q8
policy.

## What we want to do

Make Wan video support publication-ready for AbstractFramework model repos and standalone users:
fix prepare, validate q8 reload and quality, compare motion behavior against Diffusers, decide q4
policy from evidence, and keep docs/backlog clear about what is verified versus experimental.

## Why

Wan runs are expensive. A 121-frame, 50-step generation at the recommended 1280x704 settings takes
about two hours on the user's M5 Max. Users need reliable progress, correct prepared folders, and
clear guidance before uploading or depending on quantized Wan checkpoints.

## Requirements

- `mlxgen prepare` must not pass unsupported LoRA kwargs to Wan.
- Prepared q8 Wan folders must reload and generate MP4 output.
- q8 quality must be compared with BF16/source at realistic settings before public model-card
  claims go beyond "loads and runs".
- Wan timing and memory reporting must remain profile-specific: the TI2V-5B clean profile now has
  full-process physical, RSS, MLX peak, and timing measurements, while older smoke comparisons and
  future full-duration A14B claims still need the same package-level measurement discipline.
- q4 must not be published as good unless side-by-side tests show acceptable quality. If full q4
  degrades motion or detail, define a mixed q4/q8 policy and document the retained q8/BF16 layers.
- Static-feeling outputs must be evaluated with frame contact sheets, frame-difference or optical
  flow metrics, and at least one Diffusers comparison using the same prompt/settings where
  feasible.
- Video model cards should identify `pipeline_tag: text-to-video` or image-to-video coverage, not
  inherit image-only wording.

## Suggested implementation

1. Keep the `save.py` signature-based LoRA forwarding fix and tests.
2. Add a Wan q8 prepared-folder quality panel at a documented lower-cost setting and, if compute
   allows, one 121-frame quality run.
3. Run the same prompt through upstream Diffusers from the local snapshot for a short comparison,
   or record why the hardware/runtime cost blocks it.
4. Prepare q4 only after enough disk is available, then compare BF16/q8/q4 with the same prompt,
   seed, dimensions, frames, steps, guidance, and fps.
5. If q4 fails quality, inspect layer sensitivity and implement a Wan-specific quantization
   predicate rather than reusing Qwen/ERNIE rules blindly.
6. Add or update generated Hugging Face cards for Wan q8/q4 once the quantization policy is known.

## Scope

- Wan2.2 TI2V 5B prepare, q8/q4 quantization validation, motion-quality checks, and docs/model-card
  readiness.
- Text-to-video and first-frame image-to-video only.

## Non-goals

- Do not publish new q4 repos or expand q8 claims beyond the exact settings that have passed
  quality and video-health validation.
- Do not port Wan A14B, Wan VACE, Wan Animate, or video-to-video in this item.
- Do not delete large local model folders automatically to recover disk; ask first.
- Do not treat q8 smoke tests at 128x128/5 frames as quality evidence.

## Dependencies and related tasks

- [Model integration roadmap](0001_model_integration_roadmap.md)
- `src/mflux/models/common/cli/save.py`
- `src/mflux/models/wan/variants/wan2_2_ti2v.py`
- `src/mflux/models/wan/scheduler/wan_unipc_multistep_scheduler.py`
- `src/mflux/models/wan/weights/wan_weight_definition.py`
- `tests/cli/test_prepare_save.py`
- `tests/wan/test_wan_local_parity.py`
- `tests/wan/test_wan_scheduler_and_timesteps.py`
- Local Diffusers checkout reference: `diffusers/src/diffusers/pipelines/wan/`
- Local Transformers checkout reference: `transformers/src/transformers/models/umt5/`

## Expected outcomes

- Wan q8 prepare and reload are no longer blocked by the LoRA argument bug.
- The record shows which Wan quantization levels are verified, which are only smoke-tested,
  and which remain unsafe to publish.
- Static or low-motion Wan outputs are evaluated against objective frame metrics and Diffusers
  parity instead of by impression alone.
- AbstractVision can depend on a clear Wan support state and the shared step-based progress
  reporting behavior.

## Validation

- `uv run ruff check src/mflux/models/common/cli/save.py tests/cli/test_prepare_save.py`
- `uv run pytest tests/cli/test_prepare_save.py -q`
- `uv run pytest tests/wan/test_wan_scheduler_and_timesteps.py tests/wan/test_wan_progress.py -q`
- `MFLUX_RUN_LOCAL_WAN_PARITY=1 uv run pytest tests/wan/test_wan_local_parity.py -q`
- `uv run mlxgen prepare --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --quantize 8 --path
  models/wan2.2-ti2v-5b-diffusers-8bit`
- `uv run mlxgen generate --model models/wan2.2-ti2v-5b-diffusers-8bit --task text-to-video
  --width 128 --height 128 --frames 5 --steps 2 --guidance 1 --fps 8 ...`
- `uv run mlxgen prepare --model Wan-AI/Wan2.2-T2V-A14B-Diffusers --path
  models/wan2.2-t2v-a14b-diffusers-8bit --quantize 8`
- `uv run mlxgen generate --model models/wan2.2-t2v-a14b-diffusers-8bit --task text-to-video
  --width 384 --height 224 --frames 17 --steps 12 --guidance 4 --guidance-2 3 --fps 8 ...`
- Contact sheets and motion metrics for user videos under
  `validation_outputs/wan/user_video_analysis/`.

## Progress checklist

- [x] Identify the q8 prepare crash as unsupported LoRA kwargs passed into Wan.
- [x] Patch prepare backend instantiation to pass LoRA kwargs only to compatible model classes.
- [x] Add focused tests for Wan no-LoRA kwargs and Qwen LoRA preservation.
- [x] Verify Wan scheduler/timestep/progress tests.
- [x] Verify full local Wan parity fixtures against Diffusers.
- [x] Prepare a q8 Wan folder locally and measure size.
- [x] Fix generated Wan q8 model-card metadata and video usage.
- [x] Smoke-generate an MP4 from the prepared q8 folder.
- [x] Run a same-settings BF16/source versus prepared q8 short comparison.
- [x] Run a clean TI2V-5B source/BF16/q8 comparison at 1280x704, 17 frames, 20 steps,
      guidance 5, fps 24, seed 321, with `--negative-prompt ""`, and publish the MP4/contact-sheet
      evidence in `docs/assets/quantization/wan-ti2v5b-clean/`.
- [x] Generate contact sheets and motion metrics for the three reference videos.
- [x] Validate T2V-A14B mixed q8 against BF16/source with a contact sheet and frame metrics.
- [x] Add a documented TI2V-5B runtime memory profile with full-process Darwin physical footprint,
      max RSS, MLX allocator peak, logical model bytes, generation time, MP4 health, and decoded
      frame metrics.
- [ ] Generalize Wan runtime peak-memory reporting into reusable validation metadata or a documented
      release harness for future full-size A14B and q4/q8 comparisons. The harness should preserve
      phase-labeled samples and distinguish storage, Wan MLX model bytes, MLX active bytes,
      full-process physical footprint, max RSS, MLX allocator peak, cache limit, denoiser-release
      flags, block/slice cache-clearing flags, tensor-health interval, and save-health settings.
- [ ] Run a normalized Wan memory matrix before making any cross-family memory claim: same
      resolution, frames, steps, fps, seed, guidance, negative-prompt policy, cache limit,
      `--low-ram` behavior, tensor-health interval, save-health policy, and sampler.
- [ ] Add q8 quality comparison at full-duration/full-step settings where practical. TI2V-5B now
      has clean 1280x704 validation at 17 frames and 20 steps; full-duration A14B mixed q8/BF16
      still needs exact-setting validation under item 0016 before broader claims.
- [ ] Decide and validate Wan q4 or mixed q4/q8 policy.
- [x] Update generated Wan q8 model cards and public docs for the mixed q8/BF16 policy.
- [ ] Validate I2V-A14B mixed q8 with source-image conditioning before publishing an I2V repo.

## Implementation notes

Re-check free disk before preparing more Wan folders. The prepared q8 folder consumed about 17 GiB
and only about 29 GiB remained afterward. Preserve user videos and generated model folders unless
the user explicitly authorizes cleanup.
