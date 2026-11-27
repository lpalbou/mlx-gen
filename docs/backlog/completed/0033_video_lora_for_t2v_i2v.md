# Completed: Video LoRA support for T2V and I2V

## Metadata

- Created: 2026-06-07
- Status: Completed
- Completed: 2026-06-11

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: May revise the generation capability contract if LoRA becomes task-specific
  capability metadata rather than model-family metadata only.

## Context

Completed item [0007](../completed/0007_lora_capability_matrix_and_strict_application.md) covers
strict LoRA resolution/application and current image-family LoRA support. Video LoRA is a separate
problem: T2V and I2V adapters must be applied to video transformers, must preserve temporal
behavior, and must be validated with MP4 outputs rather than still images.

The immediate candidate family is Wan2.2 because it is MLX-Gen's current T2V/I2V backend. Future
video families such as LTX are tracked separately in proposed items 0009 and 0010.

## Current code reality

- Wan model constructors now accept `lora_paths`, `lora_scales`, and `lora_target_roles` through
  the public `mlxgen generate` route.
- `src/mflux/models/wan/wan_initializer.py` now applies Wan LoRAs through the shared strict LoRA
  loader and stores adapter application reports in generated video metadata.
- `src/mflux/models/wan/weights/wan_lora_mapping.py` now maps:
  - Diffusers-style Wan adapter keys;
  - non-Diffusers `diffusion_model.blocks.*` Wan adapter keys;
  - Musubi/Kohya-style `lora_unet_blocks_*` Wan adapter keys.
- MLX-Gen now supports explicit role targeting:
  - `transformer` for TI2V-5B;
  - `high_noise_transformer` and `low_noise_transformer` for A14B.
- TI2V-5B I2V now mirrors Diffusers' T2V-to-I2V image-projection expansion behavior by
  synthesizing zero LoRA tensors for missing `add_k_proj` and `add_v_proj` families when a T2V
  adapter is loaded on an I2V-capable transformer.
- `GenerationCapability` now surfaces Wan LoRA support and target roles per video route.
- Focused unit and router tests cover:
  - Wan adapter-key conversion;
  - TI2V T2V compatibility;
  - prepared-package compatibility via `base_model` fallback;
  - generated-video metadata propagation;
  - CLI forwarding of `--lora-target-roles`.

## Completion status as of 2026-06-11

Current Wan q8 public rows are now validated route-by-route with model-backed A/B artifacts:

- `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` on `wan.text-video`
  - adapter: `AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets`
  - evidence: `docs/assets/validation/wan-lora-2026-06-11/ti2v_t2v_hstoric_ab_contact_sheet.jpg`
- `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` on `wan.first-frame`
  - adapter: `ostris/wan22_5b_i2v_crush_it_lora`
  - evidence: `docs/assets/validation/wan-lora-2026-06-11/ti2v_i2v_crushit_ab_contact_sheet.jpg`
- `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` on `wan.text-video`
  - adapter: `artificialguybr/FollowCam-Redmond-WAN2-T2V-14B`
  - evidence: `docs/assets/validation/wan-lora-2026-06-11/a14b_t2v_followcam_ab_contact_sheet.jpg`
- `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` on `wan.first-frame`
  - adapter: `ostris/wan22_i2v_14b_orbit_shot_lora`
  - evidence: `docs/assets/validation/wan-lora-2026-06-11/a14b_i2v_orbit_spaceship_ab_contact_sheet.jpg`

The combined summary sheet is:

- `docs/assets/validation/wan-lora-2026-06-11/wan_video_lora_route_matrix.jpg`

All four validated rows use explicit role assignment, matching base-model cards, and same-seed A/B
comparisons with only the adapter changed.

## Problem or opportunity

Users will expect LoRA to work across text-to-image, image-to-image, text-to-video, and
image-to-video. Treating LoRA as a generic option would be wrong for video because unsupported
adapters could be ignored, partially applied to the wrong transformer, or damage temporal
consistency while still producing a plausible MP4.

## Direction

Keep Wan video LoRA tied to item 0007's strict capability contract and continue promotion
route-by-route:

1. Add task-specific capability fields through item 0007:
   - `supports_lora`;
   - `lora_supported_modes`;
   - `lora_validation_status`;
   - optional `lora_target_roles` for multi-transformer models.

   For video, `lora_target_roles` must be explicit. Candidate role names are:
   - `transformer` for single-video-transformer families such as TI2V-5B;
   - `high_noise_transformer` and `low_noise_transformer` for Wan A14B;
   - `both_transformers` only when the adapter is intentionally applied to both A14B denoisers.

2. Audit upstream and community Wan LoRA conventions:
   - key naming for Wan T2V and I2V adapters;
   - whether adapters target the video transformer, text encoder, condition embedder, or multiple
     modules;
   - whether A14B adapters target both denoisers or a single denoiser role.

3. Keep the current implementation fail-closed:
   - no silent role duplication for A14B;
   - no unsupported video-family fallback;
   - no promotion from key matching alone.

4. Continue promotion in stages:
   - keep TI2V-5B T2V as the first validated Wan row;
   - promote TI2V-5B I2V only after a public adapter shows a clear source-conditioned A/B delta;
   - promote A14B T2V only after a paired or explicitly duplicated adapter demonstrates a clear
     high-noise/low-noise effect;
   - promote A14B I2V last, because it combines dual denoisers with source-image identity
     preservation.

5. Validate by task direction:
   - T2V: prompt-only video with and without adapter, same seed/settings, visible style or motion
     difference, stable health metadata.
   - I2V: source image plus adapter, same seed/settings, source identity preserved enough for the
     adapter use case.

6. Keep unsupported video families fail-closed. Do not accept `--lora-paths` for Wan, SeedVR2, or
   future video models until a mapping and proof exist.

## Difficulty estimate

| Family | Direction | Difficulty | Reason |
| --- | --- | --- | --- |
| Wan2.2 TI2V-5B T2V | T2V | Medium to high | Ordinary linear-module injection is feasible, but MLX-Gen still needs Wan-specific adapter conversion, capability surfacing, and MP4 validation. |
| Wan2.2 TI2V-5B I2V | I2V | High | Same transformer path as T2V, plus image-conditioning projection semantics and source-identity validation. |
| Wan2.2 A14B T2V | T2V | Medium to high | Two denoiser roles and boundary routing add routing complexity, but the public adapter ecosystem is stronger and already publishes paired high-noise/low-noise LoRAs. |
| Wan2.2 A14B I2V | I2V | Very high | Same dual-transformer issue plus source-image identity and motion validation. |
| Future LTX or selected second video family | T2V/I2V | Unknown | Depends on selected backend and upstream adapter ecosystem. |
| SeedVR2 | Restoration/upscale | Not a near-term LoRA target | It is not a generative prompt adapter surface in MLX-Gen today. Use restoration controls instead. |

## Public proof candidates

The next implementation pass should not start with a generic adapter hunt. Use named public proof
candidates and record the exact file set used.

### TI2V-5B text-to-video

- `AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets`
  - base model: `Wan-AI/Wan2.2-TI2V-5B`
  - current public signal: `235` monthly downloads on Hugging Face as of `2026-06-11`
  - use case: obvious visual style shift with trigger phrase `HST style HD film, early 1900s`
  - proof value: direct T2V base-model adapter with a visible style target
- `wouterverweirder/wan_2_2_5B_woven_fabric_02-lora`
  - base model: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
  - use case: simple trigger-token texture/style proof
  - proof value: direct Diffusers-style TI2V-5B adapter if the first candidate turns out too subtle

### TI2V-5B image-to-video

- `ostris/wan22_5b_i2v_crush_it_lora`
  - base model: `Wan-AI/Wan2.2-TI2V-5B`
  - current public signal: `18` likes on Hugging Face as of `2026-06-11`
  - file: `wan22_5b_i2v_crush_it_lora.safetensors`
  - use case: explicit I2V trigger phrase `crush it`
  - proof value: direct TI2V-5B I2V adapter with a simple, documented trigger

### T2V-A14B

- `lightx2v/Wan2.2-Lightning`
  - base models: `Wan-AI/Wan2.2-T2V-A14B`, `Wan-AI/Wan2.2-I2V-A14B`, `Wan-AI/Wan2.2-TI2V-5B`
  - current public signal: `618` likes on Hugging Face as of `2026-06-11`
  - T2V proof files:
    - `Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors`
    - `Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors`
  - proof value: best public dual-noise A14B LoRA candidate for matching MLX-Gen's two-transformer
    routing
- `wangkanai/wan22-fp8-t2v-loras`
  - files:
    - `loras/wan/wan22-camera-rotation-rank16-v2.safetensors`
    - `loras/wan/wan22-light-volumetric-t2v.safetensors`
    - `loras/wan/wan22-face-naturalizer-t2v.safetensors`
  - proof value: secondary route for obvious camera or lighting changes after the core A14B LoRA
    plumbing works

### I2V-A14B

- `lightx2v/Wan2.2-Lightning`
  - I2V proof files:
    - `Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors`
    - `Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors`
  - proof value: clean paired high-noise/low-noise files aligned with the A14B two-stage route
- `lightx2v/Wan2.2-Distill-Loras`
  - current public signal: about `555k` downloads and `255` likes on Hugging Face as of
    `2026-06-11`
  - files:
    - `wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors`
    - `wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors`
  - proof value: strongest public adoption signal and explicit online-loading/offline-merging
    guidance for paired A14B I2V LoRAs
- `morphic/Wan2.2-frames-to-video`
  - base model: `Wan-AI/Wan2.2-I2V-A14B`
  - file: `lora_interpolation_high_noise_final.safetensors`
  - proof value: specialized I2V transition/interpolation adapter once the base A14B I2V LoRA path
    is working

## Why it might matter

Video LoRAs would unlock style, character, product, motion, or domain adaptation in local video
workflows. That is strategically useful for AbstractVision, but it must be implemented after the
base video routes are quality-stable enough that adapter effects can be judged.

## Promotion criteria

Promote an exact Wan row from `mapped-unvalidated` to `validated` only when:

- the route has a public adapter with a compatible model card or directly inspected tensor keys;
- the selected role assignment is explicit and recorded in capability plus output metadata;
- the model-backed MP4 A/B uses the same source or prompt, seed, frames, steps, guidance, and
  output size with and without the adapter;
- the contact sheet or MP4 shows a clear effect that matches the adapter's intended use instead of
  a subtle or ambiguous drift.

## Validation ideas

- Unit tests for unsupported LoRA rejection on Wan before support lands.
- Mapping test using a tiny synthetic adapter fixture once the target key patterns are known.
- Model-backed A/B MP4 proof with same prompt, seed, dimensions, frames, steps, guidance, and
  negative prompt.
- Contact sheet and direct MP4 links for with-LoRA versus without-LoRA outputs.
- Metadata proving requested and resolved adapter paths, scales, target roles, applied target
  counts, unmatched key counts, model package identity, and video health.

## Non-goals

- Do not implement video LoRA inside item 0007; keep that item focused on strictness and current
  image-family support.
- Do not claim support from key-pattern matching alone. A video output proof is required.
- Do not silently apply an adapter to only one A14B transformer unless the capability metadata and
  documentation say exactly which role was targeted.
- Do not make video LoRA a fallback for unsupported image/edit LoRA or vice versa.
- Do not spend time on Bonsai LoRA here; that remains a separate low-priority architectural item.
- Do not implement LTX, HunyuanVideo, or another second video family inside this item; proposed
  items 0009 and 0010 remain the place to select or spike a second video backend.

## Completion notes

This item is complete for the current Wan2.2 surface served by MLX-Gen. Follow-up work, if needed,
belongs in separate items:

- second video-family LoRA support;
- broader q4/package proof coverage;
- richer validation-registry or `mlxgen validation` integration for LoRA-specific profiles.

## Implementation notes

Start with rejection tests and capability metadata. Then compare upstream adapter naming against
MLX-Gen Wan weights before touching model execution. A successful implementation needs both a
mapping proof and a visible MP4 proof. When choosing the first route, decide explicitly whether the
priority is smallest code change (`TI2V-5B T2V`) or strongest public proof candidate set
(`A14B T2V` / `A14B I2V` via LightX2V).
