# Proposed: Video LoRA support for T2V and I2V

## Metadata

- Created: 2026-06-07
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: May revise the generation capability contract if LoRA becomes task-specific
  capability metadata rather than model-family metadata only.

## Context

Planned item [0007](../planned/0007_lora_capability_matrix_and_strict_application.md) covers
strict LoRA resolution/application and current image-family LoRA support. Video LoRA is a separate
problem: T2V and I2V adapters must be applied to video transformers, must preserve temporal
behavior, and must be validated with MP4 outputs rather than still images.

The immediate candidate family is Wan2.2 because it is MLX-Gen's current T2V/I2V backend. Future
video families such as LTX are tracked separately in proposed items 0009 and 0010.

## Current code reality

- Wan model classes do not accept `lora_paths` in their public constructor path.
- There is no Wan LoRA mapping in `src/mflux/models/common/lora/mapping/` or
  `src/mflux/models/wan/`.
- Wan A14B uses separate high-noise and low-noise transformers; a LoRA policy must decide whether
  an adapter applies to one transformer, both transformers, or a model-specific subset.
- `GenerationCapability` does not yet expose task-specific LoRA support for T2V/I2V.
- Existing LoRA mappings are image-family mappings for FLUX, FLUX.2, Qwen, and Z-Image.
- Item 0007 is expected to add strict LoRA capability metadata, loader reports, and fail-closed
  unsupported-family handling. Video LoRA should reuse that contract rather than introducing a
  parallel set of CLI flags or metadata fields.

## Problem or opportunity

Users will expect LoRA to work across text-to-image, image-to-image, text-to-video, and
image-to-video. Treating LoRA as a generic option would be wrong for video because unsupported
adapters could be ignored, partially applied to the wrong transformer, or damage temporal
consistency while still producing a plausible MP4.

## Proposed direction

Keep video LoRA proposed until image LoRA strictness lands, then investigate Wan first:

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

3. Implement a Wan LoRA mapping only when a known adapter can be loaded and visually changes a
   small video output.

4. Implement in stages:
   - TI2V-5B first, because it has one transformer role and can validate both T2V and first-frame
     I2V behavior with less routing ambiguity.
   - A14B T2V next, only after the high-noise/low-noise adapter role is explicit.
   - A14B I2V last, because it combines dual denoisers with source-image identity preservation.

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
| Wan2.2 TI2V-5B | T2V/I2V | High | 3D/video transformer mapping plus temporal validation, but only one transformer path. |
| Wan2.2 A14B T2V | T2V | High | Two denoiser roles and boundary routing make adapter targeting more complex. |
| Wan2.2 A14B I2V | I2V | Very high | Same dual-transformer issue plus source-image identity and motion validation. |
| Future LTX or selected second video family | T2V/I2V | Unknown | Depends on selected backend and upstream adapter ecosystem. |
| SeedVR2 | Restoration/upscale | Not a near-term LoRA target | It is not a generative prompt adapter surface in MLX-Gen today. Use restoration controls instead. |

## Why it might matter

Video LoRAs would unlock style, character, product, motion, or domain adaptation in local video
workflows. That is strategically useful for AbstractVision, but it must be implemented after the
base video routes are quality-stable enough that adapter effects can be judged.

## Promotion criteria

Promote to `planned/` when:

- item 0007 has fail-closed LoRA capability metadata and strict loader behavior;
- at least one Wan-compatible public LoRA or local adapter is available for validation;
- item 0015 or later Wan parity work confirms the base T2V/I2V route is healthy enough for adapter
  comparisons;
- a small MP4 validation profile is selected, such as `480x240`, 41 frames, 10 to 15 steps.
- the selected adapter's target role is known and can be recorded in capability metadata and video
  metadata before generation starts.

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
- Do not implement LTX, HunyuanVideo, or another second video family inside this item; proposed
  items 0009 and 0010 remain the place to select or spike a second video backend.

## Guidance for future agents

Start with rejection tests and capability metadata. Then compare upstream adapter naming against
MLX-Gen Wan weights before touching model execution. A successful implementation needs both a
mapping proof and a visible MP4 proof.
