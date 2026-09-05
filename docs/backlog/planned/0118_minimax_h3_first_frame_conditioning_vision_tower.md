# Planned: MiniMax-H3 first-frame conditioning through the Qwen3-VL vision tower

## Metadata

- Created: 2026-09-04
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0007](../../adr/0007_role_aware_reference_conditioning_and_factored_model_sources.md).
- ADR impact: none beyond the joint audio-video contract ADR owed by item 0117.

## Summary

Ship MiniMax-H3 `image-to-video` (the reference's FL2VA / I2VA task): one keyframe conditions both
the latent rows and the text sequence.

## Current code reality

- The layout already reserves keyframe condition rows (`keyframe_anchors`), and
  `build_row_timesteps` pins them at the 0.999 noise-augmentation level; the scheduler's
  `scale_noise` and the video VAE encoder (single frame, seed-42 posterior sample, fp16 rounding,
  latent normalization) are ported and verified.
- `MiniMaxH3.generate_video(image_path=...)` raises `NotImplementedError` because the reference also
  inserts `"<Picture i>: "` plus the keyframe's vision tokens into the Qwen3-VL sequence, and the
  vision tower (patch embedding, 2D rotary, window attention, deepstack merger injected at text
  layers 0-2) is not ported. `Qwen3VLTextModel.__call__` already accepts `deepstack_visual_embeds`.

## Scope

1. Port the Qwen3-VL vision encoder and the deepstack merger; parity vs transformers on a real
   image (tiny config exact, real weights at bf16 noise).
2. Processor: keyframe resize policy of the reference pipeline, `<Picture 1>: ` label, image token
   placement, 3-axis M-RoPE positions for image tokens.
3. Pipeline: condition latents (encode seed 42, noise-aug 0.999), condition rows first in the packed
   sequence, FL2VA prompt presentation, capability row `minimax-h3.first-frame`, CLI `--image-path`.
4. Model-backed proof: same-seed T2VA vs FL2VA contact sheets on one keyframe.

## Non-goals

Ref2VA (reference images / videos / audio, the `Ref2VA` Turbo adapters) and the Context-IR API.

## Validation

Tiny and real-weight parity for the vision tower; an FL2VA clip whose first frame matches the
keyframe (PSNR against the source) and whose motion follows the prompt.
