# Proposed: HunyuanVideo-1.5 second-family spike

## Metadata

- Created: 2026-06-18
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: Likely needs a video-backend ADR if MLX-Gen selects HunyuanVideo as the second
  major video family after Wan.

## Context

Wan remains the primary MLX-Gen video family, but HunyuanVideo-1.5 is now the strongest concrete
second-family candidate that is not already preserved in the backlog as its own bounded item.
Public source weights and Diffusers support exist for text-to-video and image-to-video, and the
published step-distilled `480p` image-to-video checkpoints are a plausible Apple Silicon fit.

This proposal does not replace the broader second-family selection item. It preserves the strongest
current non-Wan candidate as a concrete spike rather than leaving it buried in a generic list.

## Current code reality

- MLX-Gen has Wan video routes, SeedVR2 image-only restoration/upscale, and no HunyuanVideo family.
- The local Diffusers checkout includes HunyuanVideo 1.5 pipelines.
- Existing proposed item 0009 owns second-family selection at a broad roadmap level.
- Existing proposed item 0010 owns a separate LTX-family spike.

## Problem or opportunity

If MLX-Gen eventually wants a second major local video family, HunyuanVideo-1.5 is a better
bounded candidate than leading directly with larger audio-video platforms. It offers a public
Diffusers path, both T2V and I2V, and a step-distilled `480p` line that is more realistic for
Apple Silicon than a very large first port.

## Proposed direction

Keep this as a concrete spike under the broader second-family discussion:

1. Audit the official HunyuanVideo-1.5 source weights, license, and component inventory.
2. Run the upstream Diffusers `480p` step-distilled I2V path first at bounded settings.
3. Compare its user value against:
   - the existing Wan A14B fast path,
   - the planned SeedVR2 video-restoration work, and
   - the LTX-Video-first spike.
4. Promote to `planned/` only if the low-cost local path is genuinely useful and the license is
   acceptable for MLX-Gen's intended publication and derivative-weight story.

## Why it might matter

This is the clearest missing "second video family" candidate with a practical low-cost inference
story. It is more realistic for local Apple Silicon exploration than treating `LTX-2.3` or other
large audio-video platforms as the next obvious step.

## Promotion criteria

- Proposed item 0009 still recommends a second-family comparison pass.
- The official HunyuanVideo-1.5 local path is runnable with one bounded deterministic clip.
- License review is acceptable for the intended MLX-Gen and AbstractFramework use.
- The family offers materially different value than Wan rather than just duplicating it.

## Validation ideas

- One upstream Diffusers I2V smoke on the `480p` step-distilled route.
- If practical, one matching T2V smoke.
- Record model size, expected VRAM/RAM footprint, wall time, fps, dimensions, and output health.

## Non-goals

- Do not start a HunyuanVideo MLX port from this proposal alone.
- Do not treat HunyuanVideo as chosen over LTX or Wan-family extensions without a comparison pass.
- Do not fold this into Wan VACE or SeedVR2; those are different product directions.

## Implementation notes

Start with upstream Diffusers and a license review. If the bounded smoke is not practical on the
target Apple Silicon machine, keep this item proposed rather than inflating the roadmap.

## Sources checked

- HunyuanVideo-1.5 model card: https://huggingface.co/tencent/HunyuanVideo-1.5
- Diffusers HunyuanVideo 1.5 docs: https://huggingface.co/docs/diffusers/api/pipelines/hunyuan_video15
- HunyuanVideo-1.5 step-distillation note: https://huggingface.co/tencent/HunyuanVideo-1.5/blob/main/assets/step_distillation_comparison.md
- HunyuanVideo-1.5 `480p` step-distilled transformer tree: https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_i2v_step_distilled
