# Completed: Wan I2V Source Aspect Ratio Preservation

## Metadata
- Created: 2026-06-04
- Status: Completed
- Completed: 2026-06-04

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None

## Context

Wan image-to-video conditions generation on a source image. Users expect the output video to keep
the source image's composition proportions. If the source image is resized to an unrelated video
canvas, the first-frame conditioning is stretched before denoising and the whole video inherits the
wrong geometry.

## Current code reality

- `src/mflux/models/wan/variants/wan2_2_ti2v.py` validates requested `height` and `width` against
  Wan patch multiples before it knows whether image-to-video source dimensions should affect the
  output size.
- `_encode_first_frame_condition(...)` and `_encode_video_condition(...)` call
  `ImageUtil.scale_to_dimensions(...)`, which resizes directly to the selected output dimensions.
  If those dimensions have a different aspect ratio from the source image, the source is stretched.
- `src/mflux/models/wan/cli/wan_generate.py` forwards `--width`, `--height`, and `--image-path`
  into `generate_video(...)`; unified `mlxgen generate --image ...` routes to the same Wan CLI.
- Focused tests cover Wan spatial multiple rounding and I2V conditioning shapes, but not
  aspect-ratio preservation.

## Problem

Wan I2V currently lets users request a video canvas that does not match the input image ratio. The
source image is then resized non-uniformly, which can distort the first frame, reduce prompt
adherence, and create avoidable geometry artifacts.

## What we want to do

When `image_path` is provided, resolve the video output dimensions from the source image aspect
ratio and the selected model's spatial multiples before preparing latents or encoding the image
condition. The generated video should use the resolved source-ratio canvas instead of stretching the
source image into a mismatched requested canvas.

## Why

This is a direct user-facing quality issue. I2V inputs are visual references, and preserving their
proportions is a basic correctness expectation for both CLI users and Python callers.

## Requirements

- Apply to both TI2V-5B first-frame I2V and A14B I2V paths.
- Apply at the model API layer so Python callers and all CLI routes share the same behavior.
- Keep text-to-video unchanged.
- Choose output dimensions that match the source aspect ratio as closely as the selected Wan spatial
  multiples allow while staying in the user's requested scale class.
- Do not crop the image, add borders, or resize it into an unrelated aspect ratio.
- Record resolved output dimensions in `GeneratedVideo` metadata.
- Print a concise dimension-adjustment message when I2V output dimensions change.

## Suggested implementation

- Add a Wan helper that reads source image dimensions and resolves an I2V canvas from:
  - source aspect ratio;
  - requested `width * height` scale;
  - model-specific width/height patch multiples.
- Call that helper in `generate_video(...)` when `image_path` is present before latent creation.
- Reuse the resolved dimensions for first-frame conditioning, A14B video conditioning, decoded
  output metadata, and save validation.
- Add focused unit tests for common mismatched source/request cases and CLI forwarding behavior.

## Scope

- Wan model dimension resolution.
- Wan CLI documentation/help text where needed.
- Unit tests for the helper and generation path.
- Core documentation updates for I2V aspect behavior.
- Local proof outputs for TI2V-5B and A14B I2V when cached models and machine memory allow.

## Non-goals

- No outpainting, padding, canvas extension, or reframing support.
- No image cropping or letterboxing modes.
- No change to text-to-video requested dimensions.
- No broad video quality claim beyond the exact proof runs produced for this task.

## Dependencies and related tasks

- [0012 Wan2.2 A14B T2V/I2V support](../completed/0012_wan_a14b_t2v_i2v_support.md)
- [0016 Wan video integrity release gate](0016_wan_video_integrity_release_gate.md)
- [0020 Generation capability contract and route planning](../completed/0020_generation_capability_contract.md)

## Expected outcomes

- A mismatched I2V source/request pair no longer stretches the source image.
- TI2V-5B and A14B I2V use the same source-ratio resolution rule.
- Generated videos and metadata report the resolved source-ratio dimensions.
- End-user docs explain that Wan I2V auto-resolves the video canvas from the input image ratio and
  model spatial multiples.

## Validation

- Unit tests:
  - helper chooses source-ratio dimensions for 16:9 and portrait inputs;
  - text-to-video dimension behavior remains unchanged;
  - both I2V conditioning paths receive resolved dimensions;
  - CLI output metadata and progress are not regressed.
- Local proof, when model artifacts are present:
  - one TI2V-5B I2V video from a deliberately mismatched source/request pair;
  - one A14B I2V video from the same source/request pair or equivalent;
  - extracted first/last frames or video-health metadata showing the output dimensions match the
    resolved source-ratio canvas.

## Progress checklist
- [x] Review design with architecture and review lenses.
- [x] Implement source-ratio dimension resolver.
- [x] Add focused tests.
- [x] Produce local proof videos for TI2V-5B and A14B I2V when feasible.
- [x] Update core docs and LLM index.
- [x] Move this item to completed with evidence.

## Implementation notes

Prefer a single model-layer rule over CLI-only behavior. Do not introduce a border/crop mode as a
shortcut. If exact source aspect ratio is impossible under model spatial multiples at the requested
scale, choose the closest supported canvas and report the resolved dimensions clearly.

## Completion report

The implemented design resolves Wan image-to-video output dimensions before latent preparation or
image conditioning. For image-to-video calls, `width` and `height` are treated as a size target. The
model reads the input image dimensions, searches nearby model-supported spatial multiples, and
chooses the canvas closest to the source aspect ratio at the requested scale. Text-to-video keeps
the previous model-multiple normalization behavior.

Files changed for runtime behavior:

- `src/mflux/models/wan/variants/wan2_2_ti2v.py`
- `src/mflux/models/wan/cli/wan_generate.py`
- `src/mflux/utils/generated_video.py`
- `src/mflux/utils/video_util.py`

Core validation added:

- `tests/wan/test_wan_a14b_config.py`
- `tests/metadata/test_generated_video.py`

Automated checks:

- `uv run ruff check src/mflux/models/wan/variants/wan2_2_ti2v.py src/mflux/models/wan/cli/wan_generate.py src/mflux/utils/video_util.py src/mflux/utils/generated_video.py tests/wan/test_wan_a14b_config.py tests/metadata/test_generated_video.py`
- `uv run pytest tests/wan tests/metadata/test_generated_video.py tests/cli/test_mlx_gen_router.py -q`

Result: 160 passed, 7 skipped.

Local proof source:

- `docs/assets/i2v_takeoff_source.png` (`512x512`)

Local proof outputs:

- `validation_outputs/i2v_aspect_ratio_2026_06_04/ti2v5b_square_source_requested_432x240_seed7771.mp4`
- `validation_outputs/i2v_aspect_ratio_2026_06_04/ti2v5b_square_source_requested_432x240_seed7771.metadata.json`
- `validation_outputs/i2v_aspect_ratio_2026_06_04/ti2v5b_square_source_requested_432x240_seed7771.frame-strip.png`
- `validation_outputs/i2v_aspect_ratio_2026_06_04/a14b_i2v_square_source_requested_432x240_seed7772.mp4`
- `validation_outputs/i2v_aspect_ratio_2026_06_04/a14b_i2v_square_source_requested_432x240_seed7772.metadata.json`
- `validation_outputs/i2v_aspect_ratio_2026_06_04/a14b_i2v_square_source_requested_432x240_seed7772.frame-strip.png`

Both proof commands requested `432x240` video from a `512x512` source image. Both outputs resolved
to `320x320`, preserving the source image ratio without crop, border, or non-uniform resize.
Metadata records the requested dimensions, source image dimensions, and resolved output dimensions.

Residual behavior: if a source ratio cannot be represented exactly at the requested scale under the
selected Wan spatial multiple, MLX-Gen chooses the closest supported canvas and prints the resolved
dimensions before generation.
