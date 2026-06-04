# Proposed: Wan I2V prompt motion validation

## Metadata

- Created: 2026-05-27
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: None unless MLX-Gen changes its public Wan conditioning API or adds new motion
  controls.

## Context

AbstractFramework generated an image-to-video clip through MLX-Gen Wan with a source sword image
and the prompt `the sword becomes magical and emits particles`. The integration passed the prompt,
source image path, seed, 50 steps, 5.0 guidance, 10 requested frames, 10 fps, and 1280x704 output.
The returned MP4 was 9 frames and visually near-static.

## Current code reality

- `Wan2_2_TI2V.generate_video()` accepts `prompt`, `image_path`, `num_inference_steps`, `width`,
  `height`, `num_frames`, `fps`, `guidance`, and `negative_prompt`.
- Wan frame counts are normalized to `4n + 1`, so a 10-frame request becomes 9 frames.
- First-frame conditioning is implemented by resizing the input image to the target dimensions
  before encoding it.
- There is no public `strength`, motion amount, first-frame conditioning weight, or motion-control
  parameter for the Wan I2V path today.
- Existing planned item 0002 already tracks broader Wan motion parity and Diffusers comparison
  work.
- Existing planned item 0015 now owns same-settings Diffusers-vs-MLX prompt-adherence validation
  for both T2V and I2V.

## Problem or opportunity

When Wan I2V returns a near-static clip even with a motion prompt, callers cannot tell whether the
result is expected short-clip behavior, over-strong first-frame conditioning, scheduler drift, a
prompt-conditioning issue, or a missing motion-control parameter.

## Proposed direction

Add a focused Wan I2V validation pass that compares MLX-Gen against Diffusers for the same source
image, prompt, seed, dimensions, frame count, steps, fps, and guidance. Include at least one
recommended-settings run and one lower-cost smoke run. If MLX-Gen is materially less animated than
Diffusers, investigate prompt embedding, timestep/scheduler, first-frame mask/latent condition, and
VAE encode/decode parity before exposing new API controls.

## Why it might matter

AbstractVision and AbstractFlow can only pass the prompt and source image through. If MLX-Gen Wan
I2V under-responds to motion prompts, users will see correct wiring but poor creative control.

## Promotion criteria

- Repeated I2V outputs remain near-static at recommended settings, not only at very short smoke
  settings.
- A same-settings Diffusers run shows meaningfully stronger prompt-following or motion.
- A candidate fix or new conditioning control is identified in the Wan implementation.

## Validation ideas

- Generate contact sheets and adjacent-frame difference/optical-flow metrics for MLX-Gen and
  Diffusers.
- Test 121 frames, 50 steps, 24 fps at 1280x704 or 704x1280.
- Repeat the reported sword prompt with a source image already matching the target aspect ratio.
- Include a camera-motion prompt and an object-motion prompt so first-frame locking can be
  separated from prompt weakness.

## Non-goals

- This proposal does not authorize wrapper-side prompt rewriting.
- This proposal does not require AbstractVision to invent a `strength` parameter before MLX-Gen
  supports one.
- This proposal does not block short smoke tests; it only prevents treating them as quality
  evidence.

## Guidance for future agents

Start from planned items 0002 and 0015. Promote this proposal only if 0015 shows an I2V-specific
prompt or motion weakness that remains unexplained after the broader Wan parity work.
