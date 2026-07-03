# ADR 0006: Generative Video Editing Task Boundary

Status: Accepted.

## Context

MLX-Gen already has two distinct public video workflows:

- `mlxgen generate` for prompt-driven Wan text-to-video and first-frame image-to-video;
- `mlxgen upscale` for SeedVR2 image restoration and video restoration/upscale.

That split is meaningful. `generate` is the creative model-driven surface. `upscale` is the
promptless restoration surface with SeedVR2-specific chunking, audio-copy, and host-safety rules.

Upstream Wan now also exposes prompt-guided editing of an existing source video through plain
video-to-video and richer VACE conditioning. Without a durable boundary, future work could drift in
three bad directions:

- hide source-video editing behind the current image-centric planner and pretend `image-to-video`
  already covers it;
- overload `mlxgen upscale --video-path ...` even though `upscale` currently means restoration,
  not semantic rewrite;
- collapse plain video-to-video, masked edit, reference-guided edit, and control-conditioned video
  into one vague feature bucket with no stable public contract.

Current code is only partially ready for this decision. The save and metadata layer already knows
`task="video-to-video"` for SeedVR2 outputs, but unified generation planning only knows
`text-to-image`, `image-to-image`, `text-to-video`, and `image-to-video`, and Wan runtime loading
still maps all video generation to the current TI2V/I2V-shaped runtime.

## Decision

MLX-Gen will treat prompt-guided editing of an existing source video as a generative workflow owned
by `mlxgen generate`, not `mlxgen upscale`.

The public task boundary is:

- `text-to-video`: create a new video from text only;
- `image-to-video`: animate from one input image or first frame;
- `video-to-video`: prompt-guided transformation of an existing source video;
- `mlxgen upscale --video-path ...`: restoration or upscale only, with no prompt-guided semantic
  rewrite.

The first public generative video-edit implementation must be plain `video-to-video` only:

- one source video;
- one prompt;
- one output MP4;
- no public mask, reference-image, or control-video taxonomy yet.

Masks, reference images, control inputs, and conditioning scales are allowed later as capability
detail or follow-up surface once one bounded `video-to-video` route is proven locally. They are not
separate public task names at the first boundary.

The implementation boundary is also explicit:

- planner and capability ownership lives in `task_inference.py`;
- CLI argument normalization and routing live in `mlx_gen.py`;
- model-specific video-edit semantics live in a dedicated runtime/handler path;
- current Wan TI2V/I2V classes must not be silently overloaded to claim general video editing.

## Consequences

### Positive

- User-facing workflow meaning stays coherent: create, animate, restore, and edit remain distinct.
- SeedVR2 restoration semantics stay isolated from prompt-guided Wan editing semantics.
- Future Wan work can add a fifth public task without pretending the current image-count planner
  already models source-video editing safely.

### Negative

- MLX-Gen will need explicit planner, CLI, runtime, and documentation work before it can claim
  prompt-guided source-video editing.
- A later reader-first alias or command may still be worth adding if `video-to-video` proves too
  internal-sounding for naive users.

### Neutral

- Current code does not yet implement the new public task. Adoption work is tracked in backlog
  items rather than implied by this ADR alone.
- The saved-output `task="video-to-video"` metadata already used by SeedVR2 remains valid, but it
  does not by itself mean unified generation supports prompt-guided video editing.

## Enforcement

- Backlog items that add generative source-video editing must cite this ADR.
- Public docs and CLI help must keep `mlxgen upscale` described as restoration/upscale, not as a
  generic video-edit surface.
- Public docs and CLI help must not imply that `image-to-video` or `--image` already support
  source-video editing.
- Any future `video-to-video` implementation must require explicit source-video input and fail
  closed on unsupported request shapes.
- Model-specific video-edit implementations must use a dedicated handler/runtime boundary instead of
  silently branching inside the current Wan TI2V/I2V runtime.

## Validation

Compliance is validated by:

- focused CLI/help tests that keep current workflow wording truthful;
- planner and capability tests once `video-to-video` is added publicly;
- model-backed smoke proof for the first selected Wan video-edit route before docs claim support;
- backlog completion reports that record exact source clip, prompt, frames, steps, wall time, and
  peak memory evidence.

## Backlog links

- Related proposal: [0039 Wan VACE video editing and control](../backlog/proposed/0039_wan_vace_video_editing_and_control.md)
- Adoption and workflow hardening: [0072 Reader-first video workflow boundary and generative video-edit contract](../backlog/completed/0072_reader_first_video_workflow_boundary_and_generative_video_edit_contract.md)
- Reference proof tooling: [0073 Wan VACE reference validation harness and bounded source cases](../backlog/completed/0073_wan_vace_reference_validation_harness_and_bounded_source_cases.md)
- Completed implementation: [0074 Wan plain generative video-to-video route](../backlog/completed/0074_wan_plain_generative_video_to_video_route.md)
- Future follow-up: [0075 Wan VACE conditioning expansion after plain video-to-video](../backlog/proposed/0075_wan_vace_conditioning_expansion_after_plain_video_to_video.md)

## Related

- [ADR 0001: Runtime Smoke Validation For Model Routes](0001_runtime_smoke_validation_for_model_routes.md)
- [ADR 0002: No Silent Automatic Fallbacks](0002_no_silent_automatic_fallbacks.md)
- [ADR 0003: Runtime Truth Versus Consumer Convenience](0003_runtime_truth_vs_consumer_convenience.md)
- [docs/api.md](../api.md)
- [docs/getting-started.md](../getting-started.md)
- [docs/upscaling.md](../upscaling.md)
- [src/mflux/task_inference.py](../../src/mflux/task_inference.py)
- [src/mflux/cli/mlx_gen.py](../../src/mflux/cli/mlx_gen.py)
