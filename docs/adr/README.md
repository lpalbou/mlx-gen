# Architecture Decision Records

This directory contains durable engineering decisions for MLX-Gen. ADRs are policy, not task
tracking; backlog items record execution state and evidence.

| ADR | Status | Decision |
| --- | --- | --- |
| [ADR 0001](0001_runtime_smoke_validation_for_model_routes.md) | Accepted | New model routes need model-backed smoke proof before being described as working. |
| [ADR 0002](0002_no_silent_automatic_fallbacks.md) | Accepted | Ambiguous or unsupported requests fail closed; automatic fallback is explicit opt-in only. |
| [ADR 0003](0003_runtime_truth_vs_consumer_convenience.md) | Accepted | MLX-Gen owns exact runtime truth; higher-level integrations own convenience and curation. |
| [ADR 0004](0004_seedvr2_video_host_safety_and_proof_boundaries.md) | Accepted | SeedVR2 video defaults to a conservative host-safe CLI profile, and heuristic metrics alone do not justify public family-quality rankings. |
| [ADR 0005](0005_seedvr2_video_quality_proof_requires_five_second_reader_first_clips.md) | Accepted | SeedVR2 public video quality claims require at least five contiguous seconds of reader-first proof plus direct visual review artifacts. |
| [ADR 0006](0006_generative_video_editing_task_boundary.md) | Accepted | Prompt-guided source-video editing belongs to `mlxgen generate`; the first public milestone is plain `video-to-video`, not VACE conditioning. |
