# Proposed: Larger Bernini models capacity and scope gate

## Metadata

- Created: 2026-08-10
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs:
  - [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
  - [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
  - [ADR 0006](../../adr/0006_generative_video_editing_task_boundary.md)
  - [ADR 0007](../../adr/0007_role_aware_reference_conditioning_and_factored_model_sources.md)
- ADR impact: Promotion would likely require an ADR amendment only if MLX-Gen exposes the full
  planner-backed Bernini stack rather than the current renderer-only route.

## Context

The public larger Bernini-family releases are permissively licensed and technically relevant, but
they are materially larger than the current `Bernini-R-1.3B-Diffusers` row and they expand scope
in two different ways:

- `ByteDance/Bernini-R-Diffusers` is the larger 14B renderer-only diffusers package.
- `ByteDance/Bernini-R` is the original separate-checkpoint 14B renderer layout.
- `ByteDance/Bernini-Diffusers` is the full Bernini stack, which adds the planner/MLLM path on
  top of the renderer.

As of Monday, August 10, 2026, the public Hugging Face repository sizes are:

| Repository | Public role | Reported size |
| --- | --- | ---: |
| `ByteDance/Bernini-R-1.3B-Diffusers` | 1.3B renderer-only diffusers package | 28.93 GB public repo (`26.94 GiB`) |
| `ByteDance/Bernini-R-Diffusers` | 14B renderer-only diffusers package | 126.20 GB public repo (`117.53 GiB`) |
| `ByteDance/Bernini-R` | 14B renderer high/low checkpoints repo | 143.54 GB public repo (`133.67 GiB`) |
| `ByteDance/Bernini-Diffusers` | full Bernini planner + renderer stack | 192.38 GB public repo (`179.17 GiB`) |

The current mlx-gen Bernini work has not yet proved strict parity even for the official public
1.3B example matrix. Pulling larger weights before that matrix is explicit would blur two
different questions:

1. is the current 1.3B MLX port still missing an implementation detail, or
2. is a larger/publicly different Bernini scope actually required?

This item records the larger-model follow-up without letting it hide unresolved 1.3B parity work.

## Problem or opportunity

The larger public Bernini rows may matter for two legitimate reasons:

- the 14B renderer may recover quality or task robustness that the smaller 1.3B renderer misses;
- the full Bernini planner stack may be necessary for upstream examples that depend on planner
  semantics rather than renderer-only prompting.

But they also have real costs:

- storage jumps from 28.93 GB to 126.20-192.38 GB before local caching overhead;
- the full Bernini stack is not just "a larger checkpoint" but a different product surface;
- proof, download, and capacity work become much more expensive on Apple Silicon;
- a larger-model experiment can easily mask a still-unfixed 1.3B parity defect.

## Proposed direction

Treat larger Bernini models as a gated follow-up:

1. Finish the strict public `Bernini-R-1.3B` official example matrix first.
2. Use that matrix to classify each miss as one of:
   - missing mlx-gen capability;
   - unresolved MLX/reference parity defect;
   - likely 1.3B model-capacity limitation;
   - likely planner-dependent behavior.
3. Promote the 14B renderer before the full planner stack unless the blocked official case is
   clearly planner-shaped.
4. Keep the full Bernini planner stack separate from renderer-only support in backlog, docs, and
   public task claims.
5. Do not prefetch the 126-192 GB repos on a host whose disk budget is not explicitly approved for
   that download.

## Recommended promotion order

1. `ByteDance/Bernini-R-Diffusers` (14B renderer-only)
2. `ByteDance/Bernini-R` only if the raw high/low-checkpoint layout is needed for a parity check
   the diffusers package cannot answer
3. `ByteDance/Bernini-Diffusers` only if the official blocked examples or planner semantics justify
   the full-stack scope increase

## Estimated work after promotion

- 14B renderer source audit and bounded parity spike: about 1-2 engineering weeks.
- 14B renderer exact-input parity plus model-backed official-example checks: about 3-6 engineering
  weeks.
- Full Bernini planner-stack spike after a renderer baseline exists: about 4-8 additional
  engineering weeks.
- Public-quality proof, documentation, and release hardening for a promoted larger row: likely
  another 2-4 engineering weeks.

These ranges assume the current Wan-family scaffolding stays reusable and that no new Apple-Silicon
kernel gap blocks the larger rows.

## Promotion criteria

- The strict 1.3B official-example matrix is written down and current status is no longer vague.
- At least one failed official 1.3B case has evidence that points to model scope or planner scope
  rather than an ordinary MLX defect.
- Disk and runtime-memory budgets for the larger row are explicitly accepted before download.
- The larger-row scope is stated truthfully in docs and validation: 14B renderer-only and full
  Bernini are not treated as interchangeable.

## Non-goals

- Do not treat this item as proof that the 14B or full Bernini rows should be downloaded now.
- Do not fold planner-backed Bernini into the current renderer-only Bernini doc without separate
  evidence.
- Do not use larger models to retroactively claim the 1.3B row is validated.

## Implementation notes

If the 1.3B official-example matrix is still incomplete, update that matrix first and leave this
item proposed. If the 1.3B matrix is complete and the blocked rows are clearly model-scope or
planner-scope issues, promote the next larger row explicitly and preserve the exact download size,
runtime recipe, and proof boundary in the implementation item.

## Sources checked

- `ByteDance/Bernini-R-1.3B-Diffusers`: https://huggingface.co/api/models/ByteDance/Bernini-R-1.3B-Diffusers
- `ByteDance/Bernini-R-Diffusers`: https://huggingface.co/api/models/ByteDance/Bernini-R-Diffusers
- `ByteDance/Bernini-R`: https://huggingface.co/api/models/ByteDance/Bernini-R
- `ByteDance/Bernini-Diffusers`: https://huggingface.co/api/models/ByteDance/Bernini-Diffusers
- Official Bernini repository: https://github.com/bytedance/Bernini
