# Recurrent: Backlog release-state hygiene

## Metadata

- Created: 2026-06-04
- Status: Recurrent
- Completed: N/A
- Cadence: After each release, after large validation runs, and before any planned/proposed split
  that changes priorities.

## ADR status

- Governing ADRs: None
- ADR impact: None. Escalate to ADR only if this process creates a durable engineering policy
  beyond backlog maintenance.

## Purpose

Keep backlog state aligned with shipped code, public docs, validation evidence, and release claims.
The backlog is useful only when planned work names what remains rather than what already shipped.

## Trigger

- A release is tagged or published.
- A model is uploaded or a model card is materially rewritten.
- A long validation run changes a model-family support claim.
- A planned item is mostly completed but still has residual follow-up work.
- A proposed item becomes a concrete correctness or ADR-alignment issue.

## Checklist

- Recount planned, proposed, completed, deprecated, and recurrent item files.
- Check global four-digit ID uniqueness across all backlog lifecycle folders.
- Move completed work to `completed/` with a completion report instead of leaving shipped work as
  planned.
- Narrow partially completed planned items so their remaining scope is explicit.
- Promote proposed items only when evidence shows urgency, blocking risk, or a clear mandate.
- Fix stale lifecycle links after moves.
- Update `overview.md` counts, ledgers, next recommended work, and planning notes in the same pass.
- Confirm planned work cites relevant ADRs or explicitly records why none applies.

## Latest run

- 2026-06-04: Post-0.18.9 hygiene promoted LoRA strictness to planned work, narrowed Wan A14B
  boundary-memory scope to the remaining full-size retry, updated Wan q8 integrity/performance
  wording for the shipped 0.18.9 guardrail baseline, and normalized completed-item reports.
- 2026-06-04: Pre-0.18.10 hygiene checked the current taskless/capability planner against code and
  docs, added release-readiness evidence to completed item 0020, narrowed planned item 0019 to the
  remaining first-class FLUX.1 Fill outpaint/reframe adapter, and confirmed item 0016 still gates
  full-size Wan A14B q8 readiness claims.
- 2026-06-07: Post-0.18.13 priority hygiene checked planned/proposed/completed counts, confirmed
  global backlog IDs remain unique, deprioritized FIBO Edit parity work without deleting its
  history, promoted first-class outpaint/reframe and LoRA strictness in the recommendation order,
  added proposed item 0032 for SeedVR2 video restoration/upscaling, and added proposed item 0033
  for future T2V/I2V LoRA support.
- 2026-06-07: Re-scoped item 0019 after the FLUX.1 priority concern. Generative reframe starts
  with FLUX.2 and then Qwen Image Edit 2511; canvas outpaint is validated first for FLUX.2 and
  Qwen Image Edit 2511;
  Z-Image and ERNIE remain validation-gated candidates; FIBO Edit stays linked to deferred items
  0024/0027; native fill/inpaint outpaint remains separate until a fill/mask backend is proven.
- 2026-06-08: Refined LoRA backlog state after the 0.18.14 release. Item 0007 is now the next
  recommended platform task and explicitly covers task/mode capability metadata, strict runtime
  application, strict scale counts, unsupported-family rejection, metadata provenance, and
  `mlxgen prepare --lora-paths` q4/q8 bake/export risk. Proposed item 0033 remains unpromoted and
  now requires Wan target-role metadata plus MP4 A/B validation before implementation.
- 2026-06-12: Post-0.18.17 hygiene recounted backlog state, confirmed global IDs remain unique,
  refreshed the recommendation order around the shipped Wan/LightX2V work, kept residual LoRA
  proof coverage as follow-up rather than the immediate top task, and added planned item 0042 for
  the GitHub Actions Node 20 deprecation warning surfaced by release run `27440684820`.
- 2026-06-12: Follow-up hygiene for PR `#4` moved item 0042 to completed after the branch release
  rehearsal (`27443742691`) passed under the upgraded action versions and the only remaining PR CI
  failure was a pre-existing `ruff` baseline issue outside the workflow migration scope.
- 2026-06-13: Release `0.18.18` exposed one remaining Node 20 warning from
  `softprops/action-gh-release@v2`, so item 0042 was reopened, moved back to `planned/`, and
  narrowed to a final GitHub Release publication cleanup pass.
