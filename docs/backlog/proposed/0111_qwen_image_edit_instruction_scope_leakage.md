# Proposed: Qwen image edit instruction-scope leakage

## Metadata

- Created: 2026-08-13
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: none directly; candidate for a quality-gate note once characterized.
- ADR impact: none until root cause is known.

## Context

During an iterative image-editing session on a 614x512 portrait reference image
(2026-08-13), the instruction "change tie color to red" also changed the suit color
that a previous instruction had set to blue — the edit's effect leaked beyond its instructed
scope. Observed alongside (but distinct from) the aspect-compression/drift defect tracked and
fixed separately; scope leakage persists as its own behavior.

Observed on both Qwen edit and FLUX.2 Klein edit chains (2026-08-13: a Klein geometry-fixed
chain kept perfect registration while the suit shifted blue to purple during the tie edit), so
the investigation spans both families. Not yet characterized: whether the leak is model-inherent (edit models re-synthesize loosely
conditioned regions), seed/settings-sensitive, or amplified by an implementation choice
(conditioning resolution, prompt template, latent reuse). No mlx-gen defect is established.

## Demonstrated mitigation (2026-08-13)

Scope pinning in the instruction eliminates the leak in the reproduced Klein case: "change the
tie color to red, keeping the suit exactly the same blue color and everything else unchanged"
produced the red tie with the suit's mean region RGB within ~1.5% of the pre-edit frame
(`untracked/drift_matrix/fx3_pinned.png` vs `fx2.png`; plain instruction shifted the suit from
blue RGB(3,102,245) to purple RGB(97,55,114)). Documented in `docs/image-edit-modes.md`. The
remaining investigation below is about whether implementation defaults (guidance, negative
prompt) amplify the unpinned leak relative to reference implementations.

## Proposed work

1. Reproduce with a controlled protocol: fixed seed, single-instruction edits from the same
   source, measuring off-target change (masked pixel deltas outside the instructed region).
2. Compare against a reference implementation of the same checkpoint at identical settings to
   split model behavior from implementation drift (the Bernini oracle pattern:
   matched-settings, same-seed comparison).
3. If implementation-attributable, fix; if model-inherent, document mitigation guidance
   (instruction phrasing, region hints if the model supports them) in the editing docs.

## Validation expectations

- A committed comparison grid (source, per-instruction outputs, off-target delta measurements).
- Either a fix with regression coverage, or a documented model-behavior note with mitigation
  guidance.

## References

- User report 2026-08-13 (three-step edit chain: glasses, blue suit, red tie)
- Related: image-edit aspect-compression fix (see the completed item for that defect once filed)
