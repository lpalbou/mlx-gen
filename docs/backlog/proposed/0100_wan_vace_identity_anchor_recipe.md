# Proposed: document the VACE identity-anchor recipe (draft/repair tier)

## Metadata

- Created: 2026-07-25
- Status: Proposed, conditional (documentation-only recipe; ship only if the
  bracket-conditioning route 0097 does not make it redundant)
- Completed: N/A
- Effort: S (docs + one preserved proof case)

## ADR status

- Governing ADRs: [ADR 0005](../../adr/0005_seedvr2_video_quality_proof_requires_five_second_reader_first_clips.md)
  discipline applies in spirit (reader-first proof clips);
  [ADR 0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md) (the doc
  must state the quality tier honestly).
- ADR impact: None (documentation).

## Context (investigation provenance)

The 2026-07-25 storyboard consistency investigation verified that a
subject-identity anchor is wireable TODAY on the existing VACE-1.3B route
with no code changes: `--reference-image <subject>` plus a
seed-frame-preserving mask video (first frame black / rest white) — the
route guards permit the combination. It was deliberately NOT promoted to the
main storyboard path because it is a two-tier quality downgrade:

- 1.3B model class, 480p-bound, CFG-on;
- measured cost class in the prior VACE investigation: ~17 min for 17 frames
  CFG-on versus ~145 s for an 81-frame Lightning A14B scene — a draft/repair
  tool, not a chain driver.

The investigation's ranked roadmap still lists it as the only TODAY-wireable
subject-reference mechanism (A14B has no reference-image input; its Wan2.1
CLIP image cross-attention hooks exist in code but are dormant — no config
sets `added_kv_proj_dim`), so the recipe is worth one documented, proven page
for users who need to repair a single drifted scene.

## What we want to do

1. One documented recipe page (or wan-video.md section): reference image +
   seed-frame-preserving mask video + flag set, with the honest quality/cost
   tier stated up front.
2. One preserved proof case on the storyboard ship (reference = clean S1
   frame; repair the drifted S4) with reader-first clips.
3. A pointer from the host application's companion track so the host's storyboard
   docs can link it as the "repair a broken scene" path.

## Condition for shipping

If 0097 (`last_image` bracket conditioning on full-quality A14B) passes its
quality gate, that route covers most repair cases at a higher tier; then this
item should be closed as "superseded" with a short note instead of shipping.
Decide AFTER 0097's A/B.

## Non-goals

- VACE-14B port (XL; separate decision entirely).
- Making VACE the storyboard chain driver.

## Dependencies and related tasks

- [0097](0097_wan_last_image_bracket_conditioning.md) (supersession gate).
- [0039](0039_wan_vace_video_editing_and_control.md) /
  [0075](0075_wan_vace_conditioning_expansion_after_plain_video_to_video.md)
  (the broader VACE conditioning tracks).
- Host application companion track `proposed/storyboard_consistency_2026_07/`.

## Validation

- The preserved proof case above; no new runtime surface to validate.

## Progress checklist

- [ ] Wait for 0097 A/B verdict (supersession gate)
- [ ] Recipe page with honest tier statement
- [ ] Preserved storyboard-ship proof case
- [ ] Host cross-link
