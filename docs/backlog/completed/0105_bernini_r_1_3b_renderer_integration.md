# Completed: Bernini-R 1.3B renderer integration

## Metadata

- Created: 2026-08-03
- Status: Completed
- Completed: 2026-08-04

## Post-completion correction — 2026-08-04

This item is complete as a **functional runtime integration**, not as a release-quality model
validation. The original visual disposition was wrong. Full-frame 5K review plus two independent
audits found all five required quality cases fail; the validation registry is now `FAIL` and the
runtime is experimental. The corrected schema-v3 report records machine contract `true`, complete
hash-bound review `true`, visual quality `false`, and overall pass `false`.

The failure was hidden by low-resolution/sample-only review and by conflating successful tests and
decodable MP4s with acceptable generation. The ten model-backed cases below are one `run_1` matrix.
The five “cycles” were engineering phases, not five independent generation/review repetitions.
Planned item [0106](../planned/0106_bernini_full_trajectory_parity_and_release_quality.md) owns the
blocking full-trajectory parity and quality work.

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md),
  [ADR 0006](../../adr/0006_generative_video_editing_task_boundary.md), and
  [ADR 0007](../../adr/0007_role_aware_reference_conditioning_and_factored_model_sources.md)
- ADR impact: ADR 0007 establishes reference-input roles and factored component provenance.

## Context

ByteDance's Apache-2.0 `Bernini-R-1.3B-Diffusers` renderer is a Wan2.1-1.3B fine-tune for
reference-to-video and reference-guided video editing. The development host has 128 GiB unified
memory, while the target deployment envelope is about 18 GB. The official monolithic repository
is about 28.9 GB, but the renderer transformer is about 5.7 GB and its tokenizer, UMT5, and VAE
are compatible with explicitly pinned Wan components.

Bernini cannot be treated as stock Wan or VACE. Its exact semantics add
source-ID rotary phase, heterogeneous packed latent segments, target-only extraction, and
task-specific three-/four-pass guidance.

## Shipped code reality

- `BerniniRenderer` is a dedicated Wan-family runtime with official R2V, RV2V, and prompt-guided
  V2V conditioning and guidance semantics.
- `WanTransformer` retains its ordinary rectangular path and adds a narrow heterogeneous packed
  segment path with source-ID RoPE for Bernini.
- Factored model resolution pins the renderer and Wan base independently, downloads only required
  component patterns, validates compatibility, records component provenance, and preflights free
  disk space with 2 GiB headroom.
- The capability planner and unified CLI/Python runtime distinguish repeatable reference images
  from primary images and source video. Unsupported role combinations fail before model loading.
- The official 1.3B model identity is exact. The A14B `ByteDance/Bernini-R` repository and
  similarly named third-party repositories fail closed instead of silently selecting 1.3B.
- Low-RAM execution sequences component residency, streams VAE decode, preserves progress and
  failure diagnostics, and records resolved geometry, prompt token counts, reference order,
  source paths, and component revisions.
- Bernini is BF16-only. The public constructor rejects every non-`None` quantization request
  because q4 failed quality validation and the nominal q8 experiment quantized no applicable
  transformer layers.

## Delivered scope

- Reference-to-video through public `text-to-video` plus one to eight ordered reference images.
- Reference-guided video editing through public `video-to-video` plus a source video and one to
  eight references.
- Prompt-guided video editing through public `video-to-video` plus a source video.
- Unified CLI, Python planning/runtime, metadata replay, validation-registry, documentation, and
  portable proof support.

## Non-goals retained

- Wan2.2 A14B Bernini renderer support.
- Qwen2.5-VL semantic-planner integration or a public `mv2v` claim.
- New public `r2v` or `rv2v` task names.
- Model conversion, upload, cache deletion, or publication.
- Claims that the official 848x480x81-frame/40-step profile fits 18 GB.

## Implementation

1. Extended Wan rotary/transformer execution with an isolated packed-segment path while keeping
   ordinary Wan and VACE paths as regression oracles.
2. Added explicit renderer/base component sources, required-pattern resolution, compatibility
   checks, provenance, and cold-download disk preflight.
3. Added exact renderer paths: chained APG for R2V, four-branch chained CFG for RV2V, and APG for
   prompt V2V. Source/reference latents are encoded independently; the target starts from pure
   noise and packed order is video, references, target.
4. Added typed reference-role capabilities and routing without weakening the existing primary
   image/video mixed-media rejection.
5. Added a versioned proof harness that rejects stale cases, binds review decisions to artifact
   and contact-sheet hashes, exports a portable sanitized bundle, and covers integrity,
   provenance, numerical parity, memory, and real media.

## Five completed engineering phases

- [x] Cycle 1: architecture, disk, official-source, framework, and ADR review.
- [x] Cycle 2: numeric core, component sources, deterministic parity, and initialization smoke.
- [x] Cycle 3: runtime, role-aware CLI/Python integration, metadata, and contract tests.
- [x] Cycle 4: real R2V/RV2V/V2V generations, controls, contact sheets, and fixes.
- [x] Cycle 5: cross-review, regression gates, proof audit, and core documentation.

The parity skeptic concentrated on official packing, source-ID RoPE, guidance, scheduler, VAE,
identity gating, and proof sufficiency. The framework/memory skeptic concentrated on capability
compatibility, low-RAM lifecycle, factored downloads, portable evidence, media integrity, and
backlog/documentation truthfulness. Their later audit invalidated the visual pass and identified
the first-step/full-trajectory parity gap preserved in item 0106.

## Validation and proof

The durable [validation overview](../../assets/validation/bernini-r-1.3b-2026-08-04/README.md)
links the actual clips, 5K paged contact sheets, structured report, numerical parity data, logs,
inputs, memory samples, exact revisions, and SHA-256 manifest. The portable verifier checks the
exact inventory, size, and SHA-256 of every bundled artifact in addition to report/review/sheet
schema compatibility.

Model-backed coverage comprises eight 17-frame diagnostic videos and two five-frame structural
smokes. None of the required rows is accepted as quality evidence:

- eight-reference R2V is nearly static, misses requested references/action, and has cadence jumps;
- reference-guided RV2V partially transfers pinstripes but corrupts frames 13-16;
- a neutral same-seed/source/prompt pinstripe-versus-black RV2V pair held the guidance branch
  fixed and proved sensitivity to reference content;
- the corresponding black-shirt reference is retained as a negative result because it did not
  transfer faithfully;
- prompt V2V inserts a cartoon-like snowman and severely corrupts frames 13-16;
- source/reference ablations and 848/1280 condition-cap smokes exercise role and shape contracts.

These are failures, not minor limitations. Controlled 33-frame and 40-step diagnostics did not
earn a pass; the exact-upstream-prompt 33-frame R2V progressively collapses from about frame 13.

Numerical evidence includes exact Diffusers 0.35.2 UniPC schedules with four-step replay max-abs
`1.335e-5`, APG compact-case max-abs at most `5.96e-7`, transformer FP32 cosine
`0.999999869`, five-slice runtime BF16 cosine `0.99976615`, and shared-latent 17-frame VAE decode
cosine `0.99999761` with relative L2 `0.00255`. These are exact-input stage comparisons, not a
claim of cross-backend pixel identity or complete full-trajectory parity.

Final gates:

- `make test-fast`: 640 passed, 1137 deselected.
- Impacted Bernini/Wan/router/release/resolution/runtime slice: 755 passed, 7 skipped.
- The first full run collected 1,777 tests: 1,746 passed, 24 failed, and 7 skipped. Five failures
  were a Bernini-introduced ordinary-Wan scheduler-constructor regression; the fix is covered by
  the now-green step-grid module and complete Wan suite. The remaining 19 comprise one corrupt
  DepthPro cache artifact, one Z-Image LoRA bake defect in untouched code, four FIBO VLM exact-
  golden mismatches, six generated-image mismatches, and seven failures before output creation.
  They remain unresolved and are not represented as passing; no golden or cache was replaced.
- `make lint`: clean.
- Changed Python surface: 43 files format-clean; seven key source files type-check clean with
  import following disabled to isolate the changed surface from existing repository baselines.
- `make build`: wheel and source distribution built successfully.
- `git diff --check`: clean.
- Validation profile `bernini_r_1_3b_2026_08_04`: **FAIL** after corrected visual review.
- Portable manifest, structured JSON, hash-bound review, media decode, image verification, and
  changed-document local links can pass as evidence-integrity gates while overall visual quality
  remains **FAIL**.

## 18 GB conclusion

The largest bounded proof measured **9.45 GB whole-process Darwin physical footprint** on the
128 GB development host. This shows that the tested shapes fit an 18 GB-class memory envelope,
but it is not a direct measurement on one and does not make the failed outputs production
candidates.

A cold selective download is about **16.36 GiB**. With the enforced **2 GiB** safety headroom, a
fresh machine needs **18.36 GiB free disk**, so a strict cold 18 GB storage budget is slightly
short. If the pinned Wan base is already complete in cache, only missing renderer files count and
the route fits comfortably.

## References

- [Bernini user guide](../../bernini.md)
- [Portable proof overview](../../assets/validation/bernini-r-1.3b-2026-08-04/README.md)
- [Schema-v3 proof report](../../assets/validation/bernini-r-1.3b-2026-08-04/bundle/bernini_proof_report.json)
- [Hash-bound visual review](../../assets/validation/bernini-r-1.3b-2026-08-04/bundle/visual_review.md)
- [Portable manifest](../../assets/validation/bernini-r-1.3b-2026-08-04/bundle/portable_manifest.json)
- [ADR 0007](../../adr/0007_role_aware_reference_conditioning_and_factored_model_sources.md)
- Official renderer revision `ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce`
- Official source revision `2d2b4591ac053ec25c6371b01a5a6746679e5793`
