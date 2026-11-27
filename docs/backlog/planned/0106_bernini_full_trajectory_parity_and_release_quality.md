# Planned: Bernini-R 1.3B full-trajectory parity and release quality

## Metadata

- Created: 2026-08-04
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md),
  [ADR 0006](../../adr/0006_generative_video_editing_task_boundary.md), and
  [ADR 0007](../../adr/0007_role_aware_reference_conditioning_and_factored_model_sources.md)
- ADR impact: None currently. ADR 0007's role-aware references and factored-source policy remain
  valid; this item must revise an ADR only if the fix changes those durable contracts.

## Context

Completed item 0105 delivered the renderer-only BF16 runtime, but its original visual pass was
wrong. Full-frame 5K review and two independent audits found that every required visual-quality
case fails. The schema-v3 bundle now truthfully records `machine_contract_passed=true`,
`visual_review_complete=true`, `visual_quality_passed=false`, and `passed=false`; the release
registry reports `FAIL`.

The strongest implementation checks currently pass:

- 17-frame VAE encode/decode parity has shared-latent cosine `0.99999761` and relative L2
  `0.00255`;
- a runtime-BF16 transformer comparison spanning five latent slices has cosine `0.99976615` and
  relative L2 `0.021655`;
- focused scheduler, APG, source-ID, prompt-encoder, tokenizer, and component-source checks pass.

Those isolated checks do not prove the first denoise step or full trajectory on real Bernini
conditioning. The visual failures include nearly static motion, weak reference fidelity,
four-latent-slice cadence jumps, and severe tail corruption. An exact-upstream-prompt 33-frame
diagnostic also fails: it records 571 UMT5 tokens truncated to 512, progressively collapses from
about frame 13, and has a latent-boundary/non-boundary transition ratio of `2.0645`.

Evidence update from Sunday, August 9, 2026: two runtime precision fixes materially improved the
renderer math (`float64` APG projection and official-sensitive FP32 transformer paths), but they do
not close every user-visible gap. The original shirt RV2V contract still fails at the neck/collar
junction when driven from an ordinary shirt photo. The original one-image statue R2V case still
reads as scene/camera drift rather than a clean local head turn. Any result that requires prepared
inputs, trimming, tighter crops, anchored source loops, or stronger prompt restatement must not be
counted as support. The remaining open question is therefore narrower: which gaps are still
Bernini parity defects, and which are intrinsic limits of the weaker public conditioning contract.

Evidence update from Monday, August 10, 2026: the stricter target has been made explicit in the
official public 1.3B example matrix. That matrix now has accepted qualitative local evidence for
five official rows (`t2i`, `i2i`, `t2v`, `r2v`, `rv2v_case1`). The remaining open rows are
`v2v_case1`, `v2v_case2` / `mv2v`, `v2v_case3`, `r2v_case2`, and `ads2v`. The parity item therefore
has two subproblems:

1. close or explicitly reject the remaining open public rows, starting with `v2v_case1`; and
2. only after the matrix is current, decide whether the older release-bundle failures still point
   to an implementation gap, a scope gap, or simply an earlier bounded proof slice.

The bundled upstream output clips cannot close the comparison: their files do not attest the
producing checkpoint or inference recipe, and the associated upstream launcher defaults to the
14B renderer. They are qualitative targets, not 1.3B parity baselines.

## What we want to do

Find the first real-input trajectory divergence, fix it only if the evidence identifies an MLX
implementation mismatch, and earn a new visual release pass against an attested Bernini-R 1.3B
reference.

## Requirements

- Keep the public runtime and validation registry experimental/`FAIL` until every acceptance gate
  below passes.
- Work backwards from pixels: VAE decode, denoise trajectory/transformer/scheduler, then prompt and
  condition encoders.
- Export exact PyTorch inputs instead of comparing integer seeds across PyTorch and MLX.
- Build a real-input first-step fixture containing exact positive/negative prompt embeddings,
  independently encoded reference/source latents, initial noise, timestep, sigma, source IDs, and
  scheduler state.
- Compare the unconditioned, image/reference, text, and source branches independently before APG or
  chained CFG; then compare guidance reduction and the post-UniPC latent.
- Extend a passing first-step fixture to several steps and then the complete short trajectory. Log
  shapes, dtypes, cosine, relative L2, max absolute error, and per-latent-slice results.
- Check the four-frame cadence explicitly. Do not impose a public 33-frame minimum: Bernini accepts
  every `4n+1` count, and current 33-frame samples still show cadence seams.
- Obtain or generate a metadata-bearing official/Diffusers **1.3B** baseline. Record checkpoint
  revision, code revision, prompt/token truncation, negative prompt, dimensions, frames, steps,
  flow shift, guidance/APG settings, condition geometry, and exact initial noise.
- Treat the official public 1.3B example matrix as a required checklist, not as informal context.
  Required rows are `t2i`, `i2i`, `t2v`, `v2v_case1`, `v2v_case2`, `v2v_case3`, `r2v`,
  `r2v_case2`, `rv2v_case1`, and `ads2v`; if one row is intentionally excluded from the public
  mlx-gen surface, the reason must be stated explicitly and preserved in evidence.
- Resolve the public-source ambiguities before judging parity: prompt-enhancer usage, whether a row
  is renderer-only or planner-shaped, and the exact source revision and case inventory for the
  public testcase tree.
- Progressively match the reference recipe. Start with bounded shapes; do not run the full
  848x480x81/40-step profile without user approval for the compute cost.
- Distinguish "math fixed" from "contract proven". A run that only works after patched context
  references, trimmed startup frames, tighter crops, anchored source-video loops, or stronger
  prompt/task restatement does not by itself prove the original public contract.
- Preserve exact MP4s and schema-v3 5K paged proof sheets. Every MLX frame, conditioned source
  frame, reference, upstream/reference frame, and highest localized-change transition must remain
  inspectable and hash-bound.
- Keep whole-process physical-memory measurement separate from storage size and MLX component
  peaks. Recheck the 18 GB envelope on the final passing profile.
- Do not change implementation math merely to improve one prompt. Require a parity failure or a
  reproducible cross-prompt defect before patching shared Wan/Bernini code.

## Suggested implementation

1. Add an export mode to the local Diffusers Bernini-R 1.3B reference that writes the complete
   first-step fixture and provenance manifest.
2. Add an MLX replay tool that consumes that fixture without re-tokenizing, re-encoding, or
   regenerating noise.
3. Compare VAE-decoded clean/reference/source latents and the four Bernini prediction branches.
4. Compare APG/chained-CFG reduction, scheduler model input, UniPC history, and updated latent.
5. Expand from one step to five steps across at least five temporal latent slices; locate the first
   divergence and its relationship to source-aware RoPE/packed segment extraction.
6. Patch the narrowest proven mismatch and add focused regression fixtures that keep ordinary Wan,
   TI2V, A14B, and VACE behavior intact.
7. Run bounded R2V, RV2V, and V2V visual cases with stable seeds, then one attested recommended
   profile after explicit compute approval.
8. Regenerate the portable bundle, independently inspect every page and playable MP4, and promote
   the registry only if the required quality gate passes.

## Scope

- Bernini-R 1.3B BF16 renderer trajectory parity.
- R2V, RV2V, and source-only V2V guidance paths.
- Prompt/reference/source conditioning, packed transformer, UniPC state, VAE decode, temporal
  cadence, proof tooling, registry, and user documentation.

## Non-goals

- A14B renderer support or Qwen2.5-VL planner integration.
- Quantization, LoRA, masks, warm-start strength, or new public task names.
- Treating the unattested upstream MP4s as 1.3B parity evidence.
- Publishing models or proof artifacts externally.
- Hiding a quality failure behind file-decode, nonzero-motion, or unit-test success.

## Acceptance criteria

- Exact-input first-step and multi-step parity reports pass declared tolerances for every guidance
  branch, reduction, scheduler update, temporal slice, and VAE decode.
- A metadata-bearing official/Diffusers 1.3B baseline is retained with exact reproducibility data.
- Required R2V, RV2V, reference A/B, and V2V cases pass human visual review for prompt adherence,
  identity/reference fidelity, source preservation, useful motion, temporal continuity, and clean
  first/middle/final frames.
- Cadence-boundary and localized-transition diagnostics do not reveal periodic seams or corruption.
- The proof CLI exits nonzero on any missing/stale/low-resolution/unreviewed/failed evidence.
- The portable manifest verifier proves exact file inventory, size, SHA-256, report schema, review
  schema, and sheet contract.
- Focused Bernini tests, impacted Wan regressions, `make test-fast`, `make lint`, `make build`, and
  the justified slow suite pass.
- Only after all gates pass do the validation registry and user recommendations move from `FAIL`
  and experimental to a narrowly stated supported profile.

## Evidence

- [Current failed schema-v3 proof](../../assets/validation/bernini-r-1.3b-2026-08-04/README.md)
- [Official public 1.3B example parity matrix](../../assets/validation/bernini-r-1.3b-2026-08-04/official_example_parity_matrix.md)
- [Functional integration history](../completed/0105_bernini_r_1_3b_renderer_integration.md)
- [Bernini user guide](../../bernini.md)
- [ADR 0007](../../adr/0007_role_aware_reference_conditioning_and_factored_model_sources.md)
