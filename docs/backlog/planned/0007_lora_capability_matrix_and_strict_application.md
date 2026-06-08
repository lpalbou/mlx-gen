# Planned: LoRA capability matrix and strict application

## Metadata

- Created: 2026-05-28
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: May revise the generation capability contract. No new ADR is needed if LoRA remains
  task-specific capability metadata plus per-family adapter mappings. Escalate to an ADR only if
  MLX-Gen changes LoRA into a plugin/provider interface, stores adapters as a separate package
  class, or permits automatic fallback/substitution behavior.

## Context

MLX-Gen already exposes LoRA arguments and metadata, and the current public docs advertise LoRA
support. That support is real for some image families but not universal. As MLX-Gen becomes the
Apple Silicon backend for AbstractVision, callers need a reliable capability answer before they
offer LoRA controls in a UI or workflow.

Qwen-Image-Edit-2511 also makes LoRA strategically important: its model card calls out integrated
LoRA capabilities and community LoRA effects as part of the 2511 upgrade.

## Current code reality

- `src/mflux/cli/parser/parsers.py` adds `--lora-style`, `--lora-paths`, and `--lora-scales`.
- Unified `mlxgen generate` advertises `--lora-paths` and `--lora-scales` in help text, but
  `resolve_generation_plan(...)` has no `has_lora` input and `GenerationCapability` has no LoRA
  fields. The router therefore cannot centrally reject unsupported LoRA requests today.
- `src/mflux/models/common/resolution/lora_resolution.py` now fails unresolved LoRA paths before
  model load, and `docs/troubleshooting.md` tells users that requested LoRAs are required.
- `src/mflux/models/common/lora/mapping/lora_loader.py` still has loader-level silent-degradation
  paths: missing or unreadable files print an error and return from `_apply_single_lora()`, and
  zero-match adapters can finish with warnings rather than a failed generation.
- `LoraResolution.resolve_scales(...)` pads or truncates mismatched scale counts with a warning.
  That is not strict enough for user-requested adapters because scale mistakes can silently change
  the generated result.
- `src/mflux/task_inference.py` exposes generation capabilities for tasks, modes, masks,
  outpaint, and image strength, but it does not yet expose task-specific LoRA support.
- LoRA mappings exist for FLUX.1, FLUX.2, Qwen, and Z-Image:
  - `src/mflux/models/flux/weights/flux_lora_mapping.py`
  - `src/mflux/models/flux2/weights/flux2_lora_mapping.py`
  - `src/mflux/models/qwen/weights/qwen_lora_mapping.py`
  - `src/mflux/models/z_image/weights/z_image_lora_mapping.py`
- FLUX.2 and Z-Image also have training adapters.
- ERNIE and Bonsai accept `lora_paths` in constructor signatures for prepare compatibility, but
  their initializers delete those arguments and set `model.lora_paths = None`.
- Wan, SeedVR2, and FIBO do not have proven LoRA mappings in the current MLX-Gen tree. FIBO is
  already rejected when LoRA is requested, but it remains a useful negative test because FIBO Edit
  itself is currently deprioritized and unavailable through unified generation.
- `mlxgen prepare` parses LoRA flags for every model and currently rejects only FIBO explicitly.
  Signature-based forwarding prevents some constructor crashes, but it is not a capability
  contract and can still leave users with unsupported or ignored adapter requests on families that
  accept then discard LoRA kwargs.
- `src/mflux/models/common/weights/saving/model_saver.py` bakes and strips LoRA wrappers before
  save. `src/mflux/models/common/lora/mapping/lora_saver.py` skips the bake when the LoRA delta
  shape does not match the base weight. That is especially risky for q4/q8 packed linears, where a
  prepared model can look valid while the requested LoRA was not baked into the saved weights.
- User-facing docs are mixed: `docs/api.md` documents generation modes but does not yet explain a
  reliable LoRA capability contract, while inherited/model-local READMEs still include broad LoRA
  examples that may not match the unified router's support boundaries.

## Problem

LoRA should be treated as required user input, not best-effort decoration. If a user asks for a LoRA
and it is missing, corrupt, maps zero keys, or targets a family that does not support LoRA, MLX-Gen
should fail early with a clear message. Silent or warning-only behavior is dangerous because the
output image can look plausible while ignoring the requested adapter.

This is now planned because the current behavior is inconsistent with ADR 0002 and the user-facing
troubleshooting docs: resolution is strict, but application can still degrade silently later.

## What we want to do

Add a capability matrix and strict LoRA application policy in two phases:

Phase 1, strictness and introspection:

1. Add family-level capability metadata for LoRA inference and LoRA training.
2. Add task/mode-specific LoRA metadata to each `GenerationCapability` so callers can ask whether
   LoRA is supported for T2I, latent I2I, edit-reference I2I, multi-reference I2I, canvas-guided
   outpaint/reframe, T2V, or I2V.
3. Make the unified `mlxgen` router reject LoRA flags for unsupported families and unsupported
   task/mode combinations before model load.
4. Change LoRA loading so user-requested files must exist, load, and apply at least one mapped
   target.
5. Make `--lora-scales` strict: a scale list must match the adapter list exactly, and scales
   without adapters must fail before model load.
6. Keep partial-match warnings for valid adapters, but fail zero-match adapters by default.
7. Make `mlxgen prepare --lora-paths` fail closed unless the selected family and quantization mode
   has a tested, deterministic LoRA bake/export path.
8. Add docs and generated capability metadata so AbstractVision can decide whether to display LoRA
   controls.

Phase 2, visual support claims:

1. Select one known public adapter per supported image family and task direction.
2. Produce model-backed A/B proofs with identical prompt, seed, dimensions, steps, guidance, and
   input image where relevant.
3. Promote capability status from "mapped but unvalidated" to "validated" only for the exact
   model family, mode, and package class that passed visual review.

Initial support matrix should be explicit and task-aware:

| Family | Current MLX-Gen LoRA status | Task directions | Difficulty | Path |
| --- | --- | --- | --- | --- |
| FLUX.1 | Inference mapping exists in the mflux-derived code, but unified `mlxgen capabilities` is not currently centered on FLUX.1. Fill, depth, control, redux, kontext, and in-context variants share related transformer concepts but need route-specific proof. | Dedicated CLI compatibility first; unified T2I/fill only after explicit capability work | Low for loader strictness; medium for unified proof | Do not advertise FLUX.1 LoRA through unified capabilities until FLUX.1 itself is deliberately revalidated. Dedicated CLI LoRA can remain compatibility behavior if it becomes strict and documented separately. |
| FLUX.2 Klein | Inference mapping exists; training adapter exists. | T2I, latent I2I, edit-reference, multi-reference | Low to medium | Keep supported, add strict loader tests and one visible T2I plus one I2I/edit validation row per representative package. |
| Qwen Image / Qwen Image Edit / Qwen Image Edit 2509 / 2511 | Inference mapping exists. Official Qwen Image Edit 2511 advertises integrated LoRA capability and many adapters exist in the HF model tree. | T2I where the model supports it; I2I edit-reference and multi-reference for edit models | Medium | Keep supported only after visible validation with real Qwen LoRAs. Validate base Qwen Image, Qwen Image Edit, 2509, and 2511 separately because prompt/image contracts differ. |
| Z-Image / Z-Image Turbo | Inference mapping and training adapter exist; one public LoRA slow test exists. | T2I, latent I2I where routed | Low to medium | Keep supported, make strict failure behavior common with FLUX/Qwen, and preserve visible LoRA regression output. |
| ERNIE Image / Turbo | Constructors accept `lora_paths` for compatibility but initializer ignores them. | T2I, latent I2I | Medium to high | Reject LoRA flags before model load until a real ERNIE mapping and a public adapter proof exist. Mapping may be feasible, but no current code applies it. |
| Bonsai | Initializer ignores LoRA; packed ternary/low-bit layout is not a normal adapter target. | T2I | High / not priority | Reject LoRA flags. Revisit only if Bonsai publishes adapter semantics that match the packed MLX runtime. |
| FIBO / FIBO Edit | LoRA is rejected today; no proven mapping. FIBO Edit is deprioritized and not a release-quality unified edit route. | T2I only for base FIBO; FIBO Edit disabled in unified generation | High / deferred | Keep rejected. Do not spend LoRA work here until base FIBO/FIBO Edit priority changes. |
| Wan2.2 TI2V/T2V/I2V | No mapping or constructor support. A14B has separate high-noise and low-noise transformers. | T2V, I2V | High | Track separately in [proposed item 0033](../proposed/0033_video_lora_for_t2v_i2v.md). Start with Wan only after item 0015 and video integrity work are stable. |
| SeedVR2 | No LoRA mapping; current route is restoration/upscale rather than generation. | Image restoration/upscale today; video restoration proposed in item 0032 | Low value / not priority | Reject LoRA flags. Treat model-specific restoration controls such as resolution and softness separately from LoRA. |

Task-direction roadmap:

| Direction | Near-term stance | Notes |
| --- | --- | --- |
| T2I | Support for FLUX.2, Qwen Image, and Z-Image after strict application tests. FLUX.1 stays dedicated-CLI or revalidation-gated unless unified FLUX.1 support is deliberately restored. | This is the easiest surface because one prompt produces one image and visible adapter effects are easy to compare. |
| Latent I2I | Support only for families whose latent route uses the same mapped transformer and has source-preserving visual proof. | A LoRA that works for T2I can still overpower the encoded source when `--image-strength` is high. |
| Edit-reference I2I | Support only after a single-image edit proof shows both adapter effect and prompt adherence. | Qwen and FLUX.2 edit paths need separate validation from latent I2I. |
| Multi-reference I2I | Support only after a two-reference proof shows adapter effect without losing the intended reference composition. | Validate separately for Qwen 2509/2511 and FLUX.2 because their multi-reference contracts differ. |
| Canvas reframe/outpaint | Support only after item 0019's canvas route and a LoRA A/B proof pass for the specific family. | LoRA may amplify source-window drift; treat it as a separate validation row from normal edit-reference I2I. |
| T2V | Proposed, not part of this planned item. | Requires Wan-specific mapping and temporal validation; see item 0033. |
| I2V | Proposed, not part of this planned item. | Same as T2V plus source-image identity and motion validation; see item 0033. |

## Suggested implementation

Make the capability contract the source of truth:

- Extend `GenerationCapability` with LoRA fields such as:
  - `supports_lora: bool`;
  - `lora_status: "unsupported" | "mapped-unvalidated" | "validated"`;
  - `lora_validation_profile: str | None`;
  - `lora_target_roles: tuple[str, ...]` for future multi-transformer/video models.
- Keep capability metadata task/mode specific. For example, `qwen.edit` and
  `qwen.multi-reference` can have different LoRA validation status even though they share the
  same handler.
- Add `has_lora` to `resolve_generation_plan(...)` and `resolve_task(...)`, derived by the CLI
  and Python callers from `lora_paths`, `lora_scales`, or `lora_style`.
- Reject unsupported LoRA requests before loading weights. Error text should name the selected
  model, resolved public task/mode, and the closest supported alternatives when any exist.
- Add a structured loader result, for example `LoRAApplicationReport`, with adapter path, scale,
  matched key count, applied target count, unmatched key count, and target roles. Save it in
  generated metadata.
- Add a dedicated `LoRAApplicationError` or equivalent `ValueError` for unreadable files, corrupt
  files, zero matched keys, zero applied targets, missing A/B matrices, target-path misses, and
  matrix shape mismatches.
- Treat `mlxgen prepare` as a separate contract from runtime LoRA loading. Runtime LoRA can wrap
  linear layers at generation time; prepared-package LoRA baking must either prove output
  equivalence after save/reload or fail with a clear message. In particular, q4/q8 packed
  linears must not skip LoRA deltas silently.
- If a LoRA is baked into a saved package, record original adapter paths, resolved files, scales,
  bake status, target counts, and any quantization constraints in package metadata and the model
  card.
- Keep adapter downloads explicit. `mlxgen download --model <lora-repo> --all-files` can prepare
  cache state; generation must not auto-download LoRA files.
- Do not make `--lora-scales` a standalone behavior. If scales are present and paths are absent,
  fail with a parser error.

## Why it might matter

LoRA is one of the fastest ways to make MLX-Gen useful for personal styles, product references,
characters, and AbstractVision workflows. It is also one of the easiest places to lie accidentally:
loading a model and ignoring the adapter creates outputs that are technically valid but
semantically wrong.

## Scope

- Family-level LoRA capability metadata for generation and training surfaces.
- Task-direction metadata for whether a family supports LoRA in T2I, latent I2I, edit-reference
  I2I, multi-reference I2I, fill/inpaint/outpaint, T2V, or I2V.
- Strict router or model-load rejection for families without proven LoRA inference support.
- Strict loader behavior for unreadable, corrupt, zero-match, or shape-invalid user-requested
  adapters.
- Strict scale-count validation and a clear error for scales without adapter paths.
- Generated metadata that records which adapters were actually applied, their scales, and how many
  targets matched.
- Prepared-package policy for LoRA: either tested bake/export for the selected family and
  quantization mode, or early rejection before loading/saving.
- Documentation cleanup for inherited/model-local READMEs that currently make broader LoRA claims
  than the unified router can prove.
- Docs and tests that align the public troubleshooting claim with runtime behavior.

## Non-goals

- Do not implement LoRA for every family.
- Do not implement Wan/video LoRA in this item; preserve that work in proposed item 0033 unless it
  is promoted.
- Do not add automatic LoRA downloads during generation.
- Do not change existing quantization policies.
- Do not bake LoRAs into prepared packages unless that behavior is explicitly documented, tested,
  and output-equivalent after save/reload for the relevant quantization mode.

## Validation

- Unit test missing file, corrupt file, zero matched keys, partial matched keys, and successful
  application.
- Unit test strict scale behavior: too few scales, too many scales, and scales without adapter
  paths fail instead of padding or truncating.
- Router tests proving LoRA is accepted only for capabilities marked as LoRA-supported and rejected
  for ERNIE, Wan, Bonsai, FIBO, SeedVR2, and unsupported modes until implemented.
- `mlxgen prepare` tests proving unsupported LoRA is rejected for ERNIE, Bonsai, Wan, SeedVR2, and
  FIBO, instead of relying on constructor signatures or silently ignored kwargs.
- Prepared-package tests proving any claimed LoRA bake works after save/reload. For q4/q8, this
  must include an output-equivalence or strong latent/weight-effect check, not only shard count or
  file-size checks.
- `mlxgen capabilities --model ...` tests proving LoRA fields are visible per mode.
- Python API tests proving `resolve_generation_plan(... has_lora=True)` rejects unsupported modes
  before model instantiation.
- Real-image contact sheet with Qwen-Image-Edit-2511 and at least one public 2511 LoRA. Include
  single-image edit and multi-reference edit rows if both are claimed.
- Real-image proof for each task direction marked supported:
  - T2I: one model-backed row per supported image family;
  - latent I2I: one source-preserving variation row where claimed;
  - edit-reference/multi-reference: one focused Qwen or FLUX.2 edit row where claimed;
  - canvas outpaint: after item 0019 route ownership is stable for the target model family;
  - native fill/inpaint outpaint: only after item 0019 or a follow-up item provides a validated
    fill/mask backend.
- Metadata test proving generated images preserve original LoRA paths and scales.
- Metadata test proving generated images record applied target counts, not only requested paths.
- Metadata tests for FLUX.2 edit-reference and multi-reference outputs, because those routes
  currently apply transformer LoRA through initialization but must still preserve LoRA provenance
  in generated output metadata.
- Save/prepare test proving baked or stripped LoRA behavior is documented, deterministic, and not
  skipped on packed q4/q8 weights.

Recommended first visual proof set:

| Family | Minimal proof |
| --- | --- |
| Z-Image Turbo | Existing public Z-Image Turbo LoRA A/B row, because public LoRA examples already exist and the model path is lightweight compared with Qwen. |
| FLUX.2 Klein | One public FLUX.2 Klein LoRA A/B row for T2I and one edit-reference row, preferably q8 package first. |
| Qwen Image Edit 2511 | One public 2511 adapter row for single-image edit and one multi-reference row. If the adapter is an acceleration LoRA rather than style/content LoRA, validate both speed/step contract and visual quality. |
| Qwen Image / Qwen Image Edit / 2509 | Validate separately before marking supported because base generation, 2509, and 2511 use different prompt/image contracts. |

## Progress checklist

- [x] Confirm unresolved LoRA paths now fail in `LoraResolution`.
- [x] Confirm loader-level missing/unreadable and zero-match cases can still avoid hard failure.
- [ ] Add explicit family-level LoRA capability metadata.
- [ ] Add task-direction LoRA metadata so UI/API callers know which modes can accept adapters.
- [ ] Add `has_lora` planning input and route-level rejection before model load.
- [ ] Make `--lora-scales` fail when the count differs from `--lora-paths`, and fail when scales
      are provided without paths.
- [ ] Reject LoRA flags for unsupported families before generation starts.
- [ ] Reject or prove `mlxgen prepare --lora-paths` for unsupported families and q4/q8 packed
      packages; no skipped-bake prepared package may be saved as if LoRA was applied.
- [ ] Make loader-level missing, unreadable, corrupt, zero-match, and shape-invalid cases fail
      closed.
- [ ] Return and persist a structured LoRA application report.
- [ ] Add focused tests for strict LoRA application.
- [ ] Add first model-backed A/B contact sheets for Z-Image Turbo, FLUX.2 Klein, and Qwen Image
      Edit 2511 before marking those exact modes validated.
- [ ] Update docs and generated capability metadata.

## Guidance for future agents

Start with strict diagnostics and the capability matrix before adding new family mappings. If a
family cannot prove LoRA application with a known adapter and visible output difference, mark it as
unsupported rather than "probably works".

## Sources checked

- `src/mflux/models/common/lora/mapping/lora_loader.py`
- `src/mflux/models/qwen/weights/qwen_lora_mapping.py`
- `src/mflux/models/flux2/weights/flux2_lora_mapping.py`
- `src/mflux/models/z_image/weights/z_image_lora_mapping.py`
- Qwen-Image-Edit-2511 model card: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- Qwen-Image-Edit-2509 model card: https://huggingface.co/Qwen/Qwen-Image-Edit-2509
- FLUX.2 Klein 4B model card: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- Wan2.2 TI2V-5B Diffusers model card: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers
- Public Z-Image Turbo LoRA example: https://huggingface.co/renderartist/Classic-Painting-Z-Image-Turbo-LoRA
