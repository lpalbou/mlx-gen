# Planned: First-class I2I modes and outpaint/reframe UX

## Metadata
- Created: 2026-06-04
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: May revise existing ADRs or add a small task-capability ADR before closure.

The durable policy to preserve is that public tasks should describe media direction
(`text-to-image`, `image-to-image`, `text-to-video`, `image-to-video`). Backend modes such as
latent img2img, edit-conditioned I2I, multi-reference I2I, masked fill, inpainting, and
outpainting should not be exposed as fake public tasks unless an ADR explicitly accepts that
taxonomy.

## Context

MLX-Gen now has public media-direction tasks, internal image-to-image modes, and a capability
planner from [completed item 0020](../completed/0020_generation_capability_contract.md). That
solves the task taxonomy problem for text/image/video routing. This item tracks the remaining
canvas-expansion work around reframe and outpaint. Earlier planning assumed FLUX.1 Fill should be
the first implementation target because native diffusion outpainting is usually a fill/inpaint
mask contract. Current MLX-Gen validation is centered on FLUX.2 Klein and Qwen Image Edit, so the
first shipped path is a canvas-guided outpaint workflow for those edit models: build an expanded
canvas, run the edit backend, and apply adaptive source blending only when it does not create
ghosting.

This item therefore tracks two related but distinct workflows:

- Generative reframe: use an edit or latent I2I model to zoom out, extend background, or infer a
  missing object boundary. This can start with the models MLX-Gen already validates most heavily,
  but it must be described as generative and not pixel-locked.
- Canvas-guided outpaint: place the source image on a larger canvas, initialize the new area from
  edge-extended source context, run a supported edit model on the expanded canvas, then compare the
  generated source window with the original source. Apply a content-aware source blend only when
  the two are close; skip the blend when the edit model has reconstructed or moved the scene.
- Native fill/inpaint outpaint: place the source image on a larger canvas, mask only the new area,
  and let a dedicated fill/inpaint backend regenerate the masked pixels. This remains future work
  for unified `mlxgen generate`.

Standard terms:

- Inpainting: regenerate masked areas inside an existing canvas.
- Outpainting / image expansion: extend beyond the original image boundaries by placing the source
  on a larger canvas and generating the new border area.
- Generative fill / Fill: umbrella model or product term for masked filling, including inpainting
  and outpainting.
- Reframing: user-facing term for changing crop, viewpoint, or canvas. It can be a generative edit
  without hard pixel preservation. It becomes reliable outpainting only when backed by an expanded
  canvas plus mask.

UX rule: do not use the same option name for these two guarantees. `--reframe-padding` means a
fully generative zoom-out that may redraw the source. `--outpaint-padding` means canvas expansion
with edge-extended context and adaptive source blending, available only for capabilities that
advertise `supports_outpaint=true`. Exact source locking belongs to a native fill/inpaint route.

## Current code reality

- `src/mflux/task_inference.py` exposes `GenerationCapability`, `ModelCapabilities`, and
  `GenerationPlan`. Resolver APIs return public media-direction tasks and separate internal modes
  such as `latent-img2img`, `edit-reference`, `multi-reference`, `text-video`, and
  `first-frame-i2v`.
- `src/mflux/cli/mlx_gen.py` routes through `GenerationPlan.handler_id`, exposes
  `mlxgen capabilities`, and keeps `--task edit` only as a compatibility alias for image-to-image
  edit/reference mode.
- `GenerationCapability` now exposes `supports_reframe`. `resolve_generation_plan(...)`,
  `resolve_task(...)`, and `infer_task(...)` accept `has_reframe` and fail closed when the selected
  model does not advertise generative reframe support.
- The unified `mlxgen generate` router accepts `--reframe-padding`, requires exactly one source
  image, rejects explicit `--width`, `--height`, and `--canvas-policy`, validates the padding
  value, and forwards the padding request to the selected edit backend.
- `--reframe-padding` and `--outpaint-padding` are mutually exclusive. `--outpaint-padding`
  requires exactly one source image, rejects explicit `--width`, `--height`, and
  `--canvas-policy`, validates the padding value, and is rejected unless a capability advertises
  `supports_outpaint`.
- `src/mflux/utils/outpaint_util.py` implements the canvas outpaint helper used by the current
  FLUX.2 and Qwen Image Edit routes. It parses CSS-style padding, builds a larger
  edge-extended context canvas, records the source paste rectangle, creates a feathered
  preservation mask, and performs adaptive source blending after generation.
- `src/mflux/models/flux2/cli/flux2_edit_generate.py` and
  `src/mflux/models/qwen/cli/qwen_image_edit_generate.py` accept `--reframe-padding` and
  `--outpaint-padding`, build the expanded conditioning canvas in the backend, preserve side
  placement from the padding request, set exact output dimensions from that canvas, and save
  reframe/outpaint metadata.
- FLUX.2 edit plus Qwen Image Edit, Qwen Image Edit 2509, and Qwen Image Edit 2511 single-image
  capabilities now advertise `supports_outpaint=true` and `supports_reframe=true`. Other model
  families reject `--outpaint-padding` and `--reframe-padding` before loading weights.
- `tests/cli/test_mlx_gen_router.py` now covers FLUX.2 default edit/reference I2I, explicit
  latent I2I with `--image-strength`, multi-reference I2I, and mode/option rejection.
- Focused router tests cover FLUX.2 and Qwen Image Edit variant reframe routing, backend canvas
  creation, side placement, metadata, latent model rejection, and conflicting canvas-option
  rejection.
- `src/mflux/models/common/latent_creator/latent_creator.py` implements latent img2img by resizing
  the input image to the target size, encoding it, blending it with noise, and denoising.
- `src/mflux/models/flux2/variants/edit/flux2_klein_edit.py`,
  `src/mflux/models/qwen/variants/edit/qwen_image_edit.py`, and
  `src/mflux/models/fibo/variants/edit/fibo_edit.py` implement distinct image-conditioned edit
  paths. These are still image-to-image at the public API level.
- `src/mflux/models/flux/variants/fill/flux_fill.py` and
  `src/mflux/models/flux/variants/fill/mask_util.py` implement the true masked fill path for
  FLUX.1 Fill.
- `ModelConfig` still defines FLUX.1 `dev-fill`, `dev`, `schnell`, `kontext`, `redux`, `depth`,
  and control variants, but unified `mlxgen generate` currently routes the main supported image
  families through FLUX.2, Qwen, Z-Image, ERNIE, Bonsai, FIBO, Wan, and SeedVR2-specific commands.
- `src/mflux/utils/image_util.py` also has legacy `expand_image(...)` and
  `create_outpaint_mask_image(...)` helpers from the FLUX.1 Fill path.
- `src/mflux/cli/parser/parsers.py` has an `add_image_outpaint_arguments(...)` helper for legacy
  fill CLIs; unified outpaint routing uses the dedicated router option described above.
- `src/mflux/models/flux/README.md` documents outpainting with a helper script that is not present
  in the repository.
- `docs/api.md`, `docs/faq.md`, `docs/getting-started.md`, `docs/edit-capabilities.md`,
  `llms.txt`, and `llms-full.txt` document the current split: latent img2img is for whole-image
  variation/restyle with `--image-strength`; edit/reference and multi-reference I2I do not use
  `--image-strength`; generative reframe uses `--reframe-padding`; canvas outpaint uses
  `--outpaint-padding`.
- Qwen Image Edit, Qwen Image Edit 2509/2511, and FLUX.2 Klein Edit have current source/q8/q4
  reframe and canvas outpaint proof in `docs/assets/validation/reframe-outpaint-2026-06-08/`.
  FIBO Edit, ERNIE I2I, Z-Image I2I, base Qwen Image, Qwen Image 2512, and FLUX.1 Kontext should
  not be documented as reliable unified outpaint paths unless they get their own model-backed
  validation.

## Model capability assessment and priority

The implementation order should follow the model families MLX-Gen currently supports and validates
most actively. The table separates generative reframe from canvas outpaint because they have
different guarantees.

| Priority | Family/model | Current MLX-Gen status | Reframe stance | Canvas outpaint stance | Difficulty |
| --- | --- | --- | --- | --- | --- |
| 1 | FLUX.2 Klein Edit | Edit-reference and multi-reference I2I work and have recent validation artifacts. | Implemented and validated for 4B/9B source, q8, and q4 rows on cropped-starship zoom-out. | Implemented for 4B/9B source, q8, and q4 single-image edit routes with edge-extended canvas plus adaptive source blend. Native fill/inpaint masking remains separate. | Medium: routing/UX is straightforward, but validation must catch redraw/crop failures. |
| 2 | Qwen Image Edit / 2509 / 2511 | Edit and multi-reference work; 2511 parity was fixed and validated after the scheduler correction. | Implemented and validated for Qwen Image Edit, 2509, and 2511 source, q8, and q4 rows on cropped-starship zoom-out. | Implemented and validated for Qwen Image Edit, 2509, and 2511 source, q8, and q4 single-image edit routes with edge-extended canvas plus adaptive source blend. Native Qwen inpaint/outpaint parity remains future work. | Medium to high: strong model capability, but variant-specific prompt contracts matter. |
| 3 | Z-Image / Z-Image Turbo | Text-to-image and latent I2I routes exist; no edit-conditioned route. | Exploratory latent recompose/reframe only, with clear warnings. Use after FLUX.2/Qwen because it may restyle or lose identity. | Not supported. | Medium: easy to route, hard to trust. |
| 4 | ERNIE Image Turbo | Text-to-image and latent I2I route exist; Prompt Enhancer support exists. | Exploratory latent recompose/reframe only, useful if prompt adherence is strong enough. | Not supported. | Medium to high: promising instruction model, but no hard source-preservation contract. |
| 5 | FLUX.1 Fill (`dev-fill`, `black-forest-labs/FLUX.1-Fill-dev`) | Fill model and mask path exist from mflux-derived code; unified route is not current product focus. | Not the first reframe target. | Candidate native fill/inpaint backend once FLUX.1 support is deliberately revalidated. | Low to medium technically, but lower priority because current effort has centered on FLUX.2/Qwen. |
| Deferred | FIBO base / FIBO Edit | Base FIBO T2I exists; FIBO Edit is unavailable through unified generation and parity work is deprioritized. | Only consider base FIBO for creating T2I source images. Do not use FIBO Edit for reframe/outpaint work. | Not supported. Link any future FIBO Edit work to items 0024 and 0027. | High / deferred. |
| Not applicable | SeedVR2 | Restoration/upscale, not canvas generation. | Could clean a generated result after the fact, but does not invent prompt-conditioned borders. | Not supported. | Not applicable. |
| Not applicable | Wan T2V/I2V | Video generation, not still-image editing. | Out of scope for still-image reframe. | Not supported. | Not applicable. |

## Architecture decision

Decision question: how should MLX-Gen expose reframe/outpaint without overclaiming support across
image models?

Alternatives considered:

| Alternative | Steelman | Failure mode |
| --- | --- | --- |
| Mask-first FLUX.1 Fill | It is the strictest outpainting contract and preserves original pixels through a mask. | It makes FLUX.1 the first-class path even though recent validation and package work are focused on FLUX.2/Qwen. It also leaves common "zoom out with this edit model" workflows unmodeled. |
| Edit-first generative reframe | It starts with FLUX.2 and Qwen, the families users are already validating for image editing, and supports the common "extend background" and "zoom out" requests. | It may redraw the source or invent missing object parts. Poor naming would mislead users. |
| Latent-only recompose | It would reuse Z-Image and ERNIE without new edit adapters. | It has the weakest preservation guarantees and can become ordinary restyling rather than reframing. |
| Split contract | Use `reframe` for generative edit/latent workflows, and reserve `outpaint` for mask/canvas workflows. | More documentation and capability metadata are required, but the behavior is honest and reversible. |

Decision: use the split contract. Implement and validate generative reframe first for FLUX.2, then
Qwen, with Z-Image and ERNIE as lower-confidence latent recompose candidates. Implement canvas
outpaint only where a model-specific edit route can be validated with an expanded canvas and
adaptive source blending. Keep native fill/inpaint outpaint separate until a dedicated fill/mask
backend is deliberately validated for exact source locking.

Evidence that would change this decision:

- a proven FLUX.2 or Qwen native fill/inpaint outpaint pipeline lands locally;
- FLUX.1 Fill becomes a first-class supported family again with current model-backed smoke;
- generative reframe validation fails consistently for FLUX.2 and Qwen on the two canonical source
  cases below.

## Problem

Users still see multiple workflow words inside image-to-image:

- `image-to-image`
- `edit`
- `img2img`
- `reframe`
- `outpaint`

The public-task split is now clean, but reframe/outpaint still needs a first-class mode and command
surface. A request like "extend the image to the left" should not require users to manually create
an expanded canvas and mask before calling a lower-level fill script.

The larger UX problem is that users often say "outpaint" when they mean two different things:

- "make the canvas larger and continue the background" when the main object is already fully
  visible;
- "zoom out and reveal the whole object" when the source is a close crop and the missing object
  parts must be inferred.

The first is the target for canvas outpaint. The second is often better handled as generative
reframe because missing object geometry is not present in the source pixels and may need to be
inferred rather than only blended at the border.

## What we want to do

Make reframe/outpaint a first-class image-to-image workflow with honest guarantees:

1. Add a generative reframe route for edit-capable models, starting with FLUX.2 and Qwen.
2. Add a canvas outpaint route for edit-capable models only when validation proves useful
   canvas-guided expansion and acceptable border generation.
3. Keep native fill/inpaint outpaint as a separate future route that only becomes available when a
   dedicated backend is validated.
4. Teach users which guarantee they selected before model loading starts.

## Why

This improves user ergonomics and reduces false mental models. It also keeps routing aligned with
ADR 0002: MLX-Gen should infer only when the model/input contract is explicit and fail closed
otherwise. It should not silently swap model families or pretend that a generative reframe has the
same preservation guarantee as canvas outpaint.

## Requirements

- Extend the capability/plan contract with separate internal modes or workflow flags for:
  - generative reframe, where the model may redraw source content;
  - canvas-guided outpaint, where MLX-Gen uses an expanded conditioning canvas and adaptive source
    blending.
- Keep `--image-strength` latent-img2img-only; reject it for fill/outpaint modes unless a backend
  explicitly implements equivalent semantics.
- Add first-class generative reframe UX for FLUX.2 and Qwen first:
  - accept one source image;
  - accept a target canvas or padding request;
  - force an explicit reframe mode rather than silently treating mismatched dimensions as outpaint;
  - route to edit-reference I2I for FLUX.2/Qwen edit models;
  - save metadata for source path, requested target/padding, resolved dimensions, model family,
    mode, and the fact that the operation was generative.
- Add exploratory latent reframe support only after FLUX.2/Qwen validation:
  - Z-Image and ERNIE must be labeled lower-confidence because they do not have edit-conditioned
    source preservation;
  - if validation shows consistent identity loss, keep them unsupported for reframe.
- Add first-class canvas outpaint UX only for validated edit backends:
  - accept one source image;
  - accept CSS-style padding such as `0,25%,0,25%`;
  - create an expanded canvas with the source pasted at `(left, top)` and the new area initialized
    from edge-extended source context;
  - run the selected edit backend on the expanded canvas;
  - apply adaptive source blending only when the generated source window still matches the source;
  - save useful metadata for source path, padding, expanded dimensions, paste rectangle, and mode.
- Fail closed for outpaint requests unless a model has proven canvas-outpaint or native-fill
  support.
- FIBO can only be used to create source T2I images in this item. Any FIBO Edit route remains
  deferred to items 0024 and 0027.

## Suggested implementation

1. Extend `GenerationCapability`/`GenerationPlan` with an internal mode or workflow flag for
   `generative-reframe`. Do not mark it as `supports_outpaint`.
2. Add CLI/API shape for generative reframe:
   - preferred options: `--reframe-padding` and/or `--reframe-width` / `--reframe-height`;
   - reject `--reframe-*` unless the selected capability supports generative reframe;
   - ensure `--outpaint-padding` still means mask/canvas outpaint only.
3. Route FLUX.2 and Qwen Image Edit first:
   - FLUX.2: use `flux2.edit` with edit-reference mode;
   - Qwen Image Edit, Qwen Image Edit 2509, and Qwen Image Edit 2511: use `qwen.edit` for
     single-reference edit-reference mode;
   - require explicit prompts that say whether the source is fully visible or cropped.
4. Add validation assets before claiming support:
   - source A: fully isolated object, where reframe extends background;
   - source B: close-cropped object, where reframe must infer the missing object shape.
5. Add Z-Image and ERNIE only if validation shows the latent I2I path preserves enough identity to
   be useful. Otherwise keep them unsupported for reframe despite having generic I2I.
6. Add canvas outpaint for validated edit routes:
   - build the expanded canvas and feathered preservation mask with `OutpaintUtil`;
   - route only FLUX.2/Qwen Image Edit capabilities that advertise `supports_outpaint`;
   - reject multiple images and explicit canvas sizing options.
7. Add native fill/inpaint outpaint later:
   - first validate whether FLUX.1 Fill should be included in the current product surface;
   - if yes, add a dedicated fill-outpaint adapter using the existing fill utilities and
     `BoxValues`;
   - if a FLUX.2/Qwen fill or inpaint backend becomes available, prefer it over adding broad
     FLUX.1 surface area.
8. Update completions, model cards, docs, and troubleshooting around reframe versus outpaint.

## Scope

- First-class generative reframe command/API for FLUX.2 Klein 4B/9B and Qwen Image Edit original,
  2509, and 2511.
- First-class canvas outpaint command/API for FLUX.2 Klein 4B/9B and Qwen Image Edit original,
  2509, and 2511 capabilities after validation.
- Explicit unsupported or exploratory status for Z-Image and ERNIE until validation proves useful
  source preservation.
- Separate future native fill/inpaint outpaint only for a validated fill/mask backend.
- Tests and docs that distinguish reframe, latent I2I, edit-reference I2I, canvas outpaint, and
  native fill/inpaint outpaint.

## Non-goals

- Do not claim generative reframe preserves original pixels exactly.
- Do not claim FLUX.2, Qwen, Z-Image, ERNIE, FIBO, or Kontext are reliable outpainting paths unless
  they pass model-backed validation for that exact route and docs state the preservation guarantee
  precisely.
- Do not implement native Qwen inpaint/outpaint parity in this item unless proposed item 0008 is
  promoted or a small, specific Qwen fill/inpaint backend becomes available.
- Do not remove legacy model-specific CLIs in the same change.
- Do not silently choose another model family to satisfy reframe or outpaint.
- Do not run large image models just to validate routing; use a minimal model-backed smoke only
  where ADR 0001 requires it.
- Do not use FIBO Edit for this item. Link FIBO Edit questions to items 0024 and 0027.

## Dependencies and related tasks

- [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md) for model-backed smoke
  validation before claiming a route works.
- [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md) for fail-closed routing.
- [Completed item 0018](../completed/0018_taskless_generation_routing.md) for current taskless
  routing context.
- [Completed item 0020](../completed/0020_generation_capability_contract.md) for the public
  task/internal mode/capability planning baseline.
- [Planned item 0008](0008_qwen_edit_parity_expansion.md) for future Qwen
  inpainting/outpainting parity.
- [Planned item 0024](0024_fibo_edit_unified_i2i_validation.md) and
  [planned item 0027](0027_fibo_edit_diffusers_parity_release_quality.md) for deferred FIBO Edit
  parity and validation.
- `src/mflux/task_inference.py`
- `src/mflux/cli/mlx_gen.py`
- `src/mflux/models/flux2/variants/edit/`
- `src/mflux/models/qwen/variants/edit/`
- `src/mflux/models/z_image/`
- `src/mflux/models/ernie_image/`
- `src/mflux/models/flux/cli/flux_generate_fill.py`
- `src/mflux/models/flux/variants/fill/`
- `src/mflux/utils/image_util.py`
- `src/mflux/utils/box_values.py`

## Expected outcomes

- Generative reframe has a working command/API for FLUX.2 Klein 4B/9B and Qwen Image Edit
  original, 2509, and 2511 capabilities.
- Canvas outpaint has a working command/API for FLUX.2 Klein 4B/9B and Qwen Image Edit original,
  2509, and 2511 routes with model-backed proof.
- Native fill/inpaint outpaint remains separate and unavailable unless a validated fill/mask
  backend is wired.
- Docs accurately distinguish:
  - latent img2img;
  - edit-conditioned I2I;
  - multi-reference I2I;
  - generative reframe;
  - canvas-guided outpaint with adaptive source blending;
  - inpainting;
  - native fill/inpaint outpainting / image expansion.

UX expected outcome: a user can tell before running whether the operation may redraw the source
(`reframe`), is canvas-guided with adaptive blending (`outpaint`), or requires a future native
fill/inpaint route for exact source locking.

## Current release boundary

As of 2026-06-08, this item is partially implemented and remains planned for lower-confidence
latent candidates and native fill/inpaint work. The generative reframe and canvas outpaint
contracts are implemented for FLUX.2 Klein 4B/9B and Qwen Image Edit original, 2509, and 2511
single-image edit capabilities. Source, q8, and q4 rows passed the cropped-starship
`reframe_outpaint_2026_06_08` profile. The remaining work is:

1. evaluate Z-Image and ERNIE as lower-confidence latent reframe candidates only if they can
   preserve source identity on a dedicated profile;
2. decide whether any non-FLUX.2/Qwen edit route should get canvas outpaint support;
3. keep native fill/inpaint outpaint separate until FLUX.1 Fill or another fill/mask backend is
   explicitly revalidated.

## Validation

- Fast resolver tests:
  - FLUX.2 edit-capable models expose generative reframe support;
  - Qwen edit models expose generative reframe support only where the exact model route is validated;
  - Z-Image and ERNIE expose reframe only after validation, otherwise reject `--reframe-*`;
  - non-supported models reject `--outpaint-padding` unless they implement validated canvas
    outpaint or native fill/inpaint outpaint;
  - `--image-strength` remains latent-img2img-only and is rejected for edit-reference reframe.
- Fast router tests:
  - FLUX.2 plus `--reframe-padding` routes to the FLUX.2 edit backend, not the latent backend;
  - Qwen Image Edit, 2509, and 2511 plus `--reframe-padding` route to the Qwen edit backend;
  - base Qwen Image, Z-Image, and ERNIE reject or clearly label reframe until their validation
    status is promoted;
  - `--outpaint-padding` routes for FLUX.2/Qwen Image Edit capabilities and fails closed for
    Z-Image/ERNIE/base Qwen/Qwen Image 2512/FIBO unless a validated outpaint capability is added.
- Utility tests:
  - percent and pixel reframe padding resolve expected target dimensions;
  - no-op, negative, or malformed padding fails before model load;
  - `--reframe-padding` and `--outpaint-padding` cannot be used together.
- Model-backed smoke:
  - create two T2I source images:
    - source A: fully visible isolated object with room to extend background;
    - source B: close-cropped starship where part of the ship is cut by the frame and the desired
      result shows the full starship.
  - FLUX.2 must pass both source A background extension and source B zoom-out/reveal before it is
    documented as supported.
  - Qwen must run the same two-source profile before being documented as supported.
  - Z-Image and ERNIE must run the same profile and should remain unsupported if they mostly
    restyle, crop, or lose object identity.
  - any canvas outpaint route must include generated canvas/mask evidence plus output proof.
  - any future native fill/inpaint outpaint route must include backend-specific mask evidence plus
    output proof.
- Current FLUX.2 smoke evidence:
  - source/output contact sheet:
    `docs/assets/validation/reframe-2026-06-07/flux2-reframe-contact-sheet.png`;
  - exact commands:
    `docs/assets/validation/reframe-2026-06-07/reframe-command-log.md`;
  - local validation outputs:
    `validation_outputs/reframe_2026_06_07/flux2_reframe_a_background_extension.png`
    (`736x320`) and
    `validation_outputs/reframe_2026_06_07/flux2_reframe_b_reveal_full_starship.png`
    (`864x368`).
- Current Qwen Image Edit 2511 q8 smoke evidence:
  - source/output contact sheet:
    `docs/assets/validation/reframe-2026-06-07/qwen2511-q8-reframe-contact-sheet.png`;
  - exact commands:
    `docs/assets/validation/reframe-2026-06-07/reframe-command-log.md`;
  - local validation outputs:
    `validation_outputs/reframe_2026_06_07/qwen2511_q8_reframe_a_background_extension.png`
    (`736x320`) and
    `validation_outputs/reframe_2026_06_07/qwen2511_q8_reframe_b_reveal_full_starship.png`
    (`864x368`).
- Current canvas outpaint smoke evidence:
  - source/output contact sheet:
    `docs/assets/validation/outpaint-2026-06-07/outpaint-contact-sheet.png`;
  - exact commands:
    `docs/assets/validation/outpaint-2026-06-07/outpaint-command-log.md`;
  - source canvas and mask assets:
    `docs/assets/validation/outpaint-2026-06-07/source-a-outpaint-canvas.png`,
    `docs/assets/validation/outpaint-2026-06-07/source-a-outpaint-mask.png`,
    `docs/assets/validation/outpaint-2026-06-07/source-b-outpaint-canvas.png`, and
    `docs/assets/validation/outpaint-2026-06-07/source-b-outpaint-mask.png`;
  - source-interior preservation check:
    `docs/assets/validation/outpaint-2026-06-07/outpaint-preservation-check.json`;
  - local validation outputs:
    `validation_outputs/outpaint_2026_06_07/flux2_outpaint_a_background.png`,
    `validation_outputs/outpaint_2026_06_07/flux2_outpaint_b_cropped_starship.png`,
    `validation_outputs/outpaint_2026_06_07/qwen2511_q8_outpaint_a_background.png`, and
    `validation_outputs/outpaint_2026_06_07/qwen2511_q8_outpaint_b_cropped_starship.png`.
- Docs checks:
  - README, `docs/api.md`, `docs/getting-started.md`, `docs/faq.md`, `docs/troubleshooting.md`,
    completions, and generated model-card examples teach reframe as generative and outpaint as
    canvas-guided with adaptive blending, not native fill/inpaint masking.

## Progress checklist

- [x] Add generative reframe capability metadata and Python/API semantics.
- [x] Add CLI routing and parser support for `--reframe-padding`.
- [x] Add shell completion support for reframe/outpaint options.
- [x] Add dedicated reframe metadata fields.
- [x] Validate FLUX.2 on the two canonical T2I source cases.
- [x] Validate Qwen Image Edit 2511 q8 on the same two canonical source cases.
- [x] Validate Qwen Image Edit, Qwen Image Edit 2509, and Qwen Image Edit 2511 source/q8/q4 on the
      cropped-starship reframe profile.
- [ ] Decide whether Z-Image and ERNIE should be promoted or explicitly rejected for reframe.
- [x] Add canvas outpaint capability metadata and CLI routing for FLUX.2/Qwen Image Edit routes.
- [x] Add canvas/mask utility validation for `OutpaintUtil`.
- [x] Validate FLUX.2 canvas outpaint on the two canonical source cases.
- [x] Validate Qwen Image Edit 2511 q8 canvas outpaint on the same two canonical source cases.
- [x] Validate Qwen Image Edit, Qwen Image Edit 2509, and Qwen Image Edit 2511 source/q8/q4 on the
      cropped-starship outpaint profile.
- [x] Keep `--outpaint-padding` rejected for unsupported families and conflicting options.
- [x] Run focused tests.
- [x] Run model-backed FLUX.2 reframe smoke and preserve source/output proof.
- [x] Update user docs for current reframe/outpaint behavior.
- [ ] Update generated model-card guidance if prepared-package cards should mention reframe.
- [ ] Decide whether ADR 0002 needs a task-taxonomy addendum before closure.

## Guidance for the implementing agent

Re-check the code first because routing has been changing quickly. Keep the public contract simple:
media direction is the task, implementation is a backend mode. Prefer explicit errors over silent
model swaps. Do not claim outpaint support for a model unless the implementation preserves an
expanded source canvas with a mask and the output has model-backed validation evidence. Do not use
FIBO Edit as a shortcut for this item; if FIBO becomes relevant again, resume items 0024 and 0027
first.
