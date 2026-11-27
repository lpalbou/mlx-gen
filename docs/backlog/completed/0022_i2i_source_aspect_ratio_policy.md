# Completed: I2I source aspect-ratio policy

## Metadata
- Created: 2026-06-04
- Status: Completed
- Completed: 2026-06-04

## ADR status
- Governing ADRs: [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None

## Context

Wan image-to-video now resolves the generation canvas from the source image ratio before
conditioning the model, which prevents portrait or square sources from being silently stretched
into a mismatched requested video canvas. Ordinary image-to-image has the same user-facing risk:
the model may resize a source image into an explicit `--width` and `--height` even when that canvas
does not match the source ratio.

This item covers ordinary image-to-image geometry only. True outpainting/reframing remains tracked
by [0019](../planned/0019_first_class_i2i_modes_and_outpaint_reframe.md).

## Current code reality

- `src/mflux/utils/dimension_resolver.py` resolves `auto`/scale factors against a reference image
  but returns explicit integer `width`/`height` as-is.
- `src/mflux/models/common/config/config.py` floors image dimensions to 16-pixel multiples and
  resolves `None` dimensions from a reference image, but it has no source-aspect canvas policy.
- `src/mflux/models/common/latent_creator/latent_creator.py` encodes latent img2img sources by
  resizing the source image to `config.width` by `config.height`.
- `src/mflux/models/flux2/variants/edit/flux2_klein_edit.py` and
  `src/mflux/models/qwen/variants/edit/qwen_image_edit.py` use one or more reference images but do
  not centrally define how output canvas ratio should relate to the primary source image.
- CLI image routes that use `supports_dimension_scale_factor=True` default to `auto`, so the common
  CLI path often preserves the source size by default, but explicit mismatched dimensions can still
  stretch/recompose ordinary I2I.
- Direct Python calls usually have `width=1024` and `height=1024` defaults, so a rectangular source
  can be unintentionally forced into a square output unless the caller knows to pass matching
  dimensions.

## Problem

Ordinary I2I does not have a consistent geometry contract. Users can ask for a source-conditioned
edit or variation, pass a rectangular/square source image, and get a generated canvas that silently
uses a different aspect ratio. This makes visual validation confusing and can turn a style/edit
request into an unintended reframing/remix.

## What we want to do

Add a shared source-aspect canvas resolver for ordinary I2I. When an I2I mode has a primary source
image, `width` and `height` should be treated as a size target by default, while the final output
canvas preserves the primary source image aspect ratio and model multiple.

## Why

This gives users and application integrations predictable geometry. It also keeps ordinary I2I
separate from explicit canvas-changing workflows such as outpainting, reframing, fill/inpaint, and
upscale.

## Requirements

- Preserve the primary source image aspect ratio for latent img2img, edit/reference I2I, and
  multi-reference I2I.
- Use the first image as the geometry anchor for multi-reference I2I.
- Treat explicit `width` and `height` as an approximate size target, not as permission to stretch
  the source into a different ratio.
- Keep output dimensions on the model's required multiple, starting with 16-pixel multiples for
  image generation routes that use the shared `Config`.
- Keep text-to-image behavior unchanged.
- Do not change masked fill/outpaint/upscale semantics in this item.
- Preserve clean early failures for unsupported task/mode/options.

## Suggested implementation

- Add a shared image canvas resolver near `DimensionResolver`.
- Add an opt-in `preserve_image_aspect_ratio` flag to `Config`.
- Enable the flag in latent img2img and edit/reference model paths that use an ordinary source
  image, but leave fill/outpaint/upscale paths exact-canvas.
- Update Qwen edit dimension computation to use the same resolver with the first input image.
- Adjust Python defaults where needed so direct `generate_image(image_path=...)` calls can preserve
  source geometry without requiring callers to pass source dimensions manually.

## Scope

- Shared resolver implementation.
- Focused unit tests for canvas resolution and Config behavior.
- Router or model-path tests proving ordinary I2I forwards to the expected backend and preserves
  dimensions before generation.
- At least one low-cost real image-generation validation asset with command, prompt, source, and
  output paths for manual inspection.
- Core documentation updates.

## Non-goals

- Do not implement first-class outpainting/reframing; keep that in item 0019.
- Do not promise ordinary I2I preserves original pixels in place.
- Do not change video canvas policy.
- Do not run heavyweight model validations unless a lightweight local model cannot prove the
  contract.

## Dependencies and related tasks

- [0018](0018_taskless_generation_routing.md): taskless generation routing.
- [0020](0020_generation_capability_contract.md): capability contract and I2I modes.
- [0021](0021_wan_i2v_source_aspect_ratio.md): analogous I2V source-ratio behavior.
- [0019](../planned/0019_first_class_i2i_modes_and_outpaint_reframe.md): future outpaint/reframe UX.

## Expected outcomes

- Ordinary I2I no longer silently stretches a source image into a mismatched output ratio.
- CLI and Python direct calls use the same source-aspect policy for ordinary I2I routes.
- Users can reproduce a small validation image and inspect source/output dimensions.
- FAQ/API/getting-started/LLM docs explain the behavior without claiming outpainting support.

## Validation

- Unit tests for source-aspect canvas resolution across square, portrait, and landscape sources.
- Config tests showing I2I preserves source ratio and T2I remains unchanged.
- Focused model-path tests for at least Flux2 latent/edit and Qwen edit dimension behavior.
- One real low-cost I2I generation command with source, prompt, output, and image-size proof.
- `uv run ruff check .`
- Focused pytest target for new tests, plus no doc whitespace errors.

## Progress checklist
- [x] Create shared resolver and Config integration.
- [x] Enable ordinary latent/edit/multi-reference I2I paths.
- [x] Add focused tests.
- [x] Generate validation source/output assets and record exact commands.
- [x] Update core documentation.
- [x] Move this item to completed with completion evidence.

## Implementation notes

Re-check current code before editing. Keep the implementation explicit and mode-aware: ordinary I2I
preserves aspect by default, while fill/outpaint/upscale remain exact-canvas workflows.

## Design checks

- API contract review: keep public tasks as media directions and expose canvas behavior through
  capability fields. The default for ordinary I2I should be `source-aspect`; `exact-resize` must be
  an explicit opt-in.
- Code-path review: centralize ordinary I2I geometry in `Config` and the shared dimension resolver.
  Backend CLIs should pass `auto` and explicit dimensions through instead of resolving them before
  the model config can apply source-aspect policy.
- Validation review: prove behavior with a mismatched target canvas, image metadata, and a
  side-by-side exact-resize comparison. Keep edit/reference and multi-reference route behavior in
  focused unit/router tests when no local prepared edit model is available.
- Documentation review: state the current contract neutrally. Do not claim ordinary I2I preserves
  original pixels in place or provides outpainting/reframing.

## Completion report

Implemented shared ordinary I2I canvas policies:

- `source-aspect`: default for latent img2img, edit/reference I2I, and multi-reference I2I.
- `exact-resize`: explicit opt-in for exact requested dimensions.

The capability contract now exposes `canvas_policies`, `default_canvas_policy`,
`primary_image_index`, and `dimension_multiple`. Generated image metadata records final dimensions,
requested dimensions, source-image dimensions, and `canvas_policy`.

The implementation routes CLI and Python direct calls through the same config resolver. A final
audit removed backend CLI pre-resolution from ordinary image routes so `auto` dimensions remain
visible to `Config`.

## Validation evidence

Focused tests:

```sh
uv run pytest tests/test_task_inference.py tests/cli/test_mlx_gen_router.py tests/cli/test_backend_canvas_policy.py tests/common/config/test_config_dimensions.py tests/utils/test_dimension_resolver.py tests/arg_parser/test_cli_argparser.py tests/callbacks/test_progress_callbacks.py tests/image_generation/test_qwen_edit_dimensions.py tests/metadata/test_generated_image.py tests/ernie_image/test_mistral3_text_encoder.py::test_ernie_img2img_latent_helper_patchifies_and_normalizes -q
```

Result: `172 passed in 1.11s`.

Static check:

```sh
uv run ruff check .
```

Result: `All checks passed!`.

Post-fix local validation source:

```text
validation_outputs/i2i_matrix_2026_06_04/source_spaceship_432x240.png
```

Source size: `432x240`.

Base FIBO was removed from ordinary I2I capability because the prior generated proof was not a
valid I2I/edit quality validation. The route now fails before model load:

```sh
uv run mlxgen generate --model ./models/fibo-8bit --image validation_outputs/i2i_matrix_2026_06_04/source_spaceship_432x240.png --i2i-mode latent --image-strength 0.4 --width 320 --height 320 --prompt '{"short_description":"A graphite pencil sketch of the same spaceship in snow."}' --steps 2 --guidance 1 --seed 9107 --metadata --replace --output validation_outputs/i2i_matrix_2026_06_04/final_after_fix/fibo_base_should_reject.png
```

Post-fix generated compatibility outputs:

- `validation_outputs/i2i_matrix_2026_06_04/final_after_fix/flux2_9b_8bit_edit_pencil.png`
- `validation_outputs/i2i_matrix_2026_06_04/final_after_fix/flux2_9b_8bit_latent_pencil_strength035.png`
- `validation_outputs/i2i_matrix_2026_06_04/final_after_fix/flux2_9b_8bit_multi_reference_sketch_crash.png`
- `validation_outputs/i2i_matrix_2026_06_04/final_after_fix/flux2_4b_4bit_edit_pencil.png`
- `validation_outputs/i2i_matrix_2026_06_04/final_after_fix/flux2_9b_4bit_edit_pencil.png`
- `validation_outputs/i2i_matrix_2026_06_04/final_after_fix/qwen_edit_2511_4bit_crash.png`
- `validation_outputs/i2i_matrix_2026_06_04/final_after_fix/qwen_image_2512_8bit_latent_bluehour_strength05.png`
- `validation_outputs/i2i_matrix_2026_06_04/final_after_fix/qwen_image_2512_4bit_latent_bluehour_strength05.png`
- `validation_outputs/i2i_matrix_2026_06_04/final_after_fix/z_image_turbo_8bit_latent_pencil_strength05.png`
- `validation_outputs/i2i_matrix_2026_06_04/final_after_fix/z_image_turbo_4bit_latent_pencil_strength05.png`
- `validation_outputs/i2i_matrix_2026_06_04/final_after_fix/ernie_turbo_8bit_latent_pencil_strength035.png`
- `validation_outputs/i2i_matrix_2026_06_04/final_after_fix/ernie_turbo_4bit_latent_pencil_strength035.png`

All generated outputs in this matrix resolve to `432x240`, record `canvas_policy:
source-aspect`, and write metadata sidecars as JSON objects.

Contact sheet:

```text
validation_outputs/i2i_matrix_2026_06_04/final_after_fix/i2i_compatibility_contact_sheet_after_fix.png
```

Local cache checks that failed before model load because the cached package was incomplete:
FIBO Edit, Qwen Edit 2509 q4/q8, Qwen Edit 2511 q8, FLUX.2 4B q8, FLUX.2 base 9B q8, and
non-Turbo Z-Image q4/q8. Base Qwen with one image and no `--image-strength` now fails closed
instead of silently swapping to an edit checkpoint.

## Documentation updated

- `docs/api.md`
- `docs/faq.md`
- `docs/getting-started.md`
- `docs/python-integration.md`
- `docs/troubleshooting.md`
- `docs/README.md`
- `llms.txt`
- `llms-full.txt`

## Residual work

First-class reframe and outpaint follow-up work is tracked in
[0019](../planned/0019_first_class_i2i_modes_and_outpaint_reframe.md).
