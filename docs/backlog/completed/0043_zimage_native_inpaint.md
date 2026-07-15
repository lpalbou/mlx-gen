# Completed: Z-Image native inpaint

## Metadata

- Created: 2026-06-15
- Status: Completed
- Completed: 2026-06-21

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: No new ADR was required. The shipped path stays explicit, fail-closed, and inside
  the existing image-to-image contract.

## Outcome

Completed as the first narrow native inpaint route for the existing Z-Image family.

MLX-Gen now:

- exposes `z-image.inpaint` on the exact `AbstractFramework/z-image-turbo-8bit` row;
- accepts the generic `--image + --mask-path + --prompt` request shape through unified
  `mlxgen generate`;
- rejects `--mask-path` without `--image`;
- rejects `--image-strength` together with `--mask-path` instead of silently falling back to
  latent img2img;
- publishes an accepted same-prompt same-seed engine-thruster proof against the previous latent
  route.

## Closing code reality

- `src/mflux/models/z_image/variants/z_image.py` now implements native inpaint as a distinct
  masked route:
  - source image latents are encoded explicitly;
  - mask latents are built in latent space;
  - unmasked regions are preserved through per-step latent blending;
  - `image_strength` and `mask_path` stay mutually exclusive.
- `src/mflux/models/z_image/cli/z_image_turbo_generate.py` now exposes `--mask-path` and fails
  closed on invalid combinations. (Correction 2026-07-15: this item originally claimed both
  Z-Image CLIs gained `--mask-path`, but only the turbo command did; the non-turbo
  `z_image_generate.py` command gained it in completed item 0082 when the non-turbo route was
  opened.)
- `src/mflux/task_inference.py` now surfaces a distinct `z-image.inpaint` capability only on the
  exact Z-Image Turbo rows that were intentionally opened for this route.
- `tests/image_generation/test_masked_generation_routes.py`,
  `tests/cli/test_mlx_gen_router.py`, and `tests/test_task_inference.py` now lock the mask-route
  math, routing, and error contract.

## Validation

Focused automated validation:

- `tests/image_generation/test_masked_generation_routes.py`
- `tests/cli/test_mlx_gen_router.py`
- `tests/test_task_inference.py`
- `tests/arg_parser/test_cli_argparser.py`

Accepted published proof bundle:

- [native inpaint report](../../assets/validation/zimage-inpaint-2026-06-21/zimage_inpaint_report.md)
- [native inpaint command log](../../assets/validation/zimage-inpaint-2026-06-21/zimage_inpaint_command_log.md)
- [native inpaint stats](../../assets/validation/zimage-inpaint-2026-06-21/zimage_inpaint_stats_m5max.json)
- [native inpaint full contact sheet](../../assets/validation/zimage-inpaint-2026-06-21/zimage_inpaint_contact_sheet.png)
- [native inpaint crop sheet](../../assets/validation/zimage-inpaint-2026-06-21/zimage_inpaint_engine_crop_contact_sheet.png)
- [native inpaint output](../../assets/validation/zimage-inpaint-2026-06-21/zimage_inpaint_engine_q8.png)
- [latent baseline output](../../assets/validation/zimage-inpaint-2026-06-21/zimage_latent_engine_q8.png)

Accepted result on the public engine-thruster proof row:

- model: `AbstractFramework/z-image-turbo-8bit`
- latent baseline: same source, same prompt, same seed, `--image-strength 0.35`
- native inpaint: same source, same prompt, same seed, `--mask-path`
- latent baseline timing: `2.78s` generation, `5.99s` wall, `11.49 GB` max RSS
- native inpaint timing: `21.00s` generation, `26.86s` wall, `18.11 GB` max RSS

Maintenance update on the accepted route:

- the route now invalidates cached source/mask conditions when those files change in place;
- the accepted published output was refreshed after the runtime tune because the current native
  inpaint result is cleaner in the masked thruster region while keeping the same narrow public
  proof contract.

The public claim stays intentionally narrow: exact q8 Z-Image Turbo row, exact engine mask case,
same-prompt same-seed comparison against the old latent route.

## Related backlog items

- [0008 Qwen edit parity expansion](../completed/0008_qwen_edit_parity_expansion.md)
- [0045 Z-Image ControlNet follow-up](../proposed/0045_zimage_controlnet_followup.md)

## Follow-ups

- Keep `controlnet_inpaint` separate in [0045](../proposed/0045_zimage_controlnet_followup.md).
- If future Z-Image native inpaint proofs broaden beyond the engine case, publish them only after
  they are clearly good enough for the public validation surface.
