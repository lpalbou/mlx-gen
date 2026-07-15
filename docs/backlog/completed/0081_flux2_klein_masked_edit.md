# 0081 - FLUX.2 Klein Masked Edit / Inpaint

- Status: completed (2026-07-15)
- Scope: native masked edit (`--mask-path`) for all FLUX.2 Klein image routes (distilled
  4B/9B and base 4B/9B), ported from the upstream diffusers `Flux2KleinInpaintPipeline`
  (huggingface/diffusers PR #13050, closing issue #13005), including optional masked-area
  reference images on the backend command and Python API
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
  (model-backed smoke proof included below), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
  (mask/strength/outpaint combinations fail closed before weight load)
- Upstream confirmation: the Klein models support masked inpainting without new weights -
  diffusers merged `Flux2KleinInpaintPipeline` in April 2026 with clean-source conditioning
  tokens plus per-step latent compositing; ComfyUI workflows ship the same technique
- Validation: adversarial subagent review (design + code + parity attack; one major finding
  fixed - see below), focused test suites green (masked routes, router 193, task inference,
  python runtime; full fast band 633 passed), lint clean, and model-backed smoke runs on the
  exact cached q8 packages

## What shipped

1. Runtime `Flux2KleinInpaint` (`src/mflux/models/flux2/variants/edit/flux2_klein_inpaint.py`):
   full-strength denoise from seed noise with the clean source encoded once at generation
   size; the clean packed source latents ride along as conditioning tokens at reserved grid
   t-coordinate 10 (diffusers parity), optional extra reference images at t=20+; after every
   scheduler step, unmasked latents are re-composited from the source re-noised to the next
   sigma (clean at the final step). `image_strength` is rejected (mask and latent strength
   stay separate routes per the repo-wide contract). Guidance resolves per model class:
   base Klein defaults to 4.0 with true CFG (blank negative prompt), distilled Klein stays
   at 1.0 and rejects higher values.
2. Helpers (`flux2_klein_edit_helpers.py`): `prepare_inpaint_mask` binarizes the mask at
   pixel resolution (LANCZOS resize, >= 0.5 threshold) and then bilinear-interpolates it
   directly onto the packed latent grid; `MaskUtil.interpolate_bilinear` implements torch
   `F.interpolate(mode="bilinear", align_corners=False)` semantics in numpy (verified
   against torch across 32->2, 64->4, 48->3, 100->7 and upsampling shapes by the
   adversarial reviewer). `prepare_inpaint_source_conditioning` builds the source
   conditioning ids without re-encoding; `prepare_reference_image_conditioning` gained
   `t_coord_start` so masked-area references land at t=20+ without colliding with the
   source tokens; the outpaint route's preserved-latent blend moved into the shared helper
   (`preserved_source_latents`, byte-identical behavior); klein-base detection and guidance
   validation centralized (`is_base_model`, `validate_guidance`, `default_guidance`).
3. Capability layer: new `flux2.inpaint` capability (`supports_mask=true`, one source
   image) on trusted FLUX.2 identities, distilled and base; `mlxgen generate --image ...
   --mask-path ...` routes there, and `--mask-path` with two or more images or with
   `--image-strength` fails closed. Python runtime id `flux2.klein-inpaint`.
4. CLI: `mflux-generate-flux2-edit` accepts `--mask-path/--masked-image-path` (metadata
   backfill via `-C` included for free through the shared parser), rejects mask +
   reframe/outpaint padding, resolves the guidance default per model class, and passes
   `image_paths[0]` as source with `image_paths[1:]` as masked-area references
   (diffusers `image_reference` parity surface).
5. Tests: mask binarize/soft-boundary math, torch-parity golden values for the bilinear
   helper, blend preservation across steps, source/reference t-coordinate reservation,
   full masked-loop progress contract (`image-to-image` task events, conditioning sequence
   length, decode shape), image-strength rejection, router routing/rejection matrix,
   backend forwarding (distilled defaults and base CFG default), capability gating, and
   the runtime definition.

## Adversarial review

One adversarial subagent attacked the reference parity, mask math, contracts, and repo
conventions after implementation. Verdict: no critical findings; one major finding fixed
before the smoke run (the Python API guidance default silently disabled CFG for base
models; now `guidance=None` resolves per model class in one place). Minor findings fixed:
LANCZOS pixel-resolution mask resize per the documented ported-surface policy, removal of a
dummy-tensor ids construction, centralized klein-base guidance helpers instead of a
cross-class private call. The reviewer numerically verified `interpolate_bilinear` against
`torch.nn.functional.interpolate` (max deviation 4.1e-6) and confirmed the outpaint refactor
is behavior-preserving.

## Model-backed smoke proof (wiring validation, not published visual QA)

Preserved in `validation_outputs/flux2-klein-inpaint-smoke/` (local, uncommitted):

- `AbstractFramework/flux.2-klein-4b-8bit`, 4 steps, guidance 1, seed 42, 720x400 from the
  717x403 `tests/resources/glasses.jpg` source, central rectangle mask: the masked lens
  region takes the requested dark-blue tint; the repaint stops exactly at the mask boundary.
  Outside-mask mean abs pixel diff 2.37/255 (inside 55.93/255), 0.78% of outside pixels
  above 20/255 (mask-edge transition band). 28 s denoise wall time.
- `AbstractFramework/flux.2-klein-base-4b-8bit`, 8 steps, default guidance 4 (true CFG),
  same mask: dark-green tint variant, same preservation behavior. 62 s denoise.
- Reference-conditioned case (backend command): source + `tests/resources/shirt.jpg` as
  masked-area reference, "fill the masked lens area with the plaid fabric pattern": the
  masked region takes the reference's blue plaid pattern. 21 s denoise.

Follow-ups deliberately left open:

- publish a visual-QA proof bundle and validation-registry row before claiming a validated
  masked-edit row for any exact Klein package (current claim is smoke/wiring grade);
- consider a latent-space blur/feather option once upstream conventions settle
  (diffusers exposes `mask_processor.blur`; MLX-Gen currently expects pre-feathered masks).
