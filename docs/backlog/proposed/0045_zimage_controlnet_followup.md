# Proposed: Z-Image ControlNet follow-up

## Metadata

- Created: 2026-06-18
- Updated: 2026-07-15 (upstream reference now exists; port sized)
- Status: Proposed
- Completed: N/A

## 2026-07-15 refresh: upstream reference and port sizing

Diffusers now ships first-party reference code that did not exist when this item was created:
`ZImageControlNetModel` (`models/controlnets/controlnet_z_image.py`, ~860 lines) plus
`ZImageControlNetPipeline` and `ZImageControlNetInpaintPipeline`. The documented weights are the
single-file `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0` releases (the pipeline
hard-rejects the v1 checkpoint via a `control_in_dim` check - only Union 2.0/2.1 works). Honest
port sizing from the reference:

- control transformer blocks at configurable `control_layers_places` over the 30-layer host,
  a separate control patch embedder, and control noise-refiner layers in three structural
  variants keyed on `add_control_noise_refiner` (single-file config detection required);
- `from_transformer` shares nine module groups with the base transformer (t_embedder,
  x_embedder, cap_embedder, rope_embedder, refiners, pad tokens), so the MLX runtime needs
  weight-sharing plumbing that does not exist today;
- the MLX `ZImageTransformer` forward needs per-block residual injection
  (`controlnet_block_samples`);
- the control-inpaint input is `concat([VAE(control_image) 16ch, inverted mask 1ch,
  VAE(source) 16ch]) = 33ch`, not a simple masked-image concat;
- plus sidecar capability row and router injection (Qwen `control_model` pattern), CFG
  batching of control features, a multi-GB single-file download, and a torch parity harness.

Verdict recorded 2026-07-15: a full multi-day port with its own ADR-0001 proof obligations;
kept proposed rather than bundled into the masked-edit expansion (completed item 0082).

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None if this remains a narrow follow-up after native Z-Image inpaint. A small ADR
  may be warranted only if MLX-Gen decides to share one explicit control-image contract across
  several image families.

## Context

Completed item [0043](../completed/0043_zimage_native_inpaint.md) now covers the first narrow
native Z-Image inpaint route with published proof assets. Public Z-Image and Z-Image-Turbo
ControlNet weights also exist, including inpaint-related control variants. That makes a clean
follow-up worth preserving without reopening 0043 or broadening its accepted public claim.

## Current code reality

- MLX-Gen already supports Z-Image and Z-Image-Turbo text generation plus latent img2img.
- Completed item 0043 owns native Z-Image Turbo inpaint; completed item 0082 (2026-07-15)
  extended native `z-image.inpaint` to trusted non-turbo rows and created the canonical
  `docs/masked-editing.md` page. Masked editing without structure control is therefore covered.
- MLX-Gen has the full sidecar pattern to copy from Qwen: `control_model` on the capability,
  router injection of the exact sidecar spec, `--controlnet-model` conflict validation, and
  `QwenImageControlNet` as the runtime shape (`src/mflux/models/qwen/variants/controlnet/`).
- The MLX `ZImageTransformer` (`src/mflux/models/z_image/model/z_image_transformer/`) has no
  per-block residual-injection hook today.
- There is no first-class Z-Image control-image route today.

## Problem or opportunity

Once native Z-Image inpaint lands, users will reasonably ask for the same stronger
structure-constrained editing/control pattern that is becoming available in Qwen. Public
ControlNet weights make that a real possibility, but it should remain a separate decision from
native inpaint because the user contract, sidecar requirements, and validation burden are larger.

## Proposed direction

Track Z-Image ControlNet as a follow-up rather than bundling it into 0043:

1. Finish native Z-Image inpaint first. (Done: 0043 turbo, 0082 non-turbo.)
2. Audit the public ControlNet weight families for `Z-Image` and `Z-Image-Turbo`.
3. Decide whether the first MLX-Gen route should be:
   - structured control,
   - control-inpaint, or
   - one union-style route with explicit control-image metadata.
4. Keep route selection fail-closed and preserve exact model-plus-sidecar identity in metadata and
   capabilities.

## Session plan (2026-07-15, for a dedicated multi-day session)

Reference implementations to port from (local diffusers checkout):

- `src/diffusers/models/controlnets/controlnet_z_image.py` (`ZImageControlNetModel`, ~860 lines)
- `src/diffusers/pipelines/z_image/pipeline_z_image_controlnet.py` (structured control)
- `src/diffusers/pipelines/z_image/pipeline_z_image_controlnet_inpaint.py` (control-inpaint)

Execution order:

1. **Preflight** (cheap, do first): check disk (weights are a multi-GB single file from
   `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0`; the pipeline hard-rejects v1 via the
   `control_in_dim` check - download Union 2.0 or 2.1 only), then run one bounded upstream
   torch/MPS smoke to capture reference tensors for parity (per this item's promotion criteria).
2. **MLX ControlNet module**: port `ZImageControlTransformerBlock` stack with configurable
   `control_layers_places` (host transformer has 30 layers), the separate control patch
   embedder, and the control noise-refiner variants keyed on `add_control_noise_refiner`
   (single-file config detection covers three structural variants).
3. **Weight sharing**: implement `from_transformer` semantics - nine module groups are shared
   with the base transformer (t_scale, t_embedder, all_x_embedder, cap_embedder, rope_embedder,
   noise_refiner, context_refiner, x_pad_token, cap_pad_token). Map the single-file checkpoint
   keys.
4. **Transformer hook**: add optional `controlnet_block_samples` per-block residual injection
   to the MLX `ZImageTransformer` forward (upstream adds them after matching blocks), strictly
   guarded so plain runs are byte-identical.
5. **Runtime + routing**: new `ZImageControlNet` variant mirroring `QwenImageControlNet`;
   capability rows gated to the exact validated model+sidecar identity
   (`supports_control_image` for structured control; `supports_mask` + `control_model` for
   control-inpaint - note the control-inpaint input is
   `concat([VAE(control_image) 16ch, inverted mask 1ch, VAE(source) 16ch]) = 33ch`, not a
   masked-image concat); CFG batching of control features; router injection + conflict
   validation copied from the Qwen pattern. Respect the one-masked-route-per-row invariant
   pinned by `test_every_row_exposes_at_most_one_masked_route` when adding control-inpaint.
6. **Parity + proof**: export-then-compare harness against the torch reference (tools/ pattern
   from the Wan VACE port), then ADR-0001 visual proofs: same prompt/seed no-control vs
   control, and for control-inpaint the source/mask/control/result sheet. Update
   `docs/masked-editing.md` and the validation registry only after accepted proofs.

Estimated effort: multi-day. Do not attempt inside a mixed session; steps 2-4 are the risky
core and should land behind tests before any capability is exposed.

## Why it might matter

This is the clearest Z-Image follow-up after native inpaint. It extends an already-supported family
instead of introducing a new base image architecture, and it could eventually give MLX-Gen a second
strong structured-control family next to Qwen.

## Promotion criteria

- Completed item 0043 remains the accepted native-inpaint baseline with exact proof assets.
- Public ControlNet weight paths remain available and runnable.
- One bounded upstream smoke proves the conditioning contract before any MLX port work starts.

## Validation ideas

- Same prompt, seed, and source across no-control vs control runs.
- Include source image, control image, and result in the proof sheet.
- If control-inpaint is chosen, also include the mask and localized result.

## Non-goals

- Do not bundle this into the first Z-Image inpaint pass.
- Do not assume Qwen control semantics and Z-Image control semantics are identical.
- Do not create implicit auto-preprocessors from this item alone.

## Guidance for future agents

Treat this as a narrow family-adjacent follow-up. If control-image preprocessing becomes the real
user pain point, split that into a separate item instead of hiding it inside the model port.

## Sources checked

- Z-Image native docs: https://huggingface.co/docs/diffusers/api/pipelines/z_image
- Z-Image Fun ControlNet Union 2.1: https://huggingface.co/alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1
- Z-Image-Turbo Fun ControlNet Union 2.1: https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
- VideoX-Fun repository: https://github.com/aigc-apps/VideoX-Fun
