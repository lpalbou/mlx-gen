# Masked Editing

This page is the canonical masked-edit reference for MLX-Gen: the request contract, the
per-model route matrix, per-family behavior notes, and the current proof surface. Other pages
([Image edit modes](image-edit-modes.md), [API and CLI](api.md), [FAQ](faq.md),
[Qwen route matrix](qwen-route-matrix.md)) summarize and link here.

Masked editing (also called inpainting) changes only one region of an existing image: repair a
damaged area, replace one object, recolor one part, or add a new object into a selected region,
while everything outside the region stays stable.

## The Request Contract

The request shape is the same on every masked route:

```sh
mlxgen generate \
  --model <mask-capable model> \
  --image source.png \
  --mask-path mask.png \
  --prompt "What the masked region should become." \
  --output edited.png
```

- White mask pixels are repainted; black mask pixels are preserved.
- `--mask-path` requires exactly one `--image` on the unified command.
- `--image-strength` cannot be combined with `--mask-path` anywhere; masked editing is a
  separate route from latent image-to-image.
- Pre-feather masks yourself if you want soft transitions; MLX-Gen binarizes masks at 50%
  luminance (see the per-family notes for each route's latent-grid resampling).
- Masks with an alpha channel produce a one-time warning; alpha is ignored and luminance is used.
- `masked_image_path` is recorded in output metadata and replayed by `--config-from-metadata`.
- Mask design matters for removals: cover the whole object you want gone, plus a margin. If the
  mask covers only part of an object that continues outside it, every route regenerates a
  plausible continuation instead of removing it (measured in the
  [matrix bundle's](assets/validation/masked-edit-matrix-2026-07-15/README.md) limitation
  demonstration).

Use `mlxgen capabilities --model <model>` to check whether a row supports masked edit
(`supports_mask=true`) before a long run.

## Model Matrix

Proof grades come from the standardized [masked-edit 5x5 matrix](assets/validation/masked-edit-matrix-2026-07-15/README.md)
(one source, five masks, same seed: object insertion, lens recolor, arm retexture, sticker
removal, plus an unscored partial-object-removal limitation demonstration), inspectable per row
with `mlxgen validation --model <row> --profile masked_edit_matrix_5x5_2026_07_15`.

| Model row | Route | Proof grade |
| --- | --- | --- |
| `AbstractFramework/qwen-image-edit-2511-8bit` | `qwen.inpaint` (edit checkpoint) | Validated visual QA with Lightning LoRA ([edit capabilities](edit-capabilities.md)) |
| Other Qwen edit rows (original, 2509, 2511 variants) | `qwen.inpaint` | Shipped; exact visual QA on the 2511 q8 row above |
| `AbstractFramework/qwen-image-8bit` | `qwen.control-inpaint` (InstantX sidecar, auto-injected) | Validated visual QA ([edit capabilities](edit-capabilities.md)) |
| `Qwen/Qwen-Image` source, `AbstractFramework/qwen-image-4bit`, `AbstractFramework/qwen-image-2512-8bit` | `qwen.base-inpaint` (native, no sidecar) | Validated matrix: PARTIAL - insertion, retexture, and removal pass on all three rows; opaque full-region recolors stay incomplete at defaults (warm-start anchoring, see below) |
| `AbstractFramework/z-image-turbo-8bit` | `z-image.inpaint` | Validated visual QA ([edit capabilities](edit-capabilities.md)) |
| Non-turbo Z-Image rows | not supported for the moment | Withdrawn after the matrix measured reproducible geometry artifacts on masks crossing thin structures (seed- and CFG-independent); masked requests are rejected before model load. Use Z-Image Turbo instead. |
| `AbstractFramework/flux.2-klein-4b-8bit`, `AbstractFramework/flux.2-klein-base-4b-8bit` | `flux2.inpaint` | Validated matrix: PASS on all four scored cases |
| Other FLUX.2 Klein rows (9B variants, source checkpoints) | `flux2.inpaint` | Shipped; matrix proof exists on the 4B q8 rows above, plus the reference-conditioned smoke case |
| Wan video rows | `--video-mask-path` (video-side masks) | See [Wan video](wan-video.md) |

Masked routes are exposed on trusted model identities (catalog aliases, official prepared
packages, exact proven rows, or local paths with an explicit `--base-model`). Arbitrary
local folder names do not unlock masked routes.

## Per-Family Behavior

### Qwen edit checkpoints (`qwen.inpaint`)

The edit checkpoints keep the full source image as conditioning while blending unmasked latents
back each step. Best masked-edit quality in the Qwen family, and the fast public path stacks the
`lightx2v/Qwen-Image-Edit-2511-Lightning` adapter (4 steps, guidance 1); see
[FAQ](faq.md#how-do-i-do-masked-edit-or-inpaint) for the exact command.

### Base Qwen checkpoints (`qwen.base-inpaint` and `qwen.control-inpaint`)

Every base Qwen row carries exactly one masked route, keyed on the exact model string:

- The exact validated `AbstractFramework/qwen-image-8bit` row keeps the ControlNet
  control-inpaint route: MLX-Gen auto-injects the InstantX inpainting sidecar and runs the full
  requested schedule. This is the stricter backend for hard local replacements.
- Every other trusted base row (source `Qwen/Qwen-Image`, `qwen-image-4bit`, the exact
  `qwen-image-2512-8bit` row) runs native masked edit, ported from the diffusers
  `QwenImageInpaintPipeline`: no sidecar download, and the masked region starts from the
  re-noised source at the upstream-example strength `0.85` rather than pure noise. Base Qwen is
  a text-to-image model without edit-instruction training; the warm start is what anchors
  repainted content to the surrounding structure.

Because of the warm start, the native route executes fewer denoise iterations than requested
(`20` requested steps run `17`). Output metadata records the runtime truth as
`effective_steps` and `mask_strength`. This differs from the other masked routes
(Qwen edit, Z-Image, FLUX.2 Klein), which denoise the full schedule from noise.

The warm start is tunable through `--mask-strength` (Python: `mask_strength`), the upstream
`QwenImageInpaintPipeline` strength knob scoped to masked runs. The measured trade-off from
the validation matrix:

- Default `0.85` anchors repainted content to the source: object insertion, retexturing, and
  removal validate cleanly, but opaque full-region recolors (for example turning a clear lens
  into a dark tinted one) stay incomplete.
- `--mask-strength 0.95` repaints fully — the matrix recolor case becomes a complete tinted
  lens on the q4 and 2512-q8 proof rows, while on the source bf16 row the repaint covers the
  full region but the color lands on the frame ring rather than the lens (single-case
  observation) — and the higher strength weakens the anchor: masks crossing thin connected
  structures (the arm-retexture case) can produce detached-geometry artifacts at that
  setting.

Rule of thumb: keep the default for edits that should follow existing structure; raise toward
`0.95` for content-replacing edits inside well-contained masks. Describing the current
appearance in `--negative` also pushes CFG away from the anchor and can substitute at the
default strength. For the strongest masked recolors overall, use a Qwen edit checkpoint or
FLUX.2 Klein. The applied `mask_strength` and executed `effective_steps` are recorded in
metadata and replayed by `--config-from-metadata`.

Prompt like a text-to-image caption on the native route — describe the target scene ("a small
red glasses case sitting behind them"), not the editing instruction ("add a case"). Mask
resampling follows diffusers: NEAREST onto the latent grid with a 50% threshold.

### Z-Image (`z-image.inpaint`, Turbo only)

Native per-step latent compositing on Turbo rows, 9 steps without guidance. Mask resampling:
NEAREST at the latent grid, 50% threshold.

Non-turbo Z-Image masked editing is not supported for the moment: the validation matrix
measured reproducible geometry artifacts when masks cross thin connected structures
(reproduced across seeds and with CFG both on and off, while Turbo renders the same case
cleanly — evidence in the [matrix bundle](assets/validation/masked-edit-matrix-2026-07-15/README.md)).
Non-turbo masked requests are rejected before model load; use Z-Image Turbo instead.

### FLUX.2 Klein (`flux2.inpaint`, distilled and base)

Ported from the diffusers `Flux2KleinInpaintPipeline`: the clean source rides along as
conditioning tokens while unmasked latents are re-composited each step. Distilled Klein runs
masked edits in 4 steps at guidance 1; base Klein defaults to guidance 4 with true CFG. The
backend `mflux-generate-flux2-edit` command and the Python `Flux2KleinInpaint` class also accept
extra images after the source as references for the masked area ("replace the masked object
with the one in the second image"). Masks are binarized at pixel resolution and then
bilinear-downsampled onto the packed latent grid, so mask borders keep soft transition values.

### Video

Masked video-to-video uses `--video-mask-path` on the Wan A14B route (exact latent lock) and on
Wan VACE models (learned mask conditioning with `--vace-masked-region`). See
[Wan video](wan-video.md).

## Choosing A Masked Route

- Best current image quality with a fast path: Qwen Image Edit 2511 q8 + Lightning.
- Local edits without any sidecar or edit checkpoint: base Qwen native route or Z-Image Turbo.
- Strictest local replacement discipline: base-Qwen control-inpaint on the exact 8bit row.
- Fastest masked edits: FLUX.2 Klein distilled (4 steps) or Z-Image Turbo (9 steps).
- Masked edit guided by a reference object image: FLUX.2 Klein backend route.

## Proof Assets

- [Masked edit 5x5 matrix (2026-07-15)](assets/validation/masked-edit-matrix-2026-07-15/README.md):
  the standardized multi-case validation for the new routes — contact sheet, per-case zoom
  sheets, preservation metrics, per-row statuses, and the partial-object-removal limitation
  demonstration. Registry profile `masked_edit_matrix_5x5_2026_07_15`.
- [Masked edit expansion smoke bundle (2026-07-15)](assets/validation/masked-edit-2026-07-15/README.md):
  the initial single-case wiring proofs with the runtime-truth metadata sidecar.
- [Image edit capabilities](edit-capabilities.md): the validated visual-QA rows (Qwen edit 2511
  masked edit, base-Qwen control-inpaint, Z-Image Turbo inpaint) with contact sheets and command
  logs.
- [Qwen localized editing](qwen-localized-editing.md): plain-language comparison of Qwen masked
  edit, structured control, and control-inpaint.
