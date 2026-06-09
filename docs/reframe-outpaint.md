# Reframe And Outpaint

MLX-Gen exposes two experimental single-image canvas expansion workflows through `mlxgen generate`:

- `--reframe-padding` asks an edit model to generate a wider view from the source image. The model
  can redraw the source while changing the crop, viewpoint, or visible subject boundary.
- `--outpaint-padding` builds an expanded conditioning canvas, runs a supported edit model on that
  canvas, and blends the original source back only when the generated source window still matches
  closely enough.

Both options use CSS-style padding in `top,right,bottom,left` order. Percentages are relative to
the source image size. For example, `5%,80%,5%,60%` adds a small top/bottom border, more space to
the right, and a large extension to the left.

These routes are generative edit workflows. They are not native masked fill/inpaint pipelines and
do not guarantee exact pixel locking. Use them when a plausible wider view is acceptable and review
the output visually. A future native fill/inpaint route should be used when only the new border
area may change.

## Supported Models

The current experimental validation profile is `reframe_outpaint_2026_06_08`. It uses one cropped
starship source image and checks source, q8, and q4 packages for each supported family:

| Family | Source model | q8 package | q4 package | Status |
| --- | --- | --- | --- | --- |
| Qwen Image Edit | `Qwen/Qwen-Image-Edit` | `AbstractFramework/qwen-image-edit-8bit` | `AbstractFramework/qwen-image-edit-4bit` | experimental reframe and outpaint pass |
| Qwen Image Edit 2509 | `Qwen/Qwen-Image-Edit-2509` | `AbstractFramework/qwen-image-edit-2509-8bit` | `AbstractFramework/qwen-image-edit-2509-4bit` | experimental reframe and outpaint pass |
| Qwen Image Edit 2511 | `Qwen/Qwen-Image-Edit-2511` | `AbstractFramework/qwen-image-edit-2511-8bit` | `AbstractFramework/qwen-image-edit-2511-4bit` | experimental reframe and outpaint pass |
| FLUX.2 Klein 4B | `black-forest-labs/FLUX.2-klein-4B` | `AbstractFramework/flux.2-klein-4b-8bit` | `AbstractFramework/flux.2-klein-4b-4bit` | experimental reframe and outpaint pass |
| FLUX.2 Klein 9B | `black-forest-labs/FLUX.2-klein-9B` | `AbstractFramework/flux.2-klein-9b-8bit` | `AbstractFramework/flux.2-klein-9b-4bit` | experimental reframe and outpaint pass |

These options are intentionally not exposed for base Qwen Image, Qwen Image 2512, FLUX.2 Klein
Base, ERNIE Image Turbo, Z-Image, FIBO, Bonsai, Wan, or SeedVR2. Those families are text
generation, latent I2I, video, or upscale/restoration routes, or do not yet have a validated
edit-reference canvas-expansion profile.

Check support before running:

```sh
mlxgen capabilities --model AbstractFramework/qwen-image-edit-2511-8bit
```

Inspect the validation records for a package:

```sh
mlxgen validation \
  --profile reframe_outpaint_2026_06_08 \
  --model AbstractFramework/qwen-image-edit-2511-8bit
```

## Reframe Example

Use reframe when you want a model to create a wider view and you accept that the source may be
redrawn:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --image input.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the full subject and extend the background naturally." \
  --steps 16 \
  --seed 42 \
  --output reframed.png
```

## Outpaint Example

Use outpaint when you want MLX-Gen to construct a larger canvas before the edit model runs:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image input.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close crop into a wider realistic shot. Complete the missing subject and background outside the original frame." \
  --negative "text, border, frame, hard seam, duplicate subject" \
  --steps 20 \
  --guidance 4 \
  --seed 42 \
  --output outpaint.png
```

After generation, MLX-Gen compares the source window in the generated image with the original
source. If the generated source window is close, MLX-Gen blends source detail back in. If the model
has reconstructed or moved the scene, MLX-Gen keeps the generated image to avoid ghosted fragments.

## Validation Assets

The current proof set uses this source image:

![Cropped starship source](assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png)

The outpaint helper creates this wider conditioning canvas and source-window mask:

![Wide outpaint canvas](assets/validation/reframe-outpaint-2026-06-08/source-b-outpaint-canvas-wide.png)

In the mask image, black marks the original source window and white marks the generated border area.

![Wide outpaint source mask](assets/validation/reframe-outpaint-2026-06-08/source-b-outpaint-mask-wide.png)

The summary sheet shows every supported source/q8/q4 row:

![Reframe and outpaint source/q8/q4 summary](assets/validation/reframe-outpaint-2026-06-08/reframe-outpaint-base-q8-q4-summary.jpg)

Per-family contact sheets:

- [Qwen Image Edit](assets/validation/reframe-outpaint-2026-06-08/qwen-image-edit-reframe-outpaint-matrix.jpg)
- [Qwen Image Edit 2509](assets/validation/reframe-outpaint-2026-06-08/qwen-image-edit-2509-reframe-outpaint-matrix.jpg)
- [Qwen Image Edit 2511](assets/validation/reframe-outpaint-2026-06-08/qwen-image-edit-2511-reframe-outpaint-matrix.jpg)
- [FLUX.2 Klein 4B](assets/validation/reframe-outpaint-2026-06-08/flux2-klein-4b-reframe-outpaint-matrix.jpg)
- [FLUX.2 Klein 9B](assets/validation/reframe-outpaint-2026-06-08/flux2-klein-9b-reframe-outpaint-matrix.jpg)

The exact commands and validation manifest are published with the assets:

- [Command log](assets/validation/reframe-outpaint-2026-06-08/reframe-outpaint-command-log.md)
- [Validation manifest](assets/validation/reframe-outpaint-2026-06-08/reframe-outpaint-validation-manifest.json)
