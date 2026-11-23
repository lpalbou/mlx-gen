# Image Editing Modes

Open this file when a task starts from an existing image: restyle, instruction edit, masked edit,
multi-reference composition, reframe, outpaint, or structured control, and for step previews.
All of them run through `mlxgen generate`; `image-to-image` is one public task with several modes.

## Pick the mode

| Goal | Mode | Inputs | How it is selected | `--image-strength` |
| --- | --- | --- | --- | --- |
| Change mood, style, or lighting of the whole image | `latent-img2img` | one image | pass `--image-strength`, or `--i2i-mode latent` | required |
| Follow an instruction and keep the layout | `edit-reference` | one image | default on FLUX.2 Klein and Qwen edit models; or `--i2i-mode edit` | rejected |
| Change one region only | masked edit | one image + mask | `--mask-path` | rejected |
| Structure from one image, style or material from others | `multi-reference` | two or more images | repeat `--image`; the first anchors geometry | rejected |
| Reveal more of the scene (may redraw the source) | generative reframe | one image | `--reframe-padding` | rejected |
| Extend the canvas and keep the source region | outpaint | one image | `--outpaint-padding` | rejected |
| Fix the layout from an edge map or pose guide | structured control (text-to-image) | control image, no `--image` | `--controlnet-image-path` | not applicable |

Check the model first: the route must exist in `mlxgen capabilities --model <id>` with the matching
flag (`supports_image_strength`, `supports_mask`, `supports_reframe`, `supports_outpaint`,
`supports_control_image`, or a `multi-reference` row). An unsupported combination is refused
before weights load. Which family has which route is in [models.md](models.md).

## Latent img2img

```sh
mlxgen generate \
  --model "AbstractFramework/flux.2-klein-4b-8bit" \
  --image "<INPUT.png>" \
  --i2i-mode latent \
  --image-strength 0.4 \
  --prompt "<NEW LOOK FOR THE SAME SCENE>" \
  --seed 42 --metadata --output "<VARIATION.png>"
```

- `--image-strength` is in `(0, 1]`. Higher adds more noise and allows more change; lower stays
  closer to the source. `0.4` is a practical start.
- The run executes `floor(steps x image_strength)` denoise steps, and the progress bar shows that
  effective count: `--steps 50 --image-strength 0.4` runs 20.
- Z-Image, ERNIE Image Turbo, and base Qwen Image are latent-only for unmasked input and require
  an explicit `--image-strength`.

## Edit-reference and multi-reference

```sh
mlxgen generate \
  --model "AbstractFramework/qwen-image-edit-2509-8bit" \
  --image "<STRUCTURE.png>" --image "<STYLE.png>" \
  --prompt "<HOW TO COMBINE THEM>" \
  --negative "color, blur, crop, text, watermark" \
  --steps 20 --seed 42 --metadata --output "<COMPOSED.png>"
```

- One `--image` is an instruction edit; two or more select multi-reference where the model has
  that row (Qwen Image Edit 2509/2511, FLUX.2 Klein; the original `qwen-image-edit` is
  single-reference).
- Instruction edits regenerate the whole image, so a change can spill into nearby content. Name
  what must stay ("keeping the suit exactly the same blue and everything else unchanged"), or use
  a mask for a hard guarantee.
- Negative prompts: Qwen and base FLUX.2 Klein accept them (Klein needs `--guidance` above 1.0);
  distilled Klein rejects them.

## Masked edit (inpaint)

```sh
mlxgen generate \
  --model "AbstractFramework/qwen-image-edit-2511-8bit" \
  --image "<SOURCE.png>" \
  --mask-path "<MASK.png>" \
  --prompt "<WHAT THE MASKED REGION SHOULD BECOME>" \
  --seed 42 --metadata --output "<EDITED.png>"
```

- White mask pixels are repainted, black pixels are preserved. Masks are binarized at 50%
  luminance; alpha is ignored with a warning. Feather the mask yourself for soft transitions.
- Exactly one `--image`. `--image-strength` and `--outpaint-padding` cannot be combined with
  `--mask-path`.
- To remove an object, cover the whole object plus a margin. A mask over part of an object makes
  the model continue it instead of removing it.
- Routes: Qwen edit models (`qwen.inpaint`), base Qwen models (`qwen.base-inpaint`, or
  `qwen.control-inpaint` on the exact `AbstractFramework/qwen-image-8bit` row), Z-Image Turbo
  (`z-image.inpaint`), FLUX.2 Klein distilled and base (`flux2.inpaint`). Non-Turbo Z-Image
  rejects masked requests.
- Base Qwen native masked edit warm-starts from the source: `--mask-strength` (default `0.85`)
  tunes it; raise toward `0.95` when the content must be replaced, for example a full recolor.
- Masked routes require a trusted model identity: a catalog alias, a published package id, or a
  local folder with an explicit `--base-model`.

## Output size for image-to-image

- Default `--canvas-policy source-aspect`: the first image sets the aspect ratio, and `--width` /
  `--height` are a size target. The saved size can differ from the request.
- `--canvas-policy exact-resize` forces the requested canvas and may reshape the source. With it,
  `--resize-mode crop` center-crops and `--resize-mode pad` letterboxes instead of stretching
  (`resize`, the default). `--resize-mode` exists on latent routes only; edit, control, and
  outpaint routes reject it.
- The sidecar records `canvas_policy`, `resize_mode`, `requested_width`, `requested_height`,
  `source_image_width`, `source_image_height`, and the final `width` / `height`.

## Reframe and outpaint

Both take CSS-style padding in `top,right,bottom,left` order, in pixels or percent; each side is
independent and `0` leaves an edge in place.

```sh
# Generative reframe: a wider view, the source may be redrawn.
mlxgen generate \
  --model "AbstractFramework/flux.2-klein-4b-8bit" \
  --image "<INPUT.png>" \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "<DESCRIBE THE WIDER VIEW>" \
  --steps 16 --seed 42 --metadata --output "<REFRAMED.png>"

# Outpaint: extend the canvas and hold the source region.
mlxgen generate \
  --model "AbstractFramework/flux.2-klein-4b-8bit" \
  --image "<INPUT.png>" \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "<DESCRIBE WHAT FILLS THE NEW AREA>" \
  --steps 16 --guidance 1 --seed 42 --metadata --output "<OUTPAINT.png>"
```

| Family | Reframe | Outpaint | Guidance |
| --- | --- | --- | --- |
| Qwen Image Edit / 2509 / 2511 | yes | yes, fixed `edge` canvas | model default |
| FLUX.2 Klein 4B / 9B distilled | yes | yes | `1.0` |
| FLUX.2 Klein Base 4B / 9B | no | yes | `4.0` |

Not available on base Qwen Image, Qwen Image 2512, ERNIE, Z-Image, FIBO, Bonsai, video models, or
SeedVR2: they reject the options before loading weights.

- `--outpaint-padding` computes the output size itself: do not add `--width`, `--height`, or
  `--canvas-policy`. It cannot be combined with `--reframe-padding` or `--mask-path`.
- Outpaint is generative, not a pixel lock: the source region is held during denoising and pasted
  back while it still matches. Use a mask when a region must stay untouched.
- `--outpaint-fill auto|edge|neutral|solid|blur` (FLUX.2 Klein only) chooses what fills the new
  area before denoising. `edge` continues the border texture; `neutral` gives the model a blank
  area to generate new content; `auto` (default) picks between them from the padding depth. If the
  new area comes back as stretched border texture, rerun with `--outpaint-fill neutral`.
  `--outpaint-fill-color "R,G,B"` or `"#rrggbb"` sets the color for `solid`. Qwen edit accepts
  only its fixed `edge` canvas and refuses another mode.
- `--outpaint-passes auto|1|2`: a request that pads both axes deeply is split into two single-axis
  passes by `auto`, which avoids a duplicated subject in the free corner and doubles the denoising.
- Every outpaint run prints the resolved canvas on stderr first, for example
  `Outpaint: fill=neutral, canvas 928x1536 from source 768x766, padding top=0 right=76 bottom=766 left=76.`
- Cost reference on one 432x240 source at `5%,80%,5%,60%`, Apple M5 Max, 8-bit packages:
  distilled Klein 4B 54.6 s, distilled Klein 9B 80.6 s, base Klein 4B 72.2 s, base Klein 9B
  100.0 s, Qwen Image Edit 2511 341.1 s.
- Base Klein rows publish `outpaint_recommended_lora` (`fal/flux-2-klein-4B-outpaint-lora`); it is
  optional. With it loaded, `auto` paints the green canvas the adapter expects and the prompt is
  "Fill the green spaces according to the image".

## Structured control (Qwen)

```sh
mlxgen generate \
  --model "AbstractFramework/qwen-image-8bit" \
  --prompt "<PROMPT>" \
  --controlnet-image-path "<CANNY_OR_POSE.png>" \
  --seed 42 --metadata --output "<CONTROLLED.png>"
```

- Supported on the exact `AbstractFramework/qwen-image-8bit` row (`qwen.control`). The router
  injects the sidecar `InstantX/Qwen-Image-ControlNet-Union:diffusion_pytorch_model.safetensors`,
  reported as `control_model` on the capability row. Like every other file it must already be on
  disk: when it is missing, the run stops with the download error that names the command.
- It is text-to-image guided by a control image, not an edit: do not combine it with `--image`.
  For source-image editing on that row use `--image` plus `--mask-path` (`qwen.control-inpaint`).
- `--controlnet-strength` is accepted on these routes.

## Replay and previews

- `--config-from-metadata "<FILE.metadata.json>"` (`-C`) reuses the parameters of a previous run,
  including the model, images, padding, and resolved outpaint fill; options you pass take priority.
- `--stepwise-image-output-dir "<DIR>"` writes one PNG per denoising step plus a composite strip.
  `--preview-decoder auto|tiny|full` picks the preview decoder: `tiny` uses a small published
  autoencoder (about 8% extra wall time for every-step previews instead of about 86%), available
  after `mlxgen download --model madebyollin/taef1` (FLUX.1 latent space: Z-Image) or
  `madebyollin/taef2` (FLUX.2 latent space: FLUX.2 Klein, ERNIE, Bonsai). `auto` falls back to the
  full VAE when the tiny decoder is missing; `tiny` errors instead. Qwen and FIBO always preview
  through the full VAE. Final outputs are always decoded with the full VAE.

Source docs: <https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/image-edit-modes.md>,
<https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/masked-editing.md>,
<https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/reframe-outpaint.md>,
<https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/previews.md>.
