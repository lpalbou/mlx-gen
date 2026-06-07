# API And CLI

MLX-Gen can be used from the command line or embedded in Python applications. The stable public entry point for new command-line usage is `mlxgen`.

## Command-Line Surface

Use `mlxgen --help` to see the command groups:

```sh
mlxgen --help
```

The public workflows are:

| Command | Purpose |
| --- | --- |
| `mlxgen generate` | Generate images or supported videos from a cached or prepared model. Image input selects image-to-image or image-to-video when the model supports it. |
| `mlxgen capabilities` | Inspect the public tasks, internal modes, and option support for a model without loading weights. |
| `mlxgen validation` | Inspect generated-output and benchmark records for exact model/package rows. |
| `mlxgen download` | Explicitly download model or LoRA files into the local cache. |
| `mlxgen prepare` | Create a reusable local MLX-Gen model folder, optionally quantized, and write a Hugging Face model card. |

The package also installs dedicated entry points from the mflux codebase for workflows that are not
yet routed through unified `mlxgen generate`, including `mflux-upscale-seedvr2` for SeedVR2 image
super-resolution. New generation and model-management integrations should prefer the `mlxgen`
commands above when the workflow is available there.

For a full copy/pasteable workflow that exercises T2I, I2I edit, multi-reference I2I, T2V A14B,
and I2V A14B, see [Spaceship Snow Workflow](examples/spaceship-snow.md).

## Generation Router

`mlxgen generate` chooses the backend from `--model`, optional `--family`, and image inputs. Public
tasks are media directions: `text-to-image`, `image-to-image`, `text-to-video`, and
`image-to-video`. Edit/reference behavior is an internal image-to-image mode, not a separate public
task.

```sh
mlxgen generate \
  --model z-image-turbo \
  --prompt "A product photo of a ceramic teapot" \
  --output image.png
```

Inspect a model before generation:

```sh
mlxgen capabilities --model flux2-klein-4b
```

The JSON includes each route-supported public task, internal mode, image count, route handler, and
option support. Route support means MLX-Gen knows how to dispatch and validate options; it is not a
claim that a model/package passed visual release QA. Applications can use the same contract from
Python through `get_model_capabilities(...)` and `resolve_generation_plan(...)`. For custom
repositories or local paths whose name does not identify the architecture, construct the
`ModelConfig` with the same base-model hint that you would pass to the CLI.

Most image and video backends accept a negative prompt. In the unified CLI,
`--negative-prompt` and `--negative` are aliases. Python callers pass the same value as
`negative_prompt=...` on the model-specific generation method.

Use `mlxgen validation` when you need exact release evidence for a model/package:

```sh
mlxgen validation --model AbstractFramework/qwen-image-edit-2509-8bit
```

This returns the current validation profile rows with status, prompt, source image(s), artifact
path, and reviewer notes. Route support and visual validation are intentionally separate:
`mlxgen capabilities --model briaai/Fibo-Edit` currently exposes no unified public generation
capability, while `mlxgen validation --model AbstractFramework/qwen-image-edit-2511-8bit` reports
the current Qwen 2511 edit proof rows.

### Image-To-Image Modes

`image-to-image` is one public task with several internal modes. Use `mlxgen capabilities --model
<model>` to see which modes a selected model exposes, and use `--i2i-mode` when you need to force a
specific path.

| Goal | Internal mode | Inputs | Selection rule | Uses `--image-strength`? |
| --- | --- | --- | --- | --- |
| Whole-image variation or restyle from a source image | `latent-img2img` | exactly one image | pass `--image-strength` or `--i2i-mode latent` on a model that supports latent I2I | Yes |
| Instruction edit, object/layout change, or composition-preserving style edit | `edit-reference` | one image | default for FLUX.2 and dedicated edit checkpoints when one image is supplied without `--image-strength`; or pass `--i2i-mode edit` | No |
| Reference composition from several images | `multi-reference` | two or more images | repeat `--image` on a model that supports multi-reference I2I; or pass `--i2i-mode multi-reference` | No |
| Inpainting, outpainting, or reframing with a preserved canvas | fill/outpaint mode | image plus mask/canvas | not first-class in unified `mlxgen generate` yet | No |

Use latent img2img when you want a whole-image variation driven by source-image noise injection:
restyle the whole scene, change the mood, or make a loose variation. Higher `--image-strength`
adds more noise, allows more change, and runs more effective denoise steps. Lower values stay
closer to the encoded source image.

Use edit/reference I2I when the prompt is an instruction: remove an object, change an object color,
turn a scene into a pencil sketch while preserving layout, reposition or reshape a subject, or keep
the composition stable. Edit/reference and multi-reference routes use the image(s) as conditioning
or references, so `--image-strength` is rejected before loading weights.

In `auto` mode, the selected model's default capability wins. FLUX.2 routes one image to
`edit-reference`, supports latent I2I when `--image-strength` is supplied, and supports
multi-reference I2I with two or more images. The original `Qwen/Qwen-Image-Edit` checkpoint is a
single-reference edit model in MLX-Gen. Use it for one-source semantic or appearance edits such as
pencil sketch, object-state changes, style changes, and layout-preserving instruction edits. Qwen
Image Edit 2509 and 2511 expose multi-reference edit routes through unified
`mlxgen generate` when a package supports that route. The validation command records which exact
source/prepared package rows passed visual review. Qwen Image Edit 2511 source, q8, and q4 now have
passing 2026-06-06 proof rows for single-image pencil sketch, single-image hard-landing edit, and
two-reference pencil-plus-crash composition.

Latent-only image models such as ERNIE Image Turbo, Z-Image, and base Qwen Image require explicit
`--image-strength` for `latent-img2img`. Base FIBO exposes text-to-image through unified
`mlxgen generate`. FIBO Edit is not exposed as a public `mlxgen generate` capability until it has
passing source-model and prepared-package visual proof; use the dedicated FIBO Edit command only for
experimental parity work.

### Image-To-Image Canvas Policy

Ordinary image-to-image defaults to `--canvas-policy source-aspect`. The first input image defines
the output aspect ratio for latent img2img, edit/reference I2I, and multi-reference I2I. `--width`
and `--height` are treated as a size target, then MLX-Gen chooses the nearest model-compatible
canvas that preserves the source ratio. For multi-reference I2I, the first `--image` is the
geometry anchor and later images are references.

Use `--canvas-policy exact-resize` only when you intentionally want the requested output canvas
exactly. Exact resize can reshape or recompose the source and is not a substitute for outpainting.
Generated image metadata records `canvas_policy`, requested dimensions, source-image dimensions,
and final `width`/`height` when an image input is used.

For instruction/reference image-to-image, pass one or more input images to an edit-capable model:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2509-8bit \
  --image input.png \
  --prompt "Turn the room into a pencil sketch" \
  --negative "color, blur, crop, text, watermark" \
  --output edited.png
```

For latent image-to-image variation, use a model that supports `latent-img2img` and pass
`--image-strength`. `--image-strength` is rejected for edit/reference and multi-reference modes
because those paths use source/reference images as conditioning rather than noising the source
latent:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --image input.png \
  --i2i-mode latent \
  --canvas-policy source-aspect \
  --image-strength 0.4 \
  --prompt "Make the scene more cinematic" \
  --output variation.png
```

`--task edit` remains accepted as a compatibility alias for
`--task image-to-image --i2i-mode edit`, but new commands and integrations should prefer
`--i2i-mode`.

Reframing and outpainting are not ordinary image-to-image resizing. Generic I2I with a larger
`--width` or `--height` resizes/recomposes the source instead of preserving original pixels in place.
The reliable operation is masked outpainting: create a larger canvas, paste the source image into
it, create a mask for the new area, and run a fill/inpaint model. MLX-Gen has lower-level FLUX.1
Fill support inherited from mflux, but the unified `mlxgen generate --outpaint-padding ...` flow is
not available yet.

### Negative Prompts

Use `--negative-prompt` or the shorter `--negative` alias to describe what the model should avoid:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-8bit \
  --image input.png \
  --prompt "Convert the scene into a clean graphite pencil sketch while preserving layout" \
  --negative "color, blur, crop, text, watermark" \
  --steps 30 \
  --guidance 4 \
  --output sketch.png
```

For Qwen image edit, guidance above `1` uses true classifier-free guidance when a negative prompt
is present. If you omit the option, MLX-Gen uses the official blank negative-prompt behavior for
Qwen edit models, so true CFG remains enabled by default. Passing an explicit negative prompt is
still useful for blocking concrete failure modes such as crop, blur, text, intact object state, or
unwanted color.

For Wan, omitting the option uses the model's official default negative prompt. Pass
`--negative ""` or `--negative-prompt ""` to intentionally run without a negative prompt.

Supported router families are `qwen`, `flux2`, `bonsai`, `fibo`, `z-image`, `ernie-image`, and `wan`:

```sh
mlxgen generate \
  --model ./models/qwen-image-8bit \
  --family qwen \
  --prompt "A clean studio product photo"
```

Use `--config-from-metadata` / `-C` when you want the router to read fields such as `model`, `image_path`, or `image_paths` from an existing metadata file.

Bonsai Image routes through the same text-to-image command surface. The supported ternary
checkpoint is already low-bit packed, so omit `--quantize`:

```sh
mlxgen generate \
  --model prism-ml/bonsai-image-ternary-4B-mlx-2bit \
  --prompt "A bonsai tree in a quiet ceramic studio, soft morning light" \
  --width 1024 \
  --height 1024 \
  --steps 4 \
  --guidance 1 \
  --seed 42 \
  --output bonsai.png
```

Bonsai is text-to-image only in MLX-Gen. Image input, negative prompts, and `--quantize` are
rejected before model execution.

ERNIE Image Turbo routes through the same command surface:

```sh
mlxgen generate \
  --model baidu/ERNIE-Image-Turbo \
  --prompt "A clean product photo of a ceramic mug" \
  --width 512 \
  --height 512 \
  --steps 8 \
  --guidance 1 \
  --output image.png
```

ERNIE Image Turbo supports BF16 source weights plus prepared q8/q4 folders. MLX-Gen also provides single-image latent image-to-image for ERNIE:

```sh
mlxgen generate \
  --model baidu/ERNIE-Image-Turbo \
  --image input.png \
  --prompt "Turn the scene into a pencil sketch" \
  --width 512 \
  --height 512 \
  --steps 8 \
  --guidance 3 \
  --image-strength 0.25 \
  --output edited.png
```

ERNIE image-to-image accepts exactly one input image. Multi-image edit is not supported. `--image-strength` follows latent img2img denoising semantics: higher values add more noise and allow more transformation, while lower positive values stay closer to the encoded source image.

For ERNIE image-to-image, the default source-aspect canvas policy preserves the input ratio while
using `--width` and `--height` as a size target. Use roughly `--image-strength 0.25` to `0.35` for
visible stylization, `0.45` to `0.6` for stronger source preservation, and 12-16 steps when the
output needs more polished stylization. Use Qwen Image Edit for single-image instruction edits when
source layout matters, and use FLUX.2 when the workflow needs validated multi-reference composition.

ERNIE's optional Prompt Enhancer is available with `--use-prompt-enhancer` when the full source snapshot is present. The default `mlxgen download --model baidu/ERNIE-Image-Turbo` command downloads only generation components; run `mlxgen download --model baidu/ERNIE-Image-Turbo --all-files` before using Prompt Enhancer. Prepared q8/q4 ERNIE folders created by `mlxgen prepare` do not include Prompt Enhancer files.

Wan2.2 routes through the same command surface for video generation. TI2V-5B is the smaller text-to-video and first-frame image-to-video path:

```sh
mlxgen generate \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --prompt "A short cinematic video of a glowing orange glass sphere floating above teal water" \
  --width 1280 \
  --height 704 \
  --frames 121 \
  --steps 50 \
  --guidance 5 \
  --fps 24 \
  --output video.mp4
```

T2V-A14B uses the larger two-transformer Diffusers path. `--guidance-2` is an optional
Diffusers-compatible low-noise-stage override. With no guidance flags, MLX-Gen uses the model's
two-stage defaults (`4` high-noise and `3` low-noise for T2V-A14B). If you set `--guidance` and
omit `--guidance-2`, the low-noise stage follows `--guidance`:

```sh
mlxgen generate \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "A cinematic shot of mist rolling across a teal mountain lake" \
  --width 1280 \
  --height 720 \
  --frames 81 \
  --steps 40 \
  --guidance 4 \
  --guidance-2 3 \
  --fps 16 \
  --output video.mp4
```

TI2V-5B image-to-video uses the same command with one input image:

```sh
mlxgen generate \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --image input.png \
  --prompt "A slow cinematic camera move from the input frame" \
  --width 1280 \
  --height 704 \
  --frames 121 \
  --steps 50 \
  --guidance 5 \
  --fps 24 \
  --output video.mp4
```

A14B I2V uses the separate `Wan-AI/Wan2.2-I2V-A14B-Diffusers` snapshot and the Diffusers
concatenated image-condition latent path:

```sh
mlxgen generate \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image input.png \
  --prompt "A cinematic flyby around the subject in the input image" \
  --width 1280 \
  --height 720 \
  --frames 81 \
  --steps 40 \
  --guidance 3.5 \
  --fps 16 \
  --output video.mp4
```

The TI2V-5B I2V path follows Diffusers first-frame latent conditioning: the first frame is VAE-encoded, kept active through denoising with a timestep mask, and reinserted before decode. The separate A14B I2V model uses concatenated image-condition latents instead. Multi-image/video interpolation is not enabled.

For Wan image-to-video, saved metadata records the requested dimensions, the source image
dimensions, and the resolved output dimensions.

### Wan Video Parameters

Wan uses frame-count control rather than a separate duration flag. The output duration is:

```text
duration_seconds = frames / fps
```

At the default 24 fps, `--frames 121` produces about 5.04 seconds of video, `--frames 73` produces about 3.04 seconds, and `--frames 49` produces about 2.04 seconds.

| Option | Behavior |
| --- | --- |
| `--width`, `--height` | Accepted values are model-specific. Text-to-video values are adjusted up to the selected Wan VAE/patch multiple. For image-to-video, these values are a size target: MLX-Gen resolves the final canvas from the source image aspect ratio and the selected model's spatial multiple before conditioning the model. |
| `--frames` | Number of output frames. Wan requires `4n + 1`; other values are adjusted to `4 * floor(frames / 4) + 1`. TI2V-5B default: `121`; A14B default: `81`. |
| `--fps` | MP4 playback frame rate. Any positive integer is accepted. TI2V-5B default/recommended value: `24`; A14B default/recommended value: `16`. |
| `--steps` | Denoising steps. TI2V-5B default/recommended quality value: `50`; A14B default/recommended value: `40`. Lower values run faster but reduce quality. |
| `--guidance` | Classifier-free guidance scale. TI2V-5B default: `5`; A14B default: `4`. |
| `--guidance-2` | Optional low-noise guidance scale for Wan A14B `transformer_2`. If both guidance flags are omitted, model-specific two-stage defaults are used. If `--guidance` is set and `--guidance-2` is omitted, the low-noise stage follows `--guidance`. It is rejected for single-transformer Wan models. |
| `--negative-prompt`, `--negative` | If omitted, Wan uses the model's official default negative prompt. Pass `--negative ""` to intentionally run without a negative prompt; this can be better for simple abstract scenes where the default negative prompt adds unwanted texture. |
| `--seed` | Deterministic seed. Repeat with multiple values to create multiple videos. |
| `--progress`, `--no-progress` | Show or disable the CLI video progress bar. The bar advances by denoising step and keeps the requested frame count as context. Default: `--progress true`. |
| `--low-ram` | For Wan CLI runs, clear MLX cache between transformer blocks and denoise steps, release denoisers before decode when the model instance will not be reused for another seed, and clear cache between VAE temporal decode slices. This is intended for memory pressure, not speed. |

Common Wan video sizes:

| Model | Required width/height multiple | Recommended/native quality size | Useful lower-cost sizes | Notes |
| --- | ---: | --- | --- | --- |
| TI2V-5B T2V/I2V | 32 px | `1280x704` or `704x1280` | `832x480`, `480x832`, `448x256`, `256x448` | Text-to-video `1280x720` adjusts to `1280x736`; image-to-video preserves the source image ratio at a nearby supported canvas. |
| T2V-A14B | 16 px | `1280x720` or `720x1280` | `832x480`, `480x832`, `448x256`, `256x448`, `432x240` | Text-to-video only; image input is rejected. |
| I2V-A14B | 16 px | Source-ratio canvas near `1280x720` or `720x1280` | Source-ratio canvas near `832x480`, `448x256`, or `432x240` | Requires one input image; output preserves the source image ratio at a nearby supported canvas. |

The upstream TI2V-5B guidance is 1280x704 or 704x1280, 121 frames, 50 steps, and 24 fps. The upstream A14B guidance is 1280x720 or 720x1280, 81 frames, 40 steps, `--guidance 4`, optional `--guidance-2 3`, and 16 fps. Lower resolutions, frame counts, or step counts are useful for routing and prompt checks, but they should not be treated as final quality settings.

For visual checks, use `448x256` or larger for Wan examples. Tiny square canvases such as `128x128`
are not representative of Wan video quality or prompt adherence.

Example outputs at 1280x704, 17 frames, and 20 steps:

![Wan2.2 TI2V 1280x704 text-to-video contact sheet](assets/generation/wan2.2-ti2v-5b-t2v-1280x704-17f-20steps-contact-sheet.png)

![Wan2.2 TI2V first-frame image-to-video contact sheet](assets/generation/wan2.2-ti2v-5b-i2v-bateau-1280x704-17f-20steps-contact-sheet.png)

These panels are examples at the model's spatial scale. Evaluate final visual quality with the
recommended full-resolution, frame-count, and step-count settings for your target model.

## SeedVR2 Upscale Command

SeedVR2 image super-resolution uses the dedicated `mflux-upscale-seedvr2` command. See
[Image Upscaling](upscaling.md) for a reproducible 5x source/output comparison.

```sh
mflux-upscale-seedvr2 \
  --image-path input.png \
  --resolution 1024 \
  --quantize 8 \
  --metadata \
  --output input_short_edge_1024.png
```

`--resolution` accepts either an integer target or a scale factor:

| Value | Meaning | Example from `320x192` |
| --- | --- | --- |
| `1024` | Set the shorter output edge to 1024px and preserve the source aspect ratio. | `1706x1024` after even-dimension normalization |
| `2x` | Multiply both source dimensions by 2 and preserve the source aspect ratio. | `640x384` |
| `3x` | Multiply both source dimensions by 3 and preserve the source aspect ratio. | `960x576` |

Use integer shortest-edge sizing when you want a predictable target size across mixed source image
ratios. Use scale factors when you want to compare direct 2x/3x upscaling behavior. SeedVR2 also
restores and denoises, so a target close to the source size can be useful for restoration checks but
is not a good visual proof of super-resolution. For upscale quality checks, choose a target that
materially increases the pixel dimensions.

Useful options:

| Option | Behavior |
| --- | --- |
| `--image-path` | One or more image files or directories. Directories are expanded to supported image files. |
| `--resolution` | Integer shorter-edge target or scale factor such as `2x` or `3x`. Default: `384`. |
| `--model` | Optional SeedVR2 model selector. Defaults to `seedvr2-3b`; pass `seedvr2-7b` for the larger model. |
| `--quantize` | Optional runtime quantization. Supported values are `4` and `8`. |
| `--softness` | Optional input pre-downsampling control from `0.0` to `1.0`; use about `0.25` to `0.5` when noisy source texture should be smoothed before reconstruction. |
| `--vae-tiling` | Enable tiled VAE encode/decode for very large memory-bound upscales. The default is off for best quality. |
| `--metadata` | Write a `.metadata.json` sidecar with final output dimensions, source dimensions, seed, and model details. |

## Model Management Commands

`mlxgen download` and `mlxgen prepare` are the only public MLX-Gen commands that authorize network access.

```sh
mlxgen download --model Qwen/Qwen-Image
```

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  --quantize 8
```

Use `prepare` when you need the local saved-weight folder. It is the public MLX-Gen workflow for creating quantized model folders and generated Hugging Face cards.

If a complete prepared folder exists at `./models/<repo-name>`, a matching Hugging Face handle can
resolve to it before requiring a cache snapshot. This lets applications use stable handles such as
`AbstractFramework/qwen-image-edit-2511-8bit` or
`AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` while still running from a local prepared folder.

Generation output replaces the requested `--output` path by default. Use `--replace false` or `--no-replace` to preserve an existing file and save to a suffixed filename.

Wan video failures write a compact manifest next to the intended output path, such as
`video.failure.json` for `video.mp4`. It captures the error, tensor-health report when available,
seed, prompt, dimensions, frames, steps, guidance, fps, output path, and memory-related runtime
flags.

## Python Integration

The current Python integration path uses model classes inherited from the mflux codebase. New applications can import the `mlxgen` helpers documented in [Python Integration](python-integration.md).

Python callers should prepare or download required model files before constructing model objects. Runtime constructors and generation calls do not start network downloads.

For progress monitoring, use `mflux.callbacks.ProgressEvent` and subscribe with
`model.callbacks.subscribe_progress(...)`. Image generation emits `start`, `denoise`, `complete`,
and interruption events through that subscription path. Wan video generation uses the same event
type and also accepts a direct `progress_callback` argument on `generate_video()`: model generation
emits `start`, `denoise`, `decode`, `convert`, and `generated`; the Wan CLI then emits `save` and
`complete` only after MP4 save and video-health validation succeed.

```python
from mflux.models.common.download_policy import DownloadRequiredError
from mlxgen.models.z_image import ZImageTurbo

try:
    model = ZImageTurbo(quantize=8)
except DownloadRequiredError as exc:
    print(exc.download_command)
    raise
```

## Compatibility Boundary

MLX-Gen prepared model folders use the MLX/mflux saved-weight layout and MLX quantization tensors. They are intended for MLX-Gen and compatible mflux code, not for direct Diffusers or Transformers `from_pretrained()` loading.
