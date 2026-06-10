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
| `mlxgen generate` | Generate images or supported videos from a downloaded source model or MLX-Gen model package. Image input selects image-to-image or image-to-video when the model supports it. |
| `mlxgen upscale` | Upscale and restore images with SeedVR2. |
| `mlxgen capabilities` | Inspect the public tasks, internal modes, and option support for a model without loading weights. |
| `mlxgen validation` | Inspect generated-output and benchmark records for exact model/package rows. |
| `mlxgen download` | Explicitly download model or LoRA files into the local cache. |
| `mlxgen prepare` | Create a reusable local MLX-Gen model package, optionally quantized, and write a Hugging Face model card. |

The package also installs compatibility entry points from the mflux codebase. New workflows should
prefer the `mlxgen` commands above when a matching command exists.

For a full copy/pasteable workflow that exercises T2I, I2I edit, multi-reference I2I, T2V A14B,
and I2V A14B, see [Spaceship Snow Workflow](examples/spaceship-snow.md). For practical Wan size
and runtime examples, see [Wan Video](wan-video.md).

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

LoRA support is experimental and route-specific. Capability rows include `supports_lora`, `lora_status`,
`lora_target_roles`, and `lora_validation_profile`. `mapped-unvalidated` means the route has a
mapping and strict loader path, but the exact model/package has not yet passed a visible A/B
validation with a public adapter.

Generation does not download LoRA files. Download LoRA repositories explicitly, then pass a local
`.safetensors` file or a cached Hugging Face adapter id:

```sh
mlxgen download --model lovis93/Flux-2-Multi-Angles-LoRA-v2 --all-files

mlxgen generate \
  --model <compatible-model> \
  --prompt "<prompt from the LoRA model card>" \
  --lora-paths owner/repo:adapter.safetensors \
  --lora-scales 0.9 \
  --output with_lora.png
```

The adapter must match the selected model architecture. For example,
`lovis93/Flux-2-Multi-Angles-LoRA-v2` targets `black-forest-labs/FLUX.2-dev`; MLX-Gen currently
supports FLUX.2 Klein 4B/9B, so that adapter is rejected for Klein routes. The number of
`--lora-scales` values must match the number of `--lora-paths` values exactly. See
[LoRA](lora.md) for the source/no-LoRA/with-LoRA validation method.

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
| Experimental generative reframe / zoom-out | `edit-reference` with reframe support | one image | pass `--reframe-padding` on a model whose capability has `supports_reframe=true` | No |
| Experimental canvas-guided outpaint with adaptive source blend | `edit-reference` with outpaint support | one image | pass `--outpaint-padding` on a model whose capability has `supports_outpaint=true` | No |

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
source model or MLX-Gen optimized package rows passed visual review. The reframe/outpaint
validation profile covers Qwen Image Edit, Qwen Image Edit 2509/2511, and FLUX.2 Klein
4B/9B source/q8/q4 rows.

Latent-only image models such as ERNIE Image Turbo, Z-Image, and base Qwen Image require explicit
`--image-strength` for `latent-img2img`. Base FIBO exposes text-to-image through unified
`mlxgen generate`. FIBO Edit is not exposed as a public `mlxgen generate` capability until it has
passing source-model and optimized-variant visual proof; use the dedicated FIBO Edit command only for
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

Generative reframe is experimental and available through `--reframe-padding` for edit models that advertise
`supports_reframe=true` in `mlxgen capabilities`. It asks the edit model to generate a larger view
from one source image. Padding accepts CSS-style values in `top,right,bottom,left` order. MLX-Gen
builds a larger conditioning canvas with the source pasted at that offset, then asks the edit model
to generate the larger view:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --image input.png \
  --reframe-padding "15%,35%,15%,35%" \
  --prompt "Generatively reframe this image into a wider view. Keep the subject fully visible and extend the background naturally." \
  --steps 16 \
  --seed 42 \
  --output reframed.png
```

This is a generative edit workflow. It may redraw source content, and the prompt still controls
where the model places or reconstructs the subject. Use it for zoom-out, background extension, or
revealing plausible missing object boundaries.

Canvas-guided outpaint is experimental. Use `--outpaint-padding` when you want MLX-Gen to build an
expanded canvas and guide an edit model to fill the larger view:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image input.png \
  --outpaint-padding "5%,35%,5%,35%" \
  --prompt "Outpaint this close crop into a wider realistic shot. Complete the missing background and subject outside the original frame." \
  --negative "text, border, frame, hard seam, duplicate subject" \
  --steps 24 \
  --guidance 4 \
  --seed 42 \
  --output outpaint.png
```

For current FLUX.2 Klein 4B/9B and Qwen Image Edit variants, this route creates a larger temporary
canvas, initializes the new area with edge-extended source context, and runs the edit model on the
expanded canvas. After generation, MLX-Gen compares the generated source window with the original
source. If they are close, it applies a content-aware source blend; if the model has reconstructed
or moved the scene, it skips the blend to avoid ghosted fragments. Source, q8, and q4 rows for
FLUX.2 Klein 4B/9B plus Qwen Image Edit, 2509, and 2511 are documented in
[Image Edit Capabilities](edit-capabilities.md#reframe-and-outpaint) and
[Reframe and Outpaint](reframe-outpaint.md).

This is not the same as a native fill/inpaint pipeline that receives an explicit diffusion mask.
It is not an exact pixel-lock guarantee. MLX-Gen still keeps lower-level FLUX.1 Fill support
separate from the current unified edit-reference canvas route. Z-Image, ERNIE, FIBO, base Qwen
Image, Qwen Image 2512, FLUX.2 Klein Base, latent I2I routes, video routes, and SeedVR2 reject
`--outpaint-padding` before loading weights.

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

ERNIE Image Turbo supports BF16 source weights plus MLX-Gen q8/q4 optimized packages. MLX-Gen also provides single-image latent image-to-image for ERNIE:

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

ERNIE's optional Prompt Enhancer is available with `--use-prompt-enhancer` when the full source snapshot is present. The default `mlxgen download --model baidu/ERNIE-Image-Turbo` command downloads only generation components; run `mlxgen download --model baidu/ERNIE-Image-Turbo --all-files` before using Prompt Enhancer. ERNIE q8/q4 MLX-Gen packages created by `mlxgen prepare` do not include Prompt Enhancer files.

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
| `--flow-shift` | Flow-matching scheduler shift. Defaults to the selected Wan model config. TI2V-5B defaults to `5.0` for native 720p-class runs. A14B defaults to `3.0`. For new 480p-class TI2V-5B checks such as `832x480`, pass `--flow-shift 3`. Python callers use `flow_shift=...`. |
| `--negative-prompt`, `--negative` | If omitted, Wan uses the model's official default negative prompt. Pass `--negative ""` to intentionally run without a negative prompt; this can be better for simple abstract scenes where the default negative prompt adds unwanted texture. |
| `--seed` | Deterministic seed. Repeat with multiple values to create multiple videos. |
| `--progress`, `--no-progress` | Show or disable the CLI video progress bar. The bar advances by denoising step and keeps the requested frame count as context. Default: `--progress true`. |
| `--low-ram` | For Wan CLI runs, clear MLX cache between transformer blocks and denoise steps, release denoisers before decode when the model instance will not be reused for another seed, and clear cache between VAE temporal decode slices. This is intended for memory pressure, not speed. |

Common Wan video sizes:

| Model | Required width/height multiple | Recommended/native quality size | Lower-cost diagnostic sizes | Notes |
| --- | ---: | --- | --- | --- |
| TI2V-5B T2V/I2V | 32 px | `1280x704` or `704x1280` | `832x480`, `480x832`; smaller sizes such as `448x256` are smoke checks only | Text-to-video `1280x720` adjusts to `1280x736`; image-to-video preserves the source image ratio at a nearby supported canvas. |
| T2V-A14B | 16 px | `1280x720` or `720x1280` | `832x480`, `480x832`, `448x256`, `256x448`, `432x240` | Text-to-video only; image input is rejected. |
| I2V-A14B | 16 px | Source-ratio canvas near `1280x720` or `720x1280` | Source-ratio canvas near `832x480`, `448x256`, or `432x240` | Requires one input image; output preserves the source image ratio at a nearby supported canvas. |

The upstream TI2V-5B guidance is 1280x704 or 704x1280, 121 frames, 50 steps, 24 fps, and flow shift
`5.0`. The upstream A14B guidance is 1280x720 or 720x1280, 81 frames, 40 steps, `--guidance 4`,
optional `--guidance-2 3`, flow shift `3.0`, and 16 fps. Lower resolutions, frame counts, or step
counts are useful for routing and prompt checks; for visual TI2V-5B prompt checks, use at least
`832x480` and pass
`--flow-shift 3`.

For a practical 5-second local profile, A14B T2V at `480x240` or `240x480`, `101` frames,
`20` fps, and `20` to `25` steps is a useful quality/speed point on an M5 Max. The documented
starship profile takes about 30 minutes at `480x240`. TI2V-5B at `832x480`, `25` steps, `101`
frames, and `20` fps takes about 12 minutes on the same class of machine; new 480p-class TI2V-5B
checks should include `--flow-shift 3`. TI2V-5B at `1280x704` with the same frames and steps takes
about 35 minutes and should use the default flow shift. See [Wan Video](wan-video.md) for the MP4
assets and frame strips.

For visual checks, use `448x256` or larger for Wan examples. Tiny square canvases such as `128x128`
are not representative of Wan video quality or prompt adherence.

Example outputs at 1280x704, 17 frames, and 20 steps:

![Wan2.2 TI2V 1280x704 text-to-video contact sheet](assets/generation/wan2.2-ti2v-5b-t2v-1280x704-17f-20steps-contact-sheet.png)

![Wan2.2 TI2V first-frame image-to-video contact sheet](assets/generation/wan2.2-ti2v-5b-i2v-bateau-1280x704-17f-20steps-contact-sheet.png)

These panels are examples at the model's spatial scale. Evaluate final visual quality with the
recommended full-resolution, frame-count, and step-count settings for your target model.

## SeedVR2 Upscale Command

SeedVR2 image super-resolution uses `mlxgen upscale`. See [Image Upscaling](upscaling.md) for a
reproducible 5x source/output comparison.

```sh
mlxgen upscale \
  --model AbstractFramework/seedvr2-3b-8bit \
  --image-path input.png \
  --resolution 1024 \
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
| `--model` | Optional SeedVR2 model selector. Defaults to `seedvr2-3b`, the official `ByteDance-Seed/SeedVR2-3B` source model. Use `seedvr2-7b` for the official 7B source model, `AbstractFramework/seedvr2-3b-8bit`, `AbstractFramework/seedvr2-3b-4bit`, `AbstractFramework/seedvr2-7b-8bit`, `AbstractFramework/seedvr2-7b-4bit`, or a local path such as `./models/seedvr2-7b-8bit`. |
| `--quantize` | Optional runtime quantization for source-model runs. Published q8/q4 packages do not need this flag. |
| `--softness` | Optional input smoothing from `0.0` to `1.0`. `0.0` preserves the preprocessed source most directly. Higher values pre-downsample the conditioning image before reconstruction, which can suppress source grain/JPEG texture but can also soften fine details. Try `0.25` to `0.5` for noisy or compressed sources. |
| `--vae-tiling` | Force tiled VAE encode/decode. By default, small outputs stay untiled and large outputs automatically use tiled VAE decode. |
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

Use `prepare` when you need a local MLX-Gen model package. It creates MLX-Gen saved weights,
optional quantized weights, and a generated Hugging Face card.

If a complete local MLX-Gen package exists at `./models/<repo-name>`, a matching Hugging Face handle can
resolve to it before requiring a cache snapshot. This lets applications use stable handles such as
`AbstractFramework/qwen-image-edit-2511-8bit` or
`AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` while still running from local files.

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

MLX-Gen model packages use the MLX/mflux saved-weight layout and MLX quantization tensors. They are intended for MLX-Gen and compatible mflux code, not for direct Diffusers or Transformers `from_pretrained()` loading.
