# Getting Started

This guide covers the shortest path from a fresh MLX-Gen install to a local image or video generation run.

## Install

Install MLX-Gen as a `uv` tool:

```sh
uv tool install --upgrade mlx-gen
```

Check the command surface:

```sh
mlxgen --help
```

The top-level command shows the public workflows:

- `mlxgen generate` for image generation and supported video generation.
- `mlxgen capabilities` for inspecting model tasks, image-to-image modes, and option support without loading weights.
- `mlxgen download` for explicit Hugging Face cache downloads.
- `mlxgen prepare` for reusable local MLX-Gen model packages.

## Prepare Model Files

Before generation, either download the source repository into the local Hugging Face cache or create
a reusable local MLX-Gen package. Generation uses those local files and does not start a download.

Download into the Hugging Face cache:

```sh
mlxgen download --model z-image-turbo
```

Create a reusable local MLX-Gen package with quantized weights and a generated Hugging Face model card:

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  --quantize 8
```

Use `mlxgen prepare` when you need a local MLX-Gen model package. There is no separate MLX-Gen
`save` workflow in the public documentation.

`HF_HUB_ENABLE_HF_TRANSFER=1` is optional. It can speed up explicit Hugging Face download or prepare commands when the accelerated transfer backend is available, but it is not required to authorize downloads.

Bonsai Image checkpoints are already packed MLX artifacts. Use `mlxgen download` for Bonsai, not
`mlxgen prepare`.

## Generate An Image

Run generation from a cached alias or repository:

```sh
mlxgen generate \
  --model z-image-turbo \
  --prompt "A puffin standing on a cliff" \
  --width 1280 \
  --height 500 \
  --seed 42 \
  --steps 9 \
  --quantize 8
```

Run generation from a local MLX-Gen model package:

```sh
mlxgen generate \
  --model ./models/qwen-image-8bit \
  --family qwen \
  --prompt "A clean studio product photo" \
  --output image.png
```

`--family` is useful when a local path or custom repository name does not contain a recognizable model-family name.

Run Bonsai Image from its pre-packed ternary checkpoint:

```sh
mlxgen download --model prism-ml/bonsai-image-ternary-4B-mlx-2bit

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

## Edit An Image

Pass one or more input images to the same `generate` command. MLX-Gen routes to the right
image-to-image mode from the model and image inputs:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2509-8bit \
  --image input.png \
  --prompt "Turn the room into a pencil sketch" \
  --steps 20 \
  --seed 42 \
  --output edited.png
```

Use `mlxgen capabilities --model <model>` to check whether a model supports latent img2img,
edit/reference image-to-image, or multi-reference image-to-image. `--image-strength` is for latent
img2img variation only; edit/reference models do not use it.

LoRA support is experimental. Use the same capabilities command before LoRA runs. LoRA adapters
must match the selected model family, and a visible source/no-LoRA/with-LoRA comparison is required
before treating an adapter as validated. See [LoRA](lora.md) for the current contract.

Use `qwen-image-edit` for the original single-reference edit checkpoint; use
`qwen-image-edit-2509` or `qwen-image-edit-2511` when you need multi-reference editing.
Current contact sheets and commands for Qwen Image Edit, Qwen Image Edit 2509/2511, and FLUX.2
Klein source/q8/q4 packages are published in [Image Edit Capabilities](edit-capabilities.md).
Use `--negative-prompt` or `--negative` to block concrete failure modes such as crop, blur, text,
or unwanted color. Qwen edit models use the official blank negative-prompt behavior by default when
guidance is above `1`, but explicit negative prompts are useful for stricter edits.

Ordinary image-to-image preserves the first source image's aspect ratio by default. `--width` and
`--height` act as a size target under `--canvas-policy source-aspect`; pass
`--canvas-policy exact-resize` only when you intentionally want the exact requested canvas.

Reframe and outpaint are experimental generative edit workflows. Use `--reframe-padding` when you
want an edit model to generate a wider view from one source image.
This is a generative edit:

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

Use experimental `--outpaint-padding` when you want MLX-Gen to expand the canvas and guide a
supported edit model to fill the larger view:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image input.png \
  --outpaint-padding "5%,35%,5%,35%" \
  --prompt "Outpaint this close crop into a wider realistic shot. Complete the missing subject and background outside the original frame." \
  --negative "text, border, frame, hard seam, duplicate subject" \
  --steps 24 \
  --guidance 4 \
  --seed 42 \
  --output outpaint.png
```

For current FLUX.2 Klein 4B/9B and Qwen Image Edit variants, this route uses an edge-extended
conditioning canvas and an adaptive source blend. If the generated source window still matches the
original source, MLX-Gen blends source detail back in; if the model has reconstructed the scene, it
skips the blend to avoid ghosted fragments. This is not a native fill/inpaint pipeline with an
explicit diffusion mask, and it is not an exact pixel-lock guarantee.

Current experimental reframe and outpaint proof assets are published in
[Image Edit Capabilities](edit-capabilities.md) and [Reframe and Outpaint](reframe-outpaint.md).

For a complete image workflow with included outputs, see the
[spaceship snow example](examples/spaceship-snow.md). It covers text-to-image, two single-image
edits, and multi-reference image-to-image with copy/pasteable commands.

## Upscale An Image

SeedVR2 is available through `mlxgen upscale`. It is a diffusion super-resolution/restoration
model: it can increase image size while also cleaning noise and reconstructing detail.

Use a published q8 package for a smaller reusable SeedVR2 model:

```sh
mlxgen download --model AbstractFramework/seedvr2-3b-8bit
```

Use an integer `--resolution` when you want to set the shorter output edge while preserving aspect
ratio:

```sh
mlxgen upscale \
  --model AbstractFramework/seedvr2-3b-8bit \
  --image-path input.png \
  --resolution 1024 \
  --metadata \
  --output input_short_edge_1024.png
```

Use a scale factor when you want a direct multiplier:

```sh
mlxgen upscale \
  --model AbstractFramework/seedvr2-3b-8bit \
  --image-path input.png \
  --resolution 2x \
  --softness 0.25 \
  --metadata \
  --output input_2x.png
```

For example, a `320x192` source becomes `640x384` with `--resolution 2x` and `960x576` with
`--resolution 3x`. For visual quality checks, choose a target that changes the dimensions
materially; a near-same-size target is mainly useful for restoration/denoising checks.

SeedVR2 keeps small outputs on the untiled VAE path for image quality and automatically uses tiled
VAE decode for large outputs. `--softness` controls input smoothing before reconstruction: `0.0`
keeps the source conditioning most direct, while higher values suppress grain or JPEG texture at
the cost of softer fine detail. If the source has visible grain in smooth regions, try
`--softness 0.25` to `0.5`. Add `--vae-tiling` when you also want tiled VAE encoding or the same
tiled path for smaller outputs.

The `seedvr2` and `seedvr2-3b` aliases resolve to the official upstream 3B checkpoint. To run that
source model directly, download it and pass its full handle:

```sh
mlxgen download --model ByteDance-Seed/SeedVR2-3B

mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-3B \
  --image-path input.png \
  --resolution 2x \
  --seed 42 \
  --metadata \
  --output input_seedvr2_official_3b_2x.png
```

Use `seedvr2-7b` for the official 7B source model after downloading
`ByteDance-Seed/SeedVR2-7B`. See [Image Upscaling](upscaling.md) for 5x SeedVR2 3B and 7B
examples, source/q8/q4 comparisons, package sizes, and measured memory profiles.

## Generate A Video

Wan2.2 support is available as an initial video backend. Download the source snapshot first, then
run `mlxgen generate`. TI2V-5B infers text-to-video when no image is supplied and image-to-video
when one image is supplied. The fixed A14B models are stricter: T2V-A14B rejects image inputs and
I2V-A14B requires one image.

Use TI2V-5B when you want the smaller text-to-video or first-frame image-to-video path:

```sh
mlxgen download --model Wan-AI/Wan2.2-TI2V-5B-Diffusers

mlxgen generate \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --prompt "A short cinematic video of a glowing orange glass sphere floating above teal water" \
  --width 1280 \
  --height 704 \
  --frames 121 \
  --steps 50 \
  --guidance 5 \
  --fps 24 \
  --seed 321 \
  --output video.mp4
```

Use T2V-A14B when you want the larger Diffusers-style two-transformer A14B text-to-video path:

```sh
mlxgen download --model Wan-AI/Wan2.2-T2V-A14B-Diffusers

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
  --seed 321 \
  --output video.mp4
```

`--guidance-2` is optional and only applies to Wan A14B's low-noise `transformer_2` stage. When it
and `--guidance` are both omitted, MLX-Gen uses the model's two-stage defaults. For T2V-A14B that
means `--guidance 4` for the high-noise stage and `--guidance-2 3` for the low-noise stage. If you
set `--guidance` and omit `--guidance-2`, the low-noise stage follows your `--guidance` value.

For image-to-video, pass one input image. TI2V-5B uses first-frame conditioning
route:

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

A14B I2V uses the separate `Wan-AI/Wan2.2-I2V-A14B-Diffusers` snapshot and follows Diffusers'
concatenated image-condition latent path:

```sh
mlxgen download --model Wan-AI/Wan2.2-I2V-A14B-Diffusers

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

TI2V-5B image-to-video uses the Diffusers first-frame latent-conditioning path. The separate A14B I2V route requires the complete I2V source snapshot before generation. Current Wan video support can produce MP4 output; quality, speed, and practical defaults depend strongly on model family, prompt, size, and frame count.

Wan does not have a separate duration option. Control duration with `--frames` and `--fps`: duration
is `frames / fps`, so `--frames 121 --fps 24` is about 5.04 seconds and `--frames 81 --fps 16` is
about 5.06 seconds. Wan frame counts must be `4n + 1`; MLX-Gen adjusts other values to that shape.

Width and height are adjusted up to the selected Wan model's VAE/patch multiple. For image-to-video,
MLX-Gen also preserves the source image aspect ratio: the requested `--width` and `--height` act as
a size target, and the actual output canvas is resolved from the input image ratio and model
multiples before generation.

| Model | Required multiple | Recommended/native size | Practical lower-cost sizes |
| --- | ---: | --- | --- |
| TI2V-5B T2V/I2V | 32 px | `1280x704` or `704x1280` | `832x480`, `480x832`, `448x256`, `256x448` |
| T2V-A14B | 16 px | `1280x720` or `720x1280` | `832x480`, `480x832`, `448x256`, `256x448`, `432x240` |
| I2V-A14B | 16 px | `1280x720` or `720x1280` | `832x480`, `480x832`, `448x256`, `256x448`, `432x240` |

For TI2V-5B text-to-video, `1280x720` adjusts to `1280x736`, and `432x240` adjusts to `448x256`.
For A14B text-to-video, `1280x720`, `832x480`, `448x256`, and `432x240` are valid multiples of 16.
For image-to-video, saved metadata records the requested size, source image size, and resolved
output size.

Use `448x256` or larger for visual Wan checks. Very small square canvases such as `128x128` are not
representative of Wan video quality or prompt adherence.

For practical 5-second local clips on an M5 Max, Wan A14B at `480x240` or `240x480`, `101` frames,
`20` fps, and `20` to `25` steps is a useful starting point and takes about 30 minutes in the
documented starship profile. TI2V-5B at `832x480`, `25` steps, `101` frames, and `20` fps takes
about 12 minutes, while TI2V-5B at `1280x704` with the same frames and steps takes about 35
minutes. See [Wan Video](wan-video.md) for MP4 examples and frame strips.

Wan uses the official model negative prompt by default. For simple abstract tests, pass
`--negative-prompt ""` to run without it.

Spatial-scale sanity outputs at 1280x704, 17 frames, and 20 steps:

![Wan2.2 TI2V 1280x704 text-to-video contact sheet](assets/generation/wan2.2-ti2v-5b-t2v-1280x704-17f-20steps-contact-sheet.png)

![Wan2.2 TI2V first-frame image-to-video contact sheet](assets/generation/wan2.2-ti2v-5b-i2v-bateau-1280x704-17f-20steps-contact-sheet.png)

## Next Steps

- See [Model Management](model-management.md) for the full download, prepare, and runtime failure contract.
- See [API And CLI](api.md) for the supported command surface and Python integration notes.
- See [Spaceship Snow Workflow](examples/spaceship-snow.md) for a reproducible image and Wan A14B video example with included assets.
- See [Wan Video](wan-video.md) for practical Wan2.2 sizing and 5-second comparison assets.
- See [Quantization](quantization.md) for q4/q8 behavior, Bonsai low-bit packed support, Qwen/ERNIE mixed q4/q8 policies, and Wan mixed q8/BF16 packages.
- See [Troubleshooting](troubleshooting.md) when a required artifact is missing or a local path cannot be classified.
