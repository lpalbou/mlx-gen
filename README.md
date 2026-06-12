# MLX-Gen

[![mlx-gen](https://img.shields.io/pypi/v/mlx-gen?label=mlx-gen&logo=pypi&logoColor=white)](https://pypi.org/project/mlx-gen/)
[![MLX](https://img.shields.io/pypi/v/mlx?label=MLX&logo=pypi&logoColor=white)](https://pypi.org/project/mlx/)
[![CI](https://github.com/lpalbou/mlx-gen/actions/workflows/tests.yml/badge.svg)](https://github.com/lpalbou/mlx-gen/actions/workflows/tests.yml)

MLX-Gen is a local image and video generation runtime for Apple Silicon and MLX. It exposes
`mlxgen` for text-to-image, image-to-image, text-to-video, image-to-video, model download, model
preparation, SeedVR2 image upscaling, optimized quantized model variants, and application progress
callbacks.

> [!IMPORTANT]
> MLX-Gen started as a fork of [mflux](https://github.com/filipstrand/mflux). Most credit for the
> current codebase goes to Filip Strand and the original mflux contributors. MLX-Gen keeps that
> foundation and focuses on making local generation predictable for real applications: users
> download or prepare models before running a job, then use one `mlxgen` command to generate,
> inspect supported modes, and compare generated examples and measured results. MLX-Gen contributes
> by adding tested T2I/I2I routes, Qwen Image Edit 2509/2511 routing and parity fixes, Bonsai Image
> support, Wan2.2 text-to-video and image-to-video support, model-specific mixed quantization
> policies, published quantized Hugging Face repos optimized for MLX-Gen, and progress callbacks
> for apps. The fork exists so AbstractVision and related AbstractFramework projects can move
> quickly without losing the option to merge useful changes back upstream if that becomes valuable
> for the wider mflux community.

![MLX-Gen workflow example](https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/assets/examples/spaceship-snow/mlx-gen-example.png)

This screenshot shows [AbstractFlow](https://github.com/lpalbou/abstractflow), a visual workflow
authoring tool in the [AbstractFramework](https://abstractframework.ai/) ecosystem. AbstractFlow can
orchestrate generative image/video capabilities exposed through AbstractVision and AbstractCore; the
same MLX-Gen steps can also be run directly with the command-line examples below.

## What It Does

MLX-Gen runs supported Hugging Face source models after you download them locally. It also runs
quantized model variants that are published on Hugging Face for MLX-Gen. Some of those variants keep
precision-sensitive layers at 8-bit or BF16 instead of blindly quantizing every layer, which is why
they are published as MLX-Gen-specific repos. You explicitly download or prepare models first, then
generation uses only files already on disk, which suits desktop apps, workflow engines, and
long-running local jobs.

The main capabilities are:

- text-to-image generation with Qwen Image, FLUX.2 Klein, Z-Image, ERNIE Image Turbo, Bonsai Image,
  FIBO, and their optimized quantized variants where available;
- image-to-image modes, including latent img2img, instruction/reference edits, multi-reference
  edits, and experimental reframe/outpaint workflows where the selected model supports them;
- Wan2.2 text-to-video and image-to-video, including TI2V-5B BF16/q8 packages plus A14B
  T2V/I2V BF16 and mixed q8/BF16 packages; Wan I2V resolves output size from the source
  image aspect ratio so inputs are not stretched into a mismatched canvas;
- SeedVR2 image super-resolution through `mlxgen upscale`, with official 3B/7B source support,
  published q8/q4 packages, shortest-edge target sizing, and explicit scale factors such as `2x`
  and `3x`;
- explicit `download` and `prepare` workflows for local MLX-Gen model packages;
- JSON model capability inspection before starting a heavy run;
- experimental LoRA routing and strict adapter application checks, with model-card compatibility
  preflight when cached adapter metadata is available and exact q8 proof rows for Qwen Image Edit
  original/2509/2511, Qwen Image 2512, Z-Image Turbo, FLUX.2 Klein 9B edit, ERNIE Image Turbo
  text-to-image, and all current Wan q8 video routes; the LoRA guide now includes the documented
  `720p` Wan q8-vs-BF16 LightX2V keyframe comparison, readable `41`-frame M5 Max progress
  matrices, same-seed no-LoRA-versus-Lightning A/B sheets, a `240p`-versus-`480p` T2V sweep, and
  time/RSS tables for T2V and I2V; base Qwen Image remains experimental, and Bonsai LoRA stays
  fail-closed;
- shared progress events for applications embedding MLX-Gen.

Use `mlxgen capabilities --model ...` before long image-edit runs. Capability output describes the
available route; validation reports and contact sheets describe whether an exact source handle or
MLX-Gen optimized package passed a visual release gate. Release evidence should use true handles such as
`briaai/Fibo-Edit` or `AbstractFramework/flux.2-klein-9b-8bit`, not short aliases.

## Install

Install with `uv`:

```sh
uv tool install --upgrade mlx-gen
```

Or install into an environment:

```sh
python -m pip install -U mlx-gen
```

Check the command surface:

```sh
mlxgen --help
```

## First Commands

Download model files explicitly:

```sh
mlxgen download --model AbstractFramework/flux.2-klein-9b-8bit
```

Generate an image:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --prompt "A cinematic wide shot of a compact sci-fi spaceship resting in deep snow on a frozen alien planet" \
  --width 768 \
  --height 432 \
  --steps 24 \
  --guidance 1.0 \
  --seed 6107 \
  --output spaceship.png
```

Upscale an image with SeedVR2:

```sh
mlxgen download --model AbstractFramework/seedvr2-3b-8bit

mlxgen upscale \
  --model AbstractFramework/seedvr2-3b-8bit \
  --image-path input.png \
  --resolution 2x \
  --softness 0.25 \
  --metadata \
  --output input_2x.png
```

For SeedVR2, an integer `--resolution` is the target shorter edge while values such as `2x` and
`3x` are scale factors. Both modes preserve the source aspect ratio. Use `--softness 0.25` to
`0.5` when the source has visible grain in smooth areas. Small outputs use the untiled VAE path;
large outputs automatically use tiled VAE decode, and `--vae-tiling` also forces tiled encode. See
[docs/upscaling.md](docs/upscaling.md) for a reproducible 5x SeedVR2 comparison.

Inspect model capabilities before a run:

```sh
mlxgen capabilities --model AbstractFramework/flux.2-klein-9b-8bit
```

Capabilities are route contracts: they show which tasks, I2I modes, image counts, and options the
selected model can dispatch. For release QA evidence on exact packages, use:

```sh
mlxgen validation --model AbstractFramework/qwen-image-edit-2509-8bit
```

LoRA support is experimental. For LoRA work, inspect `supports_lora` and `lora_status` in
`mlxgen capabilities`, download the adapter explicitly with `mlxgen download`, and use an adapter
trained for the selected model family. Current exact proof rows cover original Qwen Image Edit,
Qwen Image Edit 2509/2511, Qwen Image 2512, Z-Image Turbo, FLUX.2 Klein 9B edit, and ERNIE Image
Turbo text-to-image, plus the current Wan q8 public video routes. The LoRA guide also includes the
current LightX2V 4-step A14B timing comparison against the practical original Wan profiles,
same-seed no-LoRA-versus-Lightning A/B sheets, a `240p`-versus-`480p` T2V sweep, and copy-paste
download and T2V/I2V commands for `lightx2v/Wan2.2-Lightning`.
For example, a FLUX.2-dev LoRA is not accepted for FLUX.2 Klein. See [docs/lora.md](docs/lora.md)
for the A/B validation method.

Create a local MLX-Gen model package, for example an 8-bit Qwen Image package:

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  --quantize 8
```

`mlxgen generate` does not download missing files. If something is not cached, MLX-Gen raises a
clear `DownloadRequiredError` with the command to run. A complete local MLX-Gen package at
`./models/<repo-name>` can also satisfy a matching Hugging Face handle such as
`AbstractFramework/qwen-image-edit-2511-8bit`.

## Reproducible Example

The docs include a complete model-backed spaceship workflow:

- T2I: generate a spaceship in the snow.
- I2I edit: turn it into a pencil sketch.
- I2I edit: crash the same spaceship in the snow.
- I2I multi-reference: combine the crash layout and pencil-sketch style.
- T2V A14B: generate a spaceship taking off from a snow planet.
- I2V A14B: animate the generated spaceship taking off from the source image.

See [docs/examples/spaceship-snow.md](docs/examples/spaceship-snow.md) for the exact commands and
included assets.

![Spaceship mode contact sheet](https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/assets/examples/spaceship-snow/spaceship_modes_real_generation_contact_sheet.png)

For current image-edit contact sheets, command logs, and model/package status across Qwen Image
Edit, Qwen Image Edit 2509/2511, FLUX.2 Klein, and latent I2I models, see
[docs/edit-capabilities.md](docs/edit-capabilities.md).
For a plain-language guide to latent img2img, instruction edit, multi-reference composition,
generative reframe, and outpaint, see [docs/image-edit-modes.md](docs/image-edit-modes.md).

## Published Models

Quantized and BF16 model variants optimized for MLX-Gen are published under the
[AbstractFramework organization on Hugging Face](https://huggingface.co/AbstractFramework). Current
published packages include:

FLUX.2 Klein:

- `AbstractFramework/flux.2-klein-4b-4bit`
- `AbstractFramework/flux.2-klein-4b-8bit`
- `AbstractFramework/flux.2-klein-9b-4bit`
- `AbstractFramework/flux.2-klein-9b-8bit`
- `AbstractFramework/flux.2-klein-base-4b-4bit`
- `AbstractFramework/flux.2-klein-base-4b-8bit`
- `AbstractFramework/flux.2-klein-base-9b-4bit`
- `AbstractFramework/flux.2-klein-base-9b-8bit`

Qwen Image and Qwen Image Edit:

- `AbstractFramework/qwen-image-4bit`
- `AbstractFramework/qwen-image-8bit`
- `AbstractFramework/qwen-image-2512-4bit`
- `AbstractFramework/qwen-image-2512-8bit`
- `AbstractFramework/qwen-image-edit-4bit`
- `AbstractFramework/qwen-image-edit-8bit`
- `AbstractFramework/qwen-image-edit-2509-4bit`
- `AbstractFramework/qwen-image-edit-2509-8bit`
- `AbstractFramework/qwen-image-edit-2511-4bit`
- `AbstractFramework/qwen-image-edit-2511-8bit`

Z-Image, ERNIE, and FIBO:

- `AbstractFramework/z-image-4bit`
- `AbstractFramework/z-image-8bit`
- `AbstractFramework/z-image-turbo-4bit`
- `AbstractFramework/z-image-turbo-8bit`
- `AbstractFramework/ernie-image-turbo-4bit`
- `AbstractFramework/ernie-image-turbo-8bit`
- `AbstractFramework/fibo-4bit`
- `AbstractFramework/fibo-8bit`

SeedVR2 upscaling:

- `AbstractFramework/seedvr2-3b-4bit`
- `AbstractFramework/seedvr2-3b-8bit`
- `AbstractFramework/seedvr2-7b-4bit`
- `AbstractFramework/seedvr2-7b-8bit`

SeedVR2 7B can also run from the official `ByteDance-Seed/SeedVR2-7B` source model or from the
published q8/q4 package handles above. See
[docs/upscaling.md](docs/upscaling.md) and [docs/quantization.md](docs/quantization.md) for the
validated 7B source/q8/q4 profile.

Wan2.2 video:

- `AbstractFramework/wan2.2-ti2v-5b-diffusers-bf16`
- `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit`
- `AbstractFramework/wan2.2-t2v-a14b-diffusers-bf16`
- `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit`
- `AbstractFramework/wan2.2-i2v-a14b-diffusers-bf16`
- `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit`

Use `mlxgen download --model <repo-id>` to cache a published model, then pass the repository id to
the relevant command: `mlxgen generate` for image/video generation or `mlxgen upscale` for SeedVR2
upscaling. See
[docs/quantization.md](docs/quantization.md) for the complete current package matrix with source
sizes, optimized package sizes, task coverage, and quantization notes.

For Wan2.2 TI2V-5B, the published BF16 MLX-Gen package is 21.2 GiB versus 31.9 GiB for the
upstream source snapshot. It is mainly a smaller reusable source-equivalent package because
MLX-Gen already loads Wan transformer/VAE weights at BF16 runtime precision. The published q8
package is 16.9 GiB. In the documented 1280x704 benchmark profile, q8 reduced logical model
footprint and MLX allocator peak but did not reduce full-process physical peak memory. Wan TI2V-5B
q4 or mixed q4/q8 is not published as a supported package. See the exact benchmark profile in
[docs/quantization.md](docs/quantization.md).

## Wan A14B Measurements

Wan A14B was measured on an Apple M5 Max with 128 GB unified memory. The published-card benchmark
uses small, repeatable low-RAM runs and records full-process Darwin physical footprint, RSS, MLX
allocator peak, and generation time. Use these values for the listed profiles; memory and runtime
scale with resolution, frame count, step count, cache settings, and image-to-video conditioning.

| Model | Package | Disk | Physical Peak | Max RSS | MLX Peak | Time | Profile |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Wan2.2 T2V-A14B | BF16 | 64.1 GiB | 33.0 GiB | 31.8 GiB | 27.7 GiB | 152.7 s | 384x224, 33 frames, 12 steps, 8 fps |
| Wan2.2 T2V-A14B | mixed q8/BF16 | 39.5 GiB | 20.7 GiB | 19.5 GiB | 15.5 GiB | 154.8 s | 384x224, 33 frames, 12 steps, 8 fps |
| Wan2.2 I2V-A14B | BF16 | 64.1 GiB | 33.7 GiB | 31.8 GiB | 28.2 GiB | 228.2 s | 384x384, 33 frames, 12 steps, 8 fps |
| Wan2.2 I2V-A14B | mixed q8/BF16 | 39.5 GiB | 21.5 GiB | 19.6 GiB | 15.9 GiB | 242.2 s | 384x384, 33 frames, 12 steps, 8 fps |

In these runs, mixed q8/BF16 reduces disk usage by about 38% versus BF16 MLX-Gen packages and
reduces full-process physical peak memory by about 36-37%. It is not documented as a speed
improvement. See
[docs/quantization.md](docs/quantization.md) for model-family quantization details and metrics JSON.
The 0.18.11 release also validates the published A14B q8 T2V/I2V handles on a 41-frame,
15-step, 480x240-target profile with saved MP4/contact-sheet evidence in the quantization docs.

## Ecosystem

MLX-Gen is used as the local Apple Silicon generation backend for:

- [AbstractVision](https://github.com/lpalbou/abstractvision), the direct integration layer for
  abstracting generative image and video capabilities across local and hosted providers;
- [AbstractCore](https://abstractcore.ai/), which can expose OpenAI-compatible endpoints backed by
  AbstractVision providers, including image and video capabilities;
- [AbstractFlow](https://github.com/lpalbou/abstractflow), a visual workflow authoring layer that
  can compose generative image/video nodes alongside other media, text, and agent workflows.

MLX-Gen remains useful as a standalone CLI package, and it is also designed for applications: jobs
run from model files already on disk, apps can inspect supported modes before loading weights, and
progress callbacks make long runs observable.

## Documentation

- [Getting started](docs/getting-started.md): installation, first runs, SeedVR2 upscaling, and Wan video.
- [API and CLI](docs/api.md): command surface, router behavior, image-to-image modes, experimental generative reframe, backend-specific outpaint, SeedVR2 sizing, Wan video sizes, capabilities, and Python entry points.
- [Image edit modes](docs/image-edit-modes.md): what latent img2img, edit-reference, multi-reference, generative reframe, and outpaint mean in practice, with examples.
- [Wan video](docs/wan-video.md): practical Wan2.2 T2V/I2V sizing and 5-second M5 Max comparison clips.
- [Example workflow](docs/examples/spaceship-snow.md): reproducible image and video commands.
- [Image upscaling](docs/upscaling.md): SeedVR2 sizing, published 3B/7B q8/q4 package usage, quality controls, and 5x source/output comparisons.
- [Image edit capabilities](docs/edit-capabilities.md): image-edit contact sheets, exact model/package status, and command logs.
- [Reframe and outpaint](docs/reframe-outpaint.md): experimental `--reframe-padding` and `--outpaint-padding` routes with the mixed June 8 profile plus the current FLUX.2 Klein base source-model proof.
- [Model management](docs/model-management.md): download, prepare, and run from local model files.
- [Quantization](docs/quantization.md): q8/q4/BF16 policies and measurements.
- [Python integration](docs/python-integration.md): embedding, progress callbacks, and AbstractVision/AbstractCore notes.
- [FAQ](docs/faq.md): recurring questions, image-to-image mode selection, SeedVR2 sizing, Qwen edit variants, negative prompts, experimental generative reframe, experimental canvas outpaint, Wan resolutions, and usage limits.
- [Troubleshooting](docs/troubleshooting.md): common setup and runtime failures.
- [Acknowledgements](ACKNOWLEDGEMENTS.md): upstream mflux and model-community credits.

## License

MLX-Gen is MIT licensed. Model weights remain governed by their original licenses and access terms.
