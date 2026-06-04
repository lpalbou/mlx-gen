# MLX-Gen

[![mlx-gen](https://img.shields.io/pypi/v/mlx-gen?label=mlx-gen&logo=pypi&logoColor=white)](https://pypi.org/project/mlx-gen/)
[![MLX](https://img.shields.io/pypi/v/mlx?label=MLX&logo=pypi&logoColor=white)](https://pypi.org/project/mlx/)
[![CI](https://github.com/lpalbou/mlx-gen/actions/workflows/tests.yml/badge.svg)](https://github.com/lpalbou/mlx-gen/actions/workflows/tests.yml)

MLX-Gen is a local image and video generation runtime for Apple Silicon and MLX. It exposes one
`mlxgen` command for text-to-image, image-to-image, text-to-video, image-to-video, model download,
model preparation, quantized local folders, and application progress callbacks.

> [!IMPORTANT]
> MLX-Gen started as a fork of [mflux](https://github.com/filipstrand/mflux). Most credit for the
> current codebase goes to Filip Strand and the original mflux contributors. This project keeps
> that attribution visible while publishing independently as `mlx-gen` and evolving the `mlxgen`
> command surface for current Apple Silicon workflows.

![MLX-Gen workflow example](https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/assets/examples/spaceship-snow/mlx-gen-example.png)

## What It Does

MLX-Gen runs supported Hugging Face and prepared MLX-Gen model folders without starting network
downloads during generation. You explicitly download or prepare models first, then generation is a
cache-only operation suitable for desktop apps, workflow engines, and long-running local jobs.

The main capabilities are:

- text-to-image generation with Qwen Image, FLUX.2 Klein, Z-Image, ERNIE Image Turbo, Bonsai Image,
  FIBO, and related prepared folders;
- image-to-image modes, including latent img2img, instruction/reference edits, and multi-reference
  edits where the selected model supports them;
- Wan2.2 text-to-video and image-to-video, including the TI2V-5B q8 package plus A14B
  T2V/I2V prepared BF16 and mixed q8/BF16 packages;
- explicit `download` and `prepare` workflows for reproducible local model folders;
- JSON model capability inspection before starting a heavy run;
- shared progress events for applications embedding MLX-Gen.

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

Inspect model capabilities before a run:

```sh
mlxgen capabilities --model AbstractFramework/flux.2-klein-9b-8bit
```

Create a reusable local prepared folder:

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  --quantize 8
```

`mlxgen generate` does not download missing files. If something is not cached, MLX-Gen raises a
clear `DownloadRequiredError` with the command to run.

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

## Published Models

Prepared MLX-Gen model folders are published under the
[AbstractFramework organization on Hugging Face](https://huggingface.co/AbstractFramework). Current
published examples include:

- `AbstractFramework/flux.2-klein-4b-4bit`
- `AbstractFramework/flux.2-klein-4b-8bit`
- `AbstractFramework/flux.2-klein-9b-4bit`
- `AbstractFramework/flux.2-klein-9b-8bit`
- `AbstractFramework/qwen-image-2512-4bit`
- `AbstractFramework/qwen-image-2512-8bit`
- `AbstractFramework/qwen-image-edit-2511-4bit`
- `AbstractFramework/qwen-image-edit-2511-8bit`
- `AbstractFramework/z-image-turbo-4bit`
- `AbstractFramework/z-image-turbo-8bit`
- `AbstractFramework/ernie-image-turbo-4bit`
- `AbstractFramework/ernie-image-turbo-8bit`
- `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit`
- `AbstractFramework/wan2.2-t2v-a14b-diffusers-bf16`
- `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit`
- `AbstractFramework/wan2.2-i2v-a14b-diffusers-bf16`
- `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit`

Use `mlxgen download --model <repo-id>` to cache a published model, or pass the repository id
directly to `mlxgen generate` after it is cached. See
[docs/quantization.md](docs/quantization.md) for the complete current package matrix with source
sizes, prepared package sizes, task coverage, and quantization notes.

For Wan2.2 TI2V-5B, the current published AbstractFramework prepared package is q8. The upstream
source snapshot is about 31.9 GiB and stores the transformer and VAE as FP32 while the UMT5 text
encoder is BF16. MLX-Gen loads and prepares Wan transformer/VAE weights at BF16 runtime precision,
so a future prepared BF16 TI2V-5B package would mainly be a smaller source-equivalent package, not
a separate runtime-memory optimization. Wan TI2V-5B q4 or mixed q4/q8 remains under validation and
is not published as a supported package.

## Wan A14B Measurements

Wan A14B was measured on an Apple M5 Max with 128 GB unified memory. The published-card validation
uses small, repeatable low-RAM runs and records full-process Darwin physical footprint, RSS, MLX
allocator peak, and generation time. These are validation-profile measurements, not a guarantee for
every full-size production prompt.

| Model | Package | Disk | Physical Peak | Max RSS | MLX Peak | Time | Profile |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Wan2.2 T2V-A14B | BF16 | 64.3 GiB | 33.0 GiB | 31.8 GiB | 27.7 GiB | 152.7 s | 384x224, 33 frames, 12 steps, 8 fps |
| Wan2.2 T2V-A14B | mixed q8/BF16 | 39.7 GiB | 20.7 GiB | 19.5 GiB | 15.5 GiB | 154.8 s | 384x224, 33 frames, 12 steps, 8 fps |
| Wan2.2 I2V-A14B | BF16 | 64.1 GiB | 33.7 GiB | 31.8 GiB | 28.2 GiB | 228.2 s | 384x384, 33 frames, 12 steps, 8 fps |
| Wan2.2 I2V-A14B | mixed q8/BF16 | 39.7 GiB | 21.5 GiB | 19.6 GiB | 15.9 GiB | 242.2 s | 384x384, 33 frames, 12 steps, 8 fps |

In these runs, mixed q8/BF16 reduces disk usage by about 38% versus prepared BF16 folders and
reduces full-process physical peak memory by about 36-37%. It is not documented as a speed
improvement. See [docs/quantization.md](docs/quantization.md) for model-family quantization details.

## Ecosystem

MLX-Gen is used as the local Apple Silicon generation backend for:

- [AbstractVision](https://github.com/lpalbou/abstractvision), the vision/generation layer of the
  AbstractFramework ecosystem;
- [AbstractFramework](https://github.com/lpalbou/abstractframework), the broader framework for
  local agentic and generative workflows;
- [AbstractFlow](https://github.com/lpalbou/abstractflow), a visual orchestration layer that can
  compose generative capabilities with persistent agentic tasks.

MLX-Gen remains useful as a standalone CLI package, but its cache-only runtime behavior, capability
inspection, prepared model folders, and progress callbacks are designed so applications can embed it
without surprise network transfers or ambiguous model routing.

## Documentation

- [Getting started](docs/getting-started.md): installation and first runs.
- [API and CLI](docs/api.md): command surface, router behavior, image-to-image modes, Wan video sizes, capabilities, and Python entry points.
- [Example workflow](docs/examples/spaceship-snow.md): reproducible image and video commands.
- [Model management](docs/model-management.md): download, prepare, cache-only runtime policy.
- [Quantization](docs/quantization.md): q8/q4/BF16 policies and measurements.
- [Python integration](docs/python-integration.md): embedding, progress callbacks, and AbstractVision notes.
- [FAQ](docs/faq.md): recurring questions, image-to-image mode selection, outpaint/reframe status, Wan resolutions, and usage limits.
- [Troubleshooting](docs/troubleshooting.md): common setup and runtime failures.
- [Acknowledgements](ACKNOWLEDGEMENTS.md): upstream mflux and model-community credits.

## License

MLX-Gen is MIT licensed. Model weights remain governed by their original licenses and access terms.
