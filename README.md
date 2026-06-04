# MLX-Gen

[![mlx-gen](https://img.shields.io/pypi/v/mlx-gen?label=mlx-gen&logo=pypi&logoColor=white)](https://pypi.org/project/mlx-gen/)
[![MLX](https://img.shields.io/pypi/v/mlx?label=MLX&logo=pypi&logoColor=white)](https://pypi.org/project/mlx/)
[![CI](https://github.com/lpalbou/mlx-gen/actions/workflows/tests.yml/badge.svg)](https://github.com/lpalbou/mlx-gen/actions/workflows/tests.yml)

### About

Run state-of-the-art generative image and video models locally with native MLX.

> [!IMPORTANT]
> MLX-Gen is an independent project forked from [mflux](https://github.com/filipstrand/mflux). It is currently built on the mflux codebase, with full credit to Filip Strand and the original contributors, while publishing under the `mlx-gen` package name and exposing `mlxgen` as the application import path.
>
> The project exists so compatibility fixes and capabilities can ship quickly for Apple Silicon workflows, including enabling Qwen Image/Edit support, ERNIE Image Turbo support, Bonsai low-bit FLUX.2 support, Wan video experiments, Qwen/FLUX.2 image editing, quantized model packaging, local model loading, AbstractVision integration, and release cadence. We will continue to credit and upstream focused fixes where practical, but MLX-Gen is expected to evolve and diverge rapidly as its own package.

### Table of contents

- [Relationship to mflux](#relationship-to-mflux)
- [💡 Philosophy](#-philosophy)
- [💿 Installation](#-installation)
- [Model Downloads And Preparation](#model-downloads-and-preparation)
- [Quantized Model Compatibility](#quantized-model-compatibility)
- [Progress And Video Health](#progress-and-video-health)
- [Documentation](#documentation)
- [🎨 Models](#-models)
- [✨ Features](#-features)
- [🌱 Related projects](#related-projects)
- [🙏 Acknowledgements](#-acknowledgements)
- [⚖️ License](#%EF%B8%8F-license)

---

<a id="relationship-to-mflux"></a>

### Relationship to mflux

MLX-Gen started as a fork of [mflux](https://github.com/filipstrand/mflux), which established a clear MLX-native image generation stack. This repository preserves that foundation and remains MIT licensed.

The immediate reason for the independent package is practical: MLX-Gen can iterate faster on compatibility fixes and capabilities that affect real usage, including Qwen Image/Edit quantization layouts, ERNIE Image Turbo model support, Bonsai low-bit FLUX.2 support, FLUX.2 edit behavior, local model packaging, PyPI release cadence, and Apple Silicon validation. Some of those changes are proposed upstream as small PRs; others may remain MLX-Gen-specific as the project direction diverges.

MLX-Gen also exists to power [AbstractVision](https://pypi.org/project/abstractvision/), the generative vision layer used with [AbstractCore](https://pypi.org/project/abstractcore/) in the wider [AbstractFramework](https://pypi.org/project/abstractframework/) ecosystem. That gives the package its own product requirements while keeping general fixes available for upstream mflux contributions where practical.

For now, some internals still live under `mflux.*` while MLX-Gen evolves from its forked base. New application code should use the `mlxgen` command and import path.

The project intentionally keeps mflux vocabulary in parts of the codebase and metadata while that remains useful. This preserves compatibility for existing users and keeps a possible merge-back path open if the two projects converge again.

Most credit for the current codebase goes to Filip Strand and the original mflux contributors. Changes introduced after the MLX-Gen fork are maintained here by Laurent-Philippe Albou / AbstractVision. See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for project credits.

---

### 💡 Philosophy

MLX-Gen is an independent package for running generative image and video models on MLX. It prioritizes fast local iteration, practical Apple Silicon performance, and compatibility with current model releases without coupling every change to upstream release timing.

The implementation remains intentionally direct: model code is written in MLX, with Hugging Face libraries used for tokenizers and model downloads.

---

### 💿 Installation
If you haven't already, [install `uv`](https://github.com/astral-sh/uv?tab=readme-ov-file#installation), then run:

```sh
uv tool install --upgrade mlx-gen
```

This package is published on PyPI as `mlx-gen`. The Python import for application code is `mlxgen`.

After installation, start with the MLX-Gen command help:

```sh
mlxgen --help
```

The public command surface is:

- `mlxgen generate`: generate images, edit images, or generate supported videos with a cached or prepared model.
- `mlxgen download`: explicitly download a model snapshot into the Hugging Face cache.
- `mlxgen prepare`: create a reusable local MLX-Gen model folder, optionally quantized, and write a Hugging Face model card.

Use `mlxgen prepare`, not a separate `save` command, when you want a local quantized folder to reuse or upload.

To generate your first image, use `mlxgen generate` and choose the model with `--model`:

```
mlxgen download --model z-image-turbo

mlxgen generate \
  --model z-image-turbo \
  --prompt "A puffin standing on a cliff" \
  --width 1280 \
  --height 500 \
  --seed 42 \
  --steps 9 \
  --quantize 8
```

![Puffin](src/mflux/assets/puffin.png)

The same router is also available as `mlx-gen`, `mlxgen-generate`, and `mlx-generate`.

For image editing, pass input images with `--image` or `--images`; MLX-Gen routes to the right backend from the model and image inputs:

```sh
mlxgen download --model AbstractFramework/qwen-image-edit-2511-4bit

mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-4bit \
  --image input.png \
  --prompt "Turn the room into a pencil sketch" \
  --steps 20 \
  --seed 42 \
  --output edited.png
```

For text-to-video, use Wan2.2. The smaller TI2V-5B path supports text-to-video and experimental
first-frame image-to-video:

```sh
mlxgen download --model Wan-AI/Wan2.2-TI2V-5B-Diffusers

mlxgen generate \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --task text-to-video \
  --prompt "A short cinematic video of a glowing orange glass sphere floating above teal water" \
  --width 1280 \
  --height 704 \
  --frames 121 \
  --steps 50 \
  --guidance 5 \
  --fps 24 \
  --output video.mp4
```

Wan2.2 T2V-A14B is also wired as a Diffusers-style two-transformer A14B path. `--guidance-2` is an
optional Diffusers-compatible low-noise-stage override. When both guidance flags are omitted,
MLX-Gen uses the model defaults (`4` high-noise, `3` low-noise for T2V-A14B). When `--guidance` is
set and `--guidance-2` is omitted, the low-noise stage follows `--guidance`:

```sh
mlxgen download --model Wan-AI/Wan2.2-T2V-A14B-Diffusers

mlxgen generate \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --task text-to-video \
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

Wan image-to-video is available through two paths. TI2V-5B uses an experimental first-frame
conditioning route:

```sh
mlxgen generate \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --task image-to-video \
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
mlxgen download --model Wan-AI/Wan2.2-I2V-A14B-Diffusers

mlxgen generate \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --task image-to-video \
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

The TI2V-5B I2V path follows Diffusers first-frame latent conditioning. It is not ordinary image-to-image latent initialization, and it should still be treated as early video support while quality and performance are validated. The A14B I2V route requires the complete local I2V source snapshot before generation.

Wan does not have a separate duration flag: duration is `frames / fps`. Frame counts are normalized to `4n + 1`. Width/height are normalized to the selected Wan model's VAE/patch multiple: TI2V-5B uses multiples of 32, while A14B uses multiples of 16.

See [API And CLI](docs/api.md) for spatial-scale Wan T2V/I2V sanity panels generated at 1280x704.

If a local model path or custom repository name cannot be classified from its name, add `--family qwen`, `--family flux2`, `--family bonsai`, `--family fibo`, `--family z-image`, `--family ernie-image`, or `--family wan`. The router can also read `model`, `image_path`, and `image_paths` from `--config-from-metadata`.

### Model Downloads And Preparation

MLX-Gen does not download model, tokenizer, LoRA, or Depth Pro files during generation. Generation is cache-only by default so applications can run predictable workflows without a network transfer starting in the middle of a job.

Use one of these explicit setup commands before generation:

```sh
# Download the required Hugging Face snapshot into the local cache.
mlxgen download --model Qwen/Qwen-Image

# Prepare a reusable local MLX-Gen model folder, optionally quantized.
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  --quantize 8

# Download the direct Apple Depth Pro weights used by depth workflows.
mlxgen download --model depth-pro
```

The commands have different outputs:

| Command | Use it when | Writes a local model folder | Writes a Hugging Face card |
| --- | --- | --- | --- |
| `mlxgen download` | You want the original repository cached for generation. | No | No |
| `mlxgen prepare` | You want a reusable MLX-Gen folder, usually quantized, for local reuse or upload. | Yes | Yes |
| `mlxgen generate` | You want to run inference from cached or prepared files. | No | No |

`mlxgen download` and `mlxgen prepare` are the commands that authorize network access. If you have Hugging Face's accelerated transfer backend available, you can optionally prefix those commands with `HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads.

`mlxgen prepare` also writes a Hugging Face `README.md` model card into the prepared folder. The generated card cites the original model, mflux, MLX-Gen, the `mlx-gen` version that generated the card, the quantization policy, and the default contributor attribution. Public card examples default to `AbstractFramework/<repo-name>` and include `python -m pip install -U mlx-gen` so Hugging Face readers can copy and paste a complete baseline setup.

If a required artifact is missing, MLX-Gen raises `DownloadRequiredError` with the exact command to run. See [docs/model-management.md](docs/model-management.md) for details and [docs/python-integration.md](docs/python-integration.md) for in-process usage.

### Quantized Model Compatibility

MLX-Gen supports reusable prepared folders for these primary quantized model families:

| Model family | q8 | q4 | Notes |
| --- | --- | --- | --- |
| Qwen Image and Qwen Image Edit | Supported | Mixed q4/q8 | Covers Qwen Image, Qwen Image 2512, Qwen Image Edit, 2509, and 2511. |
| ERNIE Image Turbo | Supported | Mixed q4/q8 | Text-to-image plus experimental single-image image-to-image. Prompt Enhancer is optional from a full source snapshot. |
| FLUX.2 Klein | Supported | Supported | Standard MLX quantization. 9B derivatives follow the source gated/non-commercial access requirements. |
| Bonsai Image | Pre-packed ternary 2-bit | Pre-packed ternary 2-bit | Use `mlxgen download` and `mlxgen generate`; do not run `prepare`. Binary 1-bit is detected but waiting on stock-MLX 1-bit runtime support. |
| Z-Image and Z-Image Turbo | Supported | Supported | Standard MLX quantization with model-specific generation defaults. |
| FIBO | Supported with source access | Supported with source access | Source repositories may require access approval before download or preparation. |
| Wan2.2 | Mixed q8/BF16 | Under validation | q8 prepared folders quantize bulky transformer block linears and keep sensitive conditioning/output paths, text encoders, and VAE at BF16. |

q4 and lower-bit checkpoints are not treated as blind size-only conversions. Qwen and ERNIE use mixed q4/q8 policies because fully q4 checkpoints can lose generation quality; the higher-precision paths are kept where validation shows they matter. Bonsai uses Prism's pre-packed ternary 2-bit transformer plus a 4-bit Qwen3 text encoder rather than MLX-Gen's `prepare` flow. In practice the quality strategy is similar: keep sensitive paths higher precision, then publish the smallest validated layout. See [Quantization](docs/quantization.md) for the current rules and measurements.

Wan A14B prepared BF16 and mixed q8/BF16 folders can substantially reduce storage compared with the upstream source snapshot. The mixed q8/BF16 layout also reduces measured full-process memory in validation, but it is not documented as a speed improvement. Keep full-size Wan A14B claims tied to the validation profile in the model card or release notes.

### Progress And Video Health

Python applications can subscribe to a shared progress event through `model.callbacks.subscribe_progress(...)`. The event type is `mflux.callbacks.ProgressEvent`; `event.progress` always reports denoising-step progress. Video events also include optional frame context through `frame`, `total_frames`, and `frame_progress`.

Wan video generation emits step-based progress instead of using output frames as the progress total. The CLI reserves `complete` for a video that has been saved and passed the built-in MP4 health check. If Wan generation or save validation fails, MLX-Gen writes a compact `<output>.failure.json` manifest next to the intended output path.

### Documentation

- [Getting started](docs/getting-started.md): install MLX-Gen, discover the CLI, prepare a model, and run generation.
- [Architecture](docs/architecture.md): package shape, command boundaries, model-file lifecycle, and runtime failure contract.
- [API and CLI](docs/api.md): public command surface, Python integration notes, and compatibility entry points.
- [Model management](docs/model-management.md): explicit `download` and `prepare` behavior, runtime cache policy, and model-card creation.
- [Quantization](docs/quantization.md): q4/q8 behavior, Bonsai low-bit packed support, and current Qwen/ERNIE mixed q4/q8 policies.
- [Hugging Face publishing](docs/huggingface-publishing.md): generated model cards, default `AbstractFramework/<repo-name>` usage, upload flow, and optional collection membership.
- [Python integration](docs/python-integration.md): cache-only model construction, AbstractVision integration, shared progress callbacks, and threading guidance.
- [FAQ](docs/faq.md): common questions about `prepare`, downloads, package naming, compatibility, and Wan image-to-video prompting.
- [Troubleshooting](docs/troubleshooting.md): common setup, runtime, and Wan video validation errors.

<details>
<summary>Python API</summary>

Create a standalone `generate.py` script with inline `uv` dependencies:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mlx-gen",
# ]
# ///
from mlxgen.models.z_image import ZImageTurbo

model = ZImageTurbo(quantize=8)
image = model.generate_image(
    prompt="A puffin standing on a cliff",
    seed=42,
    num_inference_steps=9,
    width=1280,
    height=500,
)
image.save("puffin.png")
```

Run it with:

```sh
mlxgen download --model z-image-turbo
uv run generate.py
```

For more Python API inspiration, look at the [CLI entry points](src/mflux/models/z_image/cli/z_image_turbo_generate.py) for the respective models.
</details>

<details>
<summary>⚠️ Troubleshooting: hf_transfer error</summary>

If you explicitly enable `HF_HUB_ENABLE_HF_TRANSFER=1` and encounter a `ValueError` because `hf_transfer` is not available, install MLX-Gen with the `hf_transfer` package included:

```sh
uv tool install --upgrade mlx-gen --with hf_transfer
```

This will enable faster model downloads from Hugging Face.

</details>

<details>
<summary>DGX / NVIDIA (uv tool install)</summary>

```sh
uv tool install --python 3.13 mlx-gen
```
</details>

---

### 🎨 Models

MLX-Gen supports the following model families. They have different strengths and weaknesses; see each model’s README for full usage details.

| Model | Release date | Size | Type | Training | Description |
| --- | --- | --- | --- | --- | --- |
|[Z-Image](src/mflux/models/z_image/README.md) | Nov 2025 | 6B | Distilled & Base | Yes | Fast, small, very good quality and realism. |
| Wan2.2 TI2V / A14B | Jul 2025 | 5B / A14B MoE | Base | No | Wan video support. TI2V-5B supports text-to-video and experimental first-frame image-to-video; T2V-A14B adds two-transformer boundary routing with optional low-noise `--guidance-2`; I2V-A14B is wired for complete local source snapshots. |
|[FLUX.2](src/mflux/models/flux2/README.md) | Jan 2026 | 4B & 9B | Distilled & Base | Yes | Fastest + smallest with very good quality and edit capabilities. |
| Bonsai Image | — | 4B | Distilled low-bit FLUX.2-derived | No | Prism low-bit text-to-image checkpoints. Ternary 2-bit runs directly in MLX-Gen; binary 1-bit is detected but blocked until stock MLX has 1-bit packed affine runtime support. |
|[FIBO](src/mflux/models/fibo/README.md) | Oct 2025+ | 8B | Distilled & Base | No | Very good JSON-based prompt understanding. Has edit capabilities. |
| ERNIE Image Turbo | Mar 2026 | 6B class | Distilled | No | Fast Apache 2.0 model from Baidu. MLX-Gen support covers text-to-image, experimental single-image image-to-image, BF16/q8/mixed q4 folders, and optional Prompt Enhancer from a full source snapshot. Use 384px+ outputs for reliable composition. |
|[SeedVR2](src/mflux/models/seedvr2/README.md) | Jun 2025 | 3B & 7B | — | No | Best upscaling model. |
|[Qwen Image](src/mflux/models/qwen/README.md) | Aug 2025+ | 20B | Base | No | Large model (slower); strong prompt understanding and world knowledge. Has edit capabilities |
|[Depth Pro](src/mflux/models/depth_pro/README.md) | Oct 2024 | — | — | No | Very fast and accurate depth estimation model from Apple. |
|[FLUX.1](src/mflux/models/flux/README.md) | Aug 2024 | 12B | Distilled & Base | No (legacy) | Legacy option with decent quality. Has edit capabilities with 'Kontext' model and upscaling support via ControlNet |

---

### ✨ Features

**General**
- Quantization and local model loading
- LoRA support (multi-LoRA, scales, library lookup)
- Metadata export + reuse, plus prompt file support

**Model-specific highlights**
- Text-to-image and image-to-image generation.
- LoRA finetuning
- In-context editing, multi-image editing, and virtual try-on
- ControlNet (Canny), depth conditioning, fill/inpainting, and Redux
- Upscaling (SeedVR2 and Flux ControlNet)
- Depth map extraction and FIBO prompt tooling (VLM inspire/refine)

See the [common README](src/mflux/models/common/README.md) for detailed usage and examples, and use the model section above to browse specific models and capabilities.

> [!NOTE]
> As MLX-Gen supports a wide variety of CLI tools and options, the easiest way to navigate the CLI in 2026 is to use a coding agent (like [Cursor](https://cursor.com), [Claude Code](https://www.anthropic.com/claude-code), or similar). Ask questions like: “Can you help me generate an image using z-image?”


---

<a id="related-projects"></a>

### 🌱 Related projects

- [MindCraft Studio](https://themindstudio.cc/mindcraft#models) — macOS app built on mflux by [@shaoju](https://github.com/shaoju)
- [Mflux-ComfyUI](https://github.com/raysers/Mflux-ComfyUI) by [@raysers](https://github.com/raysers)
- [MFLUX-WEBUI](https://github.com/CharafChnioune/MFLUX-WEBUI) by [@CharafChnioune](https://github.com/CharafChnioune)
- [mflux-fasthtml](https://github.com/anthonywu/mflux-fasthtml) by [@anthonywu](https://github.com/anthonywu)
- [mflux-streamlit](https://github.com/elitexp/mflux-streamlit) by [@elitexp](https://github.com/elitexp)

---

### 🙏 Acknowledgements

MLX-Gen exists because of the great work of:

- The MLX Team for [MLX](https://github.com/ml-explore/mlx) and [MLX examples](https://github.com/ml-explore/mlx-examples)
- Black Forest Labs for the [FLUX project](https://github.com/black-forest-labs/flux)
- Bria for the [FIBO project](https://huggingface.co/briaai/FIBO)
- Tongyi Lab for the [Z-Image project](https://tongyi-mai.github.io/Z-Image-blog/)
- Baidu for the [ERNIE Image Turbo model](https://huggingface.co/baidu/ERNIE-Image-Turbo)
- Prism ML for the [Bonsai Image low-bit MLX checkpoints](https://huggingface.co/prism-ml)
- Qwen Team for the [Qwen Image project](https://qwen.ai/blog?id=a6f483777144685d33cd3d2af95136fcbeb57652&from=research.research-list)
- ByteDance, @numz and @adrientoupet for the [SeedVR2 project](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
- Hugging Face for the [Diffusers library implementations](https://github.com/huggingface/diffusers) 
- Depth Pro authors for the [Depth Pro model](https://github.com/apple/ml-depth-pro?tab=readme-ov-file#citation)
- [mflux](https://github.com/filipstrand/mflux), Filip Strand, and the original mflux contributors and testers. MLX-Gen is currently based on that codebase and will keep acknowledging that foundation even as it evolves independently. Post-fork MLX-Gen changes are maintained by Laurent-Philippe Albou / AbstractVision.

---

### ⚖️ License

This project is licensed under the [MIT License](LICENSE).
