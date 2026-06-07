# Model Management

MLX-Gen does not download model weights, tokenizers, LoRAs, or Depth Pro weights during generation or during ordinary Python model construction.

This policy keeps CLI jobs and embedded application workflows predictable: a generation request either finds the required files locally or fails with a `DownloadRequiredError` that includes the command to run.

## Command Roles

MLX-Gen exposes three public commands for the normal model lifecycle:

| Command | Purpose | Network access | Main output |
| --- | --- | --- | --- |
| `mlxgen download` | Put an original model repository or LoRA repository in the local Hugging Face cache. | Allowed because the user asked for download. | Cached source files. |
| `mlxgen prepare` | Create a reusable local MLX-Gen model package, usually quantized. | Allowed because the user asked for preparation. | Local MLX-Gen package plus generated `README.md` card. |
| `mlxgen generate` | Generate images or supported videos from downloaded source files or local MLX-Gen packages. Image input selects image-to-image or image-to-video when supported. | Not allowed by default. | Image or video output and optional metadata. |

Use `mlxgen prepare`, not a separate `save` command, when you want a local quantized MLX-Gen
package to reuse from another project or upload to Hugging Face.

## Download A Hugging Face Snapshot

Use `mlxgen download` to populate the local Hugging Face cache:

```sh
mlxgen download --model Qwen/Qwen-Image
```

Aliases are supported when MLX-Gen knows the model:

```sh
mlxgen download --model z-image-turbo
```

For LoRA repositories, download the repository explicitly before passing it to generation:

```sh
mlxgen download --model RiverZ/normal-lora --all-files
```

`mlxgen download` is already an explicit network operation. `HF_HUB_ENABLE_HF_TRANSFER=1` is optional and only enables Hugging Face's accelerated transfer backend when that backend is available.

Use `download` when you want to run from the source model name or alias and do not need a separate
local MLX-Gen package.

## Prepare A Local MLX-Gen Package

Use `mlxgen prepare` when you want a reusable local MLX-Gen package, usually with quantized weights:

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  --quantize 8
```

`prepare` loads the source model, applies the requested quantization when `--quantize` is provided,
writes the MLX-Gen saved-weight layout to `--path`, and writes a Hugging Face `README.md` model
card into that package. The card records the `mlx-gen` version that generated it.

Then generate from the local package:

```sh
mlxgen generate \
  --model ./models/qwen-image-8bit \
  --family qwen \
  --prompt "A product photo of a ceramic teapot" \
  --output image.png
```

If the local package path does not clearly identify the model family, add `--family` during generation. The supported router families are `qwen`, `flux2`, `bonsai`, `fibo`, `z-image`, `ernie-image`, and `wan`.

SeedVR2 packages use `mlxgen upscale` instead of `mlxgen generate`:

```sh
mlxgen prepare \
  --model ByteDance-Seed/SeedVR2-3B \
  --path ./models/seedvr2-3b-8bit \
  --quantize 8

mlxgen upscale \
  --model ./models/seedvr2-3b-8bit \
  --image-path input.png \
  --resolution 2x \
  --metadata \
  --output input_2x.png
```

Use `ByteDance-Seed/SeedVR2-7B`, `AbstractFramework/seedvr2-7b-8bit`, or a path such as
`./models/seedvr2-7b-8bit` for 7B packages.

When a local MLX-Gen package name matches a Hugging Face repository basename, MLX-Gen can also resolve
the repository handle to that complete local package. For example, if
`./models/wan2.2-i2v-a14b-diffusers-8bit` exists and contains all required indexes and shard files,
then this command can run from local files without a Hugging Face cache snapshot:

```sh
mlxgen generate \
  --model AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit \
  --image source.png \
  --prompt "The spacecraft slowly lifts off" \
  --output video.mp4
```

The generated card records the source model, MLX-Gen compatibility, mflux attribution, generator version, quantization policy, and default contributor attribution. See [Hugging Face Publishing](huggingface-publishing.md) for upload and collection guidance.

## Choosing Download Or Prepare

Use `mlxgen download` when:

- you want the original repository cached locally;
- you are using a model alias or Hugging Face repository directly at generation time;
- you do not need a quantized local MLX-Gen package.

Use `mlxgen prepare` when:

- you want a local path such as `./models/qwen-image-8bit`;
- you want quantized weights with `--quantize 4` or `--quantize 8`;
- you want a generated Hugging Face model card;
- you want a package that another application, such as AbstractVision, can reference without depending on the original repository name.

Bonsai Image is an exception to the ordinary prepare flow. The Prism Bonsai repositories are
already packed MLX artifacts, so cache them with `download` and generate directly:

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

`mlxgen prepare` rejects Bonsai with a short explanation because there is no q4/q8 conversion step
to run. The ternary 2-bit checkpoint is supported; the binary 1-bit checkpoint is detected but
requires MLX runtime support that is not present in stock MLX through 0.31.2.

ERNIE Image Turbo can be downloaded from the source repo or converted into local BF16, q8, or q4
MLX-Gen packages:

```sh
mlxgen download --model baidu/ERNIE-Image-Turbo
mlxgen prepare --model baidu/ERNIE-Image-Turbo --path ./models/ernie-image-turbo
mlxgen prepare --model baidu/ERNIE-Image-Turbo --path ./models/ernie-image-turbo-8bit --quantize 8
mlxgen prepare --model baidu/ERNIE-Image-Turbo --path ./models/ernie-image-turbo-4bit --quantize 4
```

The default ERNIE download pattern fetches the BF16 generation components used by ordinary text-to-image and single-image latent image-to-image generation: tokenizer, text encoder, transformer, VAE, scheduler metadata, and repository metadata. It does not fetch ERNIE's Prompt Enhancer model. Use `mlxgen download --model baidu/ERNIE-Image-Turbo --all-files` before generation when you plan to pass `--use-prompt-enhancer`.

ERNIE 8-bit and 4-bit MLX-Gen packages load through the same `mlxgen generate` flow. The 4-bit
package uses a model-specific mixed q4/q8 policy to reduce the quality drift seen with fully q4
ERNIE checkpoints. ERNIE optimized packages contain the generation stack needed for text-to-image
and single-image latent image-to-image; they do not bundle Prompt Enhancer files.

Wan2.2 video models can be downloaded and used from the source snapshots:

```sh
mlxgen download --model Wan-AI/Wan2.2-TI2V-5B-Diffusers
mlxgen download --model Wan-AI/Wan2.2-T2V-A14B-Diffusers
mlxgen download --model Wan-AI/Wan2.2-I2V-A14B-Diffusers
```

Wan text-to-video currently uses the source snapshot plus a local-only Hugging Face UMT5 text encoder for prompt embeddings. TI2V-5B image-to-video uses the Diffusers first-frame latent-conditioning path and requires the same VAE encoder/decoder files from the source snapshot. A14B T2V uses two transformer folders, `transformer` and `transformer_2`, with boundary routing and optional low-noise `--guidance-2`. A14B I2V uses the separate `Wan-AI/Wan2.2-I2V-A14B-Diffusers` source snapshot and requires that snapshot to be complete before generation. For Wan image-to-video, MLX-Gen resolves the final output dimensions from the input image aspect ratio and model spatial multiples.

Published AbstractFramework Wan MLX-Gen packages currently include TI2V-5B BF16/q8 plus A14B
T2V/I2V BF16 and mixed q8/BF16 packages. The upstream TI2V-5B source snapshot stores its
transformer and VAE as FP32 on disk, but MLX-Gen loads and prepares those components at BF16
runtime precision; the BF16 TI2V-5B package is primarily a smaller source-equivalent
package. There is no published TI2V-5B q4 or mixed q4/q8 package yet.

For visual quality checks, use the upstream model scale. TI2V-5B uses 1280x704 or 704x1280, 121 frames, 50 steps, and 24 fps. A14B uses 1280x720 or 720x1280, 81 frames, 40 steps, `--guidance 4`, optional `--guidance-2 3`, and 16 fps. Lower settings are useful for quick command checks, not quality assessment.

## Depth Pro

Depth workflows use Apple Depth Pro weights from a direct URL rather than a Hugging Face repository. Download them explicitly:

```sh
mlxgen download --model depth-pro
```

After that, depth generation can run without starting a network transfer.

## Runtime Failure Contract

When files are missing, MLX-Gen raises `DownloadRequiredError`. The exception is also a `FileNotFoundError` for compatibility with existing callers. It exposes:

- `repo_id`
- `artifact`
- `download_command`
- `prepare_command` when a local MLX-Gen package is applicable

The human-readable message is designed for non-expert users and includes the exact command to run.
