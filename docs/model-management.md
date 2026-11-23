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

If a LoRA repository contains more than one `.safetensors` file, pass the file name as
`owner/repo:filename.safetensors`. The adapter must be trained for the selected model family; see
[LoRA](lora.md) for compatibility checks and the source/no-LoRA/with-LoRA validation method.

`mlxgen download` is already an explicit network operation. `HF_HUB_ENABLE_HF_TRANSFER=1` is optional and only enables Hugging Face's accelerated transfer backend when that backend is available.

Gated repositories need a Hugging Face sign-in before `mlxgen download`: accept the model's terms on its Hugging Face page, then set `HF_TOKEN` or run `hf auth login`.

`--model` must be a model alias or an `owner/repo` Hugging Face id; a local path is not a download source.

For a model whose catalog entry declares a download size (Bernini-R and SwiftVR), `mlxgen download` checks free space at the Hugging Face cache before transferring anything and stops with `Insufficient free space for the missing model sources` when the missing files plus 2 GiB of headroom do not fit. Sources that are already complete in the cache are not counted. Other models, and any `--all-files` download, are not checked.

Use `download` when you want to run from the source model name or alias and do not need a separate
local MLX-Gen package.

### Bernini-R pinned download

Bernini-R 1.3B downloads a selected set of files (tokenizer, UMT5 text encoder, VAE, and renderer
transformer) from one repository at a pinned revision. Two aliases are available, and each one is
its own model id for `mlxgen download` and `mlxgen generate`:

| Alias | Repository | Cold download | Free space required |
| --- | --- | ---: | ---: |
| `bernini-r-1.3b-bf16` | `AbstractFramework/bernini-r-1.3B-diffusers-bf16` | 15.60 GiB | 17.60 GiB |
| `bernini-r-1.3b` | `ByteDance/Bernini-R-1.3B-Diffusers` (FP32 source) | 26.94 GiB | 28.94 GiB |

```sh
mlxgen download --model bernini-r-1.3b-bf16
```

The free-space figure includes `2 GiB` of download headroom, and a source that is already complete
at its pinned revision is not counted. Both aliases run at the same precision and produce the same
output; see [Bernini-R 1.3B](bernini.md#download-and-capacity). The bounded low-RAM profiles
documented there peaked at 9.45 GB on the 128 GB validation host, so the tested shapes fit an
18 GB-class memory envelope, but still need an actual 18 GB-host run. This is capacity evidence
only: the Bernini visual release gate fails. Memory and disk are separate budgets: `18 GB` of free
disk is enough for neither cold download.

Do not pass `--all-files` for Bernini. The option is rejected because the route uses a pinned
subset of the repository. Bernini also rejects `mlxgen prepare` and runtime `--quantize`: the route
runs at BF16 only, and q4/q8 Bernini packages are not supported. It remains experimental after
failed visual validation.

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
  --base-model qwen-image \
  --prompt "A product photo of a ceramic teapot" \
  --output image.png
```

`--model` values are identified in two ways:

- A catalog alias (`qwen-image`), a source repository id (`Qwen/Qwen-Image`), and a published
  `AbstractFramework/<name>` package id are identified exactly.
- Any other local folder or custom repository name is identified loosely from the catalog model
  name it contains. MLX-Gen trusts only the model family from such a name, so the routes it
  publishes can be narrower than the model's own. A folder named `qwen-image-edit-2511-8bit`
  resolves to the Qwen family with the text-to-image and latent image-to-image routes only.

Pass `--base-model <alias>` to state the model exactly. It is accepted by `mlxgen generate`,
`mlxgen capabilities`, and `mlxgen prepare`, and it is required when the name contains no catalog
model name:

```sh
mlxgen capabilities \
  --model ./models/qwen-image-edit-2511-8bit \
  --base-model qwen-image-edit-2511
```

With the hint, that folder reports the Qwen edit, masked edit, reframe, outpaint, and
multi-reference routes. `--family` only checks the family: it is rejected when it differs from the
resolved family, and it cannot identify a model on its own. Wan video routes load a local folder by its
name and do not apply `--base-model`; name a local Wan folder after its model, for example
`./models/wan2.2-i2v-a14b-diffusers-8bit`. See
[Local Path Cannot Be Classified](troubleshooting.md#local-path-cannot-be-classified) for the
related error messages.

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
the repository handle to that complete local package. For a repository id such as `owner/<repo-name>`,
MLX-Gen looks for a complete package at `./models/<repo-name>` (relative to the working directory),
then at `~/models/<repo-name>`, and only then in the Hugging Face cache. A complete local folder with
that name therefore takes precedence over a cached snapshot of the repository, even when it holds
different weights; rename or move the folder to run from the cache. For example, if
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

Bernini-R is an exception: use `mlxgen download --model bernini-r-1.3b-bf16` or
`mlxgen download --model bernini-r-1.3b`. Its runtime loads a pinned revision of one repository and
intentionally has no `prepare`, q4, or q8 package route.

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

For upstream-scale visual quality checks, use the model's intended profile. TI2V-5B uses 1280x704
or 704x1280, 121 frames, 50 steps, 24 fps, and flow shift `5.0`. A14B uses 1280x720 or 720x1280,
81 frames, 40 steps, `--guidance 4`, optional `--guidance-2 3`, flow shift `3.0`, and 16 fps. For
practical five-second prompt iteration, A14B at `480x240` or `240x480`, 101 frames, 20 fps, and
`20-25` steps has produced good results on an M5 Max in about 30 minutes. For 480p-class TI2V-5B
checks such as `832x480`, pass `--flow-shift 3`. See [Wan Video](wan-video.md) for the comparison
clips.

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
- `prepare_command` when a local MLX-Gen package is applicable; it is `None` for LoRA files, Bonsai
  checkpoints, Depth Pro weights, and repositories in the `AbstractFramework/` namespace, which are
  already MLX-Gen packages

The human-readable message is designed for non-expert users and includes the exact command to run.
For a missing model its first line is:

```text
MLX-Gen will not download model files during generation.
```

For other artifacts, `model files` reads `tokenizer files`, `LoRA files`, `preview decoder files`,
or `Depth Pro weights`. The lines that follow give the `mlxgen download --model <repo-id>` command
to run, with `--all-files` for a LoRA repository. The CLI prints the message and exits non-zero
(see [exit statuses](api.md#command-line-surface)); with `--json-events` the same fields appear
under `remediation` (see [CLI Runtime Events](api.md#cli-runtime-events)).
