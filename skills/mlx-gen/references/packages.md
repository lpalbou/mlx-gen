# Model Files, Packages, And Quantization

Open this file when you need to get model files onto disk, choose between a source model and a
quantized package, build a package, or understand how a `--model` value is found.

## The three lifecycle commands

| Command | Does | Network | Writes |
| --- | --- | --- | --- |
| `mlxgen download --model <id>` | Fetches a source repository, a published package, or a LoRA repository | yes | Hugging Face cache |
| `mlxgen prepare --model <source> --path <dir> [--quantize 8]` | Loads a source model, optionally quantizes it, saves an MLX-Gen package | yes, if the source is not cached | `<dir>` with weights and a `README.md` model card |
| `mlxgen generate` / `mlxgen upscale` | Runs from files already on disk | never | outputs |

Use `download` to run a published package or a source model as-is. Use `prepare` only when you
need a local package that no published package provides.

## How `--model` is resolved

1. A path that exists on disk is used as a local folder.
2. A Hugging Face repo id (`owner/name`) first matches a complete local package at
   `./models/<name>` (relative to the working directory), so stable handles such as
   `AbstractFramework/qwen-image-edit-2511-8bit` run from a local copy.
3. Otherwise the complete snapshot in the Hugging Face cache is used.
4. Otherwise generation stops with `DownloadRequiredError`; nothing is fetched.

A catalog alias (`z-image-turbo`, `flux2-klein-4b`, ...) stands for its source repository. An
incomplete local package (missing shard or index files) is skipped, not half-loaded.

Identifying the model behind a name:

- Catalog aliases, source repo ids, and `AbstractFramework/*` package ids are identified exactly.
- A local folder or a custom repo id is identified only loosely from its name. The family is
  detected when the name contains it, but the exact variant is not trusted: a folder named
  `qwen-image-edit-2511-8bit` is treated as base Qwen Image (text and latent routes only), and a
  Z-Image Turbo folder loses its masked route.
- A name with no recognizable family fails with "Could not infer a supported backend", and
  `--family` alone does not configure it.
- So for every local folder or custom repo, pass `--base-model <catalog alias>` to `capabilities`
  and to `generate` or `upscale` (for example `--base-model qwen-image-edit-2511`), and check that
  the route you need is listed. A package stored at `./models/<repo-name>` can instead be addressed
  by its published repo id, which keeps the exact identity.

## `--quantize` versus packages

- Source model: `--quantize 8` (or `4`) quantizes at load time, every run.
- Package (`AbstractFramework/*-8bit`, `*-4bit`, `*-bf16`, or your own `prepare` output): the stored
  weights already are the quantization. Do not pass `--quantize`.
- 8-bit is the closest package to the source weights and the default recommendation where a
  validated one exists. Low-bit quality is model-specific: Qwen and ERNIE 4-bit packages are mixed
  4-bit/8-bit, FIBO keeps sensitive paths at BF16.
- Wan 8-bit packages store quantized weights but run at BF16-class memory: a disk and download
  saving only.
- Bonsai is already packed (omit `--quantize`, no `prepare`). Bernini-R is BF16 only (rejects
  `--quantize` and `prepare`). SwiftVR is BF16 only.
- MiniMax-H3 accepts only `--quantize 8` or `--quantize 4` on its source snapshot; see
  [minimax-h3.md](minimax-h3.md).
- Packages are MLX-Gen specific: they do not load with Diffusers or Transformers
  `from_pretrained()`, and third-party FP8/NVFP4 checkpoints are not MLX-Gen inputs.

## Published packages (`AbstractFramework/<name>`)

Sizes are repository totals in GiB. Run image and video packages with `mlxgen generate`, SeedVR2
packages with `mlxgen upscale`.

| Source model | Source size | Packages (size) |
| --- | ---: | --- |
| FLUX.2 Klein 4B | 22.1 | `flux.2-klein-4b-4bit` (4.3), `flux.2-klein-4b-8bit` (8.0) |
| FLUX.2 Klein 9B | 49.3 | `flux.2-klein-9b-4bit` (8.9), `flux.2-klein-9b-8bit` (16.6) |
| FLUX.2 Klein Base 4B | 22.1 | `flux.2-klein-base-4b-4bit` (4.3), `flux.2-klein-base-4b-8bit` (8.0) |
| FLUX.2 Klein Base 9B | 49.3 | `flux.2-klein-base-9b-4bit` (8.9), `flux.2-klein-base-9b-8bit` (16.6) |
| Qwen Image | 53.7 | `qwen-image-4bit` (16.2), `qwen-image-8bit` (27.5) |
| Qwen Image 2512 | 53.7 | `qwen-image-2512-4bit` (16.2), `qwen-image-2512-8bit` (27.5) |
| Qwen Image Edit | 53.8 | `qwen-image-edit-4bit` (17.0), `qwen-image-edit-8bit` (28.3) |
| Qwen Image Edit 2509 | 53.8 | `qwen-image-edit-2509-4bit` (17.0), `qwen-image-edit-2509-8bit` (28.3) |
| Qwen Image Edit 2511 | 53.8 | `qwen-image-edit-2511-4bit` (17.0), `qwen-image-edit-2511-8bit` (28.3) |
| Z-Image | 19.1 | `z-image-4bit` (5.5), `z-image-8bit` (10.2) |
| Z-Image Turbo | 30.6 | `z-image-turbo-4bit` (5.5), `z-image-turbo-8bit` (10.2) |
| ERNIE Image Turbo | 29.5 | `ernie-image-turbo-8bit` (11.5), `ernie-image-turbo-4bit` (11.5, see note) |
| FIBO | 23.8 | `fibo-4bit` (10.2), `fibo-8bit` (14.5) |
| SeedVR2 3B | 13.57 | `seedvr2-3b-4bit` (2.54), `seedvr2-3b-8bit` (4.39) |
| SeedVR2 7B | 31.63 | `seedvr2-7b-4bit` (4.79), `seedvr2-7b-8bit` (8.62) |
| Wan2.2 TI2V-5B | 31.9 | `wan2.2-ti2v-5b-diffusers-bf16` (21.2), `wan2.2-ti2v-5b-diffusers-8bit` (16.9) |
| Wan2.2 T2V-A14B | 117.5 | `wan2.2-t2v-a14b-diffusers-bf16` (64.1), `wan2.2-t2v-a14b-diffusers-8bit` (39.5) |
| Wan2.2 I2V-A14B | 117.5 | `wan2.2-i2v-a14b-diffusers-bf16` (64.1), `wan2.2-i2v-a14b-diffusers-8bit` (39.5) |
| MiniMax-H3 | about 140 GB | `minimax-h3-8bit` (about 64), `minimax-h3-4bit` (about 45) |

Notes:

- The published `ernie-image-turbo-4bit` repository carries 8-bit weights at the 8-bit size; do
  not count on it as a 4-bit memory saving.
- Bonsai (`prism-ml/bonsai-image-ternary-4B-mlx-2bit`, 3.6 GiB) and Bernini-R
  (`mlxgen download --model bernini-r-1.3b`, about 27 GiB plus 2 GiB of required free-space
  headroom) have no quantized package.
- No Wan 4-bit package is published.
- Model weights keep their original licenses and access terms. FLUX.2 Klein 9B packages are gated
  and non-commercial; MiniMax-H3 packages carry a territory-restricted license.

## Download details

```sh
mlxgen download --model "<ALIAS_OR_REPO_ID>"
mlxgen download --model "<OWNER/LORA_REPO>" --all-files        # LoRA repositories
mlxgen download --model "AbstractFramework/minimax-h3-8bit" --base-model minimax-h3-turbo-544p
mlxgen download --model depth-pro                               # Depth Pro weights (direct URL)
```

- By default only the files MLX-Gen needs are fetched (weights, tokenizers, configs);
  `--all-files` fetches the whole repository. Bernini-R rejects `--all-files`.
- Downloads resume: rerun the same command after an interruption.
- Models that declare their size are checked against free disk space first and refused with the
  required amount when they cannot fit.
- `HF_HUB_ENABLE_HF_TRANSFER=1` is optional acceleration, never required. If it errors, install
  with `uv tool install --upgrade mlx-gen --with hf_transfer` or unset it.

Failure cases: a non-repo value fails with "--model must resolve to a Hugging Face repo id for
download"; an authentication or gated-access error from Hugging Face means the user must grant
access themselves (stop, do not retry).

## Build a package with `prepare`

```sh
mlxgen download --model "Qwen/Qwen-Image"
mlxgen prepare --model "Qwen/Qwen-Image" --path ./models/qwen-image-8bit --quantize 8
mlxgen capabilities --model ./models/qwen-image-8bit --base-model qwen-image
mlxgen generate --model ./models/qwen-image-8bit --base-model qwen-image \
  --prompt "<PROMPT>" --output "<OUTPUT.png>"
```

- `prepare` loads the complete source model, so plan for the source model's memory and a long run,
  and confirm with the user first. Omit `--quantize` for a BF16 package where the family supports
  it (for example ERNIE).
- Success is the folder at `--path` containing weights and a generated `README.md` card that
  records the source model, the quantization policy, and the `mlx-gen` version.
- A folder named like a published package (`./models/<repo-name>`) also answers to the matching
  repo id.
- SeedVR2 packages are prepared the same way and run with `mlxgen upscale --model <dir>`.
- Publishing a package is the user's decision; see
  <https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/huggingface-publishing.md>.

Source docs: <https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/model-management.md>,
<https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/quantization.md>.
