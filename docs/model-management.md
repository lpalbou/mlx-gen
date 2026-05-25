# Model Management

MLX-Gen generation is cache-only by default. It will not download model weights, tokenizers, LoRAs, or Depth Pro weights during generation or during ordinary Python model construction.

This policy keeps CLI jobs and embedded application workflows predictable: a generation request either finds the required files locally or fails with a `DownloadRequiredError` that includes the command to run.

## Command Roles

MLX-Gen exposes three public commands for the normal model lifecycle:

| Command | Purpose | Network access | Main output |
| --- | --- | --- | --- |
| `mlxgen download` | Put an original model repository or LoRA repository in the local Hugging Face cache. | Allowed because the user asked for download. | Cached source files. |
| `mlxgen prepare` | Create a reusable local MLX-Gen model folder, usually quantized. | Allowed because the user asked for preparation. | Local model folder plus generated `README.md` card. |
| `mlxgen generate` | Generate or edit images from cached or prepared files. | Not allowed by default. | Image output and optional metadata. |

Use `mlxgen prepare`, not a separate `save` command, when you want a local quantized model folder to reuse from another project or upload to Hugging Face.

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

Use `download` when you want to run from the source model name or alias and do not need a separate local folder.

## Prepare A Local MLX-Gen Folder

Use `mlxgen prepare` when you want a reusable local folder, usually with quantized MLX-Gen weights:

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  -q 8
```

`prepare` loads the source model, applies the requested quantization when `-q` / `--quantize` is provided, writes the MLX-Gen saved-weight layout to `--path`, and writes a Hugging Face `README.md` model card into that folder.

Then generate from the local folder:

```sh
mlxgen generate \
  --model ./models/qwen-image-8bit \
  --family qwen \
  --prompt "A product photo of a ceramic teapot" \
  --output image.png
```

If the local folder name does not clearly identify the model family, add `--family` during generation. The supported router families are `qwen`, `flux2`, `fibo`, and `z-image`.

The generated card records the source model, MLX-Gen compatibility, mflux attribution, quantization policy, and default contributor attribution. See [Hugging Face Publishing](huggingface-publishing.md) for upload and collection guidance.

## Choosing Download Or Prepare

Use `mlxgen download` when:

- you want the original repository cached locally;
- you are using a model alias or Hugging Face repository directly at generation time;
- you do not need a quantized local folder.

Use `mlxgen prepare` when:

- you want a local path such as `./models/qwen-image-8bit`;
- you want quantized weights with `-q 4` or `-q 8`;
- you want a generated Hugging Face model card;
- you want a folder that another application, such as AbstractVision, can reference without depending on the original repository name.

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
- `prepare_command` when a local prepared folder is applicable

The human-readable message is designed for non-expert users and includes the exact command to run.
