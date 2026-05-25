# Model Management

MLX-Gen generation is cache-only by default. It will not download model weights, tokenizers, LoRAs, or Depth Pro weights during generation or during ordinary Python model construction.

This policy keeps CLI jobs and embedded application workflows predictable: a generation request either finds the required files locally or fails with a `DownloadRequiredError` that includes the command to run.

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

## Prepare A Local MLX-Gen Folder

Use `mlxgen prepare` when you want a reusable local folder, usually with quantized MLX-Gen weights:

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  -q 8
```

Then generate from the local folder:

```sh
mlxgen generate \
  --model ./models/qwen-image-8bit \
  --family qwen \
  --prompt "A product photo of a ceramic teapot" \
  --output image.png
```

`mflux-save` remains available for compatibility and uses the same explicit download permission as `mlxgen prepare`.

`mlxgen prepare` and `mflux-save` write a Hugging Face `README.md` model card into the prepared folder. The card records the source model, MLX-Gen compatibility, mflux attribution, quantization policy, and default contributor attribution.

## Depth Pro

Depth workflows use Apple Depth Pro weights from a direct URL rather than a Hugging Face repository. Download them explicitly:

```sh
mlxgen download --model depth-pro
```

After that, `mflux-save-depth` and depth generation can run without starting a network transfer.

## Runtime Failure Contract

When files are missing, MLX-Gen raises `DownloadRequiredError`. The exception is also a `FileNotFoundError` for compatibility with existing callers. It exposes:

- `repo_id`
- `artifact`
- `download_command`
- `prepare_command` when a local prepared folder is applicable

The human-readable message is designed for non-expert users and includes the exact command to run.
