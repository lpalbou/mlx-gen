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
| `mlxgen generate` | Generate or edit images from a cached or prepared model. |
| `mlxgen download` | Explicitly download model or LoRA files into the local cache. |
| `mlxgen prepare` | Create a reusable local MLX-Gen model folder, optionally quantized, and write a Hugging Face model card. |

The package also installs compatibility entry points from the mflux codebase. New MLX-Gen documentation and application integrations should prefer the `mlxgen` commands above.

## Generation Router

`mlxgen generate` chooses the backend from `--model`, optional `--family`, `--task`, and image inputs.

```sh
mlxgen generate \
  --model z-image-turbo \
  --prompt "A product photo of a ceramic teapot" \
  --output image.png
```

For edits, pass an image:

```sh
mlxgen generate \
  --model lpalbou/qwen-image-edit-2511-4bit \
  --image input.png \
  --prompt "Turn the room into a pencil sketch" \
  --output edited.png
```

Supported router families are `qwen`, `flux2`, `fibo`, and `z-image`:

```sh
mlxgen generate \
  --model ./models/qwen-image-8bit \
  --family qwen \
  --prompt "A clean studio product photo"
```

Use `--config-from-metadata` / `-C` when you want the router to read fields such as `model`, `image_path`, or `image_paths` from an existing metadata file.

## Model Management Commands

`mlxgen download` and `mlxgen prepare` are the only public MLX-Gen commands that authorize network access.

```sh
mlxgen download --model Qwen/Qwen-Image
```

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  -q 8
```

Use `prepare` when you need the local saved-weight folder. It is the public MLX-Gen workflow for creating quantized model folders and generated Hugging Face cards.

## Python Integration

The current Python integration path uses model classes inherited from the mflux codebase, with `mlxgen` available as the package identity for new applications. See [Python Integration](python-integration.md) for the current expectations.

Python callers should prepare or download required model files before constructing model objects. Runtime constructors and generation calls do not start network downloads.

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

MLX-Gen prepared model folders use the MLX/mflux saved-weight layout and MLX quantization tensors. They are intended for MLX-Gen and compatible mflux code, not for direct Diffusers or Transformers `from_pretrained()` loading.
