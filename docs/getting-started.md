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

The top-level command shows the three public workflows:

- `mlxgen generate` for image generation, image editing, and supported video generation.
- `mlxgen download` for explicit Hugging Face cache downloads.
- `mlxgen prepare` for reusable local MLX-Gen model folders.

## Prepare Model Files

Generation is cache-only. Before generation, either download the source repository into the local Hugging Face cache or prepare a reusable local folder.

Download into the Hugging Face cache:

```sh
mlxgen download --model z-image-turbo
```

Prepare a reusable local folder with quantized weights and a generated Hugging Face model card:

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  --quantize 8
```

Use `mlxgen prepare` when you need a local model folder. There is no separate MLX-Gen `save` workflow in the public documentation.

`HF_HUB_ENABLE_HF_TRANSFER=1` is optional. It can speed up explicit Hugging Face download or prepare commands when the accelerated transfer backend is available, but it is not required to authorize downloads.

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

Run generation from a prepared local folder:

```sh
mlxgen generate \
  --model ./models/qwen-image-8bit \
  --family qwen \
  --prompt "A clean studio product photo" \
  --output image.png
```

`--family` is useful when a local path or custom repository name does not contain a recognizable model-family name.

## Edit An Image

Pass one or more input images to the same `generate` command. MLX-Gen routes to the edit backend from the model and image inputs:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-4bit \
  --image input.png \
  --prompt "Turn the room into a pencil sketch" \
  --steps 20 \
  --seed 42 \
  --output edited.png
```

## Generate A Video

Wan2.2 TI2V support is available as an initial text-to-video backend. Download the source snapshot first, then run `mlxgen generate` with `--task text-to-video`:

```sh
mlxgen download --model Wan-AI/Wan2.2-TI2V-5B-Diffusers

mlxgen generate \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --task text-to-video \
  --prompt "A short cinematic video of a glowing orange glass sphere floating above teal water" \
  --width 128 \
  --height 128 \
  --frames 5 \
  --steps 4 \
  --guidance 5 \
  --fps 8 \
  --seed 321 \
  --output video.mp4
```

For image-to-video, pass one input image and switch the task:

```sh
mlxgen generate \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --task image-to-video \
  --image input.png \
  --prompt "A slow cinematic camera move from the input frame" \
  --width 256 \
  --height 256 \
  --frames 17 \
  --steps 12 \
  --guidance 5 \
  --fps 8 \
  --output video.mp4
```

Wan image-to-video uses the Diffusers first-frame latent-conditioning path. Treat current Wan video support as experimental: the pipeline can produce MP4 output, but quality, speed, and practical defaults still need broader validation.

## Next Steps

- See [Model Management](model-management.md) for the full download, prepare, and runtime failure contract.
- See [API And CLI](api.md) for the supported command surface and Python integration notes.
- See [Quantization](quantization.md) for q4/q8 behavior and current Qwen/ERNIE mixed q4/q8 policies.
- See [Troubleshooting](troubleshooting.md) when a required artifact is missing or a local path cannot be classified.
