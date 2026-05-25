# Troubleshooting

## MLX-Gen Will Not Download Files During Generation

This is expected. Runtime generation is cache-only. Run the command shown in the error message, then retry the generation.

Common commands:

```sh
mlxgen download --model Qwen/Qwen-Image
mlxgen prepare --model Qwen/Qwen-Image --path ./models/qwen-image-8bit --quantize 8
mlxgen download --model depth-pro
```

## Local Path Cannot Be Classified

When using a local model path, MLX-Gen may not be able to infer the model family from the folder name. Add `--family`:

```sh
mlxgen generate \
  --model ./models/qwen-image-8bit \
  --family qwen \
  --prompt "A clean studio product photo" \
  --output image.png
```

Supported router families are `qwen`, `flux2`, `fibo`, `z-image`, and `ernie-image`.

## ERNIE Images Look Cropped At Tiny Sizes

ERNIE Image Turbo is validated for practical generation at 384px and above. Very small outputs, such as 256x256, can crop or truncate subjects even when the pipeline is working.

Use 512x512 for small validation runs:

```sh
mlxgen generate \
  --model baidu/ERNIE-Image-Turbo \
  --prompt "A clean centered product photo of a white ceramic mug" \
  --width 512 \
  --height 512 \
  --steps 8 \
  --guidance 1 \
  --output image.png
```

## Output File Already Exists

Generation commands do not overwrite an existing output path by default. If `--output image.png` already exists, MLX-Gen saves the new image as `image_1.png`, then `image_2.png`, and prints the final path.

If the image you open has old dimensions or metadata, check the suffixed output path printed by the command.

## ERNIE Rejects Image Inputs Or Prompt Enhancer

MLX-Gen currently supports ERNIE Image Turbo as BF16 text-to-image generation. `--image`, `--images`, image-to-image/edit tasks, and `--use-prompt-enhancer` fail intentionally so applications do not accidentally run a different workflow from the one requested.

## `generate --path` Fails

`--path` belongs to `mlxgen prepare`, where it names the local model folder to create. It is not a generation option.

To prepare a quantized model folder:

```sh
mlxgen prepare --model black-forest-labs/FLUX.2-klein-4B --path models/flux.2-klein-4b-4bit --quantize 4
```

To choose the generated image path, use `--output` with `mlxgen generate`.

## LoRA Is Missing

User-requested LoRAs are required. MLX-Gen no longer ignores a missing LoRA and continues without it. Download the LoRA repository or use a local `.safetensors` file path.

```sh
mlxgen download --model RiverZ/normal-lora --all-files
```

## hf_transfer Error

`HF_HUB_ENABLE_HF_TRANSFER=1` is optional. It can make explicit Hugging Face downloads faster, but it is not required to authorize downloads.

If you enable it and the `hf_transfer` package is unavailable, install MLX-Gen with the extra package available to the environment:

```sh
uv tool install --upgrade mlx-gen --with hf_transfer
```
