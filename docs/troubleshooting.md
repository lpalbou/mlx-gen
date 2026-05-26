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

Generation commands replace the requested output path by default. If `--output image.png` already exists, MLX-Gen writes the new image to `image.png`.

Use `--replace false` or `--no-replace` when you want to preserve an existing output file. In that mode, MLX-Gen writes the new image as `image_1.png`, then `image_2.png`, and continues silently.

## ERNIE Prompt Enhancer Files Are Missing

`--use-prompt-enhancer` requires ERNIE's `pe/` and `pe_tokenizer/` files. The default ERNIE download skips those files to keep ordinary generation setup smaller.

Run:

```sh
mlxgen download --model baidu/ERNIE-Image-Turbo --all-files
```

Then retry generation with `--use-prompt-enhancer`.

## ERNIE Rejects Multiple Image Inputs Or Edit Tasks

ERNIE Image Turbo supports text-to-image and experimental single-image image-to-image. It does not support multi-image edit.

Use one input image for ERNIE image-to-image:

```sh
mlxgen generate \
  --model baidu/ERNIE-Image-Turbo \
  --image input.png \
  --prompt "Turn the scene into a pencil sketch" \
  --width 512 \
  --height 512 \
  --steps 8 \
  --guidance 3 \
  --image-strength 0.25 \
  --output edited.png
```

If you pass `--task edit`, multiple `--images`, or `--image-strength` without an image, MLX-Gen fails before loading the model and tells you which input shape ERNIE supports.

If ERNIE image-to-image does not preserve enough of the source image, increase `--image-strength`, keep the output aspect ratio close to the input aspect ratio, or use Qwen Image Edit for a true image-conditioned edit. If ERNIE preserves the source too strongly and barely applies the requested style, lower `--image-strength` or increase `--steps` to 12-16.

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
