# Troubleshooting

## MLX-Gen Will Not Download Files During Generation

This is expected. Runtime generation uses files that are already available locally. Run the command shown in the error message, then retry the generation.

Common commands:

```sh
mlxgen download --model Qwen/Qwen-Image
mlxgen prepare --model Qwen/Qwen-Image --path ./models/qwen-image-8bit --quantize 8
mlxgen download --model depth-pro
```

If you prepared a local folder under `./models/<repo-name>`, you can use either that local path or
the matching Hugging Face handle. MLX-Gen checks the prepared folder only when it is complete:

```sh
mlxgen generate --model ./models/wan2.2-i2v-a14b-diffusers-8bit ...
mlxgen generate --model AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit ...
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

Supported router families are `qwen`, `flux2`, `bonsai`, `fibo`, `z-image`, `ernie-image`, and `wan`.

## ERNIE Images Look Cropped At Tiny Sizes

ERNIE Image Turbo is validated for practical generation at 384px and above. Very small outputs, such as 256x256, can crop or truncate subjects even when the pipeline is working.

Use 512x512 for small benchmark runs:

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

ERNIE Image Turbo supports text-to-image and single-image latent image-to-image. It
does not support edit/reference or multi-reference image-to-image.

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

If you pass `--task edit`, `--i2i-mode edit`, multiple `--images`, or `--image-strength` without an
image, MLX-Gen fails before loading the model and tells you which input shape and mode ERNIE
supports.

If ERNIE image-to-image changes the source too much, lower `--image-strength` or use Qwen Image
Edit for a true image-conditioned edit. If ERNIE barely applies the requested style, raise
`--image-strength` or increase `--steps` to 12-16.

## Image-To-Image Output Size Differs From `--width` And `--height`

This is expected with the default `--canvas-policy source-aspect`. For ordinary I2I, MLX-Gen treats
`--width` and `--height` as a size target and preserves the first input image's aspect ratio. Check
the generated metadata for `canvas_policy`, `requested_width`, `requested_height`,
`source_image_width`, `source_image_height`, and final `width`/`height`.

Use `--canvas-policy exact-resize` only when you intentionally want the exact requested canvas and
accept that the model may reshape or recompose the source.

## Wan Video Quality Looks Weak At Tiny Sizes

Wan2.2 supports TI2V-5B text-to-video, TI2V-5B first-frame image-to-video, T2V-A14B text-to-video, and I2V-A14B image-to-video. Very small or very short runs are useful for quick command checks, but they are not quality settings.

Use the upstream TI2V-5B settings when validating visual quality:

```sh
mlxgen generate \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --prompt "A short cinematic video of a glowing orange glass sphere floating above teal water" \
  --width 1280 \
  --height 704 \
  --frames 121 \
  --steps 50 \
  --guidance 5 \
  --fps 24 \
  --output video.mp4
```

Use lower dimensions, frame counts, or step counts only to validate routing and MP4 writing. For
image-to-video, pass exactly one input image; MLX-Gen infers I2V from the image input and selected
Wan model. Multi-image Wan interpolation is not enabled.

For Wan image-to-video prompts, describe concrete motion rather than only a style. Name the moving
body parts or object parts, keep continuity constraints in the positive prompt, and put common
failure modes such as `static still image`, `only camera movement`, `detached arm`, `malformed
hands`, `oversized foot`, `black frames`, and `sudden scene cut` in the negative prompt. See
[How Should I Prompt Wan Image-To-Video?](faq.md#how-should-i-prompt-wan-image-to-video) for
examples.

Wan defaults to the model's official negative prompt when the option is omitted. If a simple
abstract scene turns into noisy texture, retry with `--negative-prompt ""` to disable the default
negative prompt explicitly.

For T2V-A14B source/BF16 quality checks, use 1280x720 or 720x1280, 81 frames, 40 steps, `--guidance 4`, optional `--guidance-2 3`, and 16 fps. For mixed q8/BF16 packages, use the exact documented benchmark settings when comparing published measurements, and measure your target full-size profile before planning a long job. The separate I2V-A14B path requires a complete local `Wan-AI/Wan2.2-I2V-A14B-Diffusers` snapshot and one `--image` input.

Wan uses frame-count control rather than a separate duration flag. Duration is `frames / fps`; at 24 fps, 121 frames is about 5.04 seconds, and at 16 fps, 81 frames is about 5.06 seconds. Frame counts are normalized to `4n + 1`, and width/height are normalized to the selected Wan model's VAE/patch multiple. TI2V-5B requires 32-pixel width/height multiples; A14B requires 16-pixel multiples. For image-to-video, MLX-Gen also preserves the source image aspect ratio and resolves the final output canvas from the input image plus the requested size target. See [What Wan Video Resolutions Should I Use?](faq.md#what-wan-video-resolutions-should-i-use) for the full table.

If Wan generation or MP4 save validation fails, the CLI writes a failure manifest next to the intended output path, for example `video.failure.json` for `video.mp4`. The manifest includes the error, tensor-health report when available, seed, prompt, dimensions, frames, steps, guidance, fps, output path, and memory-related runtime flags.

## `generate --path` Fails

`--path` belongs to `mlxgen prepare`, where it names the local model folder to create. It is not a generation option.

To prepare a quantized model folder:

```sh
mlxgen prepare --model black-forest-labs/FLUX.2-klein-4B --path models/flux.2-klein-4b-4bit --quantize 4
```

To choose the generated image or video path, use `--output` with `mlxgen generate`.

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
