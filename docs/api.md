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
| `mlxgen generate` | Generate images, edit images, or generate supported videos from a cached or prepared model. |
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
  --model AbstractFramework/qwen-image-edit-2511-4bit \
  --image input.png \
  --prompt "Turn the room into a pencil sketch" \
  --output edited.png
```

Supported router families are `qwen`, `flux2`, `fibo`, `z-image`, `ernie-image`, and `wan`:

```sh
mlxgen generate \
  --model ./models/qwen-image-8bit \
  --family qwen \
  --prompt "A clean studio product photo"
```

Use `--config-from-metadata` / `-C` when you want the router to read fields such as `model`, `image_path`, or `image_paths` from an existing metadata file.

ERNIE Image Turbo routes through the same command surface:

```sh
mlxgen generate \
  --model baidu/ERNIE-Image-Turbo \
  --prompt "A clean product photo of a ceramic mug" \
  --width 512 \
  --height 512 \
  --steps 8 \
  --guidance 1 \
  --output image.png
```

ERNIE Image Turbo supports BF16 source weights plus prepared q8/q4 folders. MLX-Gen also provides experimental single-image image-to-image for ERNIE:

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

ERNIE image-to-image accepts exactly one input image. Multi-image edit is not supported. `--image-strength` follows the existing MLX-Gen image-influence convention: higher values preserve more of the init image, while lower positive values allow more transformation.

For ERNIE image-to-image, preserve the source aspect ratio when choosing `--width` and `--height`. Use roughly `--image-strength 0.25` to `0.35` for visible stylization, `0.45` to `0.6` for stronger source preservation, and 12-16 steps when the output needs more polished stylization. Use Qwen Image Edit for precise object/layout-preserving edits.

ERNIE's optional Prompt Enhancer is available with `--use-prompt-enhancer` when the full source snapshot is present. The default `mlxgen download --model baidu/ERNIE-Image-Turbo` command downloads only generation components; run `mlxgen download --model baidu/ERNIE-Image-Turbo --all-files` before using Prompt Enhancer. Prepared q8/q4 ERNIE folders created by `mlxgen prepare` do not include Prompt Enhancer files.

Wan2.2 TI2V routes through the same command surface for text-to-video:

```sh
mlxgen generate \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --task text-to-video \
  --prompt "A short cinematic video of a glowing orange glass sphere floating above teal water" \
  --width 1280 \
  --height 704 \
  --frames 121 \
  --steps 50 \
  --guidance 5 \
  --fps 24 \
  --output video.mp4
```

Wan image-to-video uses the same command with one input image:

```sh
mlxgen generate \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --task image-to-video \
  --image input.png \
  --prompt "A slow cinematic camera move from the input frame" \
  --width 1280 \
  --height 704 \
  --frames 121 \
  --steps 50 \
  --guidance 5 \
  --fps 24 \
  --output video.mp4
```

The I2V path follows Diffusers first-frame latent conditioning: the first frame is VAE-encoded, kept active through denoising with a timestep mask, and reinserted before decode. Multi-image/video interpolation is not enabled.

### Wan Video Parameters

Wan uses frame-count control rather than a separate duration flag. The output duration is:

```text
duration_seconds = frames / fps
```

At the default 24 fps, `--frames 121` produces about 5.04 seconds of video, `--frames 73` produces about 3.04 seconds, and `--frames 49` produces about 2.04 seconds.

| Option | Behavior |
| --- | --- |
| `--width`, `--height` | Accepted values are at least 32 pixels. Values are adjusted down to multiples of 32 for Wan patchification, so `1280x720` becomes `1280x704`. Use `1280x704` for landscape or `704x1280` for portrait quality validation. |
| `--frames` | Number of output frames. Wan requires `4n + 1`; other values are adjusted to `4 * floor(frames / 4) + 1`. Default: `121`. |
| `--fps` | MP4 playback frame rate. Any positive integer is accepted. Default and recommended value: `24`. |
| `--steps` | Denoising steps. Default and recommended quality value: `50`. Lower values run faster but reduce quality. |
| `--guidance` | Classifier-free guidance scale. Default: `5`. |
| `--seed` | Deterministic seed. Repeat with multiple values to create multiple videos. |

The upstream TI2V-5B guidance is 1280x704 or 704x1280, 121 frames, 50 steps, and 24 fps. Lower resolutions, frame counts, or step counts are useful for quick checks, but they should not be treated as quality settings.

Spatial-scale sanity outputs at 1280x704, 17 frames, and 20 steps:

![Wan2.2 TI2V 1280x704 text-to-video contact sheet](assets/generation/wan2.2-ti2v-5b-t2v-1280x704-17f-20steps-contact-sheet.png)

![Wan2.2 TI2V first-frame image-to-video contact sheet](assets/generation/wan2.2-ti2v-5b-i2v-bateau-1280x704-17f-20steps-contact-sheet.png)

These panels are not full quality benchmarks. They exist to show that the MLX-Gen T2V and I2V paths produce coherent, non-green output once the run uses the model's spatial scale.

## Model Management Commands

`mlxgen download` and `mlxgen prepare` are the only public MLX-Gen commands that authorize network access.

```sh
mlxgen download --model Qwen/Qwen-Image
```

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  --quantize 8
```

Use `prepare` when you need the local saved-weight folder. It is the public MLX-Gen workflow for creating quantized model folders and generated Hugging Face cards.

Generation output replaces the requested `--output` path by default. Use `--replace false` or `--no-replace` to preserve an existing file and save to a suffixed filename.

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
