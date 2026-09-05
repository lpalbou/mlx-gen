# Troubleshooting

## MLX-Gen Will Not Download Files During Generation

This is expected. Runtime generation uses files that are already available locally. Run the command shown in the error message, then retry the generation.

Common commands:

```sh
mlxgen download --model Qwen/Qwen-Image
mlxgen prepare --model Qwen/Qwen-Image --path ./models/qwen-image-8bit --quantize 8
mlxgen download --model depth-pro
```

If you created a local MLX-Gen package under `./models/<repo-name>`, you can use either that local path or
the matching Hugging Face handle. MLX-Gen checks the local MLX-Gen package only when it is complete:

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

## A Package Calls `mflux-generate-flux2*` Directly And Fails

If another package shells out to `mflux-generate-flux2` or `mflux-generate-flux2-edit`, treat that
as a legacy integration path. New integrations should call `mlxgen generate` instead.

For FLUX.2, this difference matters because MLX-Gen's public FLUX.2 contract is:

- use `mlxgen generate`;
- omit `--negative-prompt` entirely;
- use `--image` for image-conditioned edit/reference workflows;
- use `mlxgen capabilities --model <flux2-model>` when you need the route contract before running.

Example migration:

```sh
# Legacy compatibility entry point
mflux-generate-flux2-edit \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --image-paths input.png \
  --prompt "Add sunglasses"
```

```sh
# Supported public entry point
mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --image input.png \
  --prompt "Add sunglasses" \
  --output edited.png
```

If the old integration also forwarded `--negative-prompt` to distilled FLUX.2 Klein weights, remove
it there: those weights have no guidance branch and the option is rejected before loading. Base
FLUX.2 Klein models accept it with `--guidance` above 1.0.

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

Use `--replace false` or `--no-replace` when you want to preserve an existing output file. In that mode, MLX-Gen writes the new image as `image_1.png`, then `image_2.png`, and keeps incrementing the suffix without overwriting the existing file.

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
accept that the model may reshape or recompose the source. When the exact canvas has a different
aspect ratio than the source, add `--resize-mode crop` (center-crop) or `--resize-mode pad`
(letterbox) to map the source onto it without distortion; the default `resize` stretches to fill.

## Outpainted Area Continues The Border Texture Instead Of Adding New Content

The space `--outpaint-padding` added returns as a stretched continuation of the source border, or as
directional streaks, rather than as new subject matter.

What you are seeing is the conditioning canvas. `edge` fill builds it by stretching a strip of the
source border outward across the padded area, so it continues an existing texture rather than
inventing one. The depth that strip covers is the *edge-fill reach*; past it, the same strip is
stretched far enough to read as one-dimensional streaks. The remedy is to pick a canvas that suits
the padding depth, and `auto` makes that choice for you.

Check which canvas the run used. Every outpaint run prints its resolved canvas on stderr before
denoising:

```text
Outpaint: fill=edge, canvas 928x1536 from source 768x766, padding top=0 right=76 bottom=766 left=76.
```

A second line follows only when `--outpaint-fill auto` resolved the mode, and it names the reason,
the padding depth, and the edge-fill reach the depth was measured against. A run that named its fill
explicitly prints the first line alone. Add `--metadata` to keep the same values in the JSON sidecar
as `outpaint_fill`, `outpaint_fill_requested`, `outpaint_fill_reason`,
`outpaint_edge_fill_reach_px`, and `outpaint_edge_fill_overreach`. To read the route contract
without running a job:

```sh
mlxgen capabilities --model AbstractFramework/flux.2-klein-base-4b-8bit
```

The `flux2.outpaint` row reports `supports_outpaint_fill`, `outpaint_fill_modes`,
`outpaint_default_fill_mode`, `outpaint_preservation`, and `outpaint_recommended_lora`.

On FLUX.2 Klein routes — distilled 4B/9B and base 4B/9B alike — ask for a blank conditioning canvas
so the model generates instead of continuing a texture:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-base-4b-8bit \
  --image portrait.png \
  --outpaint-padding "0%,10%,100%,10%" \
  --outpaint-fill neutral \
  --prompt "Extend this portrait downward to reveal the lower part of the body: the same subject in the same clothing, same lighting, same background." \
  --steps 20 \
  --guidance 4 \
  --seed 1234 \
  --output extended.png
```

Two further routes to better deep-padding results:

- Load the recommended outpaint adapter. Download it first, because generation does not download
  LoRA files. `--outpaint-fill auto` then paints the pure-green canvas the adapter is trained on:

  ```sh
  mlxgen download --model fal/flux-2-klein-4B-outpaint-lora --all-files

  mlxgen generate \
    --model AbstractFramework/flux.2-klein-base-4b-8bit \
    --image input.png \
    --outpaint-padding "5%,80%,5%,60%" \
    --prompt "Fill the green spaces according to the image" \
    --steps 20 \
    --guidance 4 \
    --seed 8612 \
    --lora-paths fal/flux-2-klein-4B-outpaint-lora:flux-outpaint-lora.safetensors \
    --lora-scales 1.0 \
    --output outpaint.png
  ```

- Extend in two moderate passes instead of one large one. Each pass gives the model a nearby edge to
  work from and runs on a smaller canvas, and outpaint attention cost grows faster than canvas area.
  This is also the route on Qwen Image Edit, which always builds an edge-extended canvas and takes
  no fill option.

`--outpaint-padding` computes the output size from the source and the padding, so do not add
`--width`, `--height`, or `--canvas-policy` to any of these commands.

To confirm the result, rerun with the same seed and read the first stderr line: it should report the
fill you asked for, and the added area should contain new content rather than stretched border
texture. See [Reframe and Outpaint](reframe-outpaint.md#the-conditioning-canvas) for the full mode
table and padding guidance, and
[Expanding On Any Side](reframe-outpaint.md#expanding-on-any-side) for `auto` results across
single-side, single-axis, and four-side padding on three source aspect ratios.

## Wan Video Quality Looks Weak At Tiny Sizes

Wan2.2 supports TI2V-5B text-to-video, TI2V-5B first-frame image-to-video, T2V-A14B text-to-video, T2V-A14B prompt-guided video-to-video (plain or masked with `--video-mask-path`), and I2V-A14B image-to-video. Very small or very short runs are useful for quick command checks, but they are not quality settings.

Use the upstream TI2V-5B settings when validating that route at its intended native scale:

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

For practical five-second prompt iteration on an M5 Max, A14B has produced stronger results than
TI2V-5B in the recorded starship-takeoff profile at much smaller dimensions: `480x240` or
`240x480`, `101` frames, 20 fps, and `20-25` steps. That profile takes about 30 minutes at
`480x240` on the recorded machine. TI2V-5B at `832x480`, `101` frames, 20 fps, and 25 steps took
about 12 minutes but was visually weaker; TI2V-5B at `1280x704` took about 35 minutes and improved
without matching the A14B result. See [Wan Video](wan-video.md) for the comparison clips and exact
prompt.

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

## Bernini Rejects Quantization Or Produces Weak Reference Fidelity

Bernini-R 1.3B is currently BF16-only. Omit `--quantize`. Generic Wan q4 was tested and produced
invalid overexposed latent-like output; nominal q8 quantized zero Bernini transformer linear
layers while labeling the run q8. Current MLX-Gen releases reject both instead of silently
returning a broken or misleading result.

The BF16 route itself is experimental and not yet promoted. Seven official public 1.3B example rows
plus `ads2v` at mid profile are qualitatively accepted with committed proof in
`docs/assets/validation/bernini-r-1.3b-2026-08-11/`. The historical schema-v3 bundle from 2026-08-04
still fails visual quality at the bounded 17-frame profile. Weak motion, missed references,
four-frame cadence jumps, or corruption from roughly frame 13 are known blockers on that historical
profile, not settings that the current docs claim to solve. Preserve the MP4 and metadata and
compare them with the committed parity bundle before assuming a successful exit means a successful
edit.

For a memory-constrained host, keep `--low-ram` enabled. The bounded model-backed proof peaked at
9.45 GB whole-process physical footprint on the 128 GB validation host, which predicts a useful
margin on an 18 GB Mac but is not a direct 18 GB-host proof. That does not mean 18 GB of fresh disk
is sufficient: the pinned factored selective download is about 16.36 GiB and requires 2 GiB
headroom, for 18.36 GiB free. Run:

```sh
mlxgen download --model bernini-r-1.3b
```

If R2V is coherent but misses requested accessories, check these before treating it as a runtime
bug:

- name references in their exact CLI order (`image0`, `image1`, ...);
- keep the prompt within the 512-token warning boundary;
- use the official 40-step quality budget and 848px default condition cap;
- use focused source images; a low-resolution 20-step proof can establish routing without
  retaining every reference detail;
- do not compare MLX and PyTorch integer seeds as pixel parity—export exact initial tensors for
  numerical comparison.

For RV2V/V2V, Bernini requires source-aspect, resize-only video conditioning. It intentionally
rejects crop, pad, exact-resize, `--video-strength`, masks, LoRA, and non-UniPC solvers. See
[Bernini-R 1.3B](bernini.md) and its [validation bundle](assets/validation/bernini-r-1.3b-2026-08-11/README.md).

## MiniMax-H3 Runs Out Of Memory Or Disk

MiniMax-H3 needs about 140 GB of disk for its snapshot and, with `--quantize 8`, peaks at about
80 GB of MLX memory (88 GB process footprint) on a `960x544` clip and 93 GB at `1344x768`. Run it on a 128 GB Mac with
`--quantize 8`, close other GPU-heavy applications, and prefer `minimax-h3-turbo-544p` over the
768p entries while iterating; the 768p canvas costs about five minutes per step. The MLX
free-buffer cache is capped at the process default (up to 8 GiB); `--mlx-cache-limit-gb` lowers or
lifts that cap. If `mlxgen download` stops early, check free disk space and rerun it (downloads
resume). The first load after a download reads 133 GB of shards from disk and takes a few minutes;
later loads are fast, and a prepared package (`mlxgen prepare --model minimax-h3 --quantize 8
--path ...`, 75 GB) skips the quantization pass on every load.

## `generate --path` Fails

`--path` belongs to `mlxgen prepare`, where it names the local MLX-Gen package to create. It is not a generation option.

To create a quantized MLX-Gen package:

```sh
mlxgen prepare --model black-forest-labs/FLUX.2-klein-4B --path models/flux.2-klein-4b-4bit --quantize 4
```

To choose the generated image or video path, use `--output` with `mlxgen generate`.

## LoRA Is Missing

LoRA support is route-specific. A requested LoRA is required: MLX-Gen fails the run rather than
continuing without it. Download the LoRA repository or use a local `.safetensors` file path.

```sh
mlxgen download --model RiverZ/normal-lora --all-files
```

If a repository contains several `.safetensors` files, specify the file:

```sh
mlxgen generate \
  --model <compatible-model> \
  --prompt "<prompt>" \
  --lora-paths owner/repo:adapter.safetensors \
  --lora-scales 0.9 \
  --output image.png
```

`--lora-scales` must have exactly one value per adapter, and it cannot be used without
`--lora-paths`.

## LoRA Is Not Compatible With The Model

LoRA adapters are trained for a specific base model. If MLX-Gen can read cached model-card
metadata, it rejects known incompatible combinations before loading model weights. It also checks
LoRA matrix shapes while applying the adapter.

For example, `lovis93/Flux-2-Multi-Angles-LoRA-v2` targets `black-forest-labs/FLUX.2-dev`.
Current FLUX.2 support in MLX-Gen is FLUX.2 Klein 4B/9B, so that adapter is not accepted for
`flux2-klein-*` or `AbstractFramework/flux.2-klein-*` models. Use an adapter trained for the exact
model family, or wait for first-class FLUX.2-dev support.

## hf_transfer Error

`HF_HUB_ENABLE_HF_TRANSFER=1` is optional. It can make explicit Hugging Face downloads faster, but it is not required to authorize downloads.

If you enable it and the `hf_transfer` package is unavailable, install MLX-Gen with the extra package available to the environment:

```sh
uv tool install --upgrade mlx-gen --with hf_transfer
```
