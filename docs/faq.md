# FAQ

## Where Is The Save Command?

Use `mlxgen prepare`.

`prepare` creates a local MLX-Gen model package: a directory with MLX-Gen weights, config files,
and a Hugging Face `README.md` model card. Use it when you want a reusable local package or a
quantized package made specifically for MLX-Gen.

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  --quantize 8
```

New MLX-Gen integrations should not call a separate save workflow; use `mlxgen prepare`.

## What Is The Difference Between Download And Prepare?

`mlxgen download` populates the local Hugging Face cache with source files. It does not create a separate MLX-Gen package and does not write a model card.

`mlxgen prepare` creates a reusable local MLX-Gen package at `--path`. It can quantize weights and writes a generated Hugging Face model card.

## Can Generate Prepare A Model Package?

No. `mlxgen generate` is only for inference. It does not accept `--path` for model preparation.

To create a model package, use:

```sh
mlxgen prepare --model black-forest-labs/FLUX.2-klein-4B --path models/flux.2-klein-4b-4bit --quantize 4
```

To choose the output image path during generation, use `--output`.

## Does Generation Download Missing Files?

No. Generation and ordinary Python model construction use files that are already available locally. Missing artifacts raise `DownloadRequiredError` with the exact `mlxgen download` or `mlxgen prepare` command to run.

## Is `HF_HUB_ENABLE_HF_TRANSFER=1` Required?

No. It is optional acceleration for explicit Hugging Face downloads and prepare operations. `mlxgen download` and `mlxgen prepare` already authorize network access.

## Which Models Should I Start With On An 18 GB, 24 GB, 32 GB, 64 GB, Or 128+ GB Mac?

Use [Model Recommendations](recommendations.md).

That page groups the current published MLX-Gen model families by unified-memory tier and uses the
measured benchmark envelope for each recommended route. It is intentionally conservative: Wan rows
prefer full-process physical-peak measurements when they are available, and the page does not
promote model families whose public memory evidence is still too thin.

## Can MLX-Gen Packages Load In Diffusers Or Transformers?

No. MLX-Gen packages use the MLX/mflux saved-weight layout and MLX quantization tensors. They are intended for MLX-Gen and compatible mflux code, not direct Diffusers or Transformers `from_pretrained()` loading.

## Can I Use The Official BFL FLUX.2 FP8 Or NVFP4 Repos Directly?

No.

MLX-Gen `generate` and `prepare` flows support the standard FLUX.2 source repositories and
MLX-Gen model packages. The official `black-forest-labs/FLUX.2-klein-base-*-fp8` and
`black-forest-labs/FLUX.2-klein-base-*-nvfp4` repositories are not direct MLX-Gen inputs.

Those repositories use a different transformer packaging format from MLX-Gen model packages, and
the current MLX-Gen loader does not treat them as ready-to-run local models. Use the BF16 source
repository or an MLX-Gen package today.

## Why Can q8 Show The Same Physical Peak As BF16?

For Wan specifically, since the 2026-06-12 runtime-precision fix the answer is simple: Wan q8
packages store transformer-block linears quantized on disk but dequantize all of them to BF16 at
load to protect output quality. At runtime a Wan q8 package therefore uses BF16-class memory
(MLX peak and physical peak match the BF16 package); q8 saves disk and download only. Plan Wan
memory as if running BF16.

For other families where q8 stays quantized at runtime, metric definitions still matter:
`Physical Peak` is a full-process Darwin high-water sample that includes MLX/Metal allocations,
any PyTorch encoders, activation graphs, decode/save buffers, and native-library transients;
`Max RSS` can under-report Apple unified-memory/Metal pressure; `MLX Peak` is only the MLX
allocator high-water mark. A quantized model can be much smaller on disk and in loaded model
memory while a specific profile shows a similar full-process peak because temporary activations
or decode buffers dominate that run. See [Quantization](quantization.md) for the current tables,
definitions, and the dated Wan correction.

## Can I Quantize ERNIE Image Turbo?

Yes. ERNIE Image Turbo supports MLX-Gen 8-bit and 4-bit optimized packages:

```sh
mlxgen prepare --model baidu/ERNIE-Image-Turbo --path ./models/ernie-image-turbo-8bit --quantize 8
mlxgen prepare --model baidu/ERNIE-Image-Turbo --path ./models/ernie-image-turbo-4bit --quantize 4
```

ERNIE q4 uses a model-specific mixed q4/q8 policy. Fully q4 ERNIE checkpoints can drift from BF16/q8 behavior, so MLX-Gen keeps Mistral3 text linears plus selected ERNIE transformer attention-output and conditioning paths at q8.

## Can I Prepare Or Quantize Bonsai Image?

No. Bonsai Image repositories from Prism are already packed MLX artifacts. Use `download` and
generate directly:

```sh
mlxgen download --model prism-ml/bonsai-image-ternary-4B-mlx-2bit

mlxgen generate \
  --model prism-ml/bonsai-image-ternary-4B-mlx-2bit \
  --prompt "A bonsai tree in a quiet ceramic studio, soft morning light" \
  --width 1024 \
  --height 1024 \
  --steps 4 \
  --guidance 1 \
  --output bonsai.png
```

The ternary 2-bit checkpoint is supported. The binary 1-bit checkpoint is detected and rejected
with an explicit unsupported-runtime message until stock MLX can execute the required 1-bit packed
affine matmul. The latest published stock MLX checked for the 0.18.7 release was 0.31.2, and it
still rejected `bits=1`.

## Does ERNIE Image Turbo Support Image Input Or Prompt Enhancer?

Prompt Enhancer is supported for ERNIE Image Turbo when the full source snapshot is available:

```sh
mlxgen download --model baidu/ERNIE-Image-Turbo --all-files

mlxgen generate \
  --model baidu/ERNIE-Image-Turbo \
  --prompt "A ceramic mug" \
  --use-prompt-enhancer
```

ERNIE Image Turbo supports single-image latent image-to-image in MLX-Gen:

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

Multi-image edit is not supported for ERNIE. ERNIE's single-image path is latent image-to-image, so
`--image-strength` follows latent img2img denoising semantics: higher values add more noise and
allow more transformation, while lower positive values stay closer to the encoded source image.

ERNIE q8/q4 MLX-Gen packages do not bundle Prompt Enhancer files; use the full source snapshot path or the Hugging Face repo after `mlxgen download --all-files` when you need `--use-prompt-enhancer`.

ERNIE LoRA support is route-specific. `AbstractFramework/ernie-image-turbo-8bit` has exact
validated text-to-image and latent img2img LoRA rows. Use
`mlxgen capabilities --model AbstractFramework/ernie-image-turbo-8bit` to inspect the exact
current status before relying on a specific adapter workflow.

## Does Bonsai Image Support LoRA?

No. MLX-Gen currently rejects Bonsai LoRA requests.

The practical blocker is architectural, not just missing validation. Bonsai runs through a packed
ternary/low-bit transformer path, while MLX-Gen's LoRA loader applies adapters by replacing normal
linear modules. That replacement boundary does not exist in the current packed Bonsai runtime, so
MLX-Gen fails closed instead of pretending the adapter applied.

## Does Wan Video Support LoRA?

Yes, for the supported Wan q8 public routes. Exact validated rows exist for:

- `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` on `wan.text-video`
- `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` on `wan.first-frame`
- `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` on `wan.text-video`
- `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` on `wan.first-frame`

Wan is not blocked the way Bonsai is. The current MLX Wan path still uses normal linear attention
and FFN layers, so LoRA injection is technically viable. Use `mlxgen capabilities --model <model>`
and check `lora_status`, `lora_target_roles`, and `lora_validation_profile` before relying on a
specific Wan LoRA workflow. For the current A14B Lightning example, start with:

```sh
mlxgen download --model lightx2v/Wan2.2-Lightning --all-files
```

Then use the paired T2V or I2V files from that repository as
`lightx2v/Wan2.2-Lightning:<subdir>/<file>.safetensors`; the copy-paste commands are in
[LoRA](lora.md). You can also point directly at absolute local `.safetensors` files after
downloading them. For Wan A14B, pass each file as its own `--lora-paths` argument rather than one
quoted combined string.

## Can I Treat MLX-Gen q8 Packages Like Third-Party FP8 Checkpoints For Lightning LoRAs?

No.

The current MLX-Gen recommendation is to use the validated `AbstractFramework/*-8bit` q8 package
for the route you want and pair it with the exact documented Lightning adapter for that route.

That is not the same thing as taking an arbitrary external FP8 checkpoint and assuming it will
behave like MLX-Gen q8. The upstream LightX2V Qwen Lightning README explicitly warns that
BF16-trained Lightning LoRAs do not automatically behave correctly on every FP8 Qwen base:

- <https://github.com/ModelTC/LightX2V-Qwen-Image-Lightning#-using-lightning-loras-with-fp8-models>

For MLX-Gen, the practical rule is simple:

- prefer the validated q8 MLX-Gen package when one exists;
- use the exact Lightning adapter example documented for that route;
- do not generalize those q8 results to unrelated external FP8 checkpoints.

## How Do I Choose Between Latent I2I And Image Edit?

MLX-Gen keeps one public `image-to-image` task and exposes different internal modes through model
capabilities. Use `mlxgen capabilities --model <model>` to inspect the selected model before a long
run. For a mode-by-mode guide with examples, see [Image Edit Modes](image-edit-modes.md).

Use latent img2img when you want a whole-image variation or broad restyle from one source image:
make the lighting more cinematic, change the mood, loosely restyle the whole scene, or preserve the
source while allowing the model to reinterpret details. Select it with `--image-strength` or
`--i2i-mode latent` on a model that supports `latent-img2img`:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --image input.png \
  --i2i-mode latent \
  --image-strength 0.35 \
  --prompt "Make the scene a moody graphite and charcoal illustration" \
  --output latent-restyle.png
```

Use edit/reference I2I when the prompt is an instruction: remove an object, change a subject's
color, turn a scene into a pencil sketch while preserving layout, reposition or reshape an object,
or keep the composition stable. Select it with an edit-capable model and no `--image-strength`, or
force it with `--i2i-mode edit`:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --image input.png \
  --i2i-mode edit \
  --prompt "Turn the scene into a clean pencil sketch while preserving the object layout" \
  --output edit-sketch.png
```

Use multi-reference I2I when two or more input images provide different references, such as one
image for content and another for style:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --image content.png \
  --image style.png \
  --prompt "Use the first image composition and the second image drawing style" \
  --output multi-reference.png
```

`--image-strength` is not used by edit/reference or multi-reference modes. Those modes use source
images as conditioning or references, not as a noised latent initialization. MLX-Gen rejects
`--image-strength` for those modes before loading weights.

For Qwen edit models, use the exact version you intend. `qwen-image-edit` is the original
single-reference edit checkpoint. Use `qwen-image-edit-2509` or `qwen-image-edit-2511` when you
need a Qwen edit model that can route multi-reference requests.

## How Do I Do Masked Edit Or Inpaint?

Use a model that supports masked edit or inpaint, then pass one input image plus `--mask-path`.
White mask pixels are repainted and black mask pixels are preserved.

Without `--mask-path`, the same edit route can still recompose the whole frame. The mask is what
keeps the change local.

Masked editing is supported on Qwen edit models, base Qwen models (native `qwen.base-inpaint`,
or the validated control-inpaint sidecar on the exact `AbstractFramework/qwen-image-8bit` row),
Z-Image Turbo, and FLUX.2 Klein distilled and base (with optional masked-area
reference images on the backend route). [Masked editing](masked-editing.md) is the canonical
page for the model matrix, per-family behavior, and proof grades.

The recommended fast public Qwen masked-edit path uses the dedicated
`lightx2v/Qwen-Image-Edit-2511-Lightning` adapter:

```sh
mlxgen download --model lightx2v/Qwen-Image-Edit-2511-Lightning --all-files

mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image input.png \
  --mask-path mask.png \
  --prompt "Repair the damaged hull inside the mask and keep the rest of the scene unchanged." \
  --steps 4 \
  --guidance 1 \
  --output repaired.png \
  --lora-paths lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
  --lora-scales 1
```

For the accepted contact sheet, timings, and full example commands, see
[Image Edit Capabilities](edit-capabilities.md) and [LoRA](lora.md).

## How Do I Use Qwen Structured Control?

Use a capability row that reports `supports_control_image=true`, then pass one control image with
`--controlnet-image-path`. This is a structured text-to-image route, not a source-image edit
route, so do not combine it with `--image`.

The current exact public proof row is `AbstractFramework/qwen-image-8bit` on `qwen.control`. The
recommended fast public path uses the dedicated `lightx2v/Qwen-Image-Lightning` adapter:

```sh
mlxgen download --model lightx2v/Qwen-Image-Lightning --all-files

mlxgen generate \
  --model AbstractFramework/qwen-image-8bit \
  --prompt "Aesthetics art, traditional asian pagoda, elaborate golden accents, sky blue and white color palette, swirling cloud pattern, digital illustration, east asian architecture, ornamental rooftop, intricate detailing on building, cultural representation." \
  --negative "blurry, low quality, distorted, deformed, text, watermark, ugly" \
  --width 576 \
  --height 864 \
  --steps 4 \
  --guidance 1 \
  --seed 5802 \
  --controlnet-image-path canny.png \
  --lora-paths lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors \
  --lora-scales 1 \
  --output controlled.png
```

The exact sidecar is injected automatically for that validated row. For the accepted contact sheet,
timings, and full example commands, see [Image Edit Capabilities](edit-capabilities.md) and
[LoRA](lora.md).

## How Is Qwen Masked Edit Different From ControlNet Inpaint?

Today, MLX-Gen ships:

- Qwen masked edit / inpaint on the Qwen edit route;
- Qwen native masked edit on base Qwen rows (`qwen.base-inpaint`, no sidecar);
- Qwen structured control on the exact validated base row;
- Qwen base control-inpaint on the exact validated `AbstractFramework/qwen-image-8bit` row.

The practical difference is:

- masked edit means “edit this existing image, but only inside the white mask”;
- structured control means “generate from text, but use a guide image to lock the layout”;
- control-inpaint means “still edit only one masked region, but use an extra control model to make
  that local replacement more disciplined.”

So control-inpaint is not mainly a new kind of prompt. It is a stricter backend for harder local
edits.

ControlNet is not a LoRA. It is an extra model package loaded beside the base model for one route.
When MLX-Gen docs say “sidecar”, that is all they mean: an extra model package loaded alongside the
main one.

For the detailed route comparison, pros/cons, and the current proof sheets, see
[Qwen localized editing](qwen-localized-editing.md).

## Which Qwen Image Edit Model Should I Use?

Use the exact Qwen edit handle for the capability you need:

| Model family | Best use | Multi-reference composition |
| --- | --- | --- |
| `Qwen/Qwen-Image-Edit` | One-source semantic or appearance edits, such as pencil sketch, object-state changes, color/style edits, and layout-preserving instruction edits. | No. MLX-Gen exposes this as `max_images=1`. |
| `Qwen/Qwen-Image-Edit-2509` | One-source edits and multi-image reference composition. | Yes, when the selected source model or MLX-Gen optimized package passes validation for the prompt profile. |
| `Qwen/Qwen-Image-Edit-2511` | One-source edits and multi-image reference composition with the 2511 checkpoint. | Yes. Source, q8, and q4 have passing 2026-06-06 proof for the documented pencil/crash/composition profile. |

For composition with multiple images, repeat `--image` and use a multi-reference edit model:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2509-8bit \
  --image crash-reference.png \
  --image sketch-reference.png \
  --prompt "Use the first image for the crashed spaceship layout and the second image for the graphite sketch style" \
  --output composition.png
```

The regular `qwen-image-edit` route intentionally rejects multi-reference input before loading
weights. Use `mlxgen capabilities --model <model>` to see the image-count contract and
`mlxgen validation --model <model>` to inspect the published release evidence for an exact package.

## How Do Negative Prompts Work?

Use `--negative-prompt` or its shorter alias `--negative` in `mlxgen generate`:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-8bit \
  --image input.png \
  --prompt "Convert the scene into a clean graphite pencil sketch while preserving layout" \
  --negative "color, blur, crop, text, watermark" \
  --steps 30 \
  --guidance 4 \
  --output sketch.png
```

Python callers pass the same value as `negative_prompt=...` on the model-specific generation
method. For Qwen image edit, guidance above `1` uses true classifier-free guidance when a negative
prompt is present. If you omit the CLI option, MLX-Gen uses the official blank negative-prompt
behavior for Qwen edit models, so true CFG remains enabled by default; explicit negative prompts are
still useful for blocking concrete failure modes such as crop, blur, text, unwanted color, or an
object remaining intact when the prompt asks for damage.

Wan video models are different: when the negative prompt is omitted, MLX-Gen uses Wan's official
default negative prompt. Pass `--negative ""` or `--negative-prompt ""` only when you intentionally
want no negative prompt.

FLUX.2 is different again: FLUX.2 Klein routes do not support negative prompts in MLX-Gen. Omit
`--negative` / `--negative-prompt` entirely for FLUX.2 runs.

## Should Integrations Call `mflux-generate-*` Commands Directly?

No. New integrations should call the public `mlxgen` commands instead:

- `mlxgen generate`
- `mlxgen upscale`
- `mlxgen capabilities`
- `mlxgen validation`
- `mlxgen download`
- `mlxgen prepare`

MLX-Gen still installs some `mflux-generate-*` compatibility entry points from the upstream
codebase, but they are not the recommended integration contract for new tools.

For example, a FLUX.2 integration should use:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --prompt "A cinematic wide shot of a compact sci-fi spaceship resting in deep snow" \
  --width 768 \
  --height 432 \
  --steps 24 \
  --guidance 1.0 \
  --seed 6107 \
  --output spaceship.png
```

Do not call `mflux-generate-flux2` or `mflux-generate-flux2-edit` directly from a package unless
you are deliberately targeting the legacy compatibility layer.

## How Does Image-To-Image Choose Output Size?

Ordinary image-to-image uses `--canvas-policy source-aspect` by default. The first input image
defines the output aspect ratio for latent img2img, edit/reference I2I, and multi-reference I2I.
`--width` and `--height` are size targets, not forced stretch dimensions. MLX-Gen resolves the
nearest model-compatible canvas that preserves the first image's ratio and stores the requested,
source, and final dimensions in image metadata.

Examples:

| Source | Request | Default output behavior |
| --- | --- | --- |
| `432x240` | `--width 320 --height 320` | wide output near the source ratio, not square |
| `512x512` | `--width 832 --height 480` | square-ish output near the source ratio, not 16:9 |
| `720x1280` | `--height 640 --width auto` | portrait output near the source ratio |

Use `--canvas-policy exact-resize` when you intentionally want the exact requested canvas:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-2512-8bit \
  --image input.png \
  --i2i-mode latent \
  --canvas-policy exact-resize \
  --width 512 \
  --height 512 \
  --image-strength 0.4 \
  --prompt "Restyle the source as a graphite sketch" \
  --output exact-square.png
```

Exact resize can reshape or recompose the source. Use it for deliberate whole-image remixes, not
for preserving original pixels in place.

## How Does SeedVR2 Upscale Sizing Work?

SeedVR2 uses `mlxgen upscale`. Its `--resolution` option supports two sizing styles:

| Form | Meaning | Example from `320x192` |
| --- | --- | --- |
| `--resolution 1024` | Set the shorter output edge to 1024px and preserve the source aspect ratio. | `1706x1024` after even-dimension normalization |
| `--resolution 2x` | Multiply both source dimensions by 2. | `640x384` |
| `--resolution 3x` | Multiply both source dimensions by 3. | `960x576` |

Shortest-edge sizing is useful when you want a predictable target class across different source
ratios. Scale factors are useful when you want direct 2x/3x comparisons.

SeedVR2 is a diffusion super-resolution/restoration model, so it may also denoise and reconstruct
detail. If the target is only slightly larger than the source, the result can mainly demonstrate
restoration rather than upscale quality. For visual upscale validation, use a source with visible
low-resolution artifacts and choose a target that materially increases pixel dimensions, such as
`2x`, `3x`, or a much larger shorter-edge target.

For noisy low-resolution sources, use `--softness 0.25` to `0.5` to smooth source grain before the
diffusion reconstruction. `--softness 0.0` keeps the source conditioning most direct. Higher values
temporarily downsample and re-enlarge the conditioning image, which can reduce grain or JPEG
texture but can also soften fine details. SeedVR2 keeps small image outputs untiled for quality and
automatically uses tiled VAE decode for large image outputs. Use `--vae-tiling` when you also want
tiled VAE encoding or the same tiled image path for smaller outputs. Video restore rejects
`--vae-tiling`; use `--low-ram` and temporal chunking instead.

See [Image Upscaling](upscaling.md) for a checked-in 5x SeedVR2 comparison where the original
source is enlarged to the generated output resolution for side-by-side assessment.

## Can SeedVR2 Restore Video?

Yes, through `mlxgen upscale --video-path ...`.

Use `--start-seconds` and `--max-frames` for a real five-second validation clip before you try a
longer source:

```sh
mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-3B \
  --video-path input.mp4 \
  --start-seconds 70 \
  --max-frames 149 \
  --resolution 1x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 29 \
  --temporal-chunk-overlap 8 \
  --low-ram \
  --mlx-cache-limit-gb 8 \
  --metadata \
  --output restored.mp4
```

Current behavior:

- source FPS is preserved by default;
- temporary SeedVR2 padding is trimmed back to the requested clip length before save;
- by default, MLX-Gen preserves the matching source audio segment when the source clip has audio;
- if copied audio cannot be proven safe, the run fails instead of silently dropping it;
- use `--drop-audio` only when you intentionally want a silent restored MP4;
- the public CLI safe profile defaults to `1x`, enables `--low-ram` automatically, uses
  `--mlx-cache-limit-gb 8` as part of the MLX cache policy, and rejects enlarged video output
  unless you explicitly pass `--force-unsafe-video-memory`;
- the public Eiffel quality proof is a five-second `70s` to `75s` reader-first clip, with safe
  bounded `1x` and explicit enlarged `2x` 3B/7B comparison MP4s, motion strips, detail crops, and
  a readable report.

It is a better fit for visibly degraded, noisy, low-resolution, or compressed footage than for
already-clean high-resolution footage. In local validation, archival material improved cleanly,
while already-clean modern footage could over-smooth instead of improving.

For long archival clips, start with:

```sh
mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-3B \
  --video-path input.mp4 \
  --resolution 1x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 29 \
  --temporal-chunk-overlap 8 \
  --low-ram \
  --mlx-cache-limit-gb 8 \
  --metadata \
  --output restored.mp4
```

Validate a five-second slice first with `--start-seconds` and `--max-frames`, then use the same
profile for the longer clip.

## Can SeedVR2 Use The Official ByteDance Checkpoint?

Yes. The `seedvr2` and `seedvr2-3b` aliases resolve to the official
`ByteDance-Seed/SeedVR2-3B` checkpoint, and `seedvr2-7b` resolves to
`ByteDance-Seed/SeedVR2-7B`. To run the 3B source model directly, download it and pass the full
Hugging Face handle:

```sh
mlxgen download --model ByteDance-Seed/SeedVR2-3B

mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-3B \
  --image-path input.png \
  --resolution 2x \
  --seed 42 \
  --metadata \
  --output input_seedvr2_official_3b_2x.png
```

For day-to-day use, prefer the published MLX-Gen packages:

```sh
mlxgen download --model AbstractFramework/seedvr2-3b-8bit

mlxgen upscale \
  --model AbstractFramework/seedvr2-3b-8bit \
  --image-path input.png \
  --resolution 2x \
  --seed 42 \
  --metadata \
  --output input_seedvr2_q8_2x.png
```

`AbstractFramework/seedvr2-3b-8bit` and `AbstractFramework/seedvr2-3b-4bit` are reusable
MLX-Gen packages generated from the official 3B source model. See [Quantization](quantization.md)
for package sizes and the measured 5x validation profile.

For 7B, download the official source and run `--model seedvr2-7b`, use
`AbstractFramework/seedvr2-7b-8bit` or `AbstractFramework/seedvr2-7b-4bit`, or prepare a local
q8/q4 package:

```sh
mlxgen prepare \
  --model ByteDance-Seed/SeedVR2-7B \
  --path ./models/seedvr2-7b-8bit \
  --quantize 8
```

The 7B source, q8, and q4 local packages are documented in [Image Upscaling](upscaling.md).

## Why Does Image-To-Image Run Fewer Steps Than `--steps`?

This is normal for latent image-to-image pipelines. They commonly start partway through the
denoising schedule instead of running all requested steps from pure noise. In MLX-Gen,
`--image-strength` is the latent img2img denoising strength: higher values add more noise, allow
more transformation, and run more effective denoise iterations. Lower values stay closer to the
encoded source image. Edit/reference I2I modes do not use `--image-strength`; they use the input
image as conditioning.

Latent image-to-image requires an explicit `--image-strength`. With `--steps 50` and
`--image-strength 0.4`, generation starts at scheduler index 30 and runs 20 denoise iterations:

```text
effective_denoise_steps = floor(steps * image_strength)
floor(50 * 0.4) = 20
```

The CLI progress bar shows the effective denoise iterations, not the original requested step count.
If you want a stronger transformation from the source image, raise `--image-strength`; if you want
more source preservation, lower it. Use text-to-image without `--image` when you want all requested
steps to run from pure noise.

## Can MLX-Gen Outpaint Or Reframe An Image?

MLX-Gen supports generative reframe for edit models that advertise
`supports_reframe=true` in `mlxgen capabilities`. Use `--reframe-padding` with one input image to
ask the edit model for a larger view:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --image input.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the full subject and extend the background naturally." \
  --steps 16 \
  --seed 42 \
  --output reframed.png
```

This is a generative edit. It is useful for zoom-out, background extension, and plausible
reconstruction of missing object boundaries, but the model may redraw source content. The prompt
remains important, especially when the source object is cropped and the missing parts must be
inferred.

Use `--outpaint-padding` when you want MLX-Gen to create a larger canvas and guide a
supported edit model to fill the expanded view:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image input.png \
  --outpaint-padding "5%,35%,5%,35%" \
  --prompt "Outpaint this close crop into a wider realistic shot. Complete the missing subject and background outside the original frame." \
  --negative "text, border, frame, hard seam, duplicate subject" \
  --steps 24 \
  --guidance 4 \
  --seed 42 \
  --output outpaint.png
```

Outpaint is backend-specific. Qwen Image Edit variants still use expanded-canvas generation plus
adaptive source restoration. Current FLUX.2 Klein strict outpaint is base-only and uses
source-locked denoising with a narrow latent transition band instead of post-generation source
pasting. The current proof is in [Image Edit Capabilities](edit-capabilities.md#reframe-and-outpaint)
and [Reframe and Outpaint](reframe-outpaint.md).

This is not a native fill/inpaint backend with an explicit diffusion mask, and it is not an exact
pixel-lock guarantee. Latent I2I models, Z-Image, ERNIE, FIBO, base Qwen Image, Qwen Image 2512,
distilled FLUX.2 Klein, Wan, SeedVR2, and unsupported edit models reject `--outpaint-padding`
before loading weights. Exact current prepared-package outpaint proof exists for
`AbstractFramework/flux.2-klein-base-4b-8bit`; broader package claims should still follow the
published validation rows.

For ordinary image-to-image, the default `source-aspect` canvas policy keeps the output ratio close
to the first source image. That prevents accidental stretching, but it does not expand the original
canvas or preserve source pixels in place. Use `--canvas-policy exact-resize` only for deliberate
whole-image recomposition. When the canvas ratio differs from the source, `--resize-mode` chooses
how source pixels map onto it: `resize` stretches to fill (default), `crop` center-crops without
distortion, and `pad` letterboxes the full source without distortion. Masks always follow the same
geometry as the source pixels.

## What Wan Video Resolutions Should I Use?

Wan width and height are normalized to the selected model's VAE/patch multiple. For text-to-video,
MLX-Gen uses the requested canvas after model-multiple normalization. Plain video-to-video uses the
same requested-canvas rule, and source frames are stretched to that canvas by default: if the
source clip's aspect ratio differs from the requested `--width`/`--height` ratio, the source is
distorted to fit and MLX-Gen prints a warning, so match the requested ratio to the source — or pass
`--canvas-policy source-aspect` to derive the canvas from the clip, or `--resize-mode crop|pad` to
map the frames without distortion. For image-to-video, MLX-Gen preserves the source image aspect
ratio: requested `--width` and `--height` define the approximate size target, and the runtime
resolves the closest supported canvas from the input image ratio before conditioning the model. The
video is generated directly at the resolved canvas; MLX-Gen does not generate a stretched canvas
and then crop or resize it back afterward. Pass `--canvas-policy exact-resize` when the requested
canvas itself is the requirement; combine it with `--resize-mode crop` or `pad` to avoid
distortion on a mismatched source.

| Model | Required multiple | Recommended/native size | Lower-cost diagnostic sizes |
| --- | ---: | --- | --- |
| TI2V-5B T2V/I2V | 32 px | `1280x704` or `704x1280` | `832x480`, `480x832`; smaller sizes such as `448x256` are smoke checks only |
| T2V-A14B | 16 px | `1280x720` or `720x1280` | `832x480`, `480x832`, `448x256`, `256x448`, `432x240` |
| I2V-A14B | 16 px | `1280x720` or `720x1280` | `832x480`, `480x832`, `448x256`, `256x448`, `432x240` |

For TI2V-5B text-to-video, `1280x720` adjusts to `1280x736`, and `432x240` adjusts to `448x256`.
For A14B text-to-video, `1280x720`, `832x480`, `448x256`, and `432x240` are already valid multiples
of 16. Plain A14B video-to-video follows that same requested-canvas rule. For TI2V-5B, use at least `832x480` for visual prompt checks; smaller canvases are useful for
route checks only. Use the
recommended/native size, frame count, and step count when judging visual quality.

## Does Wan Support Video-To-Video Editing?

Yes, in a narrow public form.

MLX-Gen currently supports prompt-guided video-to-video on `Wan2.2-T2V-A14B`. You pass one source
video plus one prompt, and MLX-Gen regenerates the clip while keeping the source clip's camera
path and composition. Subject gestures and timing are re-synthesized at typical strengths, not
copied - prompts like "keep the same motion" cannot force exact motion through. Add
`--video-mask-path` when you need preserved regions locked to the source (their motion is
preserved exactly too), or lower `--video-strength` to 0.5-0.6 at 20 steps with CFG on for
motion-preserving restyles - the measured band where source gestures survive (gesture-timing
correlation 0.86-0.90 vs 0.20 at the 0.8 default; see the motion-fidelity ladder in
[Wan Video](wan-video.md#motion-fidelity-versus-strength)). The edit still landed at 0.5 in the
measured runs, but low strength is weaker for adding brand-new objects - use a mask for those.

Use this route. The sizes and counts below are bounded diagnostic settings for a quick check, not
quality settings; for quality, use the A14B defaults (`832x480` or `1280x720`, `81` frames,
`40` steps):

```sh
mlxgen generate \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --video-path source.mp4 \
  --prompt "Keep the same setting and camera framing, but redesign the main subject" \
  --width 448 \
  --height 256 \
  --frames 17 \
  --steps 5 \
  --guidance 4 \
  --guidance-2 3 \
  --video-strength 0.7 \
  --solver unipc \
  --fps 10 \
  --seed 4242 \
  --output edited.mp4
```

`--video-strength` defaults to `0.8` and controls how far the result may move from the source: the
run denoises `floor(steps x video_strength)` effective steps (the example above resolves to `3`),
and saved metadata records both the requested `steps` and the resolved `effective_steps`. Below
roughly `0.7`, the A14B high-noise stage is skipped and `--guidance` becomes inactive; MLX-Gen
prints a warning when that happens.

Limits that matter:

- this is not SeedVR2 restore or upscale;
- plain video-to-video (no mask) re-synthesizes everything, so background text and logos drift;
  pass `--video-mask-path` to lock everything outside a mask to the source video exactly;
- source audio is copied onto the output best-effort (trimmed to the output duration); if the
  copy cannot complete, the output is saved silent with a printed reason and a manual remux
  command, and metadata records `audio_copied` / `audio_copy_reason`;
- the source is resampled onto the `--fps` timeline, so the output keeps real-time speed;
  requesting an fps above the source duplicates frames (a warning says so); matching fps passes
  frames through untouched, and metadata records `source_video_resampled`;
- the A14B route does not accept extra reference images or VACE-style controls - for
  reference-image injection and learned mask conditioning use the natively ported
  `wan-vace` model (see [Wan Video](wan-video.md#vace-reference-images-and-learned-mask-conditioning));
- source frames are stretched to the requested canvas by default, so match the aspect ratio to the
  source, derive the canvas from the clip with `--canvas-policy source-aspect`, or map without
  distortion via `--resize-mode crop|pad`;
- TI2V-5B and I2V-A14B do not currently accept `--video-path`.

## How Do I Change One Thing Without Changing Anything Else In Video-To-Video?

Use masked video-to-video: add `--video-mask-path mask.png` to the video-to-video command. This
is the recommended tool for local add/remove/replace edits ("add a red tie", "remove the logo"):
white regions of the mask are regenerated under your prompt; black regions - background AND
subject, including their exact motion and gestures - are locked to the source video at every
denoising step and match it up to VAE round-trip precision. Keep edit masks tight: motion that
crosses into the white region is re-synthesized inside it. Draw the mask over the union of the
edited region's positions if it moves. All-black masks are rejected (they would edit nothing),
and an all-white mask is equivalent to plain video-to-video. See
[Wan Video](wan-video.md#masked-video-to-video) for the full contract and a reproducible example.

For a full explanation and a reproducible example with the exact accepted command, see [Wan Video](wan-video.md).

TI2V-5B also has a flow-matching schedule shift. MLX-Gen uses the model default `5.0` for native
`1280x704` or `704x1280` runs. For new 480p-class TI2V-5B checks such as `832x480`, pass
`--flow-shift 3`.

For practical five-second prompt iteration on an M5 Max, the A14B T2V/I2V routes have looked good
at `480x240` or `240x480` with `101` frames, `20` fps, and `20-25` steps. A `101`-frame, 20 fps
clip is about five seconds and takes roughly 30 minutes at `480x240` with A14B in the recorded
starship-takeoff profile. In that same prompt family, TI2V-5B at `832x480`, `25` steps, `101`
frames, and 20 fps took about 12 minutes but was weaker visually; rerun this 480p-class profile
with `--flow-shift 3` for current testing. TI2V-5B at its `1280x704` native size took about 35
minutes and improved but still did not match the A14B result at lower resolution. See
[Wan Video](wan-video.md) for the comparison clips and commands.

For image-to-video, the source ratio controls the final canvas. These are typical resolved outputs:

| Model | Source ratio | Requested target | Generated canvas |
| --- | --- | ---: | ---: |
| I2V-A14B | `16:9` | `1280x720` | `1280x720` |
| I2V-A14B | `9:16` | `1280x720` | `720x1280` |
| I2V-A14B | `16:9` | `832x480` | `848x480` |
| I2V-A14B | `1:1` | `432x240` | `320x320` |
| TI2V-5B I2V | `20:11` | `1280x704` | `1280x704` |
| TI2V-5B I2V | `16:9` | `1280x704` | `1248x704` |
| TI2V-5B I2V | `16:9` | `432x240` | `448x256` |
| TI2V-5B I2V | `1:1` | `432x240` | `320x320` |

If you need an exact output size, prepare the source image at the same aspect ratio as the requested
target and choose dimensions that match the model multiple. For A14B I2V, a `16:9` source requested
at `1280x720` produces `1280x720`. For `480p`-class A14B I2V, `832x480` produces exactly `832x480`
when the source also has the `832:480` aspect ratio; a true `16:9` source resolves to `848x480`.
For TI2V-5B I2V, `1280x704` produces exactly `1280x704` when the source uses the same `20:11`
aspect ratio. Alternatively, pass `--canvas-policy exact-resize` to honor the requested
(multiple-adjusted) canvas directly, with `--resize-mode crop` or `pad` mapping a mismatched
source onto it without distortion.

## How Should I Prompt Wan Image-To-Video?

Wan image-to-video responds best when the input image and prompt agree on a plausible motion path.
Use a source frame with the whole subject visible, enough margin around moving limbs or objects, and
minimal occlusion for body parts that need to move. MLX-Gen resolves the output canvas from the
source image aspect ratio, so a portrait source stays portrait and a landscape source stays
landscape. TI2V-5B normalizes width and height to multiples of 32, so keep enough edge margin for
the adjusted canvas.

Keep the main subject inside the rendered frame for the whole intended motion. If a face, hand,
foot, product edge, or other identity-critical region leaves the frame and later re-enters, the
model may reconstruct it inconsistently. Once that region leaves the visible image, the model no
longer has rendered pixels to anchor its identity or geometry. When it comes back, it is effectively
reconstructed from latent and temporal context, so faces, hands, limbs, logos, edges, and other
details can drift or mutate. For human motion, a pose that already suggests the intended action is
usually more reliable than a neutral pose, but avoid edge-reaching poses unless the prompt keeps the
camera wide enough to preserve head, hands, and feet.

For faces and other front-facing identity cues, also constrain rotation. If a person turns far enough
that the face is no longer visible, the model may reassign the back of the head, hair, shoulders, or
clothing as the new front when the subject turns back. When face continuity matters, ask for
front-facing or three-quarter-front motion, keep torso pivots below about 60-90 degrees, and block
rear views in the negative prompt. This usually improves identity stability, but it can make motion
more restrained.

Wan uses the model's official default negative prompt when `--negative-prompt` is omitted. That is a
good starting point for many human/action prompts, but it can over-constrain simple abstract scenes
and introduce noisy texture. Use `--negative-prompt ""` when you intentionally want no negative
prompt, especially for minimal object, water, sky, or studio-light tests.

Write the positive prompt as a concrete motion plan instead of a general style request. Name the
subject, camera style, body parts or object parts that should move, and the continuity constraints
that should remain stable:

```text
Cinematic 5 second full-body motion video of the athlete performing controlled lateral steps:
torso pivots, head turns, arms sweeping naturally from high and low positions, legs crossing and
uncrossing, weight shifting forward and backward, knees bending and straightening, full head visible,
full body visible, arms attached naturally at shoulders and wrists, natural hands and feet,
consistent outfit, smooth studio camera.
```

For object motion, use the same pattern:

```text
Cinematic 5 second takeoff video of the spacecraft clearly lifting away from the snowy landing
field, landing gear leaving the snow, bright engines firing underneath, snow and ice blowing
outward, stable hull geometry, consistent metal panels, smooth upward camera tracking, no scene cut.
```

Use the negative prompt to block common failure modes for the subject and motion:

```text
static still image, only arms moving, only camera movement, cropped head, cropped feet, hands out of
frame, back to camera, rear view, turned away, profile-only view, over-rotated body, detached arm,
disconnected arm, detached hand, broken wrist, extra limbs, malformed hands, oversized foot, melted
foot, deformed face, duplicate body, black frames, green frames, low quality, flicker, subject exits
frame, sudden scene cut
```

Prompting reduces but does not eliminate video-model brittleness. Complex human motion can still
degrade around hands, wrists, feet, ankles, self-occluding poses, and out-of-frame re-entry. To
reduce those failures, make the subject smaller in the source frame, leave generous margins above
raised hands and below feet, choose poses whose full motion stays inside the frame, ask for a wide
or camera-follow shot, keep motion restrained near boundaries, constrain subject rotation when face
identity matters, and shorten clips when identity details approach the edge. If the action requires
the subject to leave the frame, turn away fully, or return from an occluded/back-facing pose, split it
into separate clips or use a later keyframe/image input rather than relying on one long single-image
conditioning run. For production checks, use seed sweeps and inspect decoded frames or contact
sheets rather than relying only on MP4 existence.

## Why Do Some Imports Or Paths Still Say `mflux`?

MLX-Gen is built on the mflux codebase. Some internal modules and compatibility entry points still
use `mflux.*` names, while the public package and command surface use `mlx-gen` and `mlxgen`.

## How Does This Relate To AbstractVision?

MLX-Gen is intended to be the Apple Silicon / MLX backend dependency for
[AbstractVision](https://github.com/lpalbou/abstractvision), which sits inside the wider
[AbstractFramework](https://abstractframework.ai/) ecosystem. AbstractVision abstracts generative
image and video capabilities across local and hosted providers, while MLX-Gen owns MLX model
loading, quantized local formats, capability reporting, progress callbacks, and Apple Silicon
runtime behavior.

That split also means higher-level convenience such as curated model-to-adapter pairing, preset
selection, or UI-facing adapter recommendations should live in AbstractVision. MLX-Gen is the
runtime authority: it reports which exact routes work, which options are legal, and which
adapter/base-model combinations fail closed.

[AbstractCore](https://abstractcore.ai/) can expose OpenAI-compatible endpoints backed by
AbstractVision providers. [AbstractFlow](https://github.com/lpalbou/abstractflow) can use those
capabilities in visual workflows alongside other media, text, and agent tasks.
