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

## Can MLX-Gen Packages Load In Diffusers Or Transformers?

No. MLX-Gen packages use the MLX/mflux saved-weight layout and MLX quantization tensors. They are intended for MLX-Gen and compatible mflux code, not direct Diffusers or Transformers `from_pretrained()` loading.

## Can I Use The Official BFL FLUX.2 FP8 Or NVFP4 Repos Directly?

Not today.

Current MLX-Gen `generate` and `prepare` flows support the standard FLUX.2 source repositories and
MLX-Gen model packages. The official `black-forest-labs/FLUX.2-klein-base-*-fp8` and
`black-forest-labs/FLUX.2-klein-base-*-nvfp4` repositories are not direct MLX-Gen inputs in the
current release.

Those repositories use a different transformer packaging format from MLX-Gen model packages, and
the current MLX-Gen loader does not treat them as ready-to-run local models. Use the BF16 source
repository or an MLX-Gen package today.

## Why Can q8 Show The Same Physical Peak As BF16?

Because the visible peak metric may be measuring a different thing than the persistent model
footprint.

`Storage` is the on-disk or Hugging Face repository size. `Wan MLX model` is the loaded Wan
transformer plus VAE tensor footprint. `MLX active after generation` is the MLX allocator memory
still live after `generate_video()` returns. These should drop when a q8 package actually stores
and loads quantized transformer block linears.

`Physical Peak` is a full-process Darwin high-water sample. It includes MLX/Metal allocations, the
PyTorch UMT5 prompt encoder, activation graphs, decoded video buffers, frame conversion, save
validation, Python objects, and native-library transients. `Max RSS` is resident set high-water
memory and can under-report Apple unified-memory/Metal pressure. `MLX Peak` is only the MLX
allocator high-water mark, not the whole process.

So a q8 model can be clearly smaller on disk and in loaded MLX model memory while a specific
generation profile shows a similar full-process physical peak because temporary activations or
decode/save buffers dominate that run. See [Quantization](quantization.md) for the current Wan
tables and definitions.

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

ERNIE LoRA support is experimental and route-specific. `AbstractFramework/ernie-image-turbo-8bit`
now has an exact validated text-to-image LoRA row; ERNIE latent img2img remains
`mapped-unvalidated`. Use `mlxgen capabilities --model AbstractFramework/ernie-image-turbo-8bit`
to inspect the current status before relying on a specific adapter workflow.

## Does Bonsai Image Support LoRA?

No. MLX-Gen currently rejects Bonsai LoRA requests.

The practical blocker is architectural, not just missing validation. Bonsai runs through a packed
ternary/low-bit transformer path, while MLX-Gen's LoRA loader applies adapters by replacing normal
linear modules. That replacement boundary does not exist in the current packed Bonsai runtime, so
MLX-Gen fails closed instead of pretending the adapter applied.

## Does Wan Video Support LoRA?

Yes, for the current Wan q8 public routes. Exact validated rows now exist for:

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
[LoRA](lora.md).

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
`mlxgen validation --model <model>` to inspect the current release evidence for an exact package.

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
texture but can also soften fine details. SeedVR2 keeps small outputs untiled for image quality and
automatically uses tiled VAE decode for large outputs. Use `--vae-tiling` when you also want tiled
VAE encoding or the same tiled path for smaller outputs.

See [Image Upscaling](upscaling.md) for a checked-in 5x SeedVR2 comparison where the original
source is enlarged to the generated output resolution for side-by-side assessment.

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

MLX-Gen supports experimental generative reframe for edit models that advertise
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

Use experimental `--outpaint-padding` when you want MLX-Gen to create a larger canvas and guide a
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
before loading weights. Prepared base FLUX.2 Klein q8/q4 packages also need separate visual proof
before they should be treated as release-validated outpaint packages.

For ordinary image-to-image, the default `source-aspect` canvas policy keeps the output ratio close
to the first source image. That prevents accidental stretching, but it does not expand the original
canvas or preserve source pixels in place. Use `--canvas-policy exact-resize` only for deliberate
whole-image recomposition.

## What Wan Video Resolutions Should I Use?

Wan width and height are normalized to the selected model's VAE/patch multiple. For text-to-video,
MLX-Gen uses the requested canvas after model-multiple normalization. For image-to-video, MLX-Gen
preserves the source image aspect ratio: requested `--width` and `--height` define the approximate
size target, and the runtime resolves the closest supported canvas from the input image ratio before
conditioning the model. The video is generated directly at the resolved canvas; MLX-Gen does not
generate a stretched canvas and then crop or resize it back afterward.

| Model | Required multiple | Recommended/native size | Lower-cost diagnostic sizes |
| --- | ---: | --- | --- |
| TI2V-5B T2V/I2V | 32 px | `1280x704` or `704x1280` | `832x480`, `480x832`; smaller sizes such as `448x256` are smoke checks only |
| T2V-A14B | 16 px | `1280x720` or `720x1280` | `832x480`, `480x832`, `448x256`, `256x448`, `432x240` |
| I2V-A14B | 16 px | `1280x720` or `720x1280` | `832x480`, `480x832`, `448x256`, `256x448`, `432x240` |

For TI2V-5B text-to-video, `1280x720` adjusts to `1280x736`, and `432x240` adjusts to `448x256`.
For A14B text-to-video, `1280x720`, `832x480`, `448x256`, and `432x240` are already valid multiples
of 16. For TI2V-5B, use at least `832x480` for visual prompt checks; smaller canvases are useful for
route checks only. Use the
recommended/native size, frame count, and step count when judging visual quality.

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
aspect ratio.

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

MLX-Gen is currently built on the mflux codebase. Some internal modules and compatibility entry points still use `mflux.*` names while the public package and command surface evolve under `mlx-gen` and `mlxgen`.

## How Does This Relate To AbstractVision?

MLX-Gen is intended to be the Apple Silicon / MLX backend dependency for
[AbstractVision](https://github.com/lpalbou/abstractvision), which sits inside the wider
[AbstractFramework](https://abstractframework.ai/) ecosystem. AbstractVision abstracts generative
image and video capabilities across local and hosted providers, while MLX-Gen owns MLX model
loading, quantized local formats, capability reporting, progress callbacks, and Apple Silicon
runtime behavior.

[AbstractCore](https://abstractcore.ai/) can expose OpenAI-compatible endpoints backed by
AbstractVision providers. [AbstractFlow](https://github.com/lpalbou/abstractflow) can use those
capabilities in visual workflows alongside other media, text, and agent tasks.
