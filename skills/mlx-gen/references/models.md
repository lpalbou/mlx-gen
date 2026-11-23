# Model Families

Open this file to choose a model for a task and a machine, or to look up a family's aliases,
settings, and limits. The live source of truth for one model is always
`mlxgen capabilities --model <id>` (see [capabilities-schema.md](capabilities-schema.md)).

## Choose by task

| Task | Families to consider |
| --- | --- |
| Text-to-image | FLUX.2 Klein, Z-Image, Qwen Image, ERNIE Image Turbo, Bonsai Image, FIBO |
| Whole-image restyle from one image (latent img2img) | FLUX.2 Klein, Z-Image, base Qwen Image, ERNIE Image Turbo |
| Instruction edit that keeps the layout | Qwen Image Edit, FLUX.2 Klein |
| Local masked edit | Qwen Image Edit, base Qwen Image, Z-Image Turbo, FLUX.2 Klein |
| Compose from several reference images | Qwen Image Edit 2509/2511, FLUX.2 Klein |
| Extend the canvas (reframe, outpaint) | Qwen Image Edit, FLUX.2 Klein |
| Text- or image-to-video (silent) | Wan2.2 TI2V-5B, Wan2.2 T2V-A14B / I2V-A14B |
| Prompt-guided video-to-video | Wan2.2 T2V-A14B, Wan2.1 VACE 1.3B, Bernini-R (experimental) |
| Video from reference images | Bernini-R (experimental), Wan2.1 VACE 1.3B |
| Video with a generated soundtrack | MiniMax-H3 |
| Restore or upscale an image | SeedVR2 |
| Restore or upscale a video | SeedVR2 (scales), SwiftVR (source resolution, much faster) |

## Choose by memory

Conservative starting points by unified memory, from the published measurements. Larger canvases,
more steps, longer videos, and heavier edit modes need more than the measured profile.

| Memory | Images | Edits | Video | Restoration |
| --- | --- | --- | --- | --- |
| 18 GB | `prism-ml/bonsai-image-ternary-4B-mlx-2bit`, `AbstractFramework/flux.2-klein-4b-8bit`, `AbstractFramework/ernie-image-turbo-8bit` | ERNIE 8-bit for latent restyle, Klein 4B 8-bit for lighter edits; avoid Qwen edit and control routes | No production recommendation (Bernini-R fits the tested low-RAM shapes but is experimental) | SeedVR2 3B/7B image upscaling |
| 24 GB | add `AbstractFramework/fibo-8bit`, `AbstractFramework/z-image-turbo-8bit`, `AbstractFramework/qwen-image-2512-8bit` (text only) | Z-Image Turbo 8-bit latent and inpaint; Klein 4B 8-bit stays the lighter edit default | No conservative Wan A14B recommendation | SeedVR2 image upscaling; video restore needs more headroom |
| 32 GB | same, with more headroom | same; Klein 4B and Z-Image Turbo are safer than Qwen Edit | Wan A14B is above this tier | `AbstractFramework/seedvr2-3b-8bit` five-second video restore is in range |
| 64 GB | all image families | add `AbstractFramework/qwen-image-edit-2511-8bit`, `AbstractFramework/qwen-image-8bit` control and control-inpaint | First conservative tier for Wan A14B; MiniMax-H3 with `AbstractFramework/minimax-h3-4bit` and `--low-ram` | SeedVR2 7B image upscaling; keep SeedVR2 3B for video restore |
| 128 GB or more | everything | everything | add Wan TI2V-5B 8-bit high-resolution profiles and `AbstractFramework/minimax-h3-8bit` (fits from 96 GB) | add SeedVR2 7B five-second video restore |

Measured peaks behind that table:

| Route | Model | Profile | Peak |
| --- | --- | --- | --- |
| Text-to-image | `prism-ml/bonsai-image-ternary-4B-mlx-2bit` | 512px, 4 steps | 3.57 GiB RSS |
| Text-to-image | `AbstractFramework/flux.2-klein-4b-8bit` | 512px, 4 steps | 9.23 GiB RSS |
| Text-to-image, latent | `AbstractFramework/ernie-image-turbo-8bit` | 512px, 8 steps | 12.9 GiB RSS |
| Text-to-image | `AbstractFramework/fibo-8bit` / `fibo-4bit` | 512px, 8 steps | 15.89 GB / 11.39 GB RSS |
| Text-to-image | `AbstractFramework/qwen-image-2512-8bit` | 768x336, 4 steps | 10.73 GiB RSS |
| Latent / inpaint | `AbstractFramework/z-image-turbo-8bit` | 768x432, 9 steps | 11.49 GiB / 10.57 GiB RSS |
| Edit | `AbstractFramework/qwen-image-edit-2511-8bit` | 768x432, 4 steps | 30.91 GiB RSS |
| Control-inpaint | `AbstractFramework/qwen-image-8bit` | 768x432, 4 steps | 34.94 GiB RSS |
| Image upscale | `AbstractFramework/seedvr2-3b-8bit` / `seedvr2-7b-8bit` | 5x from 133x113 | 4.73 GiB / 8.90 GiB RSS |
| Video restore | `ByteDance-Seed/SeedVR2-3B` / `SeedVR2-7B` | five seconds, 149 frames | 27.40 GB / 66.18 GB RSS |
| Text-to-video | `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` | 384x224, 33 frames, 12 steps, low-RAM | 33.0 GiB physical |
| Image-to-video | `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` | 384x384, 33 frames, 12 steps, low-RAM | 33.7 GiB physical |
| Text/image-to-video | `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` | 1280x704, 17 frames, 20 steps | 103.7 GiB physical |
| Reference-to-video | Bernini-R 1.3B BF16 | 320x192, 17 frames, 20 steps, low-RAM | 9.12 GB physical |

Not covered by a published memory benchmark: FLUX.2 Klein 9B, Qwen 4-bit edit packages, and other
Z-Image combinations. Measure a small profile first.

## Image families

`<alias>` is what `--model` accepts for the source repository; published packages are listed in
[packages.md](packages.md). Route ids are the `id` values of capability rows.

### FLUX.2 Klein (`flux2`)

- Aliases: `flux2-klein-4b`, `flux2-klein-9b` (distilled); `flux2-klein-base-4b`,
  `flux2-klein-base-9b` (base). The 9B models inherit gated, non-commercial source terms.
- Routes: `flux2.text`, `flux2.latent`, `flux2.edit`, `flux2.inpaint`, `flux2.outpaint`,
  `flux2.multi-reference`; `flux2.reframe` on distilled weights only.
- Distilled weights are step-distilled: guidance `1.0`, no negative prompt (the option is rejected
  before loading). Base weights run classifier-free guidance: guidance `4.0` by default, negative
  prompt accepted with guidance above `1.0`.
- One `--image` selects edit-reference; add `--image-strength` or `--i2i-mode latent` for latent
  img2img; two or more `--image` select multi-reference.
- Documented settings: distilled 4B at 512px and 4 steps for a fast image, 16 steps for reframe
  and outpaint; base models at 20 steps and guidance 4 for outpaint.
- FLUX.2-dev is not supported, and FLUX.2-dev LoRAs are rejected on Klein routes.

### Qwen Image and Qwen Image Edit (`qwen`)

- Aliases: `qwen-image` (text-to-image, latent, native masked edit); `qwen-image-edit`
  (single-reference edit); `qwen-image-edit-2509`, `qwen-image-edit-2511` (edit plus
  multi-reference). Qwen Image 2512 is addressed by its package id.
- Routes: base models `qwen.text`, `qwen.latent`, `qwen.base-inpaint`; the exact
  `AbstractFramework/qwen-image-8bit` row adds `qwen.control` and `qwen.control-inpaint` with an
  automatically injected ControlNet sidecar. Edit models: `qwen.edit`, `qwen.inpaint`,
  `qwen.reframe`, `qwen.outpaint`, and `qwen.multi-reference` on 2509/2511.
- Negative prompts are supported; with guidance above `1`, edit models apply the official blank
  negative prompt when you pass none.
- Documented settings: edit at 20 to 30 steps (guidance 4 with a negative prompt); 4 steps with
  guidance 1 only together with the matching Lightning LoRA ([lora.md](lora.md)).
- Edit and control routes are the heaviest image routes: 64 GB is the first comfortable tier.

### Z-Image (`z-image`)

- Aliases: `z-image`, `z-image-turbo`.
- Routes: `z-image.text`, `z-image.latent`; `z-image.inpaint` on Turbo only (masked requests on
  non-Turbo are rejected).
- Turbo takes a negative prompt but no guidance scale. Documented Turbo setting: 9 steps.
- Latent img2img needs an explicit `--image-strength`.

### ERNIE Image Turbo (`ernie-image`)

- Alias: `ernie-image-turbo`.
- Routes: `ernie-image.text`, `ernie-image.latent` (exactly one image; no edit or multi-reference).
- Documented settings: 8 steps, guidance 1 for text-to-image at 512x512; use 384px or larger
  (smaller outputs crop subjects). Latent img2img: guidance 3, `--image-strength 0.25` to `0.35`
  for visible stylization, 12 to 16 steps for more polish.
- `--use-prompt-enhancer` needs the full snapshot: `mlxgen download --model baidu/ERNIE-Image-Turbo --all-files`.
  Prepared packages do not include the enhancer.

### Bonsai Image (`bonsai`)

- Id: `prism-ml/bonsai-image-ternary-4B-mlx-2bit` (alias `bonsai`), an already packed 2-bit
  checkpoint: `download` it, never `prepare` it, and omit `--quantize`.
- Route: `bonsai.text` only. Image input, negative prompts, and LoRA are rejected.
- Documented settings: 1024x1024, 4 steps, guidance 1. The binary 1-bit checkpoint is not runnable.

### FIBO (`fibo`)

- Alias: `fibo` (gated source). Route: `fibo.text` only; no LoRA.
- FIBO Edit (`briaai/Fibo-Edit`) reports no unified route: `capabilities` returns empty arrays.

## Default step counts when `--steps` is omitted

| Alias | Steps | Alias | Steps |
| --- | ---: | --- | ---: |
| `flux2-klein-4b`, `flux2-klein-9b` | 4 | `z-image-turbo` | 9 |
| `flux2-klein-base-4b`, `flux2-klein-base-9b` | 50 | `z-image` | 50 |
| `qwen-image`, `qwen-image-edit` | 20 | `ernie-image-turbo` | 8 |
| `qwen-image-edit-2509`, `qwen-image-edit-2511` | 40 | `bonsai-image-ternary` | 4 |
| `fibo` | 50 | `wan2.2-ti2v-5b` | 50 |

Pass `--steps` explicitly in scripts so the cost is visible. Video defaults are in
[video.md](video.md) and [minimax-h3.md](minimax-h3.md).

## Video and restoration families

| Family | Ids | Details |
| --- | --- | --- |
| Wan2.2 | `wan2.2-ti2v-5b`, `wan2.2-t2v-a14b`, `wan2.2-i2v-a14b` | [video.md](video.md) |
| Wan2.1 VACE 1.3B | `wan-vace` | [video.md](video.md) |
| Bernini-R 1.3B (experimental) | `bernini-r-1.3b` | [video.md](video.md) |
| MiniMax-H3 | `minimax-h3`, `minimax-h3-turbo`, `minimax-h3-turbo-544p` | [minimax-h3.md](minimax-h3.md) |
| SeedVR2 | `seedvr2-3b` (alias `seedvr2`), `seedvr2-7b`, `seedvr2-7b-sharp` | [restoration.md](restoration.md) |
| SwiftVR | `swiftvr` (alias `swiftvr-5b`) | [restoration.md](restoration.md) |

Source docs: <https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/recommendations.md>,
<https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/api.md>.
