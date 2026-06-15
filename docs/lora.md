# LoRA

LoRA support in MLX-Gen is experimental. MLX-Gen accepts LoRA adapters only when the selected route
can apply them to the model transformer. A requested LoRA is required input: missing files,
unreadable files, incompatible matrix shapes, zero matched keys, and unsupported model families fail
before or during model setup instead of continuing without the adapter.

## Check Support First

Use `mlxgen capabilities` before starting a LoRA run:

```sh
mlxgen capabilities --model AbstractFramework/flux.2-klein-4b-8bit
```

Each capability row includes:

| Field | Meaning |
| --- | --- |
| `supports_lora` | Whether the route accepts LoRA arguments. |
| `lora_status` | `unsupported`, `mapped-unvalidated`, or `validated`. |
| `lora_target_roles` | Model components targeted by adapters, such as `transformer`. |
| `lora_validation_profile` | Validation profile id when the exact route has model-backed proof. |

`mapped-unvalidated` means MLX-Gen has a loader and mapping for the route, but that exact
model/package and task has not yet passed a visible A/B validation with an accepted adapter. Treat
LoRA routes as experimental unless a current A/B contact sheet demonstrates the intended adapter
effect for your selected model/package.

Generated output metadata now records what actually applied, not only what was requested:
`lora_application_reports`, `lora_applied_file_count`, and `lora_applied_target_count`.

When a capability row reports `lora_validation_profile`, you can inspect the accepted proof row
directly:

```sh
mlxgen validation \
  --model AbstractFramework/qwen-image-edit-8bit \
  --profile lora_qwen_edit_q8_ghibli_edit_2026_06_11
```

## Current Support Snapshot

The current LoRA surface is route-specific:

| Route family | Current status |
| --- | --- |
| `AbstractFramework/qwen-image-edit-2511-8bit`, `AbstractFramework/qwen-image-edit-2509-8bit`, `AbstractFramework/qwen-image-edit-8bit`, `AbstractFramework/qwen-image-2512-8bit`, `AbstractFramework/qwen-image-8bit` on `qwen.control`, `AbstractFramework/z-image-turbo-8bit`, `AbstractFramework/flux.2-klein-9b-8bit` edit, `AbstractFramework/ernie-image-turbo-8bit` text-to-image | Exact validated q8 proof rows exist. `AbstractFramework/qwen-image-edit-2511-8bit` currently has accepted proof rows on both `qwen.edit` and `qwen.inpaint`. `AbstractFramework/qwen-image-8bit` currently has an accepted proof row on `qwen.control` only. |
| Base Qwen Image text generation, Qwen multi-reference or canvas rows, Z-Image latent img2img, ERNIE latent img2img, and the remaining FLUX.2 package rows | `mapped-unvalidated`: the mapping works, but the exact route still lacks a strong public A/B proof. |
| `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` text-to-video, `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` first-frame image-to-video, `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` text-to-video, and `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` first-frame image-to-video | Exact validated q8 proof rows exist. |
| SeedVR2, FIBO | Unsupported today. |
| Bonsai | Unsupported and low priority. The current packed runtime does not expose the ordinary replaceable linear-module boundary that MLX-Gen's LoRA loader requires. |

## Download And Reference Adapters

Generation does not download LoRA files. Download the adapter repository explicitly:

```sh
mlxgen download --model lovis93/Flux-2-Multi-Angles-LoRA-v2 --all-files
```

Use a local `.safetensors` path or a Hugging Face repository id. If the repository contains several
adapter files, specify the file after a colon. The file part can include a subdirectory inside the
repository:

```sh
mlxgen generate \
  --model <compatible-model> \
  --prompt "<prompt from the adapter model card>" \
  --lora-paths owner/repo:adapter.safetensors \
  --lora-scales 0.9 \
  --output with_lora.png
```

The number of `--lora-scales` values must match the number of `--lora-paths` values. Passing scales
without paths fails before model load.

## Adapter Compatibility

Read the adapter model card and match its base model. A LoRA trained for one FLUX.2 variant is not
automatically compatible with another FLUX.2 variant.

For the LightX2V Qwen Lightning adapters, keep one distinction clear:

- MLX-Gen's validated public routes use the published `AbstractFramework/*-8bit` q8 packages.
- The upstream LightX2V note about BF16-versus-FP8 Lightning compatibility applies to external
  FP8 Qwen checkpoints, not to MLX-Gen's q8 packages.

In practice, MLX-Gen's current recommendation is:

1. use the validated q8 MLX-Gen packages for Qwen image and Wan video routes when you want the
   optimized public path;
2. use the exact documented Lightning adapter for that route;
3. do not assume that an arbitrary external FP8 checkpoint behaves the same way as an MLX-Gen q8
   package.

For the upstream FP8 caveat, see the LightX2V Qwen Lightning README:

- <https://github.com/ModelTC/LightX2V-Qwen-Image-Lightning#-using-lightning-loras-with-fp8-models>

The downloaded `lovis93/Flux-2-Multi-Angles-LoRA-v2` adapter targets
`black-forest-labs/FLUX.2-dev`, uses prompts that start with `<sks>`, and recommends adapter
strength around `0.8` to `1.0`. MLX-Gen currently supports FLUX.2 Klein 4B/9B, not
`black-forest-labs/FLUX.2-dev`. Passing this adapter to FLUX.2 Klein is rejected because the LoRA
matrices target a different transformer width.

Wan video LoRA is now available on the Wan routes that expose transformer LoRA targets. TI2V-5B
uses one role, `transformer`. Wan A14B uses two explicit roles, `high_noise_transformer` and
`low_noise_transformer`. MLX-Gen does not guess or silently duplicate roles for dual-transformer
A14B requests: callers must pass the intended role assignment explicitly with `--lora-target-roles`.

MLX-Gen accepts the main public Wan LoRA naming conventions, including Diffusers-style Wan
adapters and the common alternative Wan adapter layouts used in the community. The practical
question is not basic file loading anymore; it is whether a given adapter produces a strong enough
visible MP4 A/B on the exact route you want to use.

The downloaded `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` adapter targets
`Qwen/Qwen-Image-Edit-2511` and uses `<sks>` multi-angle prompt wording. MLX-Gen validates the
adapter against `AbstractFramework/qwen-image-edit-2511-8bit` through the public `mlxgen generate`
route. On the spaceship source below, base Qwen 2511 already follows many viewpoint prompts, so the
LoRA effect is visible but modest.

![Qwen Image Edit 2511 q8 multi-angle LoRA A/B](assets/validation/lora-2026-06-08/qwen2511-q8-multi-angle-lora-ab-contact-sheet.png)

The first pair used:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Use the source spaceship as the same object. <sks> back view low-angle shot wide shot. Re-render the scene from behind the spaceship at a low camera angle, keeping the icy canyon and the same vehicle design. No text, no watermark, no blur." \
  --negative "front view, same camera angle, cropped spaceship, text, watermark, blur, duplicate spaceship" \
  --width 432 \
  --height 240 \
  --steps 24 \
  --guidance 4 \
  --seed 9701 \
  --metadata \
  --replace \
  --output validation_outputs/lora_multi_angle_2026_06_08/qwen2511_q8_no_lora_back_low_wide.png \
  --i2i-mode edit

mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Use the source spaceship as the same object. <sks> back view low-angle shot wide shot. Re-render the scene from behind the spaceship at a low camera angle, keeping the icy canyon and the same vehicle design. No text, no watermark, no blur." \
  --negative "front view, same camera angle, cropped spaceship, text, watermark, blur, duplicate spaceship" \
  --width 432 \
  --height 240 \
  --steps 24 \
  --guidance 4 \
  --seed 9701 \
  --metadata \
  --replace \
  --output validation_outputs/lora_multi_angle_2026_06_08/qwen2511_q8_with_lora_back_low_wide.png \
  --i2i-mode edit \
  --lora-paths fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:qwen-image-edit-2511-multiple-angles-lora.safetensors \
  --lora-scales 0.9
```

The second pair used the same settings with `--prompt "<sks> front view high-angle shot close-up"`,
`--seed 9702`, and matching `no_lora_front_high_close.png` / `with_lora_front_high_close.png`
outputs.

`AbstractFramework/qwen-image-edit-2509-8bit` now has an exact single-image edit proof with the
stacked `lightx2v/Qwen-Image-Lightning` plus `dx8152/Qwen-Edit-2509-Multiple-angles` path. This
row is validated for `qwen.edit` only. The validated profile uses the Lightning-style settings from
the public workflow: `8` steps and `guidance 1`.

![Qwen Image Edit 2509 q8 multi-angle LoRA A/B](assets/validation/lora-2026-06-11/qwen2509_q8_multi_angle_ab_contact_sheet.png)

Commands:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2509-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Move the camera to the right. Rotate the camera 45 degrees to the right. Turn the camera to a wide-angle shot. Keep the same spaceship design, the icy canyon, the rear engines, and the wide scene composition. No text, no watermark, no blur." \
  --width 432 \
  --height 240 \
  --steps 8 \
  --guidance 1 \
  --seed 9901 \
  --metadata \
  --replace \
  --output validation_outputs/lora_strict_2026_06_11/qwen2509_q8_no_lora_angle_g1.png \
  --i2i-mode edit

mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2509-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Move the camera to the right. Rotate the camera 45 degrees to the right. Turn the camera to a wide-angle shot. Keep the same spaceship design, the icy canyon, the rear engines, and the wide scene composition. No text, no watermark, no blur." \
  --width 432 \
  --height 240 \
  --steps 8 \
  --guidance 1 \
  --seed 9901 \
  --metadata \
  --replace \
  --output validation_outputs/lora_strict_2026_06_11/qwen2509_q8_with_lora_angle_g1.png \
  --i2i-mode edit \
  --lora-paths lightx2v/Qwen-Image-Lightning:Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors dx8152/Qwen-Edit-2509-Multiple-angles:镜头转换.safetensors \
  --lora-scales 1.0 0.9
```

`AbstractFramework/qwen-image-edit-8bit` now has an exact single-image edit proof with a
Ghibli-style Qwen adapter. This row is validated for `qwen.edit` only. The current proof uses
`ghibli_style_qwen_v3.safetensors` on same-seed edit trials and produces a visible style shift
while keeping the edit route stable.

![Qwen Image Edit q8 Ghibli-style LoRA A/B](assets/validation/lora-2026-06-11/qwen_edit_q8_ghibli_trials_contact_sheet.png)

Representative command:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "ghibli style. Transform the source into a whimsical hand-painted animated film frame with soft brushwork, warm pastel sky, painterly snow, and gentle storybook lighting. Preserve the same spaceship, snowy canyon, wide framing, and overall layout." \
  --width 432 \
  --height 240 \
  --steps 24 \
  --guidance 4 \
  --seed 9951 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_lora_2026_06_11/qwen_edit_q8_ghibli_with_lora.png \
  --i2i-mode edit \
  --lora-paths /path/to/ghibli_style_qwen_v3.safetensors \
  --lora-scales 1.0
```

`AbstractFramework/qwen-image-2512-8bit` now has an exact text-to-image proof with
`prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA`. This row is validated for `qwen.text` only; the
latent img2img row remains `mapped-unvalidated`.

![Qwen Image 2512 q8 pixel art LoRA A/B](assets/validation/lora-2026-06-11/qwen2512_q8_pixel_art_ab_contact_sheet.png)

Commands:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-2512-8bit \
  --prompt "Pixel Art, a pixelated image of a space astronaut floating in zero gravity. The astronaut wears a white spacesuit with orange stripes. Earth appears in the background with blue oceans and white clouds, rendered in classic 8-bit style." \
  --negative " " \
  --width 640 \
  --height 640 \
  --steps 45 \
  --guidance 5 \
  --seed 9941 \
  --metadata \
  --replace \
  --output validation_outputs/lora_strict_2026_06_11/qwen2512_q8_no_lora_pixel_art.png

mlxgen generate \
  --model AbstractFramework/qwen-image-2512-8bit \
  --prompt "Pixel Art, a pixelated image of a space astronaut floating in zero gravity. The astronaut wears a white spacesuit with orange stripes. Earth appears in the background with blue oceans and white clouds, rendered in classic 8-bit style." \
  --negative " " \
  --width 640 \
  --height 640 \
  --steps 45 \
  --guidance 5 \
  --seed 9941 \
  --metadata \
  --replace \
  --output validation_outputs/lora_strict_2026_06_11/qwen2512_q8_with_lora_pixel_art.png \
  --lora-paths prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA:Qwen-Image-2512-Master-Pixel-Art-LoRA.safetensors \
  --lora-scales 1.0
```

For Qwen Image 2512 and Qwen Image Edit 2511, the dedicated LightX2V Lightning adapters are the
recommended fast path when you want usable results in `4` denoising steps. In practical terms,
they let you use a `4`-step workflow instead of the more typical `20`-step Qwen workflow. The
commands below assume the selected q8 package is already cached locally; the extra download is only
the LoRA repository. Those recommendations are for the validated MLX-Gen q8 packages, not for an
arbitrary external FP8 Qwen checkpoint.

![Qwen Image 2512 q8 4-step Lightning A/B](assets/validation/qwen-lightning-2026-06-15/qwen2512_q8_lightning_ab_contact_sheet.png)

The documented `AbstractFramework/qwen-image-2512-8bit` proof uses the same prompt and seed for a
source/no-LoRA/with-LoRA comparison and shows the Lightning route producing the intended image on
the `4`-step fast path.

Download the adapter repo explicitly:

```sh
mlxgen download --model lightx2v/Qwen-Image-2512-Lightning --all-files
```

Command:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-2512-8bit \
  --prompt "Cinematic wide-angle photo of a silver retro-futuristic spaceship parked on a frozen runway inside an icy canyon at sunrise, twin blue engines glowing, drifting snow, crisp metallic panel lines, photorealistic, highly detailed." \
  --negative "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, duplicate, watermark, signature, text, cartoon, anime, painting, illustration, 3d render, cgi" \
  --width 768 \
  --height 341 \
  --steps 4 \
  --guidance 1 \
  --seed 4212 \
  --metadata \
  --replace \
  --output qwen2512_lightning.png \
  --lora-paths lightx2v/Qwen-Image-2512-Lightning:Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors \
  --lora-scales 1
```

`AbstractFramework/qwen-image-edit-2511-8bit` also accepts the dedicated `4`-step Lightning edit
adapter and is a recommended way to run fast single-image Qwen edits.

![Qwen Image Edit 2511 q8 4-step Lightning A/B](assets/validation/qwen-lightning-2026-06-15/qwen2511edit_q8_lightning_ab_contact_sheet.png)

The documented proof uses the bundled spaceship source image plus a same-seed no-LoRA reference and
shows the Lightning route producing a usable `4`-step edit result.

Download the adapter repo explicitly:

```sh
mlxgen download --model lightx2v/Qwen-Image-Edit-2511-Lightning --all-files
```

Command:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Keep the same silver spaceship, icy canyon, and sunrise palette. Change the camera to a front three-quarter close shot, brighten the blue engines, add blowing snow around the landing gear, and preserve the same realistic metallic design." \
  --negative "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, duplicate, watermark, signature, text, cartoon, anime, painting, illustration, 3d render, cgi" \
  --width 768 \
  --height 432 \
  --steps 4 \
  --guidance 1 \
  --seed 5114 \
  --metadata \
  --replace \
  --output qwen2511edit_lightning.png \
  --lora-paths lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
  --lora-scales 1
```

The base `AbstractFramework/qwen-image-8bit` q8 package also has an exact validated structured-control
row when you pair the InstantX union ControlNet sidecar with the shared Qwen Lightning adapter.
This is the current recommended fast public path for Qwen structured control in `4` steps.

![Qwen Image q8 structured control Lightning proof](assets/validation/qwen-control-2026-06-15/qwen_q8_control_lightning_contact_sheet.png)

Download the two exact repositories:

```sh
hf download InstantX/Qwen-Image-ControlNet-Union \
  diffusion_pytorch_model.safetensors \
  config.json \
  conds/canny.png \
  conds/pose.png \
  README.md

mlxgen download --model lightx2v/Qwen-Image-Lightning --all-files
```

Structured-control example:

```sh
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
  --output qwen_controlled.png
```

This exact route is `qwen.control`, not base `qwen.text`, and it is a structured text-to-image
workflow rather than a source-image edit. The accepted proof compares same-prompt, same-seed
Lightning runs with and without the control image and shows the control image materially changing
layout.

Published proof artifacts:

- [structured-control command log](assets/validation/qwen-control-2026-06-15/qwen_q8_control_lightning_command_log.md)
- [structured-control timings on M5 Max](assets/validation/qwen-control-2026-06-15/qwen_q8_control_lightning_stats_m5max.json)

The same adapter is also the current recommended fast path for masked edit / inpaint on
`AbstractFramework/qwen-image-edit-2511-8bit`. The accepted proof keeps the unmasked scene stable
while changing only the masked region in two different conditions.

![Qwen Image Edit 2511 q8 masked edit Lightning proof](assets/validation/qwen-inpaint-2026-06-15/qwen2511_q8_inpaint_lightning_contact_sheet.png)

The proof uses one engine-enhancement mask and one crash-repair mask. In practical terms, Lightning
makes a `4`-step masked edit workflow usable on the validated q8 route, which is why it is the
recommended public path when you want fast Qwen inpaint runs.

To show that the mask is actually doing the localization work, MLX-Gen also publishes a same-canvas
control sheet with the Lightning adapter in both result columns. Those runs use the same
`768x432` source image, prompt, seed, and adapter. The only difference is `--mask-path`. Without
`--mask-path`, the edit recomposes the full image; with `--mask-path`, the edit stays local.

![Qwen Image Edit 2511 q8 masked edit control](assets/validation/qwen-inpaint-2026-06-15/qwen2511_q8_inpaint_mask_control_contact_sheet.png)

Download the same adapter repo:

```sh
mlxgen download --model lightx2v/Qwen-Image-Edit-2511-Lightning --all-files
```

Masked edit example:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --mask-path mask.png \
  --prompt "Keep the same silver spaceship, icy canyon, and sunrise lighting. Only inside the masked engine area, intensify both blue engines into brighter plasma thrusters, add dense blue glow and snow vapor around the thrusters, and preserve the rest of the image unchanged." \
  --negative "blurry, low quality, distorted, deformed, extra ship parts, changed camera angle, changed background, text, watermark" \
  --width 768 \
  --height 432 \
  --steps 4 \
  --guidance 1 \
  --seed 4201 \
  --output qwen2511edit_inpaint_lightning.png \
  --lora-paths lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
  --lora-scales 1
```

White mask pixels are repainted and black pixels are preserved. The exact validated command log and
timings for the two proof conditions are published here:

- [masked edit command log](assets/validation/qwen-inpaint-2026-06-15/qwen2511_q8_inpaint_lightning_command_log.md)
- [masked edit timings on M5 Max](assets/validation/qwen-inpaint-2026-06-15/qwen2511_q8_inpaint_lightning_stats_m5max.json)

`AbstractFramework/z-image-turbo-8bit` now has an exact text-to-image proof with
`renderartist/Technically-Color-Z-Image-Turbo`. This row is validated for `z-image.text` only; the
latent img2img row remains `mapped-unvalidated`.

![Z-Image Turbo q8 Technically Color LoRA A/B](assets/validation/lora-2026-06-11/zimage_q8_technically_color_ab_contact_sheet.png)

Commands:

```sh
mlxgen generate \
  --model AbstractFramework/z-image-turbo-8bit \
  --prompt "t3chnic4lly vibrant 1960s close-up of a woman sitting under a tree in a blue skirt and white blouse, she has blonde wavy short hair and a smile with green eyes lake scene by a garden with flowers in the foreground 1960s style film She's holding her hand out there is a small smooth frog in her palm, she's making eye contact with the toad." \
  --negative "JPEG Artifacts, compression, noisy, grainy, low quality, amateur" \
  --width 640 \
  --height 368 \
  --steps 9 \
  --seed 42 \
  --metadata \
  --replace \
  --output validation_outputs/lora_strict_2026_06_11/zimage_q8_no_lora.png

mlxgen generate \
  --model AbstractFramework/z-image-turbo-8bit \
  --prompt "t3chnic4lly vibrant 1960s close-up of a woman sitting under a tree in a blue skirt and white blouse, she has blonde wavy short hair and a smile with green eyes lake scene by a garden with flowers in the foreground 1960s style film She's holding her hand out there is a small smooth frog in her palm, she's making eye contact with the toad." \
  --negative "JPEG Artifacts, compression, noisy, grainy, low quality, amateur" \
  --width 640 \
  --height 368 \
  --steps 9 \
  --seed 42 \
  --metadata \
  --replace \
  --output validation_outputs/lora_strict_2026_06_11/zimage_q8_with_lora.png \
  --lora-paths renderartist/Technically-Color-Z-Image-Turbo:Technically_Color_Z_Image_Turbo_v1_renderartist_2000.safetensors \
  --lora-scales 0.5
```

`AbstractFramework/flux.2-klein-9b-8bit` now has an exact single-image edit proof with
`dx8152/Flux2-Klein-9B-Consistency`. This row is validated for `flux2.edit` only; multi-reference
and reframe/outpaint rows remain `mapped-unvalidated`.

![FLUX.2 Klein 9B q8 consistency LoRA A/B](assets/validation/lora-2026-06-11/flux2_klein9b_q8_consistency_ab_contact_sheet.png)

Commands:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Edit the source into the same spaceship after a hard landing in the snow at blue hour. Preserve the same spaceship design, hull proportions, cockpit shape, engine placement, snowy canyon layout, and wide camera angle. Add disturbed snow, bent landing struts, a shallow scrape trail, broken ice chunks, and a thin smoke plume. Keep the ship solid, sharp, and consistent." \
  --width 432 \
  --height 240 \
  --steps 20 \
  --guidance 1 \
  --seed 9801 \
  --metadata \
  --replace \
  --output validation_outputs/lora_strict_2026_06_11/flux2_klein9b_q8_no_lora_edit.png

mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Edit the source into the same spaceship after a hard landing in the snow at blue hour. Preserve the same spaceship design, hull proportions, cockpit shape, engine placement, snowy canyon layout, and wide camera angle. Add disturbed snow, bent landing struts, a shallow scrape trail, broken ice chunks, and a thin smoke plume. Keep the ship solid, sharp, and consistent." \
  --width 432 \
  --height 240 \
  --steps 20 \
  --guidance 1 \
  --seed 9801 \
  --metadata \
  --replace \
  --output validation_outputs/lora_strict_2026_06_11/flux2_klein9b_q8_with_lora_edit.png \
  --lora-paths dx8152/Flux2-Klein-9B-Consistency:Flux2-Klein-9B-consistency-V2.safetensors \
  --lora-scales 0.8
```

On this exact spaceship edit run, the with-LoRA output stayed materially closer to the source ship
layout than the no-LoRA output while still honoring the crash prompt.

`AbstractFramework/ernie-image-turbo-8bit` now has an exact text-to-image proof with
`reverentelusarca/ernie-image-elusarca-anime-style-lora`. This row is validated for
`ernie-image.text` only; the latent img2img row remains `mapped-unvalidated`.

![ERNIE Image Turbo q8 anime-style LoRA A/B](assets/validation/lora-2026-06-11/ernie_turbo_q8_anime_style_ab_contact_sheet.png)

Commands:

```sh
mlxgen generate \
  --model AbstractFramework/ernie-image-turbo-8bit \
  --prompt "elusarca anime style, a young woman with silver hair and a red trench coat standing beneath glowing lanterns in a rain-soaked alley at night, confident pose, detailed face, dramatic lighting" \
  --negative "blurry, deformed face, extra limbs, text, watermark" \
  --width 512 \
  --height 512 \
  --steps 8 \
  --guidance 1 \
  --seed 9961 \
  --metadata \
  --replace \
  --output validation_outputs/lora_strict_2026_06_11/ernie_turbo_q8_no_lora_anime.png

mlxgen generate \
  --model AbstractFramework/ernie-image-turbo-8bit \
  --prompt "elusarca anime style, a young woman with silver hair and a red trench coat standing beneath glowing lanterns in a rain-soaked alley at night, confident pose, detailed face, dramatic lighting" \
  --negative "blurry, deformed face, extra limbs, text, watermark" \
  --width 512 \
  --height 512 \
  --steps 8 \
  --guidance 1 \
  --seed 9961 \
  --metadata \
  --replace \
  --output validation_outputs/lora_strict_2026_06_11/ernie_turbo_q8_with_lora_anime.png \
  --lora-paths reverentelusarca/ernie-image-elusarca-anime-style-lora:ernie-anime-v1.safetensors \
  --lora-scales 0.9
```

This adapter produces a visibly stronger anime render while keeping the same prompt, seed, and
subject setup.

Current Wan q8 public rows now have exact route proofs:

- `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` on `wan.text-video`
- `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` on `wan.first-frame`
- `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` on `wan.text-video`
- `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` on `wan.first-frame`

![Wan TI2V-5B q8 HSToric LoRA A/B](assets/validation/wan-lora-2026-06-11/ti2v_t2v_hstoric_ab_contact_sheet.jpg)
![Wan TI2V-5B q8 Crush-It I2V LoRA A/B](assets/validation/wan-lora-2026-06-11/ti2v_i2v_crushit_ab_contact_sheet.jpg)
![Wan T2V-A14B q8 LightX2V 4-Step A/B](assets/validation/lightx2v-wan-4step-2026-06-12/a14b_t2v_lightx2v_4step_ab_contact_sheet.jpg)
![Wan I2V-A14B q8 LightX2V 4-Step A/B](assets/validation/lightx2v-wan-4step-2026-06-12/a14b_i2v_lightx2v_4step_ab_contact_sheet.jpg)

Representative commands:

```sh
mlxgen download --model AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets --all-files

mlxgen generate \
  --model AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit \
  --prompt "HST style HD film, early 1900s, autochrome, analog cinema. A horse-drawn carriage crossing a snowy town square at dusk, pedestrians in wool coats, historical street lamps glowing, gentle cinematic motion." \
  --width 832 \
  --height 480 \
  --frames 17 \
  --steps 20 \
  --guidance 4 \
  --fps 16 \
  --seed 6301 \
  --metadata \
  --output validation_outputs/wan_lora_2026_06_11/ti2v_t2v_hstoric_with_lora_q8.mp4 \
  --lora-paths AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets:HSToric_color_Wan22_5b_LoRA.safetensors \
  --lora-target-roles transformer \
  --lora-scales 0.8
```

```sh
mlxgen download --model ostris/wan22_5b_i2v_crush_it_lora --all-files

mlxgen generate \
  --model AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit \
  --image validation_outputs/wan_lora_2026_06_11/ti2v_i2v_can_source_qwen2512_q8.png \
  --prompt "crush it. An invisible hydraulic press crushes the centered aluminum soda can flat on the clean studio floor while the camera stays stable, with product-video lighting and realistic reflections." \
  --width 832 \
  --height 480 \
  --frames 41 \
  --steps 20 \
  --guidance 4 \
  --fps 20 \
  --seed 6603 \
  --metadata \
  --replace \
  --output validation_outputs/wan_lora_2026_06_11/ti2v_i2v_crushit_q8_with_lora.mp4 \
  --lora-paths ostris/wan22_5b_i2v_crush_it_lora:wan22_5b_i2v_crush_it_lora.safetensors \
  --lora-target-roles transformer \
  --lora-scales 1
```

For Wan A14B, the current recommended fast path is the official `lightx2v/Wan2.2-Lightning`
paired 4-step recipe. The accepted proof is same-seed `4`-step no-LoRA versus same-seed `4`-step
with the paired Lightning files, using:

- `steps=4`
- `flow_shift=5.0`
- `guidance=1.0`
- `guidance_2=1.0`
- explicit `high_noise_transformer` and `low_noise_transformer` roles

That accepted A/B proof is a **LoRA-effect proof**, not a fair quality comparison against the
normal longer Wan profile. The point of the same-step `4`-step no-LoRA row is only to show that
the paired LightX2V files materially change the result on the current Wan runtime.

Download the paired LightX2V Lightning files once:

```sh
mlxgen download --model lightx2v/Wan2.2-Lightning --all-files
```

After download, you can reference the paired files in either of these ways:

- public repository form: `lightx2v/Wan2.2-Lightning:<subdir>/<file>.safetensors`
- absolute local file path: `/absolute/path/to/<file>.safetensors`

For A14B, pass each adapter file as its own `--lora-paths` argument. Do not combine the two files
into one quoted string.

```sh
mlxgen generate \
  --model AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit \
  --prompt "A cinematic wide-angle movie shot of a massive futuristic starship taking off from a frozen tundra. The ship features sleek dark metallic armor. Two massive warp nacelles pulse with bright blue plasma. Violent snow squalls whip around the hull. The camera slowly tilts up as the thrusters ignite and massive clouds of snow blast away from the launch pad. Photorealistic, highly detailed, dramatic lighting." \
  --negative "oversaturated colors, overexposed, static shot, blurry details, subtitles, text, watermark, painting, illustration, ugly, deformed, broken anatomy, extra limbs, cluttered background, frozen frame, low quality, jpeg artifacts" \
  --width 480 \
  --height 240 \
  --frames 41 \
  --steps 4 \
  --guidance 1 \
  --guidance-2 1 \
  --flow-shift 5 \
  --fps 20 \
  --seed 8401 \
  --metadata \
  --replace \
  --output validation_outputs/lightx2v_wan_4step_2026_06_12/a14b_t2v_4step_lightning_q8.mp4 \
  --lora-paths \
    lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors \
    lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors \
  --lora-target-roles high_noise_transformer low_noise_transformer \
  --lora-scales 1 1

mlxgen generate \
  --model AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Starting from the input image, the silver spaceship powers up and lifts off from the frozen ground. Blue engines brighten, snow blasts outward, vapor rolls under the hull, and the camera holds the same wide icy canyon framing while the ship rises smoothly." \
  --negative "oversaturated colors, overexposed, static shot, blurry details, subtitles, text, watermark, painting, illustration, ugly, deformed, broken anatomy, extra limbs, cluttered background, frozen frame, low quality, jpeg artifacts" \
  --width 480 \
  --height 240 \
  --frames 41 \
  --steps 4 \
  --guidance 1 \
  --guidance-2 1 \
  --flow-shift 5 \
  --fps 20 \
  --seed 8402 \
  --metadata \
  --replace \
  --output validation_outputs/lightx2v_wan_4step_2026_06_12/a14b_i2v_4step_lightning_q8.mp4 \
  --lora-paths \
    lightx2v/Wan2.2-Lightning:Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors \
    lightx2v/Wan2.2-Lightning:Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors \
  --lora-target-roles high_noise_transformer low_noise_transformer \
  --lora-scales 1 1
```

If you already manage adapter files locally, the same I2V command also works with absolute paths:

```sh
mlxgen generate \
  --model AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Starting from the input image, the silver spaceship powers up and lifts off from the frozen ground. Blue engines brighten, snow blasts outward, vapor rolls under the hull, and the camera holds the same wide icy canyon framing while the ship rises smoothly." \
  --negative "oversaturated colors, overexposed, static shot, blurry details, subtitles, text, watermark, painting, illustration, ugly, deformed, broken anatomy, extra limbs, cluttered background, frozen frame, low quality, jpeg artifacts" \
  --width 480 \
  --height 240 \
  --frames 41 \
  --steps 4 \
  --guidance 1 \
  --guidance-2 1 \
  --flow-shift 5 \
  --fps 20 \
  --seed 8402 \
  --metadata \
  --replace \
  --output validation_outputs/lightx2v_wan_4step_2026_06_12/a14b_i2v_4step_lightning_q8.mp4 \
  --lora-paths \
    /absolute/path/to/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors \
    /absolute/path/to/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors \
  --lora-target-roles high_noise_transformer low_noise_transformer \
  --lora-scales 1 1
```

Wan image-to-video preserves the source image aspect ratio and rounds to a supported video size, so
the final width and height can differ slightly from the requested values.

Treat Lightning as an explicit fast recipe, not as a universal quality replacement for the
original Wan profile. The measured result is that it produces coherent local videos much faster.
Depending on the prompt and route, the original longer profile can still yield a stronger or simply
different interpretation.

The LightX2V README itself also matters here:

- it advertises A14B T2V and I2V at `480P` and `720P`, not `240p`
- it recommends prompt extension to improve detail
- it explicitly says the T2V model can still show artifacts on scenes with very large motion

MLX-Gen also includes a simple `240p` versus `480p` T2V sweep on the same Apple `M5 Max`:

![Wan T2V-A14B q8 LightX2V resolution sweep on M5 Max](assets/validation/lightx2v-wan-4step-2026-06-12/a14b_t2v_lightx2v_resolution_sweep_m5max.jpg)

The sweep uses the same LightX2V `4`-step recipe at `480x240` and `832x480` for `41` frames. On
the same machine, the `240p` row finishes in `92.98s` and the `480p` row finishes in `334.75s`.
The `480p` row is visibly stronger, which supports the LightX2V README's `480P` / `720P` quality
envelope claim.

For the documented `720p` T2V profile, the guide also includes a same-seed q8-versus-BF16
keyframe comparison using the same LightX2V recipe:

![Wan T2V-A14B q8 vs BF16 keyframe comparison](assets/validation/lightx2v-wan-4step-2026-06-12/a14b_t2v_lightx2v_q8_vs_bf16_frame_compare.png)

MLX-Gen also now includes a compact `41`-frame, `20` fps progress matrix on an Apple `M5 Max`:

![Wan T2V-A14B LightX2V progress matrix on M5 Max](assets/validation/lightx2v-wan-4step-2026-06-12/a14b_t2v_lightning_quant_progress_m5max.jpg)
![Wan I2V-A14B LightX2V progress matrix on M5 Max](assets/validation/lightx2v-wan-4step-2026-06-12/a14b_i2v_lightning_quant_progress_m5max.jpg)

The T2V matrix covers the prepared variants used for the current T2V LightX2V proof:

- `AbstractFramework/wan2.2-t2v-a14b-diffusers-bf16` with paired LightX2V LoRAs
- `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` with paired LightX2V LoRAs
- `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` on the current practical original profile

The I2V matrix covers the prepared q8 route used for the current I2V LightX2V proof:

- `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` with paired LightX2V LoRAs
- `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` on the current practical original profile

Measured on the same Apple `M5 Max`, same seeds, and the same `41`-frame target:

| Route | Model / profile | Wall time | Max RSS |
| --- | --- | ---: | ---: |
| T2V | `wan2.2-t2v-a14b` prepared BF16 + LightX2V `4`-step | `87.99s` | `32.77 GiB` |
| T2V | `wan2.2-t2v-a14b` prepared q8 + LightX2V `4`-step | `92.98s` | `13.45 GiB` |
| T2V | `wan2.2-t2v-a14b` prepared q8 original `20`-step | `516.45s` | `13.83 GiB` |
| I2V | `wan2.2-i2v-a14b` prepared q8 + LightX2V `4`-step | `87.88s` | `13.45 GiB` |
| I2V | `wan2.2-i2v-a14b` prepared q8 original `20`-step | `446.17s` | `13.73 GiB` |

On this `41`-frame profile, the Lightning recipe produces:

- `5.55x` faster q8 T2V than the current practical q8 original profile
- `5.08x` faster q8 I2V than the current practical q8 original profile

The machine-readable measurements are in
`docs/assets/validation/lightx2v-wan-4step-2026-06-12/a14b_lightning_quant_stats_m5max.json`.

Practical reading:

- for fast local previsualization, the `240p` `4`-step recipe is useful
- for presentation quality, stay closer to `480P` / `720P` or use the longer original Wan profile
- for this family, I2V currently tolerates the quick `240p` path better than T2V because the
  source frame preserves composition and detail

The validated Wan runs used exact base-model adapters, explicit role assignment, and same-seed A/B
comparisons across TI2V-5B and A14B public rows.

Combined route matrix: [summary sheet](assets/validation/wan-lora-2026-06-11/wan_video_lora_route_matrix.jpg)

Base `AbstractFramework/qwen-image-8bit` remains experimental. Exact-base adapters now load
cleanly on both the base Qwen and original Qwen edit routes, but only the original
`AbstractFramework/qwen-image-edit-8bit` single-image edit row has an accepted exact proof today.

## A/B Validation Method

Do not judge a LoRA from a single output. Use the same source, prompt, dimensions, seed, steps, and
guidance with and without the adapter.

For image-to-image LoRAs, keep the source image fixed:

```sh
mlxgen generate \
  --model <compatible-edit-model> \
  --image source.png \
  --prompt "<adapter-specific prompt>" \
  --width 432 \
  --height 240 \
  --steps 24 \
  --guidance 4 \
  --seed 42 \
  --output no_lora.png

mlxgen generate \
  --model <compatible-edit-model> \
  --image source.png \
  --prompt "<adapter-specific prompt>" \
  --width 432 \
  --height 240 \
  --steps 24 \
  --guidance 4 \
  --seed 42 \
  --lora-paths owner/repo:adapter.safetensors \
  --lora-scales 0.9 \
  --output with_lora.png
```

For text-to-image LoRAs, keep the prompt and seed fixed. Use a contact sheet that shows the source
or baseline, the no-LoRA output, and the with-LoRA output side by side. The with-LoRA output should
show the adapter's intended effect while preserving the requested prompt and source constraints.

## Current Experimental Boundaries

- Exact validated rows today are:
  - `AbstractFramework/qwen-image-edit-8bit` on `qwen.edit`;
  - `AbstractFramework/qwen-image-edit-2511-8bit` on `qwen.edit`;
  - `AbstractFramework/qwen-image-edit-2509-8bit` on `qwen.edit`;
  - `AbstractFramework/qwen-image-2512-8bit` on `qwen.text`;
  - `AbstractFramework/z-image-turbo-8bit` on `z-image.text`;
  - `AbstractFramework/flux.2-klein-9b-8bit` on `flux2.edit`;
  - `AbstractFramework/ernie-image-turbo-8bit` on `ernie-image.text`.
- Adjacent rows remain experimental. That includes Qwen base generation,
  Qwen multi-reference, Qwen reframe/outpaint, Z-Image latent img2img, ERNIE latent img2img,
  FLUX.2 multi-reference, and FLUX.2 reframe/outpaint.
- Original `AbstractFramework/qwen-image-8bit` still has no exact validated text row. The public
  `AbstractFramework/qwen-image-8bit` package is now complete locally, and the exact-base
  `flymy-ai/qwen-image-realism-lora` adapter loads cleanly on the route. It still needs a stronger
  visible A/B before the row can be promoted.
- Bonsai, FIBO, and SeedVR2 reject LoRA in unified generation. Bonsai stays fail-closed because
  its packed runtime does not expose the normal replaceable linear-module boundary that the current
  LoRA loader requires.
- Wan video LoRA is now part of the unified video routes, and all current Wan q8 public rows have
  exact route-level proof in the contact sheets above.
- `mlxgen prepare --lora-paths` is rejected until save/reload behavior is proven for the selected
  family and quantization mode.
