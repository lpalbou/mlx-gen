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
model/package and task has not yet passed a visible A/B validation with a public adapter. Treat
LoRA routes as experimental unless a current A/B contact sheet demonstrates the intended adapter
effect for your selected model/package.

## Download And Reference Adapters

Generation does not download LoRA files. Download the adapter repository explicitly:

```sh
mlxgen download --model lovis93/Flux-2-Multi-Angles-LoRA-v2 --all-files
```

Use a local `.safetensors` path or a Hugging Face repository id. If the repository contains several
adapter files, specify the file after a colon:

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

The downloaded `lovis93/Flux-2-Multi-Angles-LoRA-v2` adapter targets
`black-forest-labs/FLUX.2-dev`, uses prompts that start with `<sks>`, and recommends adapter
strength around `0.8` to `1.0`. MLX-Gen currently supports FLUX.2 Klein 4B/9B, not
`black-forest-labs/FLUX.2-dev`. Passing this adapter to FLUX.2 Klein is rejected because the LoRA
matrices target a different transformer width.

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
outputs. The LoRA loader matched and applied all `1,680` adapter tensors for both LoRA runs.

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

- FLUX.2 Klein, Qwen image models, and Z-Image expose LoRA as `mapped-unvalidated` until exact
  public-adapter A/B rows are added. The first checked row is Qwen Image Edit 2511 q8 with
  `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`; it applies correctly, but the visible benefit is
  modest on the current spaceship prompt because base Qwen already handles viewpoint edits well.
- ERNIE, Bonsai, FIBO, Wan, and SeedVR2 reject LoRA in unified generation.
- Video LoRA is tracked separately from image LoRA and is not part of the current unified video
  routes.
- `mlxgen prepare --lora-paths` remains gated until save/reload behavior is proven for the selected
  family and quantization mode.
