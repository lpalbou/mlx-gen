# LoRA

LoRA support in MLX-Gen is route-specific and exact-row validated. The public support claim is not
"this family probably loads adapters"; it is "this exact model/package and route has a current
accepted A/B proof row."

MLX-Gen still fails closed on bad LoRA input:

- missing or unreadable adapter files;
- corrupt files;
- zero matched keys;
- zero applied targets;
- incompatible matrix shapes;
- known adapter/base-model mismatches from cached model-card metadata;
- unsupported route or model-family combinations.

Generation records what actually applied in output metadata:

- `lora_application_reports`
- `lora_applied_file_count`
- `lora_applied_target_count`

## Check Support First

Use `mlxgen capabilities` before a run:

```sh
mlxgen capabilities --model AbstractFramework/flux.2-klein-9b-8bit
```

Each capability row includes:

| Field | Meaning |
| --- | --- |
| `supports_lora` | Whether the route accepts LoRA arguments at all. |
| `lora_status` | `unsupported`, `mapped-unvalidated`, or `validated`. |
| `lora_target_roles` | Model components targeted by adapters, such as `transformer`. |
| `lora_validation_profile` | Exact proof row id for `mlxgen validation`. |

Treat only `lora_status="validated"` rows as production-supported. `mapped-unvalidated` means
MLX-Gen has a mapping and strict loader path for the route, but the exact model/package row has not
yet passed a visible accepted-adapter A/B proof.

Inspect the proof row directly:

```sh
mlxgen validation \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --profile lora_qwen2511_q8_outpaint_multi_angle_2026_06_22
```

## Current Production-Supported Surface

These are the current exact validated LoRA rows in the public MLX-Gen contract.

| Exact model | Validated routes |
| --- | --- |
| `AbstractFramework/qwen-image-edit-8bit` | `qwen.edit` |
| `AbstractFramework/qwen-image-edit-2509-8bit` | `qwen.edit` |
| `AbstractFramework/qwen-image-edit-2511-8bit` | `qwen.edit`, `qwen.inpaint`, `qwen.multi-reference`, `qwen.reframe`, `qwen.outpaint` |
| `AbstractFramework/qwen-image-8bit` | `qwen.text`, `qwen.latent`, `qwen.control`, `qwen.control-inpaint` |
| `AbstractFramework/qwen-image-2512-8bit` | `qwen.text` |
| `AbstractFramework/z-image-turbo-8bit` | `z-image.text`, `z-image.latent` |
| `AbstractFramework/ernie-image-turbo-8bit` | `ernie-image.text`, `ernie-image.latent` |
| `AbstractFramework/flux.2-klein-9b-8bit` | `flux2.edit`, `flux2.multi-reference` |
| `AbstractFramework/flux.2-klein-base-4b-8bit` | `flux2.outpaint` |
| `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` | `wan.text-video`, `wan.first-frame` |
| `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` | `wan.text-video` |
| `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` | `wan.first-frame` |

Current deliberate boundaries:

- `AbstractFramework/qwen-image-2512-8bit` latent img2img is not validated yet.
- Distilled FLUX.2 Klein LoRA support is not promoted as a general public surface. The exact
  current FLUX.2 claim is the three rows listed above and nothing broader.
- SeedVR2, FIBO, and Bonsai are not public LoRA families today.

## Published Proof Bundles

The current exact image-route completion bundle is:

- [LoRA route expansion report](assets/validation/lora-route-expansion-2026-06-22/lora_route_expansion_report.md)
- [LoRA route expansion command log](assets/validation/lora-route-expansion-2026-06-22/lora_route_expansion_command_log.md)
- [LoRA route expansion stats](assets/validation/lora-route-expansion-2026-06-22/lora_route_expansion_stats_m5max.json)
- [Z-Image Turbo q8 latent LoRA report](assets/validation/zimage-latent-lora-2026-06-24/zimage_latent_lora_report.md)
- [Z-Image Turbo q8 latent LoRA command log](assets/validation/zimage-latent-lora-2026-06-24/zimage_latent_lora_command_log.md)
- [Z-Image Turbo q8 latent LoRA stats](assets/validation/zimage-latent-lora-2026-06-24/zimage_latent_lora_stats_m5max.json)

That bundle includes accepted contact sheets for:

- base Qwen text realism;
- base Qwen latent realism;
- Z-Image Turbo latent children's-drawing style transfer;
- Qwen Image Edit 2511 multi-reference;
- Qwen Image Edit 2511 reframe;
- Qwen Image Edit 2511 outpaint;
- FLUX.2 Klein 9B multi-reference;
- FLUX.2 Klein base 4B outpaint;
- ERNIE latent img2img.

The exact Z-Image latent row is:

- `AbstractFramework/z-image-turbo-8bit` on `z-image.latent`
- adapter: `ostris/z_image_turbo_childrens_drawings:z_image_turbo_childrens_drawings.safetensors`
- proof shape: same source, prompt, seed, and latent strength; only the adapter changes

Wan video LoRA uses a separate MP4 proof bundle. See the Wan sections in this guide and
[Wan Video](wan-video.md).

## Download Adapters Explicitly

Generation does not download LoRA files. Download the adapter repository first:

```sh
mlxgen download --model fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA --all-files
```

Then reference either:

- a local `.safetensors` path, or
- a cached Hugging Face adapter id in `owner/repo:file.safetensors` form.

The file part can include a subdirectory:

```sh
mlxgen generate \
  --model <compatible-model> \
  --prompt "<prompt from the adapter model card>" \
  --lora-paths owner/repo:adapter.safetensors \
  --lora-scales 0.9 \
  --output with_lora.png
```

The number of `--lora-scales` values must match the number of `--lora-paths` values exactly.
Passing scales without paths fails before model load.

## Adapter Compatibility

Read the adapter model card and match its base model. A LoRA trained for one model family is not
automatically compatible with another.

Example:

- `lovis93/Flux-2-Multi-Angles-LoRA-v2` targets `black-forest-labs/FLUX.2-dev`
- MLX-Gen currently validates FLUX.2 Klein rows, not FLUX.2-dev
- MLX-Gen rejects that adapter on Klein routes

The same rule applies inside one family: do not assume a text-only proof automatically validates a
latent, multi-reference, reframe, or outpaint route.

## Exact Example Rows

### Base Qwen latent img2img

Validated row:

- model: `AbstractFramework/qwen-image-8bit`
- route: `qwen.latent`
- adapter: `prithivMLmods/Qwen-Image-Studio-Realism:qwen-studio-realism.safetensors`

Representative command:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-8bit \
  --image docs/assets/validation/lora-route-expansion-2026-06-22/qwen_q8_latent_source_portrait_illustration.png \
  --i2i-mode latent \
  --image-strength 0.6 \
  --prompt "Studio Realism, photorealistic portrait of the same young woman of African descent standing in the same sunlit park with arms crossed, the same loose shoulder-length curls, the same pendant necklace, and the same sleeveless taupe dress. Preserve the same pose, framing, and background layout. Natural skin texture, realistic hair strands, subtle outdoor depth of field, no text." \
  --width 512 \
  --height 512 \
  --steps 24 \
  --guidance 5 \
  --seed 4421 \
  --lora-paths prithivMLmods/Qwen-Image-Studio-Realism:qwen-studio-realism.safetensors \
  --lora-scales 1.0 \
  --output qwen_q8_latent_with_lora.png
```

### Qwen Image Edit 2511 reframe/outpaint/multi-reference

Validated rows:

- model: `AbstractFramework/qwen-image-edit-2511-8bit`
- adapters:
  - `lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`
  - `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:qwen-image-edit-2511-multiple-angles-lora.safetensors`

These rows are validated route-by-route. The Lightning adapter is the fast baseline; the
multi-angle adapter is what changes the viewpoint behavior on top of that exact route.

### FLUX.2 Klein base 4B outpaint

Validated row:

- model: `AbstractFramework/flux.2-klein-base-4b-8bit`
- route: `flux2.outpaint`
- adapter: `fal/flux-2-klein-4B-outpaint-lora:flux-outpaint-lora.safetensors`

Representative command:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-base-4b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Fill the green spaces according to the image" \
  --width 1040 \
  --height 272 \
  --steps 20 \
  --guidance 4 \
  --seed 8612 \
  --lora-paths fal/flux-2-klein-4B-outpaint-lora:flux-outpaint-lora.safetensors \
  --lora-scales 1.0 \
  --output flux2_base4b_q8_outpaint_with_lora.png
```

The base route already supports outpaint without a LoRA. This exact validated row exists to prove
the dedicated outpaint adapter on the same green-canvas route and seed, not to claim that
`flux2.outpaint` needs a LoRA to function at all.

## Lightning Caveat For Qwen

For the LightX2V Qwen Lightning adapters, keep one distinction clear:

- MLX-Gen's validated public routes use the published `AbstractFramework/*-8bit` q8 packages.
- The upstream LightX2V note about BF16-versus-FP8 Lightning compatibility applies to external
  FP8 Qwen checkpoints, not to MLX-Gen's q8 packages.

Current MLX-Gen recommendation:

1. use the validated q8 MLX-Gen package for the route you want;
2. use the exact documented Lightning adapter for that route;
3. do not generalize those results to unrelated third-party FP8 checkpoints.

Upstream reference:

- <https://github.com/ModelTC/LightX2V-Qwen-Image-Lightning#-using-lightning-loras-with-fp8-models>

## Wan Video LoRA

Wan video LoRA is now exact-route validated on the current q8 public rows.

TI2V-5B:

- `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` on `wan.text-video`
- `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` on `wan.first-frame`

A14B:

- `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` on `wan.text-video`
- `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` on `wan.first-frame`

For A14B, pass the intended `high_noise_transformer` / `low_noise_transformer` role assignment
explicitly when you use separate adapter files. MLX-Gen does not silently duplicate roles.

Video-to-video (plain and masked) on `Wan2.2-T2V-A14B` also accepts the T2V Lightning pairs
through the public `unipc` route. Bounded validation on 2026-07-04 covered
`Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1` (two seeds, conference and ship clips at
480x832/25f and 448x256/17f, plus the `--video-mask-path` combination) and
`...-Seko-V2.0` (one seed; slightly closer source tracking). Recipe: `--steps 4
--video-strength 0.75 --guidance 1 --guidance-2 1 --flow-shift 5 --solver unipc`. Negative
prompts are inert at guidance 1, unmasked runs re-synthesize the scene more than the
20-step CFG-on baseline, and the 4-step lattice cannot reach the motion-preserving strength
band (fast and motion-preserving are mutually exclusive; see
[Motion Fidelity Versus Strength](wan-video.md#motion-fidelity-versus-strength)); see also
[Wan Video](wan-video.md#fast-video-to-video-with-lightning) and
the included proof bundle [docs/assets/validation/lightning-v2v-2026-07-04/](assets/validation/lightning-v2v-2026-07-04/README.md).
The stable copy-paste adapter form works directly from the Hugging Face cache:

```sh
--lora-paths "lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors" \
             "lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors" \
--lora-target-roles high_noise_transformer low_noise_transformer
```

For the current A14B fast path:

```sh
mlxgen download --model lightx2v/Wan2.2-Lightning --all-files
```

Then use the paired T2V or I2V files shown in the current validation examples. The stable public
adapter form is `owner/repo:subdir/file.safetensors`, and absolute local file paths are also fine
after download.
