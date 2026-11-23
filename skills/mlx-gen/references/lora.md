# LoRA Adapters

Open this file before adding `--lora-paths` to a run. LoRA support is route-specific and fails
closed: a requested adapter that cannot be applied stops the run instead of being skipped.

## Workflow

1. Check the route. In `mlxgen capabilities --model <id>`, read on the row you will use:
   `supports_lora`, `lora_status` (`unsupported`, `mapped-unvalidated`, `validated`),
   `lora_target_roles`, and `lora_validation_profile`. Treat only `validated` as
   production-supported; `mapped-unvalidated` loads through a strict mapping but has no accepted
   visual proof for that exact model and route.
2. Match the adapter to the model. An adapter is trained for one base model: read its model card.
   A FLUX.2-dev adapter is rejected on FLUX.2 Klein. A proof on one route (text) does not validate
   another (latent, outpaint) of the same model.
3. Download it explicitly. Generation never downloads adapters.
4. Pass it with one scale per file.

```sh
mlxgen download --model "<OWNER/LORA_REPO>" --all-files
mlxgen generate \
  --model "<COMPATIBLE_MODEL>" \
  --prompt "<PROMPT FROM THE ADAPTER CARD>" \
  --lora-paths "<OWNER/LORA_REPO>:<SUBDIR/FILE.safetensors>" \
  --lora-scales 0.9 \
  --seed 42 --metadata \
  --output "<WITH_LORA.png>"
```

Adapter references: a local `.safetensors` path, or a cached repository as
`owner/repo:file.safetensors` (the file part may include a subdirectory). Name the file whenever
the repository holds more than one.

## Rules

- The number of `--lora-scales` values must equal the number of `--lora-paths` values.
  `--lora-scales` without `--lora-paths` fails before the model loads.
- The run fails on: a missing or unreadable file, a corrupt file, zero matched keys, zero applied
  targets, incompatible matrix shapes, a known adapter/base-model mismatch from cached model-card
  metadata, or an unsupported route.
- The sidecar records what was applied: `lora_application_reports`, `lora_applied_file_count`,
  `lora_applied_target_count`. Check them instead of assuming the adapter took effect.
- To judge an adapter, compare same-seed runs with and without it.
- Bonsai, FIBO, SeedVR2, and Bernini-R accept no LoRA.
- Use the published `AbstractFramework/*-8bit` packages for Lightning adapters; do not assume the
  same result on third-party FP8 checkpoints.

## Validated rows

| Model | Routes |
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

Inspect a proof row with the profile id from the capability row:

```sh
mlxgen validation --model "AbstractFramework/qwen-image-edit-8bit" \
  --profile lora_qwen_edit_q8_ghibli_edit_2026_06_11
```

## Documented adapters

| Use | Model | Adapter | Settings |
| --- | --- | --- | --- |
| 4-step Qwen structured control | `AbstractFramework/qwen-image-8bit` | `lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors` | `--steps 4 --guidance 1`, scale 1 |
| 4-step Qwen Edit 2511 | `AbstractFramework/qwen-image-edit-2511-8bit` | `lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | 4 steps |
| Viewpoint change on Qwen Edit 2511 | same | `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:qwen-image-edit-2511-multiple-angles-lora.safetensors` | on top of the Lightning adapter |
| Realism on base Qwen latent | `AbstractFramework/qwen-image-8bit` | `prithivMLmods/Qwen-Image-Studio-Realism:qwen-studio-realism.safetensors` | scale 1.0 |
| Outpaint on base Klein 4B | `AbstractFramework/flux.2-klein-base-4b-8bit` | `fal/flux-2-klein-4B-outpaint-lora:flux-outpaint-lora.safetensors` | `--steps 20 --guidance 4`, scale 1.0, prompt "Fill the green spaces according to the image" |
| Style on Z-Image Turbo latent | `AbstractFramework/z-image-turbo-8bit` | `ostris/z_image_turbo_childrens_drawings:z_image_turbo_childrens_drawings.safetensors` | |
| 4-step Wan A14B | `AbstractFramework/wan2.2-{t2v,i2v}-a14b-diffusers-8bit` | `lightx2v/Wan2.2-Lightning` paired high/low files | see below |

## Family notes

- **Wan A14B** has two transformers. Pass each file as its own `--lora-paths` argument and assign
  roles with `--lora-target-roles high_noise_transformer low_noise_transformer`; roles are never
  duplicated silently. TI2V-5B uses `transformer`. Example pair for text-to-video:
  `lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors`
  and `.../low_noise_model.safetensors`. See [video.md](video.md).
- **MiniMax-H3**: the list you pass replaces the automatic Turbo adapter, so include the Turbo file
  next to a style or character adapter. See [minimax-h3.md](minimax-h3.md).
- **FLUX.2 Klein**: only the three rows above are validated; distilled Klein LoRA is not a general
  supported surface.

Source docs: <https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/lora.md>.
