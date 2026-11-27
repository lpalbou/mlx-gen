# Planned: Recommend the 4-step adapters with each Wan and Qwen model

## Metadata

- Created: 2026-10-04
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md). MLX-Gen documents
  validated exact pairs; it does not pick an adapter for the caller.
- ADR impact: None.

## Context

Each Wan2.2 and Qwen model has a 4-step adapter that cuts a run from 20 to 50 steps down to 4.
The docs show these pairs inside long examples, one page at a time. A reader who downloads a
model is not told, in one place, which adapter to install with it.

## Current code reality

Pairs already validated and documented (4 steps, guidance 1):

| Model | Adapter | Where documented |
| --- | --- | --- |
| Wan2.2 T2V-A14B | `lightx2v/Wan2.2-Lightning`: `Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors` and `low_noise_model.safetensors` (roles `high_noise_transformer`, `low_noise_transformer`; guidance-2 1, flow shift 5) | [item 0040](../completed/0040_lightx2v_wan_4step_acceleration_profiles.md) |
| Wan2.2 I2V-A14B | `lightx2v/Wan2.2-Lightning`: `Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors` and `low_noise_model.safetensors` (same roles and settings) | item 0040 |
| Qwen-Image | `lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors` | `docs/faq.md`, `docs/api.md`, `docs/image-edit-modes.md` |
| Qwen-Image-Edit-2511 | `lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | `docs/faq.md`, `docs/edit-capabilities.md`, `docs/lora.md` |

Not yet validated by MLX-Gen:

- Wan2.2 TI2V-5B: `Kijai/WanVideo_comfy:FastWan/Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors`
  (FastWan, 4 steps, guidance 1). A host application ships it as its default for this model and measured
  it on 2026-10-04; MLX-Gen has no proof of its own, and the adapter's page states no license.
- Qwen-Image-2512: no 4-step pair is recorded in the docs.
- `lightx2v/Wan2.2-Distill-Loras` is listed in item 0040 as a secondary compatibility set only.

Engine fault found on 2026-10-04 while installing these pairs from an empty cache:

- `LoraResolution._is_collection_in_cache` (`src/mflux/models/common/resolution/lora_resolution.py`)
  answers "cached" as soon as any snapshot of the repository is on disk, because
  `snapshot_download(local_files_only=True)` returns the snapshot folder whether or not the
  requested file is in it. `_load_collection_from_cache` then raises
  `Could not find LoRA file '...' in downloaded files` instead of fetching the file.
- Effect: `repo:file` adapters from one repository cannot be installed one after the other. The
  second file of a Wan A14B pair fails, then the other A14B model's pair, and a second
  MiniMax-H3 schedule's `turbo_lora` through `mlxgen download`.
- A host application (its release 0.9.3) works around it in its download process by replacing
  `_load_collection_from_cache`.

## What we want to do

- Fix the cache check so a file counts as cached only when that file is on disk, and fetch it
  otherwise when downloads are allowed. Cover it with a test that installs two files of one
  repository in sequence.

- Add one table to `docs/lora.md`, "4-step adapters by model", with the pairs above, the
  download command for each, and the run settings. Link it from `docs/recommendations.md`, the
  FAQ and the model cards.
- Have `mlxgen download --model <one of these models>` print one line naming the validated
  adapter and its download command. It prints; it does not download or apply anything.
- Validate FastWan on TI2V-5B and a 4-step adapter on Qwen-Image-2512, then add them to the
  table, or record here why not.

## Scope and non-goals

- In scope: the four validated pairs, the two candidates, docs and the download hint.
- Not in scope: applying an adapter by default, bundling adapters into model packages, or
  recommending adapters that have no MLX-Gen proof.

## Dependencies and related tasks

- [0040](../completed/0040_lightx2v_wan_4step_acceleration_profiles.md) for the Wan pairs.
- [0126](0126_wan_q8_keep_8bit_at_runtime.md): its runs use the same Wan pairs.
- [0007](../completed/0007_lora_capability_matrix_and_strict_application.md) for the Qwen rows.

## Validation

- Every row of the new table runs as written from a clean cache: download, then a 4-step
  generation, with the adapter reported as fully applied.
- The download hint appears for exactly the models in the table and for no other.
- For each new pair, a same-seed comparison with and without the adapter, watched in full.

## Progress checklist

- [ ] Table in `docs/lora.md`, linked from recommendations, FAQ and model cards.
- [ ] Download hint for the listed models.
- [ ] FastWan on TI2V-5B validated or declined, with its license checked.
- [ ] Qwen-Image-2512 adapter validated or declined.
- [ ] `CHANGELOG.md` and `llms` files updated.
