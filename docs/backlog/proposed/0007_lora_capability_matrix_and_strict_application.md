# Proposed: LoRA capability matrix and strict application

## Metadata

- Created: 2026-05-28
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: Needs new ADR only if MLX-Gen changes LoRA into a plugin/provider interface rather
  than per-family mappings.

## Context

MLX-Gen already exposes LoRA arguments and metadata, and the current public docs advertise LoRA
support. That support is real for some image families but not universal. As MLX-Gen becomes the
Apple Silicon backend for AbstractVision, callers need a reliable capability answer before they
offer LoRA controls in a UI or workflow.

Qwen-Image-Edit-2511 also makes LoRA strategically important: its model card calls out integrated
LoRA capabilities and community LoRA effects as part of the 2511 upgrade.

## Current code reality

- `src/mflux/cli/parser/parsers.py` adds `--lora-style`, `--lora-paths`, and `--lora-scales`.
- `src/mflux/models/common/lora/mapping/lora_loader.py` loads LoRA files, applies family mappings,
  and prints unmatched keys. Missing or unreadable LoRA files currently print an error and return
  from `_apply_single_lora()` instead of failing the generation.
- LoRA mappings exist for FLUX.1, FLUX.2, Qwen, and Z-Image:
  - `src/mflux/models/flux/weights/flux_lora_mapping.py`
  - `src/mflux/models/flux2/weights/flux2_lora_mapping.py`
  - `src/mflux/models/qwen/weights/qwen_lora_mapping.py`
  - `src/mflux/models/z_image/weights/z_image_lora_mapping.py`
- FLUX.2 and Z-Image also have training adapters.
- ERNIE and Bonsai accept `lora_paths` in constructor signatures for prepare compatibility, but
  their initializers delete those arguments and set `model.lora_paths = None`.
- Wan, SeedVR2, and FIBO do not have proven LoRA mappings in the current MLX-Gen tree. FIBO is the
  most misleading risk if any CLI path accepts LoRA flags without applying them.

## Problem or opportunity

LoRA should be treated as required user input, not best-effort decoration. If a user asks for a LoRA
and it is missing, corrupt, maps zero keys, or targets a family that does not support LoRA, MLX-Gen
should fail early with a clear message. Silent or warning-only behavior is dangerous because the
output image can look plausible while ignoring the requested adapter.

## Proposed direction

Add a capability matrix and strict LoRA application policy:

1. Add family-level capability metadata for LoRA inference and LoRA training.
2. Make the unified `mlxgen` router reject LoRA flags for unsupported families before model load.
3. Change LoRA loading so user-requested files must exist, load, and apply at least one mapped
   target.
4. Keep partial-match warnings for valid adapters, but fail zero-match adapters by default.
5. Add docs and generated capability metadata so AbstractVision can decide whether to display LoRA
   controls.

Initial support matrix should be explicit:

| Family | Inference LoRA | Training LoRA | Action |
| --- | --- | --- | --- |
| Qwen Image/Edit | Yes | Not yet proven in MLX-Gen | Validate with real 2511 LoRAs. |
| FLUX.1/FLUX.2 | Yes | FLUX.2 yes | Keep supported, add stricter tests. |
| Z-Image | Yes | Yes | Keep supported, validate against public Z LoRAs. |
| ERNIE | No | No | Reject LoRA flags. |
| Bonsai | No | No | Reject LoRA flags; packed low-bit transformer is not a normal adapter target. |
| Wan | No | No | Reject LoRA flags until a Wan mapping and video validation exist. |
| FIBO | Unknown | No | Either implement a mapping or reject LoRA flags clearly. |
| SeedVR2 | No | No | Reject LoRA flags. |

## Why it might matter

LoRA is one of the fastest ways to make MLX-Gen useful for personal styles, product references,
characters, and AbstractVision workflows. It is also one of the easiest places to lie accidentally:
loading a model and ignoring the adapter creates outputs that are technically valid but
semantically wrong.

## Promotion criteria

- The next AbstractVision integration needs to expose LoRA controls.
- A user reports a LoRA run that completed while ignoring a missing or zero-match adapter.
- We decide to promote Qwen-Image-Edit-2511 LoRA workflows as a first-class feature.
- A new family port wants LoRA support and needs a shared test contract.

## Validation ideas

- Unit test missing file, corrupt file, zero matched keys, partial matched keys, and successful
  application.
- Router tests proving LoRA accepted for Qwen/FLUX/Z-Image and rejected for ERNIE/Wan/Bonsai until
  implemented.
- Real-image contact sheet with Qwen-Image-Edit-2511 and at least one public 2511 LoRA.
- Metadata test proving generated images preserve original LoRA paths and scales.
- Save/prepare test proving baked or stripped LoRA behavior is documented and deterministic.

## Non-goals

- This proposal does not authorize implementing LoRA for every family.
- This proposal does not require automatic LoRA downloads during generation.
- This proposal does not change existing quantization policies.

## Guidance for future agents

Start with strict diagnostics and the capability matrix before adding new family mappings. If a
family cannot prove LoRA application with a known adapter and visible output difference, mark it as
unsupported rather than "probably works".

## Sources checked

- `src/mflux/models/common/lora/mapping/lora_loader.py`
- `src/mflux/models/qwen/weights/qwen_lora_mapping.py`
- `src/mflux/models/flux2/weights/flux2_lora_mapping.py`
- `src/mflux/models/z_image/weights/z_image_lora_mapping.py`
- Qwen-Image-Edit-2511 model card: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
