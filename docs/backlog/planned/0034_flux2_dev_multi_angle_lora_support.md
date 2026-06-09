# Planned: FLUX.2-dev multi-angle LoRA support

## Metadata

- Created: 2026-06-08
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs:
  - [ADR 0001](../adr/0001_runtime_smoke_validation_for_model_routes.md)
  - [ADR 0002](../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: No new ADR is required if this remains a model-integration and validation item.
  Create an ADR only if FLUX.2-dev introduces a new adapter/plugin architecture or a fallback
  compatibility policy.

## Context

The user downloaded two public camera-angle LoRAs:

- `lovis93/Flux-2-Multi-Angles-LoRA-v2`
- `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`

The FLUX adapter is the faster candidate for first validation, but its model card declares
`base_model: black-forest-labs/FLUX.2-dev`. Current MLX-Gen FLUX.2 support is FLUX.2 Klein 4B/9B,
not FLUX.2-dev.

The local proof on 2026-06-08 showed the adapter is not shape-compatible with FLUX.2 Klein:

| Selected model | Rejection |
| --- | --- |
| `flux2-klein-4b` | Expected `context_embedder` LoRA shapes `(7680, rank)` and `(rank, 3072)`, got `(15360, 16)` and `(16, 6144)`. |
| `flux2-klein-9b` | Expected `context_embedder` LoRA shapes `(12288, rank)` and `(rank, 4096)`, got `(15360, 16)` and `(16, 6144)`. |

MLX-Gen now rejects `black-forest-labs/FLUX.2-dev` as a model handle and rejects the lovis adapter
for FLUX.2 Klein through cached model-card compatibility preflight.

## Current code reality

- FLUX.2 runtime classes and weight definitions are Klein-specific:
  - `src/mflux/models/flux2/flux2_initializer.py`
  - `src/mflux/models/flux2/weights/flux2_weight_definition.py`
  - `src/mflux/models/flux2/model/flux2_transformer/transformer.py`
- `src/mflux/models/common/resolution/config_resolution.py` intentionally rejects
  `black-forest-labs/FLUX.2-dev` instead of inferring a FLUX.1 `dev` config.
- `src/mflux/models/common/lora/lora_compatibility.py` rejects cached adapter model-card
  mismatches for known base models.
- `src/mflux/models/flux2/weights/flux2_lora_mapping.py` includes some Diffusers-style key aliases,
  but aliases alone are not a FLUX.2-dev runtime port.
- The user has the lovis LoRA locally. There is no confirmed local `black-forest-labs/FLUX.2-dev`
  source model cache in this checkout at the time this item was created.

## Problem

The first FLUX.2 LoRA validation cannot use `lovis93/Flux-2-Multi-Angles-LoRA-v2` with FLUX.2
Klein. Doing so would be a false proof because the adapter is trained for a different transformer
width and base model.

## Scope

1. Decide whether to implement first-class `black-forest-labs/FLUX.2-dev` support or to select a
   FLUX.2 Klein-compatible public adapter for the first FLUX.2 LoRA proof.
2. If choosing FLUX.2-dev:
   - confirm model license/access and local cache feasibility;
   - add explicit `ModelConfig` entries and router handling;
   - port the FLUX.2-dev weight definition and any transformer config differences from upstream
     Diffusers/Transformers;
   - add a dedicated LoRA mapping or compatibility class if dev keys differ from Klein keys;
   - keep Klein behavior unchanged.
3. If choosing a Klein-compatible adapter:
   - verify the model card declares FLUX.2 Klein or test matrix shapes before generation;
   - keep the adapter in `mapped-unvalidated` until the A/B proof passes.
4. Produce a source/no-LoRA/with-LoRA contact sheet using the same prompt, seed, dimensions, steps,
   guidance, source image when applicable, and output size.
5. Update `docs/lora.md`, `docs/edit-capabilities.md` or a dedicated validation page with the exact
   commands and artifact paths only after a real visual proof passes.

## Non-goals

- Do not silently substitute FLUX.2 Klein for FLUX.2-dev.
- Do not mark `lovis93/Flux-2-Multi-Angles-LoRA-v2` as validated on Klein.
- Do not implement Qwen multi-angle LoRA validation in this item; use item 0007 or a separate Qwen
  follow-up after FLUX.2 direction is settled.
- Do not implement video LoRA here.

## Acceptance criteria

- `mlxgen capabilities --model black-forest-labs/FLUX.2-dev` either reports a first-class FLUX.2-dev
  route with correct capability metadata or continues to fail with the current explicit unsupported
  message.
- `mlxgen generate --model flux2-klein-4b --lora-paths lovis93/Flux-2-Multi-Angles-LoRA-v2:flux-multi-angles-v2-72poses-comfy.safetensors ...`
  fails before backend dispatch with a message that names FLUX.2-dev versus FLUX.2 Klein.
- If FLUX.2-dev support is implemented, the lovis adapter runs through unified `mlxgen generate`
  with the prompt format from its model card, such as `<sks> back view eye-level shot medium shot`.
- A contact sheet shows:
  - the source or baseline input;
  - the same command without LoRA;
  - the same command with LoRA;
  - prompt, seed, dimensions, steps, guidance, adapter path, and adapter scale.
- Generated metadata records the requested adapter path and scale. A later item should add applied
  target counts once `LoRAApplicationReport` exists.

## Validation

- Unit tests for `black-forest-labs/FLUX.2-dev` model resolution.
- Unit tests proving the lovis adapter is rejected for FLUX.2 Klein through model-card preflight.
- If FLUX.2-dev is added:
  - fast config/weight-definition tests;
  - one small model-backed source run without LoRA;
  - one same-settings run with LoRA;
  - one contact sheet reviewed for a visible angle change and prompt adherence.
- If a Klein-compatible adapter is selected instead:
  - model-card or shape compatibility check;
  - source/no-LoRA/with-LoRA contact sheet;
  - exact command log in coredoc.

## Notes

- The lovis model card lists the base model as `black-forest-labs/FLUX.2-dev`, prompt format as
  `<sks> [view] [elevation] shot [distance]`, and recommended strength as `0.8 - 1.0`.
- The Qwen adapter model card lists the base model as `Qwen/Qwen-Image-Edit-2511`, prompt format as
  `<sks> [azimuth] [elevation] [distance]`, and should be validated separately with the Qwen 2511
  source/q8/q4 edit routes.
