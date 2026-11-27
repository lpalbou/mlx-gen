# Proposed: Bonsai packed-runtime LoRA support

## Metadata

- Created: 2026-06-11
- Status: Proposed
- Priority: Low / deferred
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: May require a new ADR if MLX-Gen adds a second Bonsai execution mode or a packed-kernel
  LoRA injection path.

## Context

The question is whether MLX-Gen can add LoRA support to the currently served Bonsai model family.
Unlike ERNIE, Bonsai does not run on a normal transformer made of replaceable MLX linear modules.
It serves packed low-bit checkpoints through a fused FLUX.2 Klein megakernel path.

This is explicitly lower priority than finishing the remaining image-family LoRA proofs and any
future Wan video LoRA work. Keep Bonsai fail-closed unless the runtime architecture changes.

## Current code reality

- `src/mflux/models/bonsai_image/bonsai_image_initializer.py` accepts `lora_paths` and
  `lora_scales` only for signature compatibility, then discards them.
- Bonsai loads the main transformer through `Flux2KleinFastTransformer` in
  `src/mflux/models/flux2/model/flux2_transformer/klein_fast/transformer.py`.
- `src/mflux/models/common/lora/mapping/lora_loader.py` can only replace exposed `nn.Linear`,
  `nn.QuantizedLinear`, `LoRALinear`, or `FusedLoRALinear` targets.
- The packed Bonsai transformer stores most weights inside `MegakernelWeights` and executes them
  through compiled/fused kernels, not through ordinary linear submodules.
- Public candidate `baxin/LoRA_for_Objects` is not a Bonsai/FLUX.2 adapter in practice: its
  safetensor keys are SDXL UNet-style (`unet.up_blocks...`), so it does not prove a usable Bonsai
  adapter ecosystem.
- The Bonsai model cards point back to `prism-ml/bonsai-image-ternary-4B-unpacked`, which itself is
  a FLUX.2 Klein finetune, not a statement that the packed MLX runtime can accept ordinary FLUX.2
  LoRA injection.

## Problem or opportunity

Users will reasonably expect Bonsai LoRA to behave like FLUX.2 LoRA because Bonsai is derived from
FLUX.2 Klein. The packed runtime does not actually expose the same replacement boundary, so a naïve
“support” claim would violate ADR 0002.

## Proposed direction

Keep Bonsai LoRA fail-closed in the current packed runtime. Reassess only if one of these paths is
deliberately chosen:

1. Add an explicit unpacked Bonsai route and validate LoRA there as a separate model/runtime class.
2. Add native low-rank delta injection inside the packed megakernel path.
3. Obtain a public Bonsai adapter family whose key structure and expected target modules are clearly
   documented for the packed MLX runtime.

## Why it might matter

Bonsai is attractive because it is small and fast on Apple Silicon. If LoRA ever works here, it
would provide a lightweight personalized T2I route. But the current runtime boundary is too
different to treat this as a small patch.

## Promotion criteria

- A concrete architectural choice is accepted: unpacked Bonsai LoRA route or packed-kernel LoRA.
- At least one public adapter with real Bonsai/FLUX.2 Klein-compatible keys is available.
- The chosen route has a model-backed A/B proof through `mlxgen generate`.

## Validation ideas

- Inspect one or more public Bonsai adapters and confirm the key family matches the intended route.
- If testing an unpacked route, prove strict loader behavior and a visible style delta on the same
  prompt/seed profile.
- If testing a packed route, prove the low-rank delta is actually applied inside the fused kernel
  and survives regression tests.

## Non-goals

- Do not silently reinterpret FLUX.2 Klein LoRAs as Bonsai-compatible.
- Do not add a hidden unpacked fallback when a user selects the packed Bonsai model.
- Do not claim packed Bonsai LoRA support from SDXL-style or otherwise unrelated adapters.

## Implementation notes

Start with architecture, not adapter loading. The blocker is the execution boundary, not just file
parsing.
