# Proposed: MiniMax-H3 adapter layouts, validated one real file per producer

## Metadata

- Created: 2026-09-09
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md).
- ADR impact: none. A layout that loads must load the way its producer trained it; a layout we
  cannot verify is refused, not guessed.

## Context

On 2026-09-09 the MiniMax-H3 LoRA mapping grew the original checkpoint's module names next to the
diffusers ones, because a civitai adapter (ai-toolkit, `diffusion_model.blocks.N.attn.qkv_proj`)
did not match a single key. The translation (fused QKV thirds, `fc1` half swap, `blocks.` renames,
kohya `lora_down` / `lora_up` / `.alpha`, musubi-tuner's flattened `lora_unet_` names) follows the
diffusers 0.40 converter and is unit-tested against the reference fused projections. Only one
producer has been validated end to end with a real file: ai-toolkit (`H3_Skeletor_1.4`, rank 8,
stacked on the 544p Turbo adapter, same-seed comparison).

## Current code reality

- `MiniMaxH3LoRAMapping` maps both layouts; `normalize_state_dict` un-flattens musubi keys.
- `MiniMaxH3Initializer.adapter_scale` reads PEFT metadata `alpha`, leaves kohya `.alpha` tensors
  to the loader, and runs a PEFT file without either at `alpha == rank`.
- DiffSynth-Studio adapters (PEFT `.default.` infix over `attn.qkv_proj`) are refused: their fused
  QKV rows keep the raw checkpoint's per-head interleaving, which the diffusers converter
  de-interleaves with a 128-wide head, and no real file has exercised that permutation here.
- The loader applies one `state_dict_transform` to every file of a call, so a per-file layout flag
  (which DiffSynth support needs, because the infix is stripped before the transform runs) would
  mean applying files one call at a time in the initializer.

## Scope

1. One real adapter per remaining producer (kohya / musubi-tuner, DiffSynth-Studio, a reference
   `generate.py` export), each loaded with zero unmatched keys and compared same-seed against the
   Turbo-only clip.
2. DiffSynth's de-interleave, ported from the converter, gated on a real file reproducing its
   character; until then the refusal stands.
3. A `lora_layout` value in the application report naming which layout matched, so a host can show
   it without parsing keys.

## Non-goals

Ref2VA adapters (`transformer_ref` partition): the same mapping applies once that route exists.

## Validation

Each producer's file: unmatched key count 0, the same-seed clip differs from the Turbo-only clip in
the trained subject and not in layout artifacts, and a second seed shows the same subject.
