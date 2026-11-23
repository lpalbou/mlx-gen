# Completed: MiniMax-H3 quantized packages on Hugging Face

## Metadata

- Created: 2026-11-22
- Status: Completed
- Completed: 2026-11-23

## ADR status

- Governing ADRs:
  [0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md) (model-backed proof before a
  route is described as working),
  [0002](../../adr/0002_no_silent_automatic_fallbacks.md) (no silent change of the requested
  quantization policy; a package that cannot be loaded as written is refused).
- ADR impact: none. The rule that a prepared package loads at the widths stored in its files is
  implemented for the streaming loader only; item
  [0126](../planned/0126_wan_q8_keep_8bit_at_runtime.md) decides whether it becomes an ADR for
  every family.

## Completion report - 2026-11-23

### What shipped

- `AbstractFramework/minimax-h3-8bit` (64.3 GiB): 8-bit transformer, AdaLN projections included,
  and 8-bit conditioner language layers.
- `AbstractFramework/minimax-h3-4bit` (44.9 GiB): 4-bit conditioner language layers, 6-bit
  transformer. `--quantize 4` selects this mixed policy.
- Both store the Qwen3-VL vision tower, the fp32 heads and timestep MLP, the final modulation and
  the two VAEs as released, and carry the source tokenizer files, `LICENSE`, `NOTICE` and the
  scheduler configs. Both repositories are gated; the owner chose gated publication on 2026-11-22
  after reading the license's territory clause.
- Release 0.39.0, which the packages require.

### Why the small package is not a 4-bit/8-bit mix

The owner expected the usual recipe: 4-bit with the sensitive layers at 8-bit. The measurement
(`tools/minimax_h3_quant_probe.py`, five real transformer inputs, error of the predicted velocity
against the BF16 weights) shows MiniMax-H3 has no sensitive family to protect:

- any one of the five block-linear families at 4-bit costs 8% to 24% video error on its own, and all
  of them 17% to 49%;
- a 4-bit/8-bit mix at the size of uniform 6-bit (feed-forward input and attention output at 4-bit)
  is at 13% to 25%, against 6% to 15% for uniform 6-bit; uniform 5-bit, which is smaller, is also
  ahead of the mixes;
- keeping the first and last ten blocks at 8-bit recovers little.

Same-seed clips agree: against the 8-bit clip the 6-bit transformer stays closest, then the mix,
then 4-bit. Every policy produced a coherent clip that speaks the scripted line; the argument is
fidelity to the released weights at equal size, not visible breakage. The conditioner is where
4-bit is cheap: 5.0% to 12.4% velocity error against 3.3% to 7.4% at 8-bit, for 11.7 GiB.

### Findings that changed the 8-bit policy

- **The AdaLN projections were never quantization-sensitive.** Item
  [0117](0117_minimax_h3_text_to_video_audio_runtime.md) kept them BF16 (24 GiB of the 75 GiB
  package) after 8-bit there produced noise. At that time `H3AdaLayerNormModulation` cast its input
  to `weight.dtype`, which is the packed `uint32` on a quantized linear, so `silu(temb)` was
  truncated to integers. Reproduced on the real weights: with that cast the modulation tables are
  57% to 73% off; with the compute dtype (the cast used since 0.37.0) 0.01% to 0.40%. Quantizing
  them at 8-bit adds at most 2.1 points to the 8-bit video error and removes 11.3 GiB.
- **The vision tower must not be quantized.** At any width a few very-high-norm image tokens flip.
  With an 8-bit tower the 90th-percentile per-token error of an image-conditioned sequence is 26.6%
  against 2.1% with the tower as released, and the image-to-video velocity error is 6.2% against
  3.5%. The tower is 0.6B parameters. This came from the adversarial review, which asked for the
  variant that isolates it.

### Defects found on the way, all fixed with tests

- A package downloaded from Hugging Face did not resolve from the cache: path resolution required
  scheduler configs a prepared package did not carry. Packages had only ever been run from a local
  folder.
- The local package built during item 0117 had lost the dialogue markers (`<d>` tokenized as three
  pieces): its tokenizer had been re-serialized by a loader that did not yet add the config-only
  markers. Packages now copy the source tokenizer files, and loading refuses a tokenizer that
  splits a marker.
- The LoRA wrapper derived a quantized layer's input width as `columns * (32 // bits)`, wrong at 3,
  5 and 6 bits.
- The generated model card described a text-to-image checkpoint.
- From the review: the loader accepted a package missing a module, with a tensor of another shape,
  or written at another group size; the memory preflight was wrong for `--quantize 4` and for
  packages; `prepare` only warned about a missing `LICENSE`, and would bake a Turbo adapter into a
  package described as base weights.
- From the documentation pass: `bernini-r-1.3b-bf16` could not be routed, and a missing published
  package suggested preparing a package from a package.

### Evidence

- Reports and clips: `docs/assets/quantization/minimax-h3/`; tables in
  [Quantization](../../quantization.md#minimax-h3-q8-and-mixed-q4), all generated from the report
  files.
- Each package reproduces the clip of its load-time policy byte for byte (dialogue, fox and
  image-to-video at `640x352`), and `--quantize 4` / `--quantize 8` through the shipped policy
  reproduce the package clips.
- Memory: see the memory tables in [MiniMax-H3](../../minimax-h3.md#memory-by-machine), measured at
  `960x544`, 124 frames, with the smaller tiers tested by locking RAM away on the 128 GB machine.
- The uploaded 8-bit repository was checked file by file against the local package (sizes, and
  SHA-256 for the 36 weight files).

### Method notes worth keeping

- Rehearse a policy by quantizing at load time from the source snapshot with a per-family width
  assignment; a package is only worth building once the policy is settled. A warm source load with
  quantization takes seconds, and `mlxgen prepare` about a minute.
- Whole-tensor error of the conditioner output is dominated by a handful of tokens whose norm is
  fifty times the median. Report per-token medians and percentiles, and measure the conditioner
  through the transformer.
- The first transformer captures came from the package with the broken tokenizer and had to be
  redone; capture from the artifact that will ship.
- Timings taken while other work shares the machine are not usable; memory figures are.

## Follow-ups

- [0129](../proposed/0129_router_and_docs_followups_from_the_0_39_0_documentation_pass.md): router
  messages and docs found wrong during the documentation pass.
- [0119](../proposed/0119_minimax_h3_adaln_precompute_and_long_sequence_performance.md): the AdaLN
  precompute now frees 12.9 GiB at 8-bit rather than 26 GB, which changes its priority.
- The license policy decision recorded in
  [0104](../deprecated/0104_minimax_h3_joint_audio_video_watch.md) is settled for distribution by
  the owner's choice of gated repositories; the joint audio-video contract ADR is still owed.
