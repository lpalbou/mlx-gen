# Proposed: Wan memory defaults — streamed decode and per-item transformer release

## Metadata

- Created: 2026-07-22
- Status: Implemented (streamed decode default AND per-item A14B transformer
  release+reload shipped 2026-07-22; real-checkpoint validation pending)
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: None (defaults change with identical outputs; record in CHANGELOG)

## Context

2026-07-22 audit findings in 0.23.1:
- The default (non low-ram) Wan decode materializes the full video tensor in one
  array (`wan2_2_ti2v.py` ~486-492) — ~650 MB bf16 for 121f@1280x704 plus latents —
  while the streamed slice decoder (`wan_2_2_vae.py` ~312-347) is gated behind
  `--low-ram` AND single-seed (`wan_generate.py` ~66). PIL conversion is already
  per-8-frame batch (`video_util.py` ~1226-1235), so streaming by default should
  cost nothing.
- Multi-seed A14B batches keep BOTH 14 GB transformers resident for the whole run:
  `release_inactive_denoiser = single_seed and has_transformer_2`
  (`wan_generate.py` ~62-65) — the high-noise transformer is never released after
  its phase when `--seed a b c` is used.

## Problem or opportunity

~1 GB avoidable peak on every default Wan run; ~14 GB avoidable residency on batched
A14B runs — the difference between fitting and swapping on 36-48 GB machines.

## Proposed direction

- Make streamed VAE decode the default whenever the writer consumes frames in
  batches; keep the all-at-once path only where a consumer truly needs the full
  tensor.
- Release the inactive A14B transformer per batch item and reload from the OS page
  cache (measure the reload cost — page-cache hits should make it cheap); or key the
  behavior on a memory budget.

## Implementation record (2026-07-22): streamed decode default (review item e3)

- `Wan2_2_TI2V.generate_video` now ALWAYS builds the frame-batches factory via
  `iter_decode_normalized_latent_slices`; the full-tensor decode branch is
  deleted. Decode-mode is decoupled from denoiser release:
  `release_denoisers_before_decode` semantics are unchanged, and per-slice cache
  flushes follow `clear_cache_each_step` (only low-ram pays them).
- Outputs are BITWISE identical to the removed full decode (pinned by a tiny
  random-weight parity test against the still-shipped non-streamed
  `decode_normalized_latents`, which WanVace keeps using - VACE decode is
  explicitly out of scope here).
- Second-order win: `python_runtime.generate_outputs` results no longer retain
  ~650 MB decoded tensors per item; the factory holds only latents (~3.5 MB at
  121f@1280x704 bf16).
- Decode-health surfacing moved with the decode: a non-finite VAE output now
  fails at frame materialization (save/first_frame) through the per-frame finite
  checks, not inside generate_video (test-pinned).
- `wan_decode_mode: streamed_vae_slices` and `generation_time_scope: pre-save`
  metadata extras now appear on ALL runs (generation_time no longer includes the
  VAE decode, which runs at save).
- The per-item A14B transformer release (review item e4) remains OPEN.

## e4 implementation design (2026-07-22 adversarial prioritization — for the
## implementer)
Store a reload spec at init (root_path, weight_definition, quantize arg,
per-role LoRA paths/scales — all already on the model from WanInitializer);
add `_ensure_high_noise_denoiser()` invoked from the expert-selection point
when the transformer is None and a reload spec exists: re-run the
per-component WeightLoader load + apply_and_quantize for `transformer` +
re-apply the high_noise_transformer-role LoRAs. Keep the existing raise for
the un-reloadable case (release_denoisers_before_decode). ALSO unify the
release default in generate_video itself: the Python API (python_runtime
generate_outputs) never sets release_inactive_denoiser today, so embedding
hosts get ZERO release even single-seed — moving the default into the model
(on when has_transformer_2 and reloadable) is the one-source-of-truth fix,
with an opt-out kwarg/flag. Reload cost reasoned: q8 mmap page-cache-warm
~0.3-0.7 s, cold ~2-3 s at SSD speeds — <1% of a multi-minute A14B item.
GATE runtime-quantize users OUT of auto-release (a --quantize-over-bf16
reload re-quantizes 14B per item, tens of seconds) — auto only for
disk-prequantized checkpoints. PRIMARY RISK: LoRA re-application on reload
(the owner's Lightning storyboard runs LoRAs on BOTH experts) — needs a
dedicated tiny-weights reload test (reuse test_wan_lora_mapping's
_TinyWanTransformer) before any default change. Interaction with the d12
compile flag: the compiled-callable cache must drop the high expert's entry
on release and rebuild on reload (see the release-interaction note in the
0090 status).

## Implementation record (2026-07-22): per-item release + reload (review item e4)

Implemented per the design above; decisions and deviations:

- Reload spec: `root_path`, `weight_definition`, and resolved LoRA
  paths/scales/roles were already on the model; the initializer now ALSO keeps
  `model.quantize_arg` (the original request - `model.bits` only records the
  resolved level) and `model.transformer_stored_q_level` (the transformer
  component's on-disk q level, the disk-prequantized signal the auto default
  keys on).
- `WanInitializer.reload_high_noise_transformer(model)` rebuilds ONLY the high
  expert: per-component `WeightLoader._load_component` + the same q8
  normalization/validation as init + `apply_and_quantize_single` with the
  original quantize arg, then re-fuses high-role LoRAs in original order. A
  reload that resolves different bits than init FAILS LOUDLY (checkpoint
  drifted on disk). Test-pinned bitwise on a tiny REAL WanTransformer
  (fuse -> reload -> re-fuse output equality, low-role files skipped).
- Reload is LAZY at the expert-selection point, not at generation start: V2V
  runs whose strength starts below the boundary never pay the rebuild. Config
  probes (patch multiples, channel contract) go through `_reference_denoiser()`
  (both experts share construction kwargs) so request resolution works while
  the high expert is absent.
- Default unified IN THE MODEL: `release_inactive_denoiser: bool | None = None`;
  None = auto (ON when dual-expert AND `transformer_stored_q_level` is not None
  AND the reload spec exists). The CLI forwards explicit user intent from
  `--release-inactive-denoiser` / `--no-release-inactive-denoiser`; when no
  flag is given, SINGLE-seed dual-expert CLI runs forward True (the pre-0089
  rule restored by the cycle-2 review — the process exits after one item, so
  releasing costs nothing regardless of quantization) and multi-seed runs
  forward None (model-owned auto). bf16-on-disk packages are gated out of
  auto (28 GB re-read per reload), not only runtime-quantize — opt in
  explicitly.
- Which released states stay fatal: `release_denoisers_before_decode` (low
  expert gone - never reloadable) and a released high expert WITHOUT a reload
  spec keep the existing raise.
- d12 compile interaction: release still pops the high expert's compiled entry;
  after a reload the denoise loop traces a FRESH callable against the reloaded
  module (lazy rebuild, one extra trace per item). Test-pinned.
- Metadata truth: `released_inactive_denoiser: true` and `high_noise_reloads: N`
  appear only when the behavior actually fired in that run.
- The q8-normalization notice (`wan_initializer` load path) prints once per
  reload = once per batch item; accepted as truthful load output, plus one
  explicit "Reloading Wan high-noise transformer..." line so the pause is
  attributable.

NEEDS REAL-CHECKPOINT VALIDATION (owner, per AGENTS.md):

1. One multi-seed A14B run (2+ seeds, disk-prequantized q8 package) with
   whole-process peak-RSS measurement, plus the observed per-item reload wall
   time (expected ~0.3-0.7 s page-cache-warm, ~2-3 s cold).
2. One Lightning-LoRA storyboard-profile run (LoRAs on BOTH experts,
   high_noise_transformer/low_noise_transformer roles) verifying re-fused
   output identity across items (same seed twice: item 1 vs a fresh process).

## Implementation record (2026-07-23): A14B i2v condition precision build (F2)

`_load_video_condition` built first_frame + zero_frames in float32 and
concatenated BEFORE the precision cast — a ~1 GB f32 transient at 1280x720x81
(~3 GB at 1920x1080x121) that the cast immediately halved. The padded
VAE-encode input is now built directly in `ModelConfig.precision`
(`Wan2_2_TI2V._build_first_frame_video_condition`): normalization stays in
float32, the single frame is cast first, and the zero padding is allocated in
the target dtype. The VAE encode input is BITWISE identical (elementwise cast;
zeros cast exactly), pinned by a tiny random-weight VAE encode parity test
(`tests/wan/test_wan_vae.py::test_wan_i2v_precision_condition_build_is_bitwise_identical_to_f32_concat_cast`).
Real-checkpoint validation item: none needed for output correctness (bitwise);
a peak-RSS spot check on one A14B i2v run would confirm the transient claim.

## Follow-up note (2026-07-22): video writer HD color coding

The 0.25-track writer fix pins and truthfully tags the BT.601 matrix ffmpeg was
already applying at all resolutions (metadata-only, bitwise-identical pixels).
The remaining option — switching >=720p output to a REAL BT.709 encode, matching
what HD players assume by default — would change pixel data (non-bitwise,
seed-stability caveat) and needs a visual A/B before any default flip.

## Why it might matter

Same outputs, lower peaks, especially for the machines most likely to OOM.

## Promotion criteria

Promote when Wan A14B batch usage is real (embedding hosts expose multi-video
batches today) or when a smaller-RAM user reports pressure.

## Validation ideas

Peak-RSS measurement harness before/after (whole-process, per AGENTS.md); bitwise or
threshold-identical outputs; reload-cost timing for the per-item release.

## Non-goals

Changing low-ram semantics; touching image-family decode paths.

## Guidance for future agents

Follow AGENTS.md: physical process memory for user-facing claims; distinguish
storage vs runtime improvements.
