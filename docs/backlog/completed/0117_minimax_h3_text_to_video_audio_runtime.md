# Completed: MiniMax-H3 text-to-video-with-audio runtime and Turbo adapters

## Metadata

- Created: 2026-09-04
- Status: Completed
- Completed: 2026-09-04

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md),
  [ADR 0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md).
- ADR impact: two follow-ups are owed and not yet written. (1) A generated joint audio-video
  contract ADR (`GeneratedAudio`, `GeneratedVideo.audio`, the AAC mux, the `audio_*` metadata
  fields, the sidecar-WAV fallback). (2) The license decision: the MiniMax H3 Community License
  excludes the EU development territory; the owner decided on 2026-09-04 to implement anyway and
  revisit the policy separately. Proposed item 0104 recorded the opposite gate and is deprecated
  as superseded by that decision. Until an ADR exists, this item is the decision of record.

## Context

MiniMax-H3 generates video and synchronized stereo audio from one packed sequence of text, audio
and video rows through a 27B dense transformer, conditioned on `hidden_states[50]` of
Qwen3-VL-32B. lightx2v publishes PEFT Turbo adapters that reduce the schedule to 8 (or 4)
transformer evaluations. The reference is the diffusers 0.40 modular pipeline (the official
repository ships no inference code).

## Shipped code reality

- Package `src/mflux/models/minimax_h3/`: packed layout and timestep plan
  (`latent_creator/h3_layout.py`), rectified-flow scheduler with a torch-faithful `linspace`
  (`scheduler/`), the transformer, the Qwen3-VL text stack truncated to 50 layers, the 3D-CNN /
  ViT video VAE, the DAC/BigVGAN audio VAE, weight definition and mappings, PEFT LoRA mapping,
  initializer, `MiniMaxH3` variant, and CLI (`mlxgen-generate-minimax-h3`).
- Catalog entries `minimax-h3`, `minimax-h3-turbo` (768p 8-step adapter, shifts 6/3) and
  `minimax-h3-turbo-544p` (mixed-aspect 8-step adapter, shifts 12/3); `mlxgen download` fetches the
  adapter with the snapshot; `mlxgen generate`, `mlxgen capabilities`, `mlxgen prepare` and the
  Python runtime route the family (`minimax-h3.generate`).
- `StreamingWeightLoader` (common): shard-by-shard load with quantize-as-you-go so the two 30B-class
  components never hold a full BF16 copy next to the q8 copy.
- `GeneratedAudio` + `GeneratedVideo.audio`; `VideoUtil.mux_generated_audio` (ffmpeg, PyAV fallback,
  sidecar WAV on failure) and the `audio_*` metadata fields.
- `h3_precision.disable_tf32()`: MLX 0.31 runs fp32 GEMM-class kernels in TF32 on M5 by default;
  the fp32 keep-set and the audio VAE need IEEE fp32.
- q8 policy: the AdaLN table projections (`adaln_proj`, `norm_out`) and the fp32 keep-set are never
  quantized (`TRANSFORMER_QUANTIZATION_SENSITIVE_FRAGMENTS`).

## Validation

- Layout / row timesteps bit-exact vs the reference for every geometry and anchor set; scheduler
  grids bit-exact on 600 (N, shift) combinations; `step` trajectories exact.
- Tiny-config parity vs diffusers: transformer 5 configs (batch 1 and 2, anchors) at torch's own
  fp32 noise; Qwen3-VL text stack 1e-8 incl. 3-axis M-RoPE; audio VAE 7.6e-5 / 3.3e-5 (decode /
  encode); video VAE encode 1.8e-4 (= torch fp32 vs fp64), decode 2.3e-5.
- Real weights: one-block transformer fp32 rel-rms 9e-6 / 6e-6 (video / audio), bf16 6e-3, q8
  1.3e-2 with the sensitive paths kept bf16 (1.17 when they were quantized, which produced pure
  noise end to end); Qwen3-VL layers 0-1 fp32 rel-rms 4e-6, bf16 4e-3; adapter keys 624/624 matched.
- Model-backed clips (M5 Max, q8, 124 frames): fox prompt at `960x544` Turbo 8 steps, seed 42,
  750 s total incl. the cold 133 GB load, peak MLX 80 GB; coherent motion and prompt-following,
  stereo track with L/R correlation 0.91. Sheets and MP4s in `docs/assets/minimax-h3/`.
- Tests: `tests/minimax_h3/` (scheduler literals, layout invariants, tiny end-to-end pipeline with
  audio mux, catalog/capability/predicate/LoRA-mapping contracts) and
  `tests/common/test_streaming_weight_loader.py`.

## Residual risks and follow-ups

- First-frame (FL2VA) and Ref2VA need the Qwen3-VL vision tower: planned item 0118.
- 768p costs ~5 min per step (fused attention efficiency drops at 37.8k rows); AdaLN
  precompute-and-drop would free ~26 GB: proposed item 0119.
- No published q8 package yet (a prepared package needs 62 GB of disk).
- MLX 0.31 `quantized_matmul` with bf16 scales is wrong for inputs of 32768 or more rows (the first
  `M - 32768` rows collapse to ~0); the 768p canvas packs 37,851 rows. `h3_precision.rowwise` applies
  the block's q8 linears in row chunks; any other q8 route that crosses 32768 tokens needs the same
  guard (Wan 1280x704x121 is ~30k tokens).
- Adversarial verification artifacts and MLX precision probes live in `untracked/h3_verify/`.
