# Completed: MiniMax-H3 first-frame conditioning through the Qwen3-VL vision tower

## Metadata

- Created: 2026-09-04
- Status: Completed
- Completed: 2026-09-05

## Completion report - 2026-09-05

- Shipped: `qwen3_vl_vision_model.py` (image processor with Qwen2-VL `smart_resize` and merge-block
  patch order, patch embedding, bilinear align-corners position-grid resampling, 2-axis rope, 27
  blocks, main and DeepStack mergers), `qwen3_vl_model.py` (`rope_index`, image-token scatter,
  DeepStack injection at text layers 0-2, `encode`), text-encoder weight mapping/loading for
  `model.visual.*`, the `image_path` branch of `MiniMaxH3.generate_video` (keyframe canvas from its
  aspect ratio with per-entry `canvas_short_edge`/`canvas_max_pixels`, LANCZOS stretch, `<Picture 1>`
  presentation with video-tagged vision rows, seed-42 posterior sample + fp16 rounding + normalization,
  `scale_noise(0.999)` with the condition noise drawn first, anchored layout, condition rows never
  stepped), the `minimax-h3.first-frame` capability row and the CLI/router `--image-path` path.
- Validation: tiny-config parity vs transformers 5.16 exact (vision 4e-8, rope index identical,
  image-conditioned encode 6e-8); real weights fp32: processor 6e-8, vision merged 1.2e-5, DeepStack
  features ~1e-5, layers 0-2 with image 9e-6 (compared at the same DeepStack point: torch records a
  layer's output before that layer's DeepStack add). Model-backed clip: fox keyframe through
  `minimax-h3-turbo-544p`, frame 0 at PSNR 32.3 dB to the keyframe, prompted motion follows, 735 s.
  Tests: `tests/minimax_h3/test_h3_vision.py`, image-to-video tiny pipeline test, capability test.
- Non-goals kept: Ref2VA and the `last_image` closing keyframe (cover-crop path) remain unported.
- Adversarial pass (independent scripts under `untracked/h3_verify/agent2/`): vision tower and
  processor exact vs transformers (tiny 5e-7, real fp32 1.3e-5), presentation/rope positions
  bit-exact, condition latents deterministic part 1.9e-5 (the remaining 3.7e-2 is the irreproducible
  torch posterior draw, measured at exactly sqrt(2) of a single draw), layout and per-row timesteps
  bit-exact. It found two quantization-only defects, both fixed before release: the vision patch
  embedding and the transformer's `context_embedder` cast their activations to the packed uint32
  weight dtype under q8 (`h3_precision.linear_input_dtype` now resolves the compute dtype;
  regression test in `tests/minimax_h3/test_h3_precision.py`).
- Model-backed proof rows after the quantization fixes (all `minimax-h3-turbo-544p`, q8, seed 42,
  sheets and prompts in `docs/assets/minimax-h3/`): starship keyframe (`768x432` -> `960x544`) frame 0
  at PSNR 29.4 dB, hull details kept, continuous liftoff out of frame, engine track >80% of energy
  below 200 Hz; square takeoff keyframe (`512x512` -> `544x544`) PSNR 26.5 dB, continuous liftoff,
  motion/audio-energy correlation 0.71. Same seed with the earlier prompt ("lifts off slowly, hovers
  for a moment, then climbs") gave a late partial ascent, so the docs now carry motion-prompt
  guidance (one continuous action with its end state, name what must stay unchanged, continuous
  soundscape, explicit exclusions).
- Loader follow-ups landed in the same pass: prepared packages stream shard by shard through
  `StreamingWeightLoader._load_prepared` (the whole-component path held the 75 GB package twice and
  swapped on a 128 GB Mac), and the streaming loader applies the once-per-process MLX cache-limit
  default (`RuntimeMemory.apply_default_cache_limit_once`) that `WeightLoader` already applied; the
  MiniMax-H3 CLI exposes `--mlx-cache-limit-gb`. Without the cap the 544p run held 27 GB of free
  cache above its 82 GB of active MLX memory (110 GB footprint); with it the 768p Turbo and the base
  50-step runs peaked at 93 GB and 87 GB footprints at unchanged speed.
- `--base-model` now reaches the MiniMax-H3 backend: the router's base-model gate did not list the
  `mlxgen-generate-minimax-h3` route and the backend parser had no such flag, so a prepared package
  silently ran as the base entry (50 steps at `1344x768`, hours instead of minutes). Router and CLI
  tests pin the forwarding and the resolution (`_resolve_model`), and the CLI announces which entry a
  package or repo id runs as.
- Second adversarial pass (2026-09-05, `untracked/h3_verify/agent3/`) on a laptop lid that changed
  orientation over four frames in a first-person image-to-video clip: verdict model behavior. Frame
  order and timestamps monotone; the video VAE's temporal chunk assembly (5-token chunks, overlap 2,
  five-frame cross-fade) reproduced against the reference with a mock decoder to 0.0; packed layout,
  rope positions and per-row timesteps at the clip's real geometry bit-exact; no systematic spike at
  chunk boundaries across 13 clips; the change is carried in latent tokens 26 to 27.
- Tokenizer: the Qwen2 workaround in `TokenizerLoader` dropped the seven H3-only special tokens that
  `tokenizer_config.json` declares (`<d>`, `</d>`, `<|cutoff|>`, lyric and caption markers), so
  dialogue tags tokenized as three ordinary tokens instead of ids 151669+. The workaround now adds
  `additional_special_tokens` the way `from_pretrained` does; encodings of the H3 and Qwen-image
  tokenizers are identical to transformers on dialogue, vision and plain text (unit test in
  `tests/minimax_h3/test_h3_config_and_weights.py`).
- Residual deviations recorded, not fixed: the keyframe is encoded by the bf16 video VAE while the
  reference pins the VAE to fp32 (condition mean rel-rms 2e-2 vs 2e-6; a 10 GB fp32 VAE copy would
  remove it); `MIN_PIXELS`/`MAX_PIXELS` are hardcoded to the released processor values rather than
  read from `processor/preprocessor_config.json`; the reference's `logvar` clamp to [-30, 20] is not
  applied (measured range [-11.4, -2.9], so it is currently inert).

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0007](../../adr/0007_role_aware_reference_conditioning_and_factored_model_sources.md).
- ADR impact: none beyond the joint audio-video contract ADR owed by item 0117.

## Summary

Ship MiniMax-H3 `image-to-video` (the reference's FL2VA / I2VA task): one keyframe conditions both
the latent rows and the text sequence.

## Current code reality

- The layout already reserves keyframe condition rows (`keyframe_anchors`), and
  `build_row_timesteps` pins them at the 0.999 noise-augmentation level; the scheduler's
  `scale_noise` and the video VAE encoder (single frame, seed-42 posterior sample, fp16 rounding,
  latent normalization) are ported and verified.
- `MiniMaxH3.generate_video(image_path=...)` raises `NotImplementedError` because the reference also
  inserts `"<Picture i>: "` plus the keyframe's vision tokens into the Qwen3-VL sequence, and the
  vision tower (patch embedding, 2D rotary, window attention, deepstack merger injected at text
  layers 0-2) is not ported. `Qwen3VLTextModel.__call__` already accepts `deepstack_visual_embeds`.

## Scope

1. Port the Qwen3-VL vision encoder and the deepstack merger; parity vs transformers on a real
   image (tiny config exact, real weights at bf16 noise).
2. Processor: keyframe resize policy of the reference pipeline, `<Picture 1>: ` label, image token
   placement, 3-axis M-RoPE positions for image tokens.
3. Pipeline: condition latents (encode seed 42, noise-aug 0.999), condition rows first in the packed
   sequence, FL2VA prompt presentation, capability row `minimax-h3.first-frame`, CLI `--image-path`.
4. Model-backed proof: same-seed T2VA vs FL2VA contact sheets on one keyframe.

## Non-goals

Ref2VA (reference images / videos / audio, the `Ref2VA` Turbo adapters) and the Context-IR API.

## Validation

Tiny and real-weight parity for the vision tower; an FL2VA clip whose first frame matches the
keyframe (PSNR against the source) and whose motion follows the prompt.
