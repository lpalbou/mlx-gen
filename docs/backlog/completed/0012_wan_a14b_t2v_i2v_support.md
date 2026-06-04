# Completed: Wan2.2 A14B T2V/I2V support

## Metadata

- Created: 2026-05-30
- Status: Completed
- Completed: 2026-05-31

## ADR status

- Governing ADRs:
  [ADR 0001: Runtime Smoke Validation For Model Routes](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0002: No Silent Automatic Fallbacks](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: This item triggered ADR 0001 after a real A14B run showed that mock/shape tests were
  not enough to prove the route. It also triggered ADR 0002 after Wan resolution exposed a silent
  default-config fallback risk. Future completion claims must include a model-backed smoke proof,
  and ambiguous Wan models must fail closed unless the caller explicitly supplies a supported
  model identity.

## Context

The local Hugging Face cache contains `Wan-AI/Wan2.2-T2V-A14B-Diffusers` and
`Wan-AI/Wan2.2-I2V-A14B-Diffusers`. These A14B repositories are not the same architecture as the
existing `Wan-AI/Wan2.2-TI2V-5B-Diffusers` support in MLX-Gen.

## Current code reality

- MLX-Gen exposes separate Wan configs for TI2V-5B, T2V-A14B, and I2V-A14B.
- `Wan-AI/Wan2.2-T2V-A14B-Diffusers` has two 40-layer transformers (`transformer` and
  `transformer_2`), a high/low-noise boundary ratio of `0.875`, scalar batch timesteps,
  `flow_shift=3.0`, 16-channel latents, and the Wan2.1-style VAE with spatial scale 8.
- `Wan-AI/Wan2.2-I2V-A14B-Diffusers` uses the same two-transformer A14B routing with boundary
  ratio `0.9`, but its transformer input is 36 channels because source-image conditioning is
  concatenated with denoising latents rather than applied through the TI2V expanded-timestep mask
  path.
- The local T2V-A14B and I2V-A14B caches resolve with tokenizer, text encoder, VAE,
  `transformer`, and `transformer_2`.
- The Wan weight definition is config-driven and includes `transformer_2` when the selected model
  requires it.
- Diffusers exposes an optional second low-noise guidance value for A14B, but defaults it to the
  regular guidance value when it is omitted. MLX-Gen should mirror that behavior with optional
  `--guidance-2` for A14B only.
- A failed user run showed the exact failure this item must prevent: if A14B is not recognized and
  falls through to the TI2V-5B runtime, generation can load A14B patch-embedding weights while
  preparing 48-channel TI2V latents, then fail with `input: (1,6,32,32,48)` versus
  `weight: (5120,1,2,2,16)`.

## Problem

Treating A14B as an alias for TI2V-5B would silently build the wrong transformer, scheduler,
timestep, and VAE shape. A correct port needs model-specific configuration while preserving the
existing 5B path.

## Completed work

Added A14B T2V and A14B I2V support in the same Wan backend, using model config as the source of
truth for transformer dimensions, VAE variant, scheduler shift, timestep mode, boundary routing,
and image-conditioning mode.

## Why

A14B is the higher-capacity Wan text-to-video/image-to-video family. Supporting it gives MLX-Gen a
better path for serious local video generation and keeps AbstractVision from depending on
Diffusers/PyTorch for Wan A14B workflows on Apple Silicon.

## Requirements

- Add separate model configs for T2V-A14B and I2V-A14B.
- Instantiate Wan transformers and VAE from model config instead of 5B defaults.
- Load and save `transformer_2` when the selected model requires it.
- Select `transformer` versus `transformer_2` by Diffusers-compatible boundary timesteps.
- Add optional `guidance_2` CLI/API plumbing for A14B low-noise guidance; default to the regular
  guidance value when it is omitted.
- Preserve TI2V-5B expanded-timestep and first-frame mask behavior.
- Reject source-image input for T2V-only A14B and require the complete I2V A14B source snapshot
  before generation.

## Scope

- Wan2.2 T2V-A14B text-to-video support.
- Wan2.2 I2V-A14B image-to-video support with concatenated image-condition latents.
- Focused unit and CLI tests that do not require a two-hour full-quality run.

## Non-goals

- Do not claim q8/q4 A14B quality or speed until separate quantized validations exist.
- Do not implement last-frame interpolation or prompt-extension support in this item.
- Do not automatically download gated or very large A14B weights during generation.
- Do not remove existing TI2V-5B behavior or model aliases.

## Dependencies and related tasks

- [Wan quantization and motion parity](0002_wan_quantization_motion_parity.md)
- [Wan q8 performance investigation](0005_wan_q8_performance_investigation.md)
- Local Diffusers reference: `/Users/albou/projects/gh/diffusers/src/diffusers/pipelines/wan/`
- Local Diffusers VAE reference:
  `/Users/albou/projects/gh/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py`

## Expected outcomes

- `mlxgen generate --model Wan-AI/Wan2.2-T2V-A14B-Diffusers ...` resolves to an A14B config and
  uses A14B transformer/VAE/scheduler defaults.
- A14B weight loading includes `transformer_2`.
- The same Wan backend can distinguish TI2V-5B, T2V-A14B, and I2V-A14B without command-name
  proliferation.
- Docs show practical A14B T2V and I2V commands and the current validation limits.
- Unrecognized remote Wan Hugging Face repos and ambiguous local Wan paths fail with a clear
  config-resolution error instead of silently using a default runtime shape.

## Validation

- `uv run pytest tests/wan/test_wan_a14b_config.py tests/cli/test_mlx_gen_router.py -q`
- `uv run pytest tests/wan/test_wan_scheduler_and_timesteps.py tests/wan/test_wan_progress.py -q`
- `uv run ruff check src/mflux/models/wan src/mflux/models/common/config/model_config.py tests/wan`
- Optional local source smoke after enough memory/time is available:
  `uv run mlxgen generate --model Wan-AI/Wan2.2-T2V-A14B-Diffusers --task text-to-video --prompt
  "A slow cinematic shot of teal water" --width 256 --height 256 --frames 5 --steps 2 --guidance
  1 --guidance-2 1 --fps 8 --seed 1 --output validation_outputs/wan_a14b_smoke.mp4`

## Recent validation

- `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run pytest tests/wan/test_wan_a14b_config.py
  tests/wan/test_wan_scheduler_and_timesteps.py tests/wan/test_wan_progress.py
  tests/wan/test_wan_vae.py tests/wan/test_wan_transformer.py tests/cli/test_mlx_gen_router.py
  tests/cli/test_prepare_save.py -q` passed on 2026-05-30.
- `uv run ruff check src/mflux/models/common/config/model_config.py src/mflux/models/wan
  src/mflux/models/fibo/model/fibo_vae/common/wan_2_2_resample.py src/mflux/cli/mlx_gen.py
  tests/wan/test_wan_a14b_config.py tests/cli/test_mlx_gen_router.py` passed on 2026-05-30.
- A VAE-only local-cache check loaded the T2V-A14B VAE weights, decoded a 16-channel latent to
  shape `(1, 3, 1, 64, 64)`, and encoded a 9-frame 64x80 video condition to
  shape `(1, 16, 3, 8, 10)`.
- I2V-A14B conditioning shape was validated in unit tests as `(1, 20, 3, 8, 10)` for a 9-frame
  64x80 source-image run.
- Real T2V-A14B model-backed smoke validation passed on 2026-05-30:
  `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run mlxgen generate --model Wan-AI/Wan2.2-T2V-A14B-Diffusers
  --task text-to-video --prompt "A cinematic shot of mist rolling across a teal mountain lake"
  --width 128 --height 128 --frames 5 --steps 1 --guidance 1 --fps 8 --seed 1 --output
  validation_outputs/wan_a14b_t2v_smoke_128_5f_1step.mp4 --metadata`.
  `ffprobe` reported `width=128`, `height=128`, `r_frame_rate=8/1`, `nb_frames=5`.
- Real two-stage T2V-A14B model-backed smoke validation passed on 2026-05-30:
  `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run mlxgen generate --model Wan-AI/Wan2.2-T2V-A14B-Diffusers
  --task text-to-video --prompt "A cinematic shot of mist rolling across a teal mountain lake"
  --width 128 --height 128 --frames 5 --steps 2 --guidance 4 --fps 8 --seed 2 --output
  validation_outputs/wan_a14b_t2v_smoke_128_5f_2steps_guidance4.mp4 --metadata`.
  `ffprobe` reported `width=128`, `height=128`, `r_frame_rate=8/1`, `nb_frames=5`. This is
  wiring validation only, not quality validation.
- Added fail-closed coverage for unrecognized remote Wan repos so they cannot silently run through
  the TI2V-5B default config.
- Real I2V-A14B model-backed validation passed on 2026-05-31:
  `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run mlxgen generate --model Wan-AI/Wan2.2-I2V-A14B-Diffusers
  --task image-to-video --image docs/assets/i2v_takeoff_source.png
  --prompt "A spacecraft lifts off from a snowy landing field" --width 384 --height 224 --frames 17
  --steps 12 --guidance 3.5 --fps 8 --seed 321 --output
  validation_outputs/wan/wan_a14b_i2v_takeoff_prompt_defaults_384x224_17f_12steps.mp4
  --metadata`. The MP4 frame-extraction contact sheet showed coherent non-green frames and source
  image conditioning.

## Progress checklist

- [x] Add A14B model configs and aliases.
- [x] Make Wan weight definitions dynamic and include `transformer_2`.
- [x] Make Wan model construction config-driven.
- [x] Add Wan2.1-style A14B VAE support.
- [x] Add boundary routing and optional `guidance_2`.
- [x] Add A14B image-conditioning shape support or explicit source-snapshot gate.
- [x] Add focused tests.
- [x] Update public docs and LLM indexes.
- [x] Run a real A14B T2V smoke generation and preserve output evidence.
- [x] Validate A14B I2V generation after the source snapshot is complete.
