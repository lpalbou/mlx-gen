# ERNIE-Image-Turbo MLX Port Plan

## Goal

Add first-class `baidu/ERNIE-Image-Turbo` support to MLX-Gen by porting the Hugging Face Diffusers/Transformers implementation to MLX, keeping behavior traceable to upstream source files and compatible with the existing MLX-Gen `prepare`, `generate`, model-card, and local-download policies.

This is a real model port, not a config alias. ERNIE-Image-Turbo combines:

- Diffusers `ErnieImagePipeline`
- Diffusers `ErnieImageTransformer2DModel`
- Diffusers `AutoencoderKLFlux2`
- Transformers `Mistral3Model` as text encoder
- Transformers `Ministral3ForCausalLM` as optional prompt enhancer
- `FlowMatchEulerDiscreteScheduler` with ERNIE-specific shifted sigmas

## Constraints

- Follow the upstream implementations line by line where practical:
  - `/Users/albou/projects/gh/diffusers/src/diffusers/pipelines/ernie_image/`
  - `/Users/albou/projects/gh/diffusers/src/diffusers/models/transformers/transformer_ernie_image.py`
  - `/Users/albou/projects/gh/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py`
  - `/Users/albou/projects/gh/transformers/src/transformers/models/mistral3/`
  - `/Users/albou/projects/gh/transformers/src/transformers/models/ministral3/`
- Do not auto-download model weights from `generate`; only `download` and `prepare` should perform network access.
- Keep existing Qwen, FLUX.2, FIBO, and Z-Image behavior unchanged.
- Prefer a minimal text-to-image port first. Defer image-conditioned Pixtral processor support until a pipeline actually needs image tokens.
- Treat `use_pe=False` as the initial baseline if the prompt enhancer makes the first port too large or slow to validate.

## Current Local State

The Hugging Face cache for `baidu/ERNIE-Image-Turbo` is present locally. The repo revision currently advertised by Hugging Face is `bc68c81e2a1730a394d5fc9fae70713dee940140`, with about 29.43 GiB of safetensors:

- `pe/model.safetensors`: about 7.14 GiB
- `text_encoder/model.safetensors`: about 7.17 GiB
- `transformer/diffusion_pytorch_model-00001-of-00002.safetensors`: about 9.31 GiB
- `transformer/diffusion_pytorch_model-00002-of-00002.safetensors`: about 5.66 GiB
- `vae/diffusion_pytorch_model.safetensors`: about 0.16 GiB

If another machine is missing those files, use an explicit download command rather than letting generation trigger network access:

```sh
HF_HUB_ENABLE_HF_TRANSFER=1 uv run mlxgen download --model baidu/ERNIE-Image-Turbo
```

## Proposed File Layout

```text
src/mflux/models/ernie_image/
  ernie_image_initializer.py
  variants/txt2img/ernie_image_turbo.py
  latent_creator/ernie_image_latent_creator.py
  schedulers/ernie_image_flow_match_scheduler.py
  weights/ernie_image_weight_definition.py
  weights/ernie_image_weight_mapping.py
  model/ernie_transformer/
  model/ernie_text_encoder/
  model/ernie_prompt_enhancer/
```

Add model-family wiring to:

- `src/mflux/models/common/config/model_config.py`
- `src/mflux/models/common/cli/save.py`
- `src/mflux/cli/mlx_gen.py`
- generated model-card logic if publishing quantized ERNIE weights

## Implementation Phases

### Phase 1 - Config and Routing

- Add `ModelConfig` entry and aliases for `baidu/ERNIE-Image-Turbo`, `ernie-image-turbo`, and `ernie`.
- Make `uv run mlxgen prepare --model baidu/ERNIE-Image-Turbo ...` resolve to an ERNIE backend instead of failing base-model inference.
- Add smart-router family detection to `mlxgen generate`.
- Keep any legacy `mflux-*` entrypoint hidden or omitted unless needed internally.

### Phase 2 - Tokenizer and Text Encoder

- Load ERNIE `TokenizersBackend` assets through the existing tokenizer infrastructure or a small ERNIE-specific wrapper.
- Preserve upstream behavior:
  - `add_special_tokens=True`
  - BOS insertion from `tokenizer.json`
  - no padding before text-encoder execution
  - empty prompt fallback compatible with Diffusers
- Port the language path of `Mistral3Model`:
  - 26 layers
  - hidden size 3072
  - 32 query heads, 8 KV heads, head dim 128
  - RMSNorm
  - GQA
  - Yarn RoPE and `llama_4` query scaling
- Return `hidden_states[-2][0]`, not the final normalized state.
- Skip `vision_tower` and `multi_modal_projector` in the first text-to-image port.

### Phase 3 - ERNIE Transformer

- Port `ErnieImageTransformer2DModel` from Diffusers:
  - BCHW latent input with 128 channels
  - 1x1 patch embed as squeezed Conv2d or equivalent Linear
  - text projection from 3072 to 4096
  - 36 shared AdaLN blocks
  - non-causal attention over image and text tokens
  - q/k RMSNorm
  - ND3 RoPE with axes dim `[32, 48, 48]` and theta `256`
  - GELU gated feed-forward path
  - final AdaLN and output projection
- Keep position-id construction identical to Diffusers:
  - image tokens first
  - text tokens second
  - text length offset for image temporal axis

### Phase 4 - Scheduler, Latents, and VAE Decode

- Implement ERNIE's flow-match schedule:
  - `sigmas = linspace(1.0, 0.0, steps + 1)[:-1]`
  - fixed scheduler shift from config, currently `4.0`
- Keep classifier-free guidance as upstream:
  - `uncond + guidance_scale * (cond - uncond)`
- Reuse the existing Flux2 VAE port where possible, but keep ERNIE-specific decode logic:
  - denormalize packed latents with VAE batch-norm stats
  - use BN eps `1e-5`
  - unpatchify `[B,128,H/16,W/16] -> [B,32,H/8,W/8]`
  - decode through Flux2 VAE

### Phase 5 - Prompt Enhancer

- Add prompt enhancer after the baseline image path is stable.
- Port `Ministral3ForCausalLM` using the shared decoder where possible.
- Preserve ERNIE chat-template behavior:
  - JSON payload `{"prompt": prompt, "width": width, "height": height}`
  - PE tokenizer chat template
  - configurable `use_pe`, with `False` supported for deterministic parity tests

### Phase 6 - Quantization and Publishing

- Start with BF16/local generation parity.
- Add q8 only after BF16 works.
- Research q4 sensitivity separately; do not reuse Qwen's mixed q4/q8 rule blindly.
- Generate HF model cards with:
  - original provider/model citation
  - MLX-Gen version
  - original license
  - compatibility limits
  - quantization-specific wording

## Validation

1. Config and routing tests:
   - `ModelConfig.from_name("baidu/ERNIE-Image-Turbo")`
   - `mlxgen prepare --model baidu/ERNIE-Image-Turbo --help` routing assumptions
2. Tokenizer parity:
   - English prompt
   - Chinese prompt
   - empty prompt
   - long truncated prompt
   - PE chat-template prompt
3. Text encoder parity:
   - Compare MLX `hidden_states[-2]` to local Transformers output on short prompts.
4. Transformer parity:
   - Small-config forward comparison against Diffusers.
   - Validate ND3 RoPE and attention masks separately.
5. Scheduler parity:
   - Compare sigma sequence and one scheduler step against Diffusers.
6. Decode parity:
   - Validate packed latent unpatchify and VAE decode shape/dtype/finite values.
7. Real smoke test:
   - `use_pe=False`, 1 step, fixed seed, 512 or 1024 output.
   - Then `use_pe=True`, 8 steps, fixed seed, Turbo default.
8. Save/prepare smoke:
   - q8 save/load round trip.
   - q4 only after dedicated quality and speed tests.

## Initial User-Facing Commands

Once implemented:

```sh
uv run mlxgen prepare \
  --model baidu/ERNIE-Image-Turbo \
  -q 8 \
  --path models/ernie-image-turbo-8bit
```

```sh
uv run mlxgen generate \
  --model models/ernie-image-turbo-8bit \
  --prompt "A precise architectural photo of a glass library at sunrise" \
  --steps 8 \
  --guidance 4 \
  --output ernie-turbo.png
```

## Open Decisions

- Whether prompt enhancer should be enabled by default in MLX-Gen once ported. Upstream defaults to enabled, but it adds about 7 GB of weights and another autoregressive generation step.
- Whether the first release should expose only `baidu/ERNIE-Image-Turbo` or also leave room for future non-turbo ERNIE variants.
- Whether to repair incomplete HF cache snapshots automatically in `download`, or simply provide a precise diagnostic and command.
