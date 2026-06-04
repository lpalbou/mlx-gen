# Planned: Model integration roadmap

## Metadata

- Created: 2026-05-25
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: Needs new ADR if MLX-Gen adds a plugin/provider boundary for independent model
  families or video backends. No ADR is required for narrow model wiring that follows existing
  `ModelConfig`, initializer, weight-definition, and CLI-router patterns.

## Context

MLX-Gen now has its own package identity but still carries mflux-compatible internals. The immediate
need is to choose which image and video models deserve first-class local Apple Silicon support for
standalone users and for AbstractVision, without turning every failed Hugging Face model id into a
large unplanned port.

This item records the current local cache, online model candidates, and the priority order for
text-to-image, image-to-image/edit, text-to-video, and image-to-video work.

## Current code reality

- `src/mflux/models/common/cli/save.py` chooses the prepare backend by substring:
  Qwen, FIBO, Z-Image, ERNIE, FLUX.2, otherwise FLUX.1.
- `ModelConfig.from_name()` can resolve exact aliases, explicit `--base-model`, or known alias
  substrings only. ERNIE Image Turbo and Wan2.2 TI2V are now known families; GLM and CogVideoX
  remain real backend ports rather than alias problems.
- Supported model config families include FLUX.1, FLUX.2 Klein, Qwen Image/Edit, FIBO, Z-Image,
  ERNIE Image Turbo, Wan2.2 TI2V, and SeedVR2. `mlxgen prepare` does not yet route SeedVR2.
- Local prepared folders already exist for Qwen Image, Qwen Image 2512, Qwen Image Edit variants,
  FLUX.2 Klein 4B/9B, and Z-Image-Turbo q4.
- As of 2026-05-25, the source snapshot sizes reported by Hugging Face are approximately:
  Qwen Image/Edit 54 GiB, FLUX.2 Klein Base 4B 22 GiB, FLUX.2 Klein Base 9B 49 GiB,
  Z-Image 19 GiB, Z-Image-Turbo 31 GiB, ERNIE-Image/Turbo 29 GiB, GLM-Image 33 GiB,
  FIBO/Fibo-lite/Fibo-Edit 22-24 GiB, Wan2.2 TI2V 5B 32 GiB, Wan2.2 A14B 118 GiB,
  LTX-2.3-fp8 55 GiB, and CogVideoX-2B 13 GiB.
- Local Hugging Face cache includes unsupported or partially supported candidates:
  `zai-org/GLM-Image`, `zai-org/CogVideoX-2b`, `numz/SeedVR2_comfyUI`,
  `prism-ml/bonsai-image-ternary-4B-mlx-2bit`,
  `prism-ml/bonsai-image-binary-4B-mlx-1bit`, and gated Bria FIBO repo stubs.
- ERNIE-Image-Turbo has a text-to-image MLX port: tokenizer, Mistral3 text encoder,
  ErnieImageTransformer2DModel, FlowMatch-style scheduler, Flux2-style VAE wrapper, optional
  Prompt Enhancer, `mlxgen` routing, BF16 prepare/download support, and q8/q4 prepared folders.
  Remaining ERNIE work includes image-input tasks if upstream supports them, ERNIE-Image
  non-turbo validation, and stronger Diffusers parity tests.
- Wan2.2 TI2V has an initial text-to-video and experimental first-frame image-to-video MLX port:
  Wan transformer, Wan VAE encoder/decoder, UniPC scheduler, local-only Hugging Face UMT5 prompt
  encoding, MP4 output, `mlxgen` routing, save/download wiring, focused tests, and opt-in
  Diffusers fixture checks for transformer, VAE encode/decode, prompt embeddings, scheduler replay,
  and a tiny 3-step CFG latent denoise loop. The I2V path follows Diffusers first-frame latent
  conditioning rather than ordinary img2img initialization. Low-resolution Wan smoke outputs were
  replaced in public docs after direct upstream Diffusers comparison showed that tiny settings can
  produce abstract green frames. Current docs use upstream quality-validation settings and include
  1280x704, 17-frame, 20-step spatial-scale sanity panels; full quality validation still needs the
  upstream 121-frame, 50-step, 24fps regime or a deliberately documented lower-cost equivalent.
- Shared `ProgressEvent` callbacks now cover image and Wan video generation. Image callers
  subscribe through `model.callbacks.subscribe_progress(...)`; Wan video also accepts a direct
  `generate_video(progress_callback=...)` handler. Cancellation and richer runtime state remain
  future orchestration work.
- GLM-Image is also a real port, not an alias. Its snapshot declares `GlmImagePipeline`,
  `GlmImageTransformer2DModel`, `GlmImageForConditionalGeneration`, `GlmImageProcessor`,
  `T5EncoderModel`, and `AutoencoderKL`.
- FIBO repositories are gated on Hugging Face. A user without access sees `GatedRepoError` while
  downloading protected files such as `text_encoder/config.json`.
- Current CLI error handling for gated models is too raw: users get a long traceback instead of a
  short access/login/remediation message.
- Remote AbstractFramework audit on 2026-05-25 confirmed the checked Qwen, FLUX.2 Klein,
  FLUX.2 Klein Base, Z-Image, and Z-Image-Turbo 4-bit/8-bit repositories all have safetensor
  weights, generated MLX-Gen cards, and `LICENSE.md` files. Apache repos are public; FLUX.2 9B
  derivatives are gated with `license_name: flux-non-commercial-license`.
- The Hugging Face `AbstractFramework / mlx-gen` collection still needs a separate membership
  pass. The repository token used for the audit can read the collection and write model repos, but
  receives HTTP 403 when calling `add_collection_item`.

## Problem

New model ids can look like simple `mlxgen prepare --model ...` additions, but some require complete
new backends. Without a priority map, we risk mixing quick publishing work, UX error handling, and
major model-porting work in the same release.

## What we want to do

Integrate models in an order that maximizes immediate value for AbstractVision and standalone
MLX-Gen users while preserving clean, reviewable ports:

1. Validate and publish families that are already supported locally.
2. Fix cross-cutting UX problems such as gated repos and clearer unsupported-model diagnostics.
3. Port high-value text/image models whose architectures are understood and cached locally.
4. Treat video support as a separate track because it needs new pipeline, memory, cancellation, and
   output abstractions.

## Why

AbstractVision needs a reliable Apple Silicon backend that can prepare local quantized models,
generate or edit images, start adding video generation without surprise downloads, report progress,
and fail with actionable messages. Shared progress reporting now exists; cancellation and richer
runtime state remain roadmap work. MLX-Gen also needs enough model coverage to stand alone as more
than a renamed fork.

## Priority table

| Priority | Modality | Model or family | Current evidence | Complexity | Why and implementation hints |
| --- | --- | --- | --- | --- | --- |
| P0 | Publication hygiene | Published model audit and guardrails | Current checked Qwen/FLUX/Z-Image repos are complete and have correct cards/licenses/gating, but this must become automated because HF settings are outside `mlxgen prepare`. | Low | Add a publish/audit helper that verifies safetensor presence, expected license metadata, `LICENSE.md`, `base_model`, `pipeline_tag`, `library_name`, and gated settings before or after upload. |
| P0 | T2I, I2I/edit | Existing supported families: Qwen Image/Edit, FLUX.2 Klein distilled/base, Z-Image/Z-Image-Turbo | Code paths exist. Qwen/FLUX/Z-Image 4-bit and 8-bit published sizes now look plausible. Z-Image/Z-Image-Turbo and FLUX.2 4B are Apache; FLUX.2 9B/base-9B derivatives must remain gated. | Low | Continue generation validation and keep model cards/license files/HF settings synchronized. No new backend should block this already-supported publishing work. |
| P0 | Cross-cutting | Gated and non-commercial derivative publishing policy | FLUX.2 9B derivatives now require `gated=auto`; FIBO family is gated/non-commercial. `prepare` writes local files only and cannot set HF repo settings. | Low-medium | Add release/publishing helpers or docs that run `HfApi.update_repo_settings(gated=\"auto\")`, upload upstream license files, and prevent accidental public publication of gated derivatives. |
| P1 | T2I, I2I/edit | `briaai/FIBO`, `briaai/Fibo-lite`, `briaai/Fibo-Edit` | Backend exists; access has been granted. Source sizes are 22-24 GiB. FIBO is structured/JSON-native and trained for professional controllability; Fibo-lite targets 8-step/CFG=1 inference; Fibo-Edit targets structured edits. | Medium | Validate because this is already implemented, but do not make it the default family. It is non-commercial, gated, structured-workflow-heavy, and less broadly reusable than Qwen/FLUX/Z-Image. Publish only gated derivatives with Bria license terms. |
| P1 | T2I | `baidu/ERNIE-Image-Turbo` and `baidu/ERNIE-Image` | Apache 2.0, 29 GiB source snapshot including Prompt Enhancer, strong card claims around text rendering, structured layout, complex instruction following, and 8-step Turbo inference. Turbo now has BF16, q8, q4, and optional Prompt Enhancer text-to-image support with real image validation; non-turbo ERNIE-Image remains open. | High | Continue the ERNIE port with Diffusers parity tests, non-turbo defaults, generated-card behavior, and AbstractVision-facing runtime state APIs beyond the shared progress callbacks. |
| P1 | T2I | `prism-ml/bonsai-image-ternary-4B-mlx-2bit` | Implemented. FLUX.2 Klein-shaped architecture with `transformer-packed-mflux/`, a 4-bit Qwen3 text encoder, and BF16 Flux2 VAE. Local validation shows coherent output, lower RSS than FLUX.2 Klein 4B q8, and direct `mlxgen generate` support. | Done for ternary, medium-high for future low-bit kernels | Maintain the narrow Bonsai/FLUX.2 packed-loader path using standard MLX 2-bit affine execution, not a full Prism demo dependency. Track binary 1-bit separately because stock MLX through 0.31.2 still fails the required 1-bit probe. |
| P2 | T2V, I2V | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Apache 2.0, ~32 GiB, supports both text-to-video and image-to-video at 720p/24fps, and is much smaller than A14B. Initial MLX-Gen T2V and first-frame I2V support now produces MP4 output. | Very high | Continue from the first video milestone: improve quality/performance defaults, add cancellation events, memory caps, q4/q8 validation, and a longer Diffusers parity suite. |
| P2 | T2I, I2I/edit | `HiDream-ai/HiDream-O1-Image` and `HiDream-ai/HiDream-O1-Image-Dev` | MIT, current search shows image-text-to-image tags, Qwen3-VL stack, and an existing `mlx-community/HiDream-O1-Image-Dev-mlx-bf16` checkpoint. | High | Worth researching after ERNIE because an MLX BF16 artifact exists, but it likely wants an MLX-VLM/provider boundary rather than a quick mflux-style port. |
| P2 | Video-to-video/upscale | `numz/SeedVR2_comfyUI`, `ByteDance-Seed/SeedVR2-3B/7B` | SeedVR2 code exists and source is cached, but `mlxgen prepare` does not route it. | Medium-high | Make existing upscaler usable from unified CLI and prepare/card flow before larger video generation ports. It is not T2V/I2V, but it is the lowest-risk video capability already in tree. |
| P2 | I2V, T2V, V2V, A/V | `Lightricks/LTX-2.3-fp8` | Very high online usage, ~55 GiB fp8, image-to-video/text-to-video/video-to-video/audio-video tags, custom community license. Card says full and distilled checkpoints exist and training is recommended on BF16. | Very high | Important to research, but licensing and model breadth make it riskier than Wan2.2 TI2V 5B. Needs a video/audio-capable backend decision before implementation. |
| P3 | T2V | `zai-org/CogVideoX-2b` | Apache 2.0, cached locally, ~13 GiB, older but small. | High | Good proof-of-concept video port if Wan/LTX are too large. Lower strategic priority because current ecosystem momentum is stronger around Wan/LTX. |
| P3 | T2I | `zai-org/GLM-Image` | MIT, cached locally, ~33 GiB, custom GLM image/VLM architecture. | Very high | Useful text-rendering/glyph candidate, but lower priority than ERNIE because downloads are lower and the custom VLM/VQ stack is significant. |
| P0 | T2V, I2V | `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | Apache 2.0 and strong current relevance. Initial T2V-A14B and I2V-A14B support exists with source-checkpoint MP4 smoke validation. Each source family is very large and still needs careful quality, quantization, and performance validation. | Very high | Focused item 0012 completed the initial wiring. Keep quantization, full-quality generation, and motion/prompt validation in the Wan follow-up items. |
| P4 | T2I/I2I or wrappers | FLUX.2-dev, FLUX.1-Kontext, Stable Diffusion 3.5, HunyuanImage 3.0, Sulphur-2, external MLX SD3 | Large or license-constrained, overlapping, or not native to current MLX-Gen abstractions. | High to very high | Track but do not prioritize. Use AbstractVision provider adapters if needed rather than forcing all ecosystems into MLX-Gen. |

Related focused items:

- [Bonsai ternary FLUX.2 support](../completed/0003_bonsai_ternary_flux2_support.md)
- [Bonsai binary 1-bit runtime support](../proposed/0004_bonsai_binary_1bit_runtime_support.md)

## Requirements

- Keep unsupported models from pretending to prepare successfully.
- Add clear diagnostics that distinguish unknown family, missing explicit `--base-model`, gated
  access, and missing local files.
- For every integrated family, include:
  - config aliases and route detection;
  - weight definitions and mappings;
  - download patterns;
  - generation/edit/upscale validation artifacts;
  - q4/q8 behavior and size evidence where quantization is supported;
  - generated Hugging Face card behavior;
  - user docs and Python API examples.
- Do not allow generation commands to auto-download models in the middle of a workflow.
- Keep the package standalone while preserving enough mflux vocabulary to make upstream merges
  possible later.

## Suggested implementation

1. Add friendly Hugging Face access errors around `mlxgen download` and `mlxgen prepare`.
2. Add unsupported-family diagnostics for GLM/CogVideoX and unsupported Wan modes that say a
   backend path is required, instead of suggesting an arbitrary `--base-model`.
3. Validate and publish existing supported local-cache q4/q8 models before starting ERNIE.
4. For ERNIE, add stronger Diffusers comparison tests around the transformer, text encoder,
   Prompt Enhancer, and scheduler, then decide non-turbo scope from measured behavior.
5. For video, create a separate API design note or ADR before expanding beyond the first Wan T2V
   milestone if cancellation, memory caps, output containers, or broader backend ownership become
   first-class architecture policy.

## Scope

- Model-family prioritization for MLX-Gen.
- Short-term CLI and prepare UX follow-ups.
- First integration hints for T2I, I2I/edit, T2V, I2V, and V2V/upscale.

## Non-goals

- Do not treat the first ERNIE or Wan implementation as the end of the integration track; parity,
  I2V, q4/q8 validation, and Python orchestration work remain follow-up scope.
- Do not implement GLM or CogVideoX in this backlog item.
- Do not publish or redistribute gated model derivatives until the license and access terms are
  explicitly checked.
- Do not make generation auto-download models as part of this work.
- Do not replace existing mflux-compatible APIs while the fork is still close enough for useful
  upstreaming.

## Dependencies and related tasks

- `src/mflux/cli/mlx_gen.py`
- `src/mflux/models/common/cli/save.py`
- `src/mflux/models/common/config/model_config.py`
- `src/mflux/models/common/resolution/config_resolution.py`
- `src/mflux/models/qwen/weights/qwen_weight_definition.py`
- `src/mflux/models/flux2/`
- `src/mflux/models/fibo/`
- `src/mflux/models/z_image/`
- `src/mflux/models/seedvr2/`
- `docs/model-management.md`
- `docs/huggingface-publishing.md`
- Potential future ADR: video backend ownership, cancellation, and richer AbstractVision runtime
  state boundaries.
- `src/mflux/models/wan/`
- `src/mflux/utils/generated_video.py`
- `src/mflux/utils/video_util.py`

## Expected outcomes

- Users get exact, actionable guidance for gated repos and unsupported families.
- Supported local-cache models can be prepared and published without custom commands outside
  `mlxgen`.
- ERNIE and GLM are treated as real ports with explicit architecture tasks.
- Video work is no longer blocked by image CLI assumptions for the first Wan text-to-video path.
- AbstractVision has a credible Apple Silicon model roadmap.

## Validation

- `uv run mlxgen prepare --model Tongyi-MAI/Z-Image-Turbo -q 4 --path models/z-image-turbo-4bit`
  either succeeds or reports an actionable existing-path/validation message.
- `uv run mlxgen prepare --model baidu/ERNIE-Image-Turbo --path models/ernie-image-turbo`
  creates a BF16 prepared folder with an ERNIE model card.
- `uv run mlxgen prepare --model baidu/ERNIE-Image-Turbo -q 4 --path models/ernie-image-turbo-4bit`
  creates a loadable q4 prepared folder with an ERNIE model card.
- `uv run mlxgen prepare --model briaai/FIBO -q 4 --path models/fibo-4bit` reports the gated repo
  remediation steps when access is missing.
- Existing Qwen mixed q4/q8 saves and FLUX.2/Z-Image saves continue to prepare and load.
- `uv run mlxgen generate --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --task text-to-video ...`
  creates an MP4 from the local source snapshot.
- `uv run mlxgen generate --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --task image-to-video --image ...`
  creates an MP4 using first-frame latent conditioning.
- `MFLUX_RUN_LOCAL_WAN_PARITY=1 uv run pytest tests/wan/test_wan_local_parity.py -q`
  validates the full Wan transformer, VAE encoder/decoder, prompt embeddings, scheduler replay,
  and a tiny 3-step CFG latent denoise loop against Diffusers-generated fixtures.
- Wan documentation distinguishes quick command checks from quality-validation settings without
  emitting runtime warnings for valid low-cost requests.
- New model cards continue to include source model, mflux acknowledgement, MLX-Gen version,
  quantization policy, AbstractFramework namespace examples, and contributor attribution.

## Progress checklist

- [x] Resolve empty public 8-bit repos by uploading weights or hiding/removing placeholders.
- [ ] Add friendly gated-repo and unsupported-family diagnostics.
- [x] Validate and publish Z-Image/Z-Image-Turbo q4/q8 from the existing backend.
- [ ] Validate FIBO family after gated access is granted.
- [x] Prepare and validate FLUX.2 Klein base variants if they are needed for training workflows.
- [ ] Add missing q8 and non-turbo Z-Image repos to the Hugging Face collection once collection
      write permission is available.
- [x] Add initial ERNIE-Image-Turbo BF16 text-to-image backend.
- [x] Enable and validate ERNIE-Image-Turbo q8/q4 prepared folders.
- [x] Add optional ERNIE Prompt Enhancer support for full source snapshots.
- [x] Add initial Wan2.2 TI2V text-to-video backend and MP4 output.
- [x] Port Wan image-to-video first-frame latent conditioning.
- [x] Add opt-in Wan full-model parity fixtures for transformer and VAE encoder checks.
- [x] Add Wan prompt-embedding parity and VAE decode parity fixtures.
- [x] Add short deterministic Wan denoise-loop parity against Diffusers.
- [x] Remove misleading tiny Wan contact-sheet examples and document low-cost checks separately
      from quality-validation settings.
- [ ] Improve Wan video quality/performance validation beyond tiny smoke runs.
- [ ] Add one full short Wan Diffusers-vs-MLX generation comparison for the same prompt, seed,
      dimensions, frames, steps, and guidance.
- [ ] Validate Wan q8/q4 preparation and decide whether a mixed quantization policy is needed.
- [ ] Add Bonsai ternary 2-bit support through a FLUX.2-compatible packed-loader path.
- [ ] Reassess Bonsai binary 1-bit only after runtime support is available or accepted by ADR.
- [ ] Add stronger ERNIE Diffusers comparison tests and non-turbo scope.
- [ ] Decide whether SeedVR2 should be unified under `mlxgen prepare` before larger video ports.
- [ ] Draft video backend/API ADR before expanding Wan or implementing CogVideoX.
- [ ] Keep docs and generated model cards synchronized with every integrated family.

## Guidance for the implementing agent

Re-check current code and local cache before implementation. Prefer small, independently releasable
changes: diagnostics first, supported-family validation second, new architecture ports third. If a
model card or Hugging Face repo says a model is gated, non-commercial, or licensed differently than
Apache/MIT, stop and record the licensing implication before publishing derived weights.
