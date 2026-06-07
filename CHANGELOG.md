# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.18.13] - 2026-06-07

### Added

- **SeedVR2 official source and packages**: route `seedvr2` and `seedvr2-3b` to the official
  `ByteDance-Seed/SeedVR2-3B` checkpoint, add `mlxgen prepare` support for reusable SeedVR2
  packages, and document the `AbstractFramework/seedvr2-3b-8bit` and
  `AbstractFramework/seedvr2-3b-4bit` package profiles.
- **Official SeedVR2 7B route and package validation**: route `seedvr2-7b` to the official
  `ByteDance-Seed/SeedVR2-7B` checkpoint, add official 7B source loading, validate local q8/q4 7B
  packages, and document storage, timing, Max RSS, and 5x contact-sheet outputs.
- **SeedVR2 3B/7B comparison sheet**: add a single side-by-side documentation asset stacking 3B
  and 7B source/q8/q4 outputs so users can compare the same 5x upscale profile directly.
- **Unified SeedVR2 upscale command**: add `mlxgen upscale` as the public SeedVR2 image
  super-resolution command while keeping `mflux-upscale-seedvr2` available for compatibility.

### Fixed

- **SeedVR2 package resolution**: reject unsupported Hugging Face-style SeedVR2 model handles before
  loading weights and preserve recognized `AbstractFramework/seedvr2-*` package handles through
  weight resolution. Missing q8/q4 packages now fail with the explicit download/prepare guidance
  instead of resolving to another SeedVR2 source.

## [0.18.12] - 2026-06-07

### Added

- **SeedVR2 upscaling guide**: add a dedicated image-upscaling documentation page with a real
  `5x` SeedVR2 comparison, including the original `133x113` source enlarged to the generated
  `658x560` output size for direct quality assessment.

### Changed

- **SeedVR2 upscale quality default**: SeedVR2 now defaults to untiled VAE encode/decode for image
  quality, with `--vae-tiling` available as an explicit memory-saving opt-in for very large
  upscales. The docs now recommend `--softness 0.25` to `0.5` for visibly noisy sources.
- **Acknowledgements**: refresh model-family credits for the routed MLX-Gen model surface,
  including FLUX, Qwen, Wan, Z-Image, ERNIE, FIBO, Bonsai, SeedVR2, and inherited mflux routes.

### Fixed

- **SeedVR2 output metadata**: `mflux-upscale-seedvr2 --metadata` now writes JSON sidecars, records
  source image path/dimensions, and reports final even output dimensions for non-16-multiple upscale
  targets.

## [0.18.11] - 2026-06-06

### Added

- **Image-edit examples**: add a current Qwen Image Edit 2511 source/q8/q4 parity sheet
  with exact commands for pencil sketch, hard-landing edit, and multi-reference composition, and
  document the existing Qwen Image Edit 2509 plus FLUX.2 Klein 4B/9B proof matrices.
- **Wan video health metadata**: saved Wan MP4 metadata now includes decoded frame/file health
  measurements such as frame count, output size, luma range, and mean temporal delta.
- **Wan failure diagnostics**: `mlxgen generate` can pass `--failure-diagnostics` to include
  runtime memory and tensor-health details in Wan failure manifests.

### Changed

- **FIBO Edit public status**: keep FIBO Edit unavailable through unified `mlxgen generate`
  capability discovery until source-model parity and release-quality edit validation pass.
- **Prepared local folder routing**: Hugging Face model handles such as
  `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` can resolve to a complete matching
  `./models/<repo-name>` local MLX-Gen package before requiring a Hugging Face cache snapshot.
- **Wan image-to-video sizing**: image-to-video now treats `width` and `height` as a size target,
  resolves the actual output canvas from the input image aspect ratio and model spatial multiples,
  and records requested, source, and resolved dimensions in MP4 metadata.

### Fixed

- **Qwen Image/Edit FlowMatch scheduler parity**: use the model-config dynamic-shift scheduler
  values and terminal sigma stretch for Qwen Image/Edit FlowMatch runs, matching the upstream
  Diffusers scheduler contract and fixing Qwen Image Edit 2511 edit/composition adherence.
- **`mlxgen` alias imports**: importing `mlxgen.models.*` no longer replaces the `mflux.models`
  parent package with an alias module, keeping mixed `mlxgen` and `mflux` imports stable in the
  same Python process.
- **Wan q8 default generation path**: tensor-health diagnostics are opt-in so ordinary Wan q8
  generation keeps the default lazy MLX execution path. The published A14B q8 T2V and I2V packages
  passed the 480x240-target, 41-frame, 15-step release profile documented in `docs/quantization.md`.

## [0.18.10] - 2026-06-04

### Added

- **Taskless generation planning**: add public model capabilities and generation-plan resolution for
  text-to-image, image-to-image, text-to-video, and image-to-video. `mlxgen capabilities` now
  reports supported public tasks, internal modes, image-count contracts, route handlers, and option
  support before weights are loaded.
- **Image-to-image mode routing**: route one-image FLUX.2 requests to edit/reference I2I by
  default, route `--image-strength` requests to latent img2img, and route repeated `--image`
  inputs to multi-reference I2I where the model supports it.
- **Reproducible multimodal example**: add a documented spaceship-in-snow workflow with real
  generated assets for T2I, I2I edit, multi-reference I2I, Wan A14B T2V, and Wan A14B I2V.

### Changed

- **Public task cleanup**: `edit` is now a compatibility alias for `image-to-image` plus an
  internal I2I mode. New integrations should use media-direction tasks and inspect the resolved
  generation plan when they need routing details.
- **Progress event consistency**: edit and fill image pipelines now report image-to-image progress
  through the shared progress callback contract instead of relying on latent-img2img inference.
- **Local model routing safety**: local paths and custom repository names now fail earlier when
  model family or base-model hints are insufficient or contradictory.
- **README and docs**: rewrite the README around the current `mlxgen` command surface, published
  AbstractFramework model repositories, Wan A14B memory measurements, and AbstractVision /
  AbstractFramework / AbstractFlow usage.

### Fixed

- **Metadata replay routing**: generation metadata that includes `image_strength` now resolves to
  latent img2img instead of routing to edit/reference I2I.
- **Unsupported option handling**: mask, outpaint, image-strength, and model-family contradictions
  are rejected before model execution when the selected model capability does not support them.

## [0.18.9] - 2026-06-03

### Added

- **Shared progress callbacks**: expose `mflux.callbacks.ProgressEvent` and
  `CallbackRegistry.subscribe_progress(...)` so text-to-image, image-to-image, text-to-video, and
  image-to-video callers can subscribe to one denoise-step progress contract.
- **Wan progress API cleanup**: Wan video generation now emits the shared progress event and still
  supports a direct `generate_video(progress_callback=...)` handler for one-shot callers.
- **Wan A14B memory modes**: add runtime lifecycle paths that can release inactive A14B denoisers
  and lower MLX/Metal memory pressure during long T2V/I2V runs.

### Changed

- **Wan CLI progress semantics**: the Wan CLI now reports denoising-step progress instead of
  presenting step progress as an output-frame counter.
- **Wan q8 policy**: Wan prepared q8 folders now use a mixed q8/BF16 policy. Transformer block
  linears are quantized at q8, while conditioning/output projection paths, VAE, text encoder,
  scheduler metadata, tokenizer files, norms, convolutions, and other sensitive or non-quantizable
  paths stay BF16.
- **Wan video memory behavior**: MP4 frame conversion now avoids full-video NumPy temporaries,
  reducing avoidable memory pressure most noticeably for larger 81/121-frame outputs.
- **Generated model cards**: prepared BF16 and quantized model cards now describe their saved-weight
  layout and Wan mixed q8/BF16 policy more precisely.

### Documentation

- Document shared progress callbacks for Python integrations, Wan A14B guidance behavior, and Wan
  q8 storage/runtime-memory measurements.
- Clarify that Wan mixed q8/BF16 improves storage and measured usage memory in validation, but is
  not currently claimed as a speed improvement.
- Link Wan follow-up validation work from the backlog without making unsupported full-size claims
  in user-facing docs.

## [0.18.8] - 2026-05-31

### Added

- **Wan2.2 A14B support**: add T2V and I2V model configs, dynamic `transformer_2` weight loading, Wan2.1-style A14B VAE support, Diffusers-compatible high/low-noise boundary routing, scalar A14B timesteps, A14B concatenated image-condition latents, and optional `--guidance-2` low-noise guidance.
- **Wan A14B docs and tests**: document TI2V-5B versus A14B defaults, local snapshot requirements, and add focused tests for A14B config, VAE mapping, I2V conditioning shape, CLI defaults, optional low-noise guidance, and guidance routing.
- **Wan console scripts**: expose `mlxgen-generate-wan` and `mflux-generate-wan` entrypoints for direct Wan CLI checks.

### Fixed

- **Wan model identity safety**: unknown or generic Wan names no longer infer the TI2V-5B runtime from the generic `wan` substring. Wan requests now require an exact supported repo or a local MLX-Gen package with a specific Wan alias.
- **Wan prompt and default handling**: Wan generation now applies model-specific default negative prompts, spatial defaults, guidance defaults, and optional A14B low-noise guidance consistently across CLI and Python generation.
- **Wan low-cost runs**: valid low-resolution or short Wan runs no longer emit runtime quality warnings; docs now describe those settings as quick command checks rather than final quality settings.
- **Wan source/runtime mismatch guard**: Wan initialization now compares available Diffusers source configs against the selected MLX-Gen runtime before loading weights, and transformer calls fail with a clear channel mismatch before entering MLX convolution.

## [0.18.7] - 2026-05-27

### Added

- **Bonsai Image ternary 2-bit support**: `mlxgen generate` can now run `prism-ml/bonsai-image-ternary-4B-mlx-2bit` directly as a FLUX.2-derived text-to-image model with Prism's pre-packed low-bit transformer, a 4-bit Qwen3 text encoder, and the Flux2 VAE.
- **Bonsai validation docs**: Add Bonsai versus FLUX.2 Klein 4B q8 quality/speed/memory documentation, validation imagery, and comparison with MLX-Gen's mixed q4/q8 policies.

### Changed

- **Bonsai binary 1-bit handling**: `prism-ml/bonsai-image-binary-4B-mlx-1bit` is detected and rejected with an explicit unsupported-runtime message. The latest published stock MLX checked for this release, `0.31.2`, still does not support the required `bits=1, group_size=128` packed affine matmul.

## [0.18.6] - 2026-05-26

### Added

- **Wan local parity fixtures**: Add opt-in full-model Wan component-parity tests for the MLX transformer, VAE encoder/decoder, prompt-embedding paths, and a tiny 3-step CFG latent denoise loop against Diffusers-generated fixtures.
- **Wan smoke-setting warnings**: Warn when Wan generation uses tiny resolution, frame, step, or fps settings that are useful only for routing/MP4 smoke tests and not visual quality assessment.
- **Wan video progress**: Show a frame-based CLI progress bar during Wan text-to-video and image-to-video generation, and expose structured `WanProgressEvent` callbacks for Python callers.

### Documentation

- Replace low-resolution Wan example panels with upstream TI2V-5B recommended settings plus 1280x704 spatial-scale examples, and document tiny runs as command checks rather than final quality settings.

## [0.18.5] - 2026-05-26

### Added

- **ERNIE q8/q4 preparation and generation**: `mlxgen prepare --model baidu/ERNIE-Image-Turbo --quantize 8|4` now creates loadable MLX-Gen packages, and ERNIE generation can run from those q8/q4 packages.
- **ERNIE Prompt Enhancer**: `mflux-generate-ernie-image` / `mlxgen generate` can now run ERNIE's optional Prompt Enhancer when a full source snapshot with `pe/` and `pe_tokenizer/` files is available.

### Changed

- **Output replacement default**: Generation saves now replace the requested output path by default. Use `--replace false` or `--no-replace` to preserve existing files and write a suffixed output.
- **ERNIE q4 policy**: ERNIE q4 now uses a model-specific mixed q4/q8 policy. q8 is used for Mistral3 text/Prompt Enhancer linears plus ERNIE transformer V/O attention, conditioning/output linears, and VAE attention, while transformer Q/K and feed-forward modules remain q4.

### Documentation

- Advertise the current quantized-model compatibility surface and document that Qwen and ERNIE q4 checkpoints use model-specific mixed q4/q8 policies validated for generation quality.

## [0.18.4] - 2026-05-25

### Added

- **ERNIE Image Turbo text-to-image**: Add an initial MLX-native port of `baidu/ERNIE-Image-Turbo`, including Mistral3 text encoding, ERNIE transformer weights, FlowMatch Euler scheduling, Flux2-style VAE decode, `mlxgen generate` routing, and `mlxgen download` / `mlxgen prepare` support.
- **ERNIE model-card support**: Generated Hugging Face cards now recognize ERNIE Image Turbo as Apache 2.0, use ERNIE-specific 512px / 8-step / guidance-1 examples, and describe BF16 prepared weights when no quantization level is used.

### Changed

- **ERNIE quantization boundary**: ERNIE `--quantize` requests now fail explicitly because q4/q8 layouts are not validated yet.
- **Small ERNIE outputs**: ERNIE CLI generation warns when width or height is below 384px because very small outputs can crop or truncate subjects.

### Documentation

- Document ERNIE routing, BF16-only prepare/generation, model-card licensing, and recommended validation dimensions.

---

## [0.18.3] - 2026-05-25

### Changed

- **Generated model-card licenses**: `mlxgen prepare` now writes Apache 2.0 license metadata and a license section for Qwen, Z-Image, and FLUX.2 Klein 4B prepared checkpoints.
- **FLUX.2 Klein 9B publishing safety**: Generated cards for FLUX.2 Klein 9B and base-9B derivatives now use `license: other`, `license_name: flux-non-commercial-license`, source license links, and Hugging Face gated-access prompts.
- **Z-Image usage defaults**: Generated Z-Image cards now show model-specific generation defaults: Turbo examples use `--steps 8 --guidance 0`, and base Z-Image examples use `--steps 50 --guidance 4`.

### Documentation

- Add backlog tracking for model-integration priorities, publication audit guardrails, and Hugging Face collection follow-up work.

---

## [0.18.2] - 2026-05-25

### Changed

- **Generated model-card namespace**: Hugging Face model cards created by `mlxgen prepare` now default usage examples to `AbstractFramework/<repo-name>`.
- **Copy/paste model-card usage**: Generated cards now include `python -m pip install -U mlx-gen` before the `mlxgen download` and `mlxgen generate` commands.

### Removed

- **Generated collection recommendation**: Generated cards no longer emit a "Recommended Hugging Face collection" sentence. Collection membership remains a separate Hugging Face publishing step.

### Documentation

- Clarify that public generated cards use pip for baseline installation while repository development and release workflows continue to use uv.

---

## [0.18.1] - 2026-05-25

### Changed

- **CLI discoverability**: `mlxgen` and `mlxgen --help` now show the top-level `generate`, `download`, and `prepare` workflows instead of dropping directly into generation arguments.
- **Prepare command naming**: `mlxgen prepare --help` now presents the command as `mlxgen prepare` and describes the generated local model folder and Hugging Face model card.
- **Generated model-card tracking**: Hugging Face model cards created by `mlxgen prepare` now record the exact `mlx-gen` version that generated them.

### Fixed

- **Generate/prepare confusion**: `mlxgen generate --path ...` now fails with an actionable message that points to `mlxgen prepare --model ... --path ... --quantize ...` and explains that image outputs use `--output`.

### Documentation

- Clarify that `mlxgen prepare` is the public MLX-Gen workflow for creating reusable local quantized model folders and generated Hugging Face cards, and prefer long-form flags such as `--quantize` in public examples.
- Generated model cards now link to the MLX-Gen project and quantization documentation when describing quantized checkpoint policies.

---

## [0.18.0] - 2026-05-25

### Added

- **Generated Hugging Face model cards**: `mlxgen prepare` now writes a `README.md` model card into MLX-Gen model packages with source-model attribution, mflux and MLX-Gen compatibility notes, quantization details, contributor attribution, and collection guidance.

### Changed

- **Download command hints**: Missing-artifact remediation now shows plain `mlxgen download` and `mlxgen prepare` commands. `HF_HUB_ENABLE_HF_TRANSFER=1` remains optional acceleration for explicit Hugging Face downloads, not the download authorization mechanism.

### Documentation

- Document Qwen q4 mixed q4/q8 policy, q8 behavior, Hugging Face model-card generation, and collection publishing workflow.

---

## [0.17.5.post3] - 2026-05-25

### Changed

- **Explicit model preparation**: Runtime generation and Python model construction no longer download missing model, tokenizer, LoRA, or Depth Pro files. Missing artifacts now raise `DownloadRequiredError` with the exact `mlxgen download` or `mlxgen prepare` command to run.
- **Smart MLX-Gen commands**: Add `mlxgen download` and `mlxgen prepare` as explicit preparation flows. `mlxgen download --model depth-pro` handles the direct Apple Depth Pro weights.
- **LoRA handling**: User-requested LoRAs are required; missing LoRA files now fail instead of being ignored.

### Documentation

- Document that generation runs from local model files, plus Python integration expectations, AbstractVision usage context, and explicit model-management workflows.

---

## [0.17.5.post2] - 2026-05-25

### 🐛 Bug Fixes

- **Qwen Image q4 saving**: Save Qwen q4 transformers as mixed q4/q8 by using q8 for image-stream modulation (`*.img_mod_linear`) and q4 for the rest of the quantizable transformer linears. This preserves coherent Qwen q4 outputs while making newly saved q4 checkpoints smaller than both q8 and the `0.17.5.post1` BF16 mixed-q4 layout.

### 📝 Compatibility

- Existing all-q4 and `0.17.5.post1` Qwen BF16 mixed-q4 checkpoints continue to load. Re-save Qwen Image / Qwen Image Edit q4 checkpoints with `0.17.5.post2` to get the smaller q4/q8 layout.

### 🧰 DX & Maintenance

- **Package distribution**: Rename this fork's PyPI distribution to `mlx-gen` while preserving the `mflux` Python module and CLI command names, and add `mlxgen` as a lightweight import alias.

---

## [0.17.5.post1] - 2026-05-24

### 🐛 Bug Fixes

- **Qwen Image Edit conditioning resolution**: Encode the transformer’s image-conditioning latents at the edit target resolution, not the vision-language conditioning resolution (≈384px by area), preventing patchy/tiled artifacts in edit outputs.
- **Qwen Image Edit default dimensions**: Preserve the first input image dimensions by default; explicit `--width`/`--height` values or scale factors still opt into resizing.
- **Qwen Image Edit CLI scheduler**: Forward `--scheduler` to the Qwen edit pipeline (previously ignored).
- **Qwen Image Edit q4 saving**: Use mixed q4 quantization for Qwen transformers by keeping conditioning, modulation, and output projections at higher precision while quantizing the bulk attention and feed-forward layers.
- **FLUX.2 Klein Edit guidance**: Allow `--guidance > 1.0` for FLUX.2 Klein edits by checking the resolved FLUX.2 model config instead of requiring a base model name; defaults remain unchanged.

### 🧰 DX & Maintenance

- **AbstractVision package**: Publish this fork as `abstractvision-mflux` on PyPI while preserving the `mflux` Python module and CLI command names.

---

## [0.17.5] - 2026-04-10

### 🐛 Bug Fixes

- **Qwen Image Edit `mflux-save`**: Route Qwen edit model names to `QwenImageEdit` and save through the same path as inference so VisionTransformer (`encoder.visual`) weights are written. Saving with `QwenImage` previously omitted those weights and led to random vision encoders after reload.
- **Battery saver callback**: Harden Apple Silicon battery detection when `system_profiler` is missing and resolve the helper script via absolute paths.

### 📝 Documentation

- **Related projects**: Clarify that MindCraft Studio is a macOS app built on mflux.

### 🧰 DX & Maintenance

- **Dependencies**: Relax the `protobuf` upper bound to allow current 7.x releases while keeping a safe ceiling below 8.0.

### 👩‍💻 Contributors

- **@anthonywu**
- **@f-gibellini**
- **@JiwaniZakir**

---

## [0.17.4] - 2026-03-28

### 🐛 Bug Fixes

- **Z-Image PEFT/ModelScope LoRA keys**: Extend the Z-Image LoRA mapping with `.default` tensor name variants so adapters in PEFT/ModelScope layouts (for example Tongyi-MAI exports) resolve and apply correctly instead of matching zero weights.

### 👩‍💻 Contributors

- **@filipstrand**

---

## [0.17.3] - 2026-03-27

### 🐛 Bug Fixes

- **FLUX.2 edit guidance metadata**: Preserve the requested guidance value for FLUX.2 Klein base image-edit runs so `mflux-info` and saved metadata report the actual guidance used instead of always showing `1.0`.

### 👩‍💻 Contributors

- **@filipstrand**

---

## [0.17.2] - 2026-03-23

### 🐛 Bug Fixes

- **Shared tokenizer cache resolution**: Fix Hugging Face tokenizer resolution when a repo is only partially cached locally, preserving offline-first behavior for valid cached layouts while retrying ambiguous cached primaries once before surfacing real load errors.

### 🧰 DX & Maintenance

- **Tokenizer resolution coverage**: Expand shared tokenizer-resolution regression tests to cover root-layout tokenizers, fallback edge cases, and refresh failure handling.

### 👩‍💻 Contributors

- **@filipstrand**

---

## [0.17.1] - 2026-03-22

### 🐛 Bug Fixes

- **Hugging Face tokenizer dependencies**: Declare `protobuf` so minimal installs (including `uv tool install mflux`) include packages Transformers may require when loading tokenizers, fixing failures such as `mflux-generate-fibo` when the tokenizer falls back off the fast path.

### 👩‍💻 Contributors

- **@filipstrand**

---

## [0.17.0] - 2026-03-20

### 🎨 New Model Support

- **FIBO Edit**: Add image-editing support for the FIBO model family.
- **FIBO Edit remove-background workflow**: Support the dedicated remove-background edit path for FIBO.

### ✨ Improvements

- **Training image scaling**: Scale training images by area rather than longest side for more consistent preprocessing.
- **MLX 0.31.x**: Allow MLX 0.31.x in dependency ranges.
- **FLUX.2 LoRA mapping**: Expand LoRA key mapping coverage for FLUX.2.

### 🐛 Bug Fixes

- **Training optimizer state**: Evaluate optimizer state after each training step as intended.
- **Local tokenizer loading**: Fix loading tokenizers from local paths.
- **Dynamic-resolution image edit**: Restore correct behavior for image edit when using dynamic resolution.

### 👩‍💻 Contributors

- **@filipstrand**
- **@icelaglace**
- **@TheOrsa**
- **@waldheinz**

---

## [0.16.9] - 2026-03-07

### ✨ Improvements

- **Broader LoRA compatibility for FLUX.2 and Z-Image**: Expand LoRA mapping coverage so more adapter key layouts resolve cleanly for FLUX.2 and Z-Image models.

### 👩‍💻 Contributors

- **@filipstrand**

---

## [0.16.8] - 2026-03-06

### ✨ Improvements

- **Local-model LoRA training**: Allow LoRA training to work when the base model is supplied from a local path, including the FLUX.2 and Z-Image training adapters.

### 📝 Documentation

- **Distilled-model step defaults**: Clarify CLI guidance so examples prefer model default inference steps unless the user intentionally overrides them.

### 👩‍💻 Contributors

- **@waldheinz**

---

## [0.16.7] - 2026-03-02

### 🎨 New Model Support

- **FIBO-Lite support**: Add support for the FIBO-Lite model variant.

### 🐛 Bug Fixes

- **FLUX.2 edit downsampling extents**: Fix downsampling in FLUX.2 edit paths so image extents are preserved.

### 👩‍💻 Contributors

- **@filipstrand**

---

## [0.16.6] - 2026-02-20

### ✨ Improvements

- **SeedVR2 7B support**: Add support for the SeedVR2 7B upscaler variant.
- **Qwen-Image parity with diffusers**: Align Qwen-Image behavior more closely with the diffusers reference implementation.
- **FIBO scheduler default**: Default FIBO `generate_image` to `flow_match_euler_discrete`.

### 🧰 DX & Maintenance

- **Repo tooling cleanup**: Remove unused Cursor command wrappers from the repository.
- **SeedVR2 7B test coverage**: Add image test support for the new SeedVR2 7B path.

### 👩‍💻 Contributors

- **@ciaranbor**
- **@icelaglace**
- **@filipstrand**

---

## [0.16.5] - 2026-02-17

### ✨ Improvements

- **FLUX.2 Klein img2img CLI parity**: Add `--image-path` and `--image-strength` to `mflux-generate-flux2`, enabling init-image driven generation with the same CLI pattern used in other generators.
- **MLX cache control**: Add `--mlx-cache-limit-gb` to cap MLX cache usage without requiring full `--low-ram` mode.

### 📝 Documentation

- **Common CLI docs**: Document `--mlx-cache-limit-gb` behavior and usage in the shared model README.

### 👩‍💻 Contributors

- **@terribilissimo**
- **@icelaglace**

---

## [0.16.4] - 2026-02-15

### 🐛 Bug Fixes

- **Training preview stability**: Always offload optimizer state during preview generation to avoid memory pressure and improve preview reliability.
- **Apple Silicon compile guard**: Narrow the M1/M2 compile fallback so it excludes Max and Ultra variants, preserving expected optimized behavior on those chips.

---

## [0.16.3] - 2026-02-14

### 🐛 Bug Fixes

- **Z-Image training preview guidance**: Fix Z-Image (non-turbo) training previews so they use the configured guidance value instead of defaulting to 0.0, ensuring preview quality matches actual CFG behavior.
- **FLUX.2 training preview guidance**: Fix FLUX.2 training previews (txt2img and edit) so they use the configured guidance value instead of forcing 1.0.

---

## [0.16.2] - 2026-02-12

### 🐛 Bug Fixes

- **Edit training preview fallback**: Fix edit auto-discovery runs (`*_in/*_out`) with monitoring enabled so fallback preview prompts use an available input image instead of requiring explicit `data/preview.*` files.

### 📝 Documentation

- **FLUX.2 training guide**: Expand the FLUX.2 LoRA training example documentation with richer guidance and examples.

---

## [0.16.1] - 2026-02-11

### 🐛 Performance regression fixes

- **M1/M2 inference performance fallback**: Disable model-level `mx.compile` prediction wrappers for Z-Image and FLUX.2 on Apple M1/M2 to avoid observed 0.16 regressions on older Apple Silicon while preserving compiled paths on newer chips.

---

## [0.16.0] - 2026-02-11

### ✨ Improvements

- **Completely rewritten training system**: Rebuild LoRA training end-to-end, replacing the DreamBooth-specific implementation with a new common training stack (dataset, state, optimizer, runner, and statistics) shared across model families.
- **New base-model support for training and inference**: Add support for `flux2-klein-base-4b`, `flux2-klein-base-9b`, and `z-image` (in addition to `z-image-turbo`) with dedicated FLUX.2 and Z-Image training adapters.
- **Performance tuning**: Improve core scheduler/model execution paths used by FLUX.2 and Z-Image.

### 🐛 Bug Fixes

- **FLUX.2 Klein 9B text encoder overrides**: Fix override resolution/application in the FLUX.2 initializer/config flow.

### 🧰 DX & Maintenance

- **FLUX.1 legacy cleanup**: Remove legacy FLUX.1 image-generation tests/resources and retire unused helper tools.
- **Dependency alignment**: Update install guidance for stable `transformers` 5.0 and refresh lockfile/dependency metadata.

### 📝 Documentation

- **Training docs refresh**: Expand and update training docs/README sections for common training, FLUX.2, and Z-Image.
- **Install troubleshooting**: Add troubleshooting guidance for `hf_transfer` installation issues.

### 👩‍💻 Contributors

- **Filip Strand (@filipstrand)**
- **Xin (@q3g)**

---

## [0.15.5] - 2026-01-26

### ✨ Improvements

- **SeedVR2 directory input**: Allow passing a folder to `--image-path` to upscale all images inside.

### 🧰 DX & Maintenance

- **Model porting guidance**: Require model README entries in the porting workflow.

### 📝 Documentation

- **SeedVR2 usage**: Document directory upscaling with CLI and Python API examples.
- **CLI docs**: Add Python API sections and improve Z-Image Turbo entry-point links.

---

## [0.15.4] - 2026-01-20

### ✨ Improvements

- **Flux2 LoRA aliasing**: Add key aliases for `base_model` prefixes to improve LoRA resolution across configs.

### 📝 Documentation

- **Agent guidance**: Clarify skill references for Cursor agents.

---

## [0.15.3] - 2026-01-19

### 🐛 Bug Fixes

- **Flux2 Klein local path**: Fix errors when using a local FLUX.2-klein-9B path in `mflux-save` and `mflux-generate-flux2`.

---

## [0.15.2] - 2026-01-19

### 🐛 Bug Fixes

- **Flux2 edit (low-ram)**: Normalize tiled VAE latents to 4D before patchifying to avoid shape errors.

---

## [0.15.1] - 2026-01-18

### 🐛 Bug Fixes

- **PyPI metadata**: Removed invalid architecture classifier that blocked uploads (`Architecture :: AArch64`).

---

## [0.15.0] - 2026-01-18

### 🎨 New Model Support

- **Flux2 Klein (4B/9B)**: Full MLX port of Flux2 Klein (including multi-image editing support).
- **New command**: `mflux-generate-flux2` for Flux2 Klein image generation.
- **New command**: `mflux-generate-flux2-edit` for Flux2 Klein image editing.

### 🔧 Improvements

- **Qwen3-VL shared module**: Extracted `qwen3_vl` into `models/common_models/` for reuse across model families (Flux2 and Fibo etc).
- **Experimental CUDA support**: Added initial CUDA backend support as an experimental feature.
- **Test Infrastructure**: Image tests are pinned to MLX v0.30.3.

### 📝 Documentation

- **README reorganization**: Reorganized the main README for better structure and readability.

---

## [0.14.2] - 2026-01-13

### 📊 Improved Metadata Handling

- **Enhanced IPTC & XMP Support**: Significant improvements to metadata reading and writing, ensuring better compatibility with professional image editing tools.
- **Robust Metadata Extraction**: Refined logic for extracting generation parameters from previously generated images.
- **New Metadata Tests**: Added comprehensive test suite for IPTC metadata building and original image info utilities.

### 🤖 DX & Maintenance

- **Cursor AI Workflows**: Introduced standardized Cursor commands and agent rules in `.cursor/` for improved development consistency and automation.
- **SeedVR2 & ControlNet Tweaks**: Minor refinements to SeedVR2 and ControlNet model implementations.
- **Documentation Updates**: Updated README and added AGENTS.md for better contributor onboarding.

---

## [0.14.1] - 2026-01-01

### 🔧 SeedVR2 Improvements

- **Enhanced Color Correction**: Implemented precise LAB histogram matching with wavelet reconstruction for superior color consistency between input and upscaled images.
- **Configurable Softness**: Added a new `--softness` parameter (0.0 to 1.0) to control input pre-downsampling, allowing for smoother upscaling results when desired.
- **RoPE Alignment**: Fixed RoPE dimension mismatch (increased to 128) to perfectly match the reference 3B transformer architecture.

### 🤖 DX & Maintenance

- **Updated `.cursorrules`**: Added standard procedure for test output preservation and release management.
- **Updated Test Infrastructure**: Updated SeedVR2 reference images and fixed dimension-related test failures.

---

## [0.14.0] - 2025-12-31

### 🎨 New Model Support

- **SeedVR2 Diffusion Upscaler**: Added initial SeedVR2 image upscaling support.
- **New command**: `mflux-upscale-seedvr2` for high-quality image upscaling.
- **Tiling support**: Tiling is enabled by default for SeedVR2 to support high-resolution upscaling on standard memory configurations.

### 🔧 Improvements

- **Global VAE Tiling Support**: Introduced a unified VAE tiling system (`VAETiler`) that supports both tiled encoding and decoding.
- **Low-RAM Mode Enhancements**: Enabling `--low-ram` now automatically activates VAE tiling across all model families (Flux, Qwen, FIBO, Z-Image), significantly reducing memory pressure for high-resolution generation on Apple Silicon.
- **Robust Offline Cache Handling**: Improved logic for detecting complete cached models on HuggingFace Hub, handling symlinks and missing files more reliably to prevent runtime errors during offline use.
- **Selective Weight Loading**: Support for loading specific weight files, enabling more flexible model configurations and better resource sharing between related models.
- **CLI UX Improvements**:
  - Multi-image generation (multiple seeds or input images) now automatically appends suffixes (`_seed_{seed}` or `_{image_name}`) to output filenames to prevent accidental overwrites.
  - Better model configuration resolution with a priority-based system for resolving ambiguous model names.
- **Enhanced Shell Completions**: Significant updates to shell completion generation to support new commands and properly handle positional arguments and subparsers.
- **Qwen Test Hardening**: Updated Qwen image generation and edit tests to use 8-bit quantization for more robust and faster testing.
- **Test Infrastructure**: Added automatic MLX version pinning (v0.29.2) in `make test-fast` to ensure consistent test environments across different development setups.

### 📝 Documentation

- Added information about pre-quantized models available on HuggingFace for easier access.

---

## [0.13.3] - 2025-12-06

### 🐛 Bug Fixes

- **LoRA save bloat prevention**: Bake and strip LoRA wrappers before sharding to avoid exploding shard counts/sizes when saving quantized models with multiple/mismatched LoRAs (see [issue #217 comment](https://github.com/filipstrand/mflux/issues/217#issuecomment-3615321206)).
- **Regression test hardening**: LoRA model-saving tests now include size guardrails (5% tolerance) while using the bundled local LoRA fixtures to catch shard bloat regressions early.

---

## [0.13.2] - 2025-12-05

### ✨ Improvements

- **Better error messages for multi-file LoRA repos**: When a HuggingFace LoRA repo contains multiple `.safetensors` files, the error message now displays copy-paste ready options instead of a raw list
- **Z-Image LoRA format support**: Added support for Kohya and ComfyUI LoRA naming conventions, enabling compatibility with more community LoRAs.

---

## [0.13.1] - 2025-12-03

### 🐛 Bug Fixes

- **FIBO VLM chat template not loaded**: Fixed issue where the FIBO VLM tokenizer's chat template was not being loaded with `transformers` v5, causing `apply_chat_template()` to fail. The tokenizer loader now properly extracts and sets the chat template from the tokenizer config.

---

## [0.13.0] - 2025-12-03

# MFLUX v.0.13.0 Release Notes

### 🎨 New Model Support

- **Z-Image Turbo Support**: Added support for [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), a fast distilled Z-Image variant optimized for speed
- **New command**: `mflux-generate-z-image-turbo` for rapid image generation (with LoRA support, img2img, and quantization)

### ✨ New Features

- **FIBO VLM Quantization Support**: The FIBO VLM commands (`mflux-fibo-inspire`, `mflux-fibo-refine`) now support quantization via the `-q` flag (3, 4, 5, 6, or 8-bit)

- **Unified `--model` argument**: The `--model` flag now accepts local paths, HuggingFace repos, or predefined model names
  - Local paths: `--model /Users/me/models/fibo-4bit` or `--model ~/my-model`
  - HuggingFace repos: `--model briaai/Fibo-mlx-4bit`
  - Predefined names: `--model dev`, `--model schnell`, `--model fibo`
  - This mirrors how LoRA paths work for a consistent UX

- **Scale Factor Dimensions for Img2Img**: Generalized the scale factor feature (e.g., `2x`, `0.5x`, `auto`) from upscaling to all img2img commands
  - Specify output dimensions relative to input image: `--width 2x --height 2x`
  - Use `auto` to match input image dimensions: `--width auto --height auto`
  - Mix scale factors with absolute values: `--width 2x --height 512`
  - Supported in: `mflux-generate`, `mflux-generate-z-image-turbo`, `mflux-generate-fibo`, `mflux-generate-kontext`, `mflux-generate-qwen`
- **DimensionResolver utility**: New `DimensionResolver.resolve()` for consistent dimension handling across commands

### 🔧 Architecture Improvements

- **Unified Resolution System**: New `resolution/` module for consistent parameter resolution across all models
  - `PathResolution`: Resolves model paths from local paths, HuggingFace repos, or predefined names
  - `LoRAResolution`: Handles LoRA path resolution from all supported formats
  - `ConfigResolution`: Centralizes configuration resolution logic  
  - `QuantizationResolution`: Determines quantization from saved models or CLI args
- **Unified Weight Loading System**: Complete rewrite of weight handling with declarative mappings
  - New `WeightLoader` with single `load(model_path)` interface
  - `WeightDefinition` classes define model structure per model family
  - `WeightMapping` declarative mappings replace imperative weight handlers
  - Removed all per-model `weight_handler_*.py` files in favor of unified system
- **Unified Tokenizer System**: New common tokenizer module
  - `TokenizerLoader.load_all()` with unified `model_path` interface
  - Removed model-specific tokenizer handlers (`clip_tokenizer.py`, `t5_tokenizer.py`, etc.)
- **Unified LoRA API**: Simplified LoRA loading to a single `lora_paths` parameter
  - All LoRA formats now resolved through `LoRALibrary.resolve_paths()`:
    - Local paths: `/path/to/lora.safetensors`
    - Registry names: `my-lora` (from `LORA_LIBRARY_PATH`)
    - HuggingFace repos: `author/model`
    - **New**: HuggingFace collections: `repo_id:filename.safetensors`
  - Simplified model initialization: just pass `lora_paths` and everything resolves automatically
- **Unified Latent Creator Interface**: Standardized `unpack_latents(latents, height, width)` signature across all model families
  - `FluxLatentCreator`, `ZImageLatentCreator`, `FiboLatentCreator`, and `QwenLatentCreator` now share the same interface
  - Moved `FIBO._unpack_latents` to `FiboLatentCreator.unpack_latents` for consistency
- **StepwiseHandler Refactor**: Fixed `StepwiseHandler` to work with all model types by accepting a `latent_creator` parameter
  - Previously hardcoded to `FluxLatentCreator`, now model-agnostic
  - Each command passes its appropriate latent creator to `CallbackManager.register_callbacks()`
- **CLI Reorganization**: Moved CLI entry points to model-specific directories (e.g., `mflux/models/flux/cli/`)

### 🔄 Breaking Changes

- **Simplified `generate_image()` API** (programmatic users only):
  - Removed `Config` class - parameters are now passed directly to `generate_image()`
  - Removed `RuntimeConfig` class - internal complexity eliminated
  - Added `Flux1` export to main `mflux` module for cleaner imports
- **LoRA API simplified** (programmatic users only):
  - Removed `lora_names` and `lora_repo_id` parameters from all model classes (`Flux1`, `QwenImage`, `QwenImageEdit`, etc.)
  - Removed `--lora-name` and `--lora-repo-id` CLI arguments
  - Removed `LoRAHuggingFaceDownloader` class

### 🔄 Breaking Changes (CLI)

- **`--path` flag removed**: The deprecated `--path` flag for loading models has been removed. Use `--model` instead for local paths, HuggingFace repos, or predefined model names.

### 📦 Dependency Updates

- **Updated `huggingface-hub`** from `>=0.24.5,<1.0` to `>=1.1.6,<2.0`
  - v1.1.6 includes fix for incomplete file listing in `snapshot_download` which could cause cache corruption
  - Removed explicit `accelerate` and `filelock` dependencies (pulled in as transitive dependencies)
- **Updated `transformers`** from `>=4.57,<5.0` to `>=5.0.0rc0,<6.0`
  - Required for `huggingface-hub` 1.x compatibility
  - Added workaround for `Qwen2Tokenizer` bug in transformers 5.0.0rc0 where vocab/merges files are not loaded correctly via `from_pretrained()`

### 🐛 Bug Fixes

- **Qwen empty negative prompt crash**: Fixed crash when running Qwen models without a `--negative-prompt` argument. Empty prompts now use a space as fallback to ensure valid tokenization.

- **`--model` flag not working**: Fixed bug where the `--model` argument wasn't being used for loading models from HuggingFace or local paths. All CLI commands now correctly use `--model` for model path resolution.
- **Model Saving Index File**: Fixed issue where locally saved models (via `mflux-save`) would fail to load when uploaded to HuggingFace, due to missing `model.safetensors.index.json`. The model saver now generates this index file alongside the safetensor shards, ensuring compatibility with both mflux and standard HuggingFace loading paths. (see [#285](https://github.com/filipstrand/mflux/issues/285))

### 🧪 Test Infrastructure

- **Test markers**: Added `fast` and `slow` pytest markers to categorize tests
  - Fast tests: Unit tests that don't generate images (parsers, schedulers, resolution, utilities)
  - Slow tests: Integration tests that generate actual images and compare to references
- **New Makefile targets**:
  - `make test-fast` - Run fast tests only (quick feedback during development)
  - `make test-slow` - Run slow tests only (image generation tests)
  - `make test` - Run all tests (unchanged)
- Run specific test categories: `pytest -m fast` or `pytest -m slow`
- **GitHub Actions CI**: Fast tests now run automatically on PRs and pushes to main

### 🔧 Internal Changes

- Simplified `WeightLoader.load()` to take a single `model_path` parameter instead of separate `repo_id` and `local_path`
- Simplified `TokenizerLoader.load_all()` with the same unified `model_path` interface
- Renamed `local_path` parameter to `model_path` in all model constructors for clarity
- Removed `quantization_util.py` - quantization now handled through `QuantizationResolution`
- Removed `lora_huggingface_downloader.py` - downloading integrated into `LoRAResolution`
- Added comprehensive test coverage for resolution modules

### 👩‍💻 Contributors

- **Filip Strand (@filipstrand)**: Z-Image Turbo support, architecture improvements, core development

---

## [0.12.1] - 2025-11-27

### 🐛 Bug Fixes

- **FIBO VLM Tokenizer Download**: Fixed an issue where the FIBO VLM tokenizer files would not download automatically when the model weights were cached but tokenizer files were missing. The initializer now properly checks for tokenizer file existence and downloads them if needed.

---

## [0.12.0] - 2025-11-27

# MFLUX v.0.12.0 Release Notes

### 🎨 New Model Support

- **Bria FIBO Support**: Added support for [FIBO](https://huggingface.co/briaai/FIBO), the first open-source JSON-native text-to-image model from [Bria.ai](https://bria.ai)
- **Three operation modes**: Generate (text-to-image with VLM expansion), Refine (structured prompt editing), and Inspire (image-to-prompt extraction)
- **New commands**:
  - `mflux-generate-fibo` - Generate images from text prompts with VLM-guided JSON expansion
  - `mflux-refine-fibo` - Refine images using structured JSON prompts for targeted attribute editing
  - `mflux-inspire-fibo` - Extract structured prompts from reference images for style transfer and remixing
- **VLM-guided JSON prompting**: Automatically expands short text prompts into 1,000+ word structured schemas using a fine-tuned Qwen3-VL model

### 🔧 Restructure and 🔄 Breaking Changes

- **Common module reorganization**: Moved shared functionality to `models/common/` for better code reuse
  - Unified latent creators across model families
  - Centralized scheduler implementations
  - Common quantization utilities
  - Shared model saving functionality

### 👩‍💻 Contributors

- **Filip Strand (@filipstrand)**: FIBO model implementation, architecture, core development

---

## [0.11.1] - 2025-11-13

# MFLUX v.0.11.1 Release Notes

### 🎨 New Model Support

- **Qwen Image Edit Support**: Added support for the Qwen Image Edit model, enabling natural language image editing capabilities
- **New command**: `mflux-generate-qwen-edit` for image editing with text instructions
- **Multiple image support**: Edit images using multiple reference images via `--image-paths` parameter
- **Model**: Uses `Qwen/Qwen-Image-Edit-2509` for high-quality image editing
- **Quantization support**: Full support for quantized models (8-bit recommended for optimal quality)

### 🔧 Improvements

- **Dedicated Qwen Image command**: Added `mflux-generate-qwen` as a dedicated command for Qwen Image model generation. The `mflux-generate` command now only supports Flux models.
- **Image comparison utility refactoring**: Refactored `image_compare.py` into a cleaner class-based structure with static methods
- **Error handling**: Moved `ReferenceVsOutputImageError` to the main exceptions module for better organization

### 🔄 Breaking Changes

⚠️ **Qwen Image Command Change**: The Qwen Image model now requires using the dedicated `mflux-generate-qwen` command instead of `mflux-generate --model qwen`. This provides better separation between Flux and Qwen model families and improves command clarity.

### 👩‍💻 Contributors

- **Filip Strand (@filipstrand)**: Qwen Image Edit model implementation, code refactoring

---

## [0.11.0] - 2025-10-14

# MFLUX v.0.11.0 Release Notes

### 🎨 New Model Support

- **Qwen Image Support**: Added support for the Qwen Image text-to-image model, enabling a new generation of visual content creation
- **New command**: `mflux-generate` now supports Qwen models for image generation
- **Qwen-specific features**: Full LoRA support with Qwen naming conventions, img2img support, and optimized weight handling
- **Qwen-Image-mflux-6bit Model**: Added [filipstrand/Qwen-Image-mflux-6bit](https://huggingface.co/filipstrand/Qwen-Image-mflux-6bit) quantized model to HF

### 🏗️ Major Architecture Improvements

- **Package Restructure**: Complete reorganization of the codebase to support multiple model architectures
  - Moved from flat structure to organized `models/` hierarchy (`models/flux/`, `models/qwen/`, `models/depth_pro/`)
  - Better separation of concerns with dedicated model, variant, tokenizer, and weight handler modules
  - Improved maintainability and extensibility for future model additions
- **Namespace Package**: Converted mflux to a namespace package (in preparation for mflux.mcp extension)
- **Common Module**: Extracted shared functionality into `models/common/` for better code reuse
  - Unified LoRA handling across different model types
  - Shared attention utilities
  - Common download and weight management utilities

### 📊 Metadata Enhancements

- **XMP/IPTC Metadata Support**: Added comprehensive metadata support for professional workflows
  - Write XMP and IPTC metadata to generated images
  - Industry-standard metadata formats for better compatibility with professional image tools
  - Enhanced metadata reading and writing capabilities
- **New `mflux-info` command**: Display detailed metadata information from generated images
  - View generation parameters, model information, and settings
  - Extract metadata from any mflux-generated image
  - Professional-grade metadata inspection

### 🔧 Scheduler System

- **Scheduler Interface**: Introduced a new scheduler abstraction for better extensibility
  - Clean interface for implementing custom sampling schedulers
  - Foundation for future scheduler additions (Euler, DPM++, etc.)
  - Current implementation: Linear scheduler (existing behavior preserved)
- **Scheduler Selection**: Added `--scheduler` command-line argument for choosing schedulers

### 🐛 Bug Fixes

- **Non-Quantized Model Loading**: Fixed critical bug where locally saved non-quantized models failed to load properly
- **Model Weight Handling**: Improved weight loading reliability for edge cases

### 🔧 Developer Experience

- **MLX 0.29.2 Support**: Updated MLX dependency to support the latest version (mlx>=0.27.0,<0.30.0)
- **Python 3.13 Support**: Unblocked sentencepiece and torch dependencies for Python 3.13
  - Updated dependency specifications for better Python 3.13 compatibility
  - Ensured smooth experience on latest Python versions
- **Test Improvements**: Enhanced image comparison logic to allow similar images that are "close enough"
  - More robust test suite that accommodates minor numerical differences
  - Reduced false positives in image generation tests
- **CI Updates**: Removed Claude CI agent (replacement coming soon)

### 🔄 Breaking Changes

⚠️ **Import Path Changes**: Due to the package restructure, some internal import paths have changed. If you're using mflux as a library and importing internal modules directly, you may need to update your imports:
- Flux modules moved from `mflux.flux.*` to `mflux.models.flux.*`
- Common utilities moved to `mflux.models.common.*`
- CLI tools remain unchanged and fully backward compatible

### 👩‍💻 Contributors

- **Filip Strand (@filipstrand)**: Qwen model support, package restructure, core development
- **Alessandro Rizzo (@azrahello)**: XMP/IPTC metadata support, info command implementation
- **Anthony Wu (@anthonywu)**: Scheduler interface, namespace package conversion, Python 3.13 improvements, bug fixes

---

## [0.10.0] - 2025-08-04

# MFLUX v.0.10.0 Release Notes

### 🎨 Model Improvements

- **FLUX.1 Krea [dev] Support!**
- **FLUX.1-Krea-dev-mflux-4bit Model**: Added [filipstrand/FLUX.1-Krea-dev-mflux-4bit](https://huggingface.co/filipstrand/FLUX.1-Krea-dev-mflux-4bit) quantized model to HF
- **FLUX.1-Kontext-dev-mflux-4bit Model**: Added [akx/FLUX.1-Kontext-dev-mflux-4bit](https://huggingface.co/akx/FLUX.1-Kontext-dev-mflux-4bit) quantized model to HF, contributed by @akx

### ✨ New Features

- **5-bit Quantization Support**: Added support for 5-bit quantization as a new option alongside existing 3, 4, 6, and 8-bit quantization levels

### 🔧 Improvements

- **Enhanced Default Inference Steps**: Increased default inference steps for dev models from 14 to 25 for improved image quality
- **Multiple Model Aliases Support**: Improved model configuration system to properly support multiple aliases per model, making model selection more flexible and robust

### 🐛 Bug Fixes

- **LoRA Resume Training**: Fixed critical bug where adapters created after training interruption would fail to load for generation with `AttributeError: 'list' object has no attribute 'weight'`. The issue occurred because the resume loading logic wasn't properly handling layers that are legitimately lists in the transformer architecture (like `attn.to_out`). (see [#224](https://github.com/filipstrand/mflux/issues/224))

### 🔧 Technical Requirements

- **MLX Compatibility**: This release assumes MLX 0.27.0 and upwards for optimal performance and compatibility
- **MLX Compatibility for test**: Fix MLX version to 0.27.1 for image generation tests
- **Non-strict Weight Updates**: Explicitly added non-strict mode (`strict=False`) for weight updates to maintain compatibility with later MLX versions that enforce stricter weight validation by default

### 👩‍💻 Developer Experience

- **Streamlined Release Process**: Removed TestPyPi publishing step from release workflow for simplified deployment

### 🙏 Contributors

- **[@filipstrand](https://github.com/filipstrand)** - FLUX.1 Krea [dev] model support, 5-bit quantization, enhanced defaults, and various improvements
- **[@akx](https://github.com/akx)** - Added 4-bit quantized Kontext model to HF

---

## [0.9.6] - 2025-07-20

# MFLUX v.0.9.6 Release Notes

### 🔧 Technical Details

- Cap the upper MLX dependency to a known working version (0.26.1) to avoid compatibility issues with newer MLX releases that enforce stricter weight validation (see [#238](https://github.com/filipstrand/mflux/pull/238))

## [0.9.5] - 2025-07-17

# MFLUX v.0.9.5 Release Notes

### 🐛 Bug Fixes

- **Fixed faulty imports**: Corrected import issues in the mflux module to ensure proper package initialization and functionality

## [0.9.4] - 2025-07-17

# MFLUX v.0.9.4 Release Notes

### 🛠️ Dependency Updates

- Expanded MLX dependency range from `mlx>=0.22.0,<=0.26.1` to `mlx>=0.22.0,<0.27.0` to support newer MLX versions

### 🔧 Developer Experience

- Refactor the release script into a reusable Python module for better maintainability

## [0.9.3] - 2025-07-08

# MFLUX v.0.9.3 Release Notes

### 😖 Revert "Offline Resilience" change

On a "cold start" where user has not previously downloaded the requested model, the workflow does not successfully request the download of all the expected files, blocking the image generation workflow for first time users. The feature will be re-evaluated carefully after this hot fix.

## [0.9.2] - 2025-07-08

# MFLUX v.0.9.2 Release Notes

### 🏗️ Build System Improvements

- **Updated build backend**: Migrated from setuptools to modern `uv build` backend for faster and more reliable package builds
- **Enhanced artifact exclusion**: Optimized distribution packages by excluding documentation assets (~27MB) and example images (~5MB) from published packages
- **New `make build` command**: Added development build command for testing distribution packages and validating sizes

### 🗃️ Offline Resilience

- **Local-first behavior**: Implemented cache-first downloading to improve resilience when HuggingFace Hub or network connectivity is unavailable
- **Graceful fallback**: System automatically uses cached model files when available, falling back to downloads only when necessary
- **Improved reliability**: Enhanced model loading reliability in environments with unstable internet connections

### 🔧 Developer Experience

- **Release script improvements**: Enhanced release automation with better error handling and duplicate version detection
- **Build system fixes**: Fixed minor typos in Makefile that could cause build issues

## Contributors

- **Anthony Wu (@anthonywu)**: Build system modernization, offline resilience implementation
- **Filip Strand (@filipstrand)**: Release automation improvements, build fixes

---

## [0.9.1] - 2025-07-04

# MFLUX v.0.9.1 Release Notes

### 🛠️ Dependency Fixes

- Restricted MLX dependency upper bound to **0.26.1** (`mlx>=0.22.0,<=0.26.1`) to prevent incompatibility issues with MLX 0.26.2.

### 🎨 Inpaint Mask Tool Improvements

- Enhanced interactive inpaint masking tool with additional shape options (ellipse, rectangle, and free-hand drawing).
- Added eraser mode for precise mask corrections.
- Implemented undo/redo history for non-destructive editing when crafting masks.

### 👩‍💻 Developer Experience

- Introduced initial `mypy` static-type checking configuration and performed a first round of type-hint clean-up across the codebase.
- Upgraded *pre-commit* hooks and addressed newly surfaced lint warnings for a cleaner commit experience.

## Contributors

- **Filip Strand (@filipstrand)**
- **Anthony Wu (@anthonywu)**

---

## [0.9.0] - 2025-06-28

# MFLUX v.0.9.0 Release Notes

## Major New Features

### 📸 FLUX.1 Kontext

- **Added FLUX.1 Kontext support**: Official Black Forest Labs model for character consistency, local editing, and style reference
- **New command**: `mflux-generate-kontext` for image-guided generation with text instructions
- **Advanced image editing capabilities**: Sequential editing, style transfer, character consistency, and local modifications
- **Comprehensive documentation**: Detailed prompting guide with tips, templates, and best practices
- **Automatic model handling**: Uses `dev-kontext` model configuration with optimized defaults

### 🖼️ Scale Factor Support for Image Upscaling

- **Enhanced upscaling dimensions**: Added support for scale factors (e.g., `2x`, `1.5x`) in addition to absolute pixel values
- **Mixed dimension types**: Ability to combine scale factors and absolute values (e.g., `--height 2x --width 1024`)
- **Auto dimension handling**: Use `auto` to preserve original image dimensions
- **Safety warnings**: Automatic warnings when requested dimensions exceed recommended limits
- **Pixel-perfect scaling**: Scale factors automatically align to 16-pixel boundaries for optimal results

### ⌨️ Shell Completions

- **ZSH completion support**: Full tab completion for all mflux CLI commands and arguments
- **Smart completions**: Context-aware completions for model names, quantization levels, LoRA styles, and file paths
- **Easy installation**: Simple `mflux-completions` command for automatic setup
- **Dynamic generation**: Completions stay in sync with code changes and new commands
- **Comprehensive coverage**: Supports all 15+ mflux commands with proper argument validation

### 🗂️ Cache Management Improvements

- **Platform-native caching**: Uses `platformdirs` for macOS-idiomatic cache locations (`~/Library/Caches/mflux/`)
- **Automatic migration**: Seamless migration from legacy `~/.cache/mflux` to new platform-appropriate locations
- **Environment variable support**: `MFLUX_CACHE_DIR` for custom cache locations
- **Improved organization**: Separate cache directories for different types of data (models, LoRAs, etc.)
- **Backward compatibility**: Automatic symlink creation for legacy path compatibility

## Breaking Changes

### 🔧 Python API Class Naming Standardization

- **Class rename**: `FluxInContextFill` is now `Flux1InContextFill` to follow consistent naming convention
- **Class rename**: `FluxConceptFromImage` is now `Flux1ConceptFromImage` to follow consistent naming convention
- **Breaking change for library users**: If you import these classes directly in Python code, you may need to update your imports
- **CLI tools unaffected**: All command-line tools (`mflux-generate-*`) continue to work without changes

## Contributors

Contributors:
- **Anthony Wu (@anthonywu)**: Scale factor support, shell completions, cache refactor
- **Filip Strand (@filipstrand)**: Kontext support, class naming standardization, core development

## [0.8.0] - 2025-06-14

# MFLUX v.0.8.0 Release Notes

## Experimental AI Features

### 👗 CatVTON (Virtual Try-On)
- **[EXPERIMENTAL]** Added virtual try-on capabilities using in-context learning via `mflux-generate-in-context-catvton`
- Support for person image, person mask, and garment image inputs for comprehensive virtual clothing try-on
- Automatic prompting for virtual try-on scenarios with optimized default prompts
- Side-by-side generation showing garment product shot alongside styled result
- AI-powered virtual clothing fitting with realistic lighting and fabric properties

### ✏️ IC-Edit (In-Context Editing)
- **[EXPERIMENTAL]** Added natural language image editing capabilities via `mflux-generate-in-context-edit`
- Natural language image editing using simple text instructions like "make the hair black" or "add sunglasses"
- Automatic diptych template formatting for optimal editing results
- Optimal resolution auto-sizing for 512px width (the resolution IC-Edit was trained on)
- Specialized LoRA automatically downloaded and applied for enhanced editing capabilities

## Enhanced Generation Control

### 🔎 Image Upscaling
- **Built-in upscaling capabilities**: Enhanced image quality and resolution enhancement for generated images
- Seamless integration with existing generation workflow
- Professional-grade upscaling for production-ready outputs

## Interpretability research

### 🧠 Concept Attention
- **Enhanced image generation control**: Fine-grained control over image generation focus areas using attention-based concepts
- Improved composition and subject handling for more precise artistic direction
- Advanced attention mechanisms for better understanding of prompt concepts

## Workflow & Performance Improvements

### 🪫 Battery Saver
- **Power management**: Automatic power optimization during extended generation sessions
- Configurable power-saving modes specifically designed for laptop users
- Smart resource management for long-running batch operations

### 📝 Prompt File Support
- **File-based prompt input**: Batch operations via `--prompt-file` for large-scale generation projects
- Dynamic prompt updates for large batch generation workflows
- Support for external prompt management and automation systems

### 🔄 Redux Function Balancing
- **Enhanced Redux capabilities**: Improved control over image-to-image transformation strength
- Better quality variations with adjustable parameters for more predictable results
- Refined Redux algorithm for more natural image variations

### 📥 Stdin Prompt Support
- **LLM Integration Ready**: Added support for providing prompts via stdin using `--prompt -`
- Enables seamless integration with LLMs and other text generation tools
- Supports both single-line and multi-line prompts through stdin
- Perfect for automation workflows and dynamic prompt generation
- Example usage: `echo "A beautiful landscape" | mflux-generate --prompt -`

## Developer Experience

### 🔧 LORA_LIBRARY_PATH Improvements
- **Unix-style resource discovery**: Enhanced LoRA library path handling for better organization
- Improved path handling for LoRA weight discovery across multiple directories
- Better cross-platform compatibility for LoRA management

### 🧪 Testing & Documentation
- New command-line arguments for both experimental features with comprehensive help
- Comprehensive argument parser tests for new functionality
- Updated documentation with experimental feature warnings and usage guidelines
- Added note about upcoming FLUX.1 Kontext model from Black Forest Labs

## Architecture Improvements

### 📚 Documentation Structure
- Refactored "In-Context LoRA" section to "In-Context Generation" with clear subcategories
- Enhanced documentation structure for better organization and user navigation
- Improved categorization of experimental vs stable features

### 🔄 Code Architecture Changes
- **Class rename**: `Flux1InContextLora` is now `Flux1InContextDev` to better reflect the dev model variant
- **Module reorganization**: Moved from `mflux.community.in_context_lora.flux_in_context_lora` to `mflux.community.in_context.flux_in_context_dev`
- **Breaking change for library users**: If you import the class directly, update your imports accordingly


### ⚡ Performance Optimizations
- Updated MLX dependency to latest version for improved performance and stability
- Removed PyTorch dependency for DepthPro model, significantly reducing installation requirements
- Streamlined dependencies for faster installation and reduced disk usage

## Experimental Notice

⚠️ **Important**: CatVTON and IC-Edit features are experimental and may be removed or significantly changed in future updates. These features represent cutting-edge AI capabilities that are still under active development.

## Contributors

Special thanks to the following contributors for their exceptional work since v0.7.1:
- **Anthony Wu (@anthonywu)**: Battery Saver implementation, Prompt File Support, Stdin Prompt Support, LORA_LIBRARY_PATH improvements
- **Alessandro (@azrahello)**: Redux Function Balancing enhancements
- **Filip Strand (@filipstrand)**: Core development, experimental features integration, infrastructure improvements

## [0.7.1] - 2025-05-06

# MFLUX v.0.7.1 Release Notes

## New Features

### 🎭 Multi-LoRA Support
- **Multiple LoRA Loading**: Added support for loading multiple LoRA adapters simultaneously when using the in-context feature
- Enhanced creative flexibility by combining multiple artistic styles in a single generation
- Reference: [GitHub Issue #178](https://github.com/filipstrand/mflux/issues/178)

## [0.7.0] - 2025-04-25
# MFLUX v.0.7.0 Release Notes

## Major New Features

### 🖌️ FLUX.1 Tools | Fill

- Added support for the FLUX.1-Fill model for inpainting and outpainting
- Introduced `mflux-generate-fill` command-line tool for selective image editing
- Implemented interactive mask creation tool to easily mark areas for regeneration
- Added outpainting capabilities with customizable canvas expansion
- Includes helper tools for creating outpaint image canvases and masks

### 🔍 FLUX.1 Tools | Depth

- Added support for the FLUX.1-Depth model for depth-conditioned image generation
- Implemented Apple's ML Depth Pro model in MLX for state-of-the-art depth map extraction
- Added `mflux-generate-depth` and `mflux-save-depth` command-line tools
- Added ability to use either auto-generated depth maps or custom depth maps

### 🔄 FLUX.1 Tools | Redux

- Added Redux tool as a new image variation technique
- Implemented a different approach compared to standard image-to-image generation
- Uses image embedding joined with T5 text encodings for more natural variations
- Added Redux-specific weight handlers and initialization

## New Models

### 🔎 Apple ML Depth Pro

- Added native MLX implementation of Apple's ML Depth Pro model for both separate use, and as a part of the Depth tool functionality

### 🖼️ Google SigLIP Vision Transformer

- Added SigLIP vision model for the Redux functionality

## Architecture Improvements

### 💾 Weight Management Improvements

- Added support for saving MFLUX version information in model metadata

### 🧠 Memory Optimization

- Additional improvements to the `--low-ram` option
- Better memory management for image generation models

## Contributors

- @anthonywu 
- @ssakar 
- @akx 

## [0.6.2] - 2025-03-13

# MFLUX v.0.6.2 Release Notes

## Bug Fixes

### 💾 Model Saving Fix
- **Fixed local model saving**: Resolved bug preventing users from saving models locally with `mflux-save`
- Restored full functionality for local model storage and management

## [0.6.1] - 2025-03-11

# MFLUX v.0.6.1 Release Notes

## Bug Fixes

### 🛑 Image Generation Interruption
- **Fixed interruption flow**: Properly handles interruptions during image generation, ensuring graceful stops even when no callbacks are registered
- **Keyboard interrupt handling**: Ensures image generation can be stopped via Ctrl+C in all diffusion model variants (standard Flux, ControlNet, and In-Context LoRA)
- Relocated `StopImageGenerationException` from stepwise handler to main generation functions for more robust interruption system

## Test Stability Improvements

### 🧪 Test Reliability
- **Fixed sporadic test failures**: Resolved intermittent failures in auto-seeds test case when using random seed count of 1
- Improved test consistency and reliability

## Code Quality Improvements

### 🔧 Code Standards
- **Formatting and linting fixes**: Fixed various formatting issues that were missed in the v0.6.0 release
- Enhanced code consistency and maintainability

## [0.6.0] - 2025-03-05
# MFLUX v.0.6.0 Release Notes

## Major New Features

### 🌐 Third-Party HuggingFace Model Support
- Comprehensive ModelConfig refactor to support compatible HuggingFace dev/schnell models
- Added ability to use models like `Freepik/flux.1-lite-8B-alpha` and `shuttleai/shuttle-3-diffusion`
- New `--base-model` parameter to specify which base architecture (dev or schnell) a third-party model is derived from
- Maintains backward compatibility while opening up the ecosystem to community-created models

### 🎭 In-Context LoRA
- Added support for In-Context LoRA, a powerful technique that allows you to generate images in a specific style based on a reference image without requiring model fine-tuning
- Introduced a new command-line tool: `mflux-generate-in-context`
- Includes 10 pre-defined styles from the Hugging Face ali-vilab/In-Context-LoRA repository
- Detailed documentation on how to use this feature effectively with prompting tips and best practices

### 🔌 Automatic LoRA Downloads
- Added ability to automatically download LoRAs from Hugging Face when specified by repository ID
- Simplifies workflow by eliminating the need to manually download LoRA files before use

### 🧠 Memory Optimizations
- Added `--low-ram` option to reduce GPU memory usage by constraining the MLX cache size and releasing text encoders and transformer components after use
- Implemented memory saver for ControlNet to reduce RAM requirements
- General memory usage optimizations throughout the codebase

### 🗜️ Enhanced Quantization Options
- Added support for 3-bit and 6-bit quantization (requires mlx > v0.21.0)
- Expanded quantization options now include 3, 4, 6, and 8-bit precision

## ⚠️Breaking changes

Previously saved quantized models will not work for v.0.6.0 and later.  See #149 for more details.

## Interface Improvements

### 🔧 Modified Parameters

- The previous `--init-image-path` parameter is now `--image-path` 
- The previous `--init-image-strength` parameter is now `--image-strength` 

### 🖼️ Image Generation Enhancements
- Added `--auto-seeds` option to generate multiple images with random seeds in a single command
- Added option to override previously saved test images
- Added `--controlnet-save-canny` option to save the Canny edge detection reference image used by ControlNet
- Improved handling of edge cases for img2img generation

### 🔄 Callback System
- Implemented a general callback mechanism for more flexible image generation pipelines
- Added support for before-loop callbacks to accept latents
- Enhanced StepwiseHandler to include initial latent

## Architecture Improvements

### 🏗️ Code Refactoring
- Removed 'init' prefix for a more general interface
- Removed `ConfigControlnet` - the `controlnet_strength` attribute is now on `Config`
- Simplified quantization by removing unnecessary class predicates 
- Refactored model configuration system
- Refactored transformer blocks for better maintainability
- Unified attention mechanism in single and joint attention blocks
- Added support for variable numbers of transformer blocks
- Optimized with fast SDPA (Scaled Dot-Product Attention)
- Added PromptCache for small optimization when generating with repeated prompts

### 🧰 Developer Tools
- Added Batch Image Renamer tool as an isolated uv run script
- Added descriptive comments for attention computations

## Compatibility Updates
- Updated to support the latest mlx version
- Fixed compatibility issues with HuggingFace dev/schnell models

## Bug Fixes
- Fixed handling of edge cases for img2img generation
- Various small fixes and improvements throughout the codebase


## Contributors

- @anthonywu
- @ssakar
- @azrahello
- @DanaCase

## [0.5.1] - 2024-12-23

# MFLUX v.0.5.1 Release Notes

## Bug Fixes

### 🔧 LoRA Loading Fix
- **Quantized model LoRA compatibility**: Fixed critical bug where locally saved quantized models failed to set LoRA weights
- Users can now successfully combine local quantized models with external LoRA adapters
- Improved reliability for advanced workflows combining quantization and LoRA fine-tuning

## [0.5.0] - 2024-12-22

# MFLUX v.0.5.0 Release Notes

## Major New Features

### 🎛️ DreamBooth Fine-tuning
- **DreamBooth support**: Introduced V1 of fine-tuning support in MFLUX
- Enables custom model training for personalized image generation
- Full fine-tuning pipeline with training configuration options

## Architecture Improvements

### 🔧 Weight Management Overhaul
- **Rewritten LoRA handling**: Completely rewritten LoRA weight handling system
- Improved performance and reliability for LoRA operations
- Better support for complex LoRA workflows

## Developer Experience

### 🧪 Testing & Quality
- **Enhanced test coverage**: Added comprehensive tests for new and existing features
- Multi-LoRA testing support
- Local model saving test coverage

### 📊 New Dependencies
- **Matplotlib integration**: Added matplotlib for visualizing training loss during fine-tuning
- **TOML support**: Added TOML library for better handling of MFLUX version metadata
- Enhanced configuration management

## [0.4.1] - 2024-10-29

# MFLUX v.0.4.1 Release Notes

## Bug Fixes

### 🐛 Image Generation Fixes
- **Img2img resolution fix**: Fixed img2img functionality for non-square image resolutions
- Improved compatibility with various aspect ratios

## [0.4.0] - 2024-10-28

# MFLUX v.0.4.0 Release Notes

## Major New Features

### 🖼️ Image-to-Image Generation
- **Img2Img Support**: Introduced the ability to generate images based on an initial reference image
- Transform existing images using AI-powered generation techniques
- Control the strength of transformation to balance between original image preservation and creative generation
- Perfect for iterating on designs and creating variations of existing artwork

### 📊 Metadata-Driven Generation
- **Image Generation from Metadata**: Added support to generate images directly from provided metadata files
- Streamlined workflow for recreating images with specific parameters
- Enhanced reproducibility for professional and research workflows
- Automated parameter loading from previously generated images

### 🔍 Real-time Generation Monitoring
- **Progressive Step Output**: Optionally output each step of the image generation process for real-time monitoring
- Visual feedback during generation for better understanding of the AI process
- Debug and fine-tune generation parameters by observing intermediate steps
- Educational tool for understanding diffusion model progression

## Developer Experience Improvements

### 🛠️ Enhanced Command-Line Interface
- **Improved argument handling**: Enhanced parsing and validation for command-line arguments
- Better error messages and user guidance for parameter configuration
- More intuitive command structure for complex generation workflows

### 🧪 Testing & Quality Assurance
- **Automated Testing**: Added comprehensive automatic tests for image generation and command-line argument handling
- Improved reliability and stability for all generation modes
- Continuous integration testing for better code quality

### 🔧 Development Workflow
- **Pre-Commit Hooks**: Integrated pre-commit hooks with `ruff`, `isort`, and typo checks for better code consistency
- Enhanced developer experience with automated code quality checks
- Streamlined contribution process for open source development

## [0.3.0] - 2024-09-24

# MFLUX v.0.3.0 Release Notes

## Major New Features

### 🕹️ ControlNet Support
- **ControlNet Canny support**: Added Canny edge detection ControlNet functionality for precise image control
- Enhanced control over image generation with edge-guided conditioning

## Model Export Improvements

### 📦 Advanced Model Export
- **Quantized model export with LoRA**: Added ability to export quantized models with LoRA weights baked in
- Streamlined deployment for fine-tuned models

## Developer Experience

### 🛠️ Development Tools
- **Enhanced development workflow**: Improved developer experience with uv, ruff, makefile, pre-commit hooks
- Better code quality tools and automated checks
- Streamlined contribution process

## Legal & Licensing

### ⚖️ Open Source License
- **Official MIT license**: Established clear open source licensing for the project
- Legal clarity for users and contributors

## [0.2.1] - 2024-09-14

# MFLUX v.0.2.1 Release Notes

## Improvements

### 🔧 LoRA Enhancements
- **Enhanced LoRA support**: Improved compatibility and performance for LoRA weight loading
- Better integration with existing workflows
- Refined handling of LoRA adapters

## [0.2.0] - 2024-09-07

# MFLUX v.0.2.0 Release Notes

## Major Milestone

### 🚀 Official PyPI Release
- **First official PyPI release**: `pip install mflux` - making MFLUX easily installable for everyone
- Big thanks to @deto for letting us have the "mflux" name on PyPI!

## New Features

### 🎨 Core Image Generation
- **Command-line tools**: Introduced dedicated commands for better user experience
  - `mflux-generate` for generating images
  - `mflux-save` for saving quantized models to disk
- **🗜️ Quantization support**: Added support for quantized models with 4-bit and 8-bit precision
- **LoRA weights**: Added support for loading trained LoRA (Low-Rank Adaptation) weights
- **Automatic metadata**: Images now automatically save metadata when generated

## Developer Experience

### 📦 Distribution
- Official packaging and distribution through PyPI
- Simplified installation process for end users
- Professional project structure and naming
