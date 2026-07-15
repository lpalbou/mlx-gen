# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.20.0] - 2026-07-15

This release ships two image/video editing expansions: the natively ported Wan2.1-VACE-1.3B
conditioning route and FLUX.2 Klein masked edit.

### Added

- **FLUX.2 Klein masked edit / inpaint** (`--mask-path` on all FLUX.2 Klein image-to-image
  routes, distilled and base): localized edits that repaint white mask pixels and preserve
  black pixels through per-step latent compositing, ported from the diffusers
  `Flux2KleinInpaintPipeline` semantics (clean source latents ride along as conditioning
  tokens at reserved grid coordinates; the mask is binarized at pixel resolution and then
  bilinear-interpolated to the packed latent grid with torch-`F.interpolate` parity).
  Unified `mlxgen generate --image ... --mask-path ...` selects the new `flux2.inpaint`
  capability (one source image; `--image-strength` stays rejected with a mask). The backend
  `mflux-generate-flux2-edit` command and the Python `Flux2KleinInpaint` runtime additionally
  accept extra images as masked-area references (diffusers `image_reference` parity: source
  conditioning at t=10, references at t=20+). Base Klein models default to guidance 4.0 with
  true CFG on this route; distilled Klein models stay at guidance 1.0. The route is
  smoke-validated on the exact `AbstractFramework/flux.2-klein-4b-8bit` and
  `flux.2-klein-base-4b-8bit` packages (masked lens recolor: mean abs pixel diff 2.37/255
  outside the mask vs 55.93/255 inside, plus a reference-conditioned fill case); a published
  visual-QA proof row remains follow-up work, so `mlxgen validation` does not yet list
  masked-edit rows for Klein packages.

- **Native Wan2.1-VACE-1.3B port** (`Wan-AI/Wan2.1-VACE-1.3B-diffusers`, alias `wan-vace` -
  VACE only exists as a Wan2.1 release; this is the runtime's first Wan2.1 model):
  reference-image-guided generation (`--reference-image`, repeatable - inject a pictured
  object/subject into a new scene) and learned masked source-video editing
  (`--video-path` + `--video-mask-path`), with `--conditioning-scale` controlling the VACE
  branch strength. Single-transformer Wan2.1 backbone with 15 VACE control blocks, shared
  UMT5/wan21-VAE/UniPC runtime; `--video-strength` and `guidance_2` are rejected on VACE
  models (no SDEdit warm start, no boundary routing), and the planner reports the
  capability accordingly. `--vace-masked-region` selects the masked-edit semantics:
  `generate` (default) gray-fills the editable region per the official VACE inpainting
  convention so the model synthesizes new structure there; `repaint` keeps the source
  content as conditioning for restyle-in-place edits (both recorded in metadata and
  replayed). Ported stage-by-stage against the diffusers reference: mask-channel
  preparation and the UniPC schedule are bit-exact, transformer deltas sit at the model's
  intrinsic fp32 sensitivity floor (verified by a noise-injection probe), and the bounded
  CFG-loop error matches the analytic guidance amplification model. Capability proof bundle
  in `docs/assets/validation/wan-vace-2026-07-06/` with request/inputs/output panels and
  controls: a masked object REPLACEMENT judged against strict vision criteria (generate
  mode: silhouette IoU vs source 0.16-0.20 where repaints measure 0.73-0.88, in-mask change
  63.8 vs a 3.8-3.9 codec-floor background; repaint mode restyles in place at IoU 0.80-0.88;
  the upstream pipeline fed un-blanked inputs fails identically to repaint while drifting
  the whole frame, 2161 s CPU), a reference-injection identity proof via a same-seed
  no-reference control (segmented subject on white transfers; crops and full-scene
  references measurably fail - documented as user guidance), a flag-free defaults run
  (832x480x81f/30 steps: 1 h 56 min, 31.7 GiB peak), and the bf16 + fp32 parity
  comparisons; parity tools in `tools/wan_vace_parity_export.py` /
  `tools/wan_vace_parity_compare.py`.

## [0.19.0] - 2026-07-06

This release folds in the never-published 0.18.26 truth patch (its corrections are listed
under Fixed below) and ships the video-to-video temporal/audio features, the router and Wan
structural cleanup, and the measured motion-fidelity ladder.

### Added

- **Measured motion-fidelity ladder for Wan video-to-video** in `docs/wan-video.md`: a
  strength-vs-gesture-preservation table backed by a same-seed 20-step proof matrix (0.5 / 0.6 /
  0.7 / 0.8) with per-run metrics, contact sheets, and a null-row metric floor, plus a paired
  Lightning-point control run showing prompt gesture language recovers the class of motion but
  not its timing. Proof assets in `docs/assets/validation/motion-ladder-2026-07-05/`. The FAQ's
  strength advice now cites the measured band (gestures survive at 0.5-0.6, r 0.86-0.90; the
  0.8 default re-synthesizes them, r 0.20), and the docs state plainly that the 4-step Lightning
  fast path and the motion-preserving band are mutually exclusive.

- **`metadata_schema_version` in image and video metadata** (sidecar and embedded), starting at
  `1`, with a documented additive-only evolution policy in `docs/api.md`. Consumers can now
  detect structural metadata changes without parsing `mflux_version`.

- **Wan video-to-video audio copy-through**: when the source clip has audio, the matching
  segment (trimmed to the output duration) is copied onto the saved MP4, in both regular and
  `--low-ram` batch save paths. Best-effort by design: on failure the video is saved silent, a
  warning prints the reason and a manual `ffmpeg` remux command, and the sidecar records
  `audio_present` / `audio_copied` / `audio_copy_mode` / `audio_copy_reason`. Unlike the strict
  SeedVR2 restore contract, a failed mux never discards a finished generation (documented in the
  README and `docs/wan-video.md`).

### Changed

- **Router option surface single-sourced**: the `mlxgen generate` parser and its re-emission of
  consumed flags are now driven by one descriptor table (`router_options.py`). A completeness
  test maps every parser action to exactly one descriptor with a declared forwarding fate, and a
  round-trip test asserts each consumed flag reaches the backend (from argv and from metadata),
  so a future consumed-but-unforwarded flag fails CI instead of silently running defaults.
  Backend parsers, planner constraints, and metadata replay are not unified in this pass.
- **Shell completions coverage tested against `pyproject` scripts**: completions now exist for
  `mflux-generate-wan`/`mlxgen-generate-wan`, `mflux-generate-ernie-image`, and
  `mflux-generate-bonsai` (generated from the real entrypoint parsers); a truth test fails when
  a new console script ships without a completion or a documented exclusion. The `mlxgen`
  router aliases are excluded until subcommand-aware completion exists.
- **User-mask loading centralized in `MaskUtil`**: Qwen edit inpaint, Qwen control inpaint,
  Z-Image inpaint, and Wan masked video-to-video now share one loader with an explicit,
  documented resampling policy (reference-ported surfaces keep their reference's resampling -
  NEAREST for the diffusers-ported paths; in-house Wan masked V2V uses BOX). Pixel behavior is
  unchanged. Each surface now warns once per generation when a mask carries an alpha channel
  (previously only Wan did).
- **Wan runtime decomposition, phase 1**: the `generate_video` validation/resolution head moved
  into `WanVideoRequest.resolve` and the twin per-branch metadata blocks collapsed into one
  shared builder, removing the duplication where the 0.18.25 `steps` replay bug lived. No
  behavior change; all helpers remain on the model class.
- **Wan video-to-video temporal contract**: the source clip is now resampled onto the requested
  `--fps` timeline at decode, so the output keeps real-time speed regardless of the source frame
  rate (previously the first `--frames` source frames were consumed as-is and re-timed, changing
  playback speed on fps mismatch - a limitation the truth-patch band documented and this
  release removes). Matching source/requested fps passes frames through untouched
  (bit-identical with the 0.18.25 decode behavior).
  Downsampling prints an informational note; upsampling above the source fps duplicates frames
  and prints a warning. Metadata gains `source_video_resampled`; the V2V latent cache key now
  includes fps. Frame-exact first-N extraction is available by re-encoding the source to the
  target fps beforehand.

### Fixed

- **`mlxgen generate --debug` was silently dropped**: the router consumed the flag without
  re-emitting it, so backends never enabled debug logging (the help text claimed otherwise).
  The flag is now re-emitted to all routes, and the Wan CLI gained `--debug` support wired to
  LoRA debug logging. Third instance of the consumed-but-not-forwarded router bug class, now
  structurally closed (see below).
- **Metadata-sourced `video_strength` validated late**: a `video_strength` out of `(0, 1]`
  inside `--config-from-metadata` used to fail only after the multi-minute Wan weight load. The
  router now backfills and re-emits it like `video_mask_path`, so the backend parser rejects
  invalid values at parse time.
- **`mflux-completions` crashed on every invocation**: a duplicated `add_lora_arguments` call in
  the `mflux-upscale-controlnet` completion branch raised `ArgumentError: conflicting option
  string --lora-style` during generation, so completion install and `--print` both failed. The
  duplicate is removed and a truth test now generates every command's parser in CI.
- **`mflux-generate-z-image` shell completion was empty**: the command was listed in the
  completions generator without a parser branch; it now completes the real z-image options.
- **Removed the "motion anchor" overclaim from video-to-video docs**: prose that promised
  the source clip anchors "motion" (gesture-level) is corrected across `docs/wan-video.md`,
  `docs/getting-started.md`, and `docs/faq.md` - camera path, framing, and layout survive;
  subject gestures and timing are re-synthesized at the default strength (the measured ladder
  below quantifies where the transition happens), and the FAQ example prompt no longer
  suggests "keep the same motion" can force motion through.
- **CI test gate**: CI and the release workflow now run the full no-weights band
  (`-m "not slow and not high_memory_requirement"`, 1244+ tests) instead of `-m fast`
  (425 tests). 57% of tests - including every video-to-video contract test - previously ran in no
  CI band, and v0.18.25 shipped with a statically red release-date pin test that CI never
  executed. The red pin is also fixed (`PACKAGED_RELEASE_DATE` now matches the changelog date of the packaged version).
- **Wan q8 memory truth**: `docs/quantization.md` now states that since the 2026-06-12
  runtime-precision fix, Wan q8 packages (A14B and TI2V-5B) dequantize all transformer-block
  linears to BF16 at load - q8 is a storage/download saving only, and runtime memory matches the
  BF16 packages. The stale pre-fix rows are annotated with a dated correction and a re-measured
  MLX-peak value at the exact documented profile (27.8 GiB q8 vs 27.7 GiB BF16); the same
  correction propagates to README, FAQ, recommendations tiers (Wan A14B moves from the 24/32 GB
  tiers to 64 GB), and the llms context files. Separately, runtime metadata gains a
  `darwin_peak_physical_footprint_bytes` lifetime high-water field via `proc_pid_rusage` for
  future runs, and the rusage helper struct was completed to the full v4 layout (the previous
  truncated layout under-allocated the flavor-4 buffer).
- **V2V temporal and audio truth**: removed the false claim that video-to-video keeps "clip
  timing"; `docs/wan-video.md` and `docs/faq.md` now document the exact contract (first-N frames
  re-timed to `--fps`, no temporal resampling, audio track dropped). The runtime now warns with
  the exact speed factor on fps mismatch and warns when the source has an audio track; metadata
  records `source_video_audio_present`.
- **Lightning V2V contract**: the documented recipe now uses the copy-pasteable
  `owner/repo:subdir/file.safetensors` adapter form; `wan.video-video` has a LoRA validation
  registry row (`lora_wan_a14b_q8_lightning_v2v_2026_07_04`), so `mlxgen capabilities` reports the
  documented recipe as `validated` instead of `mapped-unvalidated`; the matrix proof bundle is
  included in-repo under `docs/assets/validation/lightning-v2v-2026-07-04/`.
- **Doc drift**: README no longer advertises LoRA-guide artifacts that were removed in the
  0.18.24 docs refresh; public docs no longer link proof bundles inside the git-ignored
  `validation_outputs/` folder.

## [0.18.25] - 2026-07-05

### Added

- **Masked Wan video-to-video**: new `--video-mask-path` role on the `Wan2.2-T2V-A14B`
  video-to-video route. White mask regions are regenerated under the prompt; black regions are
  locked to the source video at every denoising step (including UniPC corrector state) and match
  the source up to VAE round-trip precision after the final clean composite. Static image masks,
  binarized at 50% on the latent grid; all-black masks fail before model load; recorded in
  metadata and replayed by `--config-from-metadata`. Capabilities schema bumped to version 4
  with `supports_video_mask`.

- **Lightning fast video-to-video recipe**: bounded validation of the `lightx2v/Wan2.2-Lightning`
  T2V-A14B 4-step LoRA pairs (Seko-V1.1 and Seko-V2.0) on the public video-to-video route via
  the on-grid recipe (`--steps 4 --video-strength 0.75 --guidance 1 --flow-shift 5 --solver
  unipc`), including the masked combination whose preserved regions stay at the H.264 re-encode
  floor while cutting the denoise loop from 28 to 3 transformer forwards. Documented with the
  strength-lattice and inert-negative-prompt caveats.

- **Wan plain video-to-video route**: public prompt-guided source-video editing on
  `Wan2.2-T2V-A14B` through `mlxgen generate --video-path ...` with `--video-strength`
  (default `0.8`), unipc-only solver enforcement, fail-closed rejection on TI2V-5B and I2V-A14B,
  and included reproducible proof artifacts under `docs/assets/examples/spaceship-v2v/`.
- **V2V observability**: saved metadata records requested `steps` plus `effective_steps`,
  `video_strength`, `high_noise_stage_skipped`, and source-clip frame count, duration, and fps;
  the runtime warns when a low `video_strength` skips the A14B high-noise stage (making
  `--guidance` inactive) and when source frames are stretched to a mismatched canvas.

### Fixed

- **Router `--video-strength` forwarding**: `mlxgen generate` consumed `--video-strength` for
  validation but never forwarded it to the Wan backend, so every V2V run silently used the
  `0.8` default; the router now re-emits the flag and rejects out-of-range values before any
  model work.
- **V2V metadata replay**: metadata previously recorded the strength-truncated step count as
  `steps`, so `--config-from-metadata` replays shrank the schedule on every round trip; `steps`
  now records the requested count.
- **Early source-video validation**: unreadable or too-short `--video-path` inputs now fail
  before the multi-minute A14B weight load instead of after prompt encoding.
- **Source latent cache staleness**: image/video conditioning cache keys now include file
  mtime and size, so overwriting a source file inside one Python session re-encodes instead of
  silently reusing stale latents.

### Changed

- **Dependency hygiene**: `accelerate` moved from runtime dependencies to the `dev` extra; it is
  only needed by the upstream Diffusers reference probe in `tools/`, not by the MLX runtime.
  `ftfy` stays in runtime dependencies because Wan prompt cleaning uses it for upstream-parity
  text normalization.

## [0.18.24] - 2026-06-30

### Added

- **Python runtime loading and multi-output reuse**: add public
  `resolve_generation_runtime(...)`, `load_generation_model(...)`, and loaded-runtime
  `generate_output(...)` / `generate_outputs(...)` helpers for the unified `mlxgen generate`
  families, with shared output naming, overwrite-safe collision handling, and published
  reuse-vs-reload validation across Qwen masked edit, FLUX.2 multi-reference, Wan A14B
  image-to-video, and large Z-Image generation.
- **Machine-readable runtime events**: add `--json-events` for `mlxgen generate` and
  `mlxgen upscale`, with structured runtime progress, saved-artifact terminal events, diagnostics
  paths, and remediation objects for actionable failures.
- **Model recommendations guide**: add a memory-tier recommendation page for `18 GB`, `24 GB`,
  `32 GB`, `64 GB`, and `128+ GB` Macs based on published MLX-Gen memory measurements.

### Changed

- **Core documentation refresh**: rewrite the main user-facing docs so they describe the shipped
  CLI and Python capabilities directly, including seed-driven multi-output generation, metadata
  save behavior, Python runtime ownership, and conservative model recommendations by memory tier.

### Fixed

- **Image finalization memory tail**: default image save is now one-pass and metadata-light,
  `--embed-metadata` is explicit opt-in, and the published `4096x4096` save-phase probe measured
  peak sampled RSS `-51.2316%` and peak Darwin physical footprint `-54.1496%` versus the legacy
  three-pass path.
- **Runtime contract correctness**: fix progress terminal semantics, Qwen control/control-inpaint
  no-LoRA completion, Z-Image CFG math, and variant-sensitive capability spoofing for local/custom
  model identities.

## [0.18.23] - 2026-06-28

### Added

- **Runtime memory telemetry**: generated image/video metadata and video failure manifests now
  include MLX allocator, process RSS, Darwin physical-footprint, cache-policy, and timing records
  when available. Set `MFLUX_RUNTIME_MEMORY_TELEMETRY=0` to disable metadata collection.
- **Generation memory benchmarks**: add repeatable local benchmark tooling and profiles for
  prompt-materialization release, callback/output retention, telemetry overhead, Wan low-RAM
  profiles, and SeedVR2 image/video memory behavior.
- **Qwen route matrix**: add a first-class Qwen route matrix page that maps MLX-Gen capability ids
  to the upstream Diffusers Qwen pipelines and to the exact accepted proof surfaces already
  shipped in the docs.

### Changed

- **Low-RAM generation behavior**: `--low-ram` now applies a default MLX cache limit and clears
  cache at more model-family denoise/decode boundaries while preserving the same generation
  surfaces and documented quality profiles.
- **CLI and prepare defaults**: centralize model inference-step defaults, reject non-positive
  step counts before loading weights, and resolve `mlxgen prepare` backends from model
  configuration instead of only from string matching.
- **LoRA exact-row completion**: close the remaining production-support gaps for the current Qwen
  and FLUX.2 image LoRA surface. The exact validated public rows now also include
  `AbstractFramework/qwen-image-8bit` on `qwen.latent`,
  `AbstractFramework/qwen-image-edit-2511-8bit` on `qwen.multi-reference`, `qwen.reframe`, and
  `qwen.outpaint`,
  `AbstractFramework/flux.2-klein-9b-8bit` on `flux2.multi-reference`, and
  `AbstractFramework/flux.2-klein-base-4b-8bit` on `flux2.outpaint`. The route-expansion proof
  bundle now covers the full accepted June 22 exact-row set with contact sheets, command logs, and
  loader metrics.
- **Public contract wording**: remove stale "experimental" wording from the exact validated
  Qwen/FLUX.2 LoRA rows and from the shipped reframe/outpaint route documentation. The docs now
  describe a narrower but honest production-supported surface: exact validated rows are public
  support, `mapped-unvalidated` rows are not.
- **LoRA proof bundle cleanup**: tighten the published June 22 route-expansion bundle after a
  stricter review. The exploratory seed-sweep sheet was removed from `docs/assets/validation`,
  the Qwen 2511 A/B sheets now use prompt-matched Lightning baselines, and every accepted contact
  sheet now shows readable route/model labels plus the exact prompt and key generation parameters.
- **Z-Image latent LoRA proof**: restore the exact `AbstractFramework/z-image-turbo-8bit`
  `z-image.latent` row with a new same-source A/B proof bundle using
  `ostris/z_image_turbo_childrens_drawings`. The published bundle now shows a readable source /
  baseline / with-LoRA sheet, the exact reproduction commands, and the exact scope of the accepted
  q8 latent style-transfer row.

### Fixed

- **SeedVR2 video temporal continuity**: SeedVR2 video restore now rejects multi-chunk profiles
  below `29` source frames and `8` overlap frames on both the CLI and Python API, keeps restored
  output frame count/FPS aligned with the requested source window, and preserves the current
  quality-first `29/8` video profile for both 3B and 7B.
- **Release publishing order**: the GitHub release workflow now publishes to PyPI first and only
  creates or updates the GitHub Release after PyPI succeeds, so failed package publication does
  not leave a misleading GitHub release.
- **Distribution hygiene**: keep `twine` out of default user installs and in the `dev`/`release`
  extras, and make the package build cleanup target match `mlx-gen` artifacts.

## [0.18.22] - 2026-06-22

### Added

- **Qwen base control-inpaint proof**: add the first exact `qwen.control-inpaint` public row on
  `AbstractFramework/qwen-image-8bit`, inject the exact
  `InstantX/Qwen-Image-ControlNet-Inpainting` sidecar through unified `mlxgen generate`, validate
  the route with two same-source same-mask same-seed q8 Lightning rows, and publish a dedicated
  contact sheet, command log, and M5 Max timings.
- **Z-Image Turbo native inpaint proof**: add the first exact `z-image.inpaint` public row on
  `AbstractFramework/z-image-turbo-8bit`, validate it with a same-prompt same-seed engine-thruster
  comparison against the old latent route, and publish a full-sheet plus masked-area crop proof.

### Changed

- **Masked-route docs and help**: document the new base-Qwen control-inpaint and Z-Image Turbo
  native-inpaint routes across the README, API reference, image-edit capability guide, FAQ,
  Qwen localized-edit guide, and top-level `mlxgen generate --help`, so `--mask-path` now
  describes the current shipped route surface instead of the earlier planned-only state.
- **Qwen and Z-Image masked-route runtime**: tighten the shipped localized-edit routes without
  expanding the public API. Base-Qwen control-inpaint now skips inactive negative-prompt work on
  the exact `guidance=1` Lightning proof path and records only the effective negative prompt in
  metadata. Base-Qwen and Z-Image native inpaint now also invalidate cached source/mask conditions
  when those files change in place. The published proof bundles were refreshed with the accepted
  current timings: Qwen control-inpaint engine/repair at `17.29s` / `24.65s` and `17.74s` /
  `23.86s`, and Z-Image Turbo native inpaint at `21.00s` / `26.86s` on the documented M5 Max
  proof rows.
- **Secondary CLI contract alignment**: unified `mlxgen generate` now forwards
  `--controlnet-strength` to the exact base-Qwen `qwen.control-inpaint` route and accepts an
  explicit `--controlnet-model` only when it matches the validated inpainting sidecar. The direct
  non-turbo `mflux-generate-z-image` command no longer advertises `--mask-path`; native
  `z-image.inpaint` remains the Turbo-only public masked route.

## [0.18.21] - 2026-06-21

### Changed

- **SeedVR2 video audio contract**: SeedVR2 restored MP4s now preserve the matching source audio
  segment by default when the source clip has audio. If copied audio cannot be proven safe,
  MLX-Gen fails the run instead of silently publishing a muted output. `mlxgen upscale` now
  exposes `--drop-audio` as the explicit opt-out for intentionally silent restored MP4s. The
  shared post-write path records `audio_copied`, `audio_copy_mode`, and `audio_copy_reason`, and
  the published Air France `25s–35s` proof bundle remains the release evidence for the copied-audio
  path.

## [0.18.20] - 2026-06-21

### Changed

- **SeedVR2 video restoration**: `mlxgen upscale` accepts `--video-path` in addition to
  `--image-path`, preserves source FPS by default, trims padded frames back to the requested clip
  length, records source-video and chunk metadata in `.metadata.json`, documents the current
  explicit audio contract (`audio_copied=false`), and rejects `--vae-tiling` on video input.
  Public docs now center the June 21 five-second Eiffel proof bundle: safe bounded
  `1x 29/8` `3B` and `7B` runs plus explicit enlarged `2x 29/8` `3B` and `7B` comparison runs,
  each with restored MP4s, comparison MP4s, motion strips, contact sheets, readable labels, and
  reproduction commands. Older exploratory SeedVR2 video artifacts were removed from the published
  docs assets so the versioned validation surface only reflects the accepted current proof set.
- **SeedVR2 video host safety**: the public SeedVR2 video CLI path is now conservative by default:
  it enables `--low-ram` automatically, defaults omitted video resolution to `1x`, uses
  `--mlx-cache-limit-gb 8` as part of the MLX cache policy in safe mode, runs video restore
  through sequential temporal chunking with a fresh model per video seed, serializes video work
  through a runtime lock, and fails closed on enlarged video output unless
  `--force-unsafe-video-memory` is passed explicitly. The planner now reserves resident-weight
  headroom up front so the safe preflight chunk budget agrees with the runtime per-chunk budget.
  The CLI now also rejects streamed SeedVR2 chunk profiles below `9` frames because they can
  preserve frame count while still breaking temporal continuity.
- **SeedVR2 7B route hardening**: add a first-class `seedvr2-7b-sharp` source route for the
  official `seedvr2_ema_7b_sharp.pth` checkpoint, apply the SeedVR2 image `--color-correction`
  flag end-to-end, record SeedVR2 checkpoint provenance in output metadata, and fail closed if the
  loaded SeedVR2 transformer or VAE weights do not exactly cover the runtime parameter tree.
- **SeedVR2 temporal VAE and chunk repair**: align the streamed video planner with real model
  windows, restore clip-global streamed noise continuity, implement the official temporal-upsample
  contract, and enable SeedVR2 VAE causal slicing with temporal memory handoff. On the accepted
  Eiffel `2x 29/8` proof clip, the path reduced peak MLX usage to `34.40 GB` for `3B`
  and `44.27 GB` for `7B`, while the accepted `1x 29/8` proof shows correct frame count, FPS, and
  late-tail continuity for both models.
- **Qwen localized-edit docs**: add a dedicated public guide that explains Qwen masked edit,
  Qwen structured control, and the planned control-inpaint slice in plain language, including
  what ControlNet means, what “sidecar” means, when control-inpaint is likely to help, and why
  the extra control model is not a LoRA.
- **Legacy FLUX.2 CLI guidance**: make the legacy `mflux-generate-flux2` and
  `mflux-generate-flux2-edit` entry points identify themselves as compatibility commands, point
  new integrations to `mlxgen generate`, document the FLUX.2 negative-prompt exception more
  clearly, and add a dedicated troubleshooting migration note for packages that still shell out to
  the legacy commands.

## [0.18.19] - 2026-06-15

### Added

- **Qwen masked edit / inpaint proof**: add first-class `--mask-path` support on the Qwen edit
  route, plus a validated `AbstractFramework/qwen-image-edit-2511-8bit` q8 masked-edit proof row
  with a two-condition contact sheet, command log, and M5 Max timings for the regular `20`-step
  path versus the dedicated `lightx2v/Qwen-Image-Edit-2511-Lightning` `4`-step path.
- **Qwen masked edit control sheet**: add a same-prompt same-seed no-mask-versus-mask Lightning
  control sheet, showing that the dedicated mask is what keeps the Qwen 2511 q8 edit localized.
- **Qwen structured control proof**: add the first exact public `--controlnet-image-path` route on
  `AbstractFramework/qwen-image-8bit`, with the exact InstantX union ControlNet sidecar, a
  dedicated `qwen.control` capability row, a same-seed no-control-versus-control contact sheet,
  command log, and M5 Max timings using `lightx2v/Qwen-Image-Lightning` as the recommended `4`-step
  path.

### Changed

- **Qwen image-edit docs**: document masked edit / inpaint in the image-edit guide, the capability
  matrix, the API reference, the FAQ, and the LoRA guide, and surface the dedicated Qwen 2511
  Lightning adapter as the recommended fast public path for masked edits on the validated q8 row.
- **Qwen structured-control docs**: document the exact `qwen.control` route, make the
  `--image` versus `--controlnet-image-path` workflow boundary explicit, and publish the accepted
  base Qwen q8 structured-control proof in the capability guide, API reference, FAQ, and LoRA
  guide.
- **Adapter guidance boundary**: clarify that MLX-Gen owns exact route truth, capability
  reporting, and fail-closed adapter checks, while higher-level model-to-adapter convenience and
  curation belong in higher-level integrations. The LoRA and quantization docs now also make the
  boundary between validated MLX-Gen q8 packages and arbitrary third-party FP8 checkpoint guidance
  explicit.

## [0.18.18] - 2026-06-13

### Changed

- **Wan LightX2V public adapter syntax**: document the supported A14B adapter forms more clearly,
  including the stable `repo:subdir/file.safetensors` syntax, equivalent absolute local file
  paths after download, and the requirement to pass paired A14B adapter files as separate
  `--lora-paths` arguments.

### Fixed

- **Wan mixed-base LoRA compatibility checks**: accept the official `lightx2v/Wan2.2-Lightning`
  I2V and T2V subpath references on the matching A14B routes instead of rejecting them when the
  adapter repository declares several compatible Wan base models.
- **Nested LoRA file resolution**: resolve Hugging Face adapter references with subdirectories such
  as `owner/repo:subdir/file.safetensors` correctly in the local LoRA cache, so the documented
  LightX2V Wan commands work as written.

## [0.18.17] - 2026-06-12

### Added

- **LightX2V Wan Lightning 4-step proof**: add exact q8 A14B text-to-video and first-frame
  image-to-video same-seed A/B contact sheets using the official `lightx2v/Wan2.2-Lightning`
  paired high-noise and low-noise LoRAs at `4` steps, `flow_shift=5.0`, `guidance=1.0`, and
  `guidance_2=1.0`.
- **LightX2V Wan longer-run timing evidence**: add `81`-frame, `20` fps speed comparisons against
  the current practical original A14B profiles, showing `8.27x` faster text-to-video and `6.13x`
  faster first-frame image-to-video runs with the explicit LightX2V 4-step recipe.
- **LightX2V Wan compact proof matrix**: add `41`-frame `M5 Max` progress contact sheets for
  prepared BF16/q8 T2V plus q8 I2V, along with wall-time and max-RSS measurements for the current
  LightX2V `4`-step recipe and the current practical original q8 profiles.
- **LightX2V T2V quality-envelope follow-up**: add an `M5 Max` T2V step sweep (`4`, `6`, `8`
  steps) plus a `832x480` probe, and document that the weaker `240p` T2V look is mainly a fast
  low-resolution tradeoff rather than a LoRA loader or scheduler bug.
- **LightX2V Wan public proof refresh**: add a readable `720p` q8-versus-BF16 keyframe sheet and
  a working-only `240p`-versus-`480p` T2V sweep for the current public docs.

### Changed

- **Wan A14B route-level LoRA proof ids**: the public A14B capability rows now surface the accepted
  LightX2V 4-step validation profiles instead of the earlier effect-specific A/B examples.
- **Wan q8 LightX2V 720p stability**: improve the documented `720p` A14B q8 Lightning profile and
  add a same-seed q8-versus-BF16 comparison sheet for the `1280x720`, `41`-frame, `4`-step route.

## [0.18.16] - 2026-06-11

### Added

- **Wan video LoRA runtime**: add Wan-specific LoRA mapping, explicit Wan CLI support for
  `--lora-paths`, `--lora-scales`, and `--lora-target-roles`, generated-video metadata for applied
  Wan adapters, and focused tests for TI2V-5B single-transformer plus A14B dual-transformer LoRA
  routing.
- **Original Qwen Image Edit q8 LoRA proof**: add an accepted single-image edit A/B contact sheet
  for `AbstractFramework/qwen-image-edit-8bit` using the current Ghibli-style adapter profile.
- **Wan TI2V-5B q8 text-to-video LoRA proof**: add an exact q8 text-to-video A/B contact sheet
  using `AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets`.
- **Wan video LoRA proof assets**: add exact q8 proof contact sheets for TI2V I2V, T2V-A14B, and
  I2V-A14B, plus a combined Wan route matrix built from the final A/B artifacts.

### Changed

- **LoRA support snapshot**: Wan video LoRA is now documented as exact-route validated on all
  current Wan q8 public rows: TI2V-5B text-to-video, TI2V-5B first-frame image-to-video,
  T2V-A14B text-to-video, and I2V-A14B first-frame image-to-video. Original Qwen Image Edit q8 is
  now also treated as an accepted exact `qwen.edit` proof row.
- **LoRA validation command surface**: `mlxgen validation --profile <lora_validation_profile>` now
  resolves the current exact LoRA proof rows exposed by `mlxgen capabilities`.
- **Backlog status**: close the Wan video LoRA item as completed, keep Wan follow-up work on
  parity/performance/extra package variants separate, and narrow the remaining LoRA backlog to
  unfinished image-family proofs plus Bonsai deferral.

## [0.18.15] - 2026-06-11

### Added

- **LoRA capability reporting**: `mlxgen capabilities` now reports route-level LoRA support,
  validation status, target roles, and validation-profile ids for applications that need to decide
  whether to expose adapter controls.
- **LoRA documentation**: add a dedicated LoRA guide covering explicit adapter download, strict
  scale matching, adapter/base-model compatibility, and source/no-LoRA/with-LoRA validation.
- **Qwen Image Edit 2511 LoRA proof**: add a Qwen 2511 q8 multiple-angle LoRA A/B contact sheet
  and command log using `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`.
- **Qwen 2509, Qwen 2512, Z-Image Turbo, and ERNIE LoRA proofs**: add exact q8 proof rows,
  contact sheets, and public command examples for `AbstractFramework/qwen-image-edit-2509-8bit`,
  `AbstractFramework/qwen-image-2512-8bit`, `AbstractFramework/z-image-turbo-8bit`, and
  `AbstractFramework/ernie-image-turbo-8bit`.
- **ERNIE Image Turbo LoRA support**: add public-route LoRA loading, capability surfacing, strict
  metadata reporting, and an exact q8 anime-style A/B proof for
  `AbstractFramework/ernie-image-turbo-8bit`.
- **Wan practical video examples**: add a dedicated Wan video guide with `101`-frame, 20 fps M5 Max
  comparison clips for TI2V-5B at `832x480` and `1280x704` plus A14B T2V at `480x240`.
- **Wan TI2V-5B parity backlog**: add a planned item for source-model TI2V-5B math and behavior
  comparison against official Wan plus local Diffusers/Transformers references.
- **Wan flow-shift control**: add `--flow-shift` and Python `flow_shift=` support for Wan video
  generation, with metadata recording and tests for explicit overrides, so lower-resolution
  TI2V-5B runs can use the `3.0` schedule shift recommended by Wan references for 480p-class
  profiles while native TI2V-5B keeps its model default.

### Changed

- **Experimental feature wording**: mark LoRA and reframe/outpaint documentation as experimental
  while preserving the existing fail-closed adapter and capability contracts.
- **LoRA support snapshot**: document the exact validated q8 rows, keep base Qwen Image
  experimental, and explicitly deprioritize Bonsai packed-runtime LoRA work.
- **Wan guidance**: document that A14B at `480x240` or `240x480`, `101` frames, 20 fps, and
  `20-25` steps is the preferred practical M5 Max profile for the recorded starship prompt, and
  document `--flow-shift 3` for new 480p-class TI2V-5B checks.
- **Wan video LoRA planning**: refine backlog and docs to distinguish the single-transformer
  TI2V-5B path from dual-transformer A14B routes, and record named public proof candidates for
  future Wan LoRA validation.

### Fixed

- **LoRA fail-closed behavior**: requested LoRA adapters now fail on missing files, unreadable
  files, zero matched keys, zero applied layers, incompatible matrix shapes, and incompatible
  cached model-card base metadata instead of continuing without the requested adapter.
- **Qwen Image Edit 2511 LoRA mapping**: accept the Diffusers `transformer.transformer_blocks.*`
  `lora_A`/`lora_B` adapter key format, including Qwen modulation layers, so compatible Qwen 2511
  LoRAs apply instead of being rejected as zero-match adapters.
- **Qwen 2509 and Qwen 2512 LoRA mappings**: accept the public Diffusers modulation-key variants
  used by current Qwen 2509 edit and Qwen 2512 text adapters, allowing exact matched-key
  application on the validated q8 routes.
- **FLUX.2-dev LoRA routing**: `black-forest-labs/FLUX.2-dev` is not inferred as a supported
  FLUX.2 Klein route, and FLUX.2-dev adapters such as
  `lovis93/Flux-2-Multi-Angles-LoRA-v2` are rejected for FLUX.2 Klein models.
- **Bonsai LoRA fail-closed boundary**: Bonsai capability surfacing and planning now stay explicit
  about LoRA being unsupported on the packed ternary runtime instead of implying that ordinary
  adapter injection should work there.

## [0.18.14] - 2026-06-08

### Added

- **Generative reframe and canvas-guided outpaint**: add `--reframe-padding` and
  `--outpaint-padding` support for validated FLUX.2 Klein 4B/9B and Qwen Image Edit original,
  2509, and 2511 routes, including model capability reporting, router validation, generated
  metadata, source/q8/q4 proof contact sheets, and a `mlxgen validation` profile.

### Changed

- **Outpaint source blending**: canvas-guided outpaint now uses edge-extended conditioning and an
  adaptive source blend. MLX-Gen blends source detail back only when the generated source window
  still matches the original source; otherwise it keeps the generated canvas to avoid ghosted
  fragments.
- **I2I documentation**: clarify the difference between latent I2I, edit-reference I2I,
  multi-reference I2I, generative reframe, canvas-guided outpaint, and future native fill/inpaint
  outpaint.

### Fixed

- **SeedVR2 large-output decode**: keep small upscales on the untiled VAE path while automatically
  using tiled VAE decode for large outputs, preventing invalid full-frame decodes without requiring
  users to opt into tiled VAE encoding.

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
