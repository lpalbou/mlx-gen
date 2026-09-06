# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.36.0] - 2026-09-06

Peak memory on MiniMax-H3, measured rather than projected, plus a guard that reports a request too
large for the machine instead of leaving the OS to kill it. The capabilities payload moves to
`schema_version` 16.

### Added

- **`measured_runs` on the capability row**, replacing the singular `measured_peak`. Each run carries
  its `outcome` (`completed` or `killed`), `footprint_bytes` (the highest process footprint observed,
  which is a peak for a completed run and a lower bound for a killed one), `mlx_peak_bytes`,
  `terminated_at`, and the conditions that make it reproducible: canvas, frames, steps, quantization,
  cache limit, whether the conditioner was released, and the machine's total memory. Runs that were
  killed are published too, because a configuration that did not fit is what a host most needs.
- **`max_validated_frames` beside `max_frames`.** The first says how far the measurements go, the
  second is the decode grid's `17n + 5` bound and holds on any machine. Publishing only the second
  invites projecting a memory budget from a number that was never about memory.
- **`peak_bytes_fixed` and `peak_bytes_per_packed_row`.** Peak memory follows the packed sequence
  length, not the canvas and the frame count separately: a 243-frame `960x544` request and a
  124-frame `1344x768` request differ by 0.5% in packed rows and reached the same MLX peak. These
  publish the measured line through that axis, as bytes and not as a guarantee that a run fits.
- **`packed_sequence_length(width, height, frames)`** in the MiniMax-H3 layout module, so a caller
  can size a request from the layout instead of reverse-engineering it.
- **A request preflight.** Before a run the runtime estimates the peak from the packed rows, refuses
  what cannot fit at all, and warns when a request is close enough to the limit that other resident
  processes decide the outcome. It reports and stops; it never alters the request.

### Fixed

- **Low-RAM mode no longer loses its cache tightening silently.** An explicit `--mlx-cache-limit-gb`
  overrides it, which is deliberate and which a documented restore profile relies on, but the
  override is now announced. The cache limit is the largest single term in this model's footprint:
  the same configuration measured under a raised limit came out 16.8 GB higher, more than three times
  the difference between the two canvases.

## [0.35.0] - 2026-09-06

MiniMax-H3 becomes drivable by an embedding host: progress on the Python path, a corrected duration
contract, and a capability row that describes the soundtrack, the frame grid and the entry defaults.
The capabilities payload moves to `schema_version` 15. Every field added below is additive with a
falsy or null default, so an application can gate on `schema_version >= 15` and older readers are
unaffected.

### Added

- **The generation capability row describes generated audio and duration** (all fields additive with
  falsy defaults). `generates_audio`, `audio_channels`, `audio_sample_rate` and
  `supports_audio_shift` describe a route that composes a soundtrack with the picture, distinct from
  a restoration row's `supports_audio_passthrough`. `min_frames`, `max_frames`, `frame_multiple`,
  `frame_remainder`, `frame_rounding` and `output_fps` publish the duration contract, so a host can
  build a frame control without rediscovering the grid; `supports_fps: false` only ever said the
  caller may not choose a rate, never that the route writes 24 fps.
- **`prompt_sections` publishes a model's structured prompt.** Each section carries its `key`, the
  engine's literal `label`, the `option` and `parameter` that fill it, its `role` and whether it is
  `required`. A host can render the fields, address them, and recognise a label already present in a
  prompt it was given.
- **Per-entry defaults on the row**: `default_steps`, `default_width`, `default_height`,
  `default_frames`, `default_video_shift` and `default_audio_shift`, so a host seeds its controls
  from the catalog instead of copying constants that drift.
- **`--release-text-encoder` on MiniMax-H3.** Drops the Qwen3-VL conditioner once the prompt is
  encoded. `generate_video` already accepted it; the CLI now exposes it, and the row advertises it as
  `supports_text_encoder_release`.
- **`--progress` and `--replace` on MiniMax-H3**, matching every other generate CLI. Only the
  negative forms existed, so the documented convention failed on this one model.
- **Rows publish a precision and memory contract**: `weight_precision`,
  `unquantized_weights_bytes`, `recommended_quantize`, `validated_quantization_bits`, and a
  `measured_peak` carrying a measured peak process footprint together with the canvas, frame count
  and quantization it was measured at. Sizes are bytes, because the difference between 134.2 GB and
  125 GiB is the difference between a run that starts and one that does not. MiniMax-H3 fills all of
  them; other families publish nulls until their figures are measured.
- **`supports_flow_shift` and `default_flow_shift` on every generation row**, named for the mechanism
  rather than for one family's CLI spelling. Wan calls this control `--flow-shift` and MiniMax-H3
  calls it `--video-shift`, but the sigma transform is the same and both write the same `flow_shift`
  metadata key, so both families publish `true` with their own default. `--audio-shift` stays a
  separate field, since a route that schedules audio has two.
- **`flow_shift_option` and `flow_shift_parameter` on every generation row.** The control is one
  concept with two spellings, so the row now carries the CLI option and the Python keyword this route
  uses (`--flow-shift` / `flow_shift` on Wan, `--video-shift` / `video_shift` on MiniMax-H3). Both are
  `null` exactly when `supports_flow_shift` is `false`.
- **`universal_options` on the capabilities payload.** The options every generate route in this build
  accepts, `--low-ram` today. An application reads the build rather than gating on `schema_version`.
- **`--low-ram` on MiniMax-H3**, which was the one generate route rejecting a shared-parser option.
  It tightens the MLX cache and releases the conditioner after encoding, so one host-level toggle
  now works across every generate route.
- **`supports_guidance` on every generation row.** Whether a guidance scale steers the route, which
  is independent of `supports_negative_prompt`: distilled FLUX.2 Klein has guidance but no negative
  prompt, and Z-Image Turbo has a negative prompt but no guidance.
- **MiniMax-H3 refuses a load that cannot fit.** Unquantized the weights are 125 GiB resident, 97% of
  a 128 GiB Mac, and the load used to end in the OS killing the process with nothing said. The
  runtime now compares the resident size it will need against the machine's own memory and stops with
  an error naming `--quantize 8`. It never quantizes on the caller's behalf.

### Fixed

- **MiniMax-H3 reported no progress through the Python API.** The variant emitted only to the
  `progress_callback` argument and never to the model's callback registry, which is what
  `load_generation_model` subscribes to, so an embedding host saw nothing for the length of a run.
- **Progress phase names now match the rest of the catalog.** MiniMax-H3 emitted `denoising` where
  every other runtime emits `denoise`, so a host matching the house name dropped every step event. It
  also emitted `complete` from the model before the file was written and again after saving; the
  model now emits `generated`, leaving `complete` as the single terminal event that carries the
  output path.
- **The documented maximum frame count could never be generated.** The duration ceiling applies to
  the count after it is snapped up to the `17n + 5` grid, so 362 frames is 15.083 seconds and always
  failed while the error message and docs named it as the bound. The real maximum is 345, the error
  now lists every accepted count, and an off-grid request warns that it was rounded up and records
  `requested_frames` in the metadata beside the resolved `frames`.
- **A structured prompt plus `--soundscape` or `--music` sent the model two of the same section.**
  A prompt carrying a section label now refuses the option that would duplicate it, and section
  labels are recognised only where they open a line rather than anywhere in the text.
- **The two Turbo entries published the same label.** `minimax-h3-turbo` and `minimax-h3-turbo-544p`
  both read `MiniMax-H3 Turbo`, hiding the choice between an 11-minute and a 34-minute clip; each
  entry now names its canvas.
- **Unknown Python-runtime keywords fail with an actionable error.** `generate_outputs` splatted
  keywords straight into the model, so a keyword from another family surfaced as a bare `TypeError`
  from inside the executor; it now names the parameter, the route and what the route accepts.
- **Video health validation describes the file that ships.** It ran before the audio mux rewrote the
  container, so the recorded result described a file that no longer existed in that form.
- **The fallback audio sidecar no longer overwrites an existing file.** A failed mux writes
  `clip.wav` beside `clip.mp4` through the same collision handling as every other artifact.

## [0.34.0] - 2026-09-05

MiniMax-H3: text-to-video and first-frame image-to-video with a synchronized stereo soundtrack,
natively on MLX, with the lightx2v Turbo adapters and a prepared-package workflow.

### Added

- **MiniMax-H3 (`minimax-h3`, `minimax-h3-turbo`, `minimax-h3-turbo-544p`).** One prompt produces a
  24 fps clip of 5 to 15 seconds and its stereo soundtrack, saved together in one MP4 with an AAC
  track. The Turbo entries attach the lightx2v 8-step adapters automatically (768p adapter with
  shifts 6/3; 544p mixed-aspect adapter with shifts 12/3) and `mlxgen download` fetches them with
  the snapshot. `--prompt`, `--soundscape`, and `--music` fill the model's three structured prompt
  sections, and complete structured prompts pass through verbatim. The soundtrack covers ambient
  sound, a score, and spoken dialogue: write a line with the model's `(S1)` speaker ids and
  `<d>[Language] ...</d>` tags and the model generates the voice and animates the speaker's mouth.
  Routed through `mlxgen generate`, `mlxgen capabilities`, `mlxgen prepare`, and the Python runtime
  (`MiniMaxH3.generate_video`). Every component (packed layout, rectified-flow schedulers,
  transformer, Qwen3-VL conditioner, video and audio VAEs) is a direct port of the diffusers 0.40
  reference and matches it at fp32 rounding noise on real weights. See `docs/minimax-h3.md`.
- **MiniMax-H3 image-to-video.** `--image-path` starts the clip from a keyframe: the canvas follows
  the keyframe's aspect ratio at the entry's short edge, the image is stretched onto it and
  conditions both the packed latent rows and the text sequence through a native port of the
  Qwen3-VL vision tower (image processor, DeepStack injection, 3-axis rope index), verified against
  transformers on real weights. Advertised as the `minimax-h3.first-frame` capability row.
- **Prepared MiniMax-H3 packages.** `mlxgen prepare --model minimax-h3 --quantize 8 --path ...`
  writes the 75 GB mixed q8/BF16 package once; load it with `--model <path>` and pick a Turbo
  schedule with `--base-model minimax-h3-turbo-544p` or `minimax-h3-turbo`. A same-seed clip from
  the package is byte-identical to the load-time `--quantize 8` clip.
- **Generated audio on `GeneratedVideo`.** `video.audio` carries a `GeneratedAudio`
  (`waveform`, `sample_rate`); `save()` muxes it as AAC (ffmpeg, PyAV fallback, sidecar WAV when
  muxing fails) and records the `audio_*` metadata fields.
- **Shard-streaming quantized loading.** `--quantize 8` on MiniMax-H3 quantizes each shard as it
  loads, so the two 30B-class components never hold a full BF16 copy next to the q8 copy; the
  measured `960x544` run peaks at 80 GB of MLX memory on a 128 GB Mac. Prepared MiniMax-H3 packages
  stream the same way, so loading one costs the stored 75 GB rather than a second in-memory copy.
  The streaming loader applies the same once-per-process MLX cache-limit default as the other model
  loaders, and `mlxgen generate` on MiniMax-H3 accepts `--mlx-cache-limit-gb`.

### Notes

- MiniMax-H3 reference-to-video and the closing keyframe are not available yet. The 768p canvas
  costs about 4.5 minutes per step on an Apple M5 Max; the 544p adapter is the iteration setting.
- MiniMax-H3 is released under the MiniMax H3 Community License, which restricts use in some
  territories; read it on the model card before downloading or distributing weights or outputs.

## [0.33.1] - 2026-09-02

Video restore routes work again on FFmpeg 9.

### Fixed

- **Video restore routes on FFmpeg 9.** SeedVR2 and SwiftVR decode source frame windows through
  ffmpeg with `-vsync 0`, and FFmpeg 9.0 removed that option (deprecated since 5.1). Every restore
  run on a current Homebrew ffmpeg (9.0.1) failed at decode time with
  `Unrecognized option 'vsync'`. The decoder now passes the equivalent `-fps_mode passthrough`,
  accepted by FFmpeg 5.1 and newer, so FFmpeg 5.1 is the oldest supported release. Reported in
  #12, fixed in #14.

  The video encode color pipeline was re-verified on FFmpeg 9.0.1: the pinned BT.601 conversion
  still produces YUV bytes identical to ffmpeg's default conversion at 320x240 and 1280x720, and
  the `smpte170m` / `bt709` VUI tags are intact.

## [0.33.0] - 2026-09-02

Outpaint on both axes without duplicating the subject, the original crop restored on every route,
and negative prompts on FLUX.2 Klein base weights.

### Added

- **`--outpaint-passes {auto,1,2}`** on every outpaint route. A request that pads a vertical side
  and a horizontal side deeply opens a free corner sharing neither a row nor a column with the
  source, and the model paints a second copy of the subject into it. `auto` (default) runs such a
  request as two single-axis passes - the deeper axis first on the source, then the other axis on
  that output - when the shallower axis depth exceeds the route's published
  `outpaint_auto_split_corner_ratio` (0.3); `2` splits any request that pads both axes and rejects
  one it cannot split; `1` forces a single canvas and prints a warning naming the corner. The final
  canvas is exactly the one-pass canvas, the source keeps its position to within one dimension
  multiple, and each pass restores its source before the next starts. Measured on a 432x240 source
  grown 256 px on one axis and 148 px on the other with distilled 9B, three seeds per orientation:
  the split is clean on every seed where one pass duplicated on every seed; every other outpaint
  route is clean on the same geometry. Recorded in metadata as `outpaint_passes`,
  `outpaint_passes_requested`, `outpaint_pass_paddings`, `outpaint_pass_fills`,
  `outpaint_pass_reason`, `outpaint_pass_source_restore_differences` and
  `outpaint_pass_source_restore_applied`; the resolved count replays through
  `--config-from-metadata`.
- **Negative prompts on FLUX.2 Klein base weights.** `--negative-prompt` / `--negative` is accepted
  on the text-to-image, latent image-to-image, edit, masked-edit and outpaint routes of
  `flux2-klein-base-4b`, `flux2-klein-base-9b` and their prepared packages, and runs
  classifier-free guidance against it. It needs `--guidance` above 1.0; omitting `--guidance` with
  a negative prompt selects the base default of 4.0. Distilled Klein weights have no guidance
  branch and keep rejecting it, before weights load, both in `mlxgen generate` and in the backend
  commands. Python callers pass `negative_prompt=` on every FLUX.2 Klein `generate_image`.
- **Capability schema v12** (additive): `supports_negative_prompt` on every generation row, and
  `outpaint_pass_modes`, `outpaint_default_passes` and `outpaint_auto_split_corner_ratio` on
  outpaint-capable rows, so an application can predict a second pass and know whether a route takes
  a negative prompt before starting a job.
- **Python**: `prepare_outpaint(..., passes=...)` and `run_outpaint(..., passes=...)`;
  `OutpaintSession.pass_plan`, `.passes`, `.pass_fill_plans`, `.geometry` (the original source in
  the final canvas) and `.pass_canvases`; `resolve_outpaint_pass_plan(...)` and `OutpaintPassPlan`
  for callers that plan without a session; `OutpaintUtil.source_lock_box(...)` and
  `attach_source_lock_mask(...)`. `generate_outputs(...)` takes `generate_method=` to route a
  multi-pass pipeline through the wrapper's seed loop.

### Changed

- **Every outpaint route restores the original crop.** The FLUX.2 Klein routes hold the source in
  latent space during denoising, as before, and the shared layer then pastes the original crop back
  over the decoded result while the generated source window still matches it, the same strategy the
  Qwen route publishes. The FLUX.2 rows therefore publish
  `outpaint_preservation: "adaptive-content-aware-source-blend"` instead of
  `latent-locked-transition-band-no-postblend`, which is no longer emitted; generated metadata
  records the new value and `outpaint_source_restore_applied: true`. Applications that matched the
  old string should treat both values as "original crop restored".
- **Qwen Image Edit outpaint holds the source in latent space** through its masked-edit input
  while the padded area is denoised (a mask the run writes beside each canvas, black over the
  source minus the same 24 px transition band the FLUX.2 lock uses). A prompt asking for a wider
  view can no longer recompose the crop; on the recorded validation envelope the drift under the
  restore drops from 9.35 to 1.37 with the composition unchanged. `outpaint_preservation` is
  unchanged on Qwen rows.
- The outpaint restore threshold is 24 mean-abs on every route (it was 12 on Qwen): a window held
  by the latent lock measures 1.4 to 15.6 across the recorded runs, and a recomposed window,
  which the lock now rules out, measures above 60.
- The two-deep-axis warning fires only for a run that denoises the free corner in one pass
  (`--outpaint-passes 1`, or a request too small to split), names why, and reports the true corner
  after the dimension round-up.

## [0.32.1] - 2026-09-02

### Fixed

- **Outpaint could return the conditioning canvas instead of generated content.** On
  `flux2.outpaint`, the padded region is released for the model to invent, but the same region was
  also supplied as noise-free reference tokens at its own position on every step. Reconstructing
  that reference is a valid solution to the denoising problem, so some runs returned the padding
  essentially unchanged: a directional smear with `edge` fill, or a flat block with `neutral`.
  Whether a given run was affected depended on the source content and the noise draw rather than on
  the request, which is why the same padding depth could succeed on one image and fail on another.
  The route now conditions only on canvas cells that contain real source pixels, so the whole
  source, the transition band and every seam cell still guide the run at their true positions while
  the invented region is genuinely free.

  Measured on a 640x448 source extended 115px (Apple M5 Max, 40-core GPU, 128 GB), using padding
  detail relative to source detail — at or above 1.0 means generated content, well below means the
  canvas came back: the affected case moves from **0.57 to 1.54**, and an unaffected case is
  unchanged at **1.56 to 1.57**.

## [0.32.0] - 2026-09-01

Outpaint on every edit-reference route, and a Python API for it.

### Added

- **Strict outpaint on distilled FLUX.2 Klein 4B and 9B.** Those models now publish
  `flux2.outpaint` alongside `flux2.reframe` and run canvas expansion through the same
  source-locked denoising as the base models, at guidance 1.0 (the weights are step-distilled and
  do not take CFG; the route resolves that default for you). Measured on the published starship
  source at `5%,80%,5%,60%`: distilled 4B q8 completes in 54.6 s and distilled 9B q8 in 80.6 s on
  an Apple M5 Max, 40-core GPU, 128 GB unified memory. Evidence:
  `flux2_klein_outpaint_latent_lock_2026_09_01`.
- **`mflux.outpaint`, a model-agnostic outpaint API.** `run_outpaint(...)` performs an outpaint
  end to end from a loaded runtime; `prepare_outpaint(...)` returns an inspectable session
  (resolved fill mode, canvas geometry, preservation strategy) for applications that own their own
  seed loop; `prepare_reframe(...)` covers the reframe canvas. The conditioning-canvas keywords,
  the fill policy, the preservation strategy and the recorded metadata all follow the selected
  route, so switching model families is a change of model name. Exported lazily from `mflux` and
  `mlxgen`, and documented in [Python Integration](docs/python-integration.md).
- **`post_process` on `generate_outputs`/`generate_output`** runs a callable on each artifact after
  generation and before save, so a workflow with a post-decode step reuses the runtime's seed loop,
  progress events and save semantics instead of reimplementing them.
- **`outpaint_preservation` capability field** states how a route keeps the source region
  (`latent-locked-transition-band-no-postblend` or `adaptive-content-aware-source-blend`), so an
  application can tell what to promise about the original crop before starting a job. Capability
  `schema_version` is now `11`.
- **An outpaint model matrix** in [Reframe And Outpaint](docs/reframe-outpaint.md): every supported
  route on one source and one padding value, with the full prompt, per-route timings and
  source-drift measurements, plus a reproducible command log.

### Changed

- **One implementation of the outpaint conditioning canvas.** The fill policy, the fail-closed
  guard, the resolved-canvas notice and the metadata writer are shared by every backend and by the
  Python API. Qwen outpaint runs now print the resolved canvas and record the same
  `outpaint_fill*` metadata as the FLUX.2 routes; the canvas those runs produce is unchanged.
- **`--outpaint-fill` and `--outpaint-fill-color` are validated against the route** before
  dispatch. Asking a fixed-canvas route such as `qwen.outpaint` for a different fill mode now
  reports which canvas that route uses instead of an argument-parser error.
- **`outpaint_recommended_lora` and the `outpaint_validated_*` fields are published per route.**
  A route advertises an adapter recommendation or a validated envelope only where evidence for
  that route exists.

## [0.31.0] - 2026-08-31

Outpaint conditioning-canvas contract.

### Added

- **`--outpaint-fill {auto,edge,neutral,solid,blur}`** selects the canvas `--outpaint-padding`
  pastes the source onto before denoising, on FLUX.2 Klein base routes. `edge` stretches the source
  border strip outward, `neutral` paints a flat per-side border color sampled from the source,
  `solid` takes a color from the new `--outpaint-fill-color` (`R,G,B` or `#rrggbb`), and `blur`
  uses a blurred scaled copy. Both options round-trip through `-C` generation metadata.
- **`auto` (the default) selects the fill from the padding depth and the loaded adapter.** It keeps
  `edge` while every padded side stays within the depth the border strip covers, switches to
  `neutral` past that so the model generates new subject matter instead of continuing a stretched
  strip, and switches to `solid` green when a green-border outpaint adapter is loaded. Every
  outpaint run prints the resolved fill mode, the canvas size, and the reason.
- **Outpaint capability fields.** `supports_outpaint_fill`, `outpaint_fill_modes`,
  `outpaint_default_fill_mode`, `outpaint_auto_edge_fill_max_stretch`, `outpaint_recommended_lora`,
  `outpaint_validated_padding`, `outpaint_validated_fill_mode`, and
  `outpaint_validated_max_canvas_pixels` are published on outpaint-capable capability records, so
  an application can read the conditioning-canvas contract and the validated envelope from
  `mlxgen capabilities` JSON. Capability `schema_version` is now `10`.
- **Generation metadata** records `outpaint_fill`, `outpaint_fill_color`, `outpaint_fill_requested`,
  `outpaint_fill_reason`, `outpaint_edge_fill_reach_px`, and `outpaint_edge_fill_overreach`.

### Changed

- **Edge fill bounds how far it stretches the sampled border strip.** Padding deeper than the strip
  covers grows the strip instead of the stretch, and cross-fades toward the neutral background. All
  published reframe and outpaint validation profiles are inside that bound and their conditioning
  canvases are unchanged.
- **The outpaint transition band is derived per side.** A side with no padding keeps its source
  pixels locked instead of surrendering a band of them to regeneration.

### Notes

- Outpaint composites the source through a VAE encode/decode round trip rather than pasting the
  original pixels back, so the source region is reproduced, not preserved bit-for-bit. Use
  `--outpaint-fill` and padding depth to control the generated area; use masked editing
  (`--mask-path`, see [Masked Editing](docs/masked-editing.md)) when a region must stay untouched.
- Outpaint cost scales with the expanded canvas, and attention cost grows faster than canvas area.
  Extending in two moderate passes is cheaper and generally better than one large pass.
## [0.30.1] - 2026-08-19

Adversarial post-release audit of 0.30.0: two capability-contract corrections, dispatcher
precedence, and diagnostics fixes. No new features.

### Fixed

- **The SwiftVR capability row no longer declares color-correction modes the route refuses.**
  `color_correction_modes` on `swiftvr.restore-video` read `["wavelet", "lab", "off"]` while the
  route accepts only `"off"`; a host offering the declared flag saw the refusal only after the 5B
  weight load, as an uncaught error. The row now reads `["off"]`, and the CLI refuses
  `--color-correction wavelet|lab` at parse time with the route's own wording, before any weights
  are read.
- **An explicit SeedVR2 alias wins over directory contents again.** `--model seedvr2-3b --path
  <directory holding a SwiftVR checkpoint>` classified as SwiftVR, silently running a different
  model family than the one named; at 0.29.0 the alias ignored `--path` entirely. Positive SeedVR2
  recognition now precedes the by-content directory probe. By-content detection is unchanged when
  no positive handle is given.
- **`seedvr2.restore-image` no longer declares `dimension_multiple: 16`.** The image route accepts
  any geometry, pads internally, and crops back to the exact requested size, so the row now reads
  `1`; the video row keeps `16`, which its center-crop actually enforces.
- **Short-stream diagnostics are exact.** The pyav fallback reported one frame fewer than the
  stream delivered and could crash instead of raising the actionable error when nothing decoded;
  the suggested `--max-frames` cap is now clip-relative, so it is correct under `--start-seconds`
  on both decode backends.
- **Documentation showed a `capabilities` invocation that has never worked.** `mlxgen capabilities
  --family seedvr2` requires `--model` (`--family` is a detection override for local paths, as at
  0.29.0); the examples now use `--model seedvr2-3b`.

### Notes

- The restoration capability record gains `color_correction_modes` per route;
  `RESTORE_CAPABILITIES_SCHEMA_VERSION` moves 1 -> 2. The `mlxgen capabilities` payload keeps
  `schema_version` 9: only declared values were corrected, no payload field changed shape.
- `SEEDVR2_HANDLES` is now defined beside `SWIFTVR_HANDLES` in the dispatch module and re-exported
  from `restore_capabilities` unchanged.
- New regression tests pin all of the above: declared color modes against the route guard, alias
  precedence over directory contents, short-stream counts on both backends, and `{value}`
  interpolation in refusal messages.

## [0.30.0] - 2026-08-18

SwiftVR one-step restoration, and a capability contract for both restoration families.

### Added

- **SwiftVR one-step video restoration** through `mlxgen upscale --model swiftvr`. SwiftVR is a
  one-step restorer on a Wan2.2-TI2V-5B backbone (`H-oliday/SwiftVR`, Apache-2.0), so the existing
  Wan transformer serves it through three additive seams and only the mask-free shifted-window
  attention, the restoration-aware autoencoder, and the causal chunk protocol are new. Measured on
  an M4 Max over 121 frames at 384x288 at `1x`: 11.5 s at 11.1 GB peak MLX, against 471.0 s at
  57.9 GB for SeedVR2-3B and 458.8 s at 67.8 GB for SeedVR2-7B. The route restores at the source
  resolution only, runs bf16 only, and trims clip length to `4a + 1` frames while reporting the
  trim. Ported against the reference implementation with numerical parity at cosine 1.00000000 in
  fp32 for the transformer block, the windowed attention, and the autoencoder, including across
  chunk boundaries.
- **A restoration capability contract.** `mlxgen capabilities` now emits a `restoration` array
  beside the existing `capabilities` array, and `--family` accepts `seedvr2` and `swiftvr`. Each
  route row reports canonical identity, `accepted_media`, `max_images`/`max_videos`,
  `supports_scaling` and `scale_factors`, `supports_quantization` and `weight_precision`, the
  `frame_multiple`/`frame_remainder` clip contract, and the temporal chunking contract. Asking
  whether a restoration model accepts stills is now a field lookup (`max_images`) rather than a
  handle-string match. Capabilities `schema_version` moves 8 -> 9.
- **`SeedVR2.restore_image_to_path(...)`**, the write-to-disk image route that pairs with
  `restore_video_to_path(...)`. It is a thin adapter over `generate_image(...)` and adds no
  restoration behaviour. Both families now name their write-to-disk routes the same way, so a
  caller selects a route by input kind rather than by model family.
- **A five-second SwiftVR/SeedVR2 comparison bundle** under
  `docs/assets/validation/swiftvr-vs-seedvr2-2026-08-17/`: restored clips for all three candidates,
  a four-panel comparison video, a contiguous motion strip, two magnified detail crops, metrics, and
  a validation report.

### Fixed

- **Video runs no longer fail on sources whose container over-reports its frame count.** Some MP4
  containers keep every sample while an edit list shortens playback, so `nb_frames` counts more
  frames than the stream will decode and a restore planned from that number died mid-run with
  `Could not decode all requested video windows`. The frame count is now cross-checked against
  `duration * fps` and, when the two disagree beyond rounding, resolved with an exact decode count
  cached per file. Well-formed sources are unaffected and pay nothing. A stream that is short while
  its metadata is self-consistent still fails, but now names the mismatch and suggests
  `--max-frames`.
- **SeedVR2 video restore works at 480x360-class geometries that previously could not run.** Three
  code paths computed the restored geometry independently and disagreed for scale-factor
  resolutions: for a 480x360 source at `1x` the CLI preflight reported 468x352 and the runtime noise
  provider used 464x352 while preprocessing produced 480x352, so the streamed noise provider
  allocated latents 58 wide against an encode 60 wide and the run raised
  `SeedVR2 streamed video noise slice shape mismatch`. All three now derive the geometry from the
  preprocessing path, and the preflight reports the geometry the VAE actually receives.

### Notes

- SwiftVR restores video only. A single frame is a legal clip for its chunk protocol and does run,
  but it restores stills measurably worse than SeedVR2 on identical input - acceptable at 1:1, with
  facial features losing structure under magnification - because the causal autoencoder state is
  seeded from a replicated frame rather than real temporal context. Stills route to SeedVR2, with a
  three-subject contact sheet, face detail crop and metrics in the validation bundle.
- SwiftVR has no learned upscaler. Upstream reaches larger outputs by bilinear-resizing the degraded
  input and restoring at that size; MLX-Gen has not matched or measured that step, so `--resolution`
  other than `1x` is refused rather than approximated. Resize the source yourself and restore at
  `1x` for the same order of operations, or use SeedVR2 for a learned upscale.
- On Apple Silicon SwiftVR is an offline batch route, not a real-time one: 1080p measures about
  1.1 FPS on an M4 Max. Upstream's real-time framing describes datacenter GPUs.
- SwiftVR reinterprets texture where SeedVR2 recovers it. On grain-heavy archival material SwiftVR
  produces a clean but visibly synthetic image while SeedVR2-7B keeps the most natural texture and
  scores the highest fidelity to the source. Choose SwiftVR for throughput and SeedVR2-7B for
  fidelity, and read the detail crops rather than a sharpness metric.
- Restoration at 480x352 with SeedVR2 exceeds the host-safe memory budget for its minimum permitted
  29-frame chunk, and forcing it through produces intermittent corrupted frames. 320x240 and
  384x288 are clean. Tracked in backlog item 0115.
- Consumers reading `mlxgen capabilities` should accept `schema_version` 9. The `capabilities` array
  is unchanged for every generation route; the addition is the sibling `restoration` array.

## [0.29.0] - 2026-08-15

Generation previews with tiny autoencoders.

### Added

- **Step-wise generation previews with tiny autoencoders.** `--stepwise-image-output-dir` renders
  previews with the published tiny decoder for the model's latent space: `madebyollin/taef1` for
  the FLUX.1 latent space (flux, Z-Image) and `madebyollin/taef2` for the FLUX.2 latent space
  (FLUX.2 Klein, ERNIE-Image, Bonsai). Measured on M5 Max at `512x512` with FLUX.2 Klein 4B q8,
  a frame decodes in 19 ms versus 203 ms through the full VAE, so previewing every step costs
  about 8% extra wall time instead of about 86% — which makes continuous live preview practical
  for interactive applications. Other families preview through the full VAE.
- **`--preview-decoder`** selects `auto` (default: tiny decoder when one is published for the
  family and downloaded, full VAE otherwise), `tiny` (required, errors when unavailable), or
  `full`. Final outputs are always decoded with the full VAE, so a run with previews enabled
  produces a byte-identical image to the same run without them.
- **`ImageUtil.to_pil_image(decoded_latents)`** converts decoded latents to a `PIL.Image` without
  generation metadata, for applications rendering live preview frames. `docs/python-integration.md`
  documents the `PreviewDecoder` callback path, and `docs/previews.md` covers supported families,
  reproducible tiny-versus-full commands, measured fidelity, and decode costs.

### Notes

- Tiny decoders are mapped explicitly per latent space rather than by latent-channel count.
  Families that share a VAE share a decoder; families without a published decoder are unaffected
  and continue to preview through their own VAE.
- Preview decoding is approximate by design: fidelity against the full VAE decode of the same
  latent measures 34.8 dB PSNR (SSIM 0.975) at the final step on Z-Image Turbo, with differences
  concentrated in fine texture. Previews are not suitable for final-quality decisions.

## [0.28.0] - 2026-08-14

Wan default shot length, SeedVR2 image-restoration geometry fidelity, and documentation-site
release.

### Fixed

- **SeedVR2 image restoration returns the exact requested geometry.** Previously the shortest
  edge was silently floored to a multiple of 16 before scaling, so `--resolution 1x` resampled
  the whole image (for example `1451x1600` came back as `1440x1586`, displacing every feature
  toward the top-left), and internal padding filled with black, bleeding a dark edge into the
  restored content near the padded border. `1x` is now a pixel-exact identity on geometry
  (output size equals input size, zero shift at every corner), integer shorter-edge targets
  keep the exact aspect ratio, and internal padding is reflective and cropped away after
  decoding. Video restoration keeps the official center-crop-to-16 behavior.

### Added

- **Automatic step selection for SeedVR2 image restoration, with a `--steps` override.** On
  flat or dark content such as astrophotography or night scenes, the official one-step estimate
  retains a measurable residue of the sampling noise that decodes as a regular 8-pixel mesh
  texture and washes out the faintest structure, while on detail-rich content extra steps
  instead synthesize texture that is not in the source. The default is now adaptive: MLX-Gen
  runs the official single step, measures the mesh signature of the decoded output against the
  source, and re-runs at 4 steps only when the artifact is present (validated across
  astrophoto, portrait, super-resolution, and landscape content: only the astrophoto triggers
  refinement). The decision is recorded in metadata (`steps_mode`, `one_step_residue_pct`), and
  `--steps 1`-`4` forces a fixed count. See "Steps for Flat or Dark Content" in
  `docs/upscaling.md`.
- **Documentation site on GitHub Pages**: every published release now builds and deploys the
  documentation with MkDocs (`.github/workflows/docs.yml`), staging the doc set with validation
  media linked on GitHub instead of shipped in the site.

### Changed

- **Wan generation defaults to 81 frames across the family** (previously TI2V-5B and the shared
  Python-API default used 121). The A14B models train at 81 frames, and longer single shots
  drift toward a ping-pong ending that returns to the first frame; 81 keeps default shots inside
  every model's stable range. TI2V-5B's native 121-frame shots stay available by passing
  `--frames 121` (or `num_frames=121`) explicitly. `docs/wan-video.md` and the FAQ document the
  recommendation and the continuation routes for longer clips.

## [0.27.0] - 2026-08-13

Bernini-R 1.3B BF16 package and Qwen image-edit geometry-stability release.

### Added

- **Bernini-R 1.3B BF16 package (`bernini-r-1.3b-bf16`)**: new catalog entry for
  `AbstractFramework/bernini-r-1.3B-diffusers-bf16`, a runtime-dtype repack of the pinned
  ByteDance source (BF16 text encoder and transformer with their FP32 keep-sets preserved,
  FP32 VAE). The download is ~15.6 GiB instead of ~27 GiB, revision-pinned with byte-count
  preflights, and produces bit-identical output to the FP32 source package (verified on image
  and video use cases at identical settings and seed). Aliases: `bernini-r-1.3b-bf16`,
  `bernini-bf16`.

### Fixed

- **Image-edit geometry drift across iterative edit chains (Qwen edit and FLUX.2 Klein edit)**:
  the edited source image is now conditioned at the resolved generation canvas in both
  families, so the reference and target position grids always match and no source pixels are
  lost per pass. Qwen edit previously conditioned at the area-normalized ~1MP size regardless
  of the requested output size; any mismatch pulled every edit toward a zoom-crop of its own
  input (measured up to 18% horizontal warp in a single 614x512 edit). FLUX.2 Klein edit
  previously conditioned at source-derived floor-16 dimensions with a center crop, losing a
  sliver of the source every pass (+0.5% horizontal drift per iteration in edit chains).
  Iterative edits on both families are now registration-stable (scale 1.000, 0px translation
  across three-edit chains). Automatic-canvas single edits are unchanged (Qwen bit-identical),
  inpaint paths keep their existing contracts, and additional reference images keep per-image
  sizing. The Qwen change is a deliberate geometry-stability deviation from the upstream
  pipeline, which keeps the mismatch at explicit sizes.

### Changed

- `docs/bernini.md` recommends the BF16 package for download and documents both packages'
  sizes and runtime precision; `README.md` and `llms-full.txt` updated to match.
- `docs/image-edit-modes.md` documents the iterative-edit geometry-stability contract.

## [0.26.0] - 2026-08-13

Bernini-R 1.3B renderer and Wan conditioning release. Ships the experimental
Bernini reference-to-video / video-edit renderer (0105), Wan A14B multi-frame
context head conditioning (0102), and Wan A14B SVI 2.0 Pro chain conditioning
(0103). Capabilities `schema_version` advances 5 -> 7 for the new Wan fields.

### Added

- **Bernini-R 1.3B reference video renderer (0105)**: the unified `mlxgen generate` command and
  Python runtime support reference-to-video with 1-8 ordinary `--reference-image` inputs,
  prompt-guided video-to-video, and reference-guided video editing with `--video` plus
  references. The dedicated Wan-family runtime matches Bernini's packed source-aware RoPE,
  target-only extraction, source ordering, independent VAE conditioning, official UniPC schedule,
  task-specific chained APG/CFG semantics, and the official upstream task taxonomy (`t2i`, `i2i`,
  `t2v`, `r2v`, `rv2v`, `v2v`, `mv2v`, `ads2v`) used by the public-case parity harness. Setup
  uses a bounded, revision-pinned factored download from the Wan2.1 base components and official
  1.3B renderer; `--all-files`, `prepare`, runtime q4/q8, and LoRA are rejected because only the
  BF16 source route is numerically credible. The route ships as **EXPERIMENTAL** and is not yet
  promoted: the historical 2026-08-04 schema-v3 bundle still fails at its bounded 17-frame profile,
  but seven official public 1.3B rows plus `ads2v` at mid profile are qualitatively accepted with
  committed proof (full prompts and contact sheets in
  `docs/assets/validation/bernini-r-1.3b-2026-08-11/`). Oracle-dispositioned rows `r2v_case2` and
  `v2v_case3` fail at their official recipes in both implementations; documented tuned recovery
  recipes are bundled separately. See [Bernini-R 1.3B](docs/bernini.md) for commands, capacity
  limits, the official example catalog, and the parity matrix.

- **Wan A14B i2v SVI 2.0 Pro conditioning (0103)**: `--svi-anchor-image`,
  `--svi-motion-latent`, `--svi-motion-latent-count`, `--svi-lora-high`,
  `--svi-lora-low` (Python: `generate_video(svi_anchor_image_path=...,
  svi_motion_latent_path=..., svi_motion_latent_count=...,
  svi_motion_latent_export_path=...)` plus the `svi_lora_high_path`/
  `svi_lora_low_path` constructor pair) bring Stable Video Infinity 2.0 Pro
  (ICLR'26 Oral, vita-epfl `svi_wan22` branch) chain conditioning to the Wan
  2.2 A14B image-to-video route: every clip conditions on
  `[anchor_latent, motion_latent?, zero-latents]` — one PERSISTENT identity
  anchor re-injected into every clip, plus the previous clip's final denoised
  latent (handed over losslessly through an exported
  `*.svi_latent.safetensors` sidecar, never a pixel round-trip), padded with
  true zero latents (NOT the stock VAE-encoded zero frames — the two
  conventions are mutually unintelligible). The SVI error-recycling LoRA pair
  loads at scale 1.0 under a strict key-match contract
  (`unmatched_key_count == 0` per file, verified 800/800 on the official
  `vita-video-gen/svi-model` v2.0 Pro pack) and is re-fused on per-item
  high-noise expert reloads; PEFT adapter-infix keys
  (`lora_A.default.weight`) now normalize for single-adapter files. SVI mode
  and the pack gate each other loudly in BOTH directions (running either
  without the other produces garbage per the upstream warning), and the mode
  conflicts explicitly with `--image-path`, `--last-image`,
  `--context-frames`, and `--video-path`; TI2V-5B and VACE reject before
  weight load. Continuation clips >65 frames print a trained-length advisory.
  Metadata records the full SVI truth (`svi_anchor_image_path`,
  `svi_motion_latent_path`/`_count`, `svi_motion_latent_export`,
  `svi_assembly_trim_frames` = 1 + 4x count for continuation clips, and the
  per-expert LoRA key-match reports), replays through
  `--config-from-metadata`, and is advertised as the additive `supports_svi`
  field on the `wan.first-frame` capability row (capabilities
  `schema_version` 6 -> 7). Use a unique seed per clip (author guidance:
  identical seeds accumulate artifacts).

- **Wan A14B i2v multi-frame context head conditioning (0102)**:
  `--context-frames <png>...` (Python: `generate_video(context_image_paths=[...])`)
  extends the conditioned head of a Wan A14B image-to-video clip with the
  ordered frames that FOLLOW `--image-path` in the motion being continued —
  the SkyReels-V2/SVI-class multi-frame handover that lets a storyboard
  continuation inherit the predecessor's real momentum instead of restarting
  from one frozen frame. The head `[image, *context]` must fill whole 4x VAE
  latent groups (4/8/12 context frames = heads 5/9/13; the misuse of passing
  all K frames to `--context-frames` fails loudly on that count check), needs
  `frames >= head + 4`, composes with `--last-image`, and maps every frame
  through the same canvas and `--resize-mode` as the first frame. Fails
  loudly on every route without the A14B 36-channel i2v conditioning layout:
  text/video-to-video, TI2V-5B (`expand_timesteps`), and Wan VACE — the CLI
  rejects before the multi-minute weight load. `--context-noise <0-1000>`
  (Python: `context_noise=...`) optionally perturbs the conditioned head in
  latent space (SkyReels `addnoise_condition` precedent, ~20), deterministic
  per seed and outside the condition cache. Recorded in metadata
  (`context_image_paths`, `context_noise`), replayed by
  `--config-from-metadata`, recorded in failure manifests, and advertised as
  the additive `supports_context_frames` field on the `wan.first-frame`
  capability row (capabilities `schema_version` 5 -> 6). Ships EXPERIMENTAL
  with a measured zero-shot probe (backlog 0102): on a Lightning 4-step
  continuation pair the K=5 head carried the source clip's motion speed
  across the seam (magnitude ratio 0.90 vs the single-frame baseline's 1.90
  double-speed restart; direction cosine 0.999) at the cost of a mild
  ~2-frame flare/exposure step at the conditioned-to-free boundary (luma
  delta ~3.2/255 against the source clip's own 1.15 max; structurally clean
  in stills). K=1 behavior is bitwise unchanged.

## [0.25.0] - 2026-07-25

Load-path performance and Wan conditioning release. The performance wave
(backlog 0093/0094/0095) makes page-cold weight loads fault in at sequential
SSD speed, caps the previously unbounded MLX buffer cache with a
machine-derived default, and makes the flux2 prompt and compiled-predict
caches real. The Wan wave (backlog 0097/0098/0099) adds EXPERIMENTAL A14B
first+last bracket conditioning (`--last-image`; the capabilities contract
bumps `schema_version` 4 -> 5 with the additive `supports_last_image` field),
an honest prompt-truncation warning with token-count metadata, and explicit
denoising step grids for on-grid Lightning sampling.

Release validation: same-seed TI2V-5B output is bitwise identical to 0.24.0
under identical dependency versions (raw-YUV frame comparison; evidence in
backlog 0101).

### Added

- **Wan A14B i2v first+last bracket conditioning (0097)**: `--last-image <path>`
  (Python: `generate_video(last_image_path=...)`) pins the END of a Wan A14B
  image-to-video clip to a second anchor image, porting the diffusers
  `WanImageToVideoPipeline` `last_image` variant exactly (condition video
  `[first, zeros x (frames-2), last]`, mask endpoints kept at 1). The last image
  maps through the same canvas and `--resize-mode` as the first frame. Recorded
  in metadata (`last_image_path`), replayed by `--config-from-metadata`, and
  advertised as an additive `supports_last_image` capability field on the
  `wan.first-frame` row (capabilities `schema_version` 5). Fails loudly on every route without the A14B
  36-channel i2v conditioning layout: text/video-to-video, TI2V-5B
  (`expand_timesteps`), and Wan VACE. Official first+last training exists for
  Wan 2.1 (FLF2V); on Wan 2.2 A14B this ships as EXPERIMENTAL with a measured
  probe: on the preserved storyboard case (Lightning 4-step, 480x240x33f,
  target = a real future frame of the same shot), the bracketed clip ended at
  final-frame MAE 4.6/255 from the target (NCC 0.995, sharpness preserved at
  711 vs target 713) against the first-frame-only baseline's MAE 56.1
  (NCC 0.335, drift-to-whiteout sharpness 55), with no mid-clip exposure
  jumps (max brightness step 5.9 vs baseline 17.5). One probe pair; see
  backlog item 0097 for bounds before relying on exact end-frame adherence.
- **Wan prompt-truncation warning (0098)**: Wan prompt encoding silently
  right-truncated prompts beyond `--max-sequence-length` (512) UMT5 tokens;
  the trailing text simply never conditioned the video. Encoding now measures
  the real token counts, prints one stderr warning per truncated prompt naming
  the counts ("prompt truncated: 547 -> 512 UMT5 tokens"), and records
  `prompt_tokens` / `prompt_truncated` (plus `negative_prompt_tokens` /
  `negative_prompt_truncated` when CFG actually encodes the negative) in the
  metadata sidecar so hosts can surface headroom.
- **Wan explicit denoising step grids (0099)**: `--denoising-step-list 1000 750
  500 250` (Python: `generate_video(denoising_step_list=[...])`) runs the exact
  timestep grid the lightx2v distill contract specifies
  (`Wan22StepDistillScheduler` `denoising_step_list`) instead of a count-derived
  schedule, on both the unipc and euler solvers. The transformer sees exactly
  the requested timesteps; sigma follows the flow-matching identity
  `t / 1000` (with the count path's same leading `1e-6` guard on the UniPC
  solver). Grid entries are final, already-shifted timesteps, so the flag is
  mutually exclusive with `--steps` and `--flow-shift` (clear errors, no silent
  reconciliation) and rejected for video-to-video (strength truncation would
  drop grid points) and on Wan VACE. Metadata records the grid (`steps` becomes
  the grid length, `flow_shift` is recorded as null) and
  `--config-from-metadata` replays it. This enables on-grid 4/8-step Lightning
  variants and honest Lightning-vs-base comparisons.

- **Sequential weight prefetch at load (0093)**: weight files selected for a load from
  HF-repo layouts are sequentially read into the OS page cache before first use, so
  page-cold weights fault in at sequential SSD speed instead of random-access speed
  (measured mechanism: an 8 GB page-cold component served random-order access at
  ~157 MB/s effective; after a 0.9 s sequential prefetch at ~8.8 GB/s the same access
  pattern ran ~12x faster, bytes identical). Prepared MLX-Gen packages are deliberately
  NOT prefetched: they are written in module-tree order, materialize near-sequentially
  on their own, and the prefetch measurably regressed their cold loads (+1.7-2.1 s
  end-to-end on the Klein 9B q8 package, reproduced twice against page-cold verified
  0.000 residency). Already-resident files are detected via `mincore` and skipped
  (~0.03 s/GB probe cost on warm reloads, zero bytes re-read), and the prefetch is
  skipped when the selected files exceed half of physical RAM (low-RAM protection) or
  when `MFLUX_NO_WEIGHT_PREFETCH=1` is set.
- **Default MLX buffer-cache limit (0094)**: when no cache limit is set, model load now
  applies a machine-derived default — `total RAM / 8`, clamped to `[1 GiB, 8 GiB]`
  (the ceiling matches the 8 GiB cap validated for Wan A14B transient buffers in the
  embedding-host precedent; the floor matches the long-standing low-RAM default) —
  exactly once per process, announced with one stderr line. Previously a bare Python-API
  session ran uncapped (measured 32.9 GB MLX free cache after two Klein 9B q8
  generations, evicting page cache system-wide). `--mlx-cache-limit-gb` and low-RAM
  behavior are unchanged and win over the default; `MFLUX_MLX_CACHE_LIMIT_GB` serves
  Python-API hosts; `-1` (flag or env) is the explicit unlimited opt-out. Hosts that
  cap the cache directly through `mx.set_cache_limit` before loading are detected
  (a pre-existing limit at or below half of physical RAM is treated as deliberate,
  preserved, and announced) — the default never overrides a host-managed limit.
- **FLUX.2 prompt cache and compiled-predict reuse (0095)**: the flux2 family now
  actually reads its `prompt_cache` (it was write-only) — identical prompts on a
  resident instance skip the ~1 s Qwen3 re-encode across txt2img, edit, inpaint, and
  outpaint, with a bounded 8-entry LRU (~100 MB worst case at Klein 9B embeds). The
  per-call `mx.compile` of the denoise predict is now cached per instance and reused
  across same-structure calls, and invalidated on transformer replacement, LoRA
  application, training preview, assistant LoRA scale toggling (training preview
  context exit), LoRA bake at `save_model`, and low-RAM transformer release (compiled
  callables must not keep freed weights alive or replay stale baked constants).

## [0.24.0] - 2026-07-23

Embedded-host performance and generation-geometry release. The first wave
(backlog 0086/0087) removes the dominant per-video fixed costs — the per-prompt
UMT5 text-encoder load and the post-save health re-decode. The second wave
(backlog 0088/0089/0090) trims the import graph, changes Wan decode memory
behavior, and adds the geometry contract (`--resize-mode`, Wan
`--canvas-policy`) plus opt-in step-loop performance options.

Release validation: same-seed TI2V-5B output is bitwise identical to 0.23.1
under identical dependency versions (raw-YUV frame comparison; evidence in
backlog 0091), disk-cache hits reproduce their miss runs exactly, and the
local Wan parity suite passes 6/6. Same-seed outputs can shift when the host's
torch/transformers versions change, because UMT5 text-encoder kernels differ
across those versions; that is upstream behavior, not an mlx-gen change.

### Added

- **Wan UMT5 prompt-embed disk cache (0086)**: identical
  (encoder snapshot, tokenized prompt, sequence length, precision) encodes are now
  served from a small exact safetensors cache under the user cache dir
  (LRU-bounded, 64 entries) instead of reloading the ~11 GB torch text encoder.
  A cache hit does not import torch at all (tokenization moved to numpy tensors).
  Corrupt entries are dropped loudly and re-encoded. Opt out with
  `--no-prompt-cache` (CLI) or `prompt_embed_disk_cache=False` (constructor).
- **Wan resident text encoder (0086)**: `--keep-text-encoder` /
  `keep_text_encoder_resident=True` keeps the UMT5 encoder alive between
  generations in one process, for hosts that chain new-prompt scene generations.
  Default remains load-and-release.
- **`load_generation_model(..., model_kwargs={...})`**: the Python runtime
  wrapper now forwards host-provided constructor extras to the resolved model
  class, so embedding apps can reach model-specific controls (such as the two
  options above) through the public wrapper.
- **`--no-validate-health` (0087)** on Wan video generation and SeedVR2 video
  restore: skips the post-save full-file health re-decode for embedded hosts
  that probe the saved file themselves. Default stays ON. The skip is recorded
  as `health_check: "skipped"` in the video metadata and the `save` runtime
  event.
- **Save-event metadata completeness (0087)**: the Wan `--json-events` `save`
  event now carries `fps`, `width`, `height`, and `total_frames` of the saved
  output, so hosts can build artifact metadata with zero probe decodes.

### Import diet, Wan streamed decode, opt-in Wan compile, video color tags

Second wave of the same audit cycle, covering backlog 0088/0089/0090 plus one
cross-repo color bug the audit measured.

#### Changed

- **Import-graph diet (0088)**: `import mflux` no longer pulls huggingface_hub
  (with httpx+rich, ~0.4-0.6 s), PIL (~0.2 s of eager plugin registration via a
  module-scope `PIL.Image.init()` that is now deleted - PIL self-registers on
  first open/save, test-verified), or numpy. Measured on M5 Max: warm
  `import mflux` 237-274 ms -> 57-60 ms wall (cold-ish shell measurements were
  1.34-1.62 s before); the remaining cost is `mlx.core` (~28 ms). The pure
  output-path logic moved from `ImageUtil` into `cli/output_paths.py` (public
  `ImageUtil.resolve_output_path` delegates, surface unchanged), huggingface_hub
  imports became function-local at all six module-scope sites, and
  `tests/test_import_hygiene.py` now gates dependency creep in CI
  (torch/transformers/tokenizers/matplotlib/httpx/huggingface_hub/cv2/av/rich/PIL
  must stay off the import). Note: the 0088 document's "module-scope mx.compile
  ~1.7 s" claim was re-measured as wrong (~0.2 ms; compilation happens on first
  call) and corrected in place.
- **matplotlib is now the `concept` extra (0088)**: it was a hard install
  dependency used only for concept-attention heatmaps. Heatmap rendering
  without it now fails loudly with `pip install mlx-gen[concept]`; the dev
  extra still includes it. `uv.lock` regenerated.
- **Wan streamed VAE decode is the default (0089 e3)**: `Wan2_2_TI2V` now always
  decodes through the per-slice streaming path behind a frame-batches factory;
  the full-tensor decode branch is gone. Frames are BITWISE identical
  (test-pinned against the non-streamed decoder, which WanVace still uses), and
  the default 121f@1280x704 bf16 run no longer materializes a ~650 MB decoded
  tensor - `python_runtime.generate_outputs` results now retain only latents
  (~3.5 MB) per item until save. `release_denoisers_before_decode` semantics are
  unchanged; per-slice cache flushes follow `--low-ram`. All runs now carry the
  `wan_decode_mode: streamed_vae_slices` / `generation_time_scope: pre-save`
  metadata extras (generation_time is recorded before the decode, which runs at
  save), and a non-finite decode surfaces at frame materialization through the
  per-frame finite checks.
- **Video writer color metadata (cross-repo bug)**: the ffmpeg writer emitted
  untagged yuv420p that ffmpeg had coded with the BT.601 matrix at every
  resolution (probed: pure-red luma 81 at 64p/720p/1080p), so players assuming
  BT.709 for >=720p (AVFoundation/Quick Look) visibly shifted colors on HD
  clips while in-app chains stayed consistent. The writer now pins the
  conversion it already performs (`scale=out_color_matrix=bt601`) and tags the
  stream truthfully (`colorspace=smpte170m`, primaries/transfer `bt709` for the
  sRGB-origin frames). Pixel data verified bitwise identical at 64x64 and
  1280x720; seed stability preserved. Switching HD to a real BT.709 encode is
  recorded as a non-bitwise follow-up in backlog 0089.
- **Wan A14B per-item transformer release + reload (0089 e4)**: the release
  default now lives in the model. `generate_video(release_inactive_denoiser=None)`
  auto-releases the ~14 GB high-noise expert after its per-item denoise phase
  whenever the model is dual-expert AND loaded from a disk-prequantized package,
  and lazily REBUILDS it at the next item's first high-noise timestep
  (per-component weight reload + deterministic LoRA re-fusion, pinned bitwise by
  tiny-weights tests). This closes two gaps at once: the Python API
  (`generate_outputs`) previously set no release at all - embedding hosts kept
  both experts resident even single-seed - and multi-seed CLI batches pinned
  ~28 GB for the whole run. Runtime quantization over bf16 (and bf16-on-disk)
  is gated OUT of the auto default because each reload would re-quantize or
  re-read 14B parameters; explicit intent via `--release-inactive-denoiser` /
  `--no-release-inactive-denoiser` (or the kwarg) always wins. Compiled runs
  (`--compile-transformer`) re-trace the reloaded expert instead of reusing the
  popped callable. Metadata records `released_inactive_denoiser: true` and
  `high_noise_reloads: N` when the behavior actually fired.
  `release_denoisers_before_decode` stays terminal (the low expert is never
  reloaded). Real-checkpoint validation (multi-seed peak-RSS, Lightning-LoRA
  storyboard re-fusion identity) is queued in backlog 0089.
  Correction: SINGLE-seed CLI runs keep the pre-0089 rule and
  release the inactive expert regardless of checkpoint quantization — the
  process exits after one item, so no reload is ever paid and bf16
  checkpoints lose nothing (the first cut had silently dropped that release
  for bf16/runtime-quantized single-seed runs).

#### Added

- **`--compile-transformer` for Wan (0090 d12, opt-in)**: runs each Wan
  denoiser as a compiled MLX graph (`compile_transformer=True` on the Python
  API), with an honest expected gain of ~2-6% per step. Output is NOT
  bit-identical to eager (compiled kernel fusion differs by ~5e-4), so it will
  never become a silent default. Runs that need per-block introspection or
  mid-graph cache flushes (`--low-ram`, `--tensor-health-check-interval`,
  `MFLUX_WAN_BLOCK_HEALTH`) stay eager and print one notice naming the blocker.
  Compiled runs record `compile_transformer: true` in metadata, and the
  compiled callable for the A14B high-noise expert is dropped before
  `release_inactive_denoiser` frees it so weight memory is actually released.
  Compiled-vs-eager parity (both timestep conventions), eligibility notice, and
  CLI wiring are test-pinned.
- **`--resize-mode` (resize | crop | pad), orthogonal to `--canvas-policy`**:
  the canvas policy picks the output canvas; the resize mode picks how source
  pixels map onto it. `resize` stretches to fill (the unchanged default),
  `crop` center-crops without distortion, and the new `pad` mode letterboxes
  the full source onto the canvas (aspect-preserving, black fill by default and
  configurable via `ImageUtil.scale_to_dimensions(fill_color=...)`; the Wan
  VACE reference-image canvas keeps its own upstream-parity white/BILINEAR
  path). Exposed as a `generate_image` kwarg and CLI flag on the latent
  image-to-image families (Qwen incl. native masked edit, FLUX.2 Klein,
  Z-Image incl. native inpaint, ERNIE, FIBO, FLUX.1) and on Wan
  `generate_video` (i2v first frame, v2v source frames, VACE conditioning).
  Masks always map through the SAME source-to-canvas geometry as the pixels
  (shared letterbox math; pad borders binarize to 0 = preserved), so
  inpaint/VACE alignment cannot drift. Routes with reference-pinned
  conditioning geometry (edit/reference, controlnet, outpaint) reject the flag
  loudly instead of ignoring it. Image and video metadata record `resize_mode`
  next to `canvas_policy`; Wan condition caches key on it. Defaults are
  bit-identical everywhere.
- **Wan i2v exact-canvas support (`--canvas-policy` on the wan route)**:
  image-to-video previously ALWAYS replaced the requested canvas with a
  source-ratio canvas. `generate_video(canvas_policy=...)` / wan
  `--canvas-policy` now accept `exact-resize` to honor the requested
  (validated, multiple-snapped) dimensions with the source mapped per
  `--resize-mode`, and `source-aspect` on video-to-video to derive a
  source-ratio canvas from the clip. Default (`None`) keeps each route's
  existing behavior bit-identically. The i2v resolution print names the active
  policy and the escape hatch, both values replay via
  `--config-from-metadata`, and Wan `start` runtime events (plus
  `ProgressEvent`) now carry the RESOLVED `width`/`height` so embedding hosts
  learn final geometry before container probing (0087 save-event pattern).
- **Wan VACE ratio-mismatch warning**: VACE video conditioning used to stretch
  mismatched sources silently; it now prints the same >2% aspect-ratio warning
  as plain video-to-video. Both warnings fire only when the mapping actually
  stretches (`resize`); `crop`/`pad` are aspect-preserving by construction.

#### Changed (performance, output-identical)

- **Wan A14B i2v condition builds in model precision (0089 F2)**: the
  first-frame video condition concatenated `first_frame` + `zero_frames` in
  float32 before casting (a ~1 GB f32 transient at 1280x720x81, ~3 GB at
  1920x1080x121 that the cast immediately halved). The padded VAE-encode input
  is now built directly in the model precision; normalization stays in f32, so
  the encoded tensor is BITWISE identical (test-pinned through a tiny
  random-weight VAE encode).
- **Wan RoPE per-shape cache (0090 F7)**: `WanRotaryPosEmbed` rebuilt the
  rotary embeds on every forward (~33 ms at A14B-121f token counts, ~11 ms
  TI2V, x2 with CFG) although shapes are constant within a run. The embeds now
  cache per (frames, height, width) token grid on the embed instance
  (underscore-prefixed: excluded from parameter/quantization traversal;
  bounded to 2 grids — a grid pair measures ~114 MB f32 at A14B 121f@1280x720
  per expert, a run only ever uses one grid, and a rebuild costs ~33 ms). Pure
  shape-keyed
  compute: cached vs fresh embeds are bitwise identical (test-pinned).

#### Fixed

- **Wan text-to-video now validates `canvas_policy` loudly**: the no-source
  branch skipped policy normalization entirely, so invalid values AND the
  contradictory `source-aspect` request (no source to derive an aspect from)
  were silently ignored. Both now raise; `exact-resize` stays accepted because
  it names what text-to-video already does (regression-tested).
- **Image CLIs replay `canvas_policy`/`resize_mode` from metadata**: the wan
  route replayed both via `--config-from-metadata`, but the image parsers
  replayed neither (canvas_policy had been recorded since 0022 and never
  replayed), so a faithful `-C` replay could silently reproduce different
  geometry. Both fields now replay on parsers that define the flags; explicit
  arguments still win (regression-tested).
- **Capabilities advertise `resize_modes` per route**: `mlxgen capabilities`
  and `resolve_generation_plan` now expose which source-to-canvas mapping
  modes each route's handler actually accepts (additive field, schema
  version unchanged): latent image-to-image rows, Qwen native masked edit,
  Z-Image native inpaint, and Wan i2v/v2v advertise `resize | crop | pad`;
  reference-pinned routes (edit/reference, controlnet incl. the Qwen
  control-inpaint sidecar, outpaint/reframe) and text-only routes advertise
  none. Wan i2v/v2v rows also advertise their `canvas_policies` and per-route
  default (i2v: source-aspect; v2v and VACE: exact-resize, with VACE
  exact-only), which they previously did not.

## [0.23.1] - 2026-07-18

Documentation and verification patch; no behavior changes.

### Changed

- Wan2.2 A14B video-to-video: re-verified the documented proof command end to end on 0.23.0
  (bit-identical output to the archived 0.18.24 clip across all 17 frames - the route is
  seed-stable across releases). New proof card at
  `docs/assets/examples/spaceship-v2v/README.md` with the cross-release verification record;
  the Wan video page now cites the measured runtime (`298.7 s`, `13.8 GiB` peak RSS with
  `--low-ram`) instead of a stale 90-second estimate.

## [0.23.0] - 2026-07-17

Performance release: three fixes from a structured adversarial audit, each independently
re-verified on a clean checkout before release (deterministic memory numbers reproduced
exactly; timing ratios confirmed; quality parity confirmed metric-by-metric and visually).

### Changed

- **Denoise-loop dtype fix**: masked-edit and warm-start latent blends no longer let float32
  scheduler sigmas (and float32 latent masks) promote bf16 latents, so the Z-Image and
  FLUX.2 Klein masked loops now run at bf16 like their txt2img paths. Measured on the
  Z-Image Turbo q8 masked route: MLX peak memory 21.8 GB -> 13.1 GB (-40%) and the published
  native-inpaint recommendation profile (768x432, 9 steps) drops from 18.11 GiB to
  10.57 GiB peak RSS; Klein 4B q8 masked 11.4 GB -> 10.4 GB. Wall-clock is neutral to
  slightly positive (q8 inference is weight-streaming bound). Reproducibility note:
  same-seed outputs on the Z-Image and Klein masked routes change imperceptibly (measured
  mean abs diff 0.19-0.30/255 with outside-mask preservation unchanged); Qwen routes,
  Z-Image latent img2img, and training numerics are bit-identical (training pins its
  historical float32 forward pass explicitly). Dtype contracts are pinned by new tests.
- **Video save speedup**: frame conversion now uses buffer-protocol `np.asarray` instead of
  per-pixel `getdata()` in the encoder and health validator: 33-frame 1280x704 save
  7.75 s -> 0.86 s measured, ~9-12x independently confirmed (the old path also relied on an
  API removed in Pillow 14). Byte-parity with the old conversion is test-pinned across
  image modes.
- **CLI startup speedup**: transformers/torch no longer import at module scope on any of the
  22 console-script paths (annotation-only imports deferred via TYPE_CHECKING; tokenizer
  class lookup deferred to first use): importing the qwen backend CLI drops from ~1.4 s to
  ~0.3 s (~4x, independently confirmed). A new test imports every `[project.scripts]`
  module and asserts the heavy libraries stay unloaded.
- Model recommendations page: the Z-Image Turbo native-inpaint benchmark row reflects the
  new post-fix measurement with the pre-fix value retained for context.

## [0.22.1] - 2026-07-15

Documentation-truth patch; no behavior changes.

### Changed

- Masked-edit docs: a post-release seed sweep (source bf16 seeds 42-45, q4 seeds 42-44)
  resolved the 0.22.0 single-case source-row `--mask-strength 0.95` recolor miss as per-seed
  variance, not a precision difference. The 0.95 guidance now reads uniformly across all
  three `qwen.base-inpaint` proof rows, with a retry-another-seed note for per-seed color
  fidelity (sweep sheet added to the matrix bundle).

## [0.22.0] - 2026-07-15

This release grades the masked-edit routes shipped in 0.20.0/0.21.0 with a standardized
multi-case visual-QA matrix, makes the base-Qwen warm start user-tunable, and withdraws the
one route the matrix measured as failing.

### Added

- **Masked-edit 5x5 validation matrix**: standardized multi-case visual QA (object insertion,
  lens recolor, arm retexture, sticker removal, plus an unscored partial-object-removal
  limitation demonstration; one source, same seed) for the masked-edit routes shipped in
  0.20.0/0.21.0. Results: both FLUX.2 Klein 4B q8 rows `PASS` all scored cases; the
  `qwen.base-inpaint` q4/2512-q8 rows are `PARTIAL` at default settings (warm-start anchoring
  keeps opaque full-region recolors incomplete). New registry profile
  `masked_edit_matrix_5x5_2026_07_15` for `mlxgen validation`, published bundle in
  `docs/assets/validation/masked-edit-matrix-2026-07-15/`, proof-grade updates across
  `docs/masked-editing.md` and `docs/edit-capabilities.md`, and the documented mask-design
  rule that removals need full-object masks. The source `Qwen/Qwen-Image` bf16 checkpoint ran
  the same four scored cases (`PARTIAL` aggregate with the identical recolor signature),
  closing the last wiring-shared gap on the `qwen.base-inpaint` matrix.
- **`--mask-strength` for native base-Qwen masked edit** (`qwen.base-inpaint` only): exposes
  the upstream `QwenImageInpaintPipeline` inpaint strength scoped to masked runs. The default
  `0.85` anchors repainted content to the source (best for retexture/removal); the measured
  `0.95` setting turns the matrix recolor cells from `PARTIAL` into complete recolors on both
  prepared rows (q4 and 2512-q8), at the measured cost of a weaker structural anchor (masks
  crossing thin connected structures detached on both prepared rows at `0.95` - s095
  regression sheet in the matrix bundle). On the source bf16 row the same setting repaints the
  full region but misplaces the color (single-case observation; stated in the bundle). The
  metadata key for the applied warm start is now `mask_strength` (was
  `masked_warm_start_strength` in 0.21.0; `metadata_schema_version` bumped to `2` for the
  rename, and `--config-from-metadata` reads both spellings); `mask_strength` and executed
  `effective_steps` are recorded in metadata; the option is rejected without a mask and on
  the control-inpaint sidecar row.

### Changed

- **Non-turbo Z-Image masked editing withdrawn for the moment**: the matrix measured a
  reproducible geometry artifact (across seeds and with CFG on or off) when masks cross thin
  connected structures on `AbstractFramework/z-image-8bit`, while Z-Image Turbo renders the
  same case cleanly. `z-image.inpaint` is Turbo-only again; non-turbo masked requests fail
  before model load with an actionable error, and the withdrawal evidence stays published in
  the matrix bundle and registry.
- Backlog hygiene: the three 2026-06-30 root-level `bug_*.md` reports moved into the backlog as
  completed items 0083-0085 with resolution records; completed-ledger rows 0077-0080 added and
  lifecycle counts recounted.

## [0.21.0] - 2026-07-15

This release completes the masked-edit expansion started in 0.20.0: every currently supported
image family with a viable upstream or in-repo inpaint mechanism now accepts `--mask-path`.

### Added

- **Native base-Qwen masked edit** (`qwen.base-inpaint`): `--image + --mask-path` now works on
  trusted base Qwen rows without a ControlNet sidecar (source `Qwen/Qwen-Image`,
  `AbstractFramework/qwen-image-4bit`, and the exact proven
  `AbstractFramework/qwen-image-2512-8bit` row), ported from the diffusers
  `QwenImageInpaintPipeline`. The masked region warm-starts from the re-noised source
  (upstream-example strength 0.85) so repainted content anchors to the surrounding structure;
  output metadata records the executed `effective_steps` and `masked_warm_start_strength`
  (metadata key renamed to `mask_strength` in the next release when the strength became the
  tunable `--mask-strength` option).
  The exact validated `AbstractFramework/qwen-image-8bit` row keeps control-inpaint as its
  masked route, unchanged. Visual-smoke proof bundle with outside-mask preservation
  measurements in `docs/assets/validation/masked-edit-2026-07-15/`.
- **Z-Image non-turbo native inpaint**: `z-image.inpaint` now covers trusted non-turbo Z-Image
  rows (`Tongyi-MAI/Z-Image`, `AbstractFramework/z-image-8bit`) and the
  `mflux-generate-z-image` command accepts `--mask-path`. Pass an explicit `--guidance` on
  non-turbo masked runs; the non-turbo default runs guidance-free. (Withdrawn in the next
  release after the validation matrix measured reproducible geometry artifacts on the
  non-turbo row; see the Unreleased section.)
- **`docs/masked-editing.md`**: canonical masked-edit page consolidating the request contract,
  the per-model route matrix with proof grades, and per-family behavior; the API, FAQ,
  image-edit-modes, and Qwen pages now summarize and link there.

### Changed

- Masked-capable routes selected without `--mask-path` (for example `--i2i-mode edit` on a
  base Qwen or Z-Image row) now fail at plan time with an actionable "mask-path is required"
  error instead of failing after model load. Maskless single-image requests on
  `AbstractFramework/qwen-image-8bit` now report "image-strength is required" instead of a
  generic ambiguity error.
- Shell completions for `mflux-generate-qwen`, `mflux-generate-z-image`, and
  `mflux-generate-z-image-turbo` now mirror the real parsers (mask and ControlNet options were
  missing).

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
