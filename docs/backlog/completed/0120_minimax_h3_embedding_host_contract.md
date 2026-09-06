# Completed: MiniMax-H3 embedding-host contract - progress, duration bounds, capability surface

## Metadata

- Created: 2026-09-06
- Status: Completed
- Completed: 2026-09-06

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0002](../../adr/0002_visible_runtime_defaults.md).
- ADR impact: none. The capability schema already governs itself through
  `CAPABILITIES_SCHEMA_VERSION` and its additive-field convention; this pass follows that convention
  rather than changing it.

## Origin

A feature request from BlackPixel, an embedding host that drives the runtime through
`mlxgen generate`, `mlxgen capabilities` and `mflux.load_generation_model`, filed against 0.34.0
(`0eb1409`). It listed four defects and a proposed capability contract. An adversarial pass
(`untracked/h3_verify/agent4/`) verified every claim against the released commit and corrected three
of them before implementation.

## Completion report - 2026-09-06

### Defects fixed

- **No progress through the Python API.** `MiniMaxH3._emit` was a staticmethod that only invoked the
  caller's `progress_callback`; the model's `CallbackRegistry` (set by the initializer, and the only
  thing `load_generation_model` subscribes to) was never fed. An embedding host saw nothing for the
  11 to 62 minutes of a run. `_emit` is now an instance method publishing to both sinks, matching
  `Wan2_2TI2V._emit_progress`.
- **Phase vocabulary.** MiniMax-H3 was the only runtime emitting `denoising`; the house name across
  Wan, VACE and Bernini is `denoise`, so a host matching it dropped every step event. It also emitted
  `complete` from the model before the file existed and again from the CLI after saving; the model
  now emits `generated`, leaving one terminal `complete` that carries the output path.
- **The documented maximum frame count always raised.** The duration ceiling applies to the aligned
  count, so 362 frames is 15.083 s and fails, while the error message and `docs/api.md` named 362 as
  the bound. The real maximum is 345. The request offered "relax the guard to admit 362" as an
  alternative; that was rejected because the reference applies the ceiling to the aligned count for
  exactly this case (`before_denoise.py:398-406`) and MiniMax's own card caps output at 15 s, so
  admitting 362 would have diverged from both. `MIN_NUM_FRAMES`/`MAX_NUM_FRAMES`/`valid_frame_counts`
  now live in `h3_layout.py`, the error lists all 14 accepted counts, and an off-grid request warns
  (as the reference does) and records `requested_frames` beside the resolved `frames` in metadata.
- **`--progress` / `--replace` missing.** MiniMax-H3 declared only the negative forms. The request
  said "11 of 11 other CLIs"; the real count is 18, of which 16 inherit both polarities from the
  shared parser and Wan declares them by hand.
- **Duplicate prompt sections.** `compose_prompt` passed a structured prompt through verbatim and
  still appended `--soundscape`/`--music`, sending the model two of the same section. It now refuses
  the option for a section the prompt already declares, and recognises labels only where they open a
  line (the previous substring match treated a prose mention as a section header).
- **Duplicate Turbo label.** `minimax-h3-turbo` and `minimax-h3-turbo-544p` both published
  `MiniMax-H3 Turbo`, hiding the 11-versus-34-minute choice; each entry now names its canvas.
- **Unvalidated Python-runtime keywords.** `generate_outputs` splatted keywords into the model, so a
  keyword from another family surfaced as a bare `TypeError` from inside the executor after the
  weights were resident. `_reject_unknown_generate_kwargs` now names the parameter, the route and the
  accepted set, and skips callables taking `**kwargs`.
- **Health validation described a file that no longer existed**: it ran before the audio mux rewrote
  the container. It now runs after. The fallback `.wav` sidecar also goes through
  `resolve_output_path`, so a failed mux cannot overwrite an existing file.

### Capability contract (schema 12 -> 13)

All fields additive with falsy/None defaults, so older hosts are unaffected and newer hosts gate on
`schema_version >= 13`. Added to `GenerationCapability`: `generates_audio`, `audio_channels`,
`audio_sample_rate`, `supports_audio_shift`, `supports_video_shift`, `supports_text_encoder_release`,
`prompt_sections`, `min_frames`, `max_frames`, `frame_multiple`, `frame_remainder`, `frame_rounding`,
`output_fps`, and the per-entry `default_steps` / `default_width` / `default_height` /
`default_frames` / `default_video_shift` / `default_audio_shift`.

Three proposals were dropped on the adversarial pass's advice: `supports_audio_toggle` (definitionally
equal to `generates_audio` for every current and plausible route), `canvas_multiple` (duplicates the
published `dimension_multiple`), and a row-level `supports_guidance` (near-redundant with the
published `supports_negative_prompt` for a guidance-distilled route). `audio_prompt_sections` as a
bare tuple of names was replaced by the richer `PromptSection` descriptor - `key`, the engine's
literal `label`, the `option` and `parameter` that fill it, `role`, `required` - because a host needs
the literal string to detect a label a caller already wrote, not just a count.

`--release-text-encoder` reached the CLI (the kwarg already existed on `generate_video`) and is
advertised as `supports_text_encoder_release`.

### Validation

- `tests/minimax_h3/`: registry delivery and phase order, generation without either sink, the frame
  grid and bounds, duplicate-section refusal and line-anchored label detection, the full capability
  contract, distinct labels and per-entry defaults, keyword rejection. 29 tests.
- Full no-weights band: 2863 passed, 46 skipped, 1 pre-existing local failure
  (`tests/swiftvr/test_streaming_dit.py::TestTemporalOffset`, an MLX TF32 tolerance issue on M5 that
  passes on CI - see the note in that file's history).
- Model-backed run through `load_generation_model` (`untracked/h3_out/verify_host_contract.py`),
  `minimax-h3-turbo-544p`, q8, seed 42, 124 frames: the registry delivered
  `start, denoise x8, decode, generated, save, complete` with no `denoising` and exactly one
  `complete`; `fps=24` was refused before the run with the accepted set named; the clip muxed AAC
  stereo at 32 kHz and recorded `frames: 124`, `requested_frames: 124`, and a `video_health.file`
  block describing the muxed file.

## Addendum - 2026-09-06: four follow-up asks

BlackPixel filed four more asks after reviewing 0.35.0. A second adversarial pass
(`untracked/h3_verify/agent5/`) settled each.

- **`recommended_quantize` (their highest-value item): implemented, reshaped.** The claim is
  correct - unquantized the loaded shards are 125 GiB, 97% of a 128 GiB machine, so q8 is required
  there rather than optional, and the load previously ended in a silent OS kill. A bare int was the
  wrong shape because "unrunnable" is relative to the host's memory, so the rows publish the generic
  precision contract `RestorationCapability` already had (`weight_precision`,
  `validated_quantization_bits`) plus `unquantized_weights_bytes` and `recommended_quantize`, letting
  a host decide for itself. The field alone does not help CLI users, so `MiniMaxH3Initializer`
  also refuses an impossible load before reading weights, following the SeedVR2 precedent
  (`seedvr2_upscale.py:1005`) and ADR 0002: it reports and stops, never quantizing unasked.
- **`peak_memory_gb`: implemented, not as a scalar.** The stated reason was wrong and is corrected
  in the fourth addendum: 2.75x is the latent-frame ratio, not the peak ratio. The decision stands on
  better grounds - a scalar cannot separate the fixed floor from the part a request controls. `measured_peak` carries `peak_bytes`
  with the `quantize`, `width`, `height` and `frames` it was measured at. It publishes the process
  footprint rather than the MLX allocator peak, because the footprint is what the OS kills on.
- **`supports_solver` / `supports_low_ram`: not implemented.** Both are Wan-only mechanisms -
  `--solver` is defined in exactly one file (`wan_generate.py:409`), and `--low-ram`'s effects are
  Wan and SeedVR2 specific. A boolean that is false everywhere but Wan is a Wan detail leaking into
  the shared row; it should wait for a second family to fill it. H3 already publishes the memory
  lever a host can act on, `supports_text_encoder_release`.
- **`supports_guidance`: implemented. Their push-back was right and the first round was wrong.**
  The `supports_negative_prompt` proxy is already broken today in both directions: distilled FLUX.2
  Klein publishes guidance with no negative prompt, and Z-Image Turbo the reverse. Stamped centrally
  from `ModelConfig.supports_guidance` next to the existing negative-prompt stamp, and the dead
  `supports_guidance` parameter on `_image_latent_capabilities` (accepted, never read) was removed.

Also corrected from that pass: `supports_text_encoder_release` is documented as lowering the
denoise-time peak only, since the conditioner is resident throughout the larger load-time peak; and
the ambiguous memory parenthetical in `docs/minimax-h3.md` now names which figure is the MLX peak and
which the process footprint.

Their cache-limit companion ask was self-flagged as low confidence (one failure against two
successes) and was not acted on; it is not reproducible from the evidence available.

## Second addendum - 2026-09-06: a naming defect of our own

BlackPixel reported that they had aliased our `supports_video_shift` onto their generic flow-shift
flag and silently stopped sending Wan its `--flow-shift`. They called it their bug. A third
adversarial pass (`untracked/h3_verify/agent6/`) found it was ours.

The mechanism is one: the sigma transform in `minimax_h3_scheduler.py:58` and
`wan_euler_scheduler.py:52` is identical, MiniMax-H3's own help string calls its option a "video flow
shift", and both families write the shared `flow_shift` metadata key. A shared row field named after
one family's spelling therefore published `false` on the family that used the other, which is a
positive and untrue claim. Item 0123 had already written down the warning - "Wan's shift control is
named `--flow-shift`, not `--video-shift`; decide before populating, because the name is part of the
published contract" - and the field was populated anyway.

Fixed by renaming to `supports_flow_shift` / `default_flow_shift` and stamping both centrally from
each entry's own config, so Wan and MiniMax-H3 publish `true` with their real defaults and a family
with no such control publishes `false` truthfully. Free to do here because schema 13 and 14 exist
only on this branch: no released consumer had read the old name.

Their re-raised `--low-ram` ask was also better founded than our first refusal. That option is not
Wan-specific at all: it lives in the shared `CommandLineParser.add_general_arguments()`, so every
other generate route accepts it and MiniMax-H3 was the single one rejecting it. Rather than declaring
`supports_low_ram: false`, the route now accepts the option and maps it to its own levers, tightening
the MLX cache and releasing the conditioner after encoding. `--solver` stays rejected, since it is
genuinely Wan-only and MiniMax-H3 has one scheduler.

The general problem behind both - a host cannot tell which options a route accepts without per-family
knowledge - is filed as [0124](../proposed/0124_publish_accepted_options_per_route.md), deriving the
list from each route's parser so it cannot drift. Adding one `supports_<option>` boolean per case is
the accretion that produced this defect in the first place.

## Third addendum - 2026-09-06: declaring what is already true

Two remaining asks, both "declare a fact the build already has".

- **The flow-shift spelling.** The field is named for the concept, which was the right correction,
  but a host still has to emit an actual flag or keyword and was keeping its own per-family table.
  Rows now carry `flow_shift_option` and `flow_shift_parameter` (`--flow-shift` / `flow_shift` on
  Wan, `--video-shift` / `video_shift` on MiniMax-H3), both null exactly when `supports_flow_shift`
  is false. Publishing the spelling as a value is the inverse of the defect fixed in the second
  addendum, which was naming a shared field after one spelling.
- **`--low-ram` discovery.** The host was gating on `schema_version >= 14`, and their objection is
  right that a version check is what capability rows exist to remove. But a per-row
  `supports_low_ram` would be true on every row, since every generate route accepts the option; its
  only signal would be that the field exists, which is a version check by another name. The payload
  carries `universal_options` instead: the options every generate route in this build accepts, always
  populated, so an empty list would be a real statement rather than an unfilled one.
- **`--solver` was not declared.** It is genuinely Wan-only and is the first of an unbounded series
  (`--denoising-step-list`, the `--svi-*` family, `--context-noise`). Declaring it as a boolean is the
  accretion that produced the naming defect; it is answered properly by the derived per-route option
  list in [0124](../proposed/0124_publish_accepted_options_per_route.md).

Also recorded from that pass: "supports" is the wrong verb for the low-RAM axis in general, because
SwiftVR forces it and SeedVR2 video requires it, so a future per-route field should carry
accepted/required/forced rather than a boolean.

## Fourth addendum - 2026-09-06: the peak model, measured

BlackPixel corrected the reasoning recorded above, and they were right. This item said "the peak
scales about 2.75x across the allowed frame range". That is the latent-frame ratio (37 at 124 frames
to 102 at 345), not the peak ratio. Two independent recomputations agree. The conclusion to publish a
measurement with its geometry rather than a scalar was still correct, but for the opposite reason:
the scalar is not dishonest because the frame axis moves it a lot, it is insufficient because a host
cannot separate the fixed floor from the part a request controls.

Their functional form is also right, and is now measured rather than fitted. They proposed that peak
follows the packed sequence length. A probe run for this addendum tested it on orthogonal axes: a
243-frame `960x544` request has 37,730 packed rows against the 124-frame `1344x768` request's 37,910,
a 0.5% difference, and both reached an MLX peak of 84.8 GiB. The resolution axis and the frame axis
are the same axis.

Their conclusion, that every legal configuration fits in 128 GB, is refuted. That 243-frame probe was
killed by the OS at a process footprint of at least 92.7 GiB, four denoise steps in, at recommended
settings with no cache override. It died at 72% of the machine's memory, with another application
holding 7.5 GiB: feasibility is decided by what else is resident, not by the frame grid.

Two errors in their arithmetic, both in the conservative direction and neither load-bearing: their
345-frame row counts are high by 1,010 and 1,508 rows, and their two measured points straddle
`QUANTIZED_MATMUL_MAX_ROWS`, so a one-time chunking cost was absorbed into the fitted slope.

Shipped as a result:

- `packed_sequence_length(width, height, frames)`, so a caller sizes a request from the layout rather
  than reverse-engineering it, which is how the 345-frame counts went wrong.
- A request preflight alongside the existing load preflight: it estimates the peak from the packed
  rows, refuses what cannot fit at all, and warns when a request is close enough to the limit that
  other resident processes decide the outcome. It reports and stops, never altering the request.
- `measured_runs` replaces the singular `measured_peak`, carrying `outcome`, `footprint_bytes` (a
  peak for a completed run, a lower bound for a killed one, which is why it is not called
  `peak_bytes`), `terminated_at`, and the conditions that make a number reproducible. The killed run
  is published: it is the most informative of the three.
- `max_validated_frames` beside `max_frames`, because the grid bound and the evidence bound are
  different questions and publishing only the first invites exactly the projection that failed.
- The measured line itself, `peak_bytes_fixed` and `peak_bytes_per_packed_row`, as measured bytes and
  explicitly not a feasibility guarantee.
- Low-RAM mode no longer loses its cache tightening silently to an explicit `--mlx-cache-limit-gb`.
  The precedence is deliberate and a documented restore profile depends on it, so the fix is a
  warning rather than taking the minimum. The same configuration measured under a raised cache limit
  came out 16.8 GB higher, which is more than three times the difference between the two canvases.

Declined: capping `max_frames` by memory, which would hard-code one machine into the model contract
when a larger machine runs the full grid; and publishing 243 frames as a limit, which would be false,
since that request is iso-sequence with a shipped configuration that survives.

## Follow-ups filed

- [0121](../proposed/0121_minimax_h3_config_from_metadata_replay.md) - `--config-from-metadata`
  replay for MiniMax-H3.
- [0122](../proposed/0122_declare_metadata_keys_in_metadata_schema.md) - declare the `audio_*` and
  other runtime metadata keys in `metadata_schema.py`.
- [0123](../proposed/0123_publish_duration_and_audio_fields_on_other_families.md) - publish the new
  duration and sampling fields on the non-H3 families that have real values for them.
- [0124](../proposed/0124_publish_accepted_options_per_route.md) - publish each route's accepted
  options, derived from its parser.
