# Planned: Keep Wan q8 packages 8-bit in memory

## Metadata

- Created: 2026-10-04
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs:
  [0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md) (model-backed proof before
  a route is described as working),
  [0002](../../adr/0002_no_silent_automatic_fallbacks.md) (no silent change of the requested
  quantization policy),
  [0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md) (MLX-Gen reports runtime truth;
  consumers own convenience).
- ADR impact: None expected. If this work ends in a durable rule such as "a quantized package runs
  at its stored precision, and any runtime precision change is named in the metadata", record it
  as an ADR before closing the item.

## Context

Since commit `fd88c84` (2026-06-12, "Fix Wan q8 Lightning FFN runtime precision") a Wan q8 package
is 8-bit on disk only. At load, MLX-Gen converts every attention and feed-forward linear of the
transformer blocks back to BF16, which is nearly all of the model. An "8-bit" Wan2.2 A14B run
therefore needs BF16-class memory: about 28 GiB MLX peak and 33 GiB physical at the small
384x224 benchmark profile instead of 15.5 GiB and 20.7 GiB before the change
(`docs/quantization.md`, corrections dated 2026-07-05). `docs/recommendations.md` names 64 GB as
the first conservative tier for A14B for that reason.

The conversion was the closing repair of
[item 0040](../completed/0040_lightx2v_wan_4step_acceleration_profiles.md): a q8 T2V-A14B run at
1280x720, 41 frames, 4 LightX2V steps produced corrupted output while BF16 stayed clean. The item
recorded it as "a real runtime precision bug" and kept the attention and FFN families at BF16.

It was not a precision problem. On MLX before 0.32.0, `mx.quantized_matmul` on the GPU returns
wrong values when its input has 32,768 rows or more. A 1280x720, 41-frame clip is 39,600 rows.
MLX-Gen found this kernel fault three months later, for MiniMax-H3
([item 0117](../completed/0117_minimax_h3_text_to_video_audio_runtime.md): "any other q8 route
that crosses 32768 tokens needs the same"), and fixed that route by applying quantized linears in
row chunks. The link back to the Wan conversion was never made, so Wan still pays double memory
for a fault that has an exact, cheap repair.

## Current code reality

Audited 2026-10-04 against `99fb94d` (0.38.0).

The conversion:

- `WanInitializer._normalize_runtime_sensitive_q8_paths`
  (`src/mflux/models/wan/wan_initializer.py:631`) dequantizes, for `transformer` and
  `transformer_2` at q8, every linear whose path ends in `.attn1.to_q/.to_k/.to_v/.to_out.0`,
  `.attn2.to_q/.to_k/.to_v/.to_out.0`, `.attn2.add_k_proj/.add_v_proj`, `.ffn.net.0`,
  `.ffn.net.1`, to `ModelConfig.precision` (BF16). It runs at first load (`:242`) and again
  whenever the high-noise transformer is reloaded (`:121`). It prints
  `Normalizing Wan q8 runtime-sensitive paths to BF16 at load`.
- `WanWeightDefinition.quantization_predicate`
  (`src/mflux/models/wan/weights/wan_weight_definition.py:183`) builds those same paths, plus
  `proj_out` and `condition_embedder.*`, as plain linears at `bits == 8` through
  `_is_q8_sensitive_transformer_path` (`:191`).
- `WanInitializer._validate_component_quantization_layout` (`:610`) rejects weights that still
  carry `scales`/`biases` on any of those paths.
- `tests/wan/test_wan_quantization.py` pins this:
  `test_wan_initializer_accepts_current_q8_sensitive_bf16_layout` and
  `test_wan_initializer_rejects_old_q8_ffn_layout`.
- There is no flag, parameter or environment variable. The conversion is unconditional.
- The published q8 packages store the block linears 8-bit and were not rebuilt for the June
  change, so no package needs republishing whichever way this item ends.
- Before `fd88c84` the sensitive list was only `proj_out` and `condition_embedder.*`. That
  earlier rule has its own evidence and is not in question here:
  [item 0002](0002_wan_quantization_motion_parity.md) measured fully-quantized Wan at 384x224,
  17 frames as static and far from BF16 (sampled MAE 99.95, temporal change 2.03 versus 16.61).
  That clip is 1,680 rows, so it is unrelated to the row limit.

The MLX fault, measured 2026-10-04 on an M5 Max with a bare `nn.QuantizedLinear` (no model):

| MLX | 32,767 rows | 32,768 rows | 39,600 rows | 75,600 rows |
| --- | ---: | ---: | ---: | ---: |
| 0.31.0 (the `dev` pin) | 0 wrong | 32 wrong | 6,848 wrong | 32,784 wrong |
| 0.31.2 | 0 wrong | 32 wrong | 6,848 wrong | 32,784 wrong |
| 0.32.0, 0.32.1, 0.32.2, 0.32.3 | 0 | 0 | 0 | 0 |

- "Wrong" rows hold the bias alone in the bare-layer test: the matmul never wrote them. Inside a
  real run that unwritten memory is whatever the allocator recycled, which is how the June runs
  reached NaN and values at the BF16 maximum.
- GPU only; the CPU is right. Independent of layer width (256 and 5120), of 8-bit versus 4-bit, of
  FP32 versus BF16 input, and of FP32 versus BF16 scales (item 0117 recorded FP32 scales as exact;
  on 0.31.2 they were not).
- The limit counts rows across the whole batch: `(2, 16384, C)` fails, `(2, 16383, C)` does not.
- The same layer called on slices below the limit equals a plain matmul on the dequantized
  weights exactly.
- `pyproject.toml` admits `mlx>=0.27.0,<0.33.0` and the `dev` extra pins `mlx==0.31.0`, so
  affected MLX versions stay supported. Versions before 0.31.0 were not measured.

The existing repair, used by one family only:

- `src/mflux/models/minimax_h3/model/h3_precision.py` defines `QUANTIZED_MATMUL_MAX_ROWS = 32767`
  and `rowwise(layer, x)`, used by `h3_attention.py` for `to_q`, `to_k`, `to_v`, `to_out[0]` and
  the feed-forward linears; `tests/minimax_h3/test_h3_precision.py` covers it.
- `rowwise` chunks axis 1 and checks `x.shape[1]` only, so a batch whose total row count crosses
  the limit while each item stays below it is not protected.

Where Wan calls the affected linears:

- `src/mflux/models/wan/model/wan_transformer/wan_attention.py`: `to_q` (`:78`), `to_k`, `to_v`
  (`:79-80`), `to_out[0]` (`:169`), `add_k_proj`, `add_v_proj` (`:190-191`).
- `src/mflux/models/wan/model/wan_transformer/wan_transformer_block.py`: `net[0]`, `net[1]`
  (`:22-24`).
- Only calls whose input is the video token sequence can cross the limit. For A14B the row count
  is `((frames - 1) / 4 + 1) * (height / 16) * (width / 16)`, which matches the recorded failure
  shape `(1, 39600, 5120)` for 1280x720, 41 frames:

| A14B clip | Rows | Side of the limit |
| --- | ---: | --- |
| 832x480, 41 frames | 17,160 | below |
| 832x480, 81 frames | 32,760 | below, by 8 rows |
| 832x480, 85 frames | 34,320 | above |
| 1280x720, 33 frames | 32,400 | below |
| 1280x720, 41 frames | 39,600 | above |
| 1280x720, 81 frames | 75,600 | above |

- TI2V-5B packs `(height / 32) * (width / 32)` rows per latent frame, so its largest documented
  clip, 1280x704 at 121 frames, is 27,280 rows and never crosses the limit. Confirm this from a
  tensor shape before relying on it.

Other facts that constrain the change:

- The metadata sidecar does not say which precision the block linears ran at. A q8 package run
  writes `quantize: 8` and `precision: mlx.core.bfloat16` whether the blocks were converted or
  not.
- `SwiftVR._assert_quantization_supported`
  (`src/mflux/models/swiftvr/variants/upscale/swiftvr.py:398`) and the Bernini q8 rejection
  (`docs/quantization.md`, "Generic Wan q8 skipped every Bernini transformer linear") both reason
  from the current sensitive list. Narrowing the list changes what a nominal q8 would do on those
  routes; their rejections must stay.
- `docs/faq.md` ("Why Can q8 Show The Same Physical Peak As BF16?"), `docs/quantization.md`
  ("Wan q8", both corrections dated 2026-07-05) and `docs/recommendations.md` (A14B rows and the
  64 GB tier) describe Wan q8 as BF16-class at runtime.
- BlackPixel 0.9.3 keeps the three Wan2.2 q8 packages 8-bit on every run that goes through its
  Low RAM mode, by replacing `_normalize_runtime_sensitive_q8_paths` and
  `_is_q8_sensitive_transformer_path` in its generation process, and slices large quantized calls
  itself on MLX before 0.32.0 (`runtime_entry.py`). It ships MLX 0.31.2. Renaming or removing
  either function silently returns that host to the BF16-class path. A public behaviour here
  retires that dependency.

## Evidence so far

All of it is local and uncommitted. The June runs are under
`validation_outputs/lightx2v_wan_4step_2026_06_12_postrevert/`; the 2026-10-04 runs, logs and
probe scripts are under `validation_outputs/wan_q8_runtime_8bit/2026-10-04_probe/`.

The June 2026 failure, re-read:

- `a14b_t2v_41f_4step_lightning_q8_1280x720_euler_seed42_health.failure.json`: non-finite values
  first appear at the output of `blocks.0.attn2.to_q`, shape `(1, 39600, 5120)`, step 1 of 4,
  68,035 values, finite range at the BF16 maximum.
- `a14b_t2v_41f_4step_no_lora_q8_1280x720_euler_seed42_health.failure.json`: the same tensor fails
  with no LoRA loaded. The fault is not a LoRA interaction, although item 0040 describes it as
  one.
- `..._health_patch_attn2family.failure.json`: with the `attn2` family moved to BF16 the failure
  moves to `blocks.0.attn1.to_q` at step 2. Each family moved to BF16 removed one set of
  quantized calls above the limit, until none were left.
- `a14b_t2v_41f_4step_lightning_q8_832x480_euler_seed42.mp4`: the same recipe at 832x480,
  17,160 rows, completed cleanly in the same session. Which linears were 8-bit in that exact run
  is not recorded; the 2026-10-04 runs below are the firm evidence below the limit.
- `frame_compare_attn_ffn/q8_old_vs_new_vs_bf16_contact_sheet.png`: in the row made before the
  FFN family was moved to BF16, frame 0 is noise, frame 5 is partly noise and later frames are
  clean. At 39,600 rows the kernel skips rows 0 to 6,847, which are the first 1.9 latent frames
  of 3,600 rows each: the start of the clip, and nothing else.
- Three runs of the same 720p case named `_lowram` completed without a health error on
  2026-06-12, hours before the conversion was committed (their exact code state is not recorded).
  A skipped row holds whatever the buffer held; with the cache cleared between blocks that can be
  finite, so a run can pass the non-finite check and the clip still be wrong.
- Across the 114 A14B q8-package runs under `validation_outputs/` that record their size, every
  failure is at or above 32,768 rows (39,600 or 75,600) and none is below.
- `wan_a14b_fullsize_q8_2026_06_06/`: 1280x720, 81 frames (75,600 rows) fails the same way at
  `blocks.0.attn2.to_q` even in a 1-step run. The 13h15m all-black run recorded in item 0002 is
  that size on the same MLX line and is very likely the same fault. It was not repeated.

Runs on 2026-10-04 (M5 Max 128 GB, MLX 0.31.2, MLX-Gen 0.37.0, q8 packages, LightX2V 4-step
pairs, guidance 1/1, flow shift 5, low-RAM, cache limit 3 GB), with the block linears kept 8-bit
by replacing the two functions above for the process:

| Run | Rows | Block linears | MLX peak | Physical peak |
| --- | ---: | --- | ---: | ---: |
| I2V-A14B 832x480, 33 frames, unipc, seed 11 | 14,040 | BF16 (default) | 31.7 GiB | 34.9 GiB |
| I2V-A14B 832x480, 33 frames, unipc, seed 11 | 14,040 | 8-bit | 18.9 GiB | 21.9 GiB |
| T2V-A14B 832x480, 33 frames, unipc, seed 11 | 14,040 | 8-bit | 18.8 GiB | 22.2 GiB |
| I2V-A14B 832x480, 81 frames, unipc, seed 11 | 32,760 | 8-bit | 23.3 GiB | 26.2 GiB |
| T2V-A14B 1280x720, 41 frames, euler, seed 42 (the June case) | 39,600 | BF16 (default) | 37.6 GiB | 40.3 GiB |
| T2V-A14B 1280x720, 41 frames, euler, seed 42 (the June case) | 39,600 | 8-bit, calls sliced | 34.1 GiB | 37.3 GiB |
| TI2V-5B 480x480, 49 frames, FastWan (earlier run, other source and seed) | 2,925 | BF16 (default) | 11.6 GiB | 13.5 GiB |
| TI2V-5B 480x480, 49 frames, FastWan, seed 11 | 2,925 | 8-bit | 8.8 GiB | 12.5 GiB |
| TI2V-5B 832x480, 25 frames, FastWan, seed 11 | 2,730 | BF16 (default) | 14.3 GiB | 17.8 GiB |
| TI2V-5B 832x480, 25 frames, FastWan, seed 11 | 2,730 | 8-bit | 14.3 GiB | 18.0 GiB |
| TI2V-5B 832x480, 81 frames, FastWan, seed 11 | 8,190 | 8-bit | 14.3 GiB | 17.8 GiB |

- The 33-frame I2V pair, compared on all 33 decoded frames: mean absolute difference 1.09 to 3.03
  on the 0-255 scale (mean 2.40), PSNR 33.1 dB at worst, frame-to-frame change 6.14 versus 6.15.
  Side by side the two clips are indistinguishable at normal viewing
  (`i2v_a14b_832x480_33f_side_by_side_8bit_left_bf16_right.mp4`,
  `i2v_a14b_832x480_33f_frame_comparison.json`).
- The June failure case, repeated both ways on the same day (MLX 0.31.2, low-RAM,
  `--tensor-health-check-interval 1`, the June prompt and negative prompt). With the block
  linears 8-bit and quantized calls sliced below the row limit it completed with no health error,
  and the clip is clean from frame 0 with the same scene and composition as its BF16 twin and as
  both June clips (`t2v_a14b_1280x720_41f_early_frames_sheet.png`,
  `t2v_a14b_1280x720_41f_side_by_side_8bit_left_bf16_right.mp4`,
  `t2v_a14b_1280x720_41f_frame_comparison.json`).
  - 8-bit against its same-day BF16 twin: mean absolute difference 13.6 (10.0 to 18.0),
    frame-to-frame change 13.07 against 13.01.
  - For scale, the June q8 run converted to BF16 against the June BF16 package: 12.8 (9.2 to
    17.0). At this size and with this fine snow texture, two 16-bit-class runs already differ by
    about that much, so the 8-bit run sits at the noise floor of this clip, not above it.
  - Both of today's clips are equally far from the June BF16-package clip (18.4 and 17.9): that
    distance is the engine version and low-RAM mode (0.18.16 without it, 0.37.0 with it), not the
    weights.
  - At this size the 8-bit runtime saves about 3 GiB, not 13: the peak is no longer set by the
    weights.
- TI2V-5B runs with its block linears 8-bit and the FastWan adapter. At 832x480, 25 frames the
  two clips differ by 0.9 to 2.0 on the 0-255 scale with identical motion
  (`ti2v_5b_832x480_25f_side_by_side_8bit_left_bf16_right.mp4`). Its peak drops at 480x480 and
  does not move at 832x480, so at that canvas something other than the transformer sets the peak.
  The two 832x480, 25-frame physical peaks (17.8 and 18.0 GiB) are within run-to-run variation.
- Both LoRA files applied fully in both modes (`Applied to 400 layers (1200/1200 keys matched)`).
- Limits of this evidence: one seed, one prompt per route, 4-step adapters only, and the
  comparisons are against the same q8 package converted at load, not against the BF16 package.
  Several runs shared the GPU, so timings are not evidence of speed. The 720p run sliced its
  quantized calls in the host process, not through MLX-Gen code.

## Problem

A Wan q8 package uses twice the memory it needs. The conversion that causes it works around an
MLX kernel fault that MLX-Gen already knows how to avoid exactly, and that MLX itself fixed in
0.32.0. The metadata hides the conversion, so a host cannot tell which run it got.

## What we want to do

Run Wan q8 packages with their block linears 8-bit in memory, at every clip size, on every
supported MLX version, with quality proven against the BF16 package; then retire the load-time
conversion as the default and correct the published memory numbers.

## Why

- Wan2.2 A14B comes within reach of 32 GB to 48 GB Macs. 832x480 at 33 frames measured 18.9 GiB
  MLX peak and 21.9 GiB physical, which on paper fits a 24 GB Mac with a 20 GB GPU limit; it has
  not been run on such a Mac. Today the docs name 64 GB as the first tier for the same package.
- A package named `-8bit` running at 16-bit is a runtime-truth gap under ADR 0003.
- One MLX fault should have one repair, shared by every family that can cross the row limit.

## Requirements

- No quantized linear call in a Wan run may reach `mx.quantized_matmul` with 32,768 rows or more
  on an MLX version that has the fault. Count rows across the whole batch.
- Results below the limit must be bit-identical to today's direct call.
- `proj_out` and `condition_embedder.*` stay BF16 (item 0002's evidence).
- Published q8 packages keep loading unchanged.
- The sidecar records the runtime precision of the block linears.
- If a BF16 runtime for the block linears remains available, it is a named option the caller asks
  for. No automatic choice by available memory inside MLX-Gen (ADR 0002, ADR 0003): a host that
  wants one implements it.
- SwiftVR and Bernini keep rejecting the quantization levels they reject today, with reasons that
  remain true.
- No public claim changes before the validation below passes.

## Suggested implementation

1. Move `QUANTIZED_MATMUL_MAX_ROWS` and `rowwise` out of `minimax_h3` into a shared module, make
   the row count the product of all leading dimensions, and keep MiniMax-H3 importing it. Port
   `tests/minimax_h3/test_h3_precision.py` with a batched case such as `(2, 16384, C)`.
2. Route the Wan calls listed above through it. Decide whether to chunk only when
   `mx.__version__ < 0.32.0`; chunking always is right on every version, so only skip it if a
   measurement shows a cost.
3. Build a validation switch first, without changing the default: one process-level way to run a
   q8 package with block linears 8-bit, with them BF16, and with the attention and FFN families
   toggled separately. The replacement used for the 2026-10-04 runs is enough to start:

   ```python
   from mflux.models.wan.wan_initializer import WanInitializer
   from mflux.models.wan.weights.wan_weight_definition import WanWeightDefinition

   WanInitializer._normalize_runtime_sensitive_q8_paths = staticmethod(lambda *a, **k: None)
   WanWeightDefinition._is_q8_sensitive_transformer_path = staticmethod(
       lambda path: path == "proj_out" or path.startswith("condition_embedder.")
   )
   ```

   Both are needed: without the second, the modules are built as plain linears and LoRA loading
   fails on the packed weight shape.
4. Run the validation matrix. If it passes, narrow `_is_q8_sensitive_transformer_path` to
   `proj_out` and `condition_embedder.*`, remove or gate the load-time conversion, and update
   `_validate_component_quantization_layout` and the two layout tests.
5. Add the block-linear runtime precision to the sidecar and to the metadata schema work in
   [item 0122](../proposed/0122_declare_metadata_keys_in_metadata_schema.md).
6. Update `docs/quantization.md`, `docs/faq.md`, `docs/recommendations.md`, the package cards if
   they quote runtime memory, `CHANGELOG.md` and the `llms` files with the re-measured numbers.
7. Tell the BlackPixel host which release makes its process-level replacement unnecessary.

## Scope

- Wan2.2 T2V-A14B, I2V-A14B and TI2V-5B q8 packages, and every route that loads them through
  `WanInitializer` (text-to-video, image-to-video, video-to-video, VACE where it uses a q8
  package).
- The shared row-chunk helper and its use by MiniMax-H3.
- Metadata, tests and docs that describe Wan q8 runtime memory.

## Non-goals

- Quantizing `condition_embedder.*` or `proj_out`.
- A Wan q4 package.
- Raising the MLX floor to 0.32.0. It would also remove the fault, but it moves the `dev` pin and
  the image-comparison baselines (0.38.0 changelog); treat it as a separate decision.
- Auditing every other quantized family for calls above the row limit. Images above roughly
  2896x2896 at 16 px per token, and SeedVR2 video windows, may cross it. File that audit as its
  own item once the shared helper exists.
- An automatic memory-based precision switch inside MLX-Gen.
- Speed claims. [Item 0005](0005_wan_q8_performance_investigation.md) owns q8 speed; record
  timings here only as input to it.

## Dependencies and related tasks

- [0002 Wan quantization and motion parity](0002_wan_quantization_motion_parity.md): the mixed
  q8/BF16 policy and the 2026-06-03 all-black full-size run.
- [0005 Wan q8 performance investigation](0005_wan_q8_performance_investigation.md).
- [0013 Wan A14B boundary memory recovery and full-size validation](0013_wan_a14b_boundary_memory_recovery.md):
  its full-size q8 validation should run after this item, on the 8-bit runtime.
- [0016 Wan video integrity release gate](../completed/0016_wan_video_integrity_release_gate.md):
  the health gates to reuse.
- [0040 LightX2V Wan 4-step acceleration profiles](../completed/0040_lightx2v_wan_4step_acceleration_profiles.md):
  the completion report that introduced the conversion. Append a dated note there when this item
  closes; do not rewrite it.
- [0089 Wan memory defaults](../proposed/0089_wan_memory_defaults.md).
- [0117 MiniMax-H3 runtime](../completed/0117_minimax_h3_text_to_video_audio_runtime.md): where
  the MLX fault and the row-chunk repair were first recorded.

## Expected outcomes

- A14B q8 at 832x480, 33 frames, low-RAM: about 19 GiB MLX peak and 22 GiB physical, instead of
  about 32 GiB and 35 GiB. The gain shrinks as the clip grows: at 1280x720, 41 frames it measured
  34.1 against 37.6 GiB MLX peak, so the memory tiers must be re-measured per size, not scaled.
- Wan q8 runs at 1280x720 and at 81 frames complete with healthy tensors and track the BF16
  package.
- The docs' memory tables and Mac tiers describe the 8-bit runtime.
- BlackPixel drops its replacement of private functions.

## Validation

Unit level:

- The shared helper is exact above the limit, bit-identical below it, and handles a batched input.
- A Wan attention block and feed-forward block with q8 linears match their dequantized reference
  at 39,600 rows.

Model-backed, on the 128 GB M5 Max, each as a same-seed set of three: BF16 package (reference),
q8 package converted at load (today's default, which sets the noise floor), q8 package with block
linears 8-bit (candidate). Run with `--tensor-health-check-interval 1` and `--metadata`.

1. T2V-A14B, 1280x720, 41 frames, 4 LightX2V steps, euler, seed 42, the prompt and negative
   prompt recorded in the June failure files, plus two more seeds. This is the case that caused
   the conversion. Seed 42 was run on 2026-10-04 as a q8 pair (see Evidence so far), with the
   slicing done by the host process; still owed are the BF16-package reference, the two other
   seeds, and the run through MLX-Gen's own helper.
2. T2V-A14B, 832x480 at 81 frames and at 85 frames: one clip on each side of the limit.
3. T2V-A14B, 1280x720, 81 frames, 4 LightX2V steps: 75,600 rows.
4. I2V-A14B, 1280x720, 41 frames, 4 LightX2V steps.
5. T2V-A14B without LoRA, 20 steps, guidance 4/3: 832x480 at 41 frames, and one 1280x720 clip.
6. TI2V-5B, 1280x704, with and without the FastWan adapter, to confirm no regression where the
   limit is never crossed.
7. One video-to-video route on the A14B q8 package at its documented size.
8. Cases 1 and 2 on both MLX 0.31.0 and the newest admitted 0.32.x.

Accept the 8-bit runtime as the default only if, for every case:

- no `TensorHealthError`, and the video-integrity gate of item 0016 passes;
- per-frame error against the BF16 package is within the range the converted-at-load run shows
  against the same reference, with no early-frame spike (the June symptom);
- frame-to-frame change stays within 10% of the BF16 run;
- every clip is watched in full at native size next to its reference, not only sampled;
- MLX peak and physical peak are recorded for all three runs.

If any case fails, keep the conversion for that envelope only, name the envelope in the docs, and
record what failed here.

## Progress checklist

- [ ] Shared row-chunk helper with batched row count; MiniMax-H3 moved onto it.
- [ ] Wan block linears routed through the helper.
- [ ] Validation switch for 8-bit, BF16 and per-family runs.
- [ ] Validation matrix run; artifacts under `docs/assets/validation/` with a report.
- [ ] Default decided from the matrix and recorded here.
- [ ] Sensitive list, load-time conversion, layout validator and tests updated.
- [ ] Runtime precision of block linears in the sidecar.
- [ ] SwiftVR and Bernini rejections re-checked.
- [ ] `docs/quantization.md`, `docs/faq.md`, `docs/recommendations.md`, `CHANGELOG.md`, `llms`
      files updated with re-measured numbers.
- [ ] Dated note appended to item 0040; follow-up audit item filed for other quantized families.
- [ ] BlackPixel told which release retires its replacement.

## Guidance for the implementing agent

- Reproduce the fault with a bare layer before touching Wan. It takes seconds and needs no model:

  ```python
  import mlx.core as mx
  from mlx import nn

  layer = nn.QuantizedLinear.from_linear(nn.Linear(256, 256), group_size=64, bits=8)
  x = mx.random.normal((1, 39600, 256))
  whole = layer(x)
  sliced = mx.concatenate([layer(x[:, i : i + 8192]) for i in range(0, 39600, 8192)], axis=1)
  print(mx.__version__, float(mx.abs(whole - sliced).max()))  # about 3 before MLX 0.32.0, 0 after
  ```

- The tensor-health check only catches non-finite values. Rows the kernel skipped can hold finite
  garbage, so a run can pass the check and still be wrong. Compare against the reference clip.
- Only one clean 8-bit run above the row limit exists so far (1280x720, 41 frames, seed 42). Do
  not generalise from it, or from the 480p runs, to other seeds, lengths or routes.
- Do not change the default before the matrix passes, and do not describe the 8-bit runtime in
  public docs before then.
- The June investigation's intermediate clips and failure files are still on disk in
  `validation_outputs/`; they are local and ignored by git, so copy what the report needs into
  `docs/assets/validation/`.
