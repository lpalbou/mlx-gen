# Planned: SeedVR2 chunk-bounded video noise

## Metadata
- Created: 2026-06-27
- Status: Planned
- Completed: N/A
- Reopened: 2026-06-27; 2026-06-28

## ADR status
- Governing ADRs: ADR 0004
- ADR impact: None

## Context
SeedVR2 video restore streams source frames and output writes, but it still builds one global noise
tensor for the whole requested clip before chunk processing.

## Current code reality
- `restore_video_to_path()` plans chunks, rejects production multi-chunk profiles below 29 frames
  or below 8 frames of overlap, and then processes each chunk through the streamed writer.
- `_build_streamed_video_noise_provider()` keeps ordinary package execution on coherent
  `seedvr2_noise_mode=global` noise. The bounded `absolute_latent_frame` path is internal
  benchmark-only because it changes restored pixels.
- ADR 0004 requires explicit host-safety boundaries for SeedVR2 video.

## Problem
Global device-resident noise means long clips still scale with total requested length, not only chunk
size.

## What we want to do
Preserve one coherent noise timeline while keeping only the active chunk slice in MLX memory.

## Why
This makes streamed video restore closer to its public memory contract and reduces unified-memory
pressure on long clips.

## Requirements
- Preserve overlap consistency across chunks.
- Avoid independent per-chunk noise that creates seam risk.
- Do not reduce temporal context below the production floor to save memory.
- Keep deterministic behavior for a fixed seed as much as practical.
- Clean up temporary backing stores on success and failure.

## Suggested implementation
Use a chunk-addressable host or disk-backed noise store for long streamed videos, and convert only
requested slices to MLX arrays during chunk processing.

## Scope
- SeedVR2 streamed video restore path.
- Unit tests for slice shape, overlap consistency, and cleanup behavior.

## Non-goals
- Do not change the in-memory single-image or direct small-video path.
- Do not weaken safe-video budget checks.
- Do not silently enable unsafe enlarged-video profiles.

## Dependencies and related tasks
- [0060 runtime memory telemetry](../../completed/0060_runtime_memory_telemetry_and_manifests.md)
- Proposed [0048 SeedVR2 enlarged-video safe-profile certification](../../proposed/0048_seedvr2_enlarged_video_safe_profile_certification.md)

## Expected outcomes
- Streamed video restore no longer keeps full-clip noise as a live MLX tensor.
- Chunk overlap slices are stable and coherent.
- Memory metadata distinguishes chunk-bounded noise mode.

## Validation
- Unit tests for a synthetic clip plan and noise store.
- Existing SeedVR2 metadata and CLI tests continue to pass.

## Progress checklist
- [x] Add chunk-addressable noise store.
- [x] Route streamed restore through chunk slice conversion.
- [x] Add cleanup and metadata.
- [x] Add focused tests.

## Guidance for the implementing agent
Do not use independent per-chunk random seeds as a shortcut; overlap consistency is part of output
quality.

## Completion report

- Date: 2026-06-27
- Original path: `docs/backlog/planned/memory/0062_seedvr2_chunk_bounded_noise.md`
- Final path: `docs/backlog/completed/0062_seedvr2_chunk_bounded_noise.md`
- Summary: Replaced always-global SeedVR2 streamed-video noise residency with a bounded provider
  that preserves the old global tensor path for small clips and uses deterministic absolute-frame
  slices for larger clips.
- Implementation: Added `SeedVR2StreamedVideoNoiseProvider`, routed restore chunk processing
  through it, recorded noise mode/version/estimated bytes in metadata, and released provider state
  before validation.
- Behavior changes: Small clips stay on materialized global noise to preserve existing behavior.
  Large clips avoid a full-clip MLX noise tensor and use deterministic absolute latent-frame noise
  to keep overlap windows coherent.
- Validation: Focused SeedVR2 chunk metadata and global-slice tests passed. The full
  `tests/arg_parser/test_seedvr2_upscale_argparser.py` file still segfaults cumulatively in the
  local pytest process after many tests, including during pytest `tmp_path` setup, so broad-file
  parser validation remains a local native-runner limitation rather than a clean pass.
- Residual risk: The large-clip provider intentionally changes the exact random sequence compared
  with a hypothetical full global tensor for very large videos, so public quality claims should use
  real visual validation on the target safe profile.

## Reopen report

- Date: 2026-06-27
- Reason: The implementation avoids full-clip noise in large mode, but closure requires quantitative
  memory statistics and quality/continuity evidence from runs that actually exercise
  `seedvr2_noise_mode=absolute_latent_frame`.
- Required evidence before closure: process-isolated baseline-versus-bounded runs or a controlled
  real-profile threshold test with peak physical footprint/RSS, MLX peak, wall time, metadata mode
  fields, overlap consistency, and temporal boundary-quality checks.

## Quantitative validation update

- Date: 2026-06-27
- Evidence artifact:
  `validation_outputs/memory/real_generation_20260627_seedvr2_r3/generation_memory_benchmark.json`.
- Real profile: `mlxgen upscale`, `AbstractFramework/seedvr2-3b-4bit`, 640x480 source, 9 frames,
  1x restore, chunk size 9, overlap 0, seed 7171. The benchmark forced the baseline
  `seedvr2_noise_mode=global` and candidate `seedvr2_noise_mode=absolute_latent_frame` with three
  fresh CLI runs per variant.
- Result: all six runs exited 0 after production fixes for video finalization. Median metadata
  physical footprint fell from 3.241 GB to 3.172 GB, median metadata RSS fell from 3.099 GB to
  3.030 GB, and median sampled RSS fell from 3.174 GB to 3.160 GB. Median MLX peak stayed flat at
  about 22.59 GB because chunk/model compute dominates this small profile.
- Quality result: saved videos were valid 9-frame 640x480 outputs, but bounded-noise output was
  not pixel-identical to global-noise output (`mae=6.6854`, `rmse=11.8118`, `psnr=26.7432 dB`).
- Status: Still planned. The candidate is stable and measured, but the small real profile is not
  enough to close the long-clip memory claim, and the bounded RNG path needs explicit visual or
  continuity acceptance for non-identical noise.

## Additional SeedVR2 1280px image validation

- Date: 2026-06-27
- Evidence artifacts:
  `validation_outputs/memory/real_generation_20260627_seedvr2_image_1280_r2/generation_memory_benchmark.json`
  and
  `validation_outputs/memory/real_generation_20260627_seedvr2_image_1280_tiling_r2/generation_memory_benchmark.json`.
- Real image profiles: `mlxgen upscale`, `AbstractFramework/seedvr2-3b-4bit` and
  `AbstractFramework/seedvr2-7b-4bit`, deterministic 960x960 source upscaled to 1280x1280, seed
  8181, two fresh CLI runs per variant, offline Hugging Face cache enforced.
- Stability fix: SeedVR2 image `--low-ram` no longer registers the generic `MemorySaver` callback.
  The callback path segfaulted native MLX/Pillow execution on 1280px image restore; the stable path
  applies the MLX cache policy before generation and preserves the default SeedVR2 VAE encode
  behavior unless `--vae-tiling` is explicit.
- `--low-ram` result: output stayed pixel-identical for both model sizes (`mae=0`, `rmse=0`,
  `max_abs=0`). Median metadata physical footprint fell from 6.423 GB to 4.249 GB for 3B
  (-33.85%) and from 8.246 GB to 6.638 GB for 7B (-19.50%). Median MLX peak stayed flat
  (13.412 GB for 3B, 12.618 GB for 7B), so this is a retained-footprint/stability improvement,
  not a peak-memory solution.
- Tuned explicit `--vae-tiling` result: SeedVR2 CLI tiling now uses 768px encode tiles with 128px
  overlap. Median MLX peak fell from 13.412 GB to 7.210 GB for 3B (-46.24%) and from 12.618 GB to
  9.569 GB for 7B (-24.16%). Output changed measurably but moderately: 3B `mae=1.969`,
  `rmse=2.9851`, `psnr=38.6316 dB`; 7B `mae=1.9514`, `rmse=3.0425`, `psnr=38.4662 dB`.
- Status: Still planned. The 1280px image path is now measured and safer, but the original
  long-video chunk-bounded-noise claim remains open. Keep SeedVR2 image `--vae-tiling` explicit
  because it is the only measured peak reducer here and it is not pixel-identical.

## Superseded quantitative completion report

- Date: 2026-06-28
- Evidence artifacts:
  `validation_outputs/memory/real_generation_20260628_0062_seedvr2_3b_bounded_scaling_r2/generation_memory_benchmark.json`,
  `validation_outputs/memory/real_generation_20260628_0062_seedvr2_3b_mode_149/generation_memory_benchmark.json`,
  and
  `validation_outputs/memory/real_generation_20260628_0062_seedvr2_7b_bounded_scaling_r2/generation_memory_benchmark.json`.
- Real source: `validation_outputs/seedvr2_video_2026_06_20/eiffel_70s_149f_source.mp4`, real
  `mlxgen upscale` video restore, offline cached `AbstractFramework/seedvr2-3b-4bit` and
  `AbstractFramework/seedvr2-7b-4bit`, positive temporal chunk overlap, and two fresh CLI runs
  per scaling variant.
- 3B bounded scaling result: frame count grew from 53 to 149 (2.81x) while median sampled RSS grew
  from 3.0875 GB to 3.1292 GB (+1.3516%), metadata Darwin physical footprint grew from 3.1292 GB
  to 3.1450 GB (+0.5034%), metadata process RSS grew from 2.9603 GB to 2.9760 GB (+0.5283%), and
  MLX peak stayed flat at 9.9736 GB. The 53-frame output used 4 chunks and the 149-frame output
  used 10 chunks, both with overlap 8.
- 3B same-frame mode result: bounded 149-frame mode compared against global 149-frame mode with the
  same input and seed. Median sampled RSS moved from 3.1384 GB global to 3.1308 GB bounded
  (-0.2422%), metadata physical footprint moved from 3.1705 GB to 3.1224 GB (-1.5164%), metadata
  process RSS moved from 3.0014 GB to 2.9536 GB (-1.5937%), and MLX peak stayed effectively flat.
  Video comparison succeeded with 149 frames at 320x240; bounded and global noise are intentionally
  not pixel-identical (`mae=6.8291`, `rmse=11.626`, `psnr=26.8616 dB`).
- 7B bounded scaling result: frame count grew from 53 to 101 (1.91x) while median sampled RSS grew
  from 5.3991 GB to 5.4240 GB (+0.4617%), metadata Darwin physical footprint grew from 5.4316 GB
  to 5.4372 GB (+0.1029%), metadata process RSS grew from 5.2616 GB to 5.2756 GB (+0.2662%), and
  median MLX peak fell from 11.7686 GB to 11.0169 GB. The 53-frame output used 7 chunks and the
  101-frame output used 13 chunks, both with overlap 4.
- Continuity checks: all scaling outputs passed video health and expected frame counts. Boundary
  continuity ratios stayed below the accepted gates: 3B 53f median boundary/non-boundary ratio
  0.8745 and max/p95 0.6469; 3B 149f 0.8198 and 0.6914; 7B 53f 0.8130 and 0.6746; 7B 101f
  0.8975 and 0.9197.
- Production hardening: SeedVR2 video restore no longer calls `os._exit(0)` after success, restores
  cyclic GC after success and failure, and video health now fails closed when a file cannot be
  decoded for visual statistics. The noise-mode byte limit override is also gated behind explicit
  internal benchmark mode, so ordinary package use cannot accidentally change SeedVR2 noise policy
  from ambient environment state. Focused SeedVR2 CLI, generated-video, and chunking tests passed
  after those changes.
- Superseded status at the time: Completed. This status was revoked by the 2026-06-28 correction
  reopen report below because the evidence accepted non-identical bounded-noise output and did not
  include a 7B normal same-video 1:1 baseline/candidate proof.

## Correction reopen report

- Date: 2026-06-28
- Reason: The 2026-06-28 completion standard was wrong. A memory optimization for a premium
  restoration path must not reduce output quality. The bounded `absolute_latent_frame` path changes
  restored pixels for the same source, seed, frame count, and chunking, so it cannot count as a
  quality-preserving memory optimization.
- Adversarial review findings: the 3B same-job A/B changed output
  (`mae=6.8291`, `rmse=11.626`, `max_abs=207`, `psnr=26.8616 dB`), while the 7B closure evidence
  used a `53f -> 101f` scaling run rather than a normal same-video 1:1 baseline/candidate test.
  Scaling evidence can show memory growth behavior, but it does not prove quality preservation for
  one restoration job.
- Code correction: normal package execution now keeps SeedVR2 streamed-video noise in `global`
  mode. The byte cap that can force `absolute_latent_frame` is gated behind explicit internal
  benchmark mode and cannot be activated by an ordinary environment variable during package use.
- Superseded normal 1:1 evidence rejected by temporal-quality review:
  `validation_outputs/memory/seedvr2_1to1_memory_quality_20260628/generation_memory_benchmark.json`
  and proof bundle `validation_outputs/seedvr2_1to1_memory_quality_20260628/`.
- Rejection note: the 2026-06-28 3B `25/8` and 7B `13/4` proof bundle is not production
  evidence. It preserved frame count but used temporal windows below the accepted `29/8` profile,
  so it is retained only as diagnostic material for the temporal-quality regression.
- 3B rejected diagnostic run: source and restored videos both contain exactly 149 frames at 320x240 and
  29.97002997002997 fps, `seedvr2_noise_mode=global`, chunk size 25, overlap 8, 10 chunks.
  Across three isolated runs, median sampled RSS was 3.1518 GB, median metadata physical footprint
  was 3.1282 GB, and median MLX peak was 9.9760 GB.
- 7B rejected diagnostic run: source and restored videos both contain exactly 149 frames at 320x240 and
  29.97002997002997 fps, `seedvr2_noise_mode=global`, chunk size 13, overlap 4, 19 chunks.
  Across three isolated runs, median sampled RSS was 5.4408 GB, median metadata physical footprint
  was 5.4489 GB, and median MLX peak was 11.0195 GB.
- Rejected diagnostic artifacts: the bundle contains the copied original MP4, copied restored 3B/7B
  MP4s, side-by-side source/restored videos, timeline contact sheets, boundary contact sheets,
  per-model quality stats, per-model memory stats, and `manifest.json`, but those artifacts are
  retained for regression diagnosis only and are not release or production proof.
- Status: Planned. Item 0062 remains open until a same-source, same-frame, same-seed 3B and 7B
  baseline/candidate run proves a real memory reduction without output degradation. For a
  memory-only implementation, the acceptance gate is pixel-identical output (`mae=0`, `rmse=0`,
  `max_abs=0`) or a stricter explicitly approved quality gate before any non-identical output is
  accepted.

## Research update

- Date: 2026-06-29
- External evidence: MLX follows a JAX-style splittable PRNG model, and the upstream MLX source
  implements `normal(...)` from keyed uniform bits rather than exposing a public counter-offset
  slice API. See the JAX PRNG design note, MLX random source, and MLX Python random docs:
  `https://docs.jax.dev/en/latest/jep/263-prng.html`,
  `https://github.com/ml-explore/mlx/blob/main/mlx/random.cpp`,
  `https://ml-explore.github.io/mlx/build/html/python/random.html`.
- Local proof: repeated smaller `mx.random.normal(...)` calls are not an exact reconstruction of
  one larger call. A local check with `seed=123` compared
  `mx.random.normal(shape=(1, 16, 5, 2, 3), key=mx.random.key(seed))` against smaller repeated
  calls and observed non-zero deltas in every candidate path (`max_abs` about `3.8301` to
  `4.5771`). Stateful seeded calls also failed exact concatenation, so a simple rolling RNG state
  does not preserve the current global-noise output.
- Current interpretation: an exact-quality chunk-bounded implementation is not available through
  current public MLX Python APIs by re-generating smaller slices. Remaining exact options are:
  `1.` materialize the exact global tensor once and spill it to a host/disk-backed exact slice
  store; `2.` add or adopt a lower-level MLX API that exposes deterministic counter-offset or
  bits-based slice generation; `3.` retire or defer the item for current supported profiles if the
  measured win remains economically negligible.
- Memory math: the current global noise bytes are
  `16 * latent_frames * latent_height * latent_width * 4`, which is approximately
  `width * height * frame_count / 4` bytes for multi-frame clips. On the accepted 149-frame
  `320x240` `29/8` proof profile this is `2.918 MB` global versus `0.614 MB` chunk-live, so the
  theoretical maximum live-noise reduction is only `2.304 MB`. At `149f 1280x720` it is
  `35.021 MB` global versus `7.373 MB` chunk-live. At `600f 1920x1080` it is `311.040 MB` global
  versus `16.589 MB` chunk-live.
- Consequence: for the currently validated low-resolution 149-frame restore proofs, item 0062 is a
  residency-cleanliness issue rather than a dominant whole-process peak-memory lever. It becomes
  more interesting only for materially longer and higher-resolution future safe-video profiles.
- Required next proof before more product effort: if this item stays planned, run one larger still
  supported SeedVR2 profile with explicit memory snapshots immediately before and after
  global-noise materialization. If the delta is still only in the MiB-to-low-tens-of-MiB range
  relative to total peak, treat the item as low-value for the current public profile set.

## Temporal quality repair update

- Date: 2026-06-28
- Reason: The normal 1:1 repair evidence above still used unsafe temporal profiles: 3B used
  `25/8`, while 7B used `13/4` and produced 19 chunks. Adversarial review found that this could
  preserve frame count while creating object-continuity distortions, especially on 7B.
- Code correction: SeedVR2 video restore now rejects production multi-chunk profiles below
  29 frames or below 8 frames of overlap from both the CLI planner and the public Python
  `restore_video_to_path()` API. The temporary internal `seedvr2_tiny_temporal_chunks` bypass was
  removed from production code after adversarial review, so user-set environment variables cannot
  reopen the unsafe path. `_aligned_chunk_overlap()` now returns stride-aligned overlap values only,
  so the previous invalid effective overlap `3` cannot be passed to the chunk planner; explicit
  CLI overlap values are either honored or rejected, not silently capped downward.
- Benchmark correction: the real 149-frame 3B and 7B restoration profiles in
  `tools/generation_memory_benchmark.py` now use `29/8`. The proof bundle now writes all production
  chunk boundaries into the boundary contact sheet and records per-boundary temporal-delta stats.
- Corrected evidence artifact:
  `validation_outputs/memory/seedvr2_temporal_quality_repair_20260628/generation_memory_benchmark.json`.
- Corrected proof bundle:
  `validation_outputs/seedvr2_temporal_quality_repair_20260628/`.
- 3B corrected proof: source and restored videos both contain exactly 149 frames at 320x240 and
  29.97002997002997 fps, `seedvr2_noise_mode=global`, chunk size 29, overlap 8, 8 chunks.
  Peak sampled RSS was 3.1625 GB, metadata physical footprint was 3.1440 GB, MLX peak was
  10.1068 GB, and video health reported temporal continuity `ok`.
- 7B corrected proof: source and restored videos both contain exactly 149 frames at 320x240 and
  29.97002997002997 fps, `seedvr2_noise_mode=global`, chunk size 29, overlap 8, 8 chunks.
  Peak sampled RSS was 5.5445 GB, metadata physical footprint was 5.5334 GB, MLX peak was
  12.4665 GB, and video health reported temporal continuity `ok`.
- Measurement caveat: the corrected proof used one isolated run per model, so it proves exact
  single-run process measurements and output artifacts, not multi-run variance. Use sampled RSS,
  metadata process RSS/physical footprint, and MLX peak as the proof fields; macOS
  `/usr/bin/time -l` peak memory footprint was not coherent for these runs.
- Post-review guardrail validation: focused SeedVR2 parser/chunking tests passed after removing the
  internal tiny-chunk bypass and adding explicit-overlap regression coverage:
  `uv run pytest tests/seedvr2/test_seedvr2_video_chunking.py tests/arg_parser/test_seedvr2_upscale_argparser.py -q`
  reported 74 passed. A real CLI rejection check for `AbstractFramework/seedvr2-7b-4bit`,
  `--max-frames 149`, `--temporal-chunk-size 13`, `--temporal-chunk-overlap 4`, and
  `--force-unsafe-video-memory` exited 2 with the expected "refuses temporal chunks smaller than
  29 frames" error before generation.
- 7B repair delta against the superseded `13/4` proof: chunk count fell from 19 to 8, restored
  temporal ratio fell from 1.5440 to 1.3370, and the new all-boundary sheet covers boundary frames
  20, 40, 60, 80, 100, 120, and 140.
- Status: Planned. This repair fixes the production temporal guardrail and invalid proof profile,
  but it still does not close the chunk-bounded-noise memory-reduction item because normal package
  execution intentionally stays on quality-preserving global noise.
