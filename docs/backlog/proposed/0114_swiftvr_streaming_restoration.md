# Proposed: SwiftVR one-step restoration as a second restore family

## Metadata

- Created: 2026-08-16
- Status: Proposed
- Completed: N/A

## Implementation status (2026-08-16)

The maintainer directed implementation on 2026-08-16, ahead of the microbenchmark this item
recommends as step 1. The port landed and is parity-verified against the torch reference. This
section supersedes the derived estimates below where they conflict; the derived reasoning is kept
because it was accurate and is worth preserving as a method.

What shipped:

- `src/mflux/models/swiftvr/` (~4,340 LOC): MFSWA attention with window index cache, RoPE temporal
  offset plus analytic table extension, ReAE autoencoder, chunk protocol, streaming DiT and ReAE,
  the restore route, and the weight definition/mapping.
- Three surgical seams keep the plain Wan route byte-identical: an optional
  `self_attention_strategy` hook on `WanAttention`, an optional `t_offset` on the Wan rotary
  embedding, and an optional precomputed `rotary_emb` on `WanTransformer.__call__`. All default to
  existing behaviour.
- A shared restoration family dispatcher at `src/mflux/models/common/cli/restore_dispatch.py`;
  `mlxgen upscale` now hosts both families and every pre-existing SeedVR2 handle resolves exactly
  as before (guarded by a regression test).
- Registry entry `swiftvr` (aliases `swiftvr`, `swiftvr-5b`) -> `H-oliday/SwiftVR`.

Numerical parity against the torch reference, same weights, fp32 unless stated:

| Component | Result |
| --- | --- |
| Full DiT block, real block-0 weights | cosine 1.00000000, max_abs 5.72e-05 on ref_absmax 43.8 |
| Full DiT block, bf16 | cosine 0.99998684 |
| MFSWA, real block-0/1 weights, both parities | cosine 1.00000000, rel_max 3.4e-06 |
| MFSWA with window >= grid vs global attention | max_abs 5.96e-08 (degenerates correctly) |
| ReAE encoder / decoder | cosine 1.00000000, rel ~1.5e-06 |
| ReAE streaming first / middle / last chunk | cosine 1.00000000 (causal state threading correct) |
| ReAE bf16 encoder / decoder | cosine 0.99985450 / 0.99998939 |

Measured throughput on this M4 Max, 128 GB, 1x restoration, end to end including model load:

| Geometry | Frames | Chunks | Wall | FPS | Peak MLX |
| --- | ---: | ---: | ---: | ---: | ---: |
| 320x240 | 149 | 7 | 7.8 s | 19.20 | 11.01 GB |
| 480x270 | 13 | 1 | 3.7 s | 3.49 | 11.17 GB |
| 1920x1080 | 13 | 1 | 18.5 s | 0.70 | 14.27 GB |
| 1920x1080 | 41 | 2 | 37.6 s | 1.09 | 14.27 GB |

**Correction to the derived band below:** this item derived 0.2-0.9 FPS at 1080p on an M4 Max. The
measured value is 1.09 FPS end to end and about 1.19 FPS excluding model load, so the derivation was
roughly 1.2-1.3x pessimistic at the top of its band. The conclusion is unchanged and now measured
rather than inferred: 1.09 FPS is about 22x short of 24 FPS, so streaming remains not viable on this
platform and the honest positioning is still offline batch restoration.

**The product thesis holds at low resolution.** Against this repo's own published SeedVR2-3B proof
geometry (149 frames at 320x240, 1x, `docs/upscaling.md:234-243`, 71.33 s / 2.09 FPS on an M5 Max),
SwiftVR runs the same geometry in 7.8 s / 19.20 FPS on an M4 Max - about 9.2x faster on slower
silicon. That is a real operating-point difference, not a marginal one.

Chunk-boundary continuity was checked on the 41-frame two-chunk 1080p run: the FIRST/LAST boundary
at frame 27->28 shows a per-frame delta of 0.85x the clip median, i.e. no seam. The largest delta in
the clip is the looped source's own content discontinuity.

Tests: 665 pass under `tests/swiftvr/` and the pre-existing suite is unchanged at 1842 passed /
7 skipped, so the Wan and SeedVR2 edits did not regress.

Scope deliberately left out of this first slice:

- **Scaling.** SwiftVR reaches non-native sizes by bilinear pre-upsampling of the degraded input.
  That is unmatched and unmeasured here, so `--resolution` other than `1x` fails closed with a
  message pointing at SeedVR2 rather than approximating upstream behaviour.
- **Quantization.** No q8/q4 prepare path; BF16 only.
- **Quality comparison.** The five-axis comparison in "Validation ideas" has NOT been run. Nothing
  in this implementation establishes that SwiftVR beats SeedVR2 on output, and the published record
  in the section below still leans the other way. No release-quality claim is made and ADR 0001
  smoke evidence has not been preserved as a proof bundle.
- **Docs.** `docs/upscaling.md` has not been updated.

This item therefore stays Proposed rather than moving to completed: the code exists and is correct,
but promotion criteria 2 (quality) and the ADR 0001 proof bundle are still open.


## ADR status

- Governing ADRs:
  - [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
  - [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
  - [ADR 0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md)
  - [ADR 0004](../../adr/0004_seedvr2_video_host_safety_and_proof_boundaries.md)
- ADR impact: none proposed. This item routes the CLI question back to ADR 0002 and ADR 0003
  rather than introducing a preset/mode axis that would need a new ADR to justify.

## Context

SwiftVR is verified, not a watchlist entry built on a search snippet. `SwiftVR: Real-Time One-Step
Generative Video Restoration`, arXiv:2606.09516v1, submitted 2026-06-08, University of Macau +
TeleAI + Nanjing University. No `journal_ref`, no DOI, no venue: treat it as a preprint.

- Code: `github.com/H-oliday/SwiftVR`, Apache-2.0, 105 stars, created 2026-05-27, last commit
  2026-06-15. Inference code is real, not promissory: 2,326 Python LOC, an 829-line transformer,
  no `TODO`/`NotImplementedError` in the inspected sources. Inference only - no training code,
  no degradation pipeline, no evaluation harness.
- Weights: `huggingface.co/H-oliday/SwiftVR`, Apache-2.0, **ungated**, 488 downloads, 27 likes.
  `transformer/diffusion_pytorch_model.safetensors` is 19,999,235,584 bytes, reconciling exactly
  as `8 + 84,728 header + 4,999,787,712 params x 4` (F32).
- Ecosystem is thin: one mirror, one unlicensed bf16 recast, one 3-star ComfyUI fork. Not in
  `diffusers`. **No MLX port exists, and no Apple Silicon measurement exists, by anyone.**

Nothing here was confused with SeedVR2, FlashVSR, RealViFormer, or STAR; those appear in the
SwiftVR paper only as cited baselines.

### Scorecard for the received proposal

| Claim | Verdict | Note |
| --- | --- | --- |
| SwiftVR exists | True | Paper, Apache-2.0 code, ungated Apache-2.0 weights |
| Backbone is Wan2.2-TI2V-5B | True, and stronger | `transformer/config.json` byte-identical to the stock Diffusers config; 825/825 tensors match name, shape and dtype |
| Dense SDPA, no vendor sparse kernel | True | `attn_mask=None, is_causal=False`; no flash-attn, xformers, triton, TensorRT or fp8 in `requirements.txt` |
| ~26 FPS at 1080p on RTX 5090 | True as a quotation | Author-reported prose, explicitly "default PyTorch SDPA path, bfloat16". No table, no VRAM figure, no warm-up disclosure |
| ~31 FPS at QHD on H100 | True | 31.32 FPS, fully tabulated |
| Meaningful component reuse | Partially true | Decomposes very unevenly; see below |

Two framing corrections, not fact corrections:

- **The 26-vs-31 pairing is not interpretable as stated.** The paper's own H100 1080p number is
  **54.42 FPS** and the proposal omits it. 54.42 -> 31.32 is a 1.74x slowdown for a 1.78x pixel
  increase (linear); 54.42 -> 26 is a 2.09x hardware gap. The two nearly cancel, producing
  illusory parity between the two quoted figures.
- **"One step" means one forward per 24-frame chunk, not per frame.** Confirmed arithmetically
  from the paper's table: latency x FPS = 23.88 frames. Output arrives in bursts with an inherent
  ~1s buffer.

### Quality versus SeedVR2 is unestablished, and the published shape is a caution

Splitting the paper's own head-to-head cells by metric family: SeedVR2-3B wins **10 of 12
full-reference** cells (on SPMCS it beats SwiftVR on LPIPS 0.2619 vs 0.2837 and DISTS 0.1410 vs
0.1535), while SwiftVR wins **14 of 16 no-reference** cells. "Looks better in isolation, matches
ground truth worse" is the signature of detail synthesis rather than recovery - precisely the
hallucination axis. The author states the mechanism in issue #14: adversarial third-stage training
exists "to produce sharper and clearer results" because flow-matching output is blurry.

Three gaps bear directly on the comparison axes: **no temporal metric of any kind** is reported,
**no face/identity metric**, and **no text/OCR metric**. The one confirmed independent quality
report is issue #9, still open - texture flickering, which the author attributes to the lightweight
autoencoder that is the paper's core speed contribution: "the autoencoder's reconstruction
capability sets the upper bound of the model's output quality."

SwiftVR's SeedVR2 baseline is also handicapped and must not be reused. The paper enables
`use_tile=True` for SeedVR2 on memory grounds, and its VideoLQ MUSIQ for SeedVR2 (40.42) is
10.7 points below SeedVR2's self-reported 51.09.

## Current code reality

SwiftVR is absent: `grep -rniIl "swiftvr" . --exclude-dir=.git` returns 0 files.

What the existing Wan2.2-TI2V-5B stack genuinely provides:

- The transformer is already config-driven. `wan_transformer.py:37-57` takes every dimension as a
  keyword argument, and the `wan2.2-ti2v-5b` overrides at `model_config.py:831` already carry
  `in_channels 48`, `out_channels 48`, `num_layers 30`, `num_attention_heads 24`,
  `attention_head_dim 128`, `ffn_dim 14336`, `patch_size [1,2,2]` - the exact values SwiftVR
  ships. Its remaining config fields (`text_dim 4096`, `freq_dim 256`, `rope_max_seq_len 1024`)
  match our constructor defaults at `wan_transformer.py:47-54`.
- The override path is a closed 15-key allow-list at `wan_initializer.py:746-767`. Unknown keys
  are **silently dropped** - a trap for any new family.
- **Causal streaming VAE decode already exists** and is the only Wan decode path:
  `wan_2_2_vae.py:399-439` is a generator carrying a 64-slot conv feature cache across temporal
  slices. This is genuine left-context, not chunk re-decoding.
- Quantize/save is family-agnostic via `ModelSaver` with per-family `quantization_predicate`
  hooks.

What the CLI looks like today:

- `mlxgen upscale` is not routed. `mlx_gen.py:482-484` dispatches to `_upscale_image`, which at
  `mlx_gen.py:494-501` is an argv shim calling `seedvr2_upscale.main()`. Model selection is a
  hand-written string ladder in `_resolve_seedvr2_model` (`seedvr2_upscale.py:114-183`) that
  raises `ValueError` for any unrecognized repo id. A second family needs a dispatch resolver.
- The subcommand help hardcodes the family (`mlx_gen.py:275`).
- **Restoration has no capability contract.** `grep -ci seedvr src/mflux/task_inference.py`
  returns 0, and `--family` choices at `mlx_gen.py:513-517` exclude restoration entirely.
- **`restore_mode` already exists and already equals `"streaming"`** for every video restore
  (`seedvr2_upscale.py:62,511,529,720`).

## Problem or opportunity

The genuine opportunity is narrow and real: **offline batch restoration throughput.** Measured on
an M5 Max for the accepted `29/8` proofs, SeedVR2-3B restores the 149-frame 320x240 source in
71.33s (2.09 FPS) at 1x and 539.03s (0.276 FPS) at 2x (`docs/upscaling.md:234-243`). Restoration is
the slowest useful thing this package does.

The three named product distinctions do not survive contact with this repository:

- **Fast preview.** There is no preview surface on the upscale route at all;
  `--stepwise-image-output-dir` is on the generation parser only (`parsers.py:336-350`, absent from
  the SeedVR2 parser at `parsers.py:154`). Restoration is one-step, so there is no trajectory to
  preview. Proposed [0113](0113_video_family_tiny_previews.md) records "seedvr2 | 16 | none
  published". A cheaper tier already ships as `seedvr2-3b-4bit`.
- **Live.** No server, daemon, or socket surface. The only stream is JSONL scalar progress; no
  pixels cross it. The contract forbids partial reads - on any exception the partial file is
  deleted (`seedvr2.py:608-614`) - and an exclusive `flock` permits one video job per machine
  (`seedvr2_upscale.py:75-100`).
- **Long video.** Framed and closed as a *memory* problem. ADR 0004 accepts verbatim that
  "rebuilding the model per seed makes multi-seed video work slower"; throughput was deliberately
  traded for stability. `_safe_chunk_frame_limit` (`seedvr2_upscale.py:424-455`) solves purely for
  memory and contains no time budget.

"Complement, not replace" therefore holds only in its weakest and most useful form: a faster
offline restoration route at the same or lower memory.

### What is genuinely new, and what the brief's premise gets wrong

The received framing implies three new subsystems. Two do not exist and the third is half-built
here, which makes the port materially easier than assumed:

- **Causal KV-cache: not required.** The README states "no rolling KV cache, no overlapped DiT
  inference", and every attention call is non-causal. Causality is chunk-level only, via a RoPE
  temporal offset, conv boundary buffers, and an optional crossfade defaulting to 0.
- **One-step distilled scheduler: not required.** There is no scheduler object. It is a flow
  reformulation, not a step distillation: constant timestep 1000.0 and one subtraction
  (`z_HQ = z_LQ - v(z_LQ, 1)`).
- **Streaming decode: partly built.** Our `iter_decode_slices` feature cache is the same shape of
  solution as SwiftVR's boundary-state threading.

The real new work:

- **The autoencoder is 100% new and is where the risk lives.** SwiftVR replaces the 704.69M-param
  Wan VAE with a 40.95M-param Restoration-aware Autoencoder (ReAE, `reae.safetensors`,
  163,797,568 bytes). Zero reuse. Architecturally simple, but it is a second trained model and per
  issue #9 it is the component that caps output quality.
- **Mask-free shifted-window attention (MFSWA).** All 30 blocks swap global 3D attention for a
  gather/scatter into fixed 16x16 windows, even layers unshifted and odd half-shifted. MLX has the
  primitives, but Metal performance on the large gather tensors is unproven.
- **RoPE temporal offset and table extension** beyond `rope_max_seq_len=1024`, plus a weight
  mapping (SeedVR2's is 692 lines, Wan's 421 - tedious rather than hard).

Reuse decomposes as: DiT module ~100% (825/825 tensors match stock Wan; sampled cosine similarity
0.9955-0.99999 against the public checkpoint, so SwiftVR is a fine-tune, not a retrain); UMT5
droppable entirely (a 4.2 MB constant embedding replaces it, though cross-attention still
executes); scheduler droppable; VAE zero reuse.

Memory is not the constraint. bf16 is 9.39 GiB and a q8 policy projects to ~5.1 GiB - **smaller
than our existing Wan TI2V-5B route at 10.6 GiB** (`docs/quantization.md:293`), because ReAE
replaces the 705M-param VAE and no text encoder loads.

## Apple Silicon feasibility

**All figures here are derived, not measured. No SwiftVR run on Apple Silicon exists. Uncertainty
is at least +/-2x.**

The memory-bandwidth ratio is the wrong bound and badly overstates feasibility. Reading all
10.08 GB of bf16 weights once per forward costs ~18.5 ms at the M4 Max's 546 GB/s, implying a
~1,300 FPS ceiling from bandwidth alone. Arithmetic intensity at 1080p is ~11,700 FLOP per byte
against a crossover near 4: this workload is compute-bound by about three orders of magnitude.
Quoting the 3.28x (RTX 5090) or 6.14x (H100) bandwidth ratio would overstate feasibility by more
than 10x.

Two independent derivations converge within 1.7x:

1. **FLOP model**, validated against the paper's H100 table (recovers 359.9/367.5/371.4 TFLOP/s
   across a 4x resolution span, 3% spread), then back-solved against four measured mlx-gen Wan
   runs (`docs/quantization.md:293-294`, `docs/wan-video.md:519-521`) for a 4.3-4.7 TFLOP/s
   cluster: ~1.6 FPS at 1080p on M5 Max, DiT only.
2. **Measured-ratio method**, no FLOP model: applying the paper's own 22.5x SwiftVR/SeedVR2-3B
   ratio to our measured 0.276 FPS at 640x480 and scaling by area: ~0.9 FPS on M5 Max.

The benchmark machine is an M5 Max, which has per-core Neural Accelerators the M4 Max lacks.
Discounting accordingly, the derived band at 1080p on an M4 Max is roughly **0.2-0.9 FPS** - 1 to
5 seconds of compute per output frame, 27x to 120x short of 24 FPS.

Streaming stops being meaningful well before 1080p: 24 FPS solves to roughly 160x90 output, 10 FPS
is unreachable even at 360p, and 1 FPS arrives around 480p. Stacking a generous 2-4x MLX win with a
2x quantization win still leaves 1-2 FPS at 1080p.

**A `restore-stream` route is not justified on Apple Silicon.** The honest positioning is offline
batch restoration with progress reporting, and upstream's "real-time" and "streaming" language must
not be inherited into our docs.

## Proposed direction

Keep this proposed. Do not start a port from this item.

1. **Verify before porting**, with a one-day microbenchmark rather than an implementation: MLX
   bf16 GEMM at `[12240,3072] x [3072,14336]`, and MFSWA gather/scatter at 1080p window geometry.
   Those two numbers confirm or refute the effective-throughput assumption everything rests on.
2. **If it holds, scope as offline restoration only.** First slice: video restore at a bounded
   resolution, no streaming, live, or preview claim.
3. **Reject the three-mode taxonomy.** Route by exact model handle as every other route does:
   `mlxgen upscale --model H-oliday/SwiftVR --video-path clip.mp4`. Reasons, in order of weight:
   - ADR 0002. A mode name is a deferred default-model decision. `--model seedvr2-7b` is
     reproducible; `--mode restore-quality` silently changes model identity across versions.
   - ADR 0003 assigns recommended presets and user-facing defaults to AbstractVision and states
     MLX-Gen must not become a recommendation engine. A quality/latency ladder is exactly that.
   - Completed [0020](../completed/0020_generation_capability_contract.md) removed `edit` from the
     public task axis for leaking a backend detail; this would add a third unsanctioned axis.
   - It cannot express what already ships: three SeedVR2 aliases against two tiers leaves
     `seedvr2-7b-sharp` unaddressable.
   - `restore-balanced` names the existing no-flag default, carrying zero information.
   - Direct collision: `restore_mode` already exists and already equals `"streaming"`, so a
     `restore-quality` run would emit `restore_mode=streaming` in its own metadata.
   - The implied ladder is contradicted by our own proof: 7B 2x is *faster* than 3B 2x (454.46s vs
     539.03s) and 3B is crisper on the archival slice. ADR 0004 forbids ranking 3B against 7B
     without direct visual proof.
4. **Close the real CLI gap instead, independently of SwiftVR.** Restoration has no capability
   surface. Adding a restoration capability descriptor (accepted inputs, scale support,
   `requires_low_ram`, safe chunk-frame limit, 4n+1 geometry, per-model rows) is the ADR 0003
   division of labour: MLX-Gen emits truth, AbstractVision names presets on top. Worth doing
   whether or not SwiftVR is ever ported.
5. **Generalize `_resolve_seedvr2_model` into a family-dispatch resolver** only when a second
   restoration family actually lands, and update the hardcoded help at `mlx_gen.py:275`.

## Promotion criteria

1. A microbenchmark on the target machine confirms effective MLX throughput within the derived
   band. If it lands at the low end, close this item rather than porting.
2. A quality read establishes SwiftVR is worth porting on output, not throughput alone. The
   published record currently shows SeedVR2-3B winning 10 of 12 full-reference cells.
3. The ReAE port is scoped honestly as a second trained model, with its known texture-flicker
   limitation accepted or mitigated up front.
4. ADR 0001 is satisfied with a model-backed runtime smoke on the real checkpoint and preserved
   artifacts. ADR 0005 applies to any video quality claim.
5. The item still outranks current Wan, Qwen, FLUX.2, and Z-Image work. Proposed
   [0009](0009_video_second_family_selection.md) holds the standing posture, and
   `docs/backlog/overview.md` adds that the next useful work is UX and contract cleanup, not
   another speculative route split.

## Validation ideas

Any SeedVR2 comparison must be generated here under SeedVR2's own protocol, never lifted from the
paper's tiled row.

- **Hallucination.** The decisive test. Restore a clip with known ground truth and report
  full-reference (LPIPS, DISTS) *and* no-reference (MUSIQ, MANIQA) side by side. Improving
  no-reference while degrading full-reference is synthesis, not recovery. Include a
  lightly-degraded control: SeedVR2's own paper admits it over-generates detail on light
  degradation.
- **Temporal stability.** A warping-error-style metric plus a flicker read on high-detail texture
  (foliage, fabric, gravel, static camera) - the failure mode independently reported in issue #9.
  Neither paper helps: SwiftVR reports no temporal metric at all.
- **Faces.** A complete gap in both papers. Build a panel of real faces at several degradation
  levels; report identity similarity plus reader-first crops.
- **Text and glyphs.** Also a complete gap. Use signage or subtitles and report OCR agreement
  against ground truth. Text is where detail synthesis fails most legibly.
- **Compression damage.** Stratify by codec and QP rather than aggregating. All three of SwiftVR's
  synthetic benchmarks are in-distribution for the RealBasicVSR degradation pipeline; real
  H.264/HEVC damage at low bitrate is covered by no published number.

Two checks worth running regardless: a DiT parity check exploiting the 825/825 tensor identity
(loading SwiftVR weights into our `WanTransformer` should work with override-value changes alone,
localizing the MFSWA delta), and a negative control confirming the 15-key allow-list at
`wan_initializer.py:748-763` does not silently drop a SwiftVR-specific override.

## Non-goals

- Do not describe SwiftVR as real-time or streaming on Apple Silicon.
- Do not add `--mode`, `restore-quality`, `restore-balanced`, or `restore-stream`.
- Do not treat SwiftVR as a SeedVR2 replacement; SeedVR2 wins most full-reference cells in
  SwiftVR's own paper.
- Do not cite SwiftVR's SeedVR2 row as a baseline; it runs tiled and does not reproduce SeedVR2's
  published numbers.
- Do not start with 4K: the paper's own limitation is 13.84 FPS at 60.91 GB on an H100.
- Do not attempt to reproduce the training. No training code or evaluation harness is released.

## Implementation notes

The temptation is that the release quality is unusually good - Apache-2.0 code, Apache-2.0 ungated
weights, a backbone we already implement, and a config byte-identical to stock Wan2.2-TI2V-5B. All
true, and none of it is the deciding factor.

The deciding factor is that the headline claim is a throughput claim measured on silicon roughly
two orders of magnitude faster than ours on this workload, and streaming is the one product
distinction that cannot survive the transfer. Check the arithmetic before the architecture.

SwiftVR is easier to port than the received framing suggests - no KV cache and no distilled
scheduler to build - but ReAE is a complete second model with an author-acknowledged quality
ceiling. If ReAE caps output quality and must also be written from scratch, that is where the risk
lives, not in the DiT.

If revisited, re-check whether anyone has independently reproduced any SwiftVR throughput number.
As of 2026-08-16 nobody has, the repo has been static for two months, and the only public attempt
(issue #16, 2.31 FPS, hardware unstated) is unanswered.

## Sources checked

- https://arxiv.org/abs/2606.09516 and https://arxiv.org/html/2606.09516v1
- https://export.arxiv.org/api/query?id_list=2606.09516 (no `journal_ref`, no DOI, no `comments`)
- https://h-oliday.github.io/SwiftVR/
- https://github.com/H-oliday/SwiftVR and https://api.github.com/repos/H-oliday/SwiftVR
- `swiftvr/models/transformer.py`, `swiftvr/models/reae.py`, `swiftvr/pipeline.py`,
  `swiftvr/streaming/{dit,tae,chunk}.py`, `README.md`, `requirements.txt` (raw.githubusercontent)
- https://huggingface.co/H-oliday/SwiftVR and
  https://huggingface.co/api/models/H-oliday/SwiftVR?blobs=true
- https://huggingface.co/H-oliday/SwiftVR/raw/main/transformer/config.json
- https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/raw/main/transformer/config.json
  (byte-identical to the SwiftVR config)
- SwiftVR issues 9 (texture flickering), 14 (flow-matching blur), 16 (2.31 FPS, unanswered),
  2 (`torch_compile` yields no speedup)
- https://github.com/Comfy-Org/ComfyUI/issues/14469 (official support request, open, no PRs)
- https://huggingface.co/api/models?search=SwiftVR (3 repos: official, one mirror, one unlicensed
  bf16 recast)
- SeedVR2 baseline: https://arxiv.org/abs/2506.05301 and https://arxiv.org/html/2506.05301v2
- https://support.apple.com/en-us/121553 (M4 Max, 546 GB/s)
- https://machinelearning.apple.com/research/exploring-llms-mlx-m5 ("up to 4x" M5-over-M4 TTFT)

Non-yielding: `arxiv.org/abs/2606.09516v2` (404, v1 is the only version);
`iceclear.github.io/projects/seedvr2/` (no throughput figures published);
`huggingface.co/api/models?search=swiftvr+mlx` (0 results);
`api.github.com/repos/huggingface/diffusers/contents/src/diffusers/pipelines/swiftvr` (404, used as
negative evidence); the NVIDIA RTX 5090 product page prints no GB/s figure, so the 1,792 GB/s value
is secondary-sourced and unverified.
