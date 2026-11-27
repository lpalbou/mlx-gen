# Completed: MiniMax-H3 staged loading, block streaming, re-indexed 8-bit weights and the 16-bit package

## Metadata

- Created: 2026-11-23
- Status: Completed
- Completed: 2026-11-23

## ADR status

- Governing ADRs:
  [0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md) (model-backed proof before a
  route is described as working),
  [0002](../../adr/0002_no_silent_automatic_fallbacks.md) (a memory option that cannot apply is
  reported; `--low-ram` does not silently become a dependency on disk speed).
- ADR impact: none.

## Request

Requested on 2026-11-23: a hybrid of a 16-bit conditioner with the 8-bit transformer, and an
assessment of every route that lowers MiniMax-H3's memory at 8-bit or better quality.
Added the same day: publish any better variant that still runs on 128 GB, document the work in a
dedicated page, and replace the published 8-bit package if the rebuilt one is an improvement.

## What shipped (release 0.40.0)

- **Conditioner read for the length of an encode** (`H3Conditioner`): the 50 Qwen3-VL layers are
  bound to pending file reads and released one by one as they run. Encode peak 3.2 GiB (4.6 GiB
  with a keyframe) at 16-bit, against 48.3 GiB held whole; output bit-identical.
- **AdaLN projections as per-schedule tables** (`H3ModulationTables`): 13.0B of the transformer's
  33.1B parameters take the timestep embedding alone, so their output for the whole schedule is
  computed in one pass from disk. Teacher-forced error against the resident projections: exactly 0.
- **VAEs read at first use**, and `--low-ram` frees the transformer before the decode.
- **`--stream-blocks`** (`H3BlockStream`): each transformer block's tensors are replaced by new
  pending reads after its call. Bit-identical output; 1.5 to 4 s per step at `960x544`.
- **8-bit levels chosen on the stored grid** (`StreamingWeightLoader.quantize`).
- **`mlxgen prepare` writes one component at a time** (`ModelSaver.save_components`).
- **A guard on weight files** that change under a loaded model (`require_unchanged`).
- **Packages:** `AbstractFramework/minimax-h3-bf16` (new), `AbstractFramework/minimax-h3-8bit`
  (rebuilt with the new level choice), cards of all three regenerated.
- **Capabilities schema 17**, the dedicated page `docs/minimax-h3-memory.md`, and the probes
  `tools/minimax_h3_memory_probe.py` and `tools/minimax_h3_quant_probe.py`.

## Evidence

Peak process footprint, `960x544`, 124 frames, 8 steps, Apple M5 Max 128 GB:

| Weights | 0.39.2 | Default | `--low-ram` | `--stream-blocks --low-ram` |
| --- | ---: | ---: | ---: | ---: |
| 8-bit package | 77.0 GiB (65.0 low-RAM) | 37.0 GiB | 26.3 GiB | 11.1 GiB |
| Mixed 4-bit | 57.6 GiB (45.6 low-RAM) | 32.5 GiB | 21.9 GiB | not measured |
| 16-bit | refused (115 GiB of weights) | 54.4 GiB | 44.0 GiB | 11.7 GiB |

- Clips from the 8-bit package and from `--quantize 4` are byte-identical to the 0.39 clips at
  `640x352` and `960x544`; `--low-ram` and `--stream-blocks` clips are byte-identical to the
  default ones; the rebuilt 8-bit package's clip equals load-time `--quantize 8`.
- Locked-RAM runs completed at 96 / 64 / 32 GB (16-bit), 48 / 36 / 24 GB (8-bit) and 32 GB (mixed
  4-bit); see the dedicated page for modes, peaks and times.
- Reports: `docs/assets/quantization/minimax-h3/reference_report.json`.

## What the measurements rejected

- **The 16-bit conditioner hybrid.** Against the 16-bit reference, the 8-bit transformer with
  16-bit conditioning scores the same video error as the 8-bit package (9.0 / 8.8 / 3.9% against
  9.0 / 8.7 / 4.1% on the dialogue clip) and is no closer at clip level. Audio improves slightly.
  With the conditioner streamed the question is moot for memory.
- **AdaLN precision as a quality lever.** The 0.2 to 2.1 points seen in item 0128's table are
  inside the metric's noise.
- **The velocity metric as a fine ruler.** A 0.1% perturbation of the 16-bit conditioning moves the
  16-bit prediction by 3 to 7% (video). The 8-bit package sits at 1.2 to 2.0 times that floor.
- **Re-indexing as a visible gain.** It lowers the weight error of a layer from 0.78% to 0.59%, the
  conditioner's median token error by about 9%, and the transformer's video error in isolation in 9
  of 9 inputs; for the complete package the mean video error is 6.23% against 6.30%, and the fox
  clip sits at 28.2 dB against 28.4 dB. It ships because it is never further from the released
  weights at the same size and keeps the package equal to load-time quantization. The replacement
  was approved on those terms.

## Verification

Three checks: the byte budget and the hypotheses; the measurement method and the
prototype; the engine diff. Changes applied: tables keyed through the model's own row-timestep
function, a noise-floor baseline, `--stream-blocks` as its own option instead of a meaning of
`--low-ram`, the file-change guard, the table cache surviving a transformer reload, one public
loader surface instead of private cross-class calls, and the tests listed in
`tests/minimax_h3/test_h3_staged_loading.py`. Items recorded as follow-ups:
[0131](../proposed/0131_minimax_h3_memory_followups.md).

## Follow-ups

[0131](../proposed/0131_minimax_h3_memory_followups.md): several blocks resident while streaming
(suggested, not required), re-indexing the other families' 8-bit packages, the request
preflight, working memory, source-snapshot quantization per encode, and the host application.
