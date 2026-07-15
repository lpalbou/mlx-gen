# Masked Edit 5x5 Matrix Validation (2026-07-15)

Standardized multi-case visual QA for the masked-edit routes shipped in 0.20.0/0.21.0. One
shared source (`tests/resources/glasses.jpg`, 717x403), five masks, seed 42, run through
unified `mlxgen generate --image ... --mask-path ...` on five original model rows plus the
same-day `Qwen/Qwen-Image` source-row addendum (six rows total).

Contact sheet: `masked_edit_matrix_contact_sheet.png`. Zoom sheets per case:
`zoom_recolor.png`, `zoom_arm.png`, `zoom_sticker.png`, `zoom_remove.png`. Per-run outputs,
masks, and `preservation_metrics.json` (outside/inside-mask mean abs pixel diff per cell) are
included. Registry profile: `masked_edit_matrix_5x5_2026_07_15` (`mlxgen validation
--model <row> --profile masked_edit_matrix_5x5_2026_07_15`).

## Scored results (four well-posed cases)

| Row | Route | Settings | insert | recolor | arm retexture | sticker removal | Aggregate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `AbstractFramework/flux.2-klein-4b-8bit` | `flux2.inpaint` | 4 steps, g1 | PASS | PASS | PASS | PASS | PASS |
| `AbstractFramework/flux.2-klein-base-4b-8bit` | `flux2.inpaint` | 20 steps, g4 | PASS | PASS | PASS | PASS | PASS |
| `Qwen/Qwen-Image` (source bf16, added post-matrix same day) | `qwen.base-inpaint` | 20 steps, g4 | PASS | PARTIAL | PASS | PASS | PARTIAL |
| `AbstractFramework/qwen-image-4bit` | `qwen.base-inpaint` | 20 steps, g4 | PASS | PARTIAL | PASS | PASS | PARTIAL |
| `AbstractFramework/qwen-image-2512-8bit` | `qwen.base-inpaint` | 20 steps, g4 | PASS | PARTIAL | PASS | PASS | PARTIAL |
| `AbstractFramework/z-image-8bit` | `z-image.inpaint` (withdrawn, see addendum) | 30 steps, g4 explicit | PASS | PARTIAL | FAIL | PASS | FAIL |

Review notes:

- Qwen `recolor` PARTIAL: the native route's 0.85 warm start anchors the masked region to the
  clear source lens, so full opaque recolors stay incomplete (navy wedge / pale wash). Object
  insertion, retexture, and removal are unaffected.
- Z-Image `arm` FAIL: the repainted arm detaches from the hinge with a floating black segment
  and a hanging flap. Reproduced with seed 43 (`zimage8_arm_seed43.png`), so the failure is
  systematic on this case rather than seed luck.
- Outside-mask preservation across all 30 scored-and-demo runs (25 original + 5 source-row):
  0.31-2.54/255 mean abs diff (16 px boundary band excluded); FLUX.2 rows sit near 1.4-2.5
  because generation runs at 720x400 versus the 717x403 source, Qwen/Z-Image rows at
  0.31-0.56.

## Unscored limitation demonstration: partial-object removal

The fifth mask (`mask_remove.png`) covers only the END of the folded temple arm - an object
that continues outside the mask - with the caption "Tortoiseshell reading glasses on a clean
plain white background." Every row regenerates a plausible arm ending instead of removing it
(Klein rows reconstruct it almost exactly; Qwen/Z-Image rows replace it with different tips).
This is included as a documented mask-design lesson, not a scored case: caption prompting
cannot express "remove" for an object the mask only partially covers. The sticker case shows
the working pattern - the removable object sits fully inside the mask with a margin, and every
row passes.

## Post-matrix probe addendum and shipped consequences (same day)

Targeted probes examined whether the PARTIAL/FAIL cells respond to settings
(`probe_zoom_recolor.png`, `probe_zoom_arm.png`, `probe2_recolor_mechanisms.png`,
`probe3_negatives.png`, `s095_regression_sheet.png`):

- **Qwen recolor: mechanism found and shipped.** Candidate mechanisms compared at the same
  seed/settings on the 2512-q8 recolor cell: a content-specific negative prompt works but is
  not generalizable; generic/failure-mode negative prompts (guidance 4 and 7) do not; more
  steps change nothing; warm-start strength `0.90` stays washed out; strength `0.95` produces
  a complete lens recolor on both qwen rows (deep navy on 2512-q8; complete coverage in a
  paler blue-gray on q4). The full 4-case regression at `0.95` (`s095_regression_sheet.png`)
  shows insertion and sticker removal stay clean while the weaker anchor makes the arm cell
  detach on BOTH rows — confirming a real trade-off rather than a free win, and the reason
  the default stays `0.85`. Shipped consequence: `--mask-strength` exposes the upstream
  inpaint strength on `qwen.base-inpaint` (default `0.85` for structure-anchored edits,
  `0.95` measured for content-replacing edits inside well-contained masks).
- **The Z-Image arm failure is intrinsic to the non-turbo rung, not a CFG setting.** CFG-off
  produces a worse balloon artifact than guidance 4; Z-Image Turbo at 9 steps renders a
  mostly clean black arm on the same case; upstream demonstrates `ZImageInpaintPipeline` on
  Turbo only. Shipped consequence: non-turbo Z-Image masked editing is withdrawn from the
  public surface for the moment — `z-image.inpaint` is Turbo-only again and non-turbo masked
  requests fail before model load. The FAIL records here stay as the withdrawal evidence.

Matrix statuses are unchanged: they grade the standardized default settings.

## Source-row addendum (same day)

The `Qwen/Qwen-Image` source bf16 checkpoint ran the four scored cases through the identical
commands (`qwensrc_*.png`, row sheet `qwensrc_row_sheet.png`), closing the "wiring-shared, no
dedicated run" gap: `PASS` on insert/arm/sticker with outside-mask preservation 0.31-0.46/255,
`PARTIAL` on recolor with the same 0.85 anchoring signature as the prepared rows. A
`--mask-strength 0.95` cross-check (`qwensrc_recolor_s095.png`, seed 42) repaints the full
region but places the navy on the frame ring with a pale lens interior. A follow-up seed
sweep (`s095_seed_sweep_sheet.png`: source seeds 42-45, q4 seeds 42-44, same settings)
resolved that miss as per-seed variance rather than a precision difference: source seeds 44
and 45 produce clean complete navy lenses, seed 43 recolors with a blend artifact, while the
q4 row shows the same spread (pale at 42, small edge artifact at 43, clean at 44). The 0.95
guidance therefore holds uniformly across all three rows, with per-seed color fidelity on
every row: retry with another seed if the color lands wrong. Denoise wall times on the bf16
checkpoint: 60-142 s per case.

## Commands

```sh
mlxgen generate \
  --model <row> \
  --image tests/resources/glasses.jpg \
  --mask-path <mask>.png \
  --prompt "<case prompt>" \
  --steps <row steps> --guidance <row guidance> --seed 42 \
  --output <row>_<case>.png
```

Exact prompts per case are recorded in the validation registry
(`src/mflux/release/validation_registry.py`, `_MASKED_MATRIX_PROMPTS`) and in each output's
metadata sidecar (working bundle: `validation_outputs/masked-edit-matrix-2026-07-15/`).
Machine: M5 Max. Denoise wall times: Klein 4B 6-7 s, Klein base 4B 65-84 s, Qwen q4 48-51 s,
Qwen 2512 q8 60-68 s, Z-Image q8 62-70 s per case.
