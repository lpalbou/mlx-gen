# Bernini-R 1.3B

MLX-Gen supports ByteDance's Bernini-R 1.3B renderer for reference-guided video generation and
editing on Apple Silicon. The integration is deliberately renderer-only: it exposes the useful
reference and source-video paths without bundling the optional Qwen2.5-VL semantic planner.

> **Experimental.** Eight of the ten upstream public example rows are accepted (`t2i`, `i2i`,
> `t2v`, `r2v`, `rv2v_case1`, `v2v_case1`, `mv2v`, and `ads2v` at mid profile), each with
> committed contact-sheet proof. The two remaining rows reflect 1.3B model limits at their
> upstream default settings; both improve substantially with the tuned settings described in
> [Task-Specific Recipes](#task-specific-recipes). Treat this route as experimental rather than
> production-ready.

Upstream public Bernini examples are broader than the currently proven mlx-gen surface. The strict
official-example target is tracked separately in the
[official public 1.3B example parity matrix](assets/validation/bernini-r-1.3b-2026-08-04/official_example_parity_matrix.md).

Row-by-row status, caveats, and proof sheets live in the
[Official Public 1.3B Examples](#official-public-13b-examples) section below; the
[official public 1.3B example parity matrix](assets/validation/bernini-r-1.3b-2026-08-04/official_example_parity_matrix.md)
carries the detailed evidence record, including the local official-implementation comparison
methodology used to validate parity.

Use Bernini when ordinary images should act as reusable visual references, or when a source video
should be edited with those references. Continue using Wan VACE for learned mask/control workflows
and Wan A14B for general text/image-to-video generation.

## Supported Workflows

| Inputs | MLX-Gen mode | What it does |
| --- | --- | --- |
| 1-8 `--reference-image` values | R2V (`reference-video`) | Generates a new video from ordered ordinary reference images and a prompt. |
| One `--video` plus 1-8 references | RV2V (`reference-video-edit`) | Edits the source clip while using the images as appearance/content references. |
| One `--video`, no references | V2V (`latent-video`) | Performs prompt-guided source-video editing or motion change. |

The renderer does not currently expose Bernini's planner, T2I/I2I convenience routes, LoRA,
video masks, warm-start strength, first/last frame anchors, multi-frame continuation, or multiple
source videos. Motion-changing prompts can be used on V2V, but MLX-Gen does not claim the full
planner-driven MV2V workflow.

Internally, the Bernini runtime also accepts the official upstream `task_type` values `t2i`,
`i2i`, `t2v`, `r2v`, `rv2v`, `v2v`, `mv2v`, and `ads2v` so the public cases can be replayed
faithfully. Those typed rows are part of the parity harness; the stable user-facing guide here
remains focused on the documented video workflows above.

## Official Public 1.3B Examples

Committed proof for every upstream public example row lives in the
[official public parity bundle](assets/validation/bernini-r-1.3b-2026-08-11/README.md). The bundle
includes a [summary contact sheet](assets/validation/bernini-r-1.3b-2026-08-11/official_public_summary_contact_sheet_preview.png)
built from the pinned accepted-case validation runs (full profile, August 2026 official-public pass).
Each case directory contains the **full upstream prompt**, GitHub-friendly preview contact sheets plus
full-resolution input/official/mlx sheets, reproduce command, generated artifact, and metadata when
available. Row status follows the
[official example parity matrix](assets/validation/bernini-r-1.3b-2026-08-04/official_example_parity_matrix.md).

| Row | Upstream case | Status | Proof (prompt + sheets) |
| --- | --- | --- | --- |
| `t2i` | `assets/testcases/t2i/t2i.json` | Accepted | [t2i](assets/validation/bernini-r-1.3b-2026-08-11/t2i/README.md) |
| `i2i` | `assets/testcases/i2i/i2i.json` | Accepted (minor wall-texture caveat) | [i2i](assets/validation/bernini-r-1.3b-2026-08-11/i2i/README.md) |
| `t2v` | `assets/testcases/t2v/t2v.json` | Accepted | [t2v](assets/validation/bernini-r-1.3b-2026-08-11/t2v/README.md) |
| `r2v` | `assets/testcases/r2v/r2v.json` | Accepted | [r2v](assets/validation/bernini-r-1.3b-2026-08-11/r2v/README.md) |
| `rv2v_case1` | `assets/testcases/rv2v/rv2v_case1.json` | Accepted (shirt-closure caveat) | [rv2v_case1](assets/validation/bernini-r-1.3b-2026-08-11/rv2v_case1/README.md) |
| `v2v_case1` | `assets/testcases/v2v/v2v_case1.json` | Accepted | [v2v_case1](assets/validation/bernini-r-1.3b-2026-08-11/v2v_case1/README.md) |
| `mv2v` | `assets/testcases/v2v/v2v_case2.json` | Accepted (three named caveats) | [mv2v](assets/validation/bernini-r-1.3b-2026-08-11/mv2v/README.md) |
| `ads2v` | `assets/testcases/rv2v/rv2v_case2.json` | Accepted at mid profile | [ads2v](assets/validation/bernini-r-1.3b-2026-08-11/ads2v/README.md) |

Rows with tuned recipes (upstream default settings under-deliver at this model size; the
[Task-Specific Recipes](#task-specific-recipes) recover them substantially):

| Row | Recipe | Status | Proof (prompt + sheets) |
| --- | --- | --- | --- |
| `r2v_case2` | Official (`r2v_case2.json`) | Partial at official recipe | [r2v_case2_official](assets/validation/bernini-r-1.3b-2026-08-11/r2v_case2_official/README.md) |
| `r2v_case2` | Tuned (`--reference-guidance 6.0`, seed 43) | Substantial recovery | [r2v_case2_tuned](assets/validation/bernini-r-1.3b-2026-08-11/r2v_case2_tuned/README.md) |
| `v2v_case3` | Official (`v2v_case3.json`) | Official recipe fails at 1.3B | [v2v_case3_official](assets/validation/bernini-r-1.3b-2026-08-11/v2v_case3_official/README.md) |
| `v2v_case3` | Recovery (`mv2v` task prefix, guidance 5.0) | Quadruped dog outcome | [v2v_case3_mv2vprefix](assets/validation/bernini-r-1.3b-2026-08-11/v2v_case3_mv2vprefix/README.md) |

The `r2v_case2` upstream prompt is 571+ UMT5 tokens and truncates to 512 on both engines. The
`ads2v` accepted proof uses the mid profile (`480` condition cap, `61` frames, `24` fps) because
the official `1280x672/121f` recipe is computationally intractable on the validation host for
either implementation.

## Download And Capacity

Two pinned packages are available. The BF16 package is the recommended download — it ships the
exact runtime dtypes and is 42% smaller:

```sh
mlxgen download --model bernini-r-1.3b-bf16
```

This alias selects the pinned `AbstractFramework/bernini-r-1.3B-diffusers-bf16` revision
(approximately 15.6 GiB cold). It stores the dtypes the runtime uses — BF16 text encoder and
transformer with their FP32 keep-sets preserved, FP32 VAE — and produces bit-identical output to
the FP32 source package at the same settings and seed (verified on image and video use cases).

The FP32 source package remains available:

```sh
mlxgen download --model bernini-r-1.3b
```

This alias selects only the pinned `ByteDance/Bernini-R-1.3B-Diffusers` revision (approximately
27 GiB cold, since the upstream checkpoint ships in FP32). The main `ByteDance/Bernini-R` A14B
repository and similarly named third-party repositories intentionally fail closed; MLX-Gen never
rewrites them to the 1.3B renderer.

Each alias is its own model id: generate with the alias you downloaded (`--model
bernini-r-1.3b-bf16` after the BF16 download). Both packages preflight with 2 GiB of free-space
headroom (17.60 GiB free for the 15.60 GiB BF16 package, 28.94 GiB free for the 26.94 GiB FP32
source), skip sources already present at their pinned revisions, and execute identically at run
time: transformer in BF16 (with small FP32 keep-sets), the UMT5 text encoder in FP32 read one
block at a time, VAE in FP32, so resident memory is far below the download size.

On the 128 GB validation host, the bounded low-RAM profiles peaked at 9.45 GB whole-process
physical footprint. A separate 33-frame, eight-reference 848-condition structural probe peaked at
8.17 GB, so those tested shapes fit inside an 18 GB memory envelope. This is not a direct 18 GB
host measurement. Capacity is separate from the official-example acceptance status above.

## Generate From References

The following commands are diagnostic bounded-profile examples. For official full-profile prompts
and settings, use the [official public parity bundle](assets/validation/bernini-r-1.3b-2026-08-11/README.md).

Reference order is semantic. The prompt names the first image `image0`, the next `image1`, and so
on. Keep each reference focused on the subject or property it is meant to contribute.

```sh
mlxgen generate \
  --model bernini-r-1.3b \
  --reference-image subject.png \
  --reference-image garment.png \
  --prompt "Place the subject from image0 in a studio, wearing the garment from image1, slowly turning toward the camera" \
  --width 320 --height 192 --frames 17 --fps 16 \
  --steps 20 --max-condition-size 256 \
  --seed 42 --low-ram --metadata \
  --output referenced.mp4
```

The renderer supports at most eight references. If the packed conditioning stream has more than
five source segments, source IDs are interpolated across the trained `[1, 5]` range instead of
extrapolating beyond it. The source video counts as one segment in RV2V, so interpolation starts
at five references plus the video; R2V starts at six references.

## Edit A Video With A Reference

```sh
mlxgen generate \
  --model bernini-r-1.3b \
  --video source.mp4 \
  --reference-image garment.jpg \
  --prompt "Replace the outer shirt with the garment from image0 while preserving the person, framing, background, and motion" \
  --width 176 --height 320 --frames 17 --fps 16 \
  --steps 20 --max-condition-size 320 \
  --seed 42 --low-ram --metadata \
  --output garment-edit.mp4
```

Bernini video input always uses `source-aspect` canvas resolution and resize-only conditioning.
Matching the official renderer exactly, the source video's long edge is capped at
`--max-condition-size` (never upscaled, snapped to multiples of 16) and that resolved size is the
output canvas; `--width`/`--height` do not override a video-driven canvas. For source-video modes,
`--frames` is a maximum:
the renderer samples at the requested fps, clamps to the source duration, and returns a `4n+1`
count no greater than the request. Read the returned `num_frames` or metadata `output_frames` for
the actual duration. Crop, pad, exact-resize, `--video-strength`, and `--video-mask-path` are
rejected.

## Prompt-Guided Video Editing

Omit references for source-only V2V:

```sh
mlxgen generate \
  --model bernini-r-1.3b \
  --video source.mp4 \
  --prompt "Add a realistic snowman beside the path while keeping the dog, camera, and winter scene unchanged" \
  --width 320 --height 176 --frames 17 --fps 16 \
  --steps 20 --max-condition-size 320 --seed 42 --low-ram --metadata \
  --output edited.mp4
```

## Defaults And Guidance

The official quality defaults are 848x480, 81 frames, 16 fps, 40 steps, UniPC, and flow shift 5;
they are not the bounded proof profile or an 18 GB-host recommendation. Reference/source inputs
default to a maximum condition side of 848 pixels. The accepted range is 16-1280 in multiples of
16. Prompts have a hard 512-token UMT5 budget and warn when truncated.

| Mode | Active guidance defaults | Inactive options |
| --- | --- | --- |
| R2V | text `4.0`, reference `4.5`, APG eta `0.5`, norm `50`, momentum `0` | source guidance |
| RV2V | text `4.0`, reference `4.5`, source `1.25` | APG controls |
| V2V | text `4.0`, APG eta `0.5`, norm `50`, momentum `0` | reference/source guidance |

Use `--reference-guidance`, `--source-guidance`, `--apg-eta`, `--apg-norm-threshold`, and
`--apg-momentum` only where the table marks them active. Metadata records both active and inactive
parameters, the selected guidance mode, reference order, condition shapes, source IDs, component
revisions, prompt truncation, and video health.

## Task-Specific Recipes

Two tuned recipes measurably improve hard cases beyond the upstream default settings:

- **Many-reference property binding (R2V with 5-8 references).** Raise the reference guidance and
  try more than one seed; binding of individual reference properties is guidance- and
  seed-sensitive at this model size:

  ```sh
  mlxgen generate --model bernini-r-1.3b \
    --reference-image ... --prompt "..." \
    --reference-guidance 6.0 --seed 43 \
    --width 848 --height 480 --frames 81 --steps 40 --output out.mp4
  ```

- **Structure-changing video edits (replace or restructure the subject).** Use the `mv2v` task
  type through the Python API; its prompt conditioning explicitly licenses changing the subject's
  pose, action, and structure, which the default `v2v` conditioning holds back:

  ```python
  video = renderer.generate_video(
      seed=42,
      prompt="Replace the humanoid robot with a four-legged robotic dog ...",
      video_path="source.mp4",
      task_type="mv2v",
      guidance=5.0,
      width=848, height=480, num_frames=81, fps=16,
      num_inference_steps=40, max_condition_size=848,
  )
  ```

Worked examples of both recipes, with contact sheets, are in the
[official public parity bundle](assets/validation/bernini-r-1.3b-2026-08-11/README.md)
(`r2v_case2_tuned` and `v2v_case3_mv2vprefix`).

## Precision And Quantization

Bernini runs **BF16-only**; omit `--quantize`. The runtime executes the transformer in BF16 with a
small FP32 keep-set for precision-sensitive transformer layers, the UMT5 text encoder in FP32 (its
BF16 weights converted one block at a time, as on every Wan route), and the VAE in FP32. Quantized (8-bit or 4-bit) Bernini variants are not supported: no low-bit policy has
passed this model's numeric and visual validation gates yet, so every `--quantize` value is
rejected rather than silently degrading output.

## Python API

```python
from pathlib import Path

from mflux.models.wan.variants import BerniniRenderer

renderer = BerniniRenderer()
video = renderer.generate_video(
    seed=42,
    prompt="Bring the subject from image0 to life in a fixed medium shot",
    reference_image_paths=[Path("subject.png")],
    width=320,
    height=192,
    num_frames=17,
    fps=16,
    num_inference_steps=20,
    max_condition_size=256,
    clear_cache_each_step=True,
    clear_cache_each_transformer_block=False,
)
video.save("referenced.mp4", export_json_metadata=True)
```

The renderer stays reusable in this form. A one-shot bounded-memory host may pass
`release_denoisers_before_decode=True`, but that releases the renderer before VAE decode; construct
a new `BerniniRenderer` before generating another clip. The CLI enables this destructive release
only for a single-seed low-RAM invocation.

The CLI remains the recommended integration surface for isolated desktop/application jobs because
it supplies final MP4 health validation, failure manifests, output naming, and process-level memory
isolation.

## Proof And Known Limits

The implementation passes scheduler/source-ID tests, APG edge cases, focused FP32/runtime-BF16
transformer comparisons, VAE encode/decode comparisons, and framework regression tests.

### Current official-example evidence

The committed
[official public parity bundle](assets/validation/bernini-r-1.3b-2026-08-11/README.md) holds every
accepted and dispositioned row with full prompts, contact sheets, and generated artifacts. Start with
the [summary contact sheet](assets/validation/bernini-r-1.3b-2026-08-11/official_public_summary_contact_sheet_preview.png)
for a one-page mlx-gen overview of all pinned rows. Use the
[official example parity matrix](assets/validation/bernini-r-1.3b-2026-08-04/official_example_parity_matrix.md)
for the row-by-row status, caveats, and oracle methodology.

### Historical schema-v3 bundle

The earlier
[schema-v3 validation bundle](assets/validation/bernini-r-1.3b-2026-08-04/README.md) used a
bounded 17-frame/20-step profile and is superseded by the full-profile parity bundle above for
official-example status. It remains available through the framework validation registry:

```sh
mlxgen validation --model bernini-r-1.3b
```

That command returns the `bernini_r_1_3b_2026_08_04` record with its package identity,
route-scoped notes, upstream input references, and direct artifact paths.

### Current limits

- `v2v_case3` and `r2v_case2` need the tuned settings in
  [Task-Specific Recipes](#task-specific-recipes) to reach their best results; at upstream default
  settings they under-deliver on subject restructuring and many-reference binding respectively.
  These are 1.3B model limits, validated against the official implementation at matched settings.
- The `r2v_case2` upstream prompt exceeds the 512-token UMT5 budget and is truncated; MLX-Gen
  warns when truncation happens.
- The full `1280x672/121f` ads2v profile exceeds what the validation host can verify; the accepted
  proof uses the mid profile.
- Bernini stays experimental and fails closed in release validation until a promotion decision.
