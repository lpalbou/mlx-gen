# Capabilities And Validation JSON

Open this file when you need to read or branch on the output of `mlxgen capabilities` or
`mlxgen validation`. Both commands print JSON to stdout, read no weights, and need no model files.

## `mlxgen capabilities --model <id> [--base-model <alias>] [--family <family>]`

Top level (`schema_version` 16):

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer. Fields are added over time; ignore fields you do not know. |
| `family` | `qwen`, `flux2`, `fibo`, `z-image`, `ernie-image`, `wan`, `minimax-h3`, `bonsai`, `seedvr2`, or `swiftvr`. |
| `label` | Human name of the resolved entry, for example `MiniMax-H3 Turbo 544p`. |
| `model_name` | The repository or path that will be loaded. |
| `capabilities` | Array of `mlxgen generate` routes. Empty: not routable through `generate`. |
| `restoration` | Array of `mlxgen upscale` routes. Empty: not routable through `upscale`. |
| `universal_options` | Options every generate route in this build accepts (`["--low-ram"]`). |

An unknown model exits with status 2 and an error on stderr; stdout is empty. Both arrays empty
means the model is known but has no public route (for example `briaai/Fibo-Edit`).

A field is `null`, `false`, or an empty array when the route does not publish that fact. Image
routes publish no `default_steps`, `default_width`, or `measured_runs`; do not read `null` as zero.

### Generation rows (`capabilities[]`)

Identity and inputs:

| Field | Meaning |
| --- | --- |
| `id` | Route id, for example `flux2.edit`, `qwen.inpaint`, `wan.first-frame`. |
| `handler_id` | Backend that serves it, for example `flux2.edit`. |
| `public_task` | `text-to-image`, `image-to-image`, `text-to-video`, `image-to-video`, `video-to-video`. |
| `mode` | Internal mode: `text-only`, `latent-img2img`, `edit-reference`, `multi-reference`, `text-video`, `first-frame-i2v`, `latent-video`, `reference-video`, `reference-video-edit`. |
| `default_for_task` | Whether this row is chosen for its task when nothing disambiguates. |
| `min_images`, `max_images` | Primary `--image` count. `max_images: null` means no upper bound. |
| `min_videos`, `max_videos` | Source video count. |
| `min_reference_images`, `max_reference_images` | `--reference-image` count, separate from `--image`. |
| `primary_image_index` | Which image anchors the geometry (`0`), or `null`. |
| `model_override`, `control_model` | A substituted model or an auto-injected sidecar, or `null`. |

Option support (send an option only when its flag is `true`):

| Field | Option it gates |
| --- | --- |
| `supports_guidance` | `--guidance` |
| `supports_negative_prompt` | `--negative-prompt` / `--negative` |
| `supports_image_strength` | `--image-strength` |
| `supports_mask` | `--mask-path` |
| `supports_control_image` | `--controlnet-image-path` (structured control on a text-to-image route) |
| `supports_reframe` | `--reframe-padding` |
| `supports_outpaint`, `supports_outpaint_fill` | `--outpaint-padding`; `--outpaint-fill` and `--outpaint-fill-color` |
| `supports_lora` | `--lora-paths`, `--lora-scales` |
| `supports_frames`, `supports_fps` | `--frames`, `--fps` |
| `supports_flow_shift`, `flow_shift_option`, `flow_shift_parameter` | The flow-shift flag as this route spells it: `--flow-shift` / `flow_shift` on Wan, `--video-shift` / `video_shift` on MiniMax-H3 |
| `supports_audio_shift` | `--audio-shift` |
| `supports_video_strength`, `supports_video_mask` | `--video-strength`, `--video-mask-path` |
| `supports_last_image`, `supports_context_frames`, `supports_svi` | `--last-image`, `--context-frames`, `--svi-anchor-image` (Wan A14B image-to-video) |
| `supports_text_encoder_release` | `--release-text-encoder` |

`supports_guidance` and `supports_negative_prompt` are independent: distilled FLUX.2 Klein has
guidance without a negative prompt, Z-Image Turbo the reverse.

Geometry and length:

| Field | Meaning |
| --- | --- |
| `dimension_multiple` | Width and height multiple, or `null`. |
| `canvas_policies`, `default_canvas_policy` | Accepted `--canvas-policy` values (`source-aspect`, `exact-resize`) and the default. |
| `resize_modes` | Accepted `--resize-mode` values (`resize`, `crop`, `pad`); empty when the route pins geometry. |
| `default_width`, `default_height`, `default_steps`, `default_frames` | The entry's defaults, where published. |
| `min_frames`, `max_frames`, `frame_multiple`, `frame_remainder`, `frame_rounding` | Frame grid: valid counts satisfy `frames % frame_multiple == frame_remainder`; `frame_rounding` says how other values are adjusted. |
| `output_fps` | Fixed output rate when `supports_fps` is `false`. |
| `default_flow_shift`, `default_audio_shift` | Schedule shift defaults. |

LoRA:

| Field | Meaning |
| --- | --- |
| `lora_status` | `unsupported`, `mapped-unvalidated`, or `validated`. Only `validated` is production-supported. |
| `lora_target_roles` | Components adapters target, for example `["transformer"]`. |
| `lora_validation_profile` | Profile id to pass to `mlxgen validation --profile`, or `null`. |

Audio and prompt structure (MiniMax-H3):

| Field | Meaning |
| --- | --- |
| `generates_audio`, `audio_channels`, `audio_sample_rate` | The route composes a soundtrack with the picture. |
| `prompt_sections` | Array of `{key, label, option, parameter, role, required}`: the structured prompt sections and the option that fills each. |

Precision and memory (where published):

| Field | Meaning |
| --- | --- |
| `weight_precision` | Precision of the released weights before quantization. |
| `unquantized_weights_bytes` | Resident size of the unquantized weights; compare with the machine's memory. |
| `recommended_quantize` | The quantization to default to, or `null`. |
| `validated_quantization_bits` | Bit widths with recorded evidence (narrower than what `--quantize` accepts). |
| `measured_runs` | Array of measured runs: `outcome` (`completed` or `killed`), `footprint_bytes`, `mlx_peak_bytes`, `terminated_at`, and the conditions `quantize`, `width`, `height`, `frames`, `steps`, `cache_limit_bytes`, `release_text_encoder`, `machine_total_ram_bytes`. For a killed run `footprint_bytes` is a lower bound. |
| `max_validated_frames` | Largest frame count with a completed measurement. |
| `peak_bytes_fixed`, `peak_bytes_per_packed_row` | Expected footprint is roughly `peak_bytes_fixed + peak_bytes_per_packed_row x packed_rows`. An estimate, not a guarantee that a run fits. |

Outpaint contract (non-null only on outpaint rows):

| Field | Meaning |
| --- | --- |
| `outpaint_fill_modes`, `outpaint_default_fill_mode` | Accepted `--outpaint-fill` values and the default (`auto` on FLUX.2 Klein, fixed `edge` on Qwen edit). |
| `outpaint_pass_modes`, `outpaint_default_passes` | Accepted `--outpaint-passes` values (`auto`, `1`, `2`) and the default. |
| `outpaint_auto_edge_fill_max_stretch`, `outpaint_auto_split_corner_ratio` | Thresholds `auto` uses to switch fill mode and to split into two passes. |
| `outpaint_preservation` | How source pixels are kept (`adaptive-content-aware-source-blend`). |
| `outpaint_recommended_lora` | Adapter measured to help on that exact route, or `null`. Optional. |
| `outpaint_validated_padding`, `outpaint_validated_fill_mode`, `outpaint_validated_max_canvas_pixels` | The envelope the published proof used; outside it outpaint is supported but unvalidated. |

### Restoration rows (`restoration[]`)

| Field | Meaning |
| --- | --- |
| `id`, `handler_id`, `mode`, `public_task` | For example `seedvr2.restore-image`, `swiftvr.restore-video`. |
| `command` | `mlxgen upscale`. |
| `accepted_media` | `["image"]` or `["video"]`; one row per input kind. |
| `min_images`, `max_images`, `min_videos`, `max_videos` | Input counts. `max_images: 0` means no stills. |
| `default_resolution` | `384` for images, `1x` for video on SeedVR2. |
| `supports_scaling`, `scale_factors`, `supports_short_side_resolution` | Whether the route can enlarge; SwiftVR reports `["1x"]` only. |
| `supports_quantization`, `quantization_bits`, `weight_precision` | Quantized package support and run precision. |
| `frame_multiple`, `frame_remainder`, `min_frames`, `max_frames` | Clip-length contract (SwiftVR: `frames % 4 == 1`). |
| `dimension_multiple`, `max_canvas_pixels` | Output geometry constraint (`16` on SeedVR2 video, `1` on images). |
| `color_correction_modes`, `default_color_correction` | Exactly the `--color-correction` values accepted. |
| `chunk_strategy`, `chunk_size_default`, `chunk_size_min`, `chunk_size_multiple`, `chunk_size_remainder`, `chunk_overlap_default`, `chunk_overlap_min`, `chunk_overlap_multiple`, `chunk_options_user_settable` | Temporal chunking contract for video. |
| `supports_steps`, `steps_min`, `steps_max`, `supports_softness`, `supports_vae_tiling`, `supports_audio_passthrough`, `supports_clip_window`, `supports_multi_seed` | Per-route option support. |

### Summarize a model's routes

```python
import json
import subprocess
import sys

MODEL = "<MODEL>"

proc = subprocess.run(["mlxgen", "capabilities", "--model", MODEL], capture_output=True, text=True)
if proc.returncode != 0:
    # Unknown id or unclassifiable name: the last stderr line names the fix.
    sys.exit(f"capabilities failed: {(proc.stderr.strip().splitlines() or ['no output'])[-1]}")
caps = json.loads(proc.stdout)
if not caps["capabilities"] and not caps["restoration"]:
    sys.exit(f"{MODEL}: no route through `mlxgen generate` or `mlxgen upscale`")
print(caps["label"], "->", caps["model_name"], "schema", caps["schema_version"])
for row in caps["capabilities"]:  # `mlxgen generate` routes
    flags = sorted(key[9:] for key, value in row.items() if key.startswith("supports_") and value is True)
    print("generate", row["id"], row["public_task"], f"images {row['min_images']}-{row['max_images']}", flags)
    if row["default_steps"] is not None:  # published only where the route declares its defaults
        print("  defaults", row["default_width"], row["default_height"], row["default_steps"], row["default_frames"])
    for run in row["measured_runs"]:
        print("  measured", run["outcome"], run["width"], run["height"], run["frames"], run["footprint_bytes"])
for row in caps["restoration"]:  # `mlxgen upscale` routes
    print("upscale", row["id"], row["accepted_media"], "default", row["default_resolution"])
```

## `mlxgen validation`

Route support and visual validation are separate. `validation` reports recorded review evidence for
exact model or package rows; it does not control what `generate` accepts.

```sh
mlxgen validation --list                                   # profile ids, titles, record counts
mlxgen validation --model "<MODEL>"                        # rows for that exact model
mlxgen validation --model "<MODEL>" --profile "<PROFILE_ID>"
```

- `--list` returns `{"profiles": [{"id", "title", "description", "canonical_source", "record_count"}]}`.
- `--model` returns `{"model", "profile_id", "status", "records": [...]}`. `status` is the overall
  result for that model in the profile: for example `PASS`, `FAIL`, or `NOT_AVAILABLE` with an
  empty `records` array when the profile holds no evidence for that model (not a failure, and not
  a statement about route support). Each record carries `status`, `mode`, `public_task`, `prompt`,
  `source_images`, `artifact_path`, `reviewer_notes`, `evidence_date`, `package_variant`, and
  `family`.
- Without `--profile`, the first profile with evidence for the model is used. Pass the
  `lora_validation_profile` value from a capability row to see a LoRA proof row.
- An unknown `--profile` id exits with a usage error naming it.
- Use exact handles (`AbstractFramework/qwen-image-edit-2509-8bit`), not short aliases: evidence is
  recorded per exact model or package.

## Python equivalents

```python
from mlxgen import TaskInferenceError, get_model_capabilities, get_model_validation, resolve_generation_plan

try:
    caps = get_model_capabilities(model="<MODEL>")          # ModelCapabilities; .to_dict() is the CLI JSON
    plan = resolve_generation_plan(model="<MODEL>", image_count=1)
except TaskInferenceError as exc:                           # unknown model or impossible request
    raise SystemExit(str(exc))
print(plan.task, plan.mode, plan.handler_id, plan.default_canvas_policy)
print(get_model_validation("<MODEL>").status)
```

`resolve_generation_plan` and `resolve_task` take the request shape as keywords: `image_count`,
`video_count`, `reference_image_count`, `task`, `i2i_mode`, `has_image_strength`,
`has_video_strength`, `has_video_mask`, `has_mask`, `has_control_image`, `has_outpaint`,
`has_reframe`, `has_lora`, plus `family` and `base_model`. They load no weights and raise
`TaskInferenceError` when the model cannot serve the request, with a message naming what it accepts.

Source docs: <https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/api.md>.
