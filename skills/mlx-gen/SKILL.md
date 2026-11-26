---
name: mlx-gen
description: Run local image and video generation on Apple Silicon with MLX through the `mlxgen` CLI (PyPI `mlx-gen`, Python imports `mlxgen` and `mflux`). Covers text-to-image, image editing (latent img2img, instruction edit, masked edit, multi-reference, reframe, outpaint), text-, image- and video-to-video (Wan2.2, Bernini-R, MiniMax-H3 with a generated soundtrack), SeedVR2 and SwiftVR image/video restoration and upscaling, explicit model download, quantized package preparation, and progress events for host applications. Use when a task mentions mlx-gen, mlxgen, mflux, MLX image or video generation on a Mac, `AbstractFramework/*-8bit` or `-4bit` Hugging Face packages, FLUX.2 Klein, Qwen Image, Z-Image, ERNIE Image, Bonsai, FIBO, Wan2.2, MiniMax-H3, SeedVR2, `mlxgen capabilities`, `DownloadRequiredError`, or `.metadata.json` generation sidecars.
license: MIT
compatibility: macOS on Apple Silicon with MLX 0.32.4 or newer (`mlx>=0.32.4,<0.33.0`, installed with the package); Python 3.10 or newer. Runs locally from model files on disk. Network access is needed only for `mlxgen download` (and for `mlxgen prepare` when the source model is not cached). Models need from a few GiB to more than 100 GiB of disk and of unified memory.
metadata:
  package-version: "0.40.2"
---

# mlx-gen

## Purpose and limits

`mlx-gen` runs image and video generation models locally on Apple Silicon through MLX. One CLI,
`mlxgen`, generates images and videos, edits images, restores and upscales, downloads models, and
prepares quantized model packages; the same routing is importable from Python as `mlxgen`. Reach
for it when the work must run on the user's Mac from files on disk, with no hosted API.

Limits to know before promising a result:

- Apple Silicon and MLX only. It is a runtime, not a model picker: curated presets and
  model-to-adapter pairing belong to AbstractVision, which uses MLX-Gen as its local backend.
- Only the supported model families run (see [references/models.md](references/models.md)).
  FLUX.2-dev, the official FLUX.2 FP8/NVFP4 repositories, and arbitrary Diffusers checkpoints are
  not inputs, and MLX-Gen packages do not load in Diffusers or Transformers.
- A route that dispatches is not a route that passed visual review. `mlxgen capabilities` says what
  can run; `mlxgen validation` says what was reviewed. Bernini-R is experimental; FIBO Edit has no
  unified route; Bonsai is text-to-image only; only MiniMax-H3 generates sound.
- Heavy runs take minutes to an hour and tens of GiB of memory: read "Rules and sharp edges" first.

## Install and verify

```sh
uv tool install --upgrade mlx-gen      # or: python -m pip install -U mlx-gen
```

Verify before anything else; this prints the installed version and reads no model weights:

```sh
if ! command -v mlxgen >/dev/null 2>&1; then
  echo "FAIL: mlxgen is not on PATH. Install it (uv tool install --upgrade mlx-gen) and open a new shell."
else
  version="$(mlxgen --help 2>/dev/null | sed -n 's/^MLX-Gen \([0-9][^ ]*\).*/\1/p' | head -n 1)"
  if [ -z "$version" ]; then
    echo "FAIL: mlxgen started but printed no version. Run 'mlxgen --help' to see the error (MLX needs macOS on Apple Silicon)."
  elif mlxgen capabilities --model z-image-turbo 2>/dev/null | grep -q '"schema_version"'; then
    echo "OK: mlx-gen $version, capability routing works"
  else
    echo "FAIL: mlx-gen $version is installed but 'mlxgen capabilities --model z-image-turbo' returned no JSON."
  fi
fi
```

Compare the printed version with `metadata.package-version` above. When they differ, the docs of
the installed version win over this skill (`mlxgen <command> --help` is always current). Gated
source repositories (FLUX.2 Klein 9B, FIBO) need the user's own Hugging Face access: ask the user
to sign in on that machine or export `HF_TOKEN` themselves, never to paste a token into the
conversation, and stop at the first authentication failure.

## Mental model

- **Files first, then generation.** `mlxgen download` fills the Hugging Face cache and
  `mlxgen prepare` writes a local package; `mlxgen generate` and `mlxgen upscale` only read files
  already on disk. Nothing downloads in the background, from the CLI or from Python constructors.
- **Error contract.** A missing model, tokenizer, or LoRA raises `DownloadRequiredError` (a
  `FileNotFoundError` subclass) carrying `repo_id`, `artifact`, `download_command`, and
  `prepare_command` (or `None`). The CLI prints its message, which starts "MLX-Gen will not
  download model files during generation." and contains the exact command, then exits non-zero.
  An impossible request (wrong input count, unsupported option) is refused before weights load: a
  CLI usage error, or `TaskInferenceError` (a `ValueError`) from the Python resolvers. A requested
  LoRA that cannot be applied fails the run instead of being skipped.
- **Model ids.** `--model` takes a catalog alias (`z-image-turbo`), a Hugging Face repo id
  (`AbstractFramework/qwen-image-8bit`), or a local folder. A repo id resolves to a complete local
  package at `./models/<repo-name>` before the Hugging Face cache. A local folder or custom repo
  name is identified only loosely: pass `--base-model <catalog alias>` with it, or it may run as
  another variant with fewer routes (`--family` alone does not configure an unknown name).
- **Routes.** Public tasks are `text-to-image`, `image-to-image`, `text-to-video`,
  `image-to-video`, and `video-to-video`. The model plus the inputs you pass select an internal
  mode (latent, edit-reference, multi-reference, masked, reframe, outpaint, first-frame, ...).
  `mlxgen capabilities --model <id>` prints every route as JSON without loading weights.
- **Quantization is either/or.** Run a source model with `--quantize 8` (quantized at load), or run
  a prepared package such as `AbstractFramework/*-8bit` with no `--quantize`. Prefer the validated
  8-bit package where one exists; 4-bit is smaller and further from the source weights.
- **Cost levers.** Model choice dominates, then canvas (`--width`, `--height`), `--steps`, and
  `--frames` for video. `--low-ram` trades speed for a lower peak. The MLX free-buffer cache is
  capped by default at total RAM / 8, clamped to 1-8 GiB (`--mlx-cache-limit-gb <gb>`, `-1`
  unlimited); raising it raises the peak.
- **Outputs.** One file per seed. `--output` is replaced if it exists unless `--no-replace`.
  `--metadata` writes a `<output stem>.metadata.json` sidecar with the resolved seed, size, steps,
  and model: the reproducibility record, replayable with `--config-from-metadata`.

## Capability map

| Capability | Entry point | Taught in |
| --- | --- | --- |
| Inspect tasks, modes, limits, defaults, memory facts | `mlxgen capabilities --model <id>` | Recipes, [capabilities-schema](references/capabilities-schema.md) |
| Inspect visual-review evidence | `mlxgen validation --model <id>` / `--list` | [capabilities-schema](references/capabilities-schema.md) |
| Download a model or LoRA repository | `mlxgen download --model <id>` | Recipes, [packages](references/packages.md) |
| Build a quantized local package | `mlxgen prepare --model <src> --path <dir> --quantize 8` | [packages](references/packages.md) |
| Text-to-image | `mlxgen generate --prompt` | Recipes, [models](references/models.md) |
| Image editing: latent, edit, masked, multi-reference, reframe, outpaint, structured control | `mlxgen generate --image` plus mode options | Recipes, [image-editing](references/image-editing.md) |
| Text-, image-, video-to-video (Wan2.2, VACE, Bernini-R) | `mlxgen generate --frames`, `--image`, `--video-path`, `--reference-image` | Recipes, [video](references/video.md) |
| Video with a generated stereo soundtrack (MiniMax-H3) | `mlxgen generate --model minimax-h3-turbo-544p` | [minimax-h3](references/minimax-h3.md) |
| Restore or upscale images and video (SeedVR2, SwiftVR) | `mlxgen upscale --image-path` / `--video-path` | Recipes, [restoration](references/restoration.md) |
| LoRA adapters | `--lora-paths`, `--lora-scales` | Rules, [lora](references/lora.md) |
| Several variations in one run | `--seed 1 2 3`, `--auto-seeds N` | Rules |
| Step previews | `--stepwise-image-output-dir`, `--preview-decoder` | [image-editing](references/image-editing.md) |
| Machine-readable progress for hosts | `--json-events` | [python-api](references/python-api.md) |
| Python runtime | `mlxgen.load_generation_model`, `get_model_capabilities`, `resolve_generation_plan` | Recipes, [python-api](references/python-api.md) |

Not available: a server or HTTP API; automatic model download or fallback to another model;
tensor batching (several seeds run one after another on one loaded model); FLUX.1 in the unified
router (only the legacy `mflux-generate-*` entry points, which integrations should not call);
reference-to-video and a closing keyframe on MiniMax-H3; a video duration flag (duration is
`frames / fps`). Side effects: `download` and `prepare` are the only commands that use the network;
`download` writes to the Hugging Face cache, `prepare` writes the package folder with a `README.md`
model card, and generation writes the output, the optional sidecar, a `<stem>.failure.json`
manifest when a Wan run fails, and a small on-disk prompt-embedding cache for Wan.

## Recipes

### End-to-end: check, download, generate an image

```sh
MODEL="AbstractFramework/flux.2-klein-4b-8bit"   # choose by memory tier: references/models.md
mlxgen capabilities --model "$MODEL" > caps.json  # exits non-zero with the reason for an unknown id
mlxgen download --model "$MODEL"                  # network and disk; fast when already cached
mlxgen generate --model "$MODEL" --prompt "<PROMPT>" \
  --width 512 --height 512 --steps 4 --guidance 1 \
  --seed 42 --metadata --output "<OUTPUT.png>"
```

Success is exit status 0, `<OUTPUT.png>` on disk, and `<OUTPUT>.metadata.json` beside it. In
`caps.json`, check that `capabilities` has a row whose `public_task` matches the job and read its
`supports_*` flags before adding options; empty `capabilities` and `restoration` arrays mean the
model has no route. On a missing-files error, run the printed `mlxgen download` command once and
retry. Steps and guidance are model-specific (distilled FLUX.2 Klein: 4 steps, guidance 1; Z-Image
Turbo: 9 steps): take them from [references/models.md](references/models.md).

### Edit an image

```sh
mlxgen generate --model "AbstractFramework/qwen-image-edit-2509-8bit" \
  --image "<INPUT.png>" --prompt "<EDIT INSTRUCTION>" \
  --steps 20 --seed 42 --metadata --output "<EDITED.png>"
```

The output keeps the source aspect ratio; `--width`/`--height` are only a size target. What you add
selects the mode: `--image-strength 0.4` for a whole-image restyle (latent img2img), `--mask-path`
for a local edit (white is repainted), a second `--image` for multi-reference, `--reframe-padding`
or `--outpaint-padding` to extend the canvas. Each needs the matching `supports_*` capability
flag. See [references/image-editing.md](references/image-editing.md).

### Upscale or restore an image

```sh
mlxgen download --model "AbstractFramework/seedvr2-3b-8bit"
mlxgen upscale --model "AbstractFramework/seedvr2-3b-8bit" \
  --image-path "<INPUT.png>" --resolution 2x --metadata --output "<UPSCALED.png>"
```

`--resolution` is a scale factor (`2x`) or an integer shorter-edge target (`1024`); the aspect
ratio is preserved and no prompt is used. Video restoration (`--video-path`) has a stricter safe
profile: see [references/restoration.md](references/restoration.md).

### Generate a video

A five-second silent clip at a practical cost. Confirm the run with the user first.

```sh
mlxgen download --model "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"
mlxgen generate --model "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit" --prompt "<PROMPT>" \
  --width 832 --height 480 --frames 101 --fps 20 --steps 25 --flow-shift 3 \
  --seed 42 --low-ram --metadata --output "<VIDEO.mp4>"
```

This profile takes about 12 minutes on an Apple M5 Max. Add `--image "<FIRST_FRAME.png>"` for
image-to-video. Frame counts are adjusted to `4n + 1` and sizes to the model's pixel multiple. A
failed Wan run leaves `<VIDEO>.failure.json`. Other models, sizes, and video-to-video:
[references/video.md](references/video.md). With sound: [references/minimax-h3.md](references/minimax-h3.md).

### Drive it from Python

```python
from mflux.models.common.download_policy import DownloadRequiredError
from mlxgen import TaskInferenceError, get_model_capabilities, load_generation_model

MODEL = "<MODEL>"  # for example "AbstractFramework/z-image-turbo-8bit"
try:
    rows = get_model_capabilities(model=MODEL).capabilities  # reads no weights
    if not any(row.public_task == "text-to-image" for row in rows):
        raise SystemExit(f"{MODEL} has no text-to-image route")
    loaded = load_generation_model(model=MODEL)  # reads weights from disk
except TaskInferenceError as exc:
    raise SystemExit(f"unroutable request: {exc}")
except DownloadRequiredError as exc:
    raise SystemExit(f"model files missing, run: {exc.download_command}")

results = loaded.generate_outputs(
    seeds=[101, 202], prompt="<PROMPT>", width=1024, height=1024, num_inference_steps=9,
    output="<OUTPUT.png>", save_kwargs={"export_json_metadata": True},
    progress_callback=lambda event: print(event.phase, event.seed, event.step, event.total_steps),
)
print([(result.seed, result.saved_path) for result in results])
```

Returns one `GeneratedOutput` per seed; the files here are `<OUTPUT>_seed_101.png` and
`<OUTPUT>_seed_202.png`. The package must be importable in that interpreter (`pip install mlx-gen`,
not only `uv tool install`). Pass `quantize=8` only for a source model. An unknown generate keyword
is refused before the run. Event fields, `--json-events` for subprocess hosts, SeedVR2, and
outpaint from Python: [references/python-api.md](references/python-api.md).

## Rules and sharp edges

**Cost comes first.** `generate`, `upscale`, `prepare`, and `download` are the expensive calls;
`--help`, `capabilities`, and `validation` are free and read no weights. Before an expensive call:

- Answer what you can from `mlxgen capabilities` and [references/models.md](references/models.md):
  whether the route exists, its limits, and the published memory figures.
- Tell the user the expected cost and get a yes for anything beyond a small image run. Documented
  reference points: FLUX.2 Klein 4B 8-bit at 512px, 9.2 GiB peak; Qwen Image Edit 2511 8-bit at
  768x432, 30.9 GiB peak; Wan A14B, about 33 GiB peak even at a small 384x224 profile, and about
  30 minutes for a five-second 480x240 clip on an Apple M5 Max; Wan TI2V-5B 8-bit at 1280x704 and
  17 frames, 103.7 GiB peak; MiniMax-H3 at `960x544` and 124 frames, 37.0 GiB peak with the 8-bit
  package (26.3 GiB with `--low-ram`, 11.1 GiB with `--stream-blocks --low-ram`) and 15 to 18
  minutes per clip (16 to 21 from the 16-bit weights; 45 to 48 minutes at `1344x768`). Downloads run from 3.6 GiB (Bonsai) to 134 GiB (MiniMax-H3 source).
- A run that does not fit can be killed by the OS mid-denoise with no output, and other
  applications' memory counts. Start with the smallest documented profile, run one heavy job at a
  time, and in Python serialize access to a loaded model.

Mistakes to avoid:

- Do not pass `--quantize` to a prepared or published package (`*-8bit`, `*-4bit`, `*-bf16`, a
  `prepare` output, Bonsai). Bernini-R rejects every `--quantize` value.
- On a missing-files error, run the printed command once; do not loop. For a published package
  id use the `download` command, not `prepare`.
- A Wan 8-bit package saves disk only: it runs at BF16-class memory. Plan Wan memory as BF16.
- `--image-strength` means latent img2img only. Edit, multi-reference, and masked routes reject
  it. Latent-only models (Z-Image, ERNIE, base Qwen) require it for an unmasked image input, and
  the run executes `floor(steps x image_strength)` steps.
- `--negative-prompt` is rejected by distilled FLUX.2 Klein, Bonsai, and MiniMax-H3; check
  `supports_negative_prompt`. Wan applies its official negative prompt unless you pass `""`.
- SeedVR2 and SwiftVR run only through `mlxgen upscale` and take no prompt; prompt-guided video
  edits belong to `mlxgen generate`.
- Pass `--seed` for reproducible work; without it the seed is taken from the clock. Several seeds (`--seed 1 2`
  or `--auto-seeds N`) append `_seed_<seed>` to the output stem, or fill `{seed}` in `--output`;
  duplicate seeds are refused.
- A local folder named `qwen-image-edit-2511-8bit` runs as base Qwen Image unless you add
  `--base-model qwen-image-edit-2511`. Give `capabilities` the same `--model` and `--base-model`
  as the run, and check that the route you need is listed.
- Output size can differ from the request (source aspect ratio, model multiples, frame grids):
  trust the sidecar, not the command line.
- LoRA: download the adapter repository first (`mlxgen download --model <repo> --all-files`), pass
  it as `--lora-paths "<owner/repo>:<file.safetensors>"` with exactly one `--lora-scales` value per
  path, and treat only rows with `lora_status: "validated"` as production-supported.
- MiniMax-H3 weights are under the MiniMax H3 Community License, whose grant excludes the EU, the
  UK, the Republic of Korea, and the USA. Confirm with the user before downloading or using them.

## When things fail

| Symptom | Cause | Fix |
| --- | --- | --- |
| "MLX-Gen will not download model files during generation." | Files are not on disk | Run the `mlxgen download` command in the message, then retry |
| "Could not infer a supported backend from model ..." or "family=... is not enough to configure model" | Unrecognized id or folder name | Pass `--base-model <catalog alias>` |
| "--path prepares a local model folder and is not a generation option" | `--path` given to `generate` | Use `mlxgen prepare` for packages, `--output` for results |
| LoRA missing, mismatched scales, or "not compatible" | Adapter not downloaded, not one scale per path, or trained for another base model | Download with `--all-files`, match the counts, use an adapter for that exact family |
| Process killed, or far slower than documented | Memory pressure | Smaller canvas, fewer frames, `--low-ram`, a 4-bit or smaller model, close other apps |
| Output size or step count differs from the request | Source-aspect canvas, model multiples, or `--image-strength` | Read the `.metadata.json` sidecar; `--canvas-policy exact-resize` forces the canvas |

More symptoms and fixes: <https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/troubleshooting.md>

## Go deeper

Skill references, each opened only when the task needs it:

- [references/models.md](references/models.md): choose a model by task and memory tier; aliases, steps, limits.
- [references/packages.md](references/packages.md): published packages and sizes, `download` versus `prepare`, building a package, id resolution.
- [references/capabilities-schema.md](references/capabilities-schema.md): JSON fields of `capabilities` and `validation`.
- [references/image-editing.md](references/image-editing.md): edit modes, masks, reframe/outpaint, control images, previews.
- [references/video.md](references/video.md): Wan2.2, VACE, and Bernini-R options, sizes, and costs.
- [references/minimax-h3.md](references/minimax-h3.md): video with sound, packages, schedules, prompting.
- [references/restoration.md](references/restoration.md): SeedVR2 and SwiftVR image and video restoration.
- [references/lora.md](references/lora.md): adapter forms, validated rows, per-family rules.
- [references/python-api.md](references/python-api.md): Python runtime, progress events, `--json-events`.

Project docs as raw Markdown:
[getting started](https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/getting-started.md),
[API and CLI reference](https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/api.md),
[FAQ](https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/faq.md),
[all docs in one file](https://raw.githubusercontent.com/lpalbou/mlx-gen/main/llms-full.txt).
