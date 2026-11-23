# Python Runtime, Progress Events, And Host Integration

Open this file when embedding MLX-Gen in an application: calling it from Python, keeping a model
loaded across requests, or driving the CLI as a subprocess with machine-readable progress.

Two integration styles:

- **Subprocess**: run `mlxgen ... --json-events` and read JSON lines. Works with a
  `uv tool install`; the process exit releases all memory. Prefer it for one-off jobs.
- **In-process**: `import mlxgen` in an interpreter where `mlx-gen` is installed
  (`pip install mlx-gen`). Keeps weights loaded between jobs. Prefer it for warm workers.

Either way, nothing downloads at run time: catch `DownloadRequiredError`, show its
`download_command`, and stop.

## Subprocess: `--json-events`

`mlxgen generate` and `mlxgen upscale` accept `--json-events`. JSON lines go to stdout, human text
moves to stderr. Each line is one event; fields that have no value are omitted.

| Field | Meaning |
| --- | --- |
| `type` | `"runtime"`. |
| `command`, `model`, `seed` | The routed command, the model, and the seed of the current item. |
| `phase` | `start`, `denoise`, `decode`, `convert`, `generated`, `save`, `complete`, `failed`, `interrupted`. |
| `task` | Public task, for example `text-to-image`. |
| `step`, `total_steps`, `progress` | Denoise progress; `progress` is `step / total_steps`. |
| `frame`, `total_frames`, `frame_progress` | Video frame context. |
| `width`, `height`, `fps` | Resolved output facts: Wan `start` events carry the final canvas; video `save` events add `fps`, `width`, `height`, `total_frames`. |
| `output_path` | On `save`, `complete`, and `failed`. |
| `health_check` | `"skipped"` when `--no-validate-health` was used. |
| `error`, `error_type` | On `failed`: the message and the exception class name. |
| `diagnostics_path` | On `failed`, when a failure manifest was written (Wan). |
| `remediation` | On `failed`, when a next step is known: `{"kind": "download-required", "repo_id", "artifact", "download_command", "prepare_command"?}` or `{"kind": "cli-usage", "usage"?}`. |

Order for an image: `start`, `denoise` per step, `generated` (image in memory), `save`,
`complete` (file written). Video adds `decode` and `convert` before `generated`, and `complete`
also means the MP4 passed its health check. Only `complete` guarantees a file on disk. Ignore
phases and fields you do not recognize; new ones can appear.

```python
import json
import subprocess
import sys

command = [
    "mlxgen", "generate",
    "--model", "<MODEL>",
    "--prompt", "<PROMPT>",
    "--seed", "42",
    "--metadata",
    "--json-events",
    "--output", "<OUTPUT.png>",
]

saved, failure = [], None
proc = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)  # human text stays on stderr
for line in proc.stdout:
    try:
        event = json.loads(line)
    except ValueError:
        continue  # not an event line
    phase = event.get("phase")
    if phase == "denoise":
        print(f"step {event.get('step')}/{event.get('total_steps')}", file=sys.stderr)
    elif phase == "complete" and event.get("output_path"):
        saved.append(event["output_path"])  # the file is on disk
    elif phase == "failed":
        failure = event
code = proc.wait()

if failure is not None:
    remediation = failure.get("remediation") or {}
    if remediation.get("kind") == "download-required":
        sys.exit(f"model files missing, run: {remediation['download_command']}")
    if remediation.get("kind") == "cli-usage":
        sys.exit(f"bad arguments: {failure.get('error')}")
    sys.exit(f"generation failed: {failure.get('error')} (diagnostics: {failure.get('diagnostics_path')})")
if code != 0 or not saved:
    sys.exit(f"mlxgen exited with {code} and reported no saved output; read its stderr")
print("saved:", saved)
```

Treat a non-zero exit or an empty `saved` list as failure even without a `failed` event: an
out-of-memory kill emits nothing.

## In-process: resolve, load, generate

All from `import mlxgen`. The resolvers read no weights.

| Call | Returns | Use |
| --- | --- | --- |
| `get_model_capabilities(*, model, model_config=None, family=None, base_model=None)` | `ModelCapabilities` (`.capabilities`, `.restoration`, `.to_dict()`) | What a model can do |
| `resolve_task(*, model, image_count=0, video_count=0, ...)` | `ResolvedTask` (`.task`) | Which public task a request shape maps to |
| `resolve_generation_plan(*, model, image_count=0, ...)` | `GenerationPlan` (`.task`, `.mode`, `.handler_id`, `.capability_id`, `.canvas_policies`, `.default_canvas_policy`, `.resize_modes`, `.control_model`, `.lora_status`) | The exact route before loading |
| `resolve_generation_runtime(*, model, ...)` | `GenerationRuntimePlan` (`.runtime_id`, `.plan`, `.cache_key(...)`, `.load(...)`) | A worker-cache key without loading |
| `load_generation_model(*, model, ..., quantize=None, model_path=None, lora_paths=None, lora_scales=None, lora_target_roles=None, model_kwargs=None)` | `LoadedGenerationModel` (`.model`, `.plan`, `.runtime_id`, `.cache_key`) | Resolve and load in one call |
| `get_model_validation(model, profile_id=...)` | `ModelValidation` (`.status`, `.records`) | Review evidence |

The request-shape keywords (`image_count`, `video_count`, `reference_image_count`, `task`,
`i2i_mode`, `has_image_strength`, `has_mask`, `has_outpaint`, `has_reframe`, `has_control_image`,
`has_video_strength`, `has_video_mask`, `has_lora`, `family`, `base_model`) are the same on every
resolver and on `load_generation_model`. Contradictions raise `TaskInferenceError` before any
weight is read, for example an image passed to a text-to-video-only model.

```python
from mflux.models.common.download_policy import DownloadRequiredError
from mlxgen import TaskInferenceError, load_generation_model

try:
    loaded = load_generation_model(
        model="AbstractFramework/qwen-image-edit-2511-8bit",
        image_count=1,  # route selection: one input image -> edit-reference
    )
except TaskInferenceError as exc:
    raise SystemExit(f"unroutable request: {exc}")
except DownloadRequiredError as exc:
    raise SystemExit(f"model files missing, run: {exc.download_command}")

print(loaded.plan.task, loaded.plan.mode, loaded.plan.handler_id)
result = loaded.generate_output(
    seed=42,
    prompt="<EDIT INSTRUCTION>",
    image_paths=["<INPUT.png>"],
    num_inference_steps=20,
    output="<EDITED.png>",
    save_kwargs={"export_json_metadata": True},
)
print(result.saved_path)
```

- `generate_output(seed=...)` returns one `GeneratedOutput`; `generate_outputs(seeds=[...])`
  returns a list, one seed at a time on the same loaded model (serial reuse, not tensor batching).
  Fields: `seed`, `task`, `artifact`, `item_index`, `item_count`, `output_path`, `saved_path`.
- With `output=...` the file is written and `saved_path` is set; several seeds get
  `_seed_<seed>` in the name. Without `output` the in-memory `artifact` is returned and saving is
  yours (`artifact.save(path, export_json_metadata=True)`).
- `overwrite` defaults to `False` here: an existing target is kept and the new file gets a
  suffixed name. (The CLI replaces by default.)
- `save_kwargs`: `export_json_metadata=True` writes the sidecar; `embed_metadata=True` embeds
  metadata in an image; `validate_health=False` skips the video health re-decode.
- Other keywords go to the route's own generate call (`generate_image` or `generate_video`):
  `prompt`, `width`, `height`, `num_inference_steps`, `guidance`, `negative_prompt`, and
  route-specific ones such as `image_path` and `image_strength` (latent), `image_paths` (edit),
  `num_frames` and `fps` (video). A keyword the route does not take is refused before the run,
  with a message naming what the route accepts.
- `quantize=8` is for source models only. `model_path="./models/<dir>"` loads a local package for
  the resolved model. `model_kwargs={...}` forwards constructor extras (Wan:
  `keep_text_encoder_resident=True`, `prompt_embed_disk_cache=False`); an unknown one raises
  `TypeError` at load.
- `post_process=<callable>` runs on each artifact after generation and before saving.
- Use `loaded.cache_key` to deduplicate loaded models in a worker pool.

## Progress callbacks

Pass `progress_callback=<callable>` to `generate_output(s)`, or subscribe on a model with
`model.callbacks.subscribe_progress(callback, task="text-to-image")`, which returns an
unsubscribe function. The callback receives a `ProgressEvent` (`from mflux.callbacks import
ProgressEvent`) with no tensors:

| Attribute | Meaning |
| --- | --- |
| `phase` | Same vocabulary as the JSON events. |
| `step`, `total_steps`, `progress` | Denoise progress (`progress` is `step / total_steps`). |
| `frame`, `total_frames`, `frame_progress` | Video context, else `None`. |
| `task`, `timestep` | Public task and current timestep. |
| `seed`, `item_index`, `item_count`, `output_path` | Added by the loaded-runtime wrapper for multi-seed runs. |
| `width`, `height` | Resolved output size on early events where known. |

- On a direct model call, `complete` means the in-memory artifact is ready, not that a file was
  saved. Through `generate_output(s)` with `output=...`, the wrapper emits `generated`, `save`,
  then `complete` once the file is written.
- Exactly one terminal phase ends a run: `complete`, `failed`, or `interrupted`.
- Exceptions raised inside a callback propagate to the caller: keep handlers small and defensive.

## Direct model classes

The route helpers cover the `mlxgen generate` families. Restoration and backend-specific control
use the model classes directly; each call produces one artifact for one seed.

```python
from mflux.models.common.config import ModelConfig
from mflux.models.common.download_policy import DownloadRequiredError
from mflux.models.seedvr2 import SeedVR2

PACKAGE = "AbstractFramework/seedvr2-3b-8bit"
try:
    model = SeedVR2(model_path=PACKAGE, model_config=ModelConfig.from_name(PACKAGE))
except DownloadRequiredError as exc:
    raise SystemExit(f"model files missing, run: {exc.download_command}")

path = model.restore_image_to_path(
    seed=42,
    image_path="<INPUT.png>",
    resolution=1024,  # shorter-edge target in pixels
    output_path="<UPSCALED.png>",
    export_json_metadata=True,
)
print(path)
```

- `SeedVR2` also has `generate_image(seed, image_path, resolution, softness=0.0, ...)` (returns
  the image in memory) and `restore_video_to_path(*, seed, video_path, resolution, output_path,
  start_seconds=0.0, max_frames=None, temporal_chunk_size=49, temporal_chunk_overlap=16,
  color_correction_mode="wavelet", drop_audio=False, export_json_metadata=False, overwrite=True,
  validate_health=True, ...)`, which returns the written `Path` and follows the same chunk and
  audio rules as the CLI ([restoration.md](restoration.md)).
- Other families follow the same shape, for example
  `from mlxgen.models.z_image import ZImageTurbo`, then `ZImageTurbo(quantize=8)` and
  `model.generate_image(seed=..., prompt=..., width=..., height=..., num_inference_steps=...)`.

## Outpaint from Python

`run_outpaint(loaded=..., source_image=..., padding="top,right,bottom,left", seeds=[...],
prompt=..., output=...)` runs the same canvas, fill, and pass decisions as the CLI on a model
loaded with `has_outpaint=True`. `prepare_outpaint(source_image=..., padding=..., model=...)`
builds the canvas and returns the geometry without loading weights (use it as a context manager),
and `outpaint_contract_for_model(model=...)` returns the route's fill contract. An unsupported
request raises `OutpaintError`. Details:
<https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/python-integration.md>.

## Process-level settings

- `MFLUX_MLX_CACHE_LIMIT_GB=<gb>` sets the MLX buffer-cache cap for Python hosts (`-1` for
  unlimited). Without it, the first weight load applies total RAM / 8 clamped to 1-8 GiB and says
  so on stderr; a limit the host already set through `mx.set_cache_limit` at or below half of
  physical RAM is kept.
- `MFLUX_NO_WEIGHT_PREFETCH=1` disables the sequential read-ahead of weight files at load.
- Treat a loaded model as stateful: serialize access to it across threads and requests.

Source docs: <https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/python-integration.md>,
<https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/api.md>.
