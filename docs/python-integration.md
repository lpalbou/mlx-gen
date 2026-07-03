# Python Integration

MLX-Gen can be embedded directly in Python. The underlying model classes remain available, but new integrations should start from the route-resolved `mlxgen` helper APIs documented below for the unified `mlxgen generate` families. SeedVR2 sits outside that planner surface: use `mlxgen upscale` on the CLI and direct `SeedVR2.generate_image(...)` / `SeedVR2.restore_video_to_path(...)` in Python.

## Cache-Only Runtime

Python callers should download or prepare models before constructing model objects. Runtime constructors and generation calls do not download missing artifacts. See [Model Management](model-management.md) for the CLI setup commands.

```python
from mflux.models.common.download_policy import DownloadRequiredError
from mlxgen.models.z_image import ZImageTurbo

try:
    model = ZImageTurbo(quantize=8)
except DownloadRequiredError as exc:
    print(exc.download_command)
    raise
```

For user-facing applications, show the exception message or the `download_command`/`prepare_command` fields and stop the workflow.

## AbstractVision

MLX-Gen is intended to be the Apple Silicon / MLX dependency for
[AbstractVision](https://github.com/lpalbou/abstractvision), while AbstractVision remains a
cross-platform image/video generation abstraction inside the wider
[AbstractFramework](https://abstractframework.ai/) ecosystem.

That split means:

- AbstractVision owns provider-neutral image/video request objects, artifact storage, capability checks, and AbstractCore integration.
- AbstractVision is the right place for higher-level convenience such as curated model-to-adapter selection and user-facing presets.
- MLX-Gen owns MLX model loading, exact route behavior, local quantized formats, capability reporting, and runtime compatibility checks.
- MLX-Gen should fail early when required local artifacts are missing so AbstractVision can surface a clear remediation message instead of starting a network transfer.

New Python integrations should prefer the route-resolved runtime planner below. Direct model classes remain available as an advanced escape hatch when an application intentionally wants backend-specific control.

[AbstractCore](https://abstractcore.ai/) can expose OpenAI-compatible endpoints backed by
AbstractVision providers, including image and video generation. In that path, MLX-Gen remains the
local Apple Silicon runtime behind the AbstractVision provider.

[AbstractFlow](https://github.com/lpalbou/abstractflow) can orchestrate these generation
capabilities visually alongside other media, text, and agent workflows. MLX-Gen remains the local
model runtime boundary: it reports capabilities, validates local artifacts, emits progress events,
and writes generated assets.

## Capability And Plan Resolution

Applications can use the same routing contract as the CLI through the public `mlxgen` package. The
resolver does not load weights. Public tasks describe media direction only:
`text-to-image`, `image-to-image`, `text-to-video`, `image-to-video`, and `video-to-video`. Image editing,
multi-reference editing, and latent image-to-image are internal modes selected by model capability,
image count, and options such as `image_strength`.

```python
from mlxgen import get_model_capabilities, get_model_validation, resolve_generation_plan, resolve_task

capabilities = get_model_capabilities(model="flux2-klein-4b")
print([capability.mode for capability in capabilities.capabilities])
# ['text-only', 'latent-img2img', 'edit-reference', 'multi-reference']

print(resolve_task(model="flux2-klein-4b").task)
# text-to-image

print(resolve_task(model="flux2-klein-4b", image_count=1).task)
# image-to-image

plan = resolve_generation_plan(model="flux2-klein-4b", image_count=1)
print(plan.task, plan.mode, plan.handler_id, plan.default_canvas_policy)
# image-to-image edit-reference flux2.edit source-aspect

latent_plan = resolve_generation_plan(
    model="flux2-klein-4b",
    image_count=1,
    i2i_mode="latent",
    has_image_strength=True,
)
print(latent_plan.task, latent_plan.mode, latent_plan.handler_id, latent_plan.canvas_policies)
# image-to-image latent-img2img flux2.generate ('source-aspect', 'exact-resize')

print(resolve_task(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers", image_count=1).task)
# image-to-video

print(resolve_task(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers", video_count=1).task)
# video-to-video

validation = get_model_validation("AbstractFramework/qwen-image-edit-2509-8bit")
print(validation.status)
# PASS
```

`--task edit` in the CLI is only a compatibility alias. Python integrations should request
`task="image-to-image"` and, when they need disambiguation, pass `i2i_mode="edit"` or
`i2i_mode="latent"`.

For ordinary image-to-image, capabilities expose the canvas contract. `default_canvas_policy` is
`"source-aspect"` for latent img2img, edit/reference I2I, and multi-reference I2I. The first input
image is the geometry anchor (`primary_image_index == 0`), and generated metadata records the
requested, source, and final dimensions.

`get_model_capabilities(...)` is route-only: it tells an application which request shapes can be
dispatched and which options are legal. `get_model_validation(...)` is release-evidence metadata:
it reports exact model/package status rows for the current validation profiles. Applications such
as AbstractVision can use capabilities for routing and validation status for UI warnings, filtering,
or release gates.

Underlying model methods are singular: each `generate_image(...)`,
`generate_video(...)`, or `restore_video_to_path(...)` call produces one output artifact for one
seed. `image_count` in the planning helpers means input-reference count for route selection, not
output count.

For multi-output generation on the unified `mlxgen generate` runtime families, use the loaded
runtime wrapper instead of rebuilding the seed loop yourself. This is serial multi-output reuse,
not tensor batching: MLX-Gen loads one model instance, runs one seed at a time, preserves exact
per-seed outputs, and can save each artifact to a distinct path with one shared progress stream.
When `overwrite=False`, the wrapper resolves the unique final path before save, so existing
targets are preserved and colliding per-seed outputs are suffixed predictably.

## Runtime Planning And Loading

For embedded workers, `mlxgen` now exposes a public runtime planner that carries the resolved route,
selected runtime class, and a stable worker-cache key base without forcing the app to map
`handler_id` values back to concrete model classes.

```python
from mlxgen import resolve_generation_runtime

runtime = resolve_generation_runtime(
    model="Qwen/Qwen-Image",
    has_control_image=True,
)
worker_key = runtime.cache_key(quantize=8, model_path="./models/qwen-image-8bit")
model = runtime.load(quantize=8, model_path="./models/qwen-image-8bit")

print(runtime.runtime_id)
# qwen.controlnet

print(runtime.plan.handler_id, runtime.plan.control_model)
# qwen.generate InstantX/Qwen-Image-ControlNet-Union:diffusion_pytorch_model.safetensors
```

If you want the convenience path that both resolves and loads the runtime, use
`load_generation_model(...)`. It returns the loaded model plus the resolved plan and cache keys.
The returned object now also owns serial multi-output execution through `generate_output(...)` and
`generate_outputs(...)`.

```python
from mlxgen import load_generation_model

loaded = load_generation_model(model="qwen-image")
results = loaded.generate_outputs(
    seeds=[101, 202, 303],
    prompt="A clean studio product photo of a ceramic teapot",
    width=1024,
    height=1024,
    guidance=1.0,
    num_inference_steps=8,
    output="teapot.png",
    save_kwargs={"export_json_metadata": True, "embed_metadata": False},
)

print([result.saved_path.name for result in results])
# ['teapot_seed_101.png', 'teapot_seed_202.png', 'teapot_seed_303.png']
```

When `output=...` is provided, `generate_outputs(...)` maps the model's in-memory `complete` event
to `generated`, then emits `save`, then emits final `complete` only after the file is written.
When `output` is omitted, the wrapper returns the in-memory artifacts and preserves the model's
original `complete` event.

Published reuse-vs-reload validation covers Qwen masked edit, FLUX.2 multi-reference edit, Wan
A14B image-to-video on a recurring short profile, and a `1024x1024` Z-Image Turbo image
generation case. See
[Python multi-output reuse validation](assets/validation/python-runtime-multi-output-2026-06-30/python_runtime_multi_output_reuse_report.md).

Qwen edit versions are distinct. `qwen-image-edit` is the original single-reference edit
checkpoint. Use `qwen-image-edit-2509` or `qwen-image-edit-2511` when you need multi-reference
capabilities and the selected package supports that route.

Negative prompts are part of the model-specific generation API. The CLI aliases
`--negative-prompt` and `--negative` both map to `negative_prompt=...` in Python. For Qwen image
edit, passing a negative prompt enables true classifier-free guidance when `guidance > 1`; when the
caller omits it, MLX-Gen applies the official blank negative-prompt behavior for Qwen edit models.

```python
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

model = QwenImageEdit(
    model_path="AbstractFramework/qwen-image-edit-8bit",
    model_config=ModelConfig.qwen_image_edit(),
)
image = model.generate_image(
    seed=9501,
    prompt="Convert the scene into a clean graphite pencil sketch while preserving layout",
    negative_prompt="color, blur, crop, text, watermark",
    image_paths=["input.png"],
    width=768,
    height=432,
    num_inference_steps=30,
    guidance=4,
)
image.save("sketch.png")
```

Contradictions fail early:

```python
from mlxgen import TaskInferenceError, resolve_task

try:
    resolve_task(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers", image_count=1)
except TaskInferenceError as exc:
    print(exc)
```

## Progress And Monitoring

MLX-Gen exposes one lightweight progress event type for applications that need to update a UI, publish job status, or integrate with an external workflow runner. The event does not include latents or model tensors.

`ProgressEvent.progress` is denoise-step progress: `step / total_steps`. Video events also carry output-frame context through `frame`, `total_frames`, and `frame_progress`.

For single-output image generation, subscribe before calling `generate_image()`:

```python
from mflux.callbacks import ProgressEvent
from mlxgen.models.z_image import ZImageTurbo


def on_progress(event: ProgressEvent) -> None:
    print(f"{event.task} {event.phase}: step {event.step}/{event.total_steps} ({event.progress:.0%})")


model = ZImageTurbo(quantize=8)
unsubscribe = model.callbacks.subscribe_progress(on_progress, task="text-to-image")
try:
    image = model.generate_image(
        seed=42,
        prompt="A clean studio product photo of a ceramic teapot",
        width=1024,
        height=1024,
        num_inference_steps=8,
    )
finally:
    unsubscribe()
```

For image-to-image, the same subscription path emits `task="image-to-image"` for latent img2img
when generation uses an input image with positive `image_strength`. Edit-conditioned backends pass
the task explicitly so applications can subscribe to `task="image-to-image"` even though those
paths do not use `image_strength`.

Wan video generation supports the same subscription path. It also accepts `progress_callback` directly on `generate_video()` when a caller wants a one-shot handler for a single run:

```python
from mflux.callbacks import ProgressEvent
from mflux.models.wan.variants import Wan2_2_TI2V


def on_progress(event: ProgressEvent) -> None:
    print(
        f"{event.phase}: frame {event.frame}/{event.total_frames}, "
        f"step {event.step}/{event.total_steps}, {event.progress:.0%}"
    )


model = Wan2_2_TI2V(model_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
video = model.generate_video(
    seed=321,
    prompt="A slow cinematic shot of a glass sphere floating above teal water",
    width=1280,
    height=704,
    num_frames=121,
    num_inference_steps=50,
    fps=24,
    progress_callback=on_progress,
)
video.save("video.mp4")
```

For Wan2.2 T2V-A14B, construct the same class with `model_config=ModelConfig.wan2_2_t2v_a14b()` or the A14B model name routed through the CLI. That same route now owns plain `video-to-video`: pass `video_path`, keep `solver="unipc"`, and use `video_strength` for the source-change amount. For Wan2.2 I2V-A14B, use `model_config=ModelConfig.wan2_2_i2v_a14b()` or the `Wan-AI/Wan2.2-I2V-A14B-Diffusers` model name and pass `image_path` to `generate_video()`. A14B boundary routing is handled internally. If both `guidance` and `guidance_2` are omitted, MLX-Gen uses the model's two-stage defaults. If `guidance` is provided and `guidance_2` is omitted, the low-noise `transformer_2` stage follows `guidance`. For Wan image-to-video, `width` and `height` are size targets; the model API resolves the final output canvas from the source image aspect ratio and model spatial multiples. For Wan video-to-video, `width` and `height` are the requested output canvas after Wan patch-multiple normalization.

Image generation emits `start` and `denoise`, followed by exactly one terminal phase: `complete`,
`failed`, or `interrupted`. For image and in-memory video APIs, `complete` means the generated
in-memory artifact is ready to return from `generate_image()` or `generate_video()`. Saving to
disk is still a separate caller action, so Python progress `complete` is not a saved-file
guarantee.

The loaded runtime wrapper augments those same progress events with `seed`, `item_index`,
`item_count`, and, when saving, `output_path`. That makes one shared callback usable for several
serial outputs without losing per-seed attribution.

Wan video generation emits `start`, `denoise`, `decode`, `convert`, and `generated` from `generate_video()`. The Wan CLI then emits `save` and reserves `complete` for a saved MP4 that passes video-health validation; it can emit `failed` instead if save or final validation fails after progress has started. Progress callback exceptions propagate to the caller, so production applications should keep handlers small and defensive.

SeedVR2 streamed video restore follows the saved-output model rather than the in-memory one: `restore_video_to_path()` emits `task="video-to-video"` and reserves `complete` for a restored MP4 whose write, metadata, optional audio copy, and optional health validation all succeeded.

For CLI integrations, prefer `mlxgen ... --json-events` over parsing human stdout. Image routes map
model `complete` to `generated`, then emit `save` and final `complete` only after the output file
is written. Wan failure events include `diagnostics_path` when a failure manifest is produced.

## Threading

MLX model instances should be treated as stateful runtime objects. Applications that multiplex user requests should serialize access to a loaded model instance unless they have tested a narrower concurrency model for that specific backend.
