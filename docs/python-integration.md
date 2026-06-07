# Python Integration

MLX-Gen can be embedded directly in Python. The current runtime still exposes most model classes through the original `mflux` package layout, and new applications can import the `mlxgen` helper APIs documented below.

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
cross-platform orchestration package inside the wider
[AbstractFramework](https://abstractframework.ai/) ecosystem.

That split means:

- AbstractVision owns provider-neutral request objects, artifact storage, capability checks, and AbstractCore plugin integration.
- MLX-Gen owns MLX model loading, model-family behavior, local quantized formats, and runtime compatibility fixes.
- MLX-Gen should fail early when required local artifacts are missing so AbstractVision can surface a clear remediation message instead of starting a network transfer.

The current integration path is still model-specific Python classes. A future higher-level facade may expose explicit model lifecycle states, but current docs only describe the APIs that exist now.

[AbstractFlow](https://github.com/lpalbou/abstractflow) can orchestrate these generation
capabilities visually alongside other persistent agentic tasks. MLX-Gen remains the local model
runtime boundary: it reports capabilities, validates local artifacts, emits progress events, and
writes generated assets.

## Capability And Plan Resolution

Applications can use the same routing contract as the CLI through the public `mlxgen` package. The
resolver does not load weights. Public tasks describe media direction only:
`text-to-image`, `image-to-image`, `text-to-video`, and `image-to-video`. Image editing,
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

For image generation, subscribe before calling `generate_image()`:

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

For Wan2.2 T2V-A14B, construct the same class with `model_config=ModelConfig.wan2_2_t2v_a14b()` or the A14B model name routed through the CLI. For Wan2.2 I2V-A14B, use `model_config=ModelConfig.wan2_2_i2v_a14b()` or the `Wan-AI/Wan2.2-I2V-A14B-Diffusers` model name and pass `image_path` to `generate_video()`. A14B boundary routing is handled internally. If both `guidance` and `guidance_2` are omitted, MLX-Gen uses the model's two-stage defaults. If `guidance` is provided and `guidance_2` is omitted, the low-noise `transformer_2` stage follows `guidance`. For Wan image-to-video, `width` and `height` are size targets; the model API resolves the final output canvas from the source image aspect ratio and model spatial multiples.

Image progress phases are `start`, `denoise`, and `complete`; image generation can also emit `interrupted` when a keyboard interruption is handled. Wan video generation emits `start`, `denoise`, `decode`, `convert`, and `generated` from `generate_video()`. The Wan CLI emits `save` and reserves `complete` for a saved MP4 that passes video-health validation. Progress callback exceptions propagate to the caller, so production applications should keep handlers small and defensive.

## Threading

MLX model instances should be treated as stateful runtime objects. Applications that multiplex user requests should serialize access to a loaded model instance unless they have tested a narrower concurrency model for that specific backend.
