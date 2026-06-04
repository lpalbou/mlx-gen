# Python Integration

MLX-Gen can be embedded directly in Python. The current runtime still exposes most model classes through the original `mflux` package layout, with `mlxgen` available as the package identity for new applications.

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

MLX-Gen is intended to be the Apple Silicon / MLX dependency for AbstractVision while AbstractVision remains a cross-platform orchestration package.

That split means:

- AbstractVision owns provider-neutral request objects, artifact storage, capability checks, and AbstractCore plugin integration.
- MLX-Gen owns MLX model loading, model-family behavior, local quantized formats, and runtime compatibility fixes.
- MLX-Gen should fail early when required local artifacts are missing so AbstractVision can surface a clear remediation message instead of starting a network transfer.

The current integration path is still model-specific Python classes. A future higher-level facade may expose explicit prepared/loaded/warmed model states, but current docs only describe the APIs that exist now.

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

For image-to-image, the same subscription path emits `task="image-to-image"` when generation uses an input image and positive `image_strength`.

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

For Wan2.2 T2V-A14B, construct the same class with `model_config=ModelConfig.wan2_2_t2v_a14b()` or the A14B model name routed through the CLI. For Wan2.2 I2V-A14B, use `model_config=ModelConfig.wan2_2_i2v_a14b()` or the `Wan-AI/Wan2.2-I2V-A14B-Diffusers` model name and pass `image_path` to `generate_video()`. A14B boundary routing is handled internally. If both `guidance` and `guidance_2` are omitted, MLX-Gen uses the model's two-stage defaults. If `guidance` is provided and `guidance_2` is omitted, the low-noise `transformer_2` stage follows `guidance`.

Image progress phases are `start`, `denoise`, and `complete`; image generation can also emit `interrupted` when a keyboard interruption is handled. Wan video generation emits `start`, `denoise`, `decode`, `convert`, and `generated` from `generate_video()`. The Wan CLI emits `save` and reserves `complete` for a saved MP4 that passes video-health validation. Progress callback exceptions propagate to the caller, so production applications should keep handlers small and defensive.

## Threading

MLX model instances should be treated as stateful runtime objects. Applications that multiplex user requests should serialize access to a loaded model instance unless they have tested a narrower concurrency model for that specific backend.
