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

print(latent_plan.resize_modes)
# ('resize', 'crop', 'pad') — routes with reference-pinned geometry advertise ()

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

For video artifacts, `save_kwargs` also accepts `validate_health=False` to skip the post-save
full-file health re-decode (for hosts that probe the saved file themselves; the skip is recorded
as `health_check: "skipped"` in the video metadata) and `export_json_metadata=True` to write the
`.metadata.json` sidecar carrying the resolved seed and parameters.

`load_generation_model(...)` also accepts `model_kwargs={...}`: host-provided constructor extras
forwarded to the resolved model class (merged last, and reflected in the returned `cache_key` so
hosts that dedupe loaded models see them as part of the identity). The current Wan options are
`keep_text_encoder_resident=True` (keep the UMT5 text encoder resident between generations in one
process, ~11 GB resident RAM for skipping the per-prompt reload — built for hosts that chain
scene generations) and `prompt_embed_disk_cache=False` (opt out of the exact on-disk prompt-embed
cache). Passing a kwarg the resolved model class does not accept raises `TypeError` at load time.

```python
loaded = load_generation_model(
    model="wan2.2-i2v-a14b",
    image_count=1,
    model_kwargs={"keep_text_encoder_resident": True},
)
```

Runtime memory defaults for Python hosts: the first weight load in a process applies a
machine-derived MLX buffer-cache limit (`total RAM / 8`, clamped to `[1 GiB, 8 GiB]`) unless a
limit was already applied, and announces it on stderr. Hosts that call `mx.set_cache_limit`
directly are safe: a pre-existing limit at or below half of physical RAM is recognized as
host-managed, preserved, and announced instead of overridden. Export
`MFLUX_MLX_CACHE_LIMIT_GB=<gb>` to pick the cap yourself, or `-1` to keep the cache
unlimited. Weight files loaded from HF-repo layouts are sequentially prefetched into the page
cache at load (prepared MLX-Gen packages materialize near-sequentially on their own and are
not prefetched; already-resident files are skipped, as is everything on machines where the
files exceed half of physical RAM); set `MFLUX_NO_WEIGHT_PREFETCH=1` to opt out.

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

## Outpaint And Reframe

Outpaint is a pipeline, not a single call: a conditioning canvas is built from the source and the
requested padding, the model denoises that canvas, and the route's preservation strategy decides
what happens to the source region afterwards. A request that pads both axes deeply runs as two
single-axis passes, the second on the first's output (`passes="auto"`, the default; `"1"` and
`"2"` name the count, and `session.pass_plan` / `session.passes` report the decision before any
weight loads). `run_outpaint(...)` runs the whole pipeline on a loaded runtime, so a host gets the
same canvas, the same fill and pass decisions, and the same metadata a CLI run produces.

```python
from mlxgen import load_generation_model, run_outpaint

loaded = load_generation_model(
    model="AbstractFramework/flux.2-klein-base-4b-8bit",
    image_count=1,
    has_outpaint=True,
)
results = run_outpaint(
    loaded=loaded,
    source_image="portrait.png",  # a 768x766 source, for the printed value below
    padding="0%,10%,100%,10%",
    seeds=[1234],
    prompt="Extend this portrait downward to reveal the lower part of the body.",
    guidance=4.0,
    num_inference_steps=20,
    output="extended.png",
)

print(results[0].saved_path.name)
# extended.png
print(results[0].artifact.extra_metadata["outpaint_fill"])
# neutral
```

`padding` takes the same CSS-style `top,right,bottom,left` string the CLI takes, and each side is
independent: `"0%,10%,100%,10%"` above leaves the top edge in place, and `"0,0,25%,0"` extends only
the bottom. One call can extend a single side, both sides of an axis, or all four at different
depths.

Swap the model name to move routes. Distilled FLUX.2 Klein 4B/9B run the same latent-locked
outpaint route as the base models and take `guidance=1.0`; Qwen Image Edit variants run
expanded-canvas generation with adaptive source restoration. Omit `guidance` and each model applies
its own default.

The route decides the fill contract. Pass `fill="edge" | "neutral" | "solid" | "blur"` (and
`fill_color=(r, g, b)` for `solid`) to name the canvas yourself on a route whose capability row has
`supports_outpaint_fill=true`. A route with a fixed canvas accepts only the mode it publishes as
`outpaint_default_fill_mode` and raises `OutpaintError` for anything else, naming the route.
`outpaint_contract_for_model(model=...)` answers both questions without loading weights: it returns
the route's `fill_modes`, `default_fill_mode`, `supports_fill_option`, `recommended_lora`, and
`preservation`.

When you own the seed loop, the save step, or want to inspect the decision before spending a
denoise, build the session yourself. `prepare_outpaint(...)` loads no model weights: it resolves the
route contract, chooses the fill, builds the canvas, and hands back the generation geometry.

```python
from mlxgen import load_generation_model, prepare_outpaint

model_name = "AbstractFramework/flux.2-klein-base-4b-8bit"

with prepare_outpaint(
    source_image="portrait.png",  # a 768x766 source, for the printed values below
    padding="0%,10%,100%,10%",
    model=model_name,
) as session:
    print(session.fill_plan.mode, session.width, session.height)
    # neutral 928 1536
    print(session.notice)
    for warning in session.warnings:
        print(warning)

    loaded = load_generation_model(model=model_name, image_count=1, has_outpaint=True)
    generated = session.generate(
        loaded.model,
        seed=1234,
        prompt="Extend this portrait downward to reveal the lower part of the body.",
        guidance=4.0,
        num_inference_steps=20,
    )
    generated.save("extended.png")
```

Building the session before loading weights is the point of the split: `session.fill_plan` and
`session.width`/`session.height` are available for a confirmation step, and an unsupported request
raises `OutpaintError` before any model is read from disk.

`session.generate(...)` supplies the conditioning keywords the route expects and finalizes the
artifact; everything else is yours. Identify the route with `model=`, `model_config=`, a
`capability=` row from `get_model_capabilities(...)`, or a `contract=` from
`outpaint_contract_for_model(...)`. The session owns a temporary workspace for the canvas, so use it
as a context manager or call `session.close()`, and pass `workspace=...` when you want to keep the
canvas.

Generative reframe uses the same expanded canvas without a fill contract or source preservation:
`prepare_reframe(source_image=..., padding=...)` returns a session with the same `width`, `height`,
`canvas_policy`, `generate(...)`, and `finalize(...)` surface.

`generate_outputs(...)` and `generate_output(...)` also take two hooks this pipeline rides on
directly. `post_process=<callable>` runs on each artifact after generation and before it is saved,
which is where a host can composite, annotate, or record its own metadata and still have it land in
the written file. `generate_method=<callable>` replaces the model's own generate call for the run
and is called once per seed as `generate_method(seed=..., **kwargs)`; `run_outpaint(...)` passes
`session.generate` through it, so a split request denoises twice per seed and still keeps the
wrapper's multi-seed loop, progress events and save semantics. Between passes the model emits a
full progress cycle per pass under the same seed and item index.

See [Reframe and Outpaint](reframe-outpaint.md) for padding guidance and the published proof runs,
and [API and CLI](api.md#outpaint-conditioning-canvas) for the published capability fields.

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

For Wan2.2 T2V-A14B, construct the same class with `model_config=ModelConfig.wan2_2_t2v_a14b()` or the A14B model name routed through the CLI. That same route now owns public `video-to-video`: pass `video_path`, keep `solver="unipc"`, use `video_strength` for the source-change amount, and add `video_mask_path` when you want preserved regions locked to the source. For Wan2.2 I2V-A14B, use `model_config=ModelConfig.wan2_2_i2v_a14b()` or the `Wan-AI/Wan2.2-I2V-A14B-Diffusers` model name and pass `image_path` to `generate_video()`. On that i2v route, `last_image_path=...` adds an optional second anchor the clip should end near (first+last bracket conditioning, experimental on Wan 2.2 A14B; the last image maps through the same canvas and `resize_mode` as the first frame, and every other Wan route rejects it). Also on that route, `context_image_paths=[...]` extends the conditioned head with the ordered frames that FOLLOW `image_path` in the motion being continued (4, 8, or 12 frames — heads of 5, 9, or 13 fill whole 4x latent groups), so a continuation inherits real momentum instead of restarting from one frozen frame; `context_noise=...` (0-1000, ~20) optionally perturbs that head SkyReels-style. Both are experimental zero-shot behaviors measured in backlog 0102, rejected on every non-A14B-i2v route, and hosts should gate on the `supports_context_frames` capability field (introduced in schema 6; current schema 11). For chained scenes, the same i2v route also ships SVI 2.0 Pro conditioning (backlog 0103, EXPERIMENTAL): construct the model with the SVI error-recycling LoRA pair (`svi_lora_high_path=...`, `svi_lora_low_path=...`; strict key-match, fixed scale 1.0) and call `generate_video(svi_anchor_image_path=...)` for the first clip, then add `svi_motion_latent_path=...` (the previous clip's exported latent; export with `svi_motion_latent_export_path=...`) and a NEW seed per clip for continuations — one persistent anchor carries identity, the latent handover carries momentum, and assembly must drop `svi_assembly_trim_frames` frames (metadata; 5 for the default one motion latent) from each continuation clip. SVI mode replaces `image_path` and conflicts with `last_image_path`/`context_image_paths`/`video_path`; hosts gate on `supports_svi` (introduced in schema 7; current schema 11). `denoising_step_list=[1000, 750, 500, 250]` runs an exact distill timestep grid instead of a step count on text/image-to-video routes; it is mutually exclusive with `num_inference_steps` and an explicit `flow_shift`. A14B boundary routing is handled internally. If both `guidance` and `guidance_2` are omitted, MLX-Gen uses the model's two-stage defaults. If `guidance` is provided and `guidance_2` is omitted, the low-noise `transformer_2` stage follows `guidance`. For Wan image-to-video, `width` and `height` are size targets; the model API resolves the final output canvas from the source image aspect ratio and model spatial multiples. For Wan video-to-video, `width` and `height` are the requested output canvas after Wan patch-multiple normalization.

MiniMax-H3 returns the generated soundtrack on the same `GeneratedVideo` object (`video.audio`, a
`GeneratedAudio` with a `(channels, samples)` float32 `waveform` and `sample_rate`), and `save()`
muxes it into the MP4:

```python
from mflux.models.common.config import ModelConfig
from mflux.models.minimax_h3.variants import MiniMaxH3

model = MiniMaxH3(model_config=ModelConfig.minimax_h3_turbo_544p(), quantize=8)
video = model.generate_video(
    seed=42,
    prompt="A red fox trots through fresh snow in a birch forest at dawn.",
    soundscape="Soft crunch of paws in dry snow, faint wind, two distant crow caws.",
    music="Sparse piano, slow tempo.",
    progress_callback=on_progress,
)
video.save("fox.mp4", export_json_metadata=True)
```

`generate_video` also takes `width`, `height`, `num_frames` (rounded up to `17n + 5`),
`num_inference_steps` (transformer evaluations), `video_shift`, `audio_shift`,
`generate_audio=False` for a silent clip, and `image_path` for first-frame image-to-video (the canvas
then follows the keyframe's aspect ratio unless `width`/`height` are given). See [MiniMax-H3 video with audio](minimax-h3.md).

Bernini uses a dedicated renderer class and keeps references separate from first-frame images:

```python
from pathlib import Path

from mflux.models.wan.variants import BerniniRenderer

renderer = BerniniRenderer()
video = renderer.generate_video(
    seed=42,
    prompt="Animate the subject from image0 in a fixed medium shot",
    reference_image_paths=[Path("subject.png")],
    width=320,
    height=192,
    num_frames=17,
    num_inference_steps=20,
    fps=16,
    max_condition_size=256,
    clear_cache_each_step=True,
    clear_cache_each_transformer_block=True,
)
video.save("referenced.mp4", export_json_metadata=True)
```

Pass `video_path=...` as well for RV2V, or pass a video without references for V2V. Construct
without `quantize`; Bernini is BF16-only. It rejects LoRA and the ordinary Wan image/strength/mask/
continuation controls. Applications can inspect `bernini.reference-video`,
`bernini.reference-video-edit`, and `bernini.video-edit` through the same capability planner before
loading weights. These capabilities advertise executable experimental routes, not a production
promotion; see the committed official parity bundle and [Bernini-R 1.3B](bernini.md) for the exact
input, memory, and failure contract.
Set `release_denoisers_before_decode=True` only for a one-shot low-memory call: it makes that
renderer instance unusable for a second generation. The unified CLI applies it only to single-seed
low-RAM jobs.

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

## Live Preview Frames

Progress events report where a run is; live previews show what it currently looks like. Register
an in-loop callback, decode the step's latents with `PreviewDecoder`, and hand the resulting
`PIL.Image` to your interface. Resolving the decoder once, before generation, keeps the loop cheap:

```python
from mflux.models.common.preview.preview_decoder import PreviewDecoder
from mflux.utils.image_util import ImageUtil

preview_decoder = PreviewDecoder.resolve(model, mode="auto")


class LivePreview:
    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        unpacked = LatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
        decoded = (
            preview_decoder.decode(unpacked, vae=model.vae)
            if preview_decoder is not None
            else model.vae.decode(unpacked)
        )
        display(ImageUtil.to_pil_image(decoded))


model.callbacks.register(LivePreview())
```

`PreviewDecoder.resolve(..., mode="auto")` returns a tiny decoder when one is published for the
model's latent space and returns `None` otherwise, so applications should keep the full-VAE
fallback shown above. Use the latent creator belonging to the model family you loaded. See
[Generation Previews](previews.md) for supported families, measured fidelity, and decode costs.

## Threading

MLX model instances should be treated as stateful runtime objects. Applications that multiplex user requests should serialize access to a loaded model instance unless they have tested a narrower concurrency model for that specific backend.
