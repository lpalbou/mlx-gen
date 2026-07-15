# Image Edit Modes

MLX-Gen exposes one public `image-to-image` task, but different models support different edit
behaviors. It also exposes a small number of adjacent structured-control routes that use a control
image instead of a source image. Use this guide when you need to choose between latent restyling,
instruction editing, masked edit/inpaint, structured control, multi-reference composition,
generative reframe, and outpaint.

For the current model-by-model proof assets, use [Image Edit Capabilities](edit-capabilities.md).
That page answers "which exact model or package passed visual QA?" This page answers "what kind of
edit should I expect from each mode?" For the current Qwen-specific route map, use
[Qwen route matrix](qwen-route-matrix.md).

## Quick Chooser

| Goal | Best mode | What to expect |
| --- | --- | --- |
| Change the overall mood, style, or lighting of one source image | `latent-img2img` | The whole image is reinterpreted from the source latent. Good for variation and restyle; weaker for precise object edits. |
| Follow an instruction while keeping the scene layout recognizable | `edit-reference` | The source image stays active as a reference. Better composition hold than latent img2img. |
| Change only one local area and keep the rest of the frame stable | masked edit / inpaint | Use an edit-capable route with `--mask-path`. White mask pixels are repainted; black pixels are preserved. |
| Use an edge map or pose guide to fix the layout while generating from text | structured control | Use a route that advertises `supports_control_image=true` and pass `--controlnet-image-path`. This is not the same as source-image edit. |
| Use one image for structure and another for style, material, or lighting | `multi-reference` | The first image anchors geometry; later images contribute additional references. |
| Reveal more of the scene around the source image | `generative reframe` | The model generates a wider view. It may redraw parts of the source image while composing the larger scene. |
| Extend the canvas beyond the crop while trying to keep the source region stable | `outpaint` | The model fills new space around the source image. This is the closest MLX-Gen route to source-preserving extension, but it is still generative. |

Use `mlxgen capabilities --model <model>` before a long run. Not every model supports every mode.

## Latent Image-To-Image

Use `latent-img2img` when you want a variation of the whole image. MLX-Gen encodes the source
image into latents, adds noise, and denoises toward the prompt.

This mode is best for:

- changing mood or time of day;
- broad style restyles;
- texture and material reinterpretation;
- "same scene, different rendering" workflows.

This mode is weaker for:

- exact object removal;
- precise layout edits;
- near-pixel-perfect source preservation;
- extending a crop without visible reinterpretation.

`--image-strength` controls how far the output may drift. Lower values stay closer to the source.
Higher values allow a stronger restyle.

Example:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-base-9b-8bit \
  --image input.png \
  --i2i-mode latent \
  --image-strength 0.35 \
  --prompt "Make the same scene feel like blue-hour science-fiction concept art with colder shadows and sharper metallic detail." \
  --output latent-restyle.png
```

Expected result: the same scene idea and camera angle remain recognizable, but the model may
reinterpret fine details across the whole frame.

## Edit-Reference

Use `edit-reference` when your prompt is an instruction and you want the source image to remain a
reference throughout generation.

This mode is best for:

- turning an image into a sketch or painting style while keeping layout;
- changing an object's state, color, or condition;
- removing or replacing elements when the selected model is good at instruction edits;
- edits where composition stability matters more than free variation.

This mode is weaker for:

- exact pixel lock;
- arbitrary zoom-out without an edit-capable model;
- multi-image composition.

Example:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2509-8bit \
  --image input.png \
  --prompt "Turn the same scene into a clean graphite sketch while preserving the object layout and camera angle." \
  --output sketch.png
```

Expected result: the scene structure should hold more tightly than latent img2img, while the prompt
changes appearance or object state.

## Masked Edit / Inpaint

Use masked edit when the change should stay local: brighten a light source, repair one damaged
region, replace one object area, or redraw part of a subject while keeping the rest of the frame
stable.

This mode is best for:

- local repairs and replacements;
- changing one object part without recomposing the whole frame;
- preserving the original framing and background outside the edited area.

This mode is weaker for:

- global restyles;
- wide composition changes;
- multi-reference compositions.

The public contract is simple:

- pass one source image with `--image`;
- pass one mask with `--mask-path`;
- white mask pixels are repainted;
- black mask pixels are preserved.

If you omit `--mask-path` and keep the same prompt and seed, an edit-capable route is still free
to recompose the frame. The mask is what turns that global edit behavior into a localized edit.

Masked editing is currently supported on Qwen edit models, base Qwen models (native, or the
validated ControlNet control-inpaint sidecar on the exact `AbstractFramework/qwen-image-8bit`
row), Z-Image Turbo, and FLUX.2 Klein (distilled and base, with optional
masked-area reference images on the backend route).

[Masked editing](masked-editing.md) is the canonical page for the full model matrix, per-family
behavior differences (schedules, guidance, mask resampling), proof grades, and route selection
advice.

Example:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image input.png \
  --mask-path mask.png \
  --prompt "Repair the damaged hull inside the mask and keep the rest of the scene unchanged." \
  --output repaired.png
```

## Structured Control

Use structured control when you want the prompt to decide appearance but a control image to decide
layout. Typical control images are edges, sketches, pose/keypoint maps, or other explicit
structure guides.

This mode is best for:

- fixing composition with a canny or sketch guide;
- following a pose map for character layout;
- keeping the prompt flexible while anchoring geometry.

This mode is weaker for:

- source-image edits that should preserve an existing frame;
- local masked repairs;
- multi-reference edit composition.

The important workflow distinction is:

- `--image` means source-image generation or editing;
- `--controlnet-image-path` means structured text-to-image control.

Do not treat `--controlnet-image-path` as another name for `--image`. The current exact public
proof row is `AbstractFramework/qwen-image-8bit` on `qwen.control`, and the route uses the exact
InstantX union ControlNet sidecar that `mlxgen generate` injects automatically for that row. See
[Image Edit Capabilities](edit-capabilities.md) for the accepted contact sheet and command log. If
you want the plain-language difference between Qwen masked edit, Qwen structured control, and Qwen
base control-inpaint, see [Qwen localized editing](qwen-localized-editing.md) and
[Qwen route matrix](qwen-route-matrix.md).

Example:

```sh
mlxgen download --model lightx2v/Qwen-Image-Lightning --all-files

mlxgen generate \
  --model AbstractFramework/qwen-image-8bit \
  --prompt "Aesthetics art, traditional asian pagoda, elaborate golden accents, sky blue and white color palette, swirling cloud pattern, digital illustration, east asian architecture, ornamental rooftop, intricate detailing on building, cultural representation." \
  --negative "blurry, low quality, distorted, deformed, text, watermark, ugly" \
  --width 576 \
  --height 864 \
  --steps 4 \
  --guidance 1 \
  --seed 5802 \
  --controlnet-image-path canny.png \
  --lora-paths lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors \
  --lora-scales 1 \
  --output controlled.png
```

## Multi-Reference

Use `multi-reference` when you want one image to provide structure and another image to provide a
different cue such as style, material, or lighting.

In MLX-Gen, the first `--image` is the geometry anchor. Later images act as additional references.

This mode is best for:

- one image for composition plus one image for style;
- combining a structural sketch with a lighting or material reference;
- controlled compositions that need more than one source cue.

This mode is weaker for:

- exact source preservation across all inputs;
- workflows where you really want a loose whole-image variation;
- arbitrary canvas extension.

Example:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image content.png \
  --image style.png \
  --prompt "Use the first image for the scene layout and the second image for the watercolor style and warm lighting." \
  --output composition.png
```

Expected result: the first image usually defines the main composition, while the second image
influences the requested style or material treatment.

## Generative Reframe

Use generative reframe when you want a wider view than the source image already contains.

Reframe is a zoom-out style edit. MLX-Gen expands the working canvas and asks the model to generate
the larger scene. Because the model is composing a wider shot, it may redraw subject details inside
the original crop.

This mode is best for:

- revealing more background;
- turning a close crop into a wider establishing shot;
- asking the model to infer plausible missing surroundings.

This mode is not the right choice when you need:

- a near-pixel-perfect source region;
- strict outpainting with minimal source reinterpretation.

Example:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --image input.png \
  --reframe-padding "20%,40%,20%,40%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot and extend the background naturally." \
  --output reframed.png
```

Expected result: a larger, more complete view. The subject may be recomposed as part of the wider
scene.

## Outpaint

Use outpaint when the main goal is extending beyond the crop while keeping the existing source
region as stable as the backend allows.

Outpaint is still generative, but it is more source-preserving than reframe. MLX-Gen currently
uses backend-specific strategies:

- Qwen Image Edit variants use a larger conditioning canvas and adaptive source restoration.
- FLUX.2 Klein base variants use source-locked denoising with a narrow latent transition band.

This mode is best for:

- extending the image left, right, top, or bottom;
- revealing missing subject boundaries outside the original crop;
- turning a close crop into a wider shot without intentionally recomposing the center.

This mode is not an exact guarantee of:

- identical source pixels;
- native masked fill/inpaint semantics;
- zero reinterpretation at the source boundary.

Example:

```sh
mlxgen generate \
  --model black-forest-labs/FLUX.2-klein-base-9B \
  --image input.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close crop into a wider realistic shot. Complete the missing subject and background outside the original frame." \
  --steps 20 \
  --guidance 4 \
  --output outpaint.png
```

Expected result: the newly added space is generated around the source crop. The center should stay
more stable than in reframe, especially on the current FLUX.2 base route, but the result is still
not a literal source-paste guarantee.

## Practical Advice

- Use `latent-img2img` for whole-image mood and style variation.
- Use `edit-reference` for instructions that should keep the scene layout recognizable.
- Use structured control when the prompt should define appearance but a control image should define layout.
- Use `multi-reference` when one image is not enough to describe the target.
- Use `reframe` when you want a wider composition and accept generative recomposition.
- Use `outpaint` when you want extension around the crop and source stability matters.

For exact route support, use `mlxgen capabilities --model <model>`.
For visual release evidence on exact models and packages, use
[Image Edit Capabilities](edit-capabilities.md) and [Reframe and Outpaint](reframe-outpaint.md).
