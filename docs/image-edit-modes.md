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
| Extend the canvas beyond the crop while trying to keep the source region stable | `outpaint` | The model fills new space around the source image, starting from a conditioning canvas you can select on FLUX.2 Klein routes. This is the closest MLX-Gen route to source-preserving extension, but it is still generative. |

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

Geometry follows the shared image-to-image contract: the default `--canvas-policy source-aspect`
keeps the output ratio close to the source, and `--resize-mode` (`resize` | `crop` | `pad`)
chooses how source pixels map onto a canvas whose ratio differs. See
[docs/api.md](api.md) for the full contract.

Edits are geometry-stable across iterations: the source image is conditioned at the resolved
generation canvas, so the subject keeps its position and proportions when you feed an edit's
output back in as the next edit's input, at any requested output size. This holds for explicit
`--width`/`--height` values as well as the automatic canvas. For iterative workflows, keep
feeding the generated file forward unchanged; resizing outputs to sizes that are not multiples
of 16 between steps reintroduces a small resampling distortion outside the pipeline.

Attribute changes can spill beyond the instructed region: instruction-driven editors regenerate
the whole image, so a color or style instruction can pull semantically adjacent content with it
(for example, a tie-color change tinting the suit). Pin the attributes you want kept by naming
them in the instruction — "change the tie color to red, keeping the suit exactly the same blue
color and everything else unchanged" — which measurably holds off-target colors in place. For a
hard guarantee, use masked editing (`--mask-path`), which locks pixels outside the mask to the
source.

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

Outpaint is still generative, but it is more source-preserving than reframe. MLX-Gen uses
backend-specific strategies, and each route publishes which one it uses as `outpaint_preservation`:

- Qwen Image Edit variants use a larger conditioning canvas and adaptive source restoration.
- Every FLUX.2 Klein model — distilled 4B/9B and base 4B/9B — uses source-locked denoising with a
  narrow latent transition band. Distilled Klein runs at guidance 1.0, base Klein at 4.0.

Padding is independent per side, so one call can extend a single edge, both edges of an axis, or
all four at different depths. See
[Expanding On Any Side](reframe-outpaint.md#expanding-on-any-side) for the coverage sheets across
three source aspect ratios.

This mode is best for:

- extending the image left, right, top, or bottom, in any combination;
- revealing missing subject boundaries outside the original crop;
- turning a close crop into a wider shot without intentionally recomposing the center.

This mode is not an exact guarantee of:

- identical source pixels;
- native masked fill/inpaint semantics;
- zero reinterpretation at the source boundary.

On the FLUX.2 Klein route the source region is decoded from latents rather than pasted back, so it
is reproduced, not preserved bit-for-bit. On the Qwen route the paste back is conditional: it
applies only while the generated source window still matches the source. When a region must stay
untouched, use masked editing (`--mask-path`, see [Masked editing](masked-editing.md)).

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

Expected result: the newly added space is generated around the source crop. The crop is held in
latent space while the new area is generated and then pasted back over the result, so its interior
is your original pixels; a narrow transition band along the seam is regenerated so the new area
blends in.
[Reframe and Outpaint](reframe-outpaint.md#what-each-model-produces) runs every supported route on
one source and one padding value so you can compare speed and how far the source moved before
choosing.

### The Conditioning Canvas

Outpaint pastes your source onto a larger canvas and asks the model to complete the added area, so
what fills that area before denoising decides much of what you get back. On FLUX.2 Klein routes —
distilled and base alike — you choose it with `--outpaint-fill`:

| Mode | What it paints | What to expect |
| --- | --- | --- |
| `auto` (default) | Picks one of the modes below from the padding depth and the loaded adapter, and prints which and why. | A sensible canvas without thinking about it. |
| `edge` | Stretches the source border strip outward. | A continuation of the texture already at the border. Deeper than that strip covers, it reads as directional streaks. |
| `neutral` | A flat per-side border color sampled from the source. | Nothing for the model to continue, so it generates new subject matter, without a hard color step at the seam. |
| `solid` | One flat color, from `--outpaint-fill-color` (`R,G,B` or `#rrggbb`). | An exact canvas color, for example the one an adapter was trained on. |
| `blur` | A blurred, scaled copy of the source. | A soft background suggestion rather than a blank one. |

Two practical rules follow from that:

- **Match the fill to the padding depth.** Edge fill continues a texture; it does not invent one.
  For a small border it is the better canvas. For a lot of new space — continuing a scene well beyond
  the source frame, say — a blank canvas plus a descriptive prompt is what generates new
  subject matter, and `auto` switches to it on its own.
- **Extend in steps rather than in one jump.** Outpaint cost scales with the expanded canvas and
  attention cost grows faster than canvas area, so two moderate passes are cheaper and generally
  better than one large pass. This is also the route on Qwen Image Edit, which always builds an
  edge-extended canvas and takes no fill option.

Naming the mode explicitly:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-base-4b-8bit \
  --image photo.png \
  --outpaint-padding "0%,10%,100%,10%" \
  --outpaint-fill neutral \
  --prompt "Extend this photo downward: the same scene continues, same lighting, same background." \
  --steps 20 \
  --guidance 4 \
  --seed 1234 \
  --output extended.png
```

`--outpaint-padding` computes the output size from the source and the padding, so do not pass
`--width`, `--height`, or `--canvas-policy` with it. For the reasoning behind `auto`, the printed
run line, and the published capability contract, see
[Reframe and Outpaint](reframe-outpaint.md#the-conditioning-canvas).

## Practical Advice

- Use `latent-img2img` for whole-image mood and style variation.
- Use `edit-reference` for instructions that should keep the scene layout recognizable.
- Use structured control when the prompt should define appearance but a control image should define layout.
- Use `multi-reference` when one image is not enough to describe the target.
- Use `reframe` when you want a wider composition and accept generative recomposition.
- Use `outpaint` when you want extension around the crop and source stability matters.
- Use `--outpaint-fill` to match the conditioning canvas to how much space you are adding: edge fill
  for a small border, a blank canvas for a lot of new subject matter.

For exact route support, use `mlxgen capabilities --model <model>`.
For visual release evidence on exact models and packages, use
[Image Edit Capabilities](edit-capabilities.md) and [Reframe and Outpaint](reframe-outpaint.md).
