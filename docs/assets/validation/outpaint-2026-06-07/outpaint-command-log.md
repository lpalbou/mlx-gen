# Outpaint Validation Commands

These commands produced the 2026-06-07 outpaint validation assets. `--outpaint-padding` creates an
expanded canvas from one source image and uses the selected edit model to generate the larger view.
Padding values use CSS-style `top,right,bottom,left` order; for example `0,25%,0,25%` extends the
right and left sides only.
After generation, MLX-Gen applies an adaptive source blend only when the generated source window is
still close to the original source; otherwise it keeps the generated canvas to avoid ghosted
fragments.

The two source images used for this profile are included in this folder as
`source-a-isolated-beacon.png` and `source-b-cropped-starship.png`.

## FLUX.2 Background Extension

```sh
uv run mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --image docs/assets/validation/outpaint-2026-06-07/source-a-isolated-beacon.png \
  --outpaint-padding "0,25%,0,25%" \
  --prompt "Outpaint this image into a wider cinematic snowy plain. Keep the beacon exactly where it is and fill the new left and right canvas with consistent snow, low horizon, soft blue sky, and matching light. No text, no frame, no duplicate beacon." \
  --steps 12 \
  --guidance 1 \
  --seed 7121 \
  --metadata \
  --replace \
  --output validation_outputs/outpaint_2026_06_07/flux2_outpaint_a_background.png
```

## FLUX.2 Cropped Starship

```sh
uv run mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --image docs/assets/validation/outpaint-2026-06-07/source-b-cropped-starship.png \
  --outpaint-padding "5%,35%,5%,35%" \
  --prompt "Outpaint this close cropped starship image into a wider view that reveals the full spacecraft in the snowy canyon. Continue the missing left and right parts of the hull outside the original frame, extend the ice cliffs and snow field naturally, preserve the original lighting and design, no text, no border." \
  --steps 12 \
  --guidance 1 \
  --seed 7122 \
  --metadata \
  --replace \
  --output validation_outputs/outpaint_2026_06_07/flux2_outpaint_b_cropped_starship.png
```

## Qwen Image Edit 2511 Background Extension

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/validation/outpaint-2026-06-07/source-a-isolated-beacon.png \
  --outpaint-padding "0,25%,0,25%" \
  --prompt "Outpaint the source into a wider realistic snowy plain. Keep the central beacon unchanged in position and scale. Fill only the newly added left and right space with continuous snow texture, a low horizon, pale winter sky, and matching soft daylight. Do not add another beacon, text, frame, or border." \
  --negative "duplicate beacon, text, border, frame, hard seam, split image, collage, distorted beacon" \
  --steps 24 \
  --guidance 4 \
  --seed 7131 \
  --metadata \
  --replace \
  --output validation_outputs/outpaint_2026_06_07/qwen2511_q8_outpaint_a_background.png
```

## Qwen Image Edit 2511 Cropped Starship

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/validation/outpaint-2026-06-07/source-b-cropped-starship.png \
  --outpaint-padding "5%,35%,5%,35%" \
  --prompt "Outpaint this close cropped starship image into a wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing central spacecraft surface consistent, and complete the missing hull, tail, engines, snow field, and ice cliffs in the newly added space. Preserve the same lighting and camera angle. No text, no frame, no border, no duplicate ship." \
  --negative "text, border, frame, hard seam, split image, collage, duplicate spacecraft, duplicated mountains, repeated mountain peaks, distorted engines, melted hull, blurry ship" \
  --steps 24 \
  --guidance 4 \
  --seed 7132 \
  --metadata \
  --replace \
  --output validation_outputs/outpaint_2026_06_07/qwen2511_q8_outpaint_b_cropped_starship.png
```
