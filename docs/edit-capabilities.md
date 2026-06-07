# Image Edit Capabilities

This page summarizes the current image-to-image edit contact sheets, command logs, and model/package status for MLX-Gen. It separates
three related concepts:

- `latent-img2img`: the source image initializes latent denoising and `--image-strength` controls
  how far the output may drift.
- `edit-reference`: the source image remains active as an edit/reference condition and the prompt
  describes an instruction such as style change, object state, or scene change.
- `multi-reference`: two or more images are supplied as references for a composition.

Use `mlxgen capabilities --model <model>` to inspect route support before a run. Use the contact
sheets and status tables below when you need visual release evidence for exact source handles or
MLX-Gen optimized packages.

## Status Labels

| Status | Meaning |
| --- | --- |
| `PASS` | The row generated and the output visually satisfied the requested edit for this profile. |
| `PARTIAL` | The row generated and is usable for some work, but one requested constraint was weak. |
| `FAIL` | The row generated or routed, but the output did not satisfy the requested edit. |

The canonical validation source image is
[`docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png`](assets/examples/spaceship-snow/01_t2i_spaceship_snow.png).

## Regular Qwen Image Edit

`Qwen/Qwen-Image-Edit` is the original single-reference edit checkpoint. It supports
`edit-reference` with one input image. It does not support multi-reference composition; use
`Qwen/Qwen-Image-Edit-2509` or `Qwen/Qwen-Image-Edit-2511` for multi-reference edit routing.

![Qwen Image Edit base, q8, and q4 proof](assets/validation/i2i-edit-5x4-2026-06-05/qwen-image-edit-base-q8-q4-vl1024-contact-sheet.jpg)

| Model | Package | Capabilities validated | Result |
| --- | --- | --- | --- |
| `Qwen/Qwen-Image-Edit` | source | pencil sketch, crash edit | `PASS` |
| `AbstractFramework/qwen-image-edit-8bit` | q8 optimized variant | pencil sketch, crash edit | `PASS` |
| `AbstractFramework/qwen-image-edit-4bit` | mixed q4/q8 optimized variant | pencil sketch, crash edit | `PASS` |

These rows used a `768x432`, 30-step, guidance `4` profile with
`--scheduler flow_match_euler_discrete`.

## Qwen 2509 And FLUX.2 Matrix

The 5x4 edit validation profile tests the same spaceship source across:

- `B`: cinematic latent/style variation;
- `C`: crash edit from the source image;
- `D`: pencil sketch edit;
- `E`: multi-reference composition from the model's own pencil/crash and cinematic rows.

| Family | Exact handles/packages | Modes validated | Result summary | Contact sheet |
| --- | --- | --- | --- | --- |
| FLUX.2 Klein 4B | `black-forest-labs/FLUX.2-klein-4B`, `AbstractFramework/flux.2-klein-4b-8bit`, `AbstractFramework/flux.2-klein-4b-4bit` | `latent-img2img`, `edit-reference`, `multi-reference` | source, q8, and q4 passed B/C/D/E | [matrix](assets/validation/i2i-edit-5x4-2026-06-05/flux2-klein-4b-variant-matrix.jpg) |
| FLUX.2 Klein 9B | `black-forest-labs/FLUX.2-klein-9B`, `AbstractFramework/flux.2-klein-9b-8bit`, `AbstractFramework/flux.2-klein-9b-4bit` | `latent-img2img`, `edit-reference`, `multi-reference` | source, q8, and q4 passed B/C/D/E | [matrix](assets/validation/i2i-edit-5x4-2026-06-05/flux2-klein-9b-variant-matrix.jpg) |
| Qwen Image Edit 2509 | `Qwen/Qwen-Image-Edit-2509`, `AbstractFramework/qwen-image-edit-2509-8bit`, `AbstractFramework/qwen-image-edit-2509-4bit` | `edit-reference`, `multi-reference` | source and q8 passed B/C/D/E; q4 passed B/C/D and was partial on E | [matrix](assets/validation/i2i-edit-5x4-2026-06-05/qwen-image-edit-2509-variant-matrix.jpg) |
| Qwen Image Edit 2511 | `Qwen/Qwen-Image-Edit-2511`, `AbstractFramework/qwen-image-edit-2511-8bit`, `AbstractFramework/qwen-image-edit-2511-4bit` | `edit-reference`, `multi-reference` | source, q8, and q4 passed the 2026-06-06 pencil/crash/composition profile | [matrix](assets/validation/qwen-edit-2511-parity-2026-06-06/qwen-image-edit-2511-source-q8-q4-parity.jpg) |
| FIBO Edit | `briaai/Fibo-Edit` | Not supported through unified `mlxgen generate` | no public image-edit support in the current release; capability discovery fails closed | N/A |

### FLUX.2 Klein 4B

This matrix validates source, q8, and q4 packages on the same canonical spaceship source. The
columns cover the standardized sequence: source image, cinematic latent variation, hard-landing
edit, pencil-sketch edit, and multi-reference composition.

![FLUX.2 Klein 4B edit matrix](assets/validation/i2i-edit-5x4-2026-06-05/flux2-klein-4b-variant-matrix.jpg)

### FLUX.2 Klein 9B

This matrix validates source, q8, and q4 packages on the same canonical spaceship source. The
columns cover the same standardized sequence as Klein 4B so the two model sizes can be compared
directly.

![FLUX.2 Klein 9B edit matrix](assets/validation/i2i-edit-5x4-2026-06-05/flux2-klein-9b-variant-matrix.jpg)

### Qwen Image Edit 2509

This matrix validates the Qwen Image Edit 2509 source checkpoint plus q8 and q4 MLX-Gen optimized packages. Source
and q8 pass the full standardized edit-reference and multi-reference sequence; q4 remains partial
on the multi-reference composition row in this profile.

![Qwen Image Edit 2509 edit matrix](assets/validation/i2i-edit-5x4-2026-06-05/qwen-image-edit-2509-variant-matrix.jpg)

### Qwen Image Edit 2511

The current Qwen Image Edit 2511 proof uses the same source image across the upstream source
checkpoint, the q8 MLX-Gen package, and the q4 MLX-Gen package. The profile validates a
single-image pencil sketch, a single-image hard-landing crash edit, and a two-reference composition
from the generated pencil and crash images.

![Qwen Image Edit 2511 source, q8, and q4 parity matrix](assets/validation/qwen-edit-2511-parity-2026-06-06/qwen-image-edit-2511-source-q8-q4-parity.jpg)

### FIBO Edit

FIBO Edit is not a supported public image-edit route in MLX-Gen at the moment.
`mlxgen capabilities --model briaai/Fibo-Edit` exposes no unified generation capabilities for this
model. The dedicated compatibility command remains for maintainer parity work, but user-facing
image editing should use Qwen Image Edit, Qwen Image Edit 2509/2511, or FLUX.2 Klein routes with passing
contact sheets.

## Latent I2I Only

Some image models support latent image-to-image variation but are not edit/reference models. In the
standard spaceship profile, `Z-Image Turbo` and `ERNIE Image Turbo` q4/q8 packages passed the
single latent cinematic variation row. `Qwen Image 2512` q4/q8 ran the latent route but did not
preserve the spaceship identity for this prompt, so it is not documented here as a good edit model.

Use latent I2I for style/variation workflows, not precise composition or object-state editing:

```sh
mlxgen generate \
  --model AbstractFramework/z-image-turbo-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --i2i-mode latent \
  --image-strength 0.35 \
  --prompt "Make this same spaceship in the snow look like polished cinematic science-fiction concept art at blue hour. Preserve the exact camera angle, ship position, snowy canyon, and overall layout. Sharpen hull panels and add cold blue shadows; no crash, no damage." \
  --width 432 \
  --height 240 \
  --steps 20 \
  --seed 9201 \
  --output output.png
```

## Exact Validation Commands

The full command logs are published with the proof assets:

- [regular Qwen Image Edit command log](assets/validation/i2i-edit-5x4-2026-06-05/qwen-image-edit-command-log.md)
- [Qwen Image Edit 2511 parity command log](assets/validation/qwen-edit-2511-parity-2026-06-06/qwen-image-edit-2511-command-log.md)
- [5x4 FLUX.2 and Qwen Image Edit 2509 command log](assets/validation/i2i-edit-5x4-2026-06-05/edit-capability-command-log.md)
- [latent I2I command log](assets/validation/i2i-edit-5x4-2026-06-05/latent-i2i-command-log.md)
