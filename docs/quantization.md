# Quantization

MLX-Gen prepares quantized model folders with the same mflux/MLX layout used for local inference. Use `mlxgen prepare --model ... --path ... --quantize ...` to create those folders. They are designed for MLX-Gen and are not Diffusers or Transformers `from_pretrained()` checkpoints.

## Compatibility Summary

The current quantized-model compatibility surface is:

| Model family | q8 prepared folders | q4 prepared folders | Notes |
| --- | --- | --- | --- |
| Qwen Image | Supported | Supported with mixed q4/q8 | Applies to Qwen Image and Qwen Image 2512 text-to-image checkpoints. |
| Qwen Image Edit | Supported | Supported with mixed q4/q8 | Applies to Qwen Image Edit, 2509, and 2511 image-edit checkpoints. |
| ERNIE Image Turbo | Supported | Supported with mixed q4/q8 | Text-to-image plus experimental single-image image-to-image. Prompt Enhancer is optional and requires a full source snapshot. |
| FLUX.2 Klein | Supported | Supported | Standard MLX quantization policy. 9B derivatives follow the source gated/non-commercial access requirements. |
| Bonsai Image | Not a prepared q8 folder | Ternary 2-bit pre-packed checkpoint supported | Bonsai checkpoints are already packed MLX artifacts. Use `mlxgen download` and `mlxgen generate`; do not run `prepare`. Binary 1-bit is detected but blocked until stock MLX supports 1-bit packed affine matmul. |
| Z-Image / Z-Image Turbo | Supported | Supported | Standard MLX quantization policy with model-specific generation defaults. |
| FIBO | Supported when source access is available | Supported when source access is available | Source repositories may require access approval before download or preparation. |
| Wan2.2 | Supported with mixed q8/BF16 | Under validation | T2V-A14B full q8 collapsed to near-black static output; MLX-Gen keeps Wan conditioning/output projection linears BF16 and quantizes the bulky transformer block linears at q8. |

MLX-Gen treats low-bit quality as model-specific, not automatic. Qwen and ERNIE use mixed q4/q8 policies because fully q4 checkpoints showed unacceptable quality loss in generation validation. Bonsai uses Prism's pre-packed ternary 2-bit transformer plus a 4-bit Qwen3 text encoder rather than MLX-Gen's `prepare` flow. q8 remains the closest prepared-folder option to BF16 when memory allows.

The difference between Bonsai ternary 2-bit and MLX-Gen's mixed q4/q8 policies is mostly packaging and runtime ownership, not the quality philosophy. Both avoid blind full low-bit conversion:

| Strategy | Used by | Quality-preserving rule | Representative footprint and runtime |
| --- | --- | --- | --- |
| Mixed q4/q8 prepared folders | Qwen Image/Edit and ERNIE Image Turbo q4 folders created by `mlxgen prepare` | q4 for bulk transformer paths, q8 for empirically sensitive linears, BF16 for non-quantizable weights and selected runtime components. | ERNIE mixed q4/q8: 8.2 GiB folder, 9.34 GiB peak RSS, 7.83 s at 512px. Qwen uses the same policy shape on larger source models. |
| Pre-packed ternary 2-bit checkpoint | Bonsai Image 2-bit from Prism | The transformer is already packed at 2-bit, the Qwen3 text encoder is 4-bit, and the Flux2 VAE stays BF16. | Bonsai ternary: 3.6 GiB cached snapshot, 3.57 GiB peak RSS, 2.92 s at 512px. |

These are not model-quality rankings across unrelated models. They show the current MLX-Gen rule: use the smallest validated layout that still stays in the same visual family as a higher-precision baseline.

## Qwen q4

Qwen Image and Qwen Image Edit use a mixed q4/q8 policy when prepared with `--quantize 4`. Fully q4 Qwen checkpoints can lose coherent generative behavior, so MLX-Gen keeps only the sensitive paths at higher precision:

- q4 for most Qwen transformer attention, feed-forward, and projection linears.
- q8 for Qwen `*.img_mod_linear` transformer modulation layers.
- q4 for group64-compatible Qwen text-encoder language linears.
- q8 for group64-compatible Qwen text-encoder visual linears.
- BF16 for the VAE, norms, embeddings, and linears that are not MLX group64-compatible.

This policy applies to Qwen q4 prepared folders only. It is used for Qwen Image and Qwen Image Edit variants, including 2509 and 2511 edit checkpoints.

Representative Qwen Image 2512 512x512 validation at 15 steps:

![Qwen Image 2512 BF16, q8, full q4, and mixed q4/q8 comparison](assets/quantization/qwen-image-2512-q4-q8-comparison.png)

## q8

Qwen q8 uses the standard MLX-Gen/mflux quantization flow: quantizable modules are saved at 8-bit where the model layout supports MLX quantization, while VAE weights and non-quantizable layers remain BF16.

Other model families use their existing model-specific quantization predicates.

## Wan q8

Wan q8 uses a mixed q8/BF16 policy. A fully q8 Wan A14B layout did not preserve video quality in
validation, so MLX-Gen quantizes the bulky transformer block linears and keeps sensitive paths at
BF16:

- q8 for quantizable Wan transformer attention and feed-forward linears.
- BF16 for the Wan VAE.
- BF16 for Wan transformer `condition_embedder.*` and `proj_out`.
- BF16 for the UMT5 text encoder, scheduler metadata, tokenizer files, norms, convolutions, and
  other non-quantizable parameters.

The upstream Wan A14B source snapshots are about 118 GiB. MLX-Gen also publishes prepared BF16
folders for users who want a smaller reusable MLX-Gen package without quantizing runtime-sensitive
weights.

The current published-card validation uses small repeatable low-RAM runs on Apple Silicon. It
records the MLX allocator peak and Darwin full-process physical footprint from model init through
MP4 save and health validation. RSS is included for comparison, but physical footprint is the more
useful MLX/Metal unified-memory signal.

| Model | Package | Disk | Physical Peak | Max RSS | MLX Peak | Generation Time | Validation Profile |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Wan2.2 T2V-A14B | BF16 | 64.3 GiB | 33.0 GiB | 31.8 GiB | 27.7 GiB | 152.7 s | 384x224, 33 frames, 12 steps, 8 fps |
| Wan2.2 T2V-A14B | mixed q8/BF16 | 39.7 GiB | 20.7 GiB | 19.5 GiB | 15.5 GiB | 154.8 s | 384x224, 33 frames, 12 steps, 8 fps |
| Wan2.2 I2V-A14B | BF16 | 64.1 GiB | 33.7 GiB | 31.8 GiB | 28.2 GiB | 228.2 s | 384x384, 33 frames, 12 steps, 8 fps |
| Wan2.2 I2V-A14B | mixed q8/BF16 | 39.7 GiB | 21.5 GiB | 19.6 GiB | 15.9 GiB | 242.2 s | 384x384, 33 frames, 12 steps, 8 fps |

In these validation runs, mixed q8/BF16 cuts disk usage by about 38% versus the prepared BF16
folders and reduces full-process physical peak memory by about 36-37%. It is not currently claimed
as a speed improvement.

Full-size Wan A14B q8 generation remains a separate validation target. Model cards and public docs
should keep full-size claims tied to the exact settings that have passed MP4 health checks and
manual visual inspection.

## ERNIE Image Turbo

ERNIE Image Turbo supports BF16, q8, and q4 text-to-image generation. Use `mlxgen prepare` to create reusable q8 or q4 folders:

```sh
mlxgen prepare --model baidu/ERNIE-Image-Turbo --path ./models/ernie-image-turbo-8bit --quantize 8
mlxgen prepare --model baidu/ERNIE-Image-Turbo --path ./models/ernie-image-turbo-4bit --quantize 4
```

ERNIE q4 uses a model-specific mixed q4/q8 policy. Fully q4 ERNIE checkpoints can drift from BF16/q8 behavior, especially on text-heavy poster prompts, so MLX-Gen keeps the sensitive text-conditioning and attention-output paths at q8:

- q4 for ERNIE transformer Q/K attention projections.
- q4 for ERNIE transformer feed-forward modules.
- q8 for ERNIE transformer V/O attention projections.
- q8 for ERNIE text projection, timestep embedding, AdaLN modulation, final norm, and final projection.
- q8 for Mistral3 text-encoder and Prompt Enhancer linears.
- q8 for quantizable ERNIE VAE attention modules.
- BF16 for norms, convolutions, and other non-quantizable parameters.

Local validation on Apple Silicon with 512x512, 8 steps, guidance 1:

| Layout | Folder Size | Peak RSS | Average Generation Time | Notes |
| --- | ---: | ---: | ---: | --- |
| BF16 source generation components | ~22.4 GiB | 23.5 GiB | 6.38 s | Fastest at 512px, largest memory footprint. |
| q8 prepared folder | 12 GiB | 12.9 GiB | 7.57 s | About half the memory footprint. |
| full q4 experimental folder | 6.2 GiB | 7.2 GiB | 9.31 s | Too much visual drift on controlled poster tests; not recommended. |
| mixed q4/q8 prepared folder | 8.2 GiB | 9.34 GiB | 7.83 s | Default q4 policy; preserves BF16/q8 behavior more closely. |

At 1024x1024 with 8 steps and guidance 1, q8 generated in 84.69 s with 12.9 GiB peak RSS, and the older full-q4 experimental layout generated in 78.94 s with 7.2 GiB peak RSS. The mixed q4/q8 default should be used for new ERNIE q4 folders because the full-q4 layout does not preserve quality reliably.

On a controlled 512x512 poster prompt with seed 123, q8 stayed close to BF16 while full q4 visibly changed text color and composition. Mean absolute pixel error against q8 was 26.00 for full q4 and 12.03 for mixed q4/q8. Across a 3-seed poster repeat, full q4 averaged 33.48 MAE against q8 while mixed q4/q8 averaged 16.39 MAE.

Prepared ERNIE q8/q4 folders contain the ordinary text-to-image generation components. ERNIE Prompt Enhancer remains an optional full-source-snapshot feature and is not bundled into prepared quantized folders.

Representative ERNIE Image Turbo 512x512 validation panels:

![ERNIE Image Turbo BF16, q8, and mixed q4/q8 poster comparison](assets/quantization/ernie-image-turbo-q4-q8-comparison.png)

![ERNIE Image Turbo full q4 versus mixed q4/q8 comparison](assets/quantization/ernie-image-turbo-full-vs-mixed-q4.png)

ERNIE Image Turbo also supports experimental single-image image-to-image in MLX-Gen. The image path is encoded with the ERNIE VAE, patchified to the ERNIE denoising latent shape, normalized with the model's VAE batch-normalization statistics, and then blended with generation noise before denoising. This is an MLX-Gen extension; use it for stylization and guided variation rather than exact multi-image editing.

The same 512x512 image-to-image pencil-sketch prompt with seed 503 stayed coherent across BF16, q8, and mixed q4/q8:

![ERNIE Image Turbo image-to-image BF16, q8, and mixed q4/q8 comparison](assets/generation/ernie-image-turbo-i2i-pencil-quant-comparison.png)

ERNIE image-to-image is not a substitute for a true edit-conditioned model. It initializes ERNIE's text-to-image denoising from encoded image latents, so it is useful for fast stylization and guided variation, while Qwen Image Edit remains better when the exact source layout must be preserved. Use dimensions that match the source aspect ratio and consider 12-16 steps for stronger stylization.

Representative ERNIE q8 image-to-image versus Qwen Image Edit 2511 q4 comparisons:

![ERNIE Image Turbo q8 versus Qwen Image Edit 2511 q4 room pencil comparison](assets/generation/ernie-vs-qwen2511-room-pencil.png)

![ERNIE Image Turbo q8 versus Qwen Image Edit 2511 q4 boat watercolor comparison](assets/generation/ernie-vs-qwen2511-boat-watercolor.png)

![ERNIE Image Turbo q8 versus Qwen Image Edit 2511 q4 local wall-color edit comparison](assets/generation/ernie-vs-qwen2511-teal-wall.png)

![ERNIE Image Turbo q8 versus Qwen Image Edit 2511 q4 object replacement comparison](assets/generation/ernie-vs-qwen2511-replace-laptop.png)

In the object-replacement comparison, ERNIE can introduce the requested fruit-basket concept, but it does so by reimagining the room from its encoded latent initialization. Qwen Image Edit keeps the source image active as conditioning throughout denoising and is therefore the better choice for exact layout and local-object edits.

## Bonsai Image

Bonsai Image support is different from ordinary q4/q8 preparation. Prism publishes Bonsai as
ready-to-run MLX artifacts:

- `prism-ml/bonsai-image-ternary-4B-mlx-2bit`: supported in MLX-Gen.
- `prism-ml/bonsai-image-binary-4B-mlx-1bit`: detected, but not runnable on stock MLX through
  0.31.2 because the active runtime cannot execute the required `bits=1, group_size=128` packed
  affine matmul.

Use `mlxgen download` to cache Bonsai, then generate directly:

```sh
mlxgen download --model prism-ml/bonsai-image-ternary-4B-mlx-2bit

mlxgen generate \
  --model prism-ml/bonsai-image-ternary-4B-mlx-2bit \
  --prompt "A bonsai tree in a quiet ceramic studio, soft morning light" \
  --width 1024 \
  --height 1024 \
  --steps 4 \
  --guidance 1 \
  --seed 42 \
  --output bonsai.png
```

Do not use `mlxgen prepare` for Bonsai. The repository already contains a low-bit packed
transformer, a 4-bit Qwen3 text encoder, and a BF16 Flux2 VAE.

Local comparison against FLUX.2 Klein 4B q8 on the same prompt, seed 42, guidance 1, and 4 steps:

| Model | Disk footprint | 512px average time | 512px peak RSS | 1024px time | 1024px peak RSS | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bonsai ternary 2-bit | 3.6 GiB cached snapshot | 2.92 s | 3.57 GiB | 5.69 s | 3.60 GiB | Coherent image quality, same visual family as Klein 4B q8. |
| FLUX.2 Klein 4B q8 | 22 GiB source cache, q8 applied at runtime | 3.55 s | 9.23 GiB | 6.81 s | 9.39 GiB | Strong baseline with much higher memory footprint. |
| Bonsai binary 1-bit | 3.2 GiB cached snapshot | Not runnable | Not runnable | Not runnable | Not runnable | Waiting on stock-MLX 1-bit packed affine runtime support; latest checked stock MLX is 0.31.2. |

The 512px timing values are three cold-process repeats captured with `/usr/bin/time -l`. The Klein
baseline used on-the-fly q8 from the parent source cache because local disk space was tight during
validation.

Representative Bonsai ternary versus FLUX.2 Klein 4B q8 validation:

![Bonsai ternary 2-bit versus FLUX.2 Klein 4B q8 comparison](assets/quantization/bonsai-ternary-vs-klein4b-q8.png)

## Other Quantized Families

FLUX.2 Klein, Z-Image, Z-Image Turbo, and FIBO currently use their standard model-specific MLX quantization predicates. The following panels are representative checks, not a claim that every prompt has identical visual behavior across quantization levels.

Z-Image 512x512 validation:

![Z-Image BF16, q8, and q4 comparison](assets/quantization/z-image-q4-q8-comparison.png)

Z-Image Turbo 512x512 validation:

![Z-Image Turbo BF16, q8, and q4 comparison](assets/quantization/z-image-turbo-q4-q8-comparison.png)

FIBO 512x512 validation:

![FIBO BF16, q8, and q4 comparison](assets/quantization/fibo-q4-q8-comparison.png)

## Compatibility

Saved MLX-Gen folders can be loaded by MLX-Gen and by compatible mflux code that understands the same saved-weight layout and quantization predicates. They are not directly readable by Diffusers or Transformers because the files contain MLX quantization tensors and the mflux/MLX component layout.
