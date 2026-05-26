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
| Z-Image / Z-Image Turbo | Supported | Supported | Standard MLX quantization policy with model-specific generation defaults. |
| FIBO | Supported when source access is available | Supported when source access is available | Source repositories may require access approval before download or preparation. |

MLX-Gen treats q4 quality as model-specific, not automatic. Qwen and ERNIE use mixed q4/q8 policies because fully q4 checkpoints showed unacceptable quality loss in generation validation. q8 remains the closest quantized option to BF16 when memory allows.

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

The q8 path was not changed by the mixed-q4 work. Qwen q8 uses the standard MLX-Gen/mflux quantization flow: quantizable modules are saved at 8-bit where the model layout supports MLX quantization, while VAE weights and non-quantizable layers remain BF16.

Other model families use their existing model-specific quantization predicates.

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
