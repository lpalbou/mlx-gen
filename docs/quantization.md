# Quantization

MLX-Gen prepares quantized model folders with the same mflux/MLX layout used for local inference. Use `mlxgen prepare --model ... --path ... -q ...` to create those folders. They are designed for MLX-Gen and are not Diffusers or Transformers `from_pretrained()` checkpoints.

## Qwen q4

Qwen Image and Qwen Image Edit use a mixed q4/q8 policy when prepared with `-q 4`. Fully q4 Qwen checkpoints can lose coherent generative behavior, so MLX-Gen keeps only the sensitive paths at higher precision:

- q4 for most Qwen transformer attention, feed-forward, and projection linears.
- q8 for Qwen `*.img_mod_linear` transformer modulation layers.
- q4 for group64-compatible Qwen text-encoder language linears.
- q8 for group64-compatible Qwen text-encoder visual linears.
- BF16 for the VAE, norms, embeddings, and linears that are not MLX group64-compatible.

This policy applies to Qwen q4 prepared folders only. It is used for Qwen Image and Qwen Image Edit variants, including 2509 and 2511 edit checkpoints.

## q8

The q8 path was not changed by the mixed-q4 work. Qwen q8 uses the standard MLX-Gen/mflux quantization flow: quantizable modules are saved at 8-bit where the model layout supports MLX quantization, while VAE weights and non-quantizable layers remain BF16.

Other model families use their existing model-specific quantization predicates.

## Compatibility

Saved MLX-Gen folders can be loaded by MLX-Gen and by compatible mflux code that understands the same saved-weight layout and quantization predicates. They are not directly readable by Diffusers or Transformers because the files contain MLX quantization tensors and the mflux/MLX component layout.
