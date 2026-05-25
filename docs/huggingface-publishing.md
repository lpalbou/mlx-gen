# Hugging Face Publishing

`mlxgen prepare` creates a `README.md` model card in the prepared model folder. The generated card cites the source model, mflux, MLX-Gen, the exact `mlx-gen` version used to generate the card, the quantization policy, the source license/access policy when MLX-Gen can infer it, and the default contributor attribution to [@lpalbou](https://huggingface.co/lpalbou).

Generated cards include `python -m pip install -U mlx-gen` in the usage block so Hugging Face readers can copy and paste a complete baseline command without needing uv. Repository development and release workflows still use uv.

Use `mlxgen prepare` before upload whenever you want to publish a quantized MLX-Gen folder. `mlxgen download` only fills the local Hugging Face cache and does not create an uploadable prepared folder.

## Prepare A Model Folder

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image-Edit-2511 \
  --path ./models/qwen-image-edit-2511-4bit \
  --quantize 4
```

The prepared folder contains the MLX-Gen saved-weight layout plus the generated `README.md` model card. The card includes a `Generated with mlx-gen <version>` line so published quantized checkpoints can be traced back to the package version that prepared them.

The prepared folder can then be uploaded to Hugging Face:

```sh
huggingface-cli upload AbstractFramework/qwen-image-edit-2511-4bit ./models/qwen-image-edit-2511-4bit .
```

The generated usage block uses the default `AbstractFramework/<repo-name>` repository id.

## Collections

Model-card metadata does not add a model to a Hugging Face collection. Use the Hugging Face UI or `huggingface_hub.HfApi.add_collection_item` after the model repository exists.

Use the collection slug from the Hugging Face collection URL:

```python
from huggingface_hub import HfApi

HfApi().add_collection_item(
    collection_slug="<collection-slug>",
    item_id="AbstractFramework/qwen-image-edit-2511-4bit",
    item_type="model",
    exists_ok=True,
)
```

## Compatibility Wording

For Qwen q4 checkpoints, the generated card describes the mixed q4/q8 policy used to preserve generative quality. For q8 checkpoints, the card states that the standard MLX-Gen/mflux q8 path is used.

See [Quantization](quantization.md) for the current Qwen q4 and q8 policy.

## License And Access Wording

Generated cards include license metadata for supported families where the source license is known:

- Qwen, Z-Image, and FLUX.2 Klein 4B derivatives are marked `license: apache-2.0`.
- FLUX.2 Klein 9B and FLUX.2 Klein base-9B derivatives are marked `license: other` with `license_name: flux-non-commercial-license`, source license links, and gated-access prompts.

When publishing a gated derivative, also configure the Hugging Face repository settings so the repository itself is gated. Model-card metadata alone does not enforce access control.
