# Hugging Face Publishing

`mlxgen prepare` and `mflux-save` create a `README.md` model card in the prepared model folder. The generated card cites the source model, mflux, MLX-Gen, the quantization policy, and the default contributor attribution to [@lpalbou](https://huggingface.co/lpalbou).

## Prepare A Model Folder

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image-Edit-2511 \
  --path ./models/qwen-image-edit-2511-4bit \
  -q 4
```

The prepared folder can then be uploaded to Hugging Face:

```sh
huggingface-cli upload lpalbou/qwen-image-edit-2511-4bit ./models/qwen-image-edit-2511-4bit .
```

The generated usage block uses the default `lpalbou/<repo-name>` repository id. Edit that line before upload if you publish under another namespace.

## Collections

Model-card metadata does not add a model to a Hugging Face collection. Use the Hugging Face UI or `huggingface_hub.HfApi.add_collection_item` after the model repository exists.

Use the collection slug from the Hugging Face collection URL:

```python
from huggingface_hub import HfApi

HfApi().add_collection_item(
    collection_slug="<collection-slug>",
    item_id="lpalbou/qwen-image-edit-2511-4bit",
    item_type="model",
    exists_ok=True,
)
```

The recommended collection for the current published checkpoints is AbstractFramework / mlx-gen.

## Compatibility Wording

For Qwen q4 checkpoints, the generated card describes the mixed q4/q8 policy used to preserve generative quality. For q8 checkpoints, the card states that the standard MLX-Gen/mflux q8 path is used.

See [Quantization](quantization.md) for the current Qwen q4 and q8 policy.
