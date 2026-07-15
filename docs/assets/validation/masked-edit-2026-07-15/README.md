# Masked Edit Expansion Proof Bundle (2026-07-15)

Model-backed proof runs for the masked-edit routes added on 2026-07-15: native base-Qwen
masked edit (`qwen.base-inpaint`) and Z-Image non-turbo native inpaint (`z-image.inpaint`
on non-turbo rows). One shared case: insert a red glasses case into a masked background
region of a 717x403 product photo (`tests/resources/glasses.jpg`), everything outside the
mask preserved.

Contact sheet: `masked_edit_expansion_contact_sheet.png` (source | mask | output per row).

## Commands

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-2512-8bit \
  --image tests/resources/glasses.jpg \
  --mask-path object_mask.png \
  --prompt "Product photo of tortoiseshell reading glasses on a white background, with a small red glasses case sitting behind them on the right side." \
  --steps 20 --guidance 4 --seed 42 \
  --output qwen2512_q8_native_inpaint_object_seed42.png
```

The `AbstractFramework/qwen-image-4bit` row ran the identical command with `--model`
swapped. The Z-Image non-turbo row ran with `--model AbstractFramework/z-image-8bit
--steps 30 --guidance 4` (pass `--guidance` explicitly on non-turbo Z-Image; the default
runs guidance-free).

## Results

| Row | Route | Denoise wall time | Outside-mask mean abs diff (16 px boundary band excluded) | Inside-mask mean abs diff |
| --- | --- | --- | --- | --- |
| `AbstractFramework/qwen-image-2512-8bit` | `qwen.base-inpaint` | 55 s (17 executed steps) | 0.46/255 | 36.10/255 |
| `AbstractFramework/qwen-image-4bit` | `qwen.base-inpaint` | 71 s (17 executed steps) | 0.46/255 | 29.36/255 |
| `AbstractFramework/z-image-8bit` | `z-image.inpaint` | 98 s (30 steps) | 0.56/255 | 136.74/255 |

The native base-Qwen route runs the upstream-example warm start (strength 0.85), so 20
requested steps execute 17 denoise iterations; the sidecar metadata records
`effective_steps` and the applied warm-start strength
(`qwen2512_q8_native_inpaint_object_seed42.metadata.json`, recorded here under the original
`masked_warm_start_strength` key; the key shipped as `mask_strength` when the strength became
the tunable `--mask-strength` option the same day).

Proof grade: model-backed visual smoke on the exact rows above (single case, same seed).
Machine: M5 Max. Working copies in `validation_outputs/masked-edit-expansion-2026-07-15/`.
