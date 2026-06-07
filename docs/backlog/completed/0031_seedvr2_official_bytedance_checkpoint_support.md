# Completed: SeedVR2 official ByteDance checkpoint and package support

## Metadata

- Created: 2026-06-07
- Status: Completed
- Completed: 2026-06-07

## Final status

MLX-Gen now uses the official `ByteDance-Seed/SeedVR2-3B` checkpoint for the `seedvr2` and
`seedvr2-3b` aliases and the official `ByteDance-Seed/SeedVR2-7B` checkpoint for `seedvr2-7b`.
It supports direct official `.pth` / `.pt` loading where available and can prepare reusable
SeedVR2 q8/q4 MLX-Gen packages. The 3B source/q8/q4 and 7B source/q8/q4 rows passed the same 5x
upscale profile from a `133x113` source to a `658x560` output.

## ADR status

- Governing ADRs:
  - [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
  - [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None. This work applies the existing runtime-smoke and no-silent-fallback rules to
  SeedVR2 source-layout resolution. It does not need a new durable policy unless later work changes
  the package-wide model-source contract.

## Context

MLX-Gen exposes SeedVR2 image super-resolution through `mlxgen upscale`; the compatibility
`mflux-upscale-seedvr2` entry point remains available for older scripts. The
official `ByteDance-Seed/SeedVR2-3B` repository ships PyTorch checkpoint files:

- `seedvr2_ema_3b.pth`
- `ema_vae.pth`
- `pos_emb.pt`
- `neg_emb.pt`

The route now supports the official 3B source layout directly and can also load prepared MLX-Gen
packages generated from that source.

## Current code reality

- `src/mflux/models/common/config/model_config.py` points `seedvr2-3b` at
  `ByteDance-Seed/SeedVR2-3B`.
- `src/mflux/models/seedvr2/weights/seedvr2_weight_definition.py` selects official source,
  prepared-package, or explicit compatibility layouts from the resolved files.
- `src/mflux/models/common/weights/loading/weight_loader.py` can load direct PyTorch checkpoint and
  tensor files where SeedVR2 needs them.
- `src/mflux/models/seedvr2/model/seedvr2_text_encoder/text_embeddings.py` can use the official
  `pos_emb.pt` tensor.

## Problem

SeedVR2 should load the official checkpoint layout when the source model is cached, fail clearly if
required official files are missing, and support reusable q8/q4 MLX-Gen packages prepared from the
official 3B source.

## What we want to do

Add official ByteDance SeedVR2 3B source support as a targeted compatibility layer:

- load official `.pth` transformer and VAE files when that layout is selected;
- load the official positive text embedding when available;
- preserve runtime `--quantize 8` and `--quantize 4` behavior for the official source path;
- add `mlxgen prepare` support for reusable SeedVR2 q8/q4 packages;
- add focused tests and a real small smoke run.

## Why

The official checkpoint is the authoritative upstream source. Supporting it directly reduces
dependency on a conversion repo, makes provenance clearer, and lets future q8/q4 MLX-Gen packages
be prepared from the upstream model rather than from an intermediate conversion.

## Requirements

- `mlxgen upscale --model ByteDance-Seed/SeedVR2-3B ...` must use the requested official
  source when the snapshot is cached.
- Missing official files must produce actionable file errors, not a silent switch to another model.
- The implementation must stay local to SeedVR2 loader/component handling unless a small shared
  loader extension is clearly reusable.
- Unit tests must cover official file resolution, torch checkpoint directory loading, and current
  safetensors compatibility.
- A model-backed smoke run must produce a real image artifact before completion.

## Suggested implementation

- Add a small loader mode for component directories containing PyTorch checkpoints.
- Let SeedVR2 weight definition choose between safetensors and official PyTorch file names based on
  the resolved source directory contents.
- Store official `pos_emb.pt` on the `SeedVR2` instance when present; otherwise use the existing
  bundled safetensors embedding.
- Add resolver tests for official handles and local official directories.
- Add focused loader tests with small synthetic `.pth` and `.pt` fixtures.

## Scope

- Official `ByteDance-Seed/SeedVR2-3B` and `ByteDance-Seed/SeedVR2-7B` source loading.
- Runtime q8/q4 smoke checks on the official 3B path when memory permits.
- Reusable q8/q4 package preparation, package-card generation, and source/q8/q4 package
  validation.
- Documentation and completion evidence for the new source layout.

## Non-goals

- Do not port unrelated SeedVR2 7B or sharp behavior unless the 3B implementation naturally shares
  safe loader support.

## Dependencies and related tasks

- [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
- [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- [Completed item 0030](../completed/0030_seedvr2_upscale_smoke_and_metadata.md)
- `src/mflux/models/seedvr2/`
- `src/mflux/models/common/weights/loading/`

## Expected outcomes

- Official 3B source loading works without relying on the `numz` safetensors conversion.
- `seedvr2` and `seedvr2-3b` resolve to the official 3B source.
- `seedvr2-7b` resolves to the official 7B source.
- q8 and q4 reusable packages pass smoke validation.
- User-facing docs explain the official source path and published q8/q4 package path plainly.

## Validation

- Focused unit tests for loader and SeedVR2 resolver behavior.
- Real smoke command with `ByteDance-Seed/SeedVR2-3B`, a tiny source image, and a small scale-factor
  output.
- Optional q8/q4 smoke runs using the same command profile.

## Progress checklist

- [x] Add official checkpoint layout detection.
- [x] Add `.pth` / `.pt` component-directory loading.
- [x] Wire official positive embedding.
- [x] Add official regular 7B checkpoint layout detection.
- [x] Add focused tests.
- [x] Run model-backed official 3B smoke validation.
- [x] Add SeedVR2 `mlxgen prepare` support.
- [x] Validate prepared q8/q4 packages.
- [x] Update docs and completion report.

## Guidance for the implementing agent

Prefer targeted loader and saver extensions local to SeedVR2. Do not silently switch model identity:
the alias source, resolved package path, and metadata should make the selected model clear.

## Completion report

### Summary

MLX-Gen now accepts the official `ByteDance-Seed/SeedVR2-3B` and
`ByteDance-Seed/SeedVR2-7B` checkpoint layouts through `mlxgen upscale`. The `seedvr2` and
`seedvr2-3b` aliases resolve to the official 3B source model; `seedvr2-7b`
resolves to the official regular 7B source model. Users can run the 3B source model directly:

```sh
mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-3B \
  --image-path tests/resources/low_res.jpg \
  --resolution 2x \
  --seed 42 \
  --metadata \
  --output validation_outputs/seedvr2_official_2026_06_07/low_res_official_seedvr2_3b_base_2x.png
```

### Files and symbols touched

- `src/mflux/models/common/weights/loading/weight_loader.py`
  - added directory loading for PyTorch checkpoint and tensor files;
  - made direct `torch_checkpoint` loading handle nested dicts and BF16 tensors consistently.
- `src/mflux/models/seedvr2/weights/seedvr2_weight_definition.py`
  - added the official 3B and regular 7B `.pth` component layouts and prepared-package layouts.
- `src/mflux/models/seedvr2/seedvr2_initializer.py`
  - resolves the source root before choosing the SeedVR2 weight definition;
  - records explicit requested source handles in runtime metadata without mutating cached defaults.
- `src/mflux/models/seedvr2/model/seedvr2_text_encoder/text_embeddings.py`
  - can prepare an official `pos_emb.pt` tensor with the same batch shape as the bundled embedding.
- `src/mflux/models/seedvr2/variants/upscale/seedvr2.py`
  - uses an official source embedding when the loaded checkpoint provides one;
  - can save reusable SeedVR2 packages through `mlxgen prepare`.
- `src/mflux/models/seedvr2/cli/seedvr2_upscale.py`
  - preserves explicit official Hugging Face handles as source paths instead of falling back to the
    default conversion route.
  - rejects unsupported Hugging Face-style SeedVR2 handles before model loading and preserves
    recognized `AbstractFramework/seedvr2-*` package handles through weight resolution.
- `src/mflux/models/common/download_policy.py`
  - SeedVR2 missing-cache hints include the supported download/prepare flow.
- Docs updated:
  - `docs/upscaling.md`
  - `docs/faq.md`
  - `docs/getting-started.md`

### Validation

Focused tests:

```sh
uv run pytest \
  tests/weights/test_seedvr2_official_checkpoint_loading.py \
  tests/arg_parser/test_seedvr2_upscale_argparser.py \
  tests/image_generation/test_seedvr2_upscale_metadata.py \
  tests/resolution/test_download_policy.py \
  -q
```

Result: `30 passed`.

Fast suite:

```sh
make test-fast
```

Result: `505 passed, 408 deselected`.

Official source smokes:

```sh
/usr/bin/time -l uv run mflux-upscale-seedvr2 \
  --model ByteDance-Seed/SeedVR2-3B \
  --image-path tests/resources/low_res.jpg \
  --resolution 2x \
  --seed 42 \
  --metadata \
  --replace \
  --output validation_outputs/seedvr2_official_2026_06_07/low_res_official_seedvr2_3b_base_2x.png

/usr/bin/time -l uv run mflux-upscale-seedvr2 \
  --model ByteDance-Seed/SeedVR2-3B \
  --image-path tests/resources/low_res.jpg \
  --resolution 2x \
  --seed 42 \
  --quantize 8 \
  --metadata \
  --replace \
  --output validation_outputs/seedvr2_official_2026_06_07/low_res_official_seedvr2_3b_q8_2x.png

/usr/bin/time -l uv run mflux-upscale-seedvr2 \
  --model ByteDance-Seed/SeedVR2-3B \
  --image-path tests/resources/low_res.jpg \
  --resolution 2x \
  --seed 42 \
  --quantize 4 \
  --metadata \
  --replace \
  --output validation_outputs/seedvr2_official_2026_06_07/low_res_official_seedvr2_3b_q4_2x.png
```

All three commands produced `640x384` outputs from the `320x192` source. Metadata now records
`"model": "ByteDance-Seed/SeedVR2-3B"` for the official source rows.

Prepared-package smoke:

```sh
/usr/bin/time -l uv run mflux-upscale-seedvr2 \
  --model AbstractFramework/seedvr2-3b-8bit \
  --image-path docs/assets/upscaling/seedvr2-5x-source.jpg \
  --resolution 5x \
  --seed 42 \
  --metadata \
  --replace \
  --output validation_outputs/seedvr2_quantized_packages_2026_06_07/seedvr2_3b_q8_package_5x.png

/usr/bin/time -l uv run mflux-upscale-seedvr2 \
  --model AbstractFramework/seedvr2-3b-4bit \
  --image-path docs/assets/upscaling/seedvr2-5x-source.jpg \
  --resolution 5x \
  --seed 42 \
  --metadata \
  --replace \
  --output validation_outputs/seedvr2_quantized_packages_2026_06_07/seedvr2_3b_q4_package_5x.png
```

The q8 and q4 package commands produced `658x560` outputs from the `133x113` source.

Visual proof:

- `validation_outputs/seedvr2_official_2026_06_07/seedvr2_official_3b_source_q8_q4_contact_sheet.jpg`
- `validation_outputs/seedvr2_quantized_packages_2026_06_07/seedvr2_3b_base_q8_q4_5x_contact_sheet.jpg`
- `validation_outputs/seedvr2_7b_quantized_packages_2026_06_07/seedvr2_7b_source_q8_q4_5x_contact_sheet.jpg`

### Behavior changes

- `seedvr2`, `seedvr2-3b`, and `ByteDance-Seed/SeedVR2-3B` use the official SeedVR2 3B source path.
- `seedvr2-7b` and `ByteDance-Seed/SeedVR2-7B` use the official regular SeedVR2 7B source path.
- Runtime `--quantize 8` and `--quantize 4` work with the official source path at smoke level.
- `mlxgen prepare` can create reusable q8/q4 SeedVR2 packages with generated model cards.
- Published package handles can be downloaded with `mlxgen download` and used with
  `mlxgen upscale`.

### Residual risks and follow-ups

- This item validates official 3B and regular official 7B. `7B-sharp` is not claimed here.
- q4 passed the documented 5x validation profile. Broader image-set validation remains useful
  before making general quality claims across all source-image types.
