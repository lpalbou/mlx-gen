# Proposed: Bonsai binary 1-bit runtime support

## Metadata

- Created: 2026-05-27
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: Needs new ADR if MLX-Gen decides to ship or depend on custom low-bit Metal kernels
  outside standard MLX.

## Context

`prism-ml/bonsai-image-binary-4B-mlx-1bit` is the footprint-oriented Bonsai Image checkpoint. It
uses a FLUX.2 Klein-shaped transformer in `transformer-packed-mflux/` with packed `uint32`
weights, BF16 scales/biases, and `bits: 1`, `group_size: 128`.

## Current code reality

- MLX-Gen has a Bonsai packed transformer loader for the ternary 2-bit checkpoint.
- Local MLX 0.31.0 and isolated stock MLX 0.31.2 `mx.quantized_matmul` probes succeed for
  `bits=2, group_size=128`, but fail for `bits=1, group_size=128`.
- `mlxgen generate --model prism-ml/bonsai-image-binary-4B-mlx-1bit ...` is detected as Bonsai
  and fails early with an explicit unsupported-runtime message instead of loading through the
  ordinary FLUX.2 path.
- The binary Bonsai model card points to Prism MLX / MLX Swift forks and says upstream kernel
  support is pending.
- The ternary 2-bit Bonsai path is validated on standard MLX and tracked in
  [completed item 0003](../completed/0003_bonsai_ternary_flux2_support.md).

## Problem or opportunity

Binary Bonsai could be valuable for very low memory devices, but supporting it before the runtime
can execute 1-bit packed affine matmuls would either require adopting Prism's MLX fork or adding a
custom MLX/Metal kernel path to MLX-Gen.

## Proposed direction

Defer binary 1-bit execution until after ternary Bonsai support lands. Add detection and a clear
unsupported-runtime error as part of the ternary work, then promote this item only if one of these
conditions is true:

- upstream MLX releases supported `bits=1, group_size=128` quantized matmul on Apple Silicon;
- MLX-Gen deliberately adopts a small custom kernel path with tests and an ADR;
- Prism's fork becomes a dependency we are comfortable carrying.

## Why it might matter

The binary transformer is smaller than the ternary transformer and may matter for iPhone/iPad or
very memory-limited Macs. It is not worth compromising the package architecture before the ternary
path proves the integration.

## Promotion criteria

- Ternary Bonsai 2-bit support works through `mlxgen generate`.
- A runtime probe proves local 1-bit execution is available, or an ADR accepts custom kernel
  maintenance cost.
- A validation panel compares binary, ternary, and FLUX.2 Klein behavior at 512x512 and 1024x1024
  with the Bonsai-recommended 4-step settings.

## Validation ideas

- `mx.quantized_matmul` capability probe for `bits=1, group_size=128`.
- A 512x512, 4-step smoke generation from `prism-ml/bonsai-image-binary-4B-mlx-1bit`.
- Contact sheet comparing binary versus ternary on the same prompt/seed.
- Memory measurement after adding a proper MLX peak-memory capture path.

Latest local validation on 2026-05-27:

```sh
uv run mlxgen generate \
  --model prism-ml/bonsai-image-binary-4B-mlx-1bit \
  --prompt "A bonsai tree" \
  --width 512 \
  --height 512 \
  --steps 4 \
  --guidance 1 \
  --output validation_outputs/bonsai/binary_should_fail.png
```

The command exits with a parser error explaining that the active MLX runtime cannot execute the
required `bits=1, group_size=128` packed affine matmul. This is the intended behavior until the
runtime capability exists.

Latest stock-MLX check on 2026-05-27:

```sh
uv run --with 'mlx==0.31.2' python - <<'PY'
import mlx.core as mx
w = mx.random.normal((128, 128)).astype(mx.bfloat16)
mx.quantize(w, group_size=128, bits=1)
PY
```

The probe still fails with `ValueError: [quantize] The requested number of bits 1 is not
supported. The supported bits are 2, 3, 4, 5, 6 and 8.`

## Non-goals

- This proposal does not authorize depending on Prism's MLX fork without a separate ADR.
- This proposal does not authorize uploading modified binary weights.

## Implementation notes

Treat 1-bit as a runtime capability question first, not a model-routing problem. If the kernel is
not available, keep the user error short and actionable.
