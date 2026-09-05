"""fp32 exactness on Apple M5-class GPUs.

MLX 0.31 runs fp32 GEMM-class kernels (matmul, batched matmul, fused attention, im2col convolutions)
in TF32 by default on M5 (`MLX_ENABLE_TF32`), a ~8e-4 relative error per product. MiniMax-H3 keeps its
input/output heads, the timestep MLP and the whole audio VAE in fp32 on purpose, and TF32 there turns
the audio decode's 1e-4 parity into a 0.26 max error after seven alias-free upsampling stages.
Disabling TF32 costs nothing for the bf16 block stack. Measured 2026-09-04 on an M5 Max, MLX 0.31.0.

MLX reads the variable at the first fp32 GEMM dispatch, so call `disable_tf32()` before any fp32
matmul runs in the process; `mlxgen generate` does this when the MiniMax-H3 runtime initializes.
"""

import os

import mlx.core as mx

# MLX 0.31 `quantized_matmul` returns wrong values for inputs with 32768 or more rows when the scales are bf16
# (relative error 0.6-1.0; the first `rows - 32768` rows collapse to ~0). fp32 scales are exact but ~4x slower
# and promote the output to fp32, so quantized linears are applied in row chunks instead. A 1344x768 clip packs
# 37,851 rows. Reproduces with K=64, N=8, M=32769 (`tests/minimax_h3/test_h3_precision.py`).
QUANTIZED_MATMUL_MAX_ROWS = 32767


def disable_tf32() -> None:
    os.environ.setdefault("MLX_ENABLE_TF32", "0")


def rowwise(layer, x: mx.array) -> mx.array:
    """Apply a linear layer to `(B, S, C)` input in chunks of at most `QUANTIZED_MATMUL_MAX_ROWS` rows."""
    rows = x.shape[1]
    if rows <= QUANTIZED_MATMUL_MAX_ROWS:
        return layer(x)
    return mx.concatenate(
        [layer(x[:, start : start + QUANTIZED_MATMUL_MAX_ROWS]) for start in range(0, rows, QUANTIZED_MATMUL_MAX_ROWS)],
        axis=1,
    )
