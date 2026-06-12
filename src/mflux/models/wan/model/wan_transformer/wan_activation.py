import math

import mlx.core as mx


class WanActivation:
    @staticmethod
    def gelu_tanh(x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        activated = 0.5 * x_float * (
            1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x_float + 0.044715 * mx.power(x_float, 3)))
        )
        return activated.astype(x.dtype)
