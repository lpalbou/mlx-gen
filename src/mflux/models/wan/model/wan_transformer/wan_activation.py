import math

import mlx.core as mx


class WanActivation:
    @staticmethod
    def gelu_tanh(x: mx.array) -> mx.array:
        return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3))))
