import mlx.core as mx
from mlx import nn


class Mistral3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.eps)
        return (self.weight.astype(mx.float32) * hidden_states).astype(input_dtype)
