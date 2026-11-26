import mlx.core as mx
from mlx import nn


class MemoryState:
    DISABLED = "disabled"
    INITIALIZING = "initializing"
    ACTIVE = "active"


class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        padding: int | tuple = 1,
        causal_temporal: bool = True,
        use_padding_causal: bool = False,
    ):
        super().__init__()
        self.causal_temporal = causal_temporal
        self.use_padding_causal = use_padding_causal

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kt, kh, kw = kernel_size
        self.weight = mx.zeros((out_channels, kt, kh, kw, in_channels))
        self.bias = mx.zeros((out_channels,))
        self.memory: mx.array | None = None

    def __call__(self, x: mx.array, memory_state: str = MemoryState.DISABLED) -> mx.array:
        kt = self.kernel_size[0]
        st = self.stride[0]

        if memory_state != MemoryState.ACTIVE:
            self.memory = None

        if not (self.causal_temporal and kt > 1):
            out = self._conv(x, temporal_padding=self.padding[0])
        else:
            frames = self._causal_frames(x, memory_state)
            if st != kt and memory_state != MemoryState.DISABLED:
                # A copy of the frames the next chunk needs, not a view that would pin this whole padded input.
                self.memory = mx.stack(frames[st - kt :], axis=2)
            if self._runs_by_taps(x):
                out = self._conv_by_taps(frames)
            else:
                out = self._conv(mx.stack(frames, axis=2), temporal_padding=0)
        # MLX frees a convolution's scratch only when its command buffer completes; evaluating each convolution
        # keeps one convolution's scratch alive instead of that of every convolution sharing a buffer. The memory is
        # evaluated too, so that it does not keep this input alive as a pending graph.
        mx.eval(out, self.memory)
        return out

    def _conv(self, x: mx.array, temporal_padding: int) -> mx.array:
        x = x.transpose(0, 2, 3, 4, 1)
        out = mx.conv_general(x, self.weight, stride=self.stride, padding=(temporal_padding, *self.padding[1:]))
        out = out + self.bias
        return out.transpose(0, 4, 1, 2, 3)

    def _conv_by_taps(self, frames: list[mx.array]) -> mx.array:
        # MLX 0.32 runs this 3D convolution as one batched 2D convolution per temporal tap, summed in tap order, and
        # keeps every tap's scratch until the whole call completes. The same sums, one output frame at a time and
        # evaluated tap by tap, give the same bits with the scratch of a single tap alive.
        kt = self.kernel_size[0]
        outputs = []
        for t in range(len(frames) - kt + 1):
            acc = None
            for k in range(kt):
                tap = mx.conv_general(frames[t + k].transpose(0, 2, 3, 1), self.weight[:, k], padding=self.padding[1:])
                acc = tap if acc is None else acc + tap
                if k < kt - 1:
                    mx.eval(acc)
            outputs.append(acc + self.bias)
            mx.eval(outputs[-1])
        return mx.stack(outputs, axis=1).transpose(0, 4, 1, 2, 3)

    def _runs_by_taps(self, x: mx.array) -> bool:
        # The cases verified bit-identical to MLX's single 3D call on MLX 0.32.5 (not MLX's own rule): batch 1, unit
        # stride, input channels a multiple of 16, output channels at most 16 or a multiple of 16, and frames of at
        # least 4096 pixels. Batch 2, output channels such as 17 to 31 or 100, and smaller frames differ. Every other
        # convolution keeps the single call. These cases were measured on MLX 0.32 (SeedVR2Util.VIDEO_VAE_MEMORY_MLX_RANGE):
        # before widening that range, re-check them with tests/seedvr2/test_seedvr2_causal_conv3d_exact.py and
        # re-measure the memory table.
        out_channels = self.weight.shape[0]
        return (
            x.shape[0] == 1
            and self.stride == (1, 1, 1)
            and x.shape[1] % 16 == 0
            and (out_channels <= 16 or out_channels % 16 == 0)
            and x.shape[3] * x.shape[4] >= 4096
        )

    def _causal_frames(self, x: mx.array, memory_state: str) -> list[mx.array]:
        frames = [x[:, :, t] for t in range(x.shape[2])]
        if self.memory is not None and memory_state == MemoryState.ACTIVE:
            memory = self.memory.astype(x.dtype)
            return [memory[:, :, t] for t in range(memory.shape[2])] + frames
        causal_pad = (2 * self.padding[0]) if self.use_padding_causal else self.kernel_size[0] - 1
        return [frames[0]] * causal_pad + frames
