import gc

import mlx.core as mx
import pytest

from mflux.models.seedvr2.model.seedvr2_vae.common.conv3d import CausalConv3d, MemoryState
from mflux.models.seedvr2.model.seedvr2_vae.vae import SeedVR2VAE
from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util

# The VAE's weights load as float16; its convolutions see bfloat16 activations after each norm and float32 ones at the
# input layer and after residual sums.
DTYPE_MIXES = {
    "bf16 input, fp16 weights": (mx.bfloat16, mx.float16),
    "fp32 input, fp16 weights": (mx.float32, mx.float16),
}


class OneCallConv3d:
    # The formulation CausalConv3d replaced: causal padding by concatenation, then one 3D convolution.
    @staticmethod
    def run(conv: CausalConv3d, x: mx.array, memory: mx.array | None) -> tuple[mx.array, mx.array | None]:
        kt, st = conv.kernel_size[0], conv.stride[0]
        if memory is not None:
            x = mx.concatenate([memory.astype(x.dtype), x], axis=2)
        else:
            pad = (2 * conv.padding[0]) if conv.use_padding_causal else kt - 1
            x = mx.concatenate([mx.repeat(x[:, :, :1], pad, axis=2), x], axis=2)
        new_memory = x[:, :, st - kt :] if st != kt else None
        out = mx.conv_general(
            x.transpose(0, 2, 3, 4, 1), conv.weight, stride=conv.stride, padding=(0, *conv.padding[1:])
        )
        return (out + conv.bias).transpose(0, 4, 1, 2, 3), new_memory

    @staticmethod
    def layer(
        channels: int,
        kernel=3,
        stride=1,
        padding=1,
        use_padding_causal=False,
        out_channels: int = 32,
        weight_dtype=mx.float16,
    ) -> CausalConv3d:
        conv = CausalConv3d(
            channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            use_padding_causal=use_padding_causal,
        )
        conv.weight = (mx.random.normal(conv.weight.shape) * 0.05).astype(weight_dtype)
        conv.bias = (mx.random.normal(conv.bias.shape) * 0.1).astype(weight_dtype)
        return conv


class TapCounter:
    def __init__(self, monkeypatch):
        self.calls = 0
        original = CausalConv3d._conv_by_taps

        def counted(conv, frames):
            self.calls += 1
            return original(conv, frames)

        monkeypatch.setattr(CausalConv3d, "_conv_by_taps", counted)


# (input channels, output channels, frame size, kernel, stride, padding, use_padding_causal, batch, takes the taps).
# The taps are taken only in the cases verified bit-identical: batch 1, unit stride, input channels a multiple of 16,
# output channels at most 16 or a multiple of 16, frames of at least 4096 pixels. Every other case keeps the single
# call and must stay identical too.
CASES = {
    "taps 32->32": (32, 32, (64, 64), 3, 1, 1, False, 1, True),
    "taps 32->32 padding causal": (32, 32, (64, 64), 3, 1, 1, True, 1, True),
    "taps padding causal, no temporal padding": (32, 32, (64, 64), 3, 1, (0, 1, 1), True, 1, True),
    "taps 16->32 64x72": (16, 32, (64, 72), 3, 1, 1, False, 1, True),
    "taps 32->3": (32, 3, (64, 64), 3, 1, 1, False, 1, True),
    "taps at exactly 4096 pixels": (32, 32, (64, 64), 3, 1, 1, False, 1, True),
    "single: 20 output channels": (32, 20, (64, 64), 3, 1, 1, False, 1, False),
    "single: batch 2": (32, 32, (64, 64), 3, 1, 1, False, 2, False),
    "single: 3 input channels": (3, 32, (64, 64), 3, 1, 1, False, 1, False),
    "single: 40x40 frames": (256, 32, (40, 40), 3, 1, 1, False, 1, False),
    "single: 3840 pixels": (32, 32, (60, 64), 3, 1, 1, False, 1, False),
    "single: 4095 pixels": (32, 32, (63, 65), 3, 1, 1, False, 1, False),
    "single: stride 2 above 4096 pixels": (32, 32, (66, 66), (3, 3, 3), (2, 2, 2), (1, 0, 0), False, 1, False),
}


@pytest.mark.fast
@pytest.mark.parametrize("mix", DTYPE_MIXES)
@pytest.mark.parametrize("case", CASES)
def test_causal_conv3d_is_bitwise_the_single_3d_convolution_across_chunks(monkeypatch, case, mix):
    cin, cout, size, kernel, stride, padding, use_padding_causal, batch, takes_taps = CASES[case]
    x_dtype, w_dtype = DTYPE_MIXES[mix]
    mx.random.seed(3)
    conv = OneCallConv3d.layer(
        cin, kernel, stride, padding, use_padding_causal, out_channels=cout, weight_dtype=w_dtype
    )
    counter = TapCounter(monkeypatch)
    # A one-frame slice in ACTIVE state, as the decoder runs its latent frames, between longer slices.
    lengths = (5, 4) if conv.stride[0] > 1 else (5, 1, 4)
    chunks = [mx.random.normal((batch, cin, length, *size)).astype(x_dtype) for length in lengths]
    memory = None
    for i, chunk in enumerate(chunks):
        state = MemoryState.INITIALIZING if i == 0 else MemoryState.ACTIVE
        expected, memory = OneCallConv3d.run(conv, chunk, memory)
        got = conv(chunk, memory_state=state)
        assert got.shape == expected.shape
        assert got.dtype == expected.dtype
        assert mx.array_equal(got, expected).item()
        assert (conv.memory is None) == (memory is None)
        if memory is not None:
            assert mx.array_equal(conv.memory, memory).item()
    assert (counter.calls == len(chunks)) if takes_taps else (counter.calls == 0)


@pytest.mark.fast
def test_the_causal_memory_takes_the_dtype_of_the_next_slice():
    mx.random.seed(5)
    conv = OneCallConv3d.layer(32)
    first = mx.random.normal((1, 32, 5, 64, 64)).astype(mx.float32)
    second = mx.random.normal((1, 32, 4, 64, 64)).astype(mx.bfloat16)

    _, memory = OneCallConv3d.run(conv, first, None)
    conv(first, memory_state=MemoryState.INITIALIZING)
    expected, _ = OneCallConv3d.run(conv, second, memory)
    got = conv(second, memory_state=MemoryState.ACTIVE)

    assert got.dtype == expected.dtype
    assert mx.array_equal(got, expected).item()


@pytest.mark.fast
def test_causal_conv3d_disabled_state_keeps_no_memory():
    conv = OneCallConv3d.layer(32)
    conv(mx.random.normal((1, 32, 5, 64, 64)), memory_state=MemoryState.INITIALIZING)
    assert conv.memory is not None

    x = mx.random.normal((1, 32, 3, 64, 64))
    expected, _ = OneCallConv3d.run(conv, x, None)
    assert mx.array_equal(conv(x), expected).item()
    assert conv.memory is None


@pytest.mark.fast
def test_the_vae_layers_that_run_on_large_frames_take_the_taps():
    # Every unit-stride causal convolution of the real VAE whose input has a multiple of 16 channels takes the taps
    # on a 64x64 frame (all of them except the 3-channel encoder input layer).
    vae = SeedVR2VAE()
    layers = [m for _, m in vae.named_modules() if isinstance(m, CausalConv3d) and m.causal_temporal]
    unit = [m for m in layers if m.stride == (1, 1, 1) and m.kernel_size[0] > 1]
    shapes = {(m.weight.shape[-1], m.weight.shape[0]) for m in unit}
    assert shapes >= {(128, 128), (256, 256), (512, 512), (128, 3), (512, 32)}
    for in_channels, out_channels in shapes:
        layer = next(m for m in unit if (m.weight.shape[-1], m.weight.shape[0]) == (in_channels, out_channels))
        assert layer._runs_by_taps(mx.zeros((1, in_channels, 1, 64, 64))) == (in_channels % 16 == 0)


@pytest.mark.fast
def test_the_taps_need_a_fraction_of_the_single_call_memory():
    # Measured on MLX 0.32.5: about 0.94 GB for the single call, 0.07 GB for the taps.
    mx.random.seed(0)
    conv = OneCallConv3d.layer(128, out_channels=128)
    x = mx.random.normal((1, 128, 5, 128, 128)).astype(mx.bfloat16)
    mx.eval(x, conv.weight, conv.bias)
    frames = conv._causal_frames(x, MemoryState.DISABLED)

    def peak(run) -> int:
        mx.clear_cache()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        out = run()
        mx.eval(out)
        used = mx.get_peak_memory() - base
        del out
        mx.clear_cache()
        return used

    single = peak(lambda: conv._conv(mx.stack(frames, axis=2), temporal_padding=0))
    taps = peak(lambda: conv(x))
    assert taps * 4 < single


@pytest.mark.fast
def test_a_slice_leaves_only_its_output_and_a_two_frame_memory_alive():
    # The output and the causal memory are evaluated, and the memory is a copy: once the caller drops its input,
    # nothing else of that input stays allocated.
    mx.random.seed(2)
    conv = OneCallConv3d.layer(32)
    mx.eval(conv.weight, conv.bias)
    gc.collect()
    base = mx.get_active_memory()
    x = mx.random.normal((1, 32, 9, 128, 128)).astype(mx.float32)
    mx.eval(x)
    input_bytes = x.nbytes

    out = conv(x, memory_state=MemoryState.INITIALIZING)
    del x
    gc.collect()

    assert conv.memory.shape[2] == 2
    retained = mx.get_active_memory() - base - out.nbytes - conv.memory.nbytes
    assert retained < input_bytes // 2


# The tap guard mirrors cases verified on MLX 0.32's convolution dispatch; the bitwise test above fails if a new MLX
# changes it. It shares the MLX range of the video memory table, which was measured with this formulation.
@pytest.mark.fast
def test_the_tap_guard_was_read_from_the_installed_mlx():
    low, high = SeedVR2Util.VIDEO_VAE_MEMORY_MLX_RANGE
    installed = tuple(int(part) for part in mx.__version__.split(".")[:3])
    assert low <= installed < high, (
        f"MLX {mx.__version__} is outside {low} to below {high}: re-check CausalConv3d._runs_by_taps against "
        "this MLX's convolution dispatch with the bitwise test above, then re-measure VIDEO_VAE_MEMORY_BUCKETS."
    )
