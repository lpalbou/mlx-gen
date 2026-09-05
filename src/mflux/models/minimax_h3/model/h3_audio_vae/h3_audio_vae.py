"""MiniMax-H3 audio autoencoder (diffusers `AutoencoderKLMiniMaxH3Audio`).

Waveform in, waveform out: a DAC-lineage strided convolutional encoder (Snake activations),
a causal-attention projection that narrows the trunk to the 32-channel latent, and a
BigVGAN decoder with anti-aliased SnakeBeta activations. The model is mono; MiniMax-H3
carries stereo as two batch items. Module and parameter names follow the checkpoint so
loading is a passthrough apart from folding `weight_norm` (`weight_g` / `weight_v`) into
plain kernels and moving convolution kernels to MLX's channels-last layout.

Public tensors use the reference `(batch, channels, samples)` layout; internally the stack
runs channels-last as MLX convolutions expect.
"""

import math

import mlx.core as mx
from mlx import nn

_SNAKE_EPS = 1e-9


class H3Snake1d(nn.Module):
    """`x + (alpha + eps)^-1 * sin(alpha * x)^2`, per-channel `alpha` stored `(1, C, 1)` as in the checkpoint."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, channels, 1))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha.reshape(1, 1, -1)
        return x + (1.0 / (alpha + _SNAKE_EPS)) * mx.square(mx.sin(alpha * x))


class H3SnakeBeta(nn.Module):
    """`x + (exp(beta) + eps)^-1 * sin(exp(alpha) * x)^2` with log-space `(C,)` parameters."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha)[None, None, :]
        beta = mx.exp(self.beta)[None, None, :]
        return x + (1.0 / (beta + _SNAKE_EPS)) * mx.square(mx.sin(alpha * x))


def _depthwise_kernel(filter_: mx.array, channels: int) -> mx.array:
    # Checkpoint filter is `(1, 1, k)`; MLX grouped conv wants `(C, k, 1)`.
    return mx.broadcast_to(filter_.reshape(1, -1, 1), (channels, filter_.shape[-1], 1))


class H3LowPassFilter1d(nn.Module):
    """Depthwise Kaiser-sinc low-pass filter with a stride (the anti-aliased downsampler)."""

    def __init__(self, stride: int, kernel_size: int):
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.filter = mx.zeros((1, 1, kernel_size))

    def __call__(self, x: mx.array) -> mx.array:
        channels = x.shape[-1]
        x = mx.pad(x, [(0, 0), (self.pad_left, self.pad_right), (0, 0)], mode="edge")
        return mx.conv1d(x, _depthwise_kernel(self.filter, channels), stride=self.stride, groups=channels)


class H3UpSample1d(nn.Module):
    """Anti-aliased `ratio`x upsampler: depthwise transposed Kaiser-sinc convolution."""

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.filter = mx.zeros((1, 1, kernel_size))

    def __call__(self, x: mx.array) -> mx.array:
        channels = x.shape[-1]
        x = mx.pad(x, [(0, 0), (self.pad, self.pad), (0, 0)], mode="edge")
        x = self.ratio * mx.conv_transpose1d(
            x, _depthwise_kernel(self.filter, channels), stride=self.stride, groups=channels
        )
        return x[:, self.pad_left : x.shape[1] - self.pad_right, :]


class H3DownSample1d(nn.Module):
    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.lowpass = H3LowPassFilter1d(stride=ratio, kernel_size=kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class H3AliasFreeActivation(nn.Module):
    """Upsample, activate, downsample — BigVGAN's alias-free activation wrapper."""

    def __init__(self, activation: nn.Module, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = activation
        self.upsample = H3UpSample1d(ratio, kernel_size)
        self.downsample = H3DownSample1d(ratio, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.downsample(self.act(self.upsample(x)))


class H3ResidualUnit(nn.Module):
    """DAC residual unit: Snake, dilated k=7 conv, Snake, k=1 conv, plus a (center-cropped) shortcut."""

    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = [
            H3Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=((7 - 1) * dilation) // 2),
            H3Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        for layer in self.block:
            residual = layer(residual)
        pad = (x.shape[1] - residual.shape[1]) // 2
        if pad > 0:
            x = x[:, pad:-pad, :]
        return x + residual


class H3EncoderBlock(nn.Module):
    """Three residual units at dilations 1/3/9, then a strided channel-doubling convolution."""

    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = [
            H3ResidualUnit(dim // 2, dilation=1),
            H3ResidualUnit(dim // 2, dilation=3),
            H3ResidualUnit(dim // 2, dilation=9),
            H3Snake1d(dim // 2),
            nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


class H3AudioEncoder(nn.Module):
    def __init__(self, d_model: int, strides: tuple[int, ...], d_latent: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            layers.append(H3EncoderBlock(d_model, stride=stride))
        layers += [H3Snake1d(d_model), nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


class H3GeGluMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        return self.w2(nn.gelu_approx(self.w0(x)) * self.w1(x))


class H3CausalAttention(nn.Module):
    """Causal self-attention that narrows `in_dim` to `out_dim`: heads are mean-pooled, then the head
    dimension is average-pooled down to `out_dim`. QKV is one bias-less projection with separate query
    and value biases and a frozen zero key bias, exactly as stored."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = mx.zeros((in_dim,))
        self.v_bias = mx.zeros((in_dim,))
        self.zero_k_bias = mx.zeros((in_dim,))
        self.proj = nn.Linear(out_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x) + mx.concatenate([self.q_bias, self.zero_k_bias, self.v_bias])
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        attention = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=1.0 / math.sqrt(self.head_dim), mask="causal"
        )
        attention = attention.mean(axis=1)  # (B, S, head_dim)
        pooled = attention.reshape(batch_size, seq_len, self.out_dim, self.head_dim // self.out_dim).mean(axis=-1)
        return self.proj(pooled)


class H3AttnProjection(nn.Module):
    """`pre_block`: residual causal attention + GeGLU that rewires `latent_dim` to `latent_channels`."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = H3CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = H3GeGluMlp(out_dim, out_dim * mlp_ratio)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class H3AMPBlock(nn.Module):
    """BigVGAN anti-aliased multi-periodicity block: per dilation a (dilated conv, plain conv) pair,
    every convolution preceded by its own alias-free SnakeBeta."""

    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...]):
        super().__init__()
        self.convs1 = [
            nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2)
            for d in dilations
        ]
        self.convs2 = [nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2) for _ in dilations]
        self.activations = [H3AliasFreeActivation(H3SnakeBeta(channels)) for _ in range(2 * len(dilations))]

    def __call__(self, x: mx.array) -> mx.array:
        for index, (conv1, conv2) in enumerate(zip(self.convs1, self.convs2)):
            residual = conv1(self.activations[2 * index](x))
            residual = conv2(self.activations[2 * index + 1](residual))
            x = x + residual
        return x


class H3BigVGANDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        upsample_initial_channel: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)
        self.ups = []
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                [
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel,
                        stride=rate,
                        padding=(kernel - rate) // 2,
                    )
                ]
            )
        self.resblocks = []
        for i in range(self.num_upsamples):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for kernel, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(H3AMPBlock(channels, kernel, tuple(dilations)))
        self.activation_post = H3AliasFreeActivation(H3SnakeBeta(channels))
        self.conv_post = nn.Conv1d(channels, 1, 7, padding=3, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i][0](x)
            summed = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j](x)
                summed = block if summed is None else summed + block
            x = summed / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return mx.clip(x, -1.0, 1.0)


class H3AudioVAE(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5),
        latent_dim: int = 2048,
        latent_channels: int = 32,
        num_attention_heads: int = 8,
        decoder_dim: int = 1024,
        decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2),
        decoder_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4),
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        sampling_rate: int = 32000,
        latents_mean: list[float] | None = None,
        latents_std: list[float] | None = None,
    ):
        super().__init__()
        self.hop_length = math.prod(encoder_rates)
        self.latent_channels = latent_channels
        self.sampling_rate = sampling_rate
        self._latents_mean = mx.array(latents_mean, dtype=mx.float32) if latents_mean else None
        self._latents_std = mx.array(latents_std, dtype=mx.float32) if latents_std else None
        self.encoder = H3AudioEncoder(encoder_dim, tuple(encoder_rates), latent_dim)
        self.pre_block = H3AttnProjection(latent_dim, latent_channels, num_attention_heads)
        self.mean_proj = nn.Conv1d(latent_channels, latent_channels, 1)
        self.logs_proj = nn.Conv1d(latent_channels, latent_channels, 1)
        self.dec_in_proj = nn.Conv1d(latent_channels, latent_dim, 1)
        self.decoder = H3BigVGANDecoder(
            latent_dim,
            decoder_dim,
            tuple(decoder_rates),
            tuple(decoder_kernel_sizes),
            tuple(resblock_kernel_sizes),
            tuple(resblock_dilation_sizes),
        )

    @property
    def latents_mean(self) -> mx.array:
        return self._latents_mean

    @property
    def latents_std(self) -> mx.array:
        return self._latents_std

    def encode(self, waveform: mx.array) -> mx.array:
        """Mono `(batch, 1, samples)` waveform to the posterior mean `(batch, latent_channels, samples / hop)`."""
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(f"`waveform` must have shape (batch, 1, samples), got {tuple(waveform.shape)}.")
        right_pad = math.ceil(waveform.shape[-1] / self.hop_length) * self.hop_length - waveform.shape[-1]
        x = mx.pad(waveform, [(0, 0), (0, 0), (0, right_pad)]).transpose(0, 2, 1)
        x = self.encoder(x)
        x = self.pre_block(x)
        return self.mean_proj(x).transpose(0, 2, 1)

    def decode(self, latents: mx.array) -> mx.array:
        """Denormalized `(batch, latent_channels, frames)` latents to a `(batch, 1, frames * hop)` waveform in [-1, 1]."""
        if latents.ndim != 3:
            raise ValueError(f"`latents` must have shape (batch, latent_channels, frames), got {tuple(latents.shape)}.")
        x = self.dec_in_proj(latents.transpose(0, 2, 1))
        return self.decoder(x).transpose(0, 2, 1)
