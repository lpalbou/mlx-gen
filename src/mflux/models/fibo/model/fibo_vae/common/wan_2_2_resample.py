import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.common.wan_2_2_causal_conv_3d import Wan2_2_CausalConv3d


class Wan2_2_Resample(nn.Module):
    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode == "upsample3d":
            self.time_conv = Wan2_2_CausalConv3d(dim, dim * 2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
            self.resample_conv = nn.Conv2d(dim, upsample_out_dim, kernel_size=3, stride=1, padding=1)
        elif mode == "upsample2d":
            self.resample_conv = nn.Conv2d(dim, upsample_out_dim, kernel_size=3, stride=1, padding=1)
            self.time_conv = None
        elif mode == "downsample2d":
            self.resample_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=0)
            self.time_conv = None
        elif mode == "downsample3d":
            self.resample_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=0)
            self.time_conv = Wan2_2_CausalConv3d(
                dim,
                dim,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                padding=(0, 0, 0),
            )
        else:
            raise ValueError(f"Unsupported resample mode: {mode}")

    def __call__(
        self,
        x: mx.array,
        block_idx: int | None = None,
        feat_cache: list[mx.array | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> mx.array:
        b, c, t, h, w = x.shape
        if self.mode in ("upsample2d", "upsample3d"):
            if self.mode == "upsample3d" and self.time_conv is not None:
                if feat_cache is not None and feat_idx is not None:
                    idx = feat_idx[0]
                    if feat_cache[idx] is None:
                        feat_cache[idx] = "Rep"
                        feat_idx[0] += 1
                    else:
                        cache_x = x[:, :, -2:, :, :]
                        if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                            cache_x = mx.concatenate([feat_cache[idx][:, :, -1:, :, :], cache_x], axis=2)
                        if cache_x.shape[2] < 2 and feat_cache[idx] == "Rep":
                            cache_x = mx.concatenate([mx.zeros_like(cache_x), cache_x], axis=2)
                        cache_arg = None if feat_cache[idx] == "Rep" else feat_cache[idx]
                        x = self.time_conv(x, cache_arg)
                        feat_cache[idx] = cache_x
                        feat_idx[0] += 1
                        x = mx.reshape(x, (b, 2, c, t, h, w))
                        x = mx.transpose(x, (0, 2, 3, 1, 4, 5))
                        x = mx.reshape(x, (b, c, t * 2, h, w))
                        t = t * 2
            x = mx.transpose(x, (0, 2, 1, 3, 4))
            x = mx.reshape(x, (b * t, c, h, w))
            x = mx.transpose(x, (0, 2, 3, 1))
            x = mx.repeat(x, 2, axis=1)
            x = mx.repeat(x, 2, axis=2)
            x = self.resample_conv(x)
            x = mx.transpose(x, (0, 3, 1, 2))
            new_c = x.shape[1]
            new_h, new_w = x.shape[2], x.shape[3]
            x = mx.reshape(x, (b, t, new_c, new_h, new_w))
            x = mx.transpose(x, (0, 2, 1, 3, 4))
            return x

        # downsample modes
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        x = mx.reshape(x, (b * t, c, h, w))
        x = mx.transpose(x, (0, 2, 3, 1))
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        x = self.resample_conv(x)
        x = mx.transpose(x, (0, 3, 1, 2))
        new_c = x.shape[1]
        new_h, new_w = x.shape[2], x.shape[3]
        x = mx.reshape(x, (b, t, new_c, new_h, new_w))
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        if self.mode == "downsample3d" and self.time_conv is not None and feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :]
                x = self.time_conv(mx.concatenate([feat_cache[idx][:, :, -1:, :, :], x], axis=2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x
