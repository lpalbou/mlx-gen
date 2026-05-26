import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.common.wan_2_2_causal_conv_3d import Wan2_2_CausalConv3d
from mflux.models.fibo.model.fibo_vae.common.wan_2_2_rms_norm import Wan2_2_RMSNorm

CACHE_T = 2


class Wan2_2_ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.norm1 = Wan2_2_RMSNorm(in_dim, images=False)
        self.conv1 = Wan2_2_CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = Wan2_2_RMSNorm(out_dim, images=False)
        self.conv2 = Wan2_2_CausalConv3d(out_dim, out_dim, 3, padding=1)

        if in_dim != out_dim:
            self.conv_shortcut = Wan2_2_CausalConv3d(in_dim, out_dim, 1, padding=0)
        else:
            self.conv_shortcut = None

    def __call__(
        self,
        x: mx.array,
        resnet_idx: int | None = None,
        block_idx: int | None = None,
        feat_cache: list[mx.array | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> mx.array:
        h = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        x = self.norm1(x)
        x = nn.silu(x)
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = self._cache_slice(x, feat_cache[idx])
            x = self.conv1(x, None if feat_cache[idx] == "Rep" else feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        x = self.norm2(x)
        x = nn.silu(x)
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = self._cache_slice(x, feat_cache[idx])
            x = self.conv2(x, None if feat_cache[idx] == "Rep" else feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)
        result = x + h
        return result

    @staticmethod
    def _cache_slice(x: mx.array, previous: mx.array | str | None) -> mx.array:
        cache_x = x[:, :, -CACHE_T:, :, :]
        if cache_x.shape[2] < CACHE_T and previous is not None and previous != "Rep":
            cache_x = mx.concatenate([previous[:, :, -1:, :, :], cache_x], axis=2)
        return cache_x
