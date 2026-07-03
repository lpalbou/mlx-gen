import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.common.wan_2_2_causal_conv_3d import Wan2_2_CausalConv3d
from mflux.models.fibo.model.fibo_vae.common.wan_2_2_mid_block import Wan2_2_MidBlock
from mflux.models.fibo.model.fibo_vae.common.wan_2_2_rms_norm import Wan2_2_RMSNorm
from mflux.models.fibo.model.fibo_vae.encoder.wan_2_2_down_block import Wan2_2_DownBlock


class Wan2_2_Encoder3d(nn.Module):
    supports_cache = True

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        attn_scales: list[float] | None = None,
        temporal_downsample: list[bool] | None = None,
        non_linearity: str = "silu",
        is_residual: bool = False,
    ):
        super().__init__()

        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temporal_downsample is None:
            temporal_downsample = [False, True, True]

        if is_residual:
            raise NotImplementedError("Residual down blocks are not implemented for Wan2_2_Encoder3d in MLX.")

        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        dims = [dim * u for u in [1] + dim_mult]
        self.temporal_downsample = temporal_downsample

        self.conv_in = Wan2_2_CausalConv3d(in_channels, dims[0], 3, padding=1)
        scale = 1.0

        self.down_blocks: list[Wan2_2_DownBlock] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            block = Wan2_2_DownBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                attn_scales=attn_scales,
                scale=scale,
                temporal_downsample=temporal_downsample[i] if i < len(temporal_downsample) else False,
                non_linearity=non_linearity,
                is_last=i == len(dim_mult) - 1,
            )
            self.down_blocks.append(block)
            if i != len(dim_mult) - 1:
                scale /= 2.0

        self.mid_block = Wan2_2_MidBlock(out_dim, non_linearity, num_layers=1)

        self.norm_out = Wan2_2_RMSNorm(out_dim, images=False)
        self.conv_out = Wan2_2_CausalConv3d(out_dim, z_dim, 3, padding=1)

    def __call__(
        self,
        x: mx.array,
        feat_cache: list[mx.array | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> mx.array:
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = self._cache_slice(x, feat_cache[idx])
            x = self.conv_in(x, None if feat_cache[idx] == "Rep" else feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x = self.norm_out(x)
        x = nn.silu(x)
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = self._cache_slice(x, feat_cache[idx])
            x = self.conv_out(x, None if feat_cache[idx] == "Rep" else feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x

    @staticmethod
    def _cache_slice(x: mx.array, previous: mx.array | str | None) -> mx.array:
        cache_x = x[:, :, -2:, :, :]
        if cache_x.shape[2] < 2 and previous is not None and previous != "Rep":
            cache_x = mx.concatenate([previous[:, :, -1:, :, :], cache_x], axis=2)
        return cache_x
