from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import mlx.core as mx
from mlx import nn

from mflux.models.common.vae.vae_util import VAEUtil
from mflux.utils.image_util import ImageUtil

if TYPE_CHECKING:
    from mflux.models.common.vae.tiling_config import TilingConfig
    from mflux.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
    from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
    from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator

    LatentCreatorType: TypeAlias = type[FiboLatentCreator | FluxLatentCreator | QwenLatentCreator | ZImageLatentCreator]


class Img2Img:
    def __init__(
        self,
        vae: nn.Module,
        latent_creator: "LatentCreatorType",
        sigmas: mx.array,
        init_time_step: int,
        image_strength: float | None,
        image_path: str | Path | None,
        tiling_config: "TilingConfig" | None = None,
        resize_mode: str = "resize",
    ):
        self.vae = vae
        self.sigmas = sigmas
        self.init_time_step = init_time_step
        self.image_strength = image_strength
        self.image_path = image_path
        self.latent_creator = latent_creator
        self.tiling_config = tiling_config
        self.resize_mode = resize_mode


class LatentCreator:
    @staticmethod
    def create_for_txt2img_or_img2img(
        seed: int,
        height: int,
        width: int,
        img2img: Img2Img,
    ) -> mx.array:
        latent_creator = img2img.latent_creator

        if img2img.image_path is None:
            # txt2img: just create noise
            return latent_creator.create_noise(seed, height, width)
        else:
            if img2img.image_strength is None or img2img.image_strength <= 0.0:
                raise ValueError("latent image-to-image requires image_strength > 0.")
            # img2img: blend encoded image with noise
            pure_noise = latent_creator.create_noise(seed, height, width)
            sample_posterior = latent_creator.__name__ == "QwenLatentCreator" and hasattr(img2img.vae, "encode_sampled")
            encoded = LatentCreator.encode_image(
                width=width,
                height=height,
                vae=img2img.vae,
                image_path=img2img.image_path,
                tiling_config=img2img.tiling_config,
                sample_posterior=sample_posterior,
                seed=seed,
                resize_mode=img2img.resize_mode,
            )
            latents = latent_creator.pack_latents(encoded, height, width)
            sigma = img2img.sigmas[img2img.init_time_step]
            return LatentCreator.add_noise_by_interpolation(clean=latents, noise=pure_noise, sigma=sigma)

    @staticmethod
    def encode_image(
        vae: nn.Module,
        image_path: str | Path,
        height: int,
        width: int,
        tiling_config: "TilingConfig" | None = None,
        sample_posterior: bool = False,
        seed: int | None = None,
        resize_mode: str = "resize",
    ) -> mx.array:
        scaled_user_image = ImageUtil.scale_to_dimensions(
            image=ImageUtil.load_image(image_path).convert("RGB"),
            target_width=width,
            target_height=height,
            resize_mode=resize_mode,
        )
        image_array = ImageUtil.to_array(scaled_user_image)
        return VAEUtil.encode(
            vae=vae,
            image=image_array,
            tiling_config=tiling_config,
            sample_posterior=sample_posterior,
            seed=seed,
        )

    @staticmethod
    def add_noise_by_interpolation(clean: mx.array, noise: mx.array, sigma: float) -> mx.array:
        # Scheduler sigmas arrive as 0-d float32 arrays; without the cast MLX promotes bf16
        # latents to float32 here and the entire downstream denoise loop silently runs at f32
        # (measured 1.5-2.5x slower per step). Blend in f32, return in the latent dtype.
        return ((1 - sigma) * clean + sigma * noise).astype(clean.dtype)
