import mlx.core as mx
from PIL import Image

from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.utils.image_util import ImageUtil
from mflux.utils.mask_util import MaskUtil


class QwenControlNetUtil:
    @staticmethod
    def create_controlnet_condition(
        *,
        vae,
        controlnet_image_path: str,
        height: int,
        width: int,
        tiling_config=None,
    ) -> mx.array:
        encoded = LatentCreator.encode_image(
            vae=vae,
            image_path=controlnet_image_path,
            height=height,
            width=width,
            tiling_config=tiling_config,
        )
        return QwenLatentCreator.pack_latents(
            latents=encoded,
            height=height,
            width=width,
            num_channels_latents=16,
        )

    @staticmethod
    def create_inpaint_controlnet_condition(
        *,
        vae,
        image_path: str,
        mask_path: str,
        height: int,
        width: int,
        tiling_config=None,
    ) -> mx.array:
        scaled_image = ImageUtil.scale_to_dimensions(
            image=ImageUtil.load_image(image_path).convert("RGB"),
            target_width=width,
            target_height=height,
        )
        image_array = ImageUtil.to_array(scaled_image)
        image_mask = QwenControlNetUtil._load_binary_mask(
            mask_path=mask_path, width=width, height=height, alpha_warning=True
        )
        masked_image = mx.where(image_mask > 0.5, -mx.ones_like(image_array), image_array)
        image_latents = VAEUtil.encode(vae=vae, image=masked_image, tiling_config=tiling_config)
        latent_mask = QwenControlNetUtil._load_binary_mask(mask_path=mask_path, width=width // 8, height=height // 8)
        control_condition = mx.concatenate([image_latents, 1 - latent_mask], axis=1)
        return QwenLatentCreator.pack_latents(
            latents=control_condition,
            height=height,
            width=width,
            num_channels_latents=17,
        )

    @staticmethod
    def _load_binary_mask(*, mask_path: str, width: int, height: int, alpha_warning: bool = False) -> mx.array:
        mask_values = MaskUtil.load_binary_mask(
            mask_path,
            target_width=width,
            target_height=height,
            resampling=Image.Resampling.NEAREST,
            alpha_warning_context="Qwen control inpaint mask" if alpha_warning else None,
        )
        return mx.array(mask_values)[None, None, :, :]
