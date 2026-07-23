import mlx.core as mx
from PIL import Image

from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.utils.mask_util import MaskUtil


class QwenEditUtil:
    CONDITION_IMAGE_SIZE = 384 * 384
    VAE_IMAGE_SIZE = 1024 * 1024

    @staticmethod
    def create_image_conditioning_latents(
        vae,
        height: int | None,
        width: int | None,
        image_paths: list[str] | str,
        tiling_config: TilingConfig | None = None,
    ) -> tuple[mx.array, mx.array, list[tuple[int, int, int]], int]:
        if not isinstance(image_paths, list):
            image_paths = [str(image_paths)]

        all_image_latents = []
        all_image_grids = []
        for image_path in image_paths:
            calc_w, calc_h = QwenEditUtil._conditioning_vae_size(image_path=image_path, width=width, height=height)
            input_image = LatentCreator.encode_image(
                vae=vae,
                image_path=image_path,
                height=calc_h,
                width=calc_w,
                tiling_config=tiling_config,
            )

            image_latents = QwenLatentCreator.pack_latents(
                latents=input_image,
                height=calc_h,
                width=calc_w,
                num_channels_latents=16,
            )
            all_image_latents.append(image_latents)
            all_image_grids.append((1, calc_h // 16, calc_w // 16))

        image_latents = mx.concatenate(all_image_latents, axis=1)

        all_image_ids = []
        for image_path in image_paths:
            calc_w, calc_h = QwenEditUtil._conditioning_vae_size(image_path=image_path, width=width, height=height)
            image_ids = QwenEditUtil._create_image_ids(
                height=calc_h,
                width=calc_w,
            )
            all_image_ids.append(image_ids)
        image_ids = mx.concatenate(all_image_ids, axis=1)

        num_images = len(image_paths)
        return image_latents, image_ids, all_image_grids, num_images

    @staticmethod
    def create_inpaint_mask_latents(
        mask_path: str,
        *,
        height: int,
        width: int,
        num_channels_latents: int = 16,
        resize_mode: str = "resize",
    ) -> mx.array:
        latent_width = width // 8
        latent_height = height // 8
        mask_values = MaskUtil.load_binary_mask(
            mask_path,
            target_width=latent_width,
            target_height=latent_height,
            resampling=Image.Resampling.NEAREST,
            alpha_warning_context="Qwen inpaint mask",
            resize_mode=resize_mode,
        )
        mask = mx.array(mask_values)[None, None, :, :]
        mask = mx.repeat(mask, repeats=num_channels_latents, axis=1)
        return QwenLatentCreator.pack_latents(
            latents=mask,
            height=height,
            width=width,
            num_channels_latents=num_channels_latents,
        )

    @staticmethod
    def blend_inpaint_latents(
        *,
        latents: mx.array,
        image_latents: mx.array,
        initial_noise: mx.array,
        mask_latents: mx.array,
        sigma: mx.array | float,
    ) -> mx.array:
        init_latents = LatentCreator.add_noise_by_interpolation(
            clean=image_latents,
            noise=initial_noise,
            sigma=sigma,
        )
        # The float32 mask would otherwise promote the composite (and every later step) to f32.
        return ((1 - mask_latents) * init_latents + mask_latents * latents).astype(latents.dtype)

    @staticmethod
    def _conditioning_vae_size(image_path: str, width: int | None = None, height: int | None = None) -> tuple[int, int]:
        if width is not None and height is not None:
            return width, height

        from PIL import Image

        with Image.open(image_path) as image:
            ratio = image.size[0] / image.size[1]
        return QwenEditUtil._area_dimensions(target_area=QwenEditUtil.VAE_IMAGE_SIZE, ratio=ratio)

    @staticmethod
    def _area_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
        width = (target_area * ratio) ** 0.5
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return int(width), int(height)

    @staticmethod
    def _create_image_ids(
        height: int,
        width: int,
    ) -> mx.array:
        latent_height = height // 16
        latent_width = width // 16

        image_ids = mx.zeros((latent_height, latent_width, 3))

        row_coords = mx.arange(0, latent_height)[:, None]
        row_coords = mx.broadcast_to(row_coords, (latent_height, latent_width))
        image_ids = mx.concatenate(
            [
                image_ids[:, :, :1],
                row_coords[:, :, None],
                image_ids[:, :, 2:],
            ],
            axis=2,
        )

        col_coords = mx.arange(0, latent_width)[None, :]
        col_coords = mx.broadcast_to(col_coords, (latent_height, latent_width))
        image_ids = mx.concatenate(
            [
                image_ids[:, :, :2],
                col_coords[:, :, None],
            ],
            axis=2,
        )

        image_ids = mx.reshape(image_ids, (latent_height * latent_width, 3))

        first_dim = mx.ones((image_ids.shape[0], 1))
        image_ids = mx.concatenate([first_dim, image_ids[:, 1:]], axis=1)

        image_ids = mx.expand_dims(image_ids, axis=0)

        return image_ids
