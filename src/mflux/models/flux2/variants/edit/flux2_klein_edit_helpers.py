from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.utils.mask_util import MaskUtil
from mflux.utils.outpaint_util import OutpaintCanvas


class _Flux2KleinEditHelpers:
    CONDITION_TARGET_AREA = 1024 * 1024

    @staticmethod
    def is_base_model(model_config) -> bool:
        model_name_lower = model_config.model_name.lower()
        base_model_lower = (model_config.base_model or "").lower()
        return "klein-base" in model_name_lower or "klein-base" in base_model_lower

    @staticmethod
    def validate_guidance(*, model_config, guidance: float) -> None:
        if guidance == 1.0:
            return
        if _Flux2KleinEditHelpers.is_base_model(model_config):
            return
        raise ValueError("guidance > 1.0 is only supported for FLUX.2 Klein base models.")

    @staticmethod
    def default_guidance(model_config) -> float:
        # Base models run true CFG like the source-locked outpaint route; distilled Klein
        # models are step-distilled and must stay at guidance 1.0.
        return 4.0 if _Flux2KleinEditHelpers.is_base_model(model_config) else 1.0

    @staticmethod
    def encode_text(
        prompt: str,
        *,
        tokenizer,
        text_encoder: Qwen3TextEncoder,
    ) -> tuple[mx.array, mx.array]:
        return Flux2PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )

    @staticmethod
    def latent_grid_from_image_size(height: int, width: int) -> tuple[int, int]:
        vae_scale_factor = 8
        effective_height = 2 * (height // (vae_scale_factor * 2))
        effective_width = 2 * (width // (vae_scale_factor * 2))
        latent_height = effective_height // 2
        latent_width = effective_width // 2
        return latent_height, latent_width

    @staticmethod
    def build_latent_ids_grid(batch_size: int, latent_height: int, latent_width: int, t_coord: int = 0) -> mx.array:
        h_ids = mx.arange(latent_height, dtype=mx.int32)
        w_ids = mx.arange(latent_width, dtype=mx.int32)
        h_grid = mx.broadcast_to(mx.expand_dims(h_ids, axis=1), (latent_height, latent_width))
        w_grid = mx.broadcast_to(mx.expand_dims(w_ids, axis=0), (latent_height, latent_width))

        flat_h = h_grid.reshape(-1)
        flat_w = w_grid.reshape(-1)
        t = mx.full(flat_h.shape, t_coord, dtype=mx.int32)
        layer_ids = mx.zeros_like(flat_h)

        coords = mx.stack([t, flat_h, flat_w, layer_ids], axis=1)
        coords = mx.expand_dims(coords, axis=0)
        return mx.broadcast_to(coords, (batch_size, coords.shape[1], coords.shape[2]))

    @staticmethod
    def prepare_generation_latents(
        *,
        seed: int,
        height: int,
        width: int,
    ) -> tuple[mx.array, mx.array, int, int]:
        return Flux2LatentCreator.prepare_packed_latents(
            seed=seed,
            height=height,
            width=width,
            batch_size=1,
        )

    @staticmethod
    def crop_to_even_spatial(latents: mx.array) -> mx.array:
        if latents.shape[2] % 2 != 0:
            latents = latents[:, :, :-1, :]
        if latents.shape[3] % 2 != 0:
            latents = latents[:, :, :, :-1]
        return latents

    @staticmethod
    def ensure_4d_latents(latents: mx.array) -> mx.array:
        if latents.ndim == 5 and latents.shape[2] == 1:
            return latents[:, :, 0, :, :]
        return latents

    @staticmethod
    def bn_normalize_vae_encoded_latents(encoded: mx.array, *, vae: Flux2VAE) -> mx.array:
        bn_mean = vae.bn.running_mean.reshape(1, -1, 1, 1).astype(encoded.dtype)
        bn_std = mx.sqrt(vae.bn.running_var.reshape(1, -1, 1, 1) + vae.bn.eps).astype(encoded.dtype)
        return (encoded - bn_mean) / bn_std

    @staticmethod
    def encode_reference_image_to_packed_latents(
        *,
        vae: Flux2VAE,
        tiling_config,
        image_path: Path | str,
        height: int,
        width: int,
    ) -> mx.array:
        encoded = LatentCreator.encode_image(
            vae=vae,
            image_path=image_path,
            height=height,
            width=width,
            tiling_config=tiling_config,
        )
        encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
        encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
        encoded = Flux2LatentCreator.patchify_latents(encoded)
        encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(encoded, vae=vae)
        return Flux2LatentCreator.pack_latents(encoded)

    @staticmethod
    def prepare_reference_image_conditioning(
        *,
        vae: Flux2VAE,
        tiling_config,
        image_paths: list[Path | str] | None = None,
        height: int,
        width: int,
        batch_size: int = 1,
        t_coord_start: int = 10,
    ):
        if not image_paths:
            return None, None

        packed_latents_list: list[mx.array] = []
        ids_list: list[mx.array] = []
        for i, p in enumerate(image_paths):
            encode_width, encode_height = _Flux2KleinEditHelpers.reference_condition_dimensions(image_path=p)
            encoded = LatentCreator.encode_image(
                vae=vae,
                image_path=p,
                height=encode_height,
                width=encode_width,
                tiling_config=tiling_config,
                resize_mode="crop",
            )
            encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
            encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
            encoded = Flux2LatentCreator.patchify_latents(encoded)
            encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(encoded, vae=vae)

            packed_latents_list.append(Flux2LatentCreator.pack_latents(encoded))
            ids_list.append(Flux2LatentCreator.prepare_grid_ids(encoded, t_coord=t_coord_start + 10 * i))

        image_latents = mx.concatenate(packed_latents_list, axis=1)
        image_latent_ids = mx.concatenate(ids_list, axis=1)

        if image_latents.shape[0] != batch_size:
            image_latents = mx.broadcast_to(image_latents, (batch_size, image_latents.shape[1], image_latents.shape[2]))
        if image_latent_ids.shape[0] != batch_size:
            image_latent_ids = mx.broadcast_to(
                image_latent_ids, (batch_size, image_latent_ids.shape[1], image_latent_ids.shape[2])
            )

        return image_latents, image_latent_ids

    @staticmethod
    def prepare_inpaint_source_conditioning(
        *,
        packed_source_latents: mx.array,
        height: int,
        width: int,
        batch_size: int = 1,
        t_coord: int = 10,
    ) -> tuple[mx.array, mx.array]:
        # The clean source latents double as reference conditioning tokens (diffusers Klein
        # inpaint feeds the encoded init image as clean context at every denoising step).
        latent_height, latent_width = _Flux2KleinEditHelpers.latent_grid_from_image_size(height, width)
        source_ids = _Flux2KleinEditHelpers.build_latent_ids_grid(
            batch_size=batch_size,
            latent_height=latent_height,
            latent_width=latent_width,
            t_coord=t_coord,
        )
        source_latents = packed_source_latents
        if source_latents.shape[0] != batch_size:
            source_latents = mx.broadcast_to(
                source_latents, (batch_size, source_latents.shape[1], source_latents.shape[2])
            )
        return source_latents, source_ids

    @staticmethod
    def prepare_inpaint_mask(
        *,
        mask_path: Path | str,
        height: int,
        width: int,
        batch_size: int = 1,
    ) -> mx.array:
        # Diffusers Klein inpaint parity: resize to pixel resolution with the VaeImageProcessor
        # default LANCZOS filter, binarize, then bilinear-interpolate directly to the packed
        # latent grid (torch F.interpolate semantics) so most cells stay hard while true
        # boundary cells keep soft values.
        latent_height = height // 16
        latent_width = width // 16
        binary_mask = MaskUtil.load_binary_mask(
            mask_path,
            target_width=width,
            target_height=height,
            resampling=Image.Resampling.LANCZOS,
            alpha_warning_context="FLUX.2 Klein inpaint mask",
        )
        latent_mask = MaskUtil.interpolate_bilinear(
            binary_mask,
            target_height=latent_height,
            target_width=latent_width,
        )
        mask_array = mx.array(latent_mask).reshape(1, latent_height * latent_width, 1)
        if batch_size > 1:
            mask_array = mx.broadcast_to(mask_array, (batch_size, mask_array.shape[1], mask_array.shape[2]))
        return mask_array

    @staticmethod
    def preserved_source_latents(
        *,
        clean_latents: mx.array,
        noise_latents: mx.array,
        sigmas: mx.array,
        timestep: int,
    ) -> mx.array:
        if timestep + 1 >= len(sigmas) - 1:
            return clean_latents
        return LatentCreator.add_noise_by_interpolation(
            clean=clean_latents,
            noise=noise_latents,
            sigma=sigmas[timestep + 1],
        )

    @staticmethod
    def prepare_outpaint_edit_mask(
        *,
        canvas: OutpaintCanvas,
        height: int,
        width: int,
        batch_size: int = 1,
        transition_px: int = 24,
    ) -> mx.array:
        latent_height = height // 16
        latent_width = width // 16
        mask = Image.new("L", (canvas.target_width, canvas.target_height), color=255)
        inset_left = min(transition_px, max(0, canvas.source_width // 2 - 1))
        inset_top = min(transition_px, max(0, canvas.source_height // 2 - 1))
        preserve_left = canvas.paste_left + inset_left
        preserve_top = canvas.paste_top + inset_top
        preserve_right = canvas.paste_left + canvas.source_width - inset_left
        preserve_bottom = canvas.paste_top + canvas.source_height - inset_top
        if preserve_right <= preserve_left or preserve_bottom <= preserve_top:
            preserve_left = canvas.paste_left
            preserve_top = canvas.paste_top
            preserve_right = canvas.paste_left + canvas.source_width
            preserve_bottom = canvas.paste_top + canvas.source_height
        mask.paste(
            0,
            (
                preserve_left,
                preserve_top,
                preserve_right,
                preserve_bottom,
            ),
        )
        mask = mask.resize((latent_width, latent_height), resample=Image.Resampling.BILINEAR)
        mask_array = mx.array(np.asarray(mask, dtype=np.float32) / 255.0).reshape(1, latent_height * latent_width, 1)
        if batch_size > 1:
            mask_array = mx.broadcast_to(mask_array, (batch_size, mask_array.shape[1], mask_array.shape[2]))
        return mask_array

    @staticmethod
    def reference_condition_dimensions(*, image_path: Path | str) -> tuple[int, int]:
        with Image.open(image_path) as image:
            width, height = image.size
        area = width * height
        if area > _Flux2KleinEditHelpers.CONDITION_TARGET_AREA:
            ratio = width / height
            target_width = (_Flux2KleinEditHelpers.CONDITION_TARGET_AREA * ratio) ** 0.5
            target_height = target_width / ratio
            width = int(round(target_width))
            height = int(round(target_height))
        multiple = 16
        width = max(multiple, (width // multiple) * multiple)
        height = max(multiple, (height // multiple) * multiple)
        return width, height
