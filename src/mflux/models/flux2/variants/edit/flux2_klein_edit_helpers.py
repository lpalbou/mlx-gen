from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.resolution.config_resolution import ConfigResolution
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
        model_name_lower = ConfigResolution.identifying_name(model_config.model_name).lower()
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
    def validate_negative_prompt(*, model_config, guidance: float, negative_prompt: str | None) -> None:
        """Reject a negative prompt the weights cannot act on.

        The negative branch only runs when guidance is above 1.0 (see `_encode_prompt_pair`), so a
        negative prompt on distilled Klein - which has no guidance branch at all - or on base Klein
        at guidance 1.0 would be silently ignored. Saying so is better than a silent no-op.
        """
        if not negative_prompt:
            return
        if not _Flux2KleinEditHelpers.is_base_model(model_config):
            raise ValueError(
                "--negative-prompt is not supported for FLUX.2 Klein distilled weights: they are step-distilled "
                "and run no classifier-free guidance branch to steer. FLUX.2 Klein base models accept it with "
                "--guidance above 1.0."
            )
        if guidance is None or guidance <= 1.0:
            raise ValueError(
                "--negative-prompt on FLUX.2 Klein base weights needs --guidance above 1.0: the negative branch "
                "only runs under classifier-free guidance."
            )

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
        prompt_cache: dict | None = None,
    ) -> tuple[mx.array, mx.array]:
        return Flux2PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
            prompt_cache=prompt_cache,
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
        canvas_image_index: int | None = 0,
    ):
        if not image_paths:
            return None, None

        packed_latents_list: list[mx.array] = []
        ids_list: list[mx.array] = []
        for i, p in enumerate(image_paths):
            if i == canvas_image_index:
                # The primary (edited) image is conditioned at the resolved
                # generation canvas with a plain resize: reference and target
                # grids share the same top-left-anchored extent, and no source
                # pixels are cropped away. Conditioning it at source-derived
                # dims instead loses a floor-16 sliver per pass ("crop"), which
                # compounds into visible horizontal drift across iterative
                # edit chains.
                encode_width, encode_height = width, height
                reference_resize_mode = "resize"
            else:
                # Secondary references keep their own per-image sizing; they
                # describe content, not the output geometry.
                encode_width, encode_height = _Flux2KleinEditHelpers.reference_condition_dimensions(image_path=p)
                reference_resize_mode = "crop"
            encoded = LatentCreator.encode_image(
                vae=vae,
                image_path=p,
                height=encode_height,
                width=encode_width,
                tiling_config=tiling_config,
                resize_mode=reference_resize_mode,
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
    def outpaint_generated_gaps(canvas: OutpaintCanvas) -> tuple[int, int, int, int]:
        # Per-side count of NEW canvas pixels, as (left, top, right, bottom). This is the
        # ground truth rather than `canvas.padding` alone: OutpaintUtil rounds the target
        # dimensions up to a multiple of 16, so a side requested as 0 can still carry a few
        # pixels of synthetic filler on the right/bottom. Left/top always equal
        # padding.left/padding.top; right/bottom are padding plus that rounding sliver.
        return (
            max(0, canvas.paste_left),
            max(0, canvas.paste_top),
            max(0, canvas.target_width - canvas.paste_left - canvas.source_width),
            max(0, canvas.target_height - canvas.paste_top - canvas.source_height),
        )

    @staticmethod
    def outpaint_preserve_box(*, canvas: OutpaintCanvas, transition_px: int = 24) -> tuple[int, int, int, int]:
        # The transition band exists so the model can blend NEW canvas into the source across
        # the seam. A side with nothing new on it has no seam, so it gets no inset: insetting
        # unconditionally handed real source pixels to the model for free (with
        # `--outpaint-padding "0%,10%,100%,10%"` the top 24 canvas rows - the top of the
        # subject's head - were marked fully editable and regenerated).
        #
        # The inset is also capped at the size of the gap, so a side that only gained a few
        # pixels from the dimension round-up never sacrifices more source than it adds.
        #
        # transition_px stays ABSOLUTE and does not scale with the canvas: the latent grid is
        # always canvas/16, so 24 canvas px is 1.5 latent cells at every resolution (measured:
        # the post-resize seam profile is identical at 256x256 and 4096x4096).
        gap_left, gap_top, gap_right, gap_bottom = _Flux2KleinEditHelpers.outpaint_generated_gaps(canvas)
        # Per-axis half-extent cap keeps opposing insets from meeting (left+right <= width - 2).
        max_inset_x = max(0, canvas.source_width // 2 - 1)
        max_inset_y = max(0, canvas.source_height // 2 - 1)
        preserve_left = canvas.paste_left + min(transition_px, gap_left, max_inset_x)
        preserve_top = canvas.paste_top + min(transition_px, gap_top, max_inset_y)
        preserve_right = canvas.paste_left + canvas.source_width - min(transition_px, gap_right, max_inset_x)
        preserve_bottom = canvas.paste_top + canvas.source_height - min(transition_px, gap_bottom, max_inset_y)
        if preserve_right <= preserve_left or preserve_bottom <= preserve_top:
            return (
                canvas.paste_left,
                canvas.paste_top,
                canvas.paste_left + canvas.source_width,
                canvas.paste_top + canvas.source_height,
            )
        return preserve_left, preserve_top, preserve_right, preserve_bottom

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
        mask.paste(0, _Flux2KleinEditHelpers.outpaint_preserve_box(canvas=canvas, transition_px=transition_px))
        mask = mask.resize((latent_width, latent_height), resample=Image.Resampling.BILINEAR)
        mask_array = mx.array(np.asarray(mask, dtype=np.float32) / 255.0).reshape(1, latent_height * latent_width, 1)
        if batch_size > 1:
            mask_array = mx.broadcast_to(mask_array, (batch_size, mask_array.shape[1], mask_array.shape[2]))
        return mask_array

    @staticmethod
    def outpaint_source_cell_mask(*, canvas: OutpaintCanvas, height: int, width: int) -> np.ndarray:
        # True for every latent cell whose 16px canvas footprint holds at least one real
        # source pixel. A cell straddling the source boundary counts as source: it carries
        # the seam the model has to continue.
        latent_height = height // 16
        latent_width = width // 16
        rows = np.zeros(latent_height, dtype=bool)
        cols = np.zeros(latent_width, dtype=bool)
        rows[canvas.paste_top // 16 : (canvas.paste_top + canvas.source_height - 1) // 16 + 1] = True
        cols[canvas.paste_left // 16 : (canvas.paste_left + canvas.source_width - 1) // 16 + 1] = True
        return rows[:, None] & cols[None, :]

    @staticmethod
    def outpaint_reference_conditioning(
        *,
        image_latents: mx.array | None,
        image_latent_ids: mx.array | None,
        canvas: OutpaintCanvas,
        height: int,
        width: int,
    ) -> tuple[mx.array | None, mx.array | None]:
        # The canvas reference tokens sit at the same (h, w) rope coordinates as the generation
        # latents, one t index apart, and they stay clean at every step. A token for a cell that
        # holds nothing but synthetic filler therefore hands the model a noise-free copy of the
        # fill at exactly the position it is supposed to invent - the padded region is free in
        # the latent lock and not free in the conditioning, and reconstructing the reference is
        # then a perfectly good solution (the reported "un-denoised conditioning canvas": an
        # edge-fill smear, or a flat block under `neutral`).
        #
        # Cells holding any real source pixel are kept, so the source, the transition band and
        # every boundary cell still condition the run at their true positions.
        if image_latents is None or image_latent_ids is None:
            return image_latents, image_latent_ids
        canvas_tokens = (height // 16) * (width // 16)
        keep = _Flux2KleinEditHelpers.outpaint_source_cell_mask(canvas=canvas, height=height, width=width).reshape(-1)
        if image_latents.shape[1] < canvas_tokens or bool(keep.all()):
            return image_latents, image_latent_ids
        indices = mx.array(np.flatnonzero(keep).astype(np.int32))
        kept_latents = mx.take(image_latents[:, :canvas_tokens, :], indices, axis=1)
        kept_ids = mx.take(image_latent_ids[:, :canvas_tokens, :], indices, axis=1)
        if image_latents.shape[1] > canvas_tokens:
            # Secondary references describe content, not the output geometry, and are untouched.
            kept_latents = mx.concatenate([kept_latents, image_latents[:, canvas_tokens:, :]], axis=1)
            kept_ids = mx.concatenate([kept_ids, image_latent_ids[:, canvas_tokens:, :]], axis=1)
        return kept_latents, kept_ids

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
