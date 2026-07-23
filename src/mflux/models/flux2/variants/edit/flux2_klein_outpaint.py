from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.inference_defaults import default_inference_steps
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
from mflux.utils.dimension_resolver import CANVAS_POLICY_EXACT_RESIZE
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil
from mflux.utils.outpaint_util import OutpaintCanvas
from mflux.utils.runtime_timer import RuntimeTimer


class Flux2KleinOutpaint(nn.Module):
    vae: Flux2VAE
    transformer: Flux2Transformer
    text_encoder: Qwen3TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig | None = None,
    ):
        super().__init__()
        Flux2Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config or ModelConfig.flux2_klein_base_9b(),
        )

    def generate_image(
        self,
        *,
        seed: int,
        prompt: str,
        canvas: OutpaintCanvas,
        num_inference_steps: int | None = None,
        guidance: float = 4.0,
        scheduler: str = "flow_match_euler_discrete",
        image_strength: float = 1.0,
        reference_image_paths: list[Path | str] | None = None,
    ) -> GeneratedImage:
        timer = RuntimeTimer()
        if num_inference_steps is None:
            num_inference_steps = default_inference_steps(self.model_config, fallback=20)
        reference_image_paths = [canvas.canvas_path] if reference_image_paths is None else reference_image_paths
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=canvas.target_height,
            width=canvas.target_width,
            guidance=guidance,
            image_path=canvas.canvas_path,
            image_strength=image_strength,
            scheduler=scheduler,
            canvas_policy=CANVAS_POLICY_EXACT_RESIZE,
        )
        prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids = self._encode_prompt_pair(
            prompt=prompt,
            negative_prompt="",
            guidance=guidance,
        )

        noise_latents, latent_ids, latent_height, latent_width = Flux2LatentCreator.prepare_packed_latents(
            seed=seed,
            height=config.height,
            width=config.width,
            batch_size=1,
        )
        clean_latents = _Flux2KleinEditHelpers.encode_reference_image_to_packed_latents(
            vae=self.vae,
            tiling_config=self.tiling_config,
            image_path=canvas.canvas_path,
            height=config.height,
            width=config.width,
        )
        latents = LatentCreator.add_noise_by_interpolation(
            clean=clean_latents,
            noise=noise_latents,
            sigma=config.scheduler.sigmas[config.init_time_step],
        )
        image_latents, image_latent_ids = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
            vae=self.vae,
            tiling_config=self.tiling_config,
            image_paths=reference_image_paths,
            height=config.height,
            width=config.width,
            batch_size=latents.shape[0],
        )
        editable_mask = _Flux2KleinEditHelpers.prepare_outpaint_edit_mask(
            canvas=canvas,
            height=config.height,
            width=config.width,
            batch_size=latents.shape[0],
        ).astype(latents.dtype)

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config, task="image-to-image")
        ctx.before_loop(latents)
        # Reuse the compiled predict across calls on a resident instance (0095).
        predict = self.compiled_predict_cache.get_or_build(
            key=("edit", negative_prompt_embeds is not None),
            weights_token=self.transformer,
            build=lambda: Flux2KleinEdit._predict(self.transformer),
        )
        for t in config.time_steps:
            try:
                noise = predict(
                    latents=latents,
                    image_latents=image_latents,
                    latent_ids=latent_ids,
                    image_latent_ids=image_latent_ids,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                    guidance=guidance,
                    timestep=config.scheduler.timesteps[t],
                )
                latents = config.scheduler.step(
                    noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas
                )
                preserved_latents = self._preserved_source_latents(
                    clean_latents=clean_latents,
                    noise_latents=noise_latents,
                    sigmas=config.scheduler.sigmas,
                    timestep=t,
                ).astype(latents.dtype)
                latents = ((1.0 - editable_mask) * preserved_latents + editable_mask * latents).astype(latents.dtype)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)
        try:
            packed_latents = latents.reshape(latents.shape[0], latent_height, latent_width, latents.shape[-1]).transpose(0, 3, 1, 2)  # fmt: off
            decoded = self.vae.decode_packed_latents(packed_latents)
            image = ImageUtil.to_image(
                decoded_latents=decoded,
                config=config,
                seed=seed,
                prompt=prompt,
                negative_prompt=None,
                quantization=self.bits,
                lora_paths=self.lora_paths,
                lora_scales=self.lora_scales,
                image_paths=[canvas.source_path],
                image_path=canvas.source_path,
                generation_time=timer.elapsed_seconds(),
                extra_metadata=LoRALoader.extra_metadata_for_model(self),
            )
        except Exception:
            ctx.failed()
            raise
        ctx.complete()
        return image

    @staticmethod
    def _preserved_source_latents(
        *,
        clean_latents: mx.array,
        noise_latents: mx.array,
        sigmas: mx.array,
        timestep: int,
    ) -> mx.array:
        return _Flux2KleinEditHelpers.preserved_source_latents(
            clean_latents=clean_latents,
            noise_latents=noise_latents,
            sigmas=sigmas,
            timestep=timestep,
        )

    def _encode_prompt_pair(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        return Flux2KleinEdit._encode_prompt_pair(
            self,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=guidance,
        )
