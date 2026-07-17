from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.inference_defaults import default_inference_steps
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition
from mflux.models.z_image.z_image_initializer import ZImageInitializer
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.dimension_resolver import CANVAS_POLICY_SOURCE_ASPECT
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil
from mflux.utils.mask_util import MaskUtil
from mflux.utils.runtime_timer import RuntimeTimer
from mflux.utils.scale_factor import ScaleFactor


class ZImage(nn.Module):
    vae: VAE
    text_encoder: TextEncoder
    transformer: ZImageTransformer

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig = ModelConfig.z_image_turbo(),
    ):
        super().__init__()
        ZImageInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
        height: int | ScaleFactor | None = None,
        width: int | ScaleFactor | None = None,
        guidance: float | None = None,
        image_path: Path | str | None = None,
        mask_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = None,
        negative_prompt: str | None = None,
        canvas_policy: str = CANVAS_POLICY_SOURCE_ASPECT,
    ) -> Image.Image:
        timer = RuntimeTimer()
        if mask_path is not None and image_path is None:
            raise ValueError("mask_path requires image_path for native Z-Image inpaint.")
        if mask_path is not None and image_strength is not None:
            raise ValueError(
                "image_strength cannot be combined with mask_path; native Z-Image inpaint is a separate route."
            )
        supports_guidance = bool(self.model_config.supports_guidance)
        if not supports_guidance:
            guidance = 0.0

        if scheduler is None:
            scheduler = "flow_match_euler_discrete" if supports_guidance else "linear"
        if num_inference_steps is None:
            num_inference_steps = default_inference_steps(self.model_config, fallback=4)

        # 0. Create a new config based on the model type and input parameters
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path,
            image_strength=image_strength,
            masked_image_path=mask_path,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            canvas_policy=canvas_policy,
            preserve_image_aspect_ratio=image_path is not None and canvas_policy == CANVAS_POLICY_SOURCE_ASPECT,
        )
        # 1. Create the initial latents
        inpaint_latents = None
        if mask_path is not None:
            latents = ZImageLatentCreator.create_noise(seed, config.height, config.width)
            inpaint_latents = self._create_inpaint_latents(
                image_path=config.image_path,
                mask_path=mask_path,
                height=config.height,
                width=config.width,
                initial_noise=latents,
            )
        else:
            latents = LatentCreator.create_for_txt2img_or_img2img(
                seed=seed,
                width=config.width,
                height=config.height,
                img2img=Img2Img(
                    vae=self.vae,
                    latent_creator=ZImageLatentCreator,
                    image_path=config.image_path,
                    sigmas=config.scheduler.sigmas,
                    init_time_step=config.init_time_step,
                    image_strength=config.image_strength,
                    tiling_config=self.tiling_config,
                ),
            )
        text_encodings, negative_encodings = self._encode_prompts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=config.guidance,
        )

        # 3. Create callback context and call before_loop
        ctx = self.callbacks.start(
            seed=seed,
            prompt=prompt,
            config=config,
            task="image-to-image" if mask_path is not None else None,
        )
        ctx.before_loop(latents)
        predict = self._predict(self.transformer)

        for t in config.time_steps:
            try:
                # 4.t Predict the noise
                sigma_t = config.scheduler.sigmas[t].reshape((1,))
                timestep = mx.ones_like(sigma_t) - sigma_t
                noise = predict(
                    latents=latents,
                    timestep=timestep,
                    sigmas=config.scheduler.sigmas,
                    text_encodings=text_encodings,
                    negative_encodings=negative_encodings,
                    guidance=config.guidance,
                )

                # 5.t Take one denoise step
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                if inpaint_latents is not None:
                    sigma = config.scheduler.sigmas[t + 1] if t < config.num_inference_steps - 1 else 0.0
                    latents = self._blend_inpaint_latents(
                        latents=latents,
                        image_latents=inpaint_latents["image"],
                        initial_noise=inpaint_latents["noise"],
                        mask_latents=inpaint_latents["mask"],
                        sigma=sigma,
                    )

                # 6.t Call subscribers in-loop
                ctx.in_loop(t, latents)

                # (Optional) Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        # 7. Call subscribers after loop
        ctx.after_loop(latents)

        # 8. Decode the latents and return the image
        try:
            decoded = self._decode_latents(latents=latents, config=config)
            image = ImageUtil.to_image(
                decoded_latents=decoded,
                config=config,
                seed=seed,
                prompt=prompt,
                quantization=self.bits,
                lora_paths=self.lora_paths,
                lora_scales=self.lora_scales,
                image_path=config.image_path,
                image_strength=config.image_strength,
                masked_image_path=mask_path,
                generation_time=timer.elapsed_seconds(),
                negative_prompt=negative_prompt,
                extra_metadata=LoRALoader.extra_metadata_for_model(self),
            )
        except Exception:
            ctx.failed()
            raise
        ctx.complete()
        return image

    def _encode_prompts(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array | None]:
        text_encodings = PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["z_image"],
            text_encoder=self.text_encoder,
            prompt_cache=self.prompt_cache,
        )
        if guidance <= 1.0:
            return text_encodings, None
        negative_text = negative_prompt if negative_prompt and negative_prompt.strip() else " "
        negative_encodings = PromptEncoder.encode_prompt(
            prompt=negative_text,
            tokenizer=self.tokenizers["z_image"],
            text_encoder=self.text_encoder,
            prompt_cache=self.prompt_cache,
        )
        return text_encodings, negative_encodings

    def _decode_latents(self, *, latents: mx.array, config: Config) -> mx.array:
        unpacked = ZImageLatentCreator.unpack_latents(latents, config.height, config.width)
        return VAEUtil.decode(vae=self.vae, latent=unpacked, tiling_config=self.tiling_config)

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=ZImageWeightDefinition,
        )

    @staticmethod
    def _predict(transformer: ZImageTransformer):
        def predict(
            latents: mx.array,
            timestep: mx.array,
            sigmas: mx.array,
            text_encodings: mx.array,
            negative_encodings: mx.array | None,
            guidance: float,
        ) -> mx.array:
            noise = transformer(
                timestep=timestep,
                x=latents,
                cap_feats=text_encodings,
                sigmas=sigmas,
            )
            if negative_encodings is None:
                return noise
            negative_noise = transformer(
                timestep=timestep,
                x=latents,
                cap_feats=negative_encodings,
                sigmas=sigmas,
            )
            return negative_noise + guidance * (noise - negative_noise)

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)

    def _create_inpaint_latents(
        self,
        *,
        image_path: Path | str | None,
        mask_path: Path | str,
        height: int,
        width: int,
        initial_noise: mx.array,
    ) -> dict[str, mx.array]:
        if image_path is None:
            raise ValueError("image_path is required for native Z-Image inpaint.")
        cache_key = (self._path_signature(image_path), self._path_signature(mask_path), height, width)
        cached = self.inpaint_condition_cache.get(cache_key)
        if cached is not None:
            return {
                "image": cached["image"],
                "mask": cached["mask"],
                "noise": mx.stop_gradient(initial_noise),
            }
        encoded = LatentCreator.encode_image(
            vae=self.vae,
            image_path=image_path,
            height=height,
            width=width,
            tiling_config=self.tiling_config,
        )
        image_latents = mx.stop_gradient(ZImageLatentCreator.pack_latents(encoded, height, width))
        mask_latents = mx.stop_gradient(
            self._create_inpaint_mask_latents(mask_path=mask_path, height=height, width=width)
        )
        detached_noise = mx.stop_gradient(initial_noise)
        mx.eval(image_latents, mask_latents, detached_noise)
        self.inpaint_condition_cache[cache_key] = {
            "image": image_latents,
            "mask": mask_latents,
        }
        return {
            "image": image_latents,
            "mask": mask_latents,
            "noise": detached_noise,
        }

    @staticmethod
    def _path_signature(path: Path | str) -> tuple[str, int | None, int | None]:
        path_obj = Path(path)
        try:
            stat = path_obj.stat()
        except OSError:
            return (str(path_obj), None, None)
        return (str(path_obj), stat.st_mtime_ns, stat.st_size)

    @staticmethod
    def _create_inpaint_mask_latents(
        *,
        mask_path: Path | str,
        height: int,
        width: int,
    ) -> mx.array:
        latent_width = width // 8
        latent_height = height // 8
        mask_values = MaskUtil.load_binary_mask(
            mask_path,
            target_width=latent_width,
            target_height=latent_height,
            resampling=Image.Resampling.NEAREST,
            alpha_warning_context="Z-Image inpaint mask",
        )
        return mx.array(mask_values)[None, None, :, :]

    @staticmethod
    def _blend_inpaint_latents(
        *,
        latents: mx.array,
        image_latents: mx.array,
        initial_noise: mx.array,
        mask_latents: mx.array,
        sigma: mx.array | float,
    ) -> mx.array:
        init_latents = LatentCreator.add_noise_by_interpolation(clean=image_latents, noise=initial_noise, sigma=sigma)
        # The float32 mask would otherwise promote the composite (and every later step) to f32.
        return ((1 - mask_latents) * init_latents + mask_latents * latents).astype(latents.dtype)
