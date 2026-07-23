from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.qwen_initializer import QwenImageInitializer
from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
from mflux.utils.dimension_resolver import CANVAS_POLICY_SOURCE_ASPECT
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil
from mflux.utils.runtime_timer import RuntimeTimer
from mflux.utils.scale_factor import ScaleFactor


class QwenImage(nn.Module):
    vae: QwenVAE
    transformer: QwenTransformer
    text_encoder: QwenTextEncoder

    # Upstream QwenImageInpaintPipeline inpaint example strength: the masked region starts
    # from the re-noised source so repainted content stays anchored to the surrounding
    # structure (base Qwen is a t2i model without edit-instruction training; a pure-noise
    # start paints unrelated content into the mask, while the signature default 0.6 barely
    # repaints at all). Measured trade-off at the extremes: 0.85 anchors structure (best for
    # retexture/removal but opaque recolors stay incomplete), 0.95 repaints fully (best for
    # recolors but thin connected structures can detach). The default stays 0.85;
    # mask_strength exposes the upstream knob for content-replacing edits.
    MASKED_EDIT_STRENGTH = 0.85

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig = ModelConfig.qwen_image(),
    ):
        super().__init__()
        QwenImageInitializer.init(
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
        num_inference_steps: int = 20,
        height: int | ScaleFactor | None = None,
        width: int | ScaleFactor | None = None,
        guidance: float = 4.0,
        image_path: Path | str | None = None,
        mask_path: Path | str | None = None,
        mask_strength: float | None = None,
        image_strength: float | None = None,
        scheduler: str = "flow_match_euler_discrete",
        negative_prompt: str | None = None,
        canvas_policy: str = CANVAS_POLICY_SOURCE_ASPECT,
        resize_mode: str = "resize",
    ) -> GeneratedImage:
        timer = RuntimeTimer()
        if mask_path is not None and image_path is None:
            raise ValueError("mask_path requires image_path for native base-Qwen masked edit.")
        if mask_path is not None and image_strength is not None:
            raise ValueError(
                "image_strength cannot be combined with mask_path; native masked edit is a separate route "
                "from latent image-to-image. Use mask_strength to tune the masked warm start."
            )
        if mask_strength is not None and mask_path is None:
            raise ValueError("mask_strength requires mask_path.")
        if mask_strength is not None and not 0.0 < mask_strength <= 1.0:
            raise ValueError("mask_strength must be in (0, 1].")
        # 0. Create a new config based on the model type and input parameters
        # Masked edit runs the upstream inpaint warm start internally; the public contract
        # keeps --image-strength and --mask-path mutually exclusive, and exposes the upstream
        # inpaint strength as the dedicated mask_strength knob instead.
        resolved_mask_strength = QwenImage.MASKED_EDIT_STRENGTH if mask_strength is None else mask_strength
        internal_image_strength = resolved_mask_strength if mask_path is not None else image_strength
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path,
            image_strength=internal_image_strength,
            masked_image_path=mask_path,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            canvas_policy=canvas_policy,
            resize_mode=resize_mode,
            preserve_image_aspect_ratio=image_path is not None and canvas_policy == CANVAS_POLICY_SOURCE_ASPECT,
        )

        # 1. Create the initial latents
        inpaint_latents = None
        if mask_path is not None:
            initial_noise = QwenLatentCreator.create_noise(seed, config.height, config.width)
            inpaint_latents = self._create_inpaint_latents(
                image_path=config.image_path,
                mask_path=mask_path,
                height=config.height,
                width=config.width,
                initial_noise=initial_noise,
                resize_mode=config.resize_mode,
            )
            latents = LatentCreator.add_noise_by_interpolation(
                clean=inpaint_latents["image"],
                noise=initial_noise,
                sigma=config.scheduler.sigmas[config.init_time_step],
            )
        else:
            latents = LatentCreator.create_for_txt2img_or_img2img(
                seed=seed,
                width=config.width,
                height=config.height,
                img2img=Img2Img(
                    vae=self.vae,
                    latent_creator=QwenLatentCreator,
                    sigmas=config.scheduler.sigmas,
                    init_time_step=config.init_time_step,
                    image_strength=config.image_strength,
                    image_path=config.image_path,
                    tiling_config=self.tiling_config,
                    resize_mode=config.resize_mode,
                ),
            )

        # 2. Encode the prompt (using native MLX encoding)
        negative_prompt = self._resolve_negative_prompt(guidance=guidance, negative_prompt=negative_prompt)
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = QwenPromptEncoder.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_cache=self.prompt_cache,
            qwen_tokenizer=self.tokenizers["qwen"],
            qwen_text_encoder=self.text_encoder,
        )
        do_true_cfg = guidance > 1.0 and negative_prompt_embeds is not None and negative_prompt_mask is not None

        # 3. Create callback context and call before_loop
        ctx = self.callbacks.start(
            seed=seed,
            prompt=prompt,
            config=config,
            task="image-to-image" if mask_path is not None else None,
        )
        ctx.before_loop(latents)

        for t in config.time_steps:
            try:
                # Scale model input if needed by the scheduler
                latents = config.scheduler.scale_model_input(latents, t)

                # 4. Predict the noise
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                )
                if do_true_cfg:
                    noise_negative = self.transformer(
                        t=t,
                        config=config,
                        hidden_states=latents,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_mask=negative_prompt_mask,
                    )
                    guided_noise = QwenImage.compute_guided_noise(noise, noise_negative, config.guidance)
                else:
                    guided_noise = noise

                # 5.t Take one denoise step
                latents = config.scheduler.step(noise=guided_noise, timestep=t, latents=latents)
                if inpaint_latents is not None:
                    sigma = config.scheduler.sigmas[t + 1] if t < config.num_inference_steps - 1 else 0.0
                    latents = QwenEditUtil.blend_inpaint_latents(
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

        # 8. Decode the latent array and return the image
        try:
            latents = QwenLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
            decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
            image = ImageUtil.to_image(
                decoded_latents=decoded,
                config=config,
                seed=seed,
                prompt=prompt,
                quantization=self.bits,
                lora_paths=self.lora_paths,
                lora_scales=self.lora_scales,
                image_path=config.image_path,
                # The masked warm start is reported as mask metadata, not as image_strength.
                image_strength=image_strength,
                masked_image_path=mask_path,
                generation_time=timer.elapsed_seconds(),
                negative_prompt=negative_prompt,
                extra_metadata=self._masked_run_metadata(
                    config=config,
                    mask_path=mask_path,
                    mask_strength=resolved_mask_strength if mask_path is not None else None,
                ),
            )
        except Exception:
            ctx.failed()
            raise
        ctx.complete()
        return image

    def _masked_run_metadata(
        self,
        *,
        config: Config,
        mask_path: Path | str | None,
        mask_strength: float | None,
    ) -> dict | None:
        # Runtime truth: the warm start truncates the schedule, so record the executed step
        # count and the applied strength beside the requested steps (Wan v2v precedent).
        extra_metadata = LoRALoader.extra_metadata_for_model(self) or {}
        if mask_path is None:
            return extra_metadata or None
        return {
            **extra_metadata,
            "effective_steps": config.num_inference_steps - config.init_time_step,
            "mask_strength": mask_strength,
        }

    def _create_inpaint_latents(
        self,
        *,
        image_path: Path | str,
        mask_path: Path | str,
        height: int,
        width: int,
        initial_noise: mx.array,
        resize_mode: str = "resize",
    ) -> dict[str, mx.array]:
        # Diffusers QwenImageInpaintPipeline semantics at the repo-wide full-strength mask
        # contract: encode the clean source once, keep unmasked latents locked to it by
        # per-step blending against the run's own initial noise.
        # The mask maps through the SAME source-to-canvas geometry as the image.
        encoded = LatentCreator.encode_image(
            vae=self.vae,
            image_path=image_path,
            height=height,
            width=width,
            tiling_config=self.tiling_config,
            resize_mode=resize_mode,
        )
        image_latents = mx.stop_gradient(
            QwenLatentCreator.pack_latents(latents=encoded, height=height, width=width)
        )
        mask_latents = mx.stop_gradient(
            QwenEditUtil.create_inpaint_mask_latents(
                mask_path=str(mask_path),
                height=height,
                width=width,
                resize_mode=resize_mode,
            ).astype(image_latents.dtype)
        )
        detached_noise = mx.stop_gradient(initial_noise)
        mx.eval(image_latents, mask_latents, detached_noise)
        return {
            "image": image_latents,
            "mask": mask_latents,
            "noise": detached_noise,
        }

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=QwenWeightDefinition,
        )

    @staticmethod
    def compute_guided_noise(
        noise: mx.array,
        noise_negative: mx.array,
        guidance: float,
    ) -> mx.array:
        combined = noise_negative + guidance * (noise - noise_negative)
        cond_norm = mx.sqrt(mx.sum(noise * noise, axis=-1, keepdims=True) + 1e-12)
        noise_norm = mx.sqrt(mx.sum(combined * combined, axis=-1, keepdims=True) + 1e-12)
        noise = combined * (cond_norm / noise_norm)
        return noise

    @staticmethod
    def _resolve_negative_prompt(guidance: float, negative_prompt: str | None) -> str | None:
        if guidance > 1.0 and negative_prompt is None:
            return " "
        return negative_prompt
