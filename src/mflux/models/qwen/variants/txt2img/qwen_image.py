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
        image_strength: float | None = None,
        scheduler: str = "flow_match_euler_discrete",
        negative_prompt: str | None = None,
        canvas_policy: str = CANVAS_POLICY_SOURCE_ASPECT,
    ) -> GeneratedImage:
        timer = RuntimeTimer()
        # 0. Create a new config based on the model type and input parameters
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path,
            image_strength=image_strength,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            canvas_policy=canvas_policy,
            preserve_image_aspect_ratio=image_path is not None and canvas_policy == CANVAS_POLICY_SOURCE_ASPECT,
        )

        # 1. Create the initial latents
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
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
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
                image_strength=config.image_strength,
                generation_time=timer.elapsed_seconds(),
                negative_prompt=negative_prompt,
                extra_metadata=LoRALoader.extra_metadata_for_model(self),
            )
        except Exception:
            ctx.failed()
            raise
        ctx.complete()
        return image

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
