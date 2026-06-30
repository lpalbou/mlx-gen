from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.inference_defaults import default_inference_steps
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.dimension_resolver import CANVAS_POLICY_SOURCE_ASPECT
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil
from mflux.utils.runtime_timer import RuntimeTimer
from mflux.utils.scale_factor import ScaleFactor


class Flux2KleinEdit(nn.Module):
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
            model_config=model_config or ModelConfig.flux2_klein_4b(),
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
        height: int | ScaleFactor | None = None,
        width: int | ScaleFactor | None = None,
        guidance: float = 1.0,
        image_paths: list[Path | str] | None = None,
        image_strength: float | None = None,
        scheduler: str = "flow_match_euler_discrete",
        canvas_policy: str = CANVAS_POLICY_SOURCE_ASPECT,
    ) -> GeneratedImage:
        timer = RuntimeTimer()
        if num_inference_steps is None:
            num_inference_steps = default_inference_steps(self.model_config, fallback=4)
        self._validate_guidance(guidance)
        if image_strength is not None:
            raise ValueError(
                "image_strength is only supported for latent image-to-image mode, not edit-reference mode."
            )

        # For metadata + dimension inference purposes, pick a primary reference image (if any).
        primary_image_path = None
        if image_paths:
            primary_image_path = image_paths[0]

        # 0. Create a new config based on the model type and input parameters
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=primary_image_path,
            image_strength=image_strength,
            scheduler=scheduler,
            canvas_policy=canvas_policy,
            preserve_image_aspect_ratio=primary_image_path is not None and canvas_policy == CANVAS_POLICY_SOURCE_ASPECT,
        )
        # 1. Encode prompt(s)
        prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids = self._encode_prompt_pair(
            prompt=prompt,
            negative_prompt="",
            guidance=guidance,
        )

        # 2. Prepare latents
        latents, latent_ids, latent_height, latent_width = _Flux2KleinEditHelpers.prepare_generation_latents(
            seed=seed,
            height=config.height,
            width=config.width,
        )

        # 3. Reference image conditioning (edit-style, concat reference tokens)
        image_latents, image_latent_ids = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
            vae=self.vae,
            tiling_config=self.tiling_config,
            image_paths=image_paths,
            height=config.height,
            width=config.width,
            batch_size=latents.shape[0],
        )

        # 4. Denoising loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config, task="image-to-image")
        ctx.before_loop(latents)
        predict = self._predict(self.transformer)
        for t in config.time_steps:
            try:
                # 4.t Predict the noise
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

                # 5.t Take one denoise step
                latents = config.scheduler.step(
                    noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas
                )

                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        # 6. Decode latents
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
                image_paths=image_paths,
                image_path=config.image_path,
                generation_time=timer.elapsed_seconds(),
                extra_metadata=LoRALoader.extra_metadata_for_model(self),
            )
        except Exception:
            ctx.failed()
            raise
        ctx.complete()
        return image

    def _encode_prompt_pair(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        prompt_embeds, text_ids = _Flux2KleinEditHelpers.encode_text(
            prompt,
            tokenizer=self.tokenizers["qwen3"],
            text_encoder=self.text_encoder,
        )
        negative_prompt_embeds = None
        negative_text_ids = None
        if guidance is not None and guidance > 1.0 and negative_prompt is not None:
            negative_prompt_embeds, negative_text_ids = _Flux2KleinEditHelpers.encode_text(
                negative_prompt,
                tokenizer=self.tokenizers["qwen3"],
                text_encoder=self.text_encoder,
            )
        return prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids

    @staticmethod
    def _predict(transformer):
        def predict(
            latents: mx.array,
            image_latents: mx.array,
            latent_ids: mx.array,
            image_latent_ids: mx.array,
            prompt_embeds: mx.array,
            text_ids: mx.array,
            negative_prompt_embeds: mx.array | None,
            negative_text_ids: mx.array | None,
            guidance: float,
            timestep: mx.array,
        ) -> mx.array:
            hidden_states = mx.concatenate([latents, image_latents], axis=1)
            img_ids = mx.concatenate([latent_ids, image_latent_ids], axis=1)

            noise = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=text_ids,
                guidance=None,
            )
            noise = noise[:, : latents.shape[1]]
            if negative_prompt_embeds is not None and negative_text_ids is not None:
                negative_noise = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=negative_text_ids,
                    guidance=None,
                )
                negative_noise = negative_noise[:, : latents.shape[1]]
                noise = negative_noise + guidance * (noise - negative_noise)
            return noise

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)

    def _validate_guidance(self, guidance: float) -> None:
        if guidance == 1.0:
            return
        if self._is_base_model():
            return
        raise ValueError("guidance > 1.0 is only supported for FLUX.2 Klein base models.")

    def _is_base_model(self) -> bool:
        model_name_lower = self.model_config.model_name.lower()
        base_model_lower = (self.model_config.base_model or "").lower()
        return "klein-base" in model_name_lower or "klein-base" in base_model_lower
