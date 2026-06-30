import math
from pathlib import Path

import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.qwen_initializer import QwenImageInitializer
from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
from mflux.utils.dimension_resolver import CANVAS_POLICY_SOURCE_ASPECT
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil
from mflux.utils.runtime_memory import RuntimeMemory
from mflux.utils.runtime_timer import RuntimeTimer
from mflux.utils.scale_factor import ScaleFactor


class QwenImageEdit(nn.Module):
    vae: QwenVAE
    transformer: QwenTransformer
    text_encoder: QwenTextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig = ModelConfig.qwen_image_edit(),
    ):
        super().__init__()
        QwenImageInitializer.init_edit(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=QwenWeightDefinition,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        image_paths: list[str],
        num_inference_steps: int | None = None,
        height: int | ScaleFactor | None = None,
        width: int | ScaleFactor | None = None,
        guidance: float = 4.0,
        image_path: Path | str | None = None,
        mask_path: Path | str | None = None,
        scheduler: str = "flow_match_euler_discrete",
        negative_prompt: str | None = None,
        canvas_policy: str = CANVAS_POLICY_SOURCE_ASPECT,
    ) -> GeneratedImage:
        timer = RuntimeTimer()
        num_inference_steps = self._default_num_inference_steps(num_inference_steps, image_paths=image_paths)
        config, vl_width, vl_height, _, _ = self._compute_dimensions(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path,
            image_paths=image_paths,
            num_inference_steps=num_inference_steps,
            canvas_policy=canvas_policy,
        )
        timesteps = config.scheduler.timesteps
        time_steps = tqdm(range(len(timesteps)))
        negative_prompt = self._resolve_negative_prompt_for_model(
            guidance=config.guidance,
            negative_prompt=negative_prompt,
            image_paths=image_paths,
        )

        # 1. Create initial latents
        latents = QwenLatentCreator.create_noise(
            seed=seed,
            width=config.width,
            height=config.height,
        )
        initial_noise = latents

        # 2. Encode the prompt
        do_true_cfg = self._should_use_true_cfg(guidance=config.guidance, negative_prompt=negative_prompt)
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = self._encode_prompts_with_images(
            prompt=prompt,
            config=config,
            vl_width=vl_width,
            vl_height=vl_height,
            image_paths=image_paths,
            negative_prompt=negative_prompt,
            encode_negative=do_true_cfg,
        )

        # 3. Generate image conditioning latents
        conditioning_width = config.width if mask_path is not None else None
        conditioning_height = config.height if mask_path is not None else None
        static_image_latents, qwen_image_ids, cond_image_grid, _ = QwenEditUtil.create_image_conditioning_latents(
            vae=self.vae,
            width=conditioning_width,
            height=conditioning_height,
            image_paths=image_paths,
            tiling_config=self.tiling_config,
        )
        mask_latents = None
        if mask_path is not None:
            mask_latents = QwenEditUtil.create_inpaint_mask_latents(
                mask_path=str(mask_path),
                height=config.height,
                width=config.width,
            ).astype(static_image_latents.dtype)

        # 4. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config, task="image-to-image")
        ctx.before_loop(latents)

        for t in time_steps:
            try:
                # 5.t Concatenate the updated latents with the static image latents
                hidden_states = mx.concatenate([latents, static_image_latents], axis=1)

                # 6.t Predict the noise
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=hidden_states,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                    qwen_image_ids=qwen_image_ids,
                    cond_image_grid=cond_image_grid,
                )[:, : latents.shape[1]]
                if do_true_cfg:
                    noise_negative = self.transformer(
                        t=t,
                        config=config,
                        hidden_states=hidden_states,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_mask=negative_prompt_mask,
                        qwen_image_ids=qwen_image_ids,
                        cond_image_grid=cond_image_grid,
                    )[:, : latents.shape[1]]
                    guided_noise = QwenImage.compute_guided_noise(noise, noise_negative, config.guidance)
                else:
                    guided_noise = noise

                # 7.t Take one denoise step
                latents = config.scheduler.step(noise=guided_noise, timestep=t, latents=latents)
                if mask_latents is not None:
                    sigma = config.scheduler.sigmas[t + 1] if t < len(timesteps) - 1 else 0.0
                    latents = QwenEditUtil.blend_inpaint_latents(
                        latents=latents,
                        image_latents=static_image_latents,
                        initial_noise=initial_noise,
                        mask_latents=mask_latents,
                        sigma=sigma,
                    )

                # 8.t Call subscribers in-loop
                ctx.in_loop(t, latents, time_steps=time_steps)

                # (Optional) Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents, time_steps=time_steps)
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{len(timesteps)}")

        # 9. Call subscribers after loop
        ctx.after_loop(latents)

        # 10. Decode the latent array and return the image
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
                image_paths=image_paths,
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

    def _encode_prompts_with_images(
        self,
        prompt: str,
        negative_prompt: str,
        image_paths: list[str],
        config,
        vl_width: int | None = None,
        vl_height: int | None = None,
        encode_negative: bool = True,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        tokenizer = self.tokenizers["qwen_vl"]
        use_picture_prefix = self._should_use_edit_plus_prompt(image_paths=image_paths)
        pos_input_ids, pos_attention_mask, pos_pixel_values, pos_image_grid_thw = tokenizer.tokenize_with_image(
            prompt,
            image_paths,
            vl_width=vl_width,
            vl_height=vl_height,
            use_picture_prefix=use_picture_prefix,
        )

        pos_hidden_states = self.qwen_vl_encoder(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            pixel_values=pos_pixel_values,
            image_grid_thw=pos_image_grid_thw,
        )

        final_prompt_embeds = pos_hidden_states[0].astype(mx.float16)
        final_prompt_mask = pos_hidden_states[1].astype(mx.float16)
        if not encode_negative:
            return RuntimeMemory.materialize_inference_tree((final_prompt_embeds, final_prompt_mask, None, None))

        neg_prompt = negative_prompt if negative_prompt is not None else ""
        neg_input_ids, neg_attention_mask, neg_pixel_values, neg_image_grid_thw = tokenizer.tokenize_with_image(
            neg_prompt,
            image_paths,
            vl_width=vl_width,
            vl_height=vl_height,
            use_picture_prefix=use_picture_prefix,
        )

        neg_hidden_states = self.qwen_vl_encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            pixel_values=neg_pixel_values,
            image_grid_thw=neg_image_grid_thw,
        )

        return RuntimeMemory.materialize_inference_tree((
            final_prompt_embeds,  # prompt_embeds
            final_prompt_mask,  # prompt_mask
            neg_hidden_states[0].astype(mx.float16),  # negative_prompt_embeds
            neg_hidden_states[1].astype(mx.float16),  # negative_prompt_mask
        ))

    @staticmethod
    def _should_use_true_cfg(guidance: float, negative_prompt: str | None) -> bool:
        return guidance > 1.0 and negative_prompt is not None

    @staticmethod
    def _resolve_negative_prompt(guidance: float, negative_prompt: str | None) -> str | None:
        if guidance > 1.0 and negative_prompt is None:
            return " "
        return negative_prompt

    def _should_use_edit_plus_prompt(self, image_paths: list[str]) -> bool:
        return QwenImageEdit._is_edit_plus_model_config(
            model_config=self.model_config,
            image_paths=image_paths,
        )

    @staticmethod
    def _is_edit_plus_model_config(model_config: ModelConfig, image_paths: list[str]) -> bool:
        del image_paths
        return bool(getattr(model_config, "transformer_overrides", {}).get("qwen_edit_plus", False))

    def _compute_dimensions(
        self,
        image_paths: list[str],
        num_inference_steps: int,
        height: int | ScaleFactor | None,
        width: int | ScaleFactor | None,
        guidance: float,
        image_path: Path | str | None,
        scheduler: str,
        canvas_policy: str = CANVAS_POLICY_SOURCE_ASPECT,
    ) -> tuple[Config, int, int, int, int]:
        reference_image_path = QwenImageEdit._dimension_reference_image_path(image_paths=image_paths)
        reference_image = ImageUtil.load_image(reference_image_path).convert("RGB")
        image_size = reference_image.size
        default_width, default_height = QwenEditUtil._area_dimensions(
            target_area=QwenEditUtil.VAE_IMAGE_SIZE,
            ratio=image_size[0] / image_size[1],
        )
        if QwenImageEdit._is_auto_dimension(width):
            width = default_width
        if QwenImageEdit._is_auto_dimension(height):
            height = default_height

        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path or reference_image_path,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            canvas_policy=canvas_policy,
            preserve_image_aspect_ratio=canvas_policy == CANVAS_POLICY_SOURCE_ASPECT,
        )
        use_width = config.width
        use_height = config.height

        condition_image_size = (
            QwenEditUtil.CONDITION_IMAGE_SIZE
            if QwenImageEdit._is_edit_plus_model_config(model_config=self.model_config, image_paths=image_paths)
            else QwenEditUtil.VAE_IMAGE_SIZE
        )
        condition_ratio = image_size[0] / image_size[1]
        vl_width = math.sqrt(condition_image_size * condition_ratio)
        vl_height = vl_width / condition_ratio
        vl_width = round(vl_width / 32) * 32
        vl_height = round(vl_height / 32) * 32

        return config, int(vl_width), int(vl_height), use_width, use_height

    def _resolve_negative_prompt_for_model(
        self,
        guidance: float,
        negative_prompt: str | None,
        image_paths: list[str],
    ) -> str | None:
        del image_paths
        return QwenImageEdit._resolve_negative_prompt(guidance=guidance, negative_prompt=negative_prompt)

    def _default_num_inference_steps(self, num_inference_steps: int | None, image_paths: list[str]) -> int:
        if num_inference_steps is not None:
            return num_inference_steps
        if QwenImageEdit._is_edit_plus_model_config(model_config=self.model_config, image_paths=image_paths):
            return 40
        return 50

    @staticmethod
    def _dimension_reference_image_path(image_paths: list[str]) -> str:
        return image_paths[0]

    @staticmethod
    def _is_auto_dimension(value: int | ScaleFactor | None) -> bool:
        return value is None or isinstance(value, ScaleFactor) and value.value == 1
