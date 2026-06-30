from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.config.inference_defaults import default_inference_steps
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_controlnet import QwenTransformerControlNet
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.qwen_initializer import QwenImageInitializer
from mflux.models.qwen.variants.controlnet.qwen_controlnet_util import QwenControlNetUtil
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
from mflux.utils.dimension_resolver import CANVAS_POLICY_SOURCE_ASPECT
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil
from mflux.utils.runtime_timer import RuntimeTimer
from mflux.utils.scale_factor import ScaleFactor


class QwenImageControlNet(nn.Module):
    vae: QwenVAE
    transformer: QwenTransformer
    transformer_controlnet: QwenTransformerControlNet
    text_encoder: QwenTextEncoder

    def __init__(
        self,
        *,
        controlnet_model: str,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig = ModelConfig.qwen_image(),
    ):
        super().__init__()
        QwenImageInitializer.init_controlnet(
            model=self,
            controlnet_model=controlnet_model,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config,
        )

    def generate_image(
        self,
        *,
        seed: int,
        prompt: str,
        controlnet_image_path: str | None = None,
        controlnet_strength: float = 0.85,
        num_inference_steps: int | None = None,
        height: int | ScaleFactor | None = None,
        width: int | ScaleFactor | None = None,
        guidance: float = 4.0,
        scheduler: str = "flow_match_euler_discrete",
        negative_prompt: str | None = None,
        image_path: Path | str | None = None,
        mask_path: Path | str | None = None,
        canvas_policy: str = CANVAS_POLICY_SOURCE_ASPECT,
    ) -> GeneratedImage:
        timer = RuntimeTimer()
        if mask_path is not None and image_path is None:
            raise ValueError("mask_path requires image_path for Qwen control-inpaint.")
        if mask_path is not None and controlnet_image_path is not None:
            raise ValueError("Qwen control-inpaint uses image_path + mask_path and cannot be combined with controlnet_image_path.")
        if mask_path is None and controlnet_image_path is None:
            raise ValueError("Qwen ControlNet generation requires either controlnet_image_path or image_path + mask_path.")
        if num_inference_steps is None:
            num_inference_steps = default_inference_steps(self.model_config, fallback=4)
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            controlnet_strength=controlnet_strength,
            canvas_policy=canvas_policy,
            preserve_image_aspect_ratio=image_path is not None and canvas_policy == CANVAS_POLICY_SOURCE_ASPECT,
        )
        negative_prompt = self._resolve_negative_prompt(guidance=config.guidance, negative_prompt=negative_prompt)
        use_cfg = QwenImageControlNet._use_classifier_free_guidance(
            guidance=config.guidance,
            negative_prompt=negative_prompt,
        )
        controlnet_condition = self._resolve_controlnet_condition(
            controlnet_image_path=controlnet_image_path,
            image_path=image_path,
            mask_path=mask_path,
            height=config.height,
            width=config.width,
        )
        latents = QwenLatentCreator.create_noise(seed=seed, width=config.width, height=config.height)
        if use_cfg:
            prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = QwenPromptEncoder.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                prompt_cache=self.prompt_cache,
                qwen_tokenizer=self.tokenizers["qwen"],
                qwen_text_encoder=self.text_encoder,
            )
            effective_negative_prompt = negative_prompt
        else:
            prompt_embeds, prompt_mask = QwenPromptEncoder.encode_positive_prompt(
                prompt=prompt,
                prompt_cache=self.prompt_cache,
                qwen_tokenizer=self.tokenizers["qwen"],
                qwen_text_encoder=self.text_encoder,
            )
            negative_prompt_embeds = None
            negative_prompt_mask = None
            effective_negative_prompt = negative_prompt
        ctx = self.callbacks.start(
            seed=seed,
            prompt=prompt,
            config=config,
            task="image-to-image" if mask_path is not None else "text-to-image",
        )
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                latents = config.scheduler.scale_model_input(latents, t)
                controlnet_block_samples = self.transformer_controlnet(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    controlnet_cond=controlnet_condition,
                    conditioning_scale=controlnet_strength,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                )
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                    controlnet_block_samples=controlnet_block_samples,
                )
                if use_cfg:
                    noise_negative = self.transformer(
                        t=t,
                        config=config,
                        hidden_states=latents,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_mask=negative_prompt_mask,
                        controlnet_block_samples=controlnet_block_samples,
                    )
                    guided_noise = QwenImageControlNet.compute_guided_noise(noise, noise_negative, config.guidance)
                else:
                    guided_noise = noise
                latents = config.scheduler.step(noise=guided_noise, timestep=t, latents=latents)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )
        ctx.after_loop(latents)
        try:
            extra_metadata = LoRALoader.extra_metadata_for_model(self) or {}
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
                image_path=image_path,
                controlnet_image_path=controlnet_image_path,
                masked_image_path=mask_path,
                generation_time=timer.elapsed_seconds(),
                negative_prompt=effective_negative_prompt,
                extra_metadata={
                    **extra_metadata,
                    "controlnet_model": self.controlnet_model,
                },
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
        return combined * (cond_norm / noise_norm)

    @staticmethod
    def _use_classifier_free_guidance(guidance: float, negative_prompt: str | None) -> bool:
        return guidance > 1.0 and negative_prompt is not None

    @staticmethod
    def _resolve_negative_prompt(guidance: float, negative_prompt: str | None) -> str | None:
        if guidance > 1.0 and negative_prompt is None:
            return " "
        return negative_prompt

    def _resolve_controlnet_condition(
        self,
        *,
        controlnet_image_path: str | None,
        image_path: Path | str | None,
        mask_path: Path | str | None,
        height: int,
        width: int,
    ) -> mx.array:
        cache_key = self._controlnet_condition_cache_key(
            controlnet_image_path=controlnet_image_path,
            image_path=image_path,
            mask_path=mask_path,
            height=height,
            width=width,
        )
        if cache_key in self.controlnet_condition_cache:
            return self.controlnet_condition_cache[cache_key]
        if mask_path is not None:
            condition = QwenControlNetUtil.create_inpaint_controlnet_condition(
                vae=self.vae,
                image_path=str(image_path),
                mask_path=str(mask_path),
                height=height,
                width=width,
                tiling_config=self.tiling_config,
            )
        else:
            condition = QwenControlNetUtil.create_controlnet_condition(
                vae=self.vae,
                controlnet_image_path=controlnet_image_path,
                height=height,
                width=width,
                tiling_config=self.tiling_config,
            )
        condition = mx.stop_gradient(condition)
        mx.eval(condition)
        self.controlnet_condition_cache[cache_key] = condition
        return condition

    def _controlnet_condition_cache_key(
        self,
        *,
        controlnet_image_path: str | None,
        image_path: Path | str | None,
        mask_path: Path | str | None,
        height: int,
        width: int,
    ) -> tuple[str, str, tuple[str, int | None, int | None] | None, tuple[str, int | None, int | None] | None, int, int]:
        if mask_path is not None:
            return (
                "inpaint",
                self.controlnet_model,
                self._path_signature(image_path),
                self._path_signature(mask_path),
                height,
                width,
            )
        return ("control", self.controlnet_model, self._path_signature(controlnet_image_path), None, height, width)

    @staticmethod
    def _path_signature(path: Path | str | None) -> tuple[str, int | None, int | None] | None:
        if path is None:
            return None
        path_obj = Path(path)
        try:
            stat = path_obj.stat()
        except OSError:
            return (str(path_obj), None, None)
        return (str(path_obj), stat.st_mtime_ns, stat.st_size)
