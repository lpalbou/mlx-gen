from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.config.config import Config
from mflux.models.common.config.inference_defaults import default_inference_steps
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.fibo.fibo_initializer import FIBOInitializer
from mflux.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator
from mflux.models.fibo.model.fibo_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.fibo.model.fibo_text_encoder.smol_lm3_3b_text_encoder import SmolLM3_3B_TextEncoder
from mflux.models.fibo.model.fibo_transformer import FiboTransformer
from mflux.models.fibo.model.fibo_vae.wan_2_2_vae import Wan2_2_VAE
from mflux.models.fibo.variants.edit.util import FIBO_EDIT_DIMENSION_MULTIPLE, FiboEditUtil
from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition
from mflux.utils.dimension_resolver import CANVAS_POLICY_EXACT_RESIZE, CANVAS_POLICY_SOURCE_ASPECT
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil
from mflux.utils.scale_factor import ScaleFactor
from mflux.utils.tensor_health import TensorHealth


class FIBOEdit(nn.Module):
    vae: Wan2_2_VAE
    transformer: FiboTransformer
    text_encoder: SmolLM3_3B_TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig = ModelConfig.fibo_edit(),
    ):
        super().__init__()
        FIBOInitializer.init(
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
        image_path: Path | str,
        mask_path: Path | str | None = None,
        num_inference_steps: int | None = None,
        height: int | ScaleFactor | None = None,
        width: int | ScaleFactor | None = None,
        guidance: float = 5.0,
        scheduler: str = "flow_match_euler_discrete",
        negative_prompt: str | None = None,
        canvas_policy: str = CANVAS_POLICY_SOURCE_ASPECT,
    ) -> GeneratedImage:
        prompt = FiboEditUtil.ensure_edit_instruction(prompt)
        auto_canvas = width is None and height is None
        width, height = FiboEditUtil.resolve_preferred_canvas_size(
            image_path=image_path,
            width=width,
            height=height,
        )
        if num_inference_steps is None:
            num_inference_steps = default_inference_steps(self.model_config, fallback=50)
        config_canvas_policy = CANVAS_POLICY_EXACT_RESIZE if auto_canvas else canvas_policy

        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            canvas_policy=config_canvas_policy,
            preserve_image_aspect_ratio=config_canvas_policy == CANVAS_POLICY_SOURCE_ASPECT,
            dimension_multiple=FIBO_EDIT_DIMENSION_MULTIPLE,
        )
        if hasattr(config.scheduler, "set_mu"):
            mu = FlowMatchEulerDiscreteScheduler._compute_linear_mu(config.image_seq_len)
            config.scheduler.set_mu(mu)

        transformer_dtype = self._transformer_dtype()
        json_prompt, encoder_hidden_states, text_encoder_layers, prompt_attention_mask = PromptEncoder.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            tokenizer=self.tokenizers["fibo"],
            text_encoder=self.text_encoder,
            guidance=config.guidance,
            dtype=transformer_dtype,
            total_transformer_layers=self._total_transformer_layers(),
        )
        latents = FiboLatentCreator.create_noise(
            seed=seed,
            width=config.width,
            height=config.height,
            dtype=encoder_hidden_states.dtype,
        )
        TensorHealth.ensure_finite(latents, name="fibo.latents.initial", phase="fibo-edit-init")

        edit_image = FiboEditUtil.load_edit_image(
            image_path=image_path,
            width=config.width,
            height=config.height,
            mask_path=mask_path,
        )
        conditioning_latents = FiboEditUtil.encode_conditioning_image(
            vae=self.vae,
            image=edit_image,
            height=config.height,
            width=config.width,
            tiling_config=self.tiling_config,
            dtype=encoder_hidden_states.dtype,
        )
        TensorHealth.ensure_finite(
            conditioning_latents,
            name="fibo.conditioning_latents",
            phase="fibo-edit-conditioning",
        )
        conditioning_image_ids = FiboEditUtil.create_conditioning_image_ids(
            height=config.height,
            width=config.width,
            dtype=encoder_hidden_states.dtype,
        )

        ctx = self.callbacks.start(seed=seed, prompt=json_prompt, config=config, task="image-to-image")
        ctx.before_loop(latents)

        for t in config.time_steps:
            try:
                hidden_states = mx.concatenate([latents, conditioning_latents], axis=1)
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=hidden_states,
                    text_encoder_layers=text_encoder_layers,
                    encoder_hidden_states=encoder_hidden_states,
                    prompt_attention_mask=prompt_attention_mask,
                    conditioning_seq_len=conditioning_latents.shape[1],
                    conditioning_image_ids=conditioning_image_ids,
                )
                TensorHealth.ensure_finite(
                    noise,
                    name="fibo.transformer_noise",
                    phase="fibo-edit-denoise",
                    step=t + 1,
                    total_steps=config.num_inference_steps,
                    timestep=float(config.scheduler.timesteps[t].item()),
                    guidance=config.guidance,
                )
                noise = noise[:, : latents.shape[1]]
                if config.guidance > 1.0:
                    noise = FIBOEdit._apply_classifier_free_guidance(noise, config.guidance)
                    TensorHealth.ensure_finite(
                        noise,
                        name="fibo.guided_noise",
                        phase="fibo-edit-cfg",
                        step=t + 1,
                        total_steps=config.num_inference_steps,
                        timestep=float(config.scheduler.timesteps[t].item()),
                        guidance=config.guidance,
                    )
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                TensorHealth.ensure_finite(
                    latents,
                    name="fibo.latents",
                    phase="fibo-edit-scheduler",
                    step=t + 1,
                    total_steps=config.num_inference_steps,
                    timestep=float(config.scheduler.timesteps[t].item()),
                    guidance=config.guidance,
                )
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        try:
            latents = FiboLatentCreator.unpack_latents(latents, config.height, config.width)
            TensorHealth.ensure_finite(latents, name="fibo.latents.unpack", phase="fibo-edit-decode")
            decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
            image = ImageUtil.to_image(
                decoded_latents=decoded,
                config=config,
                seed=seed,
                prompt=json_prompt,
                quantization=self.bits,
                image_path=config.image_path,
                masked_image_path=mask_path,
                generation_time=config.time_steps.format_dict["elapsed"],
                negative_prompt=negative_prompt,
            )
        except Exception:
            ctx.failed()
            raise
        ctx.complete()
        return image

    @staticmethod
    def _apply_classifier_free_guidance(noise: mx.array, guidance: float) -> mx.array:
        half = noise.shape[0] // 2
        noise_uncond = noise[:half]
        noise_text = noise[half:]
        return noise_uncond + guidance * (noise_text - noise_uncond)

    def _total_transformer_layers(self) -> int:
        return len(self.transformer.transformer_blocks) + len(self.transformer.single_transformer_blocks)

    def _transformer_dtype(self) -> mx.Dtype:
        for _, value in tree_flatten(self.transformer.parameters()):
            if hasattr(value, "dtype") and value.dtype in {mx.float16, mx.bfloat16, mx.float32}:
                return value.dtype
        return ModelConfig.precision

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=FIBOWeightDefinition,
        )
