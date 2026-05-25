from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.ernie_image.ernie_image_initializer import ErnieImageInitializer
from mflux.models.ernie_image.latent_creator import ErnieImageLatentCreator
from mflux.models.ernie_image.model.ernie_image_transformer import ErnieImageTransformer2DModel
from mflux.models.ernie_image.model.ernie_image_vae import ErnieImageVAE
from mflux.models.ernie_image.model.mistral3_text_encoder import Mistral3TextEncoder
from mflux.models.ernie_image.scheduler import ErnieImageScheduler
from mflux.models.ernie_image.weights.ernie_image_weight_definition import ErnieImageWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil


class ErnieImageTurbo(nn.Module):
    vae: ErnieImageVAE
    transformer: ErnieImageTransformer2DModel
    text_encoder: Mistral3TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig = ModelConfig.ernie_image_turbo(),
    ):
        super().__init__()
        ErnieImageInitializer.init(
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
        num_inference_steps: int = 8,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = 1.0,
        negative_prompt: str | None = "",
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = None,
        use_pe: bool = False,
    ) -> Image.Image:
        del image_path, image_strength, scheduler, use_pe
        guidance = 1.0 if guidance is None else float(guidance)
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler="linear",
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
        )

        batch_size = 1
        latents = ErnieImageLatentCreator.create_noise(
            seed=seed,
            width=config.width,
            height=config.height,
            batch_size=batch_size,
        )
        text_hiddens = self._encode_prompt([prompt])
        if guidance > 1.0:
            uncond_hiddens = self._encode_prompt([negative_prompt or ""])
            cfg_text_hiddens = uncond_hiddens + text_hiddens
        else:
            cfg_text_hiddens = text_hiddens
        text_bth, text_lens = self._pad_text(cfg_text_hiddens)
        ernie_scheduler = ErnieImageScheduler(num_inference_steps=config.num_inference_steps)

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for step in config.time_steps:
            try:
                if guidance > 1.0:
                    latent_model_input = mx.concatenate([latents, latents], axis=0)
                else:
                    latent_model_input = latents
                timestep = mx.ones((latent_model_input.shape[0],), dtype=ModelConfig.precision)
                timestep = timestep * ernie_scheduler.timesteps[step].astype(ModelConfig.precision)
                pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    text_bth=text_bth,
                    text_lens=text_lens,
                )
                if guidance > 1.0:
                    pred_uncond, pred_cond = mx.split(pred, 2, axis=0)
                    pred = pred_uncond + guidance * (pred_cond - pred_uncond)

                latents = ernie_scheduler.step(noise=pred, timestep=step, latents=latents)
                ctx.in_loop(step, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(step, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {step + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)
        decoded = self.vae.decode_packed_latents(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            image_path=None,
            image_strength=None,
            generation_time=config.time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=ErnieImageWeightDefinition,
        )

    def _encode_prompt(self, prompts: list[str]) -> list[mx.array]:
        text_hiddens = []
        for prompt in prompts:
            output = self.tokenizers["ernie"].tokenize(prompt)
            hidden = self.text_encoder(output.input_ids, output.attention_mask)
            num_valid = int(mx.sum(output.attention_mask[0]).item())
            text_hiddens.append(hidden[0, :num_valid, :])
        return text_hiddens

    @staticmethod
    def _pad_text(text_hiddens: list[mx.array]) -> tuple[mx.array, mx.array]:
        if not text_hiddens:
            return mx.zeros((0, 0, 3072), dtype=ModelConfig.precision), mx.zeros((0,), dtype=mx.int32)
        lengths = [hidden.shape[0] for hidden in text_hiddens]
        max_len = max(lengths)
        padded = [
            mx.pad(hidden, [(0, max_len - hidden.shape[0]), (0, 0)]).astype(ModelConfig.precision)
            for hidden in text_hiddens
        ]
        return mx.stack(padded, axis=0), mx.array(lengths, dtype=mx.int32)
