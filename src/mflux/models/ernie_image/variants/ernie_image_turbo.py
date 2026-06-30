import json
from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.ernie_image.ernie_image_initializer import ErnieImageInitializer
from mflux.models.ernie_image.latent_creator import ErnieImageLatentCreator
from mflux.models.ernie_image.model.ernie_image_transformer import ErnieImageTransformer2DModel
from mflux.models.ernie_image.model.ernie_image_vae import ErnieImageVAE
from mflux.models.ernie_image.model.mistral3_text_encoder import Mistral3TextEncoder
from mflux.models.ernie_image.scheduler import ErnieImageScheduler
from mflux.models.ernie_image.weights.ernie_image_weight_definition import ErnieImageWeightDefinition
from mflux.utils.dimension_resolver import CANVAS_POLICY_SOURCE_ASPECT
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil
from mflux.utils.runtime_memory import RuntimeMemory
from mflux.utils.runtime_timer import RuntimeTimer
from mflux.utils.scale_factor import ScaleFactor


class ErnieImageTurbo(nn.Module):
    vae: ErnieImageVAE
    transformer: ErnieImageTransformer2DModel
    text_encoder: Mistral3TextEncoder
    prompt_enhancer: nn.Module | None

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
        height: int | ScaleFactor | None = None,
        width: int | ScaleFactor | None = None,
        guidance: float | None = 1.0,
        negative_prompt: str | None = "",
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = None,
        use_pe: bool = False,
        pe_system_prompt: str | None = None,
        pe_temperature: float = 0.6,
        pe_top_p: float = 0.95,
        pe_max_new_tokens: int | None = None,
        canvas_policy: str = CANVAS_POLICY_SOURCE_ASPECT,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        extra_metadata: dict | None = None,
    ) -> Image.Image:
        timer = RuntimeTimer()
        del scheduler
        if image_path is None:
            image_strength = None
        if use_pe:
            pe_width = 1024 if width is None else width
            pe_height = 1024 if height is None else height
            prompt = self._enhance_prompt(
                prompt=prompt,
                width=pe_width,
                height=pe_height,
                seed=seed,
                system_prompt=pe_system_prompt,
                temperature=pe_temperature,
                top_p=pe_top_p,
                max_new_tokens=pe_max_new_tokens,
            )
        guidance = 1.0 if guidance is None else float(guidance)
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler="linear",
            image_path=image_path,
            image_strength=image_strength,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            canvas_policy=canvas_policy,
            preserve_image_aspect_ratio=image_path is not None and canvas_policy == CANVAS_POLICY_SOURCE_ASPECT,
        )

        batch_size = 1
        ernie_scheduler = ErnieImageScheduler(num_inference_steps=config.num_inference_steps)
        latents = self._prepare_generation_latents(
            seed=seed,
            config=config,
            scheduler=ernie_scheduler,
            batch_size=batch_size,
        )
        text_hiddens = self._encode_prompt([prompt])
        if guidance > 1.0:
            uncond_hiddens = self._encode_prompt([negative_prompt or ""])
            cfg_text_hiddens = uncond_hiddens + text_hiddens
        else:
            cfg_text_hiddens = text_hiddens
        text_bth, text_lens = self._pad_text(cfg_text_hiddens)

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
        try:
            decoded = self.vae.decode_packed_latents(latents)
            image = ImageUtil.to_image(
                decoded_latents=decoded,
                config=config,
                seed=seed,
                prompt=prompt,
                quantization=self.bits,
                lora_paths=self.lora_paths if lora_paths is None else lora_paths,
                lora_scales=self.lora_scales if lora_scales is None else lora_scales,
                image_path=config.image_path,
                image_strength=config.image_strength,
                generation_time=timer.elapsed_seconds(),
                negative_prompt=negative_prompt,
                extra_metadata=extra_metadata or LoRALoader.extra_metadata_for_model(self),
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
            weight_definition=ErnieImageWeightDefinition,
        )

    def _encode_prompt(self, prompts: list[str]) -> list[mx.array]:
        text_hiddens = []
        for prompt in prompts:
            output = self.tokenizers["ernie"].tokenize(prompt)
            hidden = self.text_encoder(output.input_ids, output.attention_mask)
            num_valid = int(mx.sum(output.attention_mask[0]).item())
            text_hiddens.append(RuntimeMemory.materialize_inference_tree(hidden[0, :num_valid, :]))
        return text_hiddens

    def _enhance_prompt(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        system_prompt: str | None,
        temperature: float,
        top_p: float,
        max_new_tokens: int | None,
    ) -> str:
        self._ensure_prompt_enhancer()
        tokenizer = self.tokenizers["ernie_prompt_enhancer"].tokenizer
        user_content = json.dumps({"prompt": prompt, "width": width, "height": height}, ensure_ascii=False)
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(
            input_text,
            add_special_tokens=False,
            truncation=True,
            padding=False,
            max_length=tokenizer.model_max_length,
        )
        input_ids = mx.array([tokenized["input_ids"]], dtype=mx.int32)
        output_ids = self.prompt_enhancer.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens or tokenizer.model_max_length,
            eos_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        generated_ids = output_ids[0, input_ids.shape[1] :].tolist()
        if tokenizer.eos_token_id in generated_ids:
            generated_ids = generated_ids[: generated_ids.index(tokenizer.eos_token_id)]
        enhanced = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return enhanced or prompt

    def _ensure_prompt_enhancer(self) -> None:
        if self.prompt_enhancer is None:
            ErnieImageInitializer.init_prompt_enhancer(self)

    def _prepare_generation_latents(
        self,
        *,
        seed: int,
        config: Config,
        scheduler: ErnieImageScheduler,
        batch_size: int,
    ) -> mx.array:
        noise = ErnieImageLatentCreator.create_noise(
            seed=seed,
            width=config.width,
            height=config.height,
            batch_size=batch_size,
        )
        if config.image_path is None:
            return noise
        if config.image_strength is None or config.image_strength <= 0.0:
            raise ValueError("latent image-to-image requires image_strength > 0.")

        encoded = LatentCreator.encode_image(
            vae=self.vae,
            image_path=config.image_path,
            height=config.height,
            width=config.width,
            tiling_config=self.tiling_config,
        )
        encoded = self._ensure_4d_latents(encoded)
        encoded = self._crop_to_even_spatial(encoded)
        encoded = self._match_latent_spatial_size(
            encoded=encoded,
            target_height=noise.shape[2] * 2,
            target_width=noise.shape[3] * 2,
        )
        encoded = ErnieImageLatentCreator.patchify_latents(encoded)
        encoded = self._bn_normalize_vae_encoded_latents(encoded)

        sigma = scheduler.sigmas[config.init_time_step]
        return LatentCreator.add_noise_by_interpolation(clean=encoded, noise=noise, sigma=sigma)

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
        return RuntimeMemory.materialize_inference_tree((mx.stack(padded, axis=0), mx.array(lengths, dtype=mx.int32)))

    @staticmethod
    def _ensure_4d_latents(latents: mx.array) -> mx.array:
        if latents.ndim == 5 and latents.shape[2] == 1:
            return latents[:, :, 0, :, :]
        return latents

    @staticmethod
    def _crop_to_even_spatial(latents: mx.array) -> mx.array:
        if latents.shape[2] % 2 != 0:
            latents = latents[:, :, :-1, :]
        if latents.shape[3] % 2 != 0:
            latents = latents[:, :, :, :-1]
        return latents

    @staticmethod
    def _match_latent_spatial_size(
        *,
        encoded: mx.array,
        target_height: int,
        target_width: int,
    ) -> mx.array:
        _, _, height, width = encoded.shape
        if height != target_height:
            if height > target_height:
                offset = (height - target_height) // 2
                encoded = encoded[:, :, offset : offset + target_height, :]
            else:
                pad_total = target_height - height
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                encoded = mx.pad(encoded, ((0, 0), (0, 0), (pad_before, pad_after), (0, 0)))
        if width != target_width:
            if width > target_width:
                offset = (width - target_width) // 2
                encoded = encoded[:, :, :, offset : offset + target_width]
            else:
                pad_total = target_width - width
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                encoded = mx.pad(encoded, ((0, 0), (0, 0), (0, 0), (pad_before, pad_after)))
        return encoded

    def _bn_normalize_vae_encoded_latents(self, encoded: mx.array) -> mx.array:
        bn_mean = self.vae.bn.running_mean.reshape(1, -1, 1, 1).astype(encoded.dtype)
        bn_std = mx.sqrt(self.vae.bn.running_var.reshape(1, -1, 1, 1) + self.vae.bn.eps).astype(encoded.dtype)
        return (encoded - bn_mean) / bn_std
