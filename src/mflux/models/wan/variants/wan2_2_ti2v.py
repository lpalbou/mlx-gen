import gc
import html
import io
import re
import shutil
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.wan.latent_creator import WanTimestepPolicy
from mflux.models.wan.model.wan_transformer import WanTransformer
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.scheduler import WanUniPCMultistepScheduler
from mflux.models.wan.wan_initializer import WanInitializer
from mflux.models.wan.weights import WanWeightDefinition
from mflux.utils.generated_video import GeneratedVideo
from mflux.utils.image_util import ImageUtil
from mflux.utils.video_util import VideoUtil


class Wan2_2_TI2V(nn.Module):
    RECOMMENDED_WIDTH = 1280
    RECOMMENDED_HEIGHT = 704
    RECOMMENDED_AREA = RECOMMENDED_WIDTH * RECOMMENDED_HEIGHT
    RECOMMENDED_FRAMES = 121
    RECOMMENDED_STEPS = 50
    RECOMMENDED_FPS = 24

    transformer: WanTransformer
    vae: Wan2_2_VAE

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig = ModelConfig.wan2_2_ti2v_5b(),
    ):
        super().__init__()
        WanInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
        )

    def generate_video(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 50,
        height: int = RECOMMENDED_HEIGHT,
        width: int = RECOMMENDED_WIDTH,
        num_frames: int = RECOMMENDED_FRAMES,
        fps: int = RECOMMENDED_FPS,
        guidance: float | None = 5.0,
        negative_prompt: str | None = "",
        image_path: Path | str | None = None,
        max_sequence_length: int = 512,
    ) -> GeneratedVideo:
        start_time = time.time()
        height, width = self._validated_spatial_size(height=height, width=width)
        num_frames = self._validated_frame_count(num_frames)
        guidance = 5.0 if guidance is None else float(guidance)
        self._warn_if_smoke_settings(
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            fps=fps,
        )
        batch_size = 1
        is_image_to_video = image_path is not None

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=guidance > 1.0,
            max_sequence_length=max_sequence_length,
        )

        scheduler = WanUniPCMultistepScheduler()
        scheduler.set_timesteps(num_inference_steps)
        latents = self.prepare_latents(
            seed=seed,
            batch_size=batch_size,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        first_frame_mask = None
        condition = None
        if is_image_to_video:
            first_frame_mask = WanTimestepPolicy.first_frame_mask(latent_shape=latents.shape)
            condition = self._encode_first_frame_condition(
                image_path=image_path,
                height=height,
                width=width,
            )

        for timestep in scheduler.timesteps.tolist():
            if first_frame_mask is not None and condition is not None:
                latent_model_input = WanTimestepPolicy.apply_first_frame_condition(
                    latents=latents,
                    condition=condition,
                    first_frame_mask=first_frame_mask,
                ).astype(ModelConfig.precision)
                expanded_timestep = WanTimestepPolicy.expand_from_mask(
                    mask=first_frame_mask,
                    batch_size=batch_size,
                    timestep=timestep,
                    patch_size=self.transformer.patch_size,
                )
            else:
                latent_model_input = latents.astype(ModelConfig.precision)
                expanded_timestep = WanTimestepPolicy.expand_for_text_to_video(
                    latent_shape=latents.shape,
                    timestep=timestep,
                    patch_size=self.transformer.patch_size,
                )
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=expanded_timestep,
                encoder_hidden_states=prompt_embeds,
            )
            if negative_prompt_embeds is not None:
                noise_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=expanded_timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                )
                noise_pred = noise_uncond + guidance * (noise_pred - noise_uncond)

            latents = scheduler.step(noise_pred.astype(mx.float32), timestep, latents, return_dict=False)[0]
            mx.eval(latents)

        if first_frame_mask is not None and condition is not None:
            latents = WanTimestepPolicy.apply_first_frame_condition(
                latents=latents,
                condition=condition,
                first_frame_mask=first_frame_mask,
            )
        decoded = self.vae.decode_normalized_latents(latents.astype(ModelConfig.precision))
        mx.eval(decoded)
        return VideoUtil.to_video(
            decoded_latents=decoded,
            fps=fps,
            model_config=self.model_config,
            seed=seed,
            prompt=prompt,
            steps=num_inference_steps,
            guidance=guidance,
            quantization=self.bits,
            generation_time=time.time() - start_time,
            task="image-to-video" if is_image_to_video else "text-to-video",
            image_path=image_path,
            negative_prompt=negative_prompt,
        )

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str | None,
        do_classifier_free_guidance: bool,
        max_sequence_length: int = 512,
    ) -> tuple[mx.array, mx.array | None]:
        prompts = [prompt]
        if not do_classifier_free_guidance:
            return self._get_t5_prompt_embeds(prompts, max_sequence_length=max_sequence_length), None
        prompts.append(negative_prompt or "")
        embeds = self._get_t5_prompt_embeds(prompts, max_sequence_length=max_sequence_length)
        return embeds[0:1], embeds[1:2]

    def prepare_latents(
        self,
        seed: int,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
    ) -> mx.array:
        mx.random.seed(seed)
        latent_frames = (num_frames - 1) // Wan2_2_VAE.TEMPORAL_SCALE + 1
        shape = (
            batch_size,
            self.transformer.in_channels,
            latent_frames,
            height // Wan2_2_VAE.SPATIAL_SCALE,
            width // Wan2_2_VAE.SPATIAL_SCALE,
        )
        return mx.random.normal(shape, dtype=mx.float32)

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=WanWeightDefinition,
        )
        self._copy_runtime_assets(base_path)

    def _get_t5_prompt_embeds(self, prompts: list[str], max_sequence_length: int) -> mx.array:
        try:
            import torch
            from transformers import UMT5EncoderModel
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:
            raise RuntimeError("Wan prompt encoding requires torch and transformers.") from exc

        text_encoder_path = self.root_path / "text_encoder"
        if not text_encoder_path.exists():
            raise FileNotFoundError(
                f"Wan text encoder files were not found in {text_encoder_path}. "
                "Run `mlxgen download --model Wan-AI/Wan2.2-TI2V-5B-Diffusers` first."
            )

        cleaned = [self._prompt_clean(prompt) for prompt in prompts]
        tokenizer = self.tokenizers["wan"].tokenizer
        text_inputs = tokenizer(
            cleaned,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        transformers_verbosity = transformers_logging.get_verbosity()
        try:
            transformers_logging.set_verbosity_error()
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                text_encoder = UMT5EncoderModel.from_pretrained(
                    text_encoder_path,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                )
        finally:
            transformers_logging.set_verbosity(transformers_verbosity)
        text_encoder.eval()
        if hasattr(text_encoder, "shared") and hasattr(text_encoder, "encoder"):
            text_encoder.encoder.embed_tokens = text_encoder.shared

        with torch.no_grad():
            output = text_encoder(
                text_inputs.input_ids,
                text_inputs.attention_mask,
            ).last_hidden_state
        seq_lens = text_inputs.attention_mask.gt(0).sum(dim=1).long()
        padded = torch.stack(
            [
                torch.cat(
                    [
                        hidden[:seq_len_int],
                        hidden.new_zeros(max_sequence_length - seq_len_int, hidden.size(1)),
                    ]
                )
                for hidden, seq_len in zip(output, seq_lens)
                for seq_len_int in [int(seq_len.item())]
            ],
            dim=0,
        )
        embeds = mx.array(padded.float().cpu().numpy()).astype(ModelConfig.precision)
        del text_encoder
        gc.collect()
        return embeds

    def _encode_first_frame_condition(self, image_path: Path | str | None, height: int, width: int) -> mx.array:
        if image_path is None:
            raise ValueError("Wan image-to-video requires image_path.")
        image = ImageUtil.scale_to_dimensions(ImageUtil.load_image(image_path), target_width=width, target_height=height)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = image_np[None, ...]
        image_mx = mx.array(image_np)
        image_mx = mx.transpose(image_mx, (0, 3, 1, 2))
        image_mx = ImageUtil._normalize(image_mx)
        condition = self.vae.encode_normalized(image_mx.astype(ModelConfig.precision))
        mx.eval(condition)
        return condition.astype(mx.float32)

    def _copy_runtime_assets(self, base_path: str) -> None:
        target = Path(base_path)
        for subdir in ("text_encoder", "scheduler"):
            source = self.root_path / subdir
            if source.exists():
                shutil.copytree(source, target / subdir, dirs_exist_ok=True)
        model_index = self.root_path / "model_index.json"
        if model_index.exists():
            shutil.copy2(model_index, target / "model_index.json")

    @staticmethod
    def _validated_spatial_size(height: int, width: int) -> tuple[int, int]:
        multiple = Wan2_2_VAE.SPATIAL_SCALE * 2
        calc_height = height // multiple * multiple
        calc_width = width // multiple * multiple
        if calc_height <= 0 or calc_width <= 0:
            raise ValueError(f"Wan height and width must be at least {multiple}px.")
        if (height, width) != (calc_height, calc_width):
            print(
                "`height` and `width` must be multiples of "
                f"({multiple}, {multiple}) for Wan patchification. "
                f"Adjusting ({height}, {width}) -> ({calc_height}, {calc_width})."
            )
        return calc_height, calc_width

    @staticmethod
    def _validated_frame_count(num_frames: int) -> int:
        if num_frames < 1:
            raise ValueError("Wan num_frames must be at least 1.")
        if num_frames % Wan2_2_VAE.TEMPORAL_SCALE != 1:
            adjusted = num_frames // Wan2_2_VAE.TEMPORAL_SCALE * Wan2_2_VAE.TEMPORAL_SCALE + 1
            print(
                f"`frames - 1` must be divisible by {Wan2_2_VAE.TEMPORAL_SCALE}. "
                f"Adjusting {num_frames} -> {adjusted}."
            )
            num_frames = adjusted
        return max(num_frames, 1)

    @staticmethod
    def _warn_if_smoke_settings(
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        fps: int,
    ) -> None:
        issues = []
        if height * width < Wan2_2_TI2V.RECOMMENDED_AREA:
            issues.append(
                f"resolution below {Wan2_2_TI2V.RECOMMENDED_WIDTH}x{Wan2_2_TI2V.RECOMMENDED_HEIGHT}/"
                f"{Wan2_2_TI2V.RECOMMENDED_HEIGHT}x{Wan2_2_TI2V.RECOMMENDED_WIDTH} area"
            )
        if num_frames < Wan2_2_TI2V.RECOMMENDED_FRAMES:
            issues.append("frame count below 121")
        if num_inference_steps < Wan2_2_TI2V.RECOMMENDED_STEPS:
            issues.append("steps below 50")
        if fps != Wan2_2_TI2V.RECOMMENDED_FPS:
            issues.append("fps differs from 24")
        if issues:
            print(
                "Wan2.2 TI2V warning: these settings are suitable only for wiring/smoke tests, not quality "
                "validation. Upstream recommends 1280x704 or 704x1280, 121 frames, 50 steps, and 24 fps. "
                f"Detected: {', '.join(issues)}.",
                file=sys.stderr,
            )

    @staticmethod
    def _prompt_clean(text: str) -> str:
        try:
            import ftfy
        except ImportError:
            ftfy = None
        if ftfy is not None:
            text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return re.sub(r"\s+", " ", text).strip()
