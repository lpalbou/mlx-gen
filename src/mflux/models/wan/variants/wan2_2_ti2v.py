import gc
import html
import io
import re
import shutil
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import mlx.core as mx
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
from mflux.utils.video_util import VideoUtil


class Wan2_2_TI2V(nn.Module):
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
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        fps: int = 16,
        guidance: float | None = 5.0,
        negative_prompt: str | None = "",
        image_path: Path | str | None = None,
        max_sequence_length: int = 512,
    ) -> GeneratedVideo:
        if image_path is not None:
            raise NotImplementedError(
                "Wan2.2 image-to-video is not enabled yet. The text-to-video path is implemented first; "
                "I2V needs the Diffusers first-frame latent conditioning path."
            )

        start_time = time.time()
        height, width = self._validated_spatial_size(height=height, width=width)
        num_frames = self._validated_frame_count(num_frames)
        guidance = 5.0 if guidance is None else float(guidance)
        batch_size = 1

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

        for timestep in scheduler.timesteps.tolist():
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
            task="text-to-video",
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
                "Run `mlxgen download --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --all-files` first."
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
    def _prompt_clean(text: str) -> str:
        try:
            import ftfy
        except ImportError:
            ftfy = None
        if ftfy is not None:
            text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return re.sub(r"\s+", " ", text).strip()
