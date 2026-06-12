import gc
import html
import io
import math
import re
import shutil
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.callbacks import ProgressCallback, ProgressEvent
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.wan.latent_creator import WanTimestepPolicy
from mflux.models.wan.model.wan_transformer import WanBlockHealthContext, WanTransformer
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.scheduler import WanEulerScheduler, WanUniPCMultistepScheduler
from mflux.models.wan.wan_initializer import WanInitializer
from mflux.models.wan.weights import WanWeightDefinition
from mflux.utils.exceptions import ModelConfigError
from mflux.utils.generated_video import GeneratedVideo
from mflux.utils.image_util import ImageUtil
from mflux.utils.tensor_health import TensorHealth
from mflux.utils.video_util import VideoUtil

_GUIDANCE_2_UNSET = object()
_WAN_DEFAULT_SOLVER = "unipc"
_WAN_SOLVERS = ("unipc", "euler")


class Wan2_2_TI2V(nn.Module):
    RECOMMENDED_WIDTH = 1280
    RECOMMENDED_HEIGHT = 704
    RECOMMENDED_AREA = RECOMMENDED_WIDTH * RECOMMENDED_HEIGHT
    RECOMMENDED_FRAMES = 121
    RECOMMENDED_STEPS = 50
    RECOMMENDED_FPS = 24

    transformer: WanTransformer | None
    transformer_2: WanTransformer | None
    vae: Wan2_2_VAE

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_target_roles: list[str] | None = None,
    ):
        super().__init__()
        model_config = self._resolve_model_config(model_path=model_path, model_config=model_config)
        WanInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            lora_target_roles=lora_target_roles,
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
        guidance: float | None = None,
        guidance_2: float | None | object = _GUIDANCE_2_UNSET,
        flow_shift: float | None = None,
        solver: str | None = None,
        negative_prompt: str | None = None,
        image_path: Path | str | None = None,
        max_sequence_length: int = 512,
        progress_callback: ProgressCallback | None = None,
        release_inactive_denoiser: bool = False,
        release_denoisers_before_decode: bool = False,
        clear_cache_each_step: bool = False,
        clear_cache_each_transformer_block: bool = False,
        tensor_health_check_interval: int | None = None,
    ) -> GeneratedVideo:
        start_time = time.time()
        health_check_interval = self._validate_tensor_health_check_interval(tensor_health_check_interval)
        if (
            guidance_2 is not _GUIDANCE_2_UNSET
            and guidance_2 is not None
            and self._wan_config("boundary_ratio", None) is None
        ):
            raise ValueError("guidance_2 is only supported for Wan models with two-transformer boundary routing.")
        is_image_to_video = image_path is not None
        task = "image-to-video" if is_image_to_video else "text-to-video"
        if image_path is not None and not self._supports_image_to_video():
            raise ValueError(f"{self.model_config.model_name} does not support image-to-video input.")
        self._validate_denoisers_available()
        height, width, spatial_metadata = self._resolve_video_spatial_size(
            height=height,
            width=width,
            image_path=image_path,
        )
        num_frames = self._validated_frame_count(num_frames)
        guidance, guidance_2 = self._resolve_guidance_pair(guidance=guidance, guidance_2=guidance_2)
        self._validate_guidance_values(guidance=guidance, guidance_2=guidance_2)
        flow_shift = self._resolve_flow_shift(flow_shift)
        solver = self._resolve_solver(solver)
        negative_prompt = self._resolve_negative_prompt(negative_prompt)
        self._validate_runtime_contract(is_image_to_video=is_image_to_video)
        progress_registry = getattr(self, "callbacks", None)
        self._emit_progress(
            progress_callback,
            phase="start",
            frame=0,
            total_frames=num_frames,
            step=0,
            total_steps=num_inference_steps,
            task=task,
            registry=progress_registry,
        )
        batch_size = 1

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=guidance > 1.0,
            max_sequence_length=max_sequence_length,
        )
        self._require_tensor_health(prompt_embeds, phase="prompt-encoding", name="prompt_embeds")
        if negative_prompt_embeds is not None:
            self._require_tensor_health(
                negative_prompt_embeds,
                phase="prompt-encoding",
                name="negative_prompt_embeds",
            )

        scheduler = self._create_scheduler(flow_shift=flow_shift, solver=solver)
        scheduler.set_timesteps(num_inference_steps)
        boundary_timestep = self._boundary_timestep(scheduler)
        latents = self.prepare_latents(
            seed=seed,
            batch_size=batch_size,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        self._require_tensor_health(latents, phase="prepare-latents", name="latents")
        first_frame_mask = None
        condition = None
        if is_image_to_video:
            if self._uses_expanded_timesteps():
                first_frame_mask = WanTimestepPolicy.first_frame_mask(latent_shape=latents.shape)
                condition = self._encode_first_frame_condition(
                    image_path=image_path,
                    height=height,
                    width=width,
                )
                self._require_tensor_health(condition, phase="image-conditioning", name="condition")
            else:
                condition = self._encode_video_condition(
                    image_path=image_path,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    batch_size=batch_size,
                )
                self._require_tensor_health(condition, phase="image-conditioning", name="condition")

        total_steps = len(scheduler.timesteps)
        high_noise_denoiser_released = False
        for step_index, timestep in enumerate(scheduler.timesteps.tolist()):
            step_number = step_index + 1
            should_check_tensors = TensorHealth.should_check_step(step_number, total_steps, health_check_interval)
            high_noise_denoiser_released = self._maybe_release_high_noise_denoiser(
                timestep=timestep,
                boundary_timestep=boundary_timestep,
                release_inactive_denoiser=release_inactive_denoiser,
                already_released=high_noise_denoiser_released,
            )
            current_transformer, current_guidance = self._select_transformer_and_guidance(
                timestep=timestep,
                boundary_timestep=boundary_timestep,
                guidance=guidance,
                guidance_2=guidance_2,
            )
            denoiser_name = self._denoiser_name(current_transformer)
            block_health_context = WanBlockHealthContext(
                step=step_number,
                total_steps=total_steps,
                timestep=timestep,
                denoiser=denoiser_name,
                guidance=current_guidance,
            )
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
                    patch_size=current_transformer.patch_size,
                )
            elif is_image_to_video and condition is not None:
                latent_model_input = mx.concatenate([latents, condition], axis=1).astype(ModelConfig.precision)
                expanded_timestep = self._batch_timestep(batch_size=batch_size, timestep=timestep)
            else:
                latent_model_input = latents.astype(ModelConfig.precision)
                if self._uses_expanded_timesteps():
                    expanded_timestep = WanTimestepPolicy.expand_for_text_to_video(
                        latent_shape=latents.shape,
                        timestep=timestep,
                        patch_size=current_transformer.patch_size,
                    )
                else:
                    expanded_timestep = self._batch_timestep(batch_size=batch_size, timestep=timestep)
            noise_pred = current_transformer(
                hidden_states=latent_model_input,
                timestep=expanded_timestep,
                encoder_hidden_states=prompt_embeds,
                clear_cache_each_block=clear_cache_each_transformer_block,
                block_health_context=block_health_context,
            )
            if should_check_tensors or clear_cache_each_transformer_block:
                self._materialize_denoise_prediction(
                    noise_pred,
                    clear_cache=clear_cache_each_transformer_block,
                )
            if should_check_tensors:
                self._require_tensor_health(
                    noise_pred,
                    phase="conditional-denoise-prediction",
                    name="noise_pred",
                    step=step_number,
                    total_steps=total_steps,
                    timestep=timestep,
                    denoiser=denoiser_name,
                    guidance=current_guidance,
                )
            if negative_prompt_embeds is not None:
                noise_uncond = current_transformer(
                    hidden_states=latent_model_input,
                    timestep=expanded_timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    clear_cache_each_block=clear_cache_each_transformer_block,
                    block_health_context=block_health_context,
                )
                if should_check_tensors or clear_cache_each_transformer_block:
                    self._materialize_denoise_prediction(
                        noise_uncond,
                        clear_cache=clear_cache_each_transformer_block,
                    )
                if should_check_tensors:
                    self._require_tensor_health(
                        noise_uncond,
                        phase="unconditional-denoise-prediction",
                        name="noise_uncond",
                        step=step_number,
                        total_steps=total_steps,
                        timestep=timestep,
                        denoiser=denoiser_name,
                        guidance=current_guidance,
                    )
                noise_pred = noise_uncond + current_guidance * (noise_pred - noise_uncond)
                if should_check_tensors or clear_cache_each_transformer_block:
                    self._materialize_denoise_prediction(
                        noise_pred,
                        clear_cache=clear_cache_each_transformer_block,
                    )
                if should_check_tensors:
                    self._require_tensor_health(
                        noise_pred,
                        phase="guided-denoise-prediction",
                        name="noise_pred",
                        step=step_number,
                        total_steps=total_steps,
                        timestep=timestep,
                        denoiser=denoiser_name,
                        guidance=current_guidance,
                    )

            latents = scheduler.step(noise_pred.astype(mx.float32), timestep, latents, return_dict=False)[0]
            mx.eval(latents)
            if should_check_tensors:
                self._require_tensor_health(
                    latents,
                    phase="scheduler-step",
                    name="latents",
                    step=step_number,
                    total_steps=total_steps,
                    timestep=timestep,
                    denoiser=denoiser_name,
                    guidance=current_guidance,
                )
            self._emit_progress(
                progress_callback,
                phase="denoise",
                frame=self._progress_frame_for_step(
                    step_index=step_index,
                    total_steps=total_steps,
                    total_frames=num_frames,
                ),
                total_frames=num_frames,
                step=step_number,
                total_steps=total_steps,
                task=task,
                timestep=timestep,
                registry=progress_registry,
            )
            del current_transformer, latent_model_input, expanded_timestep, noise_pred
            if "noise_uncond" in locals():
                del noise_uncond
            self._cleanup_step_cache(clear_cache=clear_cache_each_step)

        if first_frame_mask is not None and condition is not None:
            latents = WanTimestepPolicy.apply_first_frame_condition(
                latents=latents,
                condition=condition,
                first_frame_mask=first_frame_mask,
            )
            self._require_tensor_health(latents, phase="final-conditioning", name="latents")
        del prompt_embeds, negative_prompt_embeds, scheduler
        if "noise_uncond" in locals():
            del noise_uncond
        if "condition" in locals():
            del condition
        if "first_frame_mask" in locals():
            del first_frame_mask
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        if release_denoisers_before_decode:
            self._release_denoisers()
        self._require_tensor_health(latents, phase="pre-decode", name="latents")
        self._emit_progress(
            progress_callback,
            phase="decode",
            frame=num_frames,
            total_frames=num_frames,
            step=total_steps,
            total_steps=total_steps,
            task=task,
            registry=progress_registry,
        )
        decoded = self.vae.decode_normalized_latents(
            latents.astype(ModelConfig.precision),
            clear_cache_each_slice=release_denoisers_before_decode,
        )
        mx.eval(decoded)
        mx.synchronize()
        mx.clear_cache()
        self._require_tensor_health(decoded, phase="vae-decode", name="decoded")
        self._emit_progress(
            progress_callback,
            phase="convert",
            frame=num_frames,
            total_frames=num_frames,
            step=total_steps,
            total_steps=total_steps,
            task=task,
            registry=progress_registry,
        )
        video = VideoUtil.to_video(
            decoded_latents=decoded,
            fps=fps,
            model_config=self.model_config,
            seed=seed,
            prompt=prompt,
            steps=num_inference_steps,
            guidance=guidance,
            guidance_2=guidance_2,
            flow_shift=flow_shift,
            solver=solver,
            quantization=self.bits,
            generation_time=time.time() - start_time,
            task=task,
            image_path=image_path,
            negative_prompt=negative_prompt,
            source_width=spatial_metadata.get("source_width"),
            source_height=spatial_metadata.get("source_height"),
            requested_width=spatial_metadata.get("requested_width"),
            requested_height=spatial_metadata.get("requested_height"),
            lora_paths=getattr(self, "lora_paths", None),
            lora_scales=getattr(self, "lora_scales", None),
            extra_metadata={
                **(LoRALoader.extra_metadata_for_model(self) or {}),
                "lora_target_roles": getattr(self, "lora_target_roles", None) or None,
            },
        )
        self._emit_progress(
            progress_callback,
            phase="generated",
            frame=num_frames,
            total_frames=num_frames,
            step=total_steps,
            total_steps=total_steps,
            task=task,
            registry=progress_registry,
        )
        return video

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
        latent_frames = (num_frames - 1) // self.vae.temporal_scale + 1
        shape = (
            batch_size,
            self.vae.z_dim,
            latent_frames,
            height // self.vae.spatial_scale,
            width // self.vae.spatial_scale,
        )
        return mx.random.normal(shape, dtype=mx.float32)

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=getattr(self, "weight_definition", WanWeightDefinition.for_config(self.model_config)),
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
                f"Run `mlxgen download --model {self.model_config.model_name}` first."
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
        image = ImageUtil.scale_to_dimensions(
            ImageUtil.load_image(image_path), target_width=width, target_height=height
        )
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = image_np[None, ...]
        image_mx = mx.array(image_np)
        image_mx = mx.transpose(image_mx, (0, 3, 1, 2))
        image_mx = ImageUtil._normalize(image_mx)
        condition = self.vae.encode_normalized(image_mx.astype(ModelConfig.precision))
        mx.eval(condition)
        return condition.astype(mx.float32)

    def _encode_video_condition(
        self,
        image_path: Path | str | None,
        height: int,
        width: int,
        num_frames: int,
        batch_size: int,
    ) -> mx.array:
        if image_path is None:
            raise ValueError("Wan image-to-video requires image_path.")
        image = ImageUtil.scale_to_dimensions(
            ImageUtil.load_image(image_path), target_width=width, target_height=height
        )
        image_np = np.array(image).astype(np.float32) / 255.0
        image_mx = mx.array(image_np[None, ...])
        image_mx = mx.transpose(image_mx, (0, 3, 1, 2))
        image_mx = ImageUtil._normalize(image_mx)
        first_frame = image_mx[:, :, None, :, :]
        zero_frames = mx.zeros(
            (batch_size, first_frame.shape[1], num_frames - 1, height, width), dtype=first_frame.dtype
        )
        video_condition = mx.concatenate([first_frame, zero_frames], axis=2).astype(ModelConfig.precision)
        latent_condition = self.vae.encode_normalized(video_condition).astype(mx.float32)
        latent_frames = latent_condition.shape[2]
        latent_height = latent_condition.shape[3]
        latent_width = latent_condition.shape[4]
        mask_np = np.ones((batch_size, 1, num_frames, latent_height, latent_width), dtype=np.float32)
        mask_np[:, :, 1:] = 0
        mask = mx.array(mask_np)
        first_frame_mask = mx.repeat(mask[:, :, 0:1], self.vae.temporal_scale, axis=2)
        mask = mx.concatenate([first_frame_mask, mask[:, :, 1:]], axis=2)
        mask = mx.reshape(mask, (batch_size, -1, self.vae.temporal_scale, latent_height, latent_width))
        mask = mx.transpose(mask, (0, 2, 1, 3, 4))
        condition = mx.concatenate([mask[:, :, :latent_frames], latent_condition], axis=1)
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

    def _validated_spatial_size(self, height: int, width: int) -> tuple[int, int]:
        multiple_h = self.vae.spatial_scale * self.transformer.patch_size[1]
        multiple_w = self.vae.spatial_scale * self.transformer.patch_size[2]
        if height <= 0 or width <= 0:
            raise ValueError(f"Wan height and width must be at least ({multiple_h}, {multiple_w})px.")
        calc_height = math.ceil(height / multiple_h) * multiple_h
        calc_width = math.ceil(width / multiple_w) * multiple_w
        if (height, width) != (calc_height, calc_width):
            print(
                "`height` and `width` must be multiples of "
                f"({multiple_h}, {multiple_w}) for Wan patchification. "
                f"Adjusting ({height}, {width}) -> ({calc_height}, {calc_width})."
            )
        return calc_height, calc_width

    def _resolved_spatial_size(
        self,
        *,
        height: int,
        width: int,
        image_path: Path | str | None,
    ) -> tuple[int, int]:
        resolved_height, resolved_width, _ = self._resolve_video_spatial_size(
            height=height,
            width=width,
            image_path=image_path,
        )
        return resolved_height, resolved_width

    def _resolve_video_spatial_size(
        self,
        *,
        height: int,
        width: int,
        image_path: Path | str | None,
    ) -> tuple[int, int, dict[str, int]]:
        if image_path is None:
            resolved_height, resolved_width = self._validated_spatial_size(height=height, width=width)
            return resolved_height, resolved_width, {}
        source_image = ImageUtil.load_image(image_path)
        resolved_height, resolved_width = self._validated_i2v_spatial_size(
            height=height,
            width=width,
            source_height=source_image.height,
            source_width=source_image.width,
        )
        return (
            resolved_height,
            resolved_width,
            {
                "source_width": source_image.width,
                "source_height": source_image.height,
                "requested_width": width,
                "requested_height": height,
            },
        )

    def _validated_i2v_spatial_size(
        self,
        *,
        height: int,
        width: int,
        source_height: int,
        source_width: int,
    ) -> tuple[int, int]:
        multiple_h = self.vae.spatial_scale * self.transformer.patch_size[1]
        multiple_w = self.vae.spatial_scale * self.transformer.patch_size[2]
        if height <= 0 or width <= 0:
            raise ValueError(f"Wan height and width must be at least ({multiple_h}, {multiple_w})px.")
        if source_height <= 0 or source_width <= 0:
            raise ValueError("Wan image-to-video source image must have positive dimensions.")
        calc_height, calc_width = self._closest_spatial_size_for_ratio(
            requested_height=height,
            requested_width=width,
            source_height=source_height,
            source_width=source_width,
            multiple_h=multiple_h,
            multiple_w=multiple_w,
        )
        if (height, width) != (calc_height, calc_width):
            print(
                "Wan image-to-video preserves the input image aspect ratio. "
                f"Adjusting requested video size ({height}, {width}) with source image "
                f"({source_height}, {source_width}) -> ({calc_height}, {calc_width})."
            )
        return calc_height, calc_width

    @staticmethod
    def _closest_spatial_size_for_ratio(
        *,
        requested_height: int,
        requested_width: int,
        source_height: int,
        source_width: int,
        multiple_h: int,
        multiple_w: int,
    ) -> tuple[int, int]:
        target_ratio = source_width / source_height
        target_area = requested_width * requested_height
        ideal_height = math.sqrt(target_area / target_ratio)
        ideal_width = ideal_height * target_ratio
        max_height = max(multiple_h, math.ceil((ideal_height * 2) / multiple_h) * multiple_h)
        max_width = max(multiple_w, math.ceil((ideal_width * 2) / multiple_w) * multiple_w)
        candidates: set[tuple[int, int]] = set()

        for candidate_height in range(multiple_h, max_height + multiple_h, multiple_h):
            ideal_candidate_width = candidate_height * target_ratio
            for width_count in Wan2_2_TI2V._nearby_axis_counts(ideal_candidate_width, multiple_w):
                candidates.add((candidate_height, width_count * multiple_w))

        for candidate_width in range(multiple_w, max_width + multiple_w, multiple_w):
            ideal_candidate_height = candidate_width / target_ratio
            for height_count in Wan2_2_TI2V._nearby_axis_counts(ideal_candidate_height, multiple_h):
                candidates.add((height_count * multiple_h, candidate_width))

        def score(candidate: tuple[int, int]) -> tuple[float, float, float, int, int]:
            candidate_height, candidate_width = candidate
            candidate_ratio = candidate_width / candidate_height
            candidate_area = candidate_width * candidate_height
            ratio_error = abs(math.log(candidate_ratio / target_ratio))
            area_error = abs(math.log(candidate_area / target_area))
            shape_error = abs(candidate_width - ideal_width) / ideal_width
            shape_error += abs(candidate_height - ideal_height) / ideal_height
            return (ratio_error * 10.0 + area_error, ratio_error, area_error, candidate_area, int(shape_error * 1e9))

        return min(candidates, key=score)

    @staticmethod
    def _nearby_axis_counts(ideal_size: float, multiple: int) -> set[int]:
        ideal_count = ideal_size / multiple
        base_counts = {math.floor(ideal_count), round(ideal_count), math.ceil(ideal_count)}
        return {max(1, count + offset) for count in base_counts for offset in range(-2, 3)}

    def _validated_frame_count(self, num_frames: int) -> int:
        if num_frames < 1:
            raise ValueError("Wan num_frames must be at least 1.")
        if num_frames % self.vae.temporal_scale != 1:
            adjusted = num_frames // self.vae.temporal_scale * self.vae.temporal_scale + 1
            print(f"`frames - 1` must be divisible by {self.vae.temporal_scale}. Adjusting {num_frames} -> {adjusted}.")
            num_frames = adjusted
        return max(num_frames, 1)

    def _validate_runtime_contract(self, *, is_image_to_video: bool) -> None:
        task = self._wan_config("task", "text-image-to-video")
        if task == "image-to-video" and not is_image_to_video:
            raise ValueError(f"{self.model_config.model_name} requires image-to-video input.")

        expected_config = self.model_config.transformer_overrides
        expected_vae_config = expected_config.get("vae_config", {})
        expected_transformer_channels = int(expected_config.get("in_channels", self.transformer.in_channels))
        expected_output_channels = int(expected_config.get("out_channels", self.transformer.out_channels))
        expected_vae_channels = int(expected_vae_config.get("z_dim", self.vae.z_dim))
        expected_transformer_2 = bool(expected_config.get("has_transformer_2", False))

        if int(self.transformer.in_channels) != expected_transformer_channels:
            self._raise_runtime_contract_mismatch(
                "transformer.in_channels",
                actual=int(self.transformer.in_channels),
                expected=expected_transformer_channels,
            )
        if int(self.transformer.out_channels) != expected_output_channels:
            self._raise_runtime_contract_mismatch(
                "transformer.out_channels",
                actual=int(self.transformer.out_channels),
                expected=expected_output_channels,
            )
        if int(self.vae.z_dim) != expected_vae_channels:
            self._raise_runtime_contract_mismatch(
                "vae.z_dim",
                actual=int(self.vae.z_dim),
                expected=expected_vae_channels,
            )
        if (self.transformer_2 is not None) != expected_transformer_2:
            self._raise_runtime_contract_mismatch(
                "transformer_2",
                actual="present" if self.transformer_2 is not None else "absent",
                expected="present" if expected_transformer_2 else "absent",
            )

        transformer_channels = int(self.transformer.in_channels)
        expected_channels = int(self.vae.z_dim)
        if is_image_to_video and not self._uses_expanded_timesteps():
            expected_channels += 20
        if transformer_channels != expected_channels:
            raise ValueError(
                "Wan runtime config mismatch: "
                f"{self.model_config.model_name} transformer expects {transformer_channels} input channels, "
                f"but the selected VAE/input path provides {expected_channels}. "
                "This usually means the model weights were paired with the wrong Wan config; refusing to continue."
            )

    def _validate_denoisers_available(self) -> None:
        expected_transformer_2 = bool(self._wan_config("has_transformer_2", False))
        if self.transformer is None or (expected_transformer_2 and self.transformer_2 is None):
            raise ValueError(
                "Wan denoisers have been released after a previous low-memory generation. "
                "Create a new Wan2_2_TI2V instance to generate another video."
            )

    @staticmethod
    def _resolve_model_config(model_path: str | None, model_config: ModelConfig | None) -> ModelConfig:
        if model_config is not None:
            return model_config
        if model_path is None:
            return ModelConfig.wan2_2_ti2v_5b()
        try:
            return ModelConfig.from_name(model_path)
        except ModelConfigError as exc:
            raise ValueError(
                f"Cannot infer a supported Wan model config from {model_path}. "
                "Pass model_config explicitly; MLX-Gen will not fall back to another Wan architecture."
            ) from exc

    def _raise_runtime_contract_mismatch(self, key: str, actual, expected) -> None:
        raise ValueError(
            "Wan runtime config mismatch: "
            f"{self.model_config.model_name} has {key}={actual!r}, but the selected config expects {expected!r}. "
            "Pass the exact Wan model/config that matches these weights; MLX-Gen will not fall back silently."
        )

    @staticmethod
    def _progress_frame_for_step(step_index: int, total_steps: int, total_frames: int) -> int:
        if total_steps <= 0 or total_frames <= 1:
            return 0
        return min(total_frames - 1, int(((step_index + 1) * (total_frames - 1)) / total_steps))

    @staticmethod
    def _emit_progress(
        progress_callback: ProgressCallback | None,
        *,
        phase: str,
        frame: int,
        total_frames: int,
        step: int,
        total_steps: int,
        task: str | None = None,
        timestep: int | float | None = None,
        registry: CallbackRegistry | None = None,
    ) -> None:
        if progress_callback is None and registry is None:
            return
        event = ProgressEvent(
            phase=phase,
            frame=frame,
            total_frames=total_frames,
            step=step,
            total_steps=total_steps,
            task=task,
            timestep=timestep,
        )
        if progress_callback is not None:
            progress_callback(event)
        if registry is not None:
            registry.emit_progress(event)

    def _require_tensor_health(
        self,
        tensor: mx.array,
        *,
        phase: str,
        name: str,
        step: int | None = None,
        total_steps: int | None = None,
        timestep: int | float | None = None,
        denoiser: str | None = None,
        guidance: float | None = None,
    ) -> None:
        TensorHealth.ensure_finite(
            tensor,
            name=name,
            phase=f"wan-{phase}",
            step=step,
            total_steps=total_steps,
            timestep=timestep,
            denoiser=denoiser,
            guidance=guidance,
        )

    @staticmethod
    def _validate_tensor_health_check_interval(interval: int | None) -> int | None:
        if interval is None:
            return None
        if interval <= 0:
            raise ValueError("tensor_health_check_interval must be greater than zero.")
        return int(interval)

    def _denoiser_name(self, transformer) -> str:
        if transformer is self.transformer:
            return "high"
        if transformer is self.transformer_2:
            return "low"
        return transformer.__class__.__name__

    def _select_transformer_and_guidance(
        self,
        *,
        timestep: int,
        boundary_timestep: float | None,
        guidance: float,
        guidance_2: float | None,
    ) -> tuple[WanTransformer, float]:
        if boundary_timestep is None or timestep >= boundary_timestep:
            if self.transformer is None:
                raise ValueError("Wan high-noise transformer was released before a high-noise timestep.")
            return self.transformer, guidance
        if self.transformer_2 is None:
            raise ValueError("Wan model config requested low-noise routing but transformer_2 is missing.")
        if guidance_2 is None:
            raise ValueError("Wan low-noise routing requires guidance_2.")
        return self.transformer_2, guidance_2

    def _boundary_timestep(self, scheduler: WanUniPCMultistepScheduler) -> float | None:
        boundary_ratio = self._wan_config("boundary_ratio", None)
        if boundary_ratio is None:
            return None
        return float(boundary_ratio) * scheduler.num_train_timesteps

    def _uses_expanded_timesteps(self) -> bool:
        return bool(self._wan_config("expand_timesteps", True))

    def _supports_image_to_video(self) -> bool:
        return bool(self._wan_config("supports_image_to_video", True))

    def _default_guidance(self) -> float:
        return float(self._wan_config("default_guidance", 5.0))

    def _default_guidance_2(self) -> float | None:
        value = self._wan_config("default_guidance_2", None)
        return None if value is None else float(value)

    def _default_flow_shift(self) -> float:
        return float(self._wan_config("flow_shift", 5.0))

    def _default_solver(self) -> str:
        return str(self._wan_config("default_solver", _WAN_DEFAULT_SOLVER))

    def _resolve_flow_shift(self, flow_shift: float | None) -> float:
        resolved = self._default_flow_shift() if flow_shift is None else float(flow_shift)
        if not np.isfinite(resolved) or resolved <= 0:
            raise ValueError(f"Wan flow_shift must be a finite positive value, got {flow_shift!r}.")
        return resolved

    def _resolve_solver(self, solver: str | None) -> str:
        resolved = self._default_solver() if solver is None else str(solver).strip().lower()
        if resolved not in _WAN_SOLVERS:
            raise ValueError(f"Wan solver must be one of {_WAN_SOLVERS}, got {solver!r}.")
        return resolved

    def _create_scheduler(self, *, flow_shift: float, solver: str):
        if solver == "unipc":
            return WanUniPCMultistepScheduler(flow_shift=flow_shift)
        if solver == "euler":
            return WanEulerScheduler(flow_shift=flow_shift)
        raise ValueError(f"Wan solver must be one of {_WAN_SOLVERS}, got {solver!r}.")

    def _resolve_guidance_pair(
        self, guidance: float | None, guidance_2: float | None | object
    ) -> tuple[float, float | None]:
        guidance_was_default = guidance is None
        resolved_guidance = self._default_guidance() if guidance is None else float(guidance)
        if guidance_2 is _GUIDANCE_2_UNSET:
            default_guidance_2 = self._default_guidance_2()
            if default_guidance_2 is not None:
                resolved_guidance_2 = default_guidance_2 if guidance_was_default else resolved_guidance
            else:
                resolved_guidance_2 = None
        else:
            if guidance_2 is None:
                resolved_guidance_2 = (
                    resolved_guidance if self._wan_config("boundary_ratio", None) is not None else None
                )
            else:
                resolved_guidance_2 = float(guidance_2)
        return resolved_guidance, resolved_guidance_2

    @staticmethod
    def _validate_guidance_values(*, guidance: float, guidance_2: float | None) -> None:
        if not np.isfinite(guidance):
            raise ValueError(f"Wan guidance must be finite, got {guidance!r}.")
        if guidance_2 is not None and not np.isfinite(guidance_2):
            raise ValueError(f"Wan guidance_2 must be finite, got {guidance_2!r}.")

    def _default_negative_prompt(self) -> str:
        return str(self._wan_config("default_negative_prompt", ""))

    def _resolve_negative_prompt(self, negative_prompt: str | None) -> str:
        if negative_prompt is None:
            return self._default_negative_prompt()
        return negative_prompt

    def _wan_config(self, key: str, default):
        return self.model_config.transformer_overrides.get(key, default)

    @staticmethod
    def _batch_timestep(batch_size: int, timestep: int) -> mx.array:
        return mx.full((batch_size,), timestep, dtype=mx.float32)

    def _maybe_release_high_noise_denoiser(
        self,
        *,
        timestep: int,
        boundary_timestep: float | None,
        release_inactive_denoiser: bool,
        already_released: bool,
    ) -> bool:
        if (
            already_released
            or not release_inactive_denoiser
            or boundary_timestep is None
            or timestep >= boundary_timestep
            or self.transformer_2 is None
        ):
            return already_released
        self._release_high_noise_denoiser()
        return True

    def _release_high_noise_denoiser(self) -> None:
        self.transformer = None
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    def _release_denoisers(self) -> None:
        self.transformer = None
        self.transformer_2 = None
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    @staticmethod
    def _cleanup_step_cache(*, clear_cache: bool) -> None:
        if not clear_cache:
            return
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    @staticmethod
    def _materialize_denoise_prediction(prediction: mx.array, *, clear_cache: bool) -> None:
        mx.eval(prediction)
        if clear_cache:
            mx.synchronize()
            mx.clear_cache()

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
