import gc
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map
from PIL import Image

from mflux.callbacks import ProgressCallback
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.wan.model.wan_transformer import WanBlockHealthContext
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.variants.wan2_2_ti2v import Wan2_2_TI2V
from mflux.models.wan.wan_text_encoder_loader import WanTextEncoderLoader
from mflux.models.wan.weights import WanWeightDefinition
from mflux.utils.dimension_resolver import (
    CANVAS_POLICY_EXACT_RESIZE,
    CANVAS_POLICY_SOURCE_ASPECT,
    RESIZE_MODE_RESIZE,
    DimensionResolver,
)
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.generated_video import GeneratedVideo
from mflux.utils.image_util import ImageUtil
from mflux.utils.runtime_memory import RuntimeMemory
from mflux.utils.tensor_health import TensorHealth
from mflux.utils.video_util import VideoUtil


class _MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average: mx.array | int = 0

    def update(self, value: mx.array) -> mx.array:
        self.running_average = value + self.momentum * self.running_average
        return self.running_average


class BerniniRenderer(Wan2_2_TI2V):
    RECOMMENDED_WIDTH = 848
    RECOMMENDED_HEIGHT = 480
    RECOMMENDED_FRAMES = 81
    RECOMMENDED_STEPS = 40
    RECOMMENDED_FPS = 16
    RECOMMENDED_MAX_CONDITION_SIZE = 848
    MIN_PROVEN_CONVERGED_FRAMES = 25
    MAX_PROVEN_SHORT_DEBUG_STEPS = 12
    SYSTEM_PROMPTS = {
        "t2i": "You are a helpful assistant specialized in text-to-image generation.",
        "t2v": "You are a helpful assistant specialized in text-to-video generation.",
        "i2i": "You are a helpful assistant specialized in image editing.",
        "v2v": "You are a helpful assistant specialized in video editing.",
        "mv2v": (
            "You are a helpful assistant for editing. You might need to adjust the video's style, lighting, "
            "colors, textures, and the subject's pose or action."
        ),
        "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
        "rv2v": "You are a helpful assistant specialized in video editing with reference.",
        "ads2v": "You are a helpful assistant specialized in ads insertion.",
    }
    GUIDANCE_MODE_BY_TASK = {
        "t2i": "t2v_apg",
        "t2v": "t2v_apg",
        "i2i": "v2v",
        "v2v": "v2v_apg",
        "mv2v": "v2v_apg",
        "r2v": "r2v_apg",
        "rv2v": "rv2v",
        "ads2v": "rv2v",
    }
    SUPPORTED_GUIDANCE_MODES = frozenset({"rv2v", "v2v", "r2v_apg", "v2v_apg", "t2v", "t2v_apg"})
    ALLOWED_GUIDANCE_MODES_BY_TASK = {
        "t2i": frozenset({"t2v", "t2v_apg"}),
        "t2v": frozenset({"t2v", "t2v_apg"}),
        "i2i": frozenset({"v2v"}),
        "v2v": frozenset({"v2v", "v2v_apg"}),
        "mv2v": frozenset({"v2v", "v2v_apg"}),
        "r2v": frozenset({"r2v_apg"}),
        "rv2v": frozenset({"rv2v"}),
        "ads2v": frozenset({"rv2v", "v2v_apg"}),
    }
    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_target_roles: list[str] | None = None,
        svi_lora_high_path: str | None = None,
        svi_lora_low_path: str | None = None,
        keep_text_encoder_resident: bool = False,
        prompt_embed_disk_cache: bool = True,
    ):
        if quantize is not None:
            raise ValueError(
                "Bernini-R currently supports BF16 inference only. Lower-bit blanket quantization produced "
                "invalid overexposed outputs, while the generic Wan 8-bit predicate quantizes no Bernini "
                "transformer linears and would report misleading metadata. Omit --quantize."
            )
        resolved_config = (
            ModelConfig.bernini_r_1_3b()
            if model_config is None and model_path is None
            else self._resolve_model_config(model_path=model_path, model_config=model_config)
        )
        if not bool(resolved_config.transformer_overrides.get("supports_bernini_renderer", False)):
            raise ValueError(
                "BerniniRenderer requires the exact Bernini-R renderer config; ordinary Wan weights do not "
                "contain the source-ID rotary behavior used by packed reference segments."
            )
        if lora_paths or lora_scales or lora_target_roles or svi_lora_high_path or svi_lora_low_path:
            raise ValueError("LoRA loading is not validated for Bernini-R and is intentionally disabled.")
        super().__init__(
            quantize=quantize,
            model_path=model_path,
            model_config=resolved_config,
            keep_text_encoder_resident=keep_text_encoder_resident,
            prompt_embed_disk_cache=prompt_embed_disk_cache,
        )
        self._promote_vae_to_float32()

    def generate_video(  # noqa: PLR0915
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
        height: int = RECOMMENDED_HEIGHT,
        width: int = RECOMMENDED_WIDTH,
        num_frames: int = RECOMMENDED_FRAMES,
        fps: int = RECOMMENDED_FPS,
        guidance: float | None = None,
        guidance_2: float | None | object = None,
        flow_shift: float | None = None,
        solver: str | None = None,
        denoising_step_list: list[int] | None = None,
        negative_prompt: str | None = None,
        image_path: Path | str | None = None,
        last_image_path: Path | str | None = None,
        context_image_paths: list[Path | str] | None = None,
        context_noise: float | None = None,
        svi_anchor_image_path: Path | str | None = None,
        svi_motion_latent_path: Path | str | None = None,
        svi_motion_latent_count: int = 1,
        svi_motion_latent_export_path: Path | str | None = None,
        video_path: Path | str | None = None,
        reference_video_paths: list[Path | str] | None = None,
        video_strength: float | None = None,
        video_mask_path: Path | str | None = None,
        canvas_policy: str | None = None,
        resize_mode: str = "resize",
        max_sequence_length: int = 512,
        progress_callback: ProgressCallback | None = None,
        release_inactive_denoiser: bool | None = None,
        release_denoisers_before_decode: bool = False,
        clear_cache_each_step: bool = False,
        clear_cache_each_transformer_block: bool = False,
        tensor_health_check_interval: int | None = None,
        compile_transformer: bool = False,
        reference_image_paths: list[Path | str] | None = None,
        reference_guidance: float | None = None,
        source_guidance: float | None = None,
        apg_eta: float | None = None,
        apg_norm_threshold: float | None = None,
        apg_momentum: float | None = None,
        max_condition_size: int = RECOMMENDED_MAX_CONDITION_SIZE,
        system_prompt: str | None = None,
        guidance_mode: str | None = None,
        task_type: str | None = None,
    ) -> GeneratedVideo:
        start_time = time.time()
        image_path = Path(image_path) if image_path is not None else None
        reference_image_paths = [Path(path) for path in (reference_image_paths or [])]
        reference_video_paths = [Path(path) for path in (reference_video_paths or [])]
        video_path = Path(video_path) if video_path is not None else None
        has_primary_image = image_path is not None
        has_primary_video = video_path is not None
        has_any_video_condition = has_primary_video or bool(reference_video_paths)
        if not has_any_video_condition:
            normalized_canvas_policy = (
                CANVAS_POLICY_EXACT_RESIZE
                if canvas_policy is None
                else DimensionResolver.normalize_canvas_policy(canvas_policy)
            )
            if normalized_canvas_policy != CANVAS_POLICY_EXACT_RESIZE:
                raise ValueError(
                    "Bernini-R reference-to-video has no source canvas; omit canvas_policy or use exact-resize."
                )
            canvas_policy = CANVAS_POLICY_EXACT_RESIZE
        else:
            canvas_policy = DimensionResolver.normalize_canvas_policy(canvas_policy)
        guidance_mode = self._resolved_guidance_mode(
            guidance_mode=guidance_mode,
            task_type=task_type,
            has_video=has_primary_video,
            has_image=has_primary_image,
            num_reference_images=len(reference_image_paths),
            num_reference_videos=len(reference_video_paths),
        )
        task_type = self._resolved_task_type(
            task_type=task_type,
            guidance_mode=guidance_mode,
            has_video=has_primary_video,
            has_image=has_primary_image,
            num_reference_images=len(reference_image_paths),
            num_reference_videos=len(reference_video_paths),
        )
        self._validate_bernini_request(
            num_inference_steps=num_inference_steps,
            fps=fps,
            guidance_2=guidance_2,
            denoising_step_list=denoising_step_list,
            image_path=image_path,
            last_image_path=last_image_path,
            context_image_paths=context_image_paths,
            context_noise=context_noise,
            svi_anchor_image_path=svi_anchor_image_path,
            svi_motion_latent_path=svi_motion_latent_path,
            svi_motion_latent_count=svi_motion_latent_count,
            svi_motion_latent_export_path=svi_motion_latent_export_path,
            video_path=video_path,
            reference_video_paths=reference_video_paths,
            video_strength=video_strength,
            video_mask_path=video_mask_path,
            reference_image_paths=reference_image_paths,
            release_inactive_denoiser=release_inactive_denoiser,
            compile_transformer=compile_transformer,
            max_condition_size=max_condition_size,
            max_sequence_length=max_sequence_length,
            tensor_health_check_interval=tensor_health_check_interval,
            guidance_mode=guidance_mode,
        )
        del (
            guidance_2,
            denoising_step_list,
            last_image_path,
            context_image_paths,
            context_noise,
            svi_anchor_image_path,
            svi_motion_latent_path,
            svi_motion_latent_count,
            svi_motion_latent_export_path,
            video_strength,
            video_mask_path,
            release_inactive_denoiser,
            compile_transformer,
        )

        requested_height, requested_width = self._validated_bernini_spatial_size(height=height, width=width)
        requested_frames = self._validated_frame_count(num_frames)
        resize_mode = DimensionResolver.normalize_resize_mode(resize_mode)
        if video_path is not None and canvas_policy != CANVAS_POLICY_SOURCE_ASPECT:
            raise ValueError(
                "Bernini-R video conditioning supports only source-aspect output, matching the official "
                "renderer. exact-resize is not yet visually proven for this route."
            )
        if resize_mode != RESIZE_MODE_RESIZE:
            raise ValueError(
                "Bernini-R supports only resize mode 'resize'; crop and pad are not official renderer modes."
            )
        num_inference_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else self._wan_config("default_steps", self.RECOMMENDED_STEPS)
        )
        text_guidance = self._resolved_task_or_config_float(
            guidance,
            task_default=None,
            config_key="default_guidance",
            fallback=4.0,
            label="guidance",
        )
        reference_guidance = self._resolved_task_or_config_float(
            reference_guidance,
            task_default=None,
            config_key="default_reference_guidance",
            fallback=4.5,
            label="reference_guidance",
        )
        source_guidance = self._resolved_task_or_config_float(
            source_guidance,
            task_default=None,
            config_key="default_source_guidance",
            fallback=1.25,
            label="source_guidance",
        )
        apg_eta = self._resolved_finite_float(
            apg_eta,
            config_key="default_apg_eta",
            fallback=0.5,
            label="apg_eta",
        )
        apg_norm_threshold = self._resolved_finite_float(
            apg_norm_threshold,
            config_key="default_apg_norm_threshold",
            fallback=50.0,
            label="apg_norm_threshold",
        )
        apg_momentum = self._resolved_finite_float(
            apg_momentum,
            config_key="default_apg_momentum",
            fallback=0.0,
            label="apg_momentum",
        )
        if apg_norm_threshold < 0:
            raise ValueError("apg_norm_threshold must be greater than or equal to zero.")
        flow_shift = self._resolved_finite_float(
            flow_shift,
            config_key="flow_shift",
            fallback=3.0,
            label="flow_shift",
        )
        solver = solver or str(self._wan_config("default_solver", "unipc"))
        if solver != "unipc":
            raise ValueError("Bernini-R currently supports only the unipc solver used by the official renderer.")
        negative_prompt = self._resolve_negative_prompt(negative_prompt)

        system_prompt = self._resolved_system_prompt(
            guidance_mode=guidance_mode,
            system_prompt=system_prompt,
            task_type=task_type,
        )
        effective_prompt = self._effective_prompt(system_prompt=system_prompt, prompt=prompt)
        task = self._public_task(task_type=task_type, has_video=has_primary_video)
        scheduler = self._create_scheduler(flow_shift=flow_shift, solver=solver)
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps.tolist()
        total_steps = len(timesteps)
        condition_plan = self._plan_condition_metadata(
            video_path=video_path,
            requested_height=requested_height,
            requested_width=requested_width,
            requested_frames=requested_frames,
            fps=fps,
            canvas_policy=canvas_policy,
            max_condition_size=max_condition_size,
        )
        height = int(condition_plan["output_height"])
        width = int(condition_plan["output_width"])
        num_frames = int(condition_plan["output_frames"])
        if task not in {"text-to-image", "image-to-image"}:
            self._validate_supported_frame_step_domain(
                num_frames=num_frames,
                num_inference_steps=total_steps,
            )
        progress_registry = getattr(self, "callbacks", None)
        self._emit_progress(
            progress_callback,
            phase="start",
            frame=0,
            total_frames=num_frames,
            step=0,
            total_steps=total_steps,
            task=task,
            registry=progress_registry,
            width=width,
            height=height,
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=effective_prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            max_sequence_length=max_sequence_length,
            clean_prompts=False,
        )
        if negative_prompt_embeds is None:
            raise RuntimeError("Bernini-R requires an unconditional text embedding for its guidance branches.")
        self._require_tensor_health(prompt_embeds, phase="prompt-encoding", name="prompt_embeds")
        self._require_tensor_health(
            negative_prompt_embeds,
            phase="prompt-encoding",
            name="negative_prompt_embeds",
        )

        prepared_conditions = self._prepare_condition_latents(
            image_path=image_path,
            video_path=video_path,
            reference_video_paths=reference_video_paths,
            reference_image_paths=reference_image_paths,
            requested_height=requested_height,
            requested_width=requested_width,
            requested_frames=requested_frames,
            fps=fps,
            canvas_policy=canvas_policy,
            resize_mode=resize_mode,
            max_condition_size=max_condition_size,
            clear_cache=clear_cache_each_step,
            condition_plan=condition_plan,
        )
        if len(prepared_conditions) == 3:
            video_condition, reference_conditions, condition_metadata = prepared_conditions
            reference_video_conditions = []
        else:
            video_condition, reference_video_conditions, reference_conditions, condition_metadata = prepared_conditions
        self._require_condition_plan_match(condition_plan=condition_plan, condition_metadata=condition_metadata)
        combined_conditions = self._combined_condition_segments(
            video_condition=video_condition,
            reference_video_conditions=reference_video_conditions,
            reference_conditions=reference_conditions,
        )

        latents = self.prepare_latents(
            seed=seed,
            batch_size=1,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        self._require_tensor_health(latents, phase="prepare-latents", name="latents")
        # Whole-step cleanup already handles the ordinary low-RAM path.
        # Clearing again after every guidance branch adds a large sync penalty
        # on video cases, so keep branch-level cache flushes only for the
        # per-transformer-block emergency policy.
        branch_cache_clear = clear_cache_each_transformer_block
        r2v_buffers = (
            [_MomentumBuffer(apg_momentum), _MomentumBuffer(apg_momentum)] if guidance_mode == "r2v_apg" else None
        )
        v2v_buffer = _MomentumBuffer(apg_momentum) if guidance_mode == "v2v_apg" else None

        for step_index, timestep in enumerate(timesteps):
            step_number = step_index + 1
            check_tensors = TensorHealth.should_check_step(
                step_number,
                total_steps,
                tensor_health_check_interval,
            )
            branch_kwargs = {
                "target": latents,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "timestep": timestep,
                "step_number": step_number,
                "total_steps": total_steps,
                "clear_cache_each_block": clear_cache_each_transformer_block,
                "clear_branch_cache": branch_cache_clear,
                "check_tensors": check_tensors,
            }
            if guidance_mode == "r2v_apg":
                sigma = self._apg_sigma(scheduler=scheduler, fallback_index=step_index)
                noise_pred = self._r2v_noise_prediction(
                    reference_conditions=reference_conditions,
                    reference_guidance=reference_guidance,
                    text_guidance=text_guidance,
                    sigma=sigma,
                    buffers=r2v_buffers,
                    eta=apg_eta,
                    norm_threshold=apg_norm_threshold,
                    **branch_kwargs,
                )
            elif guidance_mode == "rv2v":
                noise_pred = self._rv2v_noise_prediction(
                    video_condition=video_condition,
                    reference_video_conditions=reference_video_conditions,
                    reference_conditions=reference_conditions,
                    source_guidance=source_guidance,
                    reference_guidance=reference_guidance,
                    text_guidance=text_guidance,
                    **branch_kwargs,
                )
            elif guidance_mode in {"v2v", "t2v"}:
                noise_pred = self._cfg_noise_prediction(
                    condition_segments=combined_conditions,
                    text_guidance=text_guidance,
                    **branch_kwargs,
                )
            else:
                sigma = self._apg_sigma(scheduler=scheduler, fallback_index=step_index)
                noise_pred = self._single_condition_apg_noise_prediction(
                    condition_segments=combined_conditions,
                    text_guidance=text_guidance,
                    sigma=sigma,
                    buffer=v2v_buffer,
                    eta=apg_eta,
                    norm_threshold=apg_norm_threshold,
                    **branch_kwargs,
                )
            mx.eval(noise_pred)
            if check_tensors:
                self._require_tensor_health(
                    noise_pred,
                    phase="guided-denoise-prediction",
                    name="noise_pred",
                    step=step_number,
                    total_steps=total_steps,
                    timestep=timestep,
                    denoiser="bernini-renderer",
                    guidance=text_guidance,
                )
            latents = scheduler.step(noise_pred.astype(mx.float32), timestep, latents, return_dict=False)[0]
            mx.eval(latents)
            if check_tensors:
                self._require_tensor_health(
                    latents,
                    phase="scheduler-step",
                    name="latents",
                    step=step_number,
                    total_steps=total_steps,
                    timestep=timestep,
                    denoiser="bernini-renderer",
                    guidance=text_guidance,
                )
            self._emit_progress(
                progress_callback,
                phase="denoise",
                frame=self._progress_frame_for_step(step_index, total_steps, num_frames),
                total_frames=num_frames,
                step=step_number,
                total_steps=total_steps,
                task=task,
                timestep=timestep,
                registry=progress_registry,
            )
            del noise_pred
            self._cleanup_step_cache(clear_cache=clear_cache_each_step)

        condition_shapes = self._condition_shapes(combined_conditions)
        source_ids = self._configured_source_ids(len(combined_conditions))
        del (
            prompt_embeds,
            negative_prompt_embeds,
            scheduler,
            video_condition,
            reference_video_conditions,
            reference_conditions,
            combined_conditions,
        )
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
        decode_latents = RuntimeMemory.materialize_inference_tree(latents.astype(ModelConfig.precision))
        del latents
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

        def frame_batches_factory():
            decoded_slices = self.vae.iter_decode_normalized_latent_slices(
                decode_latents,
                clear_cache_each_slice=clear_cache_each_step,
                tile_spatial=clear_cache_each_step,
            )
            return VideoUtil.decoded_latent_slices_to_frame_batches(
                decoded_slices,
                batch_size=8,
                total_frames=num_frames,
            )

        extra_metadata = {
            **(getattr(self, "_last_prompt_truncation", None) or {}),
            **self._bernini_extra_metadata(
                guidance_mode=guidance_mode,
                reference_image_paths=reference_image_paths,
                reference_video_paths=reference_video_paths,
                text_guidance=text_guidance,
                reference_guidance=reference_guidance,
                source_guidance=source_guidance,
                apg_eta=apg_eta,
                apg_norm_threshold=apg_norm_threshold,
                apg_momentum=apg_momentum,
                max_condition_size=max_condition_size,
                system_prompt=system_prompt,
                effective_prompt=effective_prompt,
                unipc_flow_sigma_schedule=str(self._wan_config("unipc_flow_sigma_schedule", "endpoint-inclusive")),
                source_ids=source_ids,
                condition_shapes=condition_shapes,
                condition_metadata=condition_metadata,
                component_source_provenance=getattr(self, "component_source_provenance", {}),
                factored_component_sources=bool(getattr(self, "factored_component_sources", False)),
                vae_low_memory_policy_active=clear_cache_each_step,
                clear_cache_each_transformer_block=clear_cache_each_transformer_block,
                release_denoisers_before_decode=release_denoisers_before_decode,
                task_type=task_type,
            ),
        }
        video = VideoUtil.to_video_from_frame_batches(
            frame_batches_factory=frame_batches_factory,
            fps=fps,
            model_config=self.model_config,
            seed=seed,
            prompt=prompt,
            steps=num_inference_steps,
            guidance=text_guidance,
            guidance_2=None,
            flow_shift=flow_shift,
            solver=solver,
            quantization=self.bits,
            generation_time=time.time() - start_time,
            height=height,
            width=width,
            frame_count=num_frames,
            task=task,
            image_path=None,
            video_path=video_path,
            negative_prompt=negative_prompt,
            source_width=condition_metadata.get("source_width"),
            source_height=condition_metadata.get("source_height"),
            requested_width=requested_width,
            requested_height=requested_height,
            canvas_policy=canvas_policy,
            resize_mode=resize_mode,
            lora_paths=None,
            lora_scales=None,
            extra_metadata=extra_metadata,
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

    def generate_image(
        self,
        *,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
        height: int = RECOMMENDED_HEIGHT,
        width: int = RECOMMENDED_WIDTH,
        guidance: float | None = None,
        flow_shift: float | None = None,
        solver: str | None = None,
        negative_prompt: str | None = None,
        image_path: Path | str | None = None,
        canvas_policy: str | None = None,
        resize_mode: str = "resize",
        max_sequence_length: int = 512,
        progress_callback: ProgressCallback | None = None,
        clear_cache_each_step: bool = False,
        clear_cache_each_transformer_block: bool = False,
        tensor_health_check_interval: int | None = None,
        compile_transformer: bool = False,
        reference_guidance: float | None = None,
        source_guidance: float | None = None,
        apg_eta: float | None = None,
        apg_norm_threshold: float | None = None,
        apg_momentum: float | None = None,
        max_condition_size: int = RECOMMENDED_MAX_CONDITION_SIZE,
        system_prompt: str | None = None,
        guidance_mode: str | None = None,
        task_type: str | None = None,
    ) -> GeneratedImage:
        resolved_task_type = task_type or ("i2i" if image_path is not None else "t2i")
        artifact = self.generate_video(
            seed=seed,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=1,
            fps=1,
            guidance=guidance,
            flow_shift=flow_shift,
            solver=solver,
            negative_prompt=negative_prompt,
            image_path=image_path,
            canvas_policy=canvas_policy,
            resize_mode=resize_mode,
            max_sequence_length=max_sequence_length,
            progress_callback=progress_callback,
            clear_cache_each_step=clear_cache_each_step,
            clear_cache_each_transformer_block=clear_cache_each_transformer_block,
            tensor_health_check_interval=tensor_health_check_interval,
            compile_transformer=compile_transformer,
            reference_guidance=reference_guidance,
            source_guidance=source_guidance,
            apg_eta=apg_eta,
            apg_norm_threshold=apg_norm_threshold,
            apg_momentum=apg_momentum,
            max_condition_size=max_condition_size,
            system_prompt=system_prompt,
            guidance_mode=guidance_mode,
            task_type=resolved_task_type,
        )
        source_image_width = None
        source_image_height = None
        if image_path is not None:
            source = ImageUtil.load_image(image_path)
            source_image_width = source.width
            source_image_height = source.height
        return GeneratedImage(
            image=artifact.first_frame(),
            model_config=self.model_config,
            seed=seed,
            prompt=prompt,
            steps=artifact.steps,
            guidance=artifact.guidance,
            precision=artifact.precision,
            quantization=artifact.quantization,
            generation_time=artifact.generation_time,
            height=artifact.height,
            width=artifact.width,
            image_path=image_path,
            negative_prompt=artifact.negative_prompt,
            canvas_policy=artifact.canvas_policy,
            resize_mode=artifact.resize_mode,
            requested_width=artifact.requested_width,
            requested_height=artifact.requested_height,
            source_image_width=source_image_width,
            source_image_height=source_image_height,
            extra_metadata=artifact.extra_metadata,
        )

    def _predict_branch(
        self,
        *,
        role: str,
        target: mx.array,
        condition_segments: list[mx.array],
        source_ids: list[float],
        text_embeds: mx.array,
        timestep: int | float,
        step_number: int,
        total_steps: int,
        clear_cache_each_block: bool,
        clear_branch_cache: bool,
        check_tensors: bool,
    ) -> mx.array:
        if len(condition_segments) != len(source_ids):
            raise ValueError("Bernini condition segments and source IDs must have the same length.")
        block_context = WanBlockHealthContext(
            step=step_number,
            total_steps=total_steps,
            timestep=timestep,
            denoiser=f"bernini-{role}",
            guidance=None,
        )
        prediction = self.transformer.forward_packed(
            latent_segments=[
                *(segment.astype(ModelConfig.precision) for segment in condition_segments),
                target.astype(ModelConfig.precision),
            ],
            source_ids=[*source_ids, 0.0],
            timestep=self._batch_timestep(batch_size=int(target.shape[0]), timestep=timestep),
            encoder_hidden_states=text_embeds,
            target_segment_index=-1,
            clear_cache_each_block=clear_cache_each_block,
            block_health_context=block_context,
        ).astype(mx.float32)
        mx.eval(prediction)
        if check_tensors:
            self._require_tensor_health(
                prediction,
                phase="branch-denoise-prediction",
                name=f"noise_pred.{role}",
                step=step_number,
                total_steps=total_steps,
                timestep=timestep,
                denoiser=f"bernini-{role}",
            )
        if clear_branch_cache:
            mx.synchronize()
            mx.clear_cache()
        return prediction

    def _prepare_packed_branch(
        self,
        *,
        target: mx.array,
        condition_segments: list[mx.array],
        source_ids: list[float],
    ):
        return self.transformer.prepare_packed_segments(
            latent_segments=[
                *(segment.astype(ModelConfig.precision) for segment in condition_segments),
                target.astype(ModelConfig.precision),
            ],
            source_ids=[*source_ids, 0.0],
            target_segment_index=-1,
        )

    def _predict_prepacked_branch(
        self,
        *,
        role: str,
        prepared,
        text_embeds: mx.array,
        timestep: int | float,
        step_number: int,
        total_steps: int,
        clear_cache_each_block: bool,
        clear_branch_cache: bool,
        check_tensors: bool,
    ) -> mx.array:
        block_context = WanBlockHealthContext(
            step=step_number,
            total_steps=total_steps,
            timestep=timestep,
            denoiser=f"bernini-{role}",
            guidance=None,
        )
        prediction = self.transformer.forward_prepacked(
            prepared=prepared,
            timestep=self._batch_timestep(batch_size=int(prepared.target_shape[0]), timestep=timestep),
            encoder_hidden_states=text_embeds,
            clear_cache_each_block=clear_cache_each_block,
            block_health_context=block_context,
        ).astype(mx.float32)
        mx.eval(prediction)
        if check_tensors:
            self._require_tensor_health(
                prediction,
                phase="branch-denoise-prediction",
                name=f"noise_pred.{role}",
                step=step_number,
                total_steps=total_steps,
                timestep=timestep,
                denoiser=f"bernini-{role}",
            )
        if clear_branch_cache:
            mx.synchronize()
            mx.clear_cache()
        return prediction

    def _rv2v_noise_prediction(
        self,
        *,
        target: mx.array,
        video_condition: mx.array,
        reference_video_conditions: list[mx.array] | None = None,
        reference_conditions: list[mx.array],
        prompt_embeds: mx.array,
        negative_prompt_embeds: mx.array,
        source_guidance: float,
        reference_guidance: float,
        text_guidance: float,
        **branch_kwargs,
    ) -> mx.array:
        combined = [
            video_condition,
            *(reference_video_conditions or []),
            *reference_conditions,
        ]
        combined_ids = self._configured_source_ids(len(combined))
        empty = self._predict_branch(
            role="rv2v-empty",
            target=target,
            condition_segments=[],
            source_ids=[],
            text_embeds=negative_prompt_embeds,
            **branch_kwargs,
        )
        video = self._predict_branch(
            role="rv2v-video",
            target=target,
            condition_segments=[video_condition],
            source_ids=[1.0],
            text_embeds=negative_prompt_embeds,
            **branch_kwargs,
        )
        result = empty + source_guidance * (video - empty)
        mx.eval(result)
        del empty
        video_references = self._predict_branch(
            role="rv2v-video-references",
            target=target,
            condition_segments=combined,
            source_ids=combined_ids,
            text_embeds=negative_prompt_embeds,
            **branch_kwargs,
        )
        result = result + reference_guidance * (video_references - video)
        mx.eval(result)
        del video
        video_references_text = self._predict_branch(
            role="rv2v-video-references-text",
            target=target,
            condition_segments=combined,
            source_ids=combined_ids,
            text_embeds=prompt_embeds,
            **branch_kwargs,
        )
        result = result + text_guidance * (video_references_text - video_references)
        mx.eval(result)
        return result

    def _r2v_noise_prediction(
        self,
        *,
        target: mx.array,
        reference_conditions: list[mx.array],
        prompt_embeds: mx.array,
        negative_prompt_embeds: mx.array,
        reference_guidance: float,
        text_guidance: float,
        sigma: mx.array,
        buffers: list[_MomentumBuffer],
        eta: float,
        norm_threshold: float,
        **branch_kwargs,
    ) -> mx.array:
        reference_ids = self._configured_source_ids(len(reference_conditions))
        empty_v = self._predict_branch(
            role="r2v-empty",
            target=target,
            condition_segments=[],
            source_ids=[],
            text_embeds=negative_prompt_embeds,
            **branch_kwargs,
        )
        reference_v = self._predict_branch(
            role="r2v-references",
            target=target,
            condition_segments=reference_conditions,
            source_ids=reference_ids,
            text_embeds=negative_prompt_embeds,
            **branch_kwargs,
        )
        noisy_target = target.astype(mx.float32)
        sigma_value = sigma.astype(mx.float32)
        empty_sample = noisy_target - sigma_value * empty_v
        reference_sample = noisy_target - sigma_value * reference_v
        guided_sample = empty_sample + reference_guidance * self._normalize_diff(
            reference_sample - empty_sample,
            reference_sample,
            momentum_buffer=buffers[0],
            eta=eta,
            norm_threshold=norm_threshold,
        )
        mx.eval(guided_sample)
        del empty_v
        text_v = self._predict_branch(
            role="r2v-references-text",
            target=target,
            condition_segments=reference_conditions,
            source_ids=reference_ids,
            text_embeds=prompt_embeds,
            **branch_kwargs,
        )
        text_sample = noisy_target - sigma_value * text_v
        guided_sample = guided_sample + text_guidance * self._normalize_diff(
            text_sample - reference_sample,
            text_sample,
            momentum_buffer=buffers[1],
            eta=eta,
            norm_threshold=norm_threshold,
        )
        noise_pred = (noisy_target - guided_sample) / sigma_value
        mx.eval(noise_pred)
        del reference_v, text_v
        return noise_pred

    def _v2v_noise_prediction(
        self,
        *,
        target: mx.array,
        video_condition: mx.array,
        prompt_embeds: mx.array,
        negative_prompt_embeds: mx.array,
        text_guidance: float,
        sigma: mx.array,
        buffer: _MomentumBuffer,
        eta: float,
        norm_threshold: float,
        **branch_kwargs,
    ) -> mx.array:
        uncond_v = self._predict_branch(
            role="v2v-empty-text",
            target=target,
            condition_segments=[video_condition],
            source_ids=[1.0],
            text_embeds=negative_prompt_embeds,
            **branch_kwargs,
        )
        cond_v = self._predict_branch(
            role="v2v-video-text",
            target=target,
            condition_segments=[video_condition],
            source_ids=[1.0],
            text_embeds=prompt_embeds,
            **branch_kwargs,
        )
        noisy_target = target.astype(mx.float32)
        sigma_value = sigma.astype(mx.float32)
        uncond_sample = noisy_target - sigma_value * uncond_v
        cond_sample = noisy_target - sigma_value * cond_v
        guided_sample = uncond_sample + text_guidance * self._normalize_diff(
            cond_sample - uncond_sample,
            cond_sample,
            momentum_buffer=buffer,
            eta=eta,
            norm_threshold=norm_threshold,
        )
        noise_pred = (noisy_target - guided_sample) / sigma_value
        mx.eval(noise_pred)
        del uncond_v, cond_v
        return noise_pred

    @staticmethod
    def _normalize_diff(
        diff: mx.array,
        base_pred: mx.array,
        *,
        momentum_buffer: _MomentumBuffer | None,
        eta: float,
        norm_threshold: float,
    ) -> mx.array:
        if momentum_buffer is not None:
            diff = momentum_buffer.update(diff)
        reduction_axes = (1, 3, 4) if diff.ndim == 5 else tuple(range(1, diff.ndim))
        diff_np = np.asarray(diff)
        if norm_threshold > 0:
            diff_norm = np.sqrt(np.sum(np.square(diff_np), axis=reduction_axes, keepdims=True))
            scale = np.divide(
                norm_threshold,
                diff_norm,
                out=np.full_like(diff_norm, np.inf),
                where=diff_norm != 0,
            )
            diff_np = diff_np * np.minimum(np.ones_like(diff_np), scale)
        base_np = np.asarray(base_pred)
        diff64 = diff_np.astype(np.float64, copy=False)
        base64 = base_np.astype(np.float64, copy=False)
        base_norm = np.sqrt(np.sum(np.square(base64), axis=reduction_axes, keepdims=True))
        unit_base = base64 / np.maximum(base_norm, np.float64(1e-12))
        parallel64 = np.sum(diff64 * unit_base, axis=reduction_axes, keepdims=True) * unit_base
        parallel = parallel64.astype(diff_np.dtype, copy=False)
        orthogonal = (diff64 - parallel64).astype(diff_np.dtype, copy=False)
        return mx.array(orthogonal + eta * parallel, dtype=diff.dtype)

    def _prepare_condition_latents(
        self,
        *,
        image_path: Path | None,
        video_path: Path | None,
        reference_video_paths: list[Path],
        reference_image_paths: list[Path],
        requested_height: int,
        requested_width: int,
        requested_frames: int,
        fps: int,
        canvas_policy: str,
        resize_mode: str,
        max_condition_size: int,
        clear_cache: bool,
        condition_plan: dict | None = None,
    ) -> tuple[mx.array | None, list[mx.array], list[mx.array], dict]:
        metadata: dict = {
            "output_height": requested_height,
            "output_width": requested_width,
            "output_frames": requested_frames,
        }
        video_condition = None
        reference_video_conditions = []
        if video_path is not None:
            video_pixels, video_metadata = self._preprocess_video_condition(
                video_path=video_path,
                requested_height=requested_height,
                requested_width=requested_width,
                requested_frames=requested_frames,
                fps=fps,
                canvas_policy=canvas_policy,
                resize_mode=resize_mode,
                max_condition_size=max_condition_size,
                condition_plan=condition_plan,
            )
            metadata.update(video_metadata)
            video_condition = self._encode_condition_pixels(
                video_pixels,
                name="source_video",
                clear_cache=clear_cache,
            )
            del video_pixels

        reference_video_pixel_shapes = []
        reference_video_metadata = []
        for index, path in enumerate(reference_video_paths):
            pixels, video_metadata = self._preprocess_video_condition(
                video_path=path,
                requested_height=requested_height,
                requested_width=requested_width,
                requested_frames=requested_frames,
                fps=fps,
                canvas_policy=canvas_policy,
                resize_mode=resize_mode,
                max_condition_size=max_condition_size,
            )
            reference_video_pixel_shapes.append([int(value) for value in pixels.shape])
            reference_video_metadata.append(video_metadata)
            reference_video_conditions.append(
                self._encode_condition_pixels(
                    pixels,
                    name=f"reference_video_{index}",
                    clear_cache=clear_cache,
                )
            )
            del pixels

        reference_conditions = []
        if image_path is not None:
            source_pixels = self._preprocess_reference_image(image_path, max_condition_size=max_condition_size)
            metadata["source_image_pixel_shape"] = [int(value) for value in source_pixels.shape]
            reference_conditions.append(
                self._encode_condition_pixels(
                    source_pixels,
                    name="source_image",
                    clear_cache=clear_cache,
                )
            )
            del source_pixels
        reference_pixel_shapes = []
        for index, path in enumerate(reference_image_paths):
            pixels = self._preprocess_reference_image(path, max_condition_size=max_condition_size)
            reference_pixel_shapes.append([int(value) for value in pixels.shape])
            reference_conditions.append(
                self._encode_condition_pixels(
                    pixels,
                    name=f"reference_image_{index}",
                    clear_cache=clear_cache,
                )
            )
            del pixels
        metadata["reference_video_pixel_shapes"] = reference_video_pixel_shapes
        metadata["reference_video_metadata"] = reference_video_metadata
        metadata["reference_pixel_shapes"] = reference_pixel_shapes
        return video_condition, reference_video_conditions, reference_conditions, metadata

    def _plan_condition_metadata(
        self,
        *,
        video_path: Path | None,
        requested_height: int,
        requested_width: int,
        requested_frames: int,
        fps: int,
        canvas_policy: str,
        max_condition_size: int,
    ) -> dict:
        metadata: dict[str, Any] = {
            "output_height": requested_height,
            "output_width": requested_width,
            "output_frames": requested_frames,
            "requested_output_height": requested_height,
            "requested_output_width": requested_width,
            "requested_output_frames": requested_frames,
        }
        if video_path is None:
            return metadata
        source_info = VideoUtil.inspect_video(video_path)
        total_frames = source_info.source_frame_count
        source_width = source_info.source_width
        source_height = source_info.source_height
        source_fps = source_info.fps
        if (
            total_frames is None
            or total_frames < 1
            or source_width is None
            or source_height is None
            or source_fps is None
            or source_fps <= 0
        ):
            clip = VideoUtil.read_video_clip(video_path)
            if not clip.frames:
                raise ValueError(f"Bernini-R could not decode any source frames from {video_path}.")
            total_frames = len(clip.frames)
            source_width, source_height = clip.frames[0].size
            source_fps = clip.fps
        indices = self._smart_video_indices(
            total_frames=int(total_frames),
            video_fps=float(source_fps),
            fps=fps,
            frame_factor=4,
            max_frames=requested_frames,
            add_one=True,
        )
        if canvas_policy == CANVAS_POLICY_EXACT_RESIZE:
            condition_width, condition_height = self._condition_dimensions(
                width=int(source_width),
                height=int(source_height),
                max_size=max_condition_size,
            )
            output_width, output_height = requested_width, requested_height
        else:
            # Official renderer contract (bernini/data_utils.py
            # MaxLongEdgeMinShortEdgeResize + pipeline.py __call__): a source
            # video is resized so its long edge fits max_image_size, never
            # upscaled, snapped to stride 16, and that size IS the output
            # canvas. Requested --width/--height do not override video-driven
            # output; use --max-condition-size to control the canvas.
            condition_width, condition_height = self._condition_dimensions(
                width=int(source_width),
                height=int(source_height),
                max_size=max_condition_size,
            )
            output_width, output_height = condition_width, condition_height
            if (output_height, output_width) != (requested_height, requested_width):
                print(
                    "Bernini-R source-aspect output follows the official renderer rule: the source "
                    "video's long edge is capped at --max-condition-size (never upscaled) and that "
                    "resolves the canvas: "
                    f"requested ({requested_height}, {requested_width}) -> ({output_height}, {output_width})."
                )
        metadata.update(
            source_width=int(source_width),
            source_height=int(source_height),
            source_fps=float(source_fps),
            source_frame_count=int(total_frames),
            requested_output_width=int(requested_width),
            requested_output_height=int(requested_height),
            requested_output_frames=int(requested_frames),
            source_sample_indices=[int(index) for index in indices],
            video_condition_width=int(condition_width),
            video_condition_height=int(condition_height),
            video_condition_frames=len(indices),
            output_width=int(output_width),
            output_height=int(output_height),
            output_frames=len(indices),
        )
        return metadata

    @staticmethod
    def _require_condition_plan_match(*, condition_plan: dict, condition_metadata: dict) -> None:
        for key in ("output_width", "output_height", "output_frames", "source_sample_indices"):
            if condition_plan.get(key) != condition_metadata.get(key):
                raise RuntimeError(
                    "Bernini-R source video changed between progress planning and conditioning "
                    f"for {key}: the start event reported {condition_plan.get(key)!r}, "
                    f"but conditioning resolved {condition_metadata.get(key)!r}."
                )

    def _encode_condition_pixels(self, pixels: mx.array, *, name: str, clear_cache: bool) -> mx.array:
        self._require_tensor_health(pixels, phase="conditioning-preprocess", name=f"{name}_pixels")
        # Preserve the official float32 visual input path. Component weight
        # precision is configured independently from transformer execution.
        latents = self.vae.encode_normalized(
            pixels.astype(mx.float32),
            clear_cache_each_slice=clear_cache,
            tile_spatial=clear_cache,
        ).astype(mx.float32)
        mx.eval(latents)
        self._require_tensor_health(latents, phase="conditioning-encode", name=f"{name}_latents")
        if clear_cache:
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
        return latents

    def _preprocess_video_condition(
        self,
        *,
        video_path: Path,
        requested_height: int,
        requested_width: int,
        requested_frames: int,
        fps: int,
        canvas_policy: str,
        resize_mode: str,
        max_condition_size: int,
        condition_plan: dict | None = None,
    ) -> tuple[mx.array, dict]:
        plan = condition_plan or self._plan_condition_metadata(
            video_path=video_path,
            requested_height=requested_height,
            requested_width=requested_width,
            requested_frames=requested_frames,
            fps=fps,
            canvas_policy=canvas_policy,
            max_condition_size=max_condition_size,
        )
        total_frames = int(plan["source_frame_count"])
        source_fps = float(plan["source_fps"])
        indices = [int(index) for index in plan["source_sample_indices"]]
        frames = self._read_indexed_video_frames(video_path, indices)
        if not frames:
            raise ValueError(f"Bernini-R could not decode any source frames from {video_path}.")

        condition_width = int(plan["video_condition_width"])
        condition_height = int(plan["video_condition_height"])
        if canvas_policy == CANVAS_POLICY_EXACT_RESIZE:
            processed = [
                ImageUtil.scale_to_dimensions(
                    frame.convert("RGB"),
                    target_width=condition_width,
                    target_height=condition_height,
                    resize_mode=resize_mode,
                )
                for frame in frames
            ]
            output_width, output_height = int(plan["output_width"]), int(plan["output_height"])
        else:
            processed = [
                frame.convert("RGB").resize(
                    (condition_width, condition_height),
                    resample=Image.Resampling.BICUBIC,
                )
                if frame.size != (condition_width, condition_height)
                else frame.convert("RGB")
                for frame in frames
            ]
            output_width, output_height = int(plan["output_width"]), int(plan["output_height"])

        frames_np = np.stack([np.asarray(frame, dtype=np.float32) for frame in processed], axis=0)
        frames_np = frames_np / 127.5 - 1.0
        pixels = mx.transpose(mx.array(frames_np)[None, ...], (0, 4, 1, 2, 3))
        return pixels, {
            "source_width": int(plan["source_width"]),
            "source_height": int(plan["source_height"]),
            "source_fps": float(source_fps),
            "source_frame_count": int(total_frames),
            "source_sample_indices": [int(index) for index in indices],
            "video_condition_width": int(condition_width),
            "video_condition_height": int(condition_height),
            "video_condition_frames": len(processed),
            "condition_resize_backend": (
                f"image-util-{resize_mode}" if canvas_policy == CANVAS_POLICY_EXACT_RESIZE else "pillow-bicubic"
            ),
            "output_width": int(output_width),
            "output_height": int(output_height),
            "output_frames": len(processed),
        }

    @staticmethod
    def _read_indexed_video_frames(video_path: Path, indices: list[int]) -> list[Image.Image]:
        if not indices:
            return []
        try:
            import av
        except ImportError:
            clip = VideoUtil.read_video_clip(video_path, max_frames=max(indices) + 1)
            return [clip.frames[index] for index in indices]

        frames = []
        next_index = 0
        with av.open(str(video_path)) as container:
            if not container.streams.video:
                raise ValueError(f"Bernini-R could not find a video stream in {video_path}.")
            for frame_number, frame in enumerate(container.decode(container.streams.video[0])):
                while next_index < len(indices) and indices[next_index] == frame_number:
                    frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
                    next_index += 1
                if next_index == len(indices):
                    break
        if len(frames) != len(indices):
            raise ValueError(
                f"Bernini-R requested {len(indices)} indexed frames from {video_path}, but decoded {len(frames)}."
            )
        return frames

    @classmethod
    def _preprocess_reference_image(cls, path: Path, *, max_condition_size: int) -> mx.array:
        image = ImageUtil.load_image(path)
        width, height = cls._condition_dimensions(
            width=image.width,
            height=image.height,
            max_size=max_condition_size,
        )
        if image.size != (width, height):
            image = image.resize((width, height), resample=Image.Resampling.BICUBIC)
        image_np = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return mx.transpose(mx.array(image_np)[None, None, ...], (0, 4, 1, 2, 3))

    @staticmethod
    def _condition_dimensions(
        *,
        width: int,
        height: int,
        max_size: int,
        min_size: int = 1,
        stride: int = 16,
    ) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            raise ValueError("Bernini condition dimensions must be positive.")
        scale = min(max_size / max(width, height), 1.0)
        scale = max(scale, min_size / min(width, height))
        new_width = max(stride, int(round(round(width * scale) / stride) * stride))
        new_height = max(stride, int(round(round(height * scale) / stride) * stride))
        if max(new_width, new_height) > max_size:
            scale = max_size / max(new_width, new_height)
            new_width = max(stride, int(round(round(new_width * scale) / stride) * stride))
            new_height = max(stride, int(round(round(new_height * scale) / stride) * stride))
        return new_width, new_height

    @staticmethod
    def _smart_video_indices(
        *,
        total_frames: int,
        video_fps: float,
        fps: float,
        frame_factor: int | None = None,
        min_frames: int | None = None,
        max_frames: int | None = None,
        add_one: bool = False,
    ) -> list[int]:
        if total_frames <= 0 or video_fps <= 0 or fps <= 0:
            raise ValueError("Bernini video sampling requires positive frame counts and frame rates.")
        source_total_frames = total_frames
        nframes = total_frames / video_fps * fps
        if frame_factor is not None:
            nframes = math.floor(nframes / frame_factor) * frame_factor + int(add_one)
            nframes = max(nframes, frame_factor + int(add_one))
            if video_fps == fps:
                total_frames = math.floor(total_frames / frame_factor) * frame_factor + int(add_one)
        else:
            nframes = int(nframes + int(add_one))
        nframes = int(nframes)
        indices = BerniniRenderer._torch_float32_linspace_round(
            start=0,
            stop=total_frames - 1,
            steps=nframes,
        )
        if min_frames is not None:
            if frame_factor is not None:
                min_frames = math.ceil(min_frames / frame_factor) * frame_factor
            nframes = max(min_frames + int(add_one), nframes)
        while len(indices) < nframes:
            indices.append(indices[-1])
        if max_frames is not None:
            if frame_factor is not None:
                max_frames = math.floor(max_frames / frame_factor) * frame_factor
            nframes = min(max_frames + int(add_one), nframes)
        return [max(0, min(int(index), source_total_frames - 1)) for index in indices[:nframes]]

    @staticmethod
    def _torch_float32_linspace_round(*, start: int, stop: int, steps: int) -> list[int]:
        values = BerniniRenderer._torch_float32_linspace(start=float(start), stop=float(stop), steps=steps)
        return np.rint(np.asarray(values, dtype=np.float32)).astype(np.int64).tolist()

    @staticmethod
    def _torch_float32_linspace(*, start: float, stop: float, steps: int) -> list[float]:
        if steps <= 0:
            return []
        if steps == 1:
            return [float(np.float32(start))]
        start32 = np.float32(start)
        stop32 = np.float32(stop)
        step32 = np.float32((np.float64(stop) - np.float64(start)) / np.float64(steps - 1))
        positions = np.arange(steps, dtype=np.float32)
        left = start32 + positions * step32
        right = stop32 - (np.float32(steps - 1) - positions) * step32
        values = np.where(positions < steps // 2, left, right).astype(np.float32)
        return [float(value) for value in values]

    def _validated_bernini_spatial_size(self, *, height: int, width: int) -> tuple[int, int]:
        patch_size = self.transformer.patch_size
        multiple_height = self.vae.spatial_scale * patch_size[1]
        multiple_width = self.vae.spatial_scale * patch_size[2]
        if height <= 0 or width <= 0:
            raise ValueError(f"Bernini-R height and width must be at least ({multiple_height}, {multiple_width})px.")
        resolved_height = max(multiple_height, int(round(height / multiple_height)) * multiple_height)
        resolved_width = max(multiple_width, int(round(width / multiple_width)) * multiple_width)
        if (resolved_height, resolved_width) != (height, width):
            print(
                "Bernini-R rounds target dimensions to the nearest packed-latent multiple: "
                f"({height}, {width}) -> ({resolved_height}, {resolved_width})."
            )
        return resolved_height, resolved_width

    @classmethod
    def _validate_supported_frame_step_domain(cls, *, num_frames: int, num_inference_steps: int) -> None:
        if num_frames >= cls.MIN_PROVEN_CONVERGED_FRAMES or num_inference_steps <= cls.MAX_PROVEN_SHORT_DEBUG_STEPS:
            return
        raise ValueError(
            f"Bernini-R resolved to {num_frames} effective frames for {num_inference_steps} inference steps. "
            f"Runs above {cls.MAX_PROVEN_SHORT_DEBUG_STEPS} steps currently require at least "
            f"{cls.MIN_PROVEN_CONVERGED_FRAMES} effective frames; shorter outputs are supported only for "
            f"smoke/debug runs of at most {cls.MAX_PROVEN_SHORT_DEBUG_STEPS} steps. For video-conditioned "
            "runs, source duration and output fps can resolve fewer frames than requested; use a longer "
            "source or higher output fps, and request at least 25 frames."
        )

    def _validate_bernini_request(
        self,
        *,
        num_inference_steps: int | None,
        fps: int | float,
        guidance_2: float | None | object,
        denoising_step_list: list[int] | None,
        image_path: Path | str | None,
        last_image_path: Path | str | None,
        context_image_paths: list[Path | str] | None,
        context_noise: float | None,
        svi_anchor_image_path: Path | str | None,
        svi_motion_latent_path: Path | str | None,
        svi_motion_latent_count: int,
        svi_motion_latent_export_path: Path | str | None,
        video_path: Path | None,
        reference_video_paths: list[Path],
        video_strength: float | None,
        video_mask_path: Path | str | None,
        reference_image_paths: list[Path],
        release_inactive_denoiser: bool | None,
        compile_transformer: bool,
        max_condition_size: int,
        max_sequence_length: int,
        tensor_health_check_interval: int | None,
        guidance_mode: str,
    ) -> None:
        if guidance_2 is not None:
            raise ValueError("guidance_2 is not supported on Bernini-R; it has one renderer transformer.")
        if denoising_step_list is not None:
            raise ValueError("Bernini-R does not support denoising_step_list; use num_inference_steps.")
        if last_image_path is not None:
            raise ValueError("Bernini-R does not support last_image_path bracket conditioning.")
        if context_image_paths:
            raise ValueError("Bernini-R does not support Wan context_image_paths; use reference_image_paths.")
        if context_noise is not None:
            raise ValueError("Bernini-R does not support context_noise.")
        if (
            svi_anchor_image_path is not None
            or svi_motion_latent_path is not None
            or svi_motion_latent_export_path is not None
            or svi_motion_latent_count != 1
        ):
            raise ValueError("Bernini-R does not support SVI conditioning or motion-latent handover.")
        if video_strength is not None:
            raise ValueError(
                "video_strength is not supported on Bernini-R: the source video is a packed conditioning "
                "segment, never an SDEdit warm start. Use source_guidance instead."
            )
        if video_mask_path is not None:
            raise ValueError("Bernini-R renderer-only integration does not support video_mask_path.")
        if release_inactive_denoiser:
            raise ValueError("Bernini-R has one renderer transformer; release_inactive_denoiser is not applicable.")
        if compile_transformer:
            raise ValueError("compile_transformer is not validated for heterogeneous Bernini packed branches.")
        if guidance_mode not in self.SUPPORTED_GUIDANCE_MODES:
            raise ValueError(
                f"Unknown Bernini-R guidance mode {guidance_mode!r}. "
                f"Expected one of: {', '.join(sorted(self.SUPPORTED_GUIDANCE_MODES))}."
            )
        has_image = image_path is not None
        has_video = video_path is not None
        has_reference_video = bool(reference_video_paths)
        has_reference_image = bool(reference_image_paths)
        if has_image and (has_video or has_reference_video or has_reference_image):
            raise ValueError(
                "Bernini-R source-image editing currently accepts exactly one source image and no other "
                "conditioning sources."
            )
        if has_reference_video and not has_video:
            raise ValueError("Bernini-R reference video conditioning currently requires a primary source video.")
        if guidance_mode == "rv2v" and not (has_video and (has_reference_video or has_reference_image)):
            raise ValueError("Bernini-R rv2v guidance requires a source video and at least one extra reference.")
        if guidance_mode == "r2v_apg" and (has_video or has_reference_video or has_image or not has_reference_image):
            raise ValueError("Bernini-R r2v_apg requires one or more reference images and no source video.")
        if guidance_mode in {"v2v", "v2v_apg"} and not (
            has_video or has_reference_video or has_reference_image or has_image
        ):
            raise ValueError("Bernini-R v2v guidance requires at least one conditioning source.")
        if guidance_mode in {"t2v", "t2v_apg"} and (has_video or has_reference_video or has_reference_image or has_image):
            raise ValueError("Bernini-R t2v guidance does not accept source videos or references.")
        if image_path is not None and (not Path(image_path).exists() or not Path(image_path).is_file()):
            raise ValueError(f"Bernini-R source image does not exist or is not a file: {image_path}")
        if video_path is not None and (not video_path.exists() or not video_path.is_file()):
            raise ValueError(f"Bernini-R source video does not exist or is not a file: {video_path}")
        for path in reference_video_paths:
            if not path.exists() or not path.is_file():
                raise ValueError(f"Bernini-R reference video does not exist or is not a file: {path}")
        for path in reference_image_paths:
            if not path.exists() or not path.is_file():
                raise ValueError(f"Bernini-R reference image does not exist or is not a file: {path}")
        if num_inference_steps is not None and num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be greater than zero.")
        if fps <= 0:
            raise ValueError("Bernini-R fps must be greater than zero.")
        if max_condition_size < 16:
            raise ValueError("max_condition_size must be at least 16 pixels.")
        if max_condition_size > 1280:
            raise ValueError("max_condition_size must not exceed the official Bernini-R 1280px proven bound.")
        if max_condition_size % 16 != 0:
            raise ValueError("max_condition_size must be a multiple of 16 pixels for Bernini packed latents.")
        if len(reference_image_paths) > 8:
            raise ValueError("Bernini-R currently supports at most 8 reference images, the largest official case.")
        if len(reference_video_paths) > 1:
            raise ValueError("Bernini-R currently supports at most 1 extra reference video, matching ads2v.")
        if max_sequence_length != 512:
            raise ValueError("Bernini-R requires max_sequence_length=512, matching the official renderer.")
        self._validate_tensor_health_check_interval(tensor_health_check_interval)
        if not bool(self.model_config.transformer_overrides.get("supports_bernini_renderer", False)):
            raise ValueError("BerniniRenderer was constructed with a non-Bernini model config.")
        if self.transformer is None:
            raise ValueError(
                "Bernini-R denoiser has been released. Construct a fresh BerniniRenderer for another generation."
            )
        if self.transformer_2 is not None:
            raise ValueError("Bernini-R 1.3B must have exactly one renderer transformer.")
        if not hasattr(self.transformer, "forward_packed"):
            raise ValueError("Bernini-R requires a transformer with the packed heterogeneous-segment API.")
        if getattr(self.transformer, "vace_layers", None) is not None:
            raise ValueError("Bernini-R packed source conditioning is not the Wan VACE control path.")

    def _resolved_finite_float(
        self,
        value: float | None,
        *,
        config_key: str,
        fallback: float,
        label: str,
    ) -> float:
        resolved = float(value if value is not None else self._wan_config(config_key, fallback))
        if not math.isfinite(resolved):
            raise ValueError(f"{label} must be finite, got {resolved!r}.")
        return resolved

    def _resolved_task_or_config_float(
        self,
        value: float | None,
        *,
        task_default: float | None,
        config_key: str,
        fallback: float,
        label: str,
    ) -> float:
        if value is not None:
            resolved = float(value)
        elif task_default is not None:
            resolved = float(task_default)
        else:
            resolved = float(self._wan_config(config_key, fallback))
        if not math.isfinite(resolved):
            raise ValueError(f"{label} must be finite, got {resolved!r}.")
        return resolved

    @staticmethod
    def _source_ids(
        count: int,
        max_trained_source_id: int = 5,
        interpolate_source_ids: bool = True,
    ) -> list[float]:
        if count <= 0:
            return []
        if max_trained_source_id <= 0:
            raise ValueError("max_trained_source_id must be greater than zero.")
        if interpolate_source_ids and count > max_trained_source_id:
            return BerniniRenderer._torch_float32_linspace(
                start=1.0,
                stop=float(max_trained_source_id),
                steps=count,
            )
        return [float(index) for index in range(1, count + 1)]

    def _configured_source_ids(self, count: int) -> list[float]:
        return self._source_ids(
            count,
            max_trained_source_id=int(self._wan_config("max_trained_source_id", 5)),
            interpolate_source_ids=bool(self._wan_config("interpolate_source_ids", True)),
        )

    @classmethod
    def _guidance_mode(
        cls,
        *,
        has_video: bool,
        has_image: bool,
        num_reference_images: int,
        num_reference_videos: int = 0,
    ) -> str:
        if has_video and num_reference_videos and not num_reference_images:
            return "rv2v"
        if has_video and (num_reference_images or num_reference_videos):
            return "rv2v"
        if has_video:
            return "v2v_apg"
        if has_image:
            return "v2v"
        if num_reference_images:
            return "r2v_apg"
        return "t2v_apg"

    @classmethod
    def _resolved_guidance_mode(
        cls,
        *,
        guidance_mode: str | None,
        task_type: str | None,
        has_video: bool,
        has_image: bool,
        num_reference_images: int,
        num_reference_videos: int,
    ) -> str:
        resolved = guidance_mode or (
            cls.GUIDANCE_MODE_BY_TASK[task_type]
            if task_type is not None
            else cls._guidance_mode(
                has_video=has_video,
                has_image=has_image,
                num_reference_images=num_reference_images,
                num_reference_videos=num_reference_videos,
            )
        )
        if resolved not in cls.SUPPORTED_GUIDANCE_MODES:
            raise ValueError(
                f"Unsupported Bernini-R guidance mode {resolved!r}. "
                f"Expected one of: {', '.join(sorted(cls.SUPPORTED_GUIDANCE_MODES))}."
            )
        if task_type is not None and resolved not in cls.ALLOWED_GUIDANCE_MODES_BY_TASK[task_type]:
            raise ValueError(
                f"Bernini-R task_type={task_type!r} does not support guidance_mode={resolved!r}. "
                f"Allowed modes: {sorted(cls.ALLOWED_GUIDANCE_MODES_BY_TASK[task_type])}."
            )
        return resolved

    @classmethod
    def _resolved_task_type(
        cls,
        *,
        task_type: str | None,
        guidance_mode: str,
        has_video: bool,
        has_image: bool,
        num_reference_images: int,
        num_reference_videos: int,
    ) -> str:
        if task_type is not None:
            if task_type not in cls.SYSTEM_PROMPTS:
                raise ValueError(
                    f"Unsupported Bernini-R task_type {task_type!r}. "
                    f"Expected one of: {', '.join(sorted(cls.SYSTEM_PROMPTS))}."
                )
            if guidance_mode not in cls.ALLOWED_GUIDANCE_MODES_BY_TASK[task_type]:
                raise ValueError(
                    f"Bernini-R task_type={task_type!r} does not support guidance_mode={guidance_mode!r}. "
                    f"Allowed modes: {sorted(cls.ALLOWED_GUIDANCE_MODES_BY_TASK[task_type])}."
                )
            return task_type
        if has_image:
            return "i2i"
        if has_video and num_reference_videos and not num_reference_images:
            return "ads2v"
        if has_video and (num_reference_images or num_reference_videos):
            return "rv2v"
        if has_video:
            return "v2v"
        if num_reference_images:
            return "r2v"
        return "t2v"

    @staticmethod
    def _public_task(*, task_type: str, has_video: bool) -> str:
        if task_type == "t2i":
            return "text-to-image"
        if task_type == "i2i":
            return "image-to-image"
        if has_video or task_type in {"v2v", "mv2v", "rv2v", "ads2v"}:
            return "video-to-video"
        return "text-to-video"

    def _promote_vae_to_float32(self) -> None:
        self.vae.update(
            tree_map(
                lambda value: value.astype(mx.float32) if isinstance(value, mx.array) else value,
                self.vae.parameters(),
            )
        )

    @staticmethod
    def _guidance_parameter_activity(guidance_mode: str) -> tuple[list[str], list[str]]:
        parameters = [
            "text_guidance",
            "reference_guidance",
            "source_guidance",
            "apg_eta",
            "apg_norm_threshold",
            "apg_momentum",
        ]
        active_by_mode = {
            "r2v_apg": {
                "text_guidance",
                "reference_guidance",
                "apg_eta",
                "apg_norm_threshold",
                "apg_momentum",
            },
            "rv2v": {"text_guidance", "reference_guidance", "source_guidance"},
            "v2v": {"text_guidance"},
            "v2v_apg": {"text_guidance", "apg_eta", "apg_norm_threshold", "apg_momentum"},
            "t2v": {"text_guidance"},
            "t2v_apg": {"text_guidance", "apg_eta", "apg_norm_threshold", "apg_momentum"},
        }
        if guidance_mode not in active_by_mode:
            raise ValueError(f"Unknown Bernini-R guidance mode: {guidance_mode!r}.")
        active = active_by_mode[guidance_mode]
        return (
            [parameter for parameter in parameters if parameter in active],
            [parameter for parameter in parameters if parameter not in active],
        )

    @classmethod
    def _resolved_system_prompt(
        cls,
        *,
        guidance_mode: str,
        system_prompt: str | None,
        task_type: str | None = None,
    ) -> str:
        if system_prompt:
            return system_prompt
        if task_type is not None:
            return cls.SYSTEM_PROMPTS[task_type]
        default_task_type = {
            "rv2v": "rv2v",
            "v2v": "v2v",
            "r2v_apg": "r2v",
            "v2v_apg": "v2v",
            "t2v": "t2v",
            "t2v_apg": "t2v",
        }[guidance_mode]
        return cls.SYSTEM_PROMPTS[default_task_type]

    def _effective_prompt(self, *, system_prompt: str, prompt: str) -> str:
        cleaned = self._prompt_clean(prompt)
        if not system_prompt:
            return cleaned
        if not cleaned:
            return system_prompt
        return f"{system_prompt}{cleaned}"

    def _trim_bernini_text_embeddings(
        self,
        *,
        prompt_embeds: mx.array,
        negative_prompt_embeds: mx.array | None,
        prompt: str,
        negative_prompt: str | None,
        max_sequence_length: int,
    ) -> tuple[mx.array, mx.array | None]:
        tokenizers = getattr(self, "tokenizers", None)
        if not tokenizers or "wan" not in tokenizers:
            return prompt_embeds, negative_prompt_embeds
        text_inputs = self._tokenize_prompts(
            cleaned=[prompt, negative_prompt or ""],
            max_sequence_length=max_sequence_length,
        )
        attention_mask = np.asarray(text_inputs["attention_mask"])
        if attention_mask.ndim != 2 or attention_mask.shape[0] != 2:
            return prompt_embeds, negative_prompt_embeds
        seq_lens = np.clip(attention_mask.sum(axis=1).astype(int), 1, max_sequence_length)
        trimmed_prompt = prompt_embeds[:, : min(int(seq_lens[0]), int(prompt_embeds.shape[1])), :]
        if negative_prompt_embeds is None:
            return trimmed_prompt, None
        trimmed_negative = negative_prompt_embeds[
            :,
            : min(int(seq_lens[1]), int(negative_prompt_embeds.shape[1])),
            :,
        ]
        return trimmed_prompt, trimmed_negative

    @staticmethod
    def _apg_sigma(*, scheduler, fallback_index: int) -> mx.array:
        index = 0 if getattr(scheduler, "step_index", None) is None else int(scheduler.step_index)
        if not hasattr(scheduler, "step_index"):
            index = fallback_index
        sigma = scheduler.sigmas[index]
        sigma_value = float(np.asarray(sigma, dtype=np.float32))
        if not math.isfinite(sigma_value) or sigma_value <= 0:
            raise ValueError(f"Bernini APG requires a positive finite scheduler sigma, got {sigma_value!r}.")
        return mx.array(sigma_value, dtype=mx.float32)

    @staticmethod
    def _combined_condition_segments(
        *,
        video_condition: mx.array | None,
        reference_video_conditions: list[mx.array],
        reference_conditions: list[mx.array],
    ) -> list[mx.array]:
        return ([] if video_condition is None else [video_condition]) + reference_video_conditions + reference_conditions

    def _cfg_noise_prediction(
        self,
        *,
        target: mx.array,
        condition_segments: list[mx.array],
        prompt_embeds: mx.array,
        negative_prompt_embeds: mx.array,
        text_guidance: float,
        **branch_kwargs,
    ) -> mx.array:
        source_ids = self._configured_source_ids(len(condition_segments))
        uncond = self._predict_branch(
            role="cfg-uncond",
            target=target,
            condition_segments=condition_segments,
            source_ids=source_ids,
            text_embeds=negative_prompt_embeds,
            **branch_kwargs,
        )
        cond = self._predict_branch(
            role="cfg-cond",
            target=target,
            condition_segments=condition_segments,
            source_ids=source_ids,
            text_embeds=prompt_embeds,
            **branch_kwargs,
        )
        result = uncond + text_guidance * (cond - uncond)
        mx.eval(result)
        del uncond, cond
        return result

    def _single_condition_apg_noise_prediction(
        self,
        *,
        target: mx.array,
        condition_segments: list[mx.array],
        prompt_embeds: mx.array,
        negative_prompt_embeds: mx.array,
        text_guidance: float,
        sigma: mx.array,
        buffer: _MomentumBuffer | None,
        eta: float,
        norm_threshold: float,
        **branch_kwargs,
    ) -> mx.array:
        source_ids = self._configured_source_ids(len(condition_segments))
        prepared = self._prepare_packed_branch(
            target=target,
            condition_segments=condition_segments,
            source_ids=source_ids,
        )
        uncond_v = self._predict_prepacked_branch(
            role="apg-uncond",
            prepared=prepared,
            text_embeds=negative_prompt_embeds,
            **branch_kwargs,
        )
        cond_v = self._predict_prepacked_branch(
            role="apg-cond",
            prepared=prepared,
            text_embeds=prompt_embeds,
            **branch_kwargs,
        )
        noisy_target = target.astype(mx.float32)
        sigma_value = sigma.astype(mx.float32)
        uncond_sample = noisy_target - sigma_value * uncond_v
        cond_sample = noisy_target - sigma_value * cond_v
        guided_sample = uncond_sample + text_guidance * self._normalize_diff(
            cond_sample - uncond_sample,
            cond_sample,
            momentum_buffer=buffer,
            eta=eta,
            norm_threshold=norm_threshold,
        )
        noise_pred = (noisy_target - guided_sample) / sigma_value
        mx.eval(noise_pred)
        del uncond_v, cond_v
        return noise_pred

    @staticmethod
    def _condition_shapes(conditions: list[mx.array]) -> list[list[int]]:
        return [[int(value) for value in condition.shape] for condition in conditions]

    @staticmethod
    def _bernini_extra_metadata(
        *,
        guidance_mode: str,
        reference_image_paths: list[Path],
        reference_video_paths: list[Path],
        text_guidance: float,
        reference_guidance: float,
        source_guidance: float,
        apg_eta: float,
        apg_norm_threshold: float,
        apg_momentum: float,
        max_condition_size: int,
        system_prompt: str,
        effective_prompt: str,
        unipc_flow_sigma_schedule: str,
        source_ids: list[float],
        condition_shapes: list[list[int]],
        condition_metadata: dict,
        component_source_provenance: dict | None = None,
        factored_component_sources: bool = False,
        vae_low_memory_policy_active: bool = False,
        clear_cache_each_transformer_block: bool = False,
        release_denoisers_before_decode: bool = False,
        task_type: str | None = None,
    ) -> dict:
        active_parameters, inactive_parameters = BerniniRenderer._guidance_parameter_activity(guidance_mode)
        low_ram_effective = (
            vae_low_memory_policy_active or clear_cache_each_transformer_block or release_denoisers_before_decode
        )
        return {
            "bernini_guidance_mode": guidance_mode,
            "bernini_renderer_only": True,
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "runtime_platform": platform.platform(),
            "numpy_version": np.__version__,
            "python_executable": sys.executable,
            "text_encoder_precision_policy_id": WanTextEncoderLoader.BERNINI_PRECISION_POLICY_ID,
            "text_encoder_encode_policy_id": WanTextEncoderLoader.ENCODE_POLICY_ID,
            "transformer_precision_policy_id": WanWeightDefinition.BERNINI_TRANSFORMER_PRECISION_POLICY_ID,
            "transformer_default_weight_precision": "bfloat16",
            "transformer_fp32_weight_keys": list(WanWeightDefinition.BERNINI_TRANSFORMER_FP32_KEYS),
            "transformer_fp32_weight_prefixes": list(WanWeightDefinition.BERNINI_TRANSFORMER_FP32_PREFIXES),
            "transformer_fp32_weight_fragments": list(WanWeightDefinition.BERNINI_TRANSFORMER_FP32_FRAGMENTS),
            "source_conditioning": "independent-vae-packed-segments",
            "source_video_warm_start": False,
            "branch_evaluation": "sequential",
            "text_guidance": float(text_guidance),
            "reference_guidance": float(reference_guidance),
            "source_guidance": float(source_guidance),
            "apg_eta": float(apg_eta),
            "apg_norm_threshold": float(apg_norm_threshold),
            "apg_momentum": float(apg_momentum),
            "active_guidance_parameters": active_parameters,
            "inactive_guidance_parameters": inactive_parameters,
            "apg_reduction_axes": [1, 3, 4],
            "apg_accumulator_precision": "float64-projection",
            "apg_reference_accumulator_precision": "float64",
            "system_prompt": system_prompt,
            "effective_prompt": effective_prompt,
            "bernini_task_type": task_type,
            "unipc_flow_sigma_schedule": unipc_flow_sigma_schedule,
            "reference_image_paths": [str(path) for path in reference_image_paths],
            "reference_image_count": len(reference_image_paths),
            "reference_video_paths": [str(path) for path in reference_video_paths],
            "reference_video_count": len(reference_video_paths),
            "condition_source_ids": [float(source_id) for source_id in source_ids],
            "condition_latent_shapes": condition_shapes,
            "max_condition_size": int(max_condition_size),
            "condition_resize_backend": "pillow-bicubic",
            "component_source_provenance": dict(component_source_provenance or {}),
            "factored_component_sources": bool(factored_component_sources),
            "low_ram": bool(low_ram_effective),
            "vae_low_memory_policy_active": bool(vae_low_memory_policy_active),
            "clear_cache_each_transformer_block": bool(clear_cache_each_transformer_block),
            "release_denoisers_before_decode": bool(release_denoisers_before_decode),
            "vae_feature_cache_policy_id": (
                Wan2_2_VAE.COMPACT_FEATURE_CACHE_POLICY_ID
                if vae_low_memory_policy_active
                else "wan-default-feature-cache-v1"
            ),
            "vae_encode_cache_materialization": (
                "eager-contiguous-per-slice" if vae_low_memory_policy_active else "default"
            ),
            "vae_decode_cache_materialization": (
                "eager-contiguous-per-slice" if vae_low_memory_policy_active else "default"
            ),
            "vae_spatial_tiling": bool(vae_low_memory_policy_active),
            "vae_spatial_tiling_policy_id": (
                Wan2_2_VAE.SPATIAL_TILING_POLICY_ID if vae_low_memory_policy_active else None
            ),
            "wan_decode_mode": (
                Wan2_2_VAE.TILED_DECODE_MODE if vae_low_memory_policy_active else "streamed_vae_slices"
            ),
            **condition_metadata,
        }
