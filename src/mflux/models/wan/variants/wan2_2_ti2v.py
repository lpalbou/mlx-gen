import gc
import html
import io
import math
import re
import shutil
import time
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from PIL import Image

from mflux.callbacks import ProgressCallback, ProgressEvent
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.wan.latent_creator import WanTimestepPolicy
from mflux.models.wan.model.wan_transformer import WanBlockHealthContext, WanTransformer
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.prompt_embed_store import WanPromptEmbedStore
from mflux.models.wan.scheduler import WanEulerScheduler, WanUniPCMultistepScheduler
from mflux.models.wan.variants.wan_video_request import WanVideoRequest
from mflux.models.wan.wan_initializer import WanInitializer
from mflux.models.wan.weights import WanWeightDefinition
from mflux.utils.exceptions import ModelConfigError
from mflux.utils.generated_video import GeneratedVideo
from mflux.utils.image_util import ImageUtil
from mflux.utils.mask_util import MaskUtil
from mflux.utils.runtime_memory import RuntimeMemory
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
        keep_text_encoder_resident: bool = False,
        prompt_embed_disk_cache: bool = True,
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
        # Prompt-encoding cost controls (0086). The UMT5 text encoder is an
        # ~11 GB torch load per encode: hosts that chain generations with new
        # prompts (scene-by-scene video) can keep it resident, and identical
        # re-encodes are served from an exact disk cache.
        self._keep_text_encoder_resident = keep_text_encoder_resident
        self._resident_text_encoder = None
        self._prompt_embed_store = WanPromptEmbedStore(enabled=prompt_embed_disk_cache)
        self._prompt_embed_fingerprint = None
        # Lifetime count of per-item high-noise expert reloads (0089 e4);
        # generate_video diffs it per run for truthful metadata.
        self._high_noise_reload_count = 0

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
        video_path: Path | str | None = None,
        video_strength: float | None = None,
        video_mask_path: Path | str | None = None,
        max_sequence_length: int = 512,
        progress_callback: ProgressCallback | None = None,
        release_inactive_denoiser: bool | None = None,
        release_denoisers_before_decode: bool = False,
        clear_cache_each_step: bool = False,
        clear_cache_each_transformer_block: bool = False,
        tensor_health_check_interval: int | None = None,
        compile_transformer: bool = False,
    ) -> GeneratedVideo:
        start_time = time.time()
        request = WanVideoRequest.resolve(
            self,
            guidance=guidance,
            guidance_2=guidance_2,
            guidance_2_unset_sentinel=_GUIDANCE_2_UNSET,
            height=height,
            width=width,
            num_frames=num_frames,
            image_path=image_path,
            video_path=video_path,
            video_strength=video_strength,
            video_mask_path=video_mask_path,
            flow_shift=flow_shift,
            solver=solver,
            negative_prompt=negative_prompt,
            tensor_health_check_interval=tensor_health_check_interval,
        )
        # The model owns the release default (0089 e4): None = auto, so the
        # Python API and the CLI resolve identically; True/False is user intent.
        release_inactive_denoiser = self._resolve_release_inactive_denoiser(release_inactive_denoiser)
        high_noise_reloads_before = getattr(self, "_high_noise_reload_count", 0)
        health_check_interval = request.health_check_interval
        is_image_to_video = request.is_image_to_video
        is_video_to_video = request.is_video_to_video
        task = request.task
        height, width = request.height, request.width
        spatial_metadata = dict(request.spatial_metadata)
        num_frames = request.num_frames
        video_strength = request.video_strength
        video_mask = request.video_mask
        guidance, guidance_2 = request.guidance, request.guidance_2
        flow_shift = request.flow_shift
        solver = request.solver
        negative_prompt = request.negative_prompt
        batch_size = request.batch_size
        scheduler = self._create_scheduler(flow_shift=flow_shift, solver=solver)
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps.tolist()
        if is_video_to_video:
            timesteps = self._video_to_video_timesteps(
                scheduler=scheduler,
                num_inference_steps=num_inference_steps,
                strength=video_strength,
            )
        effective_steps = len(timesteps)
        boundary_timestep = self._boundary_timestep(scheduler)
        v2v_high_noise_stage_skipped = is_video_to_video and self._skips_high_noise_stage(
            timesteps=timesteps,
            boundary_timestep=boundary_timestep,
        )
        if v2v_high_noise_stage_skipped:
            print(
                f"⚠️  Wan video-to-video with video_strength {video_strength} starts below the two-transformer "
                f"boundary: the high-noise transformer and guidance={guidance} are unused; only guidance_2="
                f"{guidance_2} applies. Increase video_strength to engage the high-noise stage."
            )
        progress_registry = getattr(self, "callbacks", None)
        self._emit_progress(
            progress_callback,
            phase="start",
            frame=0,
            total_frames=num_frames,
            step=0,
            total_steps=effective_steps,
            task=task,
            registry=progress_registry,
        )

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

        v2v_source_latents = None
        v2v_noise = None
        if is_video_to_video:
            if fps is None or fps <= 0:
                raise ValueError("Wan video-to-video requires a positive fps.")
            latents, v2v_source_latents, v2v_noise, source_video_metadata = self._prepare_video_to_video_latents(
                seed=seed,
                batch_size=batch_size,
                scheduler=scheduler,
                height=height,
                width=width,
                num_frames=num_frames,
                video_path=video_path,
                timesteps=timesteps,
                fps=fps,
            )
            spatial_metadata.update(source_video_metadata)
            self._warn_video_to_video_source_handling(
                source_metadata=spatial_metadata,
                num_frames=num_frames,
                height=height,
                width=width,
                fps=fps,
            )
            if video_mask is None:
                v2v_source_latents = None
                v2v_noise = None
        else:
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

        # Opt-in compiled denoisers (0090 d12): per-expert compiled callables,
        # ~2-6%/step, output within ~5e-4 of eager (NON-bitwise: compiled kernel
        # fusion differs) - never a silent default. Eager modes that need
        # per-block introspection or mid-graph cache flushes stay eager, with
        # one printed notice documenting the choice.
        compiled_denoisers = self._build_compiled_denoisers(
            compile_transformer=compile_transformer,
            health_check_interval=health_check_interval,
            clear_cache_each_transformer_block=clear_cache_each_transformer_block,
        )
        compile_transformer_active = bool(compiled_denoisers)

        total_steps = effective_steps
        high_noise_denoiser_released = False
        for step_index, timestep in enumerate(timesteps):
            step_number = step_index + 1
            should_check_tensors = TensorHealth.should_check_step(step_number, total_steps, health_check_interval)
            high_noise_denoiser_released = self._maybe_release_high_noise_denoiser(
                timestep=timestep,
                boundary_timestep=boundary_timestep,
                release_inactive_denoiser=release_inactive_denoiser,
                already_released=high_noise_denoiser_released,
                compiled_denoisers=compiled_denoisers,
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
                if is_video_to_video:
                    # Plain Wan V2V follows the scalar-timestep reference path rather than TI2V patch-expanded time ids.
                    expanded_timestep = self._batch_timestep(batch_size=batch_size, timestep=timestep)
                elif self._uses_expanded_timesteps():
                    expanded_timestep = WanTimestepPolicy.expand_for_text_to_video(
                        latent_shape=latents.shape,
                        timestep=timestep,
                        patch_size=current_transformer.patch_size,
                    )
                else:
                    expanded_timestep = self._batch_timestep(batch_size=batch_size, timestep=timestep)
            compiled_denoiser = compiled_denoisers.get(denoiser_name)
            if compile_transformer_active and compiled_denoiser is None:
                # A reloaded expert has no compiled entry: its callable was popped
                # at release (0090 d12), so trace a fresh one against the reloaded
                # module instead of reusing a callable closing over freed weights.
                compiled_denoiser = self._compile_denoiser(current_transformer)
                compiled_denoisers[denoiser_name] = compiled_denoiser
            if compiled_denoiser is not None:
                noise_pred = compiled_denoiser(latent_model_input, expanded_timestep, prompt_embeds)
            else:
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
                if compiled_denoiser is not None:
                    noise_uncond = compiled_denoiser(latent_model_input, expanded_timestep, negative_prompt_embeds)
                else:
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
            if video_mask is not None:
                latents = self._composite_masked_video_state(
                    scheduler=scheduler,
                    latents=latents,
                    video_mask=video_mask,
                    source_latents=v2v_source_latents,
                    noise=v2v_noise,
                )
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
        if video_mask is not None:
            # Final clean composite: preserved regions become exactly the VAE latents of the source.
            latents = video_mask * latents + (1.0 - video_mask) * v2v_source_latents
            mx.eval(latents)
            self._require_tensor_health(latents, phase="final-mask-composite", name="latents")
        del prompt_embeds, negative_prompt_embeds, scheduler, v2v_source_latents, v2v_noise, video_mask
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
        decode_latents = RuntimeMemory.materialize_inference_tree(latents.astype(ModelConfig.precision))
        # Release/reload truth for debugging (0089 e4): recorded only when the
        # behavior actually happened this run, matching the compile_transformer style.
        high_noise_reloads = getattr(self, "_high_noise_reload_count", 0) - high_noise_reloads_before
        extra_metadata = {
            **(LoRALoader.extra_metadata_for_model(self) or {}),
            "lora_target_roles": getattr(self, "lora_target_roles", None) or None,
            **({"released_inactive_denoiser": True} if high_noise_denoiser_released else {}),
            **({"high_noise_reloads": high_noise_reloads} if high_noise_reloads else {}),
        }
        del latents
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

        # Streamed slice decode is the ONLY decode path (0089 e3): the artifact
        # holds latents (~MBs) plus a factory instead of the fully decoded
        # tensor (~650 MB bf16 at 121f@1280x704). Output frames are bitwise
        # identical to the removed full-tensor decode (test-pinned); per-frame
        # finite checks cover decode health when frames materialize at save.
        # Only low-ram runs pay per-slice cache flushes.
        def frame_batches_factory():
            decoded_slices = self.vae.iter_decode_normalized_latent_slices(
                decode_latents,
                clear_cache_each_slice=clear_cache_each_step,
            )
            return VideoUtil.decoded_latent_slices_to_frame_batches(
                decoded_slices,
                batch_size=8,
                total_frames=num_frames,
            )

        video = VideoUtil.to_video_from_frame_batches(
            frame_batches_factory=frame_batches_factory,
            height=height,
            width=width,
            frame_count=num_frames,
            # generation_time must be evaluated here (pre-decode; decode runs at save).
            generation_time=time.time() - start_time,
            **self._to_video_shared_kwargs(
                seed=seed,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                fps=fps,
                guidance=guidance,
                guidance_2=guidance_2,
                flow_shift=flow_shift,
                solver=solver,
                task=task,
                image_path=image_path,
                video_path=video_path,
                negative_prompt=negative_prompt,
                spatial_metadata=spatial_metadata,
                extra_metadata=extra_metadata,
                is_video_to_video=is_video_to_video,
                video_strength=video_strength,
                video_mask_path=video_mask_path,
                effective_steps=effective_steps,
                high_noise_stage_skipped=v2v_high_noise_stage_skipped,
                decode_extras={
                    "wan_decode_mode": "streamed_vae_slices",
                    "generation_time_scope": "pre-save",
                    # Compiled runs are NON-bitwise vs eager (~5e-4); record the mode (0090 d12).
                    **({"compile_transformer": True} if compile_transformer_active else {}),
                },
            ),
        )
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

    def _prepare_video_to_video_latents(
        self,
        *,
        seed: int,
        batch_size: int,
        scheduler,
        height: int,
        width: int,
        num_frames: int,
        video_path: Path | str,
        timesteps: list[int | float],
        fps: int | float,
    ) -> tuple[mx.array, mx.array, mx.array, dict[str, int | float | None]]:
        if not timesteps:
            raise ValueError("Wan video-to-video requires at least one denoise timestep.")
        init_latents, source_metadata = self._encode_video_to_video_latents(
            video_path=video_path,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
        )
        noise = self.prepare_latents(
            seed=seed,
            batch_size=batch_size,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        latent_timestep = self._batch_timestep(batch_size=batch_size, timestep=timesteps[0])
        latents = scheduler.scale_noise(init_latents, latent_timestep, noise).astype(mx.float32)
        mx.eval(latents)
        return latents, init_latents, noise, source_metadata

    def _prepare_video_mask(
        self,
        mask_path: Path | str,
        *,
        height: int,
        width: int,
    ) -> mx.array:
        latent_height = height // self.vae.spatial_scale
        latent_width = width // self.vae.spatial_scale
        mask_values = MaskUtil.load_binary_mask(
            mask_path,
            target_width=latent_width,
            target_height=latent_height,
            resampling=Image.Resampling.BOX,
            alpha_warning_context="Wan video mask",
        )
        # White (>= 0.5) marks the region the model may change; black is preserved from the source.
        mask = mx.array(mask_values)[None, None, None, :, :]
        active_cells = float(mx.sum(mask).item())
        total_cells = float(latent_height * latent_width)
        if active_cells == 0:
            print(
                "⚠️  Wan video mask has no editable region after latent downsampling; output will be a source round-trip."
            )
        elif active_cells == total_cells:
            print("⚠️  Wan video mask covers the full canvas; this is equivalent to plain video-to-video.")
        return mask

    @staticmethod
    def _composite_masked_video_state(
        *,
        scheduler,
        latents: mx.array,
        video_mask: mx.array,
        source_latents: mx.array,
        noise: mx.array,
    ) -> mx.array:
        # After scheduler.step, step_index already points at the level of the returned latents.
        index = scheduler.step_index
        keep = 1.0 - video_mask
        sigma = scheduler.sigmas[index]
        latents = video_mask * latents + keep * (sigma * noise + (1.0 - sigma) * source_latents)
        # The UniPC corrector rebuilds each step from last_sample and the x0 history, so preserved
        # regions must be locked in scheduler state as well, not only in the returned latents.
        last_sample = getattr(scheduler, "last_sample", None)
        if last_sample is not None:
            sigma_prev = scheduler.sigmas[index - 1]
            renoised_prev = sigma_prev * noise + (1.0 - sigma_prev) * source_latents
            scheduler.last_sample = video_mask * last_sample + keep * renoised_prev
        model_outputs = getattr(scheduler, "model_outputs", None)
        if model_outputs and model_outputs[-1] is not None:
            model_outputs[-1] = video_mask * model_outputs[-1] + keep * source_latents
        return latents

    def _encode_video_to_video_latents(
        self,
        *,
        video_path: Path | str,
        height: int,
        width: int,
        num_frames: int,
        fps: int | float,
    ) -> tuple[mx.array, dict[str, int | float | None]]:
        cache_key = ("video-to-video", self._cache_path_key(video_path), height, width, num_frames, float(fps))
        cached = self._cached_tensor(cache_name="video_condition_cache", key=cache_key)
        if cached is not None:
            return cached, self._cached_video_source_metadata(cache_key)
        latents, source_metadata = self._load_video_to_video_latents(
            video_path=video_path,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
        )
        return self._store_video_latent_cache(cache_key=cache_key, latents=latents, source_metadata=source_metadata)

    def _load_video_to_video_latents(
        self,
        *,
        video_path: Path | str,
        height: int,
        width: int,
        num_frames: int,
        fps: int | float,
    ) -> tuple[mx.array, dict[str, int | float | None]]:
        clip = VideoUtil.read_video_clip(video_path, max_frames=num_frames, target_fps=float(fps))
        if clip.clip_frame_count < num_frames:
            needed_seconds = num_frames / float(fps)
            raise ValueError(
                f"Wan video-to-video needs {needed_seconds:.2f}s of source video "
                f"({num_frames} output-timeline frames at {float(fps):.3g} fps), but {video_path} "
                f"only yielded {clip.clip_frame_count} frames."
            )
        # Fill one preallocated buffer instead of stacking per-frame copies to bound transient memory.
        video_np = np.empty((1, num_frames, height, width, 3), dtype=np.float32)
        for index, frame in enumerate(clip.frames[:num_frames]):
            video_np[0, index] = np.asarray(
                ImageUtil.scale_to_dimensions(frame, target_width=width, target_height=height),
                dtype=np.float32,
            )
        video_np /= 255.0
        video_mx = mx.array(video_np)
        del video_np
        video_mx = mx.transpose(video_mx, (0, 4, 1, 2, 3))
        # Keep source-video conditioning in float32 to match the upstream Wan V2V warm-start path.
        video_mx = ImageUtil._normalize(video_mx).astype(mx.float32)
        latents = self.vae.encode_normalized(video_mx).astype(mx.float32)
        mx.eval(latents)
        source_metadata = {
            "source_width": clip.source_width,
            "source_height": clip.source_height,
            "source_video_frame_count": clip.source_frame_count,
            "source_video_duration_seconds": clip.source_duration_seconds,
            "source_video_fps": clip.fps,
            "source_video_audio_present": bool(clip.audio_present),
            "source_video_resampled": clip.sampled_fps is not None,
        }
        return latents, source_metadata

    def _store_video_latent_cache(
        self,
        *,
        cache_key: tuple,
        latents: mx.array,
        source_metadata: dict[str, int | float | None],
    ) -> tuple[mx.array, dict[str, int | float | None]]:
        cache = getattr(self, "video_condition_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "video_condition_cache", cache)
        metadata_cache = getattr(self, "video_condition_metadata_cache", None)
        if metadata_cache is None:
            metadata_cache = {}
            setattr(self, "video_condition_metadata_cache", metadata_cache)
        materialized = RuntimeMemory.materialize_inference_tree(latents)
        cache[cache_key] = materialized
        metadata_cache[cache_key] = dict(source_metadata)
        while len(cache) > 2:
            oldest_key = next(iter(cache))
            if oldest_key == cache_key:
                break
            del cache[oldest_key]
            metadata_cache.pop(oldest_key, None)
        return materialized, dict(source_metadata)

    def _cached_video_source_metadata(self, cache_key: tuple) -> dict[str, int | float | None]:
        metadata_cache = getattr(self, "video_condition_metadata_cache", None) or {}
        return dict(metadata_cache.get(cache_key, {}))

    @staticmethod
    def _warn_video_to_video_source_handling(
        *,
        source_metadata: dict,
        num_frames: int,
        height: int,
        width: int,
        fps: int | float | None,
    ) -> None:
        source_width = source_metadata.get("source_width") or 0
        source_height = source_metadata.get("source_height") or 0
        if source_height > 0 and height > 0:
            source_ratio = source_width / source_height
            target_ratio = width / height
            if target_ratio > 0 and abs(source_ratio - target_ratio) / target_ratio > 0.02:
                print(
                    f"⚠️  Wan video-to-video stretches source frames ({source_width}x{source_height}) "
                    f"to the requested canvas ({width}x{height}). Match the requested aspect ratio to the source "
                    "to avoid distortion."
                )
        source_fps = source_metadata.get("source_video_fps")
        source_duration = source_metadata.get("source_video_duration_seconds")
        used_seconds = num_frames / float(fps) if fps else None
        resampled = bool(source_metadata.get("source_video_resampled"))
        if resampled and source_fps and fps:
            if float(fps) > float(source_fps):
                print(
                    f"⚠️  Wan video-to-video resamples the source from {float(source_fps):.6g} fps up to "
                    f"{float(fps):.3g} fps by duplicating frames; expect reduced motion smoothness."
                )
            else:
                print(
                    f"Wan video-to-video resamples the source from {float(source_fps):.6g} fps to "
                    f"{float(fps):.3g} fps; the output keeps real-time speed."
                )
        if used_seconds is not None and source_duration and source_duration - used_seconds > 0.05:
            print(f"Wan video-to-video uses the first {used_seconds:.2f}s of the {float(source_duration):.2f}s source.")

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=getattr(self, "weight_definition", WanWeightDefinition.for_config(self.model_config)),
        )
        self._copy_runtime_assets(base_path)

    def _get_t5_prompt_embeds(self, prompts: list[str], max_sequence_length: int) -> mx.array:
        cleaned = [self._prompt_clean(prompt) for prompt in prompts]
        cache_key = (tuple(cleaned), max_sequence_length)
        cached = self._cached_tensor(cache_name="prompt_embed_cache", key=cache_key)
        if cached is not None:
            return cached
        # Tokenize with numpy tensors: the tokenized ids feed BOTH the exact
        # disk-cache key and the encoder, and a disk hit must not import torch.
        text_inputs = self._tokenize_prompts(cleaned=cleaned, max_sequence_length=max_sequence_length)
        disk_key = self._prompt_embed_disk_key(text_inputs=text_inputs, max_sequence_length=max_sequence_length)
        embeds = self._prompt_embed_store.load(disk_key) if disk_key is not None else None
        if embeds is not None:
            embeds = embeds.astype(ModelConfig.precision)
        else:
            embeds = self._load_t5_prompt_embeds(text_inputs=text_inputs, max_sequence_length=max_sequence_length)
            if disk_key is not None:
                self._prompt_embed_store.store(disk_key, embeds)
        return self._store_cached_tensor(cache_name="prompt_embed_cache", key=cache_key, value=embeds)

    def _tokenize_prompts(self, *, cleaned: list[str], max_sequence_length: int):
        tokenizer = self.tokenizers["wan"].tokenizer
        return tokenizer(
            cleaned,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="np",
        )

    def _prompt_embed_disk_key(self, *, text_inputs, max_sequence_length: int) -> str | None:
        if not self._prompt_embed_store.enabled:
            return None
        if self._prompt_embed_fingerprint is None:
            self._prompt_embed_fingerprint = WanPromptEmbedStore.compute_text_encoder_fingerprint(
                self.root_path / "text_encoder"
            )
        input_ids = np.ascontiguousarray(text_inputs["input_ids"])
        attention_mask = np.ascontiguousarray(text_inputs["attention_mask"])
        return WanPromptEmbedStore.compute_key(
            encoder_fingerprint=self._prompt_embed_fingerprint,
            input_ids_bytes=str(input_ids.shape).encode("utf-8") + input_ids.tobytes(),
            attention_mask_bytes=str(attention_mask.shape).encode("utf-8") + attention_mask.tobytes(),
            max_sequence_length=max_sequence_length,
            precision=str(ModelConfig.precision),
        )

    def _load_t5_prompt_embeds(self, *, text_inputs, max_sequence_length: int) -> mx.array:
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

        input_ids = torch.from_numpy(np.ascontiguousarray(text_inputs["input_ids"]))
        attention_mask = torch.from_numpy(np.ascontiguousarray(text_inputs["attention_mask"]))

        text_encoder = self._resident_text_encoder
        if text_encoder is None:
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
                input_ids,
                attention_mask,
            ).last_hidden_state
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
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
        if self._keep_text_encoder_resident:
            # Opt-in residency for hosts that chain new-prompt generations:
            # trades ~11 GB resident RAM for skipping a per-prompt reload.
            self._resident_text_encoder = text_encoder
        else:
            del text_encoder
            gc.collect()
        return embeds

    def _encode_first_frame_condition(self, image_path: Path | str | None, height: int, width: int) -> mx.array:
        cache_key = ("first-frame", self._cache_path_key(image_path), height, width)
        cached = self._cached_tensor(cache_name="image_condition_cache", key=cache_key)
        if cached is not None:
            return cached
        condition = self._load_first_frame_condition(image_path=image_path, height=height, width=width)
        return self._store_cached_tensor(cache_name="image_condition_cache", key=cache_key, value=condition)

    def _load_first_frame_condition(self, image_path: Path | str | None, height: int, width: int) -> mx.array:
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
        cache_key = ("video", self._cache_path_key(image_path), height, width, num_frames, batch_size)
        cached = self._cached_tensor(cache_name="image_condition_cache", key=cache_key)
        if cached is not None:
            return cached
        condition = self._load_video_condition(
            image_path=image_path,
            height=height,
            width=width,
            num_frames=num_frames,
            batch_size=batch_size,
        )
        return self._store_cached_tensor(cache_name="image_condition_cache", key=cache_key, value=condition)

    def _load_video_condition(
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

    @staticmethod
    def _cache_path_key(image_path: Path | str | None) -> tuple[str, int, int] | None:
        if image_path is None:
            return None
        resolved = Path(image_path).expanduser().resolve(strict=False)
        # Include mtime and size so overwriting a source file in one session invalidates cached latents.
        try:
            stat = resolved.stat()
            return (str(resolved), stat.st_mtime_ns, stat.st_size)
        except OSError:
            return (str(resolved), 0, 0)

    def _cached_tensor(self, *, cache_name: str, key: tuple) -> mx.array | None:
        cache = getattr(self, cache_name, None)
        if cache is None:
            return None
        return cache.get(key)

    def _store_cached_tensor(self, *, cache_name: str, key: tuple, value: mx.array) -> mx.array:
        cache = getattr(self, cache_name, None)
        if cache is None:
            cache = {}
            setattr(self, cache_name, cache)
        materialized = RuntimeMemory.materialize_inference_tree(value)
        cache[key] = materialized
        if len(cache) > 2:
            oldest_key = next(iter(cache))
            if oldest_key != key:
                del cache[oldest_key]
        return materialized

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
        patch_size = self._reference_denoiser().patch_size
        multiple_h = self.vae.spatial_scale * patch_size[1]
        multiple_w = self.vae.spatial_scale * patch_size[2]
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
        video_path: Path | str | None = None,
    ) -> tuple[int, int]:
        resolved_height, resolved_width, _ = self._resolve_video_spatial_size(
            height=height,
            width=width,
            image_path=image_path,
            video_path=video_path,
        )
        return resolved_height, resolved_width

    def _resolve_video_spatial_size(
        self,
        *,
        height: int,
        width: int,
        image_path: Path | str | None,
        video_path: Path | str | None = None,
    ) -> tuple[int, int, dict[str, int]]:
        if image_path is None and video_path is None:
            resolved_height, resolved_width = self._validated_spatial_size(height=height, width=width)
            return resolved_height, resolved_width, {}
        if image_path is not None:
            source_image = ImageUtil.load_image(image_path)
            source_info = {"height": source_image.height, "width": source_image.width}
        else:
            video_info = VideoUtil.inspect_video(video_path)
            source_info = {"height": video_info.source_height, "width": video_info.source_width}
            if source_info["height"] <= 0 or source_info["width"] <= 0:
                raise ValueError("Wan video-to-video source video must have positive dimensions.")
            resolved_height, resolved_width = self._validated_spatial_size(height=height, width=width)
            return (
                resolved_height,
                resolved_width,
                {
                    "source_width": source_info["width"],
                    "source_height": source_info["height"],
                    "requested_width": width,
                    "requested_height": height,
                },
            )
        resolved_height, resolved_width = self._validated_source_aspect_spatial_size(
            height=height,
            width=width,
            source_height=source_info["height"],
            source_width=source_info["width"],
            source_label="image",
        )
        return (
            resolved_height,
            resolved_width,
            {
                "source_width": source_info["width"],
                "source_height": source_info["height"],
                "requested_width": width,
                "requested_height": height,
            },
        )

    def _validated_source_aspect_spatial_size(
        self,
        *,
        height: int,
        width: int,
        source_height: int,
        source_width: int,
        source_label: str,
    ) -> tuple[int, int]:
        patch_size = self._reference_denoiser().patch_size
        multiple_h = self.vae.spatial_scale * patch_size[1]
        multiple_w = self.vae.spatial_scale * patch_size[2]
        if height <= 0 or width <= 0:
            raise ValueError(f"Wan height and width must be at least ({multiple_h}, {multiple_w})px.")
        if source_height <= 0 or source_width <= 0:
            raise ValueError(f"Wan video-to-video source {source_label} must have positive dimensions.")
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
                f"Wan {source_label}-guided video generation preserves the source aspect ratio. "
                f"Adjusting requested video size ({height}, {width}) with source {source_label} "
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

        # A released high expert (per-item release, 0089 e4) is validated through
        # the resident low expert: both are constructed from identical kwargs.
        reference_denoiser = self._reference_denoiser()
        expected_config = self.model_config.transformer_overrides
        expected_vae_config = expected_config.get("vae_config", {})
        expected_transformer_channels = int(expected_config.get("in_channels", reference_denoiser.in_channels))
        expected_output_channels = int(expected_config.get("out_channels", reference_denoiser.out_channels))
        expected_vae_channels = int(expected_vae_config.get("z_dim", self.vae.z_dim))
        expected_transformer_2 = bool(expected_config.get("has_transformer_2", False))

        if int(reference_denoiser.in_channels) != expected_transformer_channels:
            self._raise_runtime_contract_mismatch(
                "transformer.in_channels",
                actual=int(reference_denoiser.in_channels),
                expected=expected_transformer_channels,
            )
        if int(reference_denoiser.out_channels) != expected_output_channels:
            self._raise_runtime_contract_mismatch(
                "transformer.out_channels",
                actual=int(reference_denoiser.out_channels),
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

        transformer_channels = int(reference_denoiser.in_channels)
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
        # The low-noise expert is never reloadable: release_denoisers_before_decode
        # (both experts gone) and a missing transformer_2 stay fatal. A missing
        # HIGH expert is fine when the reload spec can rebuild it (0089 e4).
        if expected_transformer_2 and self.transformer_2 is None:
            self._raise_denoisers_released()
        if self.transformer is None and not self._can_reload_high_noise_denoiser():
            self._raise_denoisers_released()

    @staticmethod
    def _raise_denoisers_released() -> None:
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
                # Lazy per-item reload (0089 e4): only pay the 14 GB rebuild when
                # the schedule actually enters the high-noise phase (V2V runs
                # that start below the boundary never trigger it).
                self._ensure_high_noise_denoiser()
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

    def _supports_video_to_video(self) -> bool:
        return bool(self._wan_config("supports_video_to_video", False))

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

    @staticmethod
    def _resolve_video_strength(video_strength: float | None) -> float:
        resolved = 0.8 if video_strength is None else float(video_strength)
        if not np.isfinite(resolved) or resolved <= 0 or resolved > 1:
            raise ValueError(f"Wan video_strength must be in (0, 1], got {video_strength!r}.")
        return resolved

    def _resolve_solver(self, solver: str | None) -> str:
        resolved = self._default_solver() if solver is None else str(solver).strip().lower()
        if resolved not in _WAN_SOLVERS:
            raise ValueError(f"Wan solver must be one of {_WAN_SOLVERS}, got {solver!r}.")
        return resolved

    @staticmethod
    def _validate_video_to_video_solver(*, is_video_to_video: bool, solver: str) -> None:
        if is_video_to_video and solver != "unipc":
            raise ValueError("Wan video-to-video currently requires solver='unipc'.")

    def _create_scheduler(self, *, flow_shift: float, solver: str):
        if solver == "unipc":
            return WanUniPCMultistepScheduler(flow_shift=flow_shift)
        if solver == "euler":
            return WanEulerScheduler(flow_shift=flow_shift)
        raise ValueError(f"Wan solver must be one of {_WAN_SOLVERS}, got {solver!r}.")

    @staticmethod
    def _video_to_video_timesteps(
        *,
        scheduler,
        num_inference_steps: int,
        strength: float,
    ) -> list[int | float]:
        init_timestep = min(max(int(num_inference_steps * strength), 1), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(t_start * getattr(scheduler, "order", 1))
        return scheduler.timesteps[t_start * getattr(scheduler, "order", 1) :].tolist()

    @staticmethod
    def _skips_high_noise_stage(
        *,
        timesteps: list[int | float],
        boundary_timestep: float | None,
    ) -> bool:
        if boundary_timestep is None or not timesteps:
            return False
        return timesteps[0] < boundary_timestep

    def _to_video_shared_kwargs(
        self,
        *,
        seed: int,
        prompt: str,
        num_inference_steps: int,
        fps: int | float,
        guidance: float,
        guidance_2: float | None,
        flow_shift: float,
        solver: str,
        task: str,
        image_path: Path | str | None,
        video_path: Path | str | None,
        negative_prompt: str | None,
        spatial_metadata: dict,
        extra_metadata: dict,
        is_video_to_video: bool,
        video_strength: float | None,
        video_mask_path: Path | str | None,
        effective_steps: int,
        high_noise_stage_skipped: bool,
        decode_extras: dict | None = None,
    ) -> dict:
        # Shared artifact kwargs; generation_time and factory wiring stay at the call site.
        # WanVace builds its own to_video call and keeps decode_extras=None there.
        return {
            "fps": fps,
            "model_config": self.model_config,
            "seed": seed,
            "prompt": prompt,
            "steps": num_inference_steps,
            "guidance": guidance,
            "guidance_2": guidance_2,
            "flow_shift": flow_shift,
            "solver": solver,
            "quantization": self.bits,
            "task": task,
            "image_path": image_path,
            "video_path": video_path,
            "negative_prompt": negative_prompt,
            "source_width": spatial_metadata.get("source_width"),
            "source_height": spatial_metadata.get("source_height"),
            "requested_width": spatial_metadata.get("requested_width"),
            "requested_height": spatial_metadata.get("requested_height"),
            "lora_paths": getattr(self, "lora_paths", None),
            "lora_scales": getattr(self, "lora_scales", None),
            "extra_metadata": {
                **extra_metadata,
                **self._video_to_video_extra_metadata(
                    is_video_to_video=is_video_to_video,
                    video_strength=video_strength,
                    video_mask_path=video_mask_path,
                    effective_steps=effective_steps,
                    high_noise_stage_skipped=high_noise_stage_skipped,
                    spatial_metadata=spatial_metadata,
                ),
                **(decode_extras or {}),
            },
        }

    @staticmethod
    def _video_to_video_extra_metadata(
        *,
        is_video_to_video: bool,
        video_strength: float | None,
        video_mask_path: Path | str | None,
        effective_steps: int,
        high_noise_stage_skipped: bool,
        spatial_metadata: dict,
    ) -> dict:
        if not is_video_to_video:
            return {}
        # `steps` in metadata records the requested count so --config-from-metadata replays exactly;
        # the strength-truncated schedule length is preserved separately here.
        extra = {
            "video_strength": video_strength,
            "effective_steps": effective_steps,
            "high_noise_stage_skipped": high_noise_stage_skipped,
        }
        if video_mask_path is not None:
            extra["video_mask_path"] = str(video_mask_path)
        for key in (
            "source_video_frame_count",
            "source_video_duration_seconds",
            "source_video_fps",
            "source_video_audio_present",
            "source_video_resampled",
        ):
            if key in spatial_metadata:
                extra[key] = spatial_metadata[key]
        return extra

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
        compiled_denoisers: dict | None = None,
    ) -> bool:
        if (
            already_released
            or not release_inactive_denoiser
            or boundary_timestep is None
            or timestep >= boundary_timestep
            or self.transformer_2 is None
            # Already absent (released in a previous batch item and never needed
            # this run): nothing was released HERE, so metadata stays truthful.
            or self.transformer is None
        ):
            return already_released
        # The compiled callable closes over the transformer; drop it first so
        # the release below actually frees the weights (0090 d12).
        if compiled_denoisers:
            compiled_denoisers.pop("high", None)
        self._release_high_noise_denoiser()
        return True

    def _build_compiled_denoisers(
        self,
        *,
        compile_transformer: bool,
        health_check_interval: int | None,
        clear_cache_each_transformer_block: bool,
    ) -> dict[str, Callable]:
        if not compile_transformer:
            return {}
        # Eligibility (0090 d12): modes that need per-block introspection or
        # mid-graph cache flushes cannot run inside a compiled graph. Running
        # eager here is a documented mode choice, announced once - never silent.
        blockers = []
        if health_check_interval is not None:
            blockers.append("tensor_health_check_interval is set")
        if clear_cache_each_transformer_block:
            blockers.append("clear_cache_each_transformer_block (--low-ram) is set")
        if WanTransformer._block_health_enabled():
            blockers.append("MFLUX_WAN_BLOCK_HEALTH is enabled")
        if blockers:
            print(
                "compile_transformer requested but running eager: "
                + "; ".join(blockers)
                + ". These modes need per-block checks or cache flushes that cannot run inside a compiled graph."
            )
            return {}
        # One compiled callable per expert, closing over block_health_context=None
        # and no per-block cache clearing (guaranteed by the eligibility gate).
        # Shapes are constant within a run, so each expert traces exactly once.
        # A released high expert (per-item release, 0089 e4) has no entry here;
        # the denoise loop traces one lazily after the reload.
        compiled = {}
        if self.transformer is not None:
            compiled["high"] = self._compile_denoiser(self.transformer)
        if self.transformer_2 is not None:
            compiled["low"] = self._compile_denoiser(self.transformer_2)
        return compiled

    @staticmethod
    def _compile_denoiser(transformer: WanTransformer) -> Callable:
        return mx.compile(
            lambda hidden_states, timestep, encoder_hidden_states: transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
            )
        )

    def _resolve_release_inactive_denoiser(self, requested: bool | None) -> bool:
        if requested is not None:
            return bool(requested)
        # Auto-release (0089 e4) only when the freed 14 GB expert reloads cheaply:
        # dual-expert config with a DISK-prequantized checkpoint (reload is an
        # mmap read + LoRA re-fusion) and a captured reload spec. Runtime
        # quantization over bf16 stays opt-in because each reload would
        # re-quantize 14B parameters (tens of seconds per batch item).
        if not bool(self._wan_config("has_transformer_2", False)):
            return False
        if getattr(self, "transformer_stored_q_level", None) is None:
            return False
        return self._can_reload_high_noise_denoiser()

    def _can_reload_high_noise_denoiser(self) -> bool:
        # Reload spec presence (0089 e4): checkpoint identity and weight layout
        # are retained at init; LoRA paths/scales/roles stay resolved on the
        # model. A missing low expert means release_denoisers_before_decode ran,
        # which stays a terminal state.
        return (
            getattr(self, "root_path", None) is not None
            and getattr(self, "weight_definition", None) is not None
            and getattr(self, "transformer_2", None) is not None
        )

    def _ensure_high_noise_denoiser(self) -> None:
        if not self._can_reload_high_noise_denoiser():
            raise ValueError("Wan high-noise transformer was released before a high-noise timestep.")
        WanInitializer.reload_high_noise_transformer(self)
        self._high_noise_reload_count = getattr(self, "_high_noise_reload_count", 0) + 1

    def _reference_denoiser(self) -> WanTransformer:
        # Config probes (patch multiples, channel contract) while the high expert
        # is released per item (0089 e4): both A14B experts are built from
        # identical transformer kwargs, so the resident one is an exact proxy.
        denoiser = self.transformer if self.transformer is not None else self.transformer_2
        if denoiser is None:
            self._raise_denoisers_released()
        return denoiser

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
