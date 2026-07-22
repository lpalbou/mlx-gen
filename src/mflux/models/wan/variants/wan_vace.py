import gc
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.wan.model.wan_transformer import WanBlockHealthContext
from mflux.models.wan.variants.wan2_2_ti2v import Wan2_2_TI2V
from mflux.utils.generated_video import GeneratedVideo
from mflux.utils.image_util import ImageUtil
from mflux.utils.mask_util import MaskUtil
from mflux.utils.video_util import VideoUtil


class WanVace(Wan2_2_TI2V):
    # Native port of the diffusers WanVACEPipeline conditioning flow on the shared Wan runtime:
    # a single Wan2.1 transformer with VACE control blocks. Inherits weights init, UMT5 prompt
    # encoding, the wan21 VAE, scheduler construction, and save plumbing from the Wan family
    # runtime; everything VACE-specific lives here.
    RECOMMENDED_WIDTH = 832
    RECOMMENDED_HEIGHT = 480
    RECOMMENDED_FRAMES = 81
    RECOMMENDED_STEPS = 30
    RECOMMENDED_FPS = 16

    def generate_video(  # noqa: PLR0915
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = RECOMMENDED_STEPS,
        height: int = RECOMMENDED_HEIGHT,
        width: int = RECOMMENDED_WIDTH,
        num_frames: int = RECOMMENDED_FRAMES,
        fps: int = RECOMMENDED_FPS,
        guidance: float | None = None,
        guidance_2: float | None = None,
        flow_shift: float | None = None,
        solver: str | None = None,
        negative_prompt: str | None = None,
        image_path: Path | str | None = None,
        video_path: Path | str | None = None,
        video_strength: float | None = None,
        video_mask_path: Path | str | None = None,
        masked_region_mode: str = "generate",
        reference_image_paths: list[Path | str] | None = None,
        conditioning_scale: float = 1.0,
        max_sequence_length: int = 512,
        progress_callback=None,
        release_inactive_denoiser: bool | None = None,
        release_denoisers_before_decode: bool = False,
        clear_cache_each_step: bool = False,
        clear_cache_each_transformer_block: bool = False,
        tensor_health_check_interval: int | None = None,
        compile_transformer: bool = False,
    ) -> GeneratedVideo:
        start_time = time.time()
        if guidance_2 is not None:
            raise ValueError("Wan VACE uses a single transformer; guidance_2 is not supported.")
        if video_strength is not None:
            raise ValueError(
                "Wan VACE has no SDEdit warm start; video_strength is not supported. "
                "Use the mask and conditioning_scale to control how much changes."
            )
        if image_path is not None:
            raise ValueError("Wan VACE does not take image_path; pass reference_image_paths instead.")
        if solver is not None and solver != "unipc":
            raise ValueError("Wan VACE currently supports only the unipc solver.")
        if masked_region_mode not in ("generate", "repaint"):
            raise ValueError(f"masked_region_mode must be 'generate' or 'repaint', got {masked_region_mode!r}.")
        if self.transformer is None:
            raise ValueError(
                "Wan VACE denoiser has been released (release_denoisers_before_decode). "
                "Construct a fresh WanVace for another generation."
            )
        if self.transformer.vace_layers is None:
            raise ValueError(
                "This transformer has no VACE layers. WanVace requires the VACE checkpoint; construct it with "
                "model_path='Wan-AI/Wan2.1-VACE-1.3B-diffusers' or model_config=ModelConfig.from_name('wan2.1-vace-1.3b')."
            )
        self._validate_tensor_health_check_interval(tensor_health_check_interval)
        del release_inactive_denoiser  # single transformer: nothing to release mid-loop
        if compile_transformer:
            # Announced, never silent: the flag is accepted (the CLI passes it
            # unconditionally) but VACE stays eager — the conditioning branch
            # is untraced.
            print("compile_transformer is not supported on Wan VACE; running eager.")
        del compile_transformer
        height, width = self._validated_vace_spatial_size(height=height, width=width)
        num_frames = self._validated_vace_frame_count(num_frames)
        guidance = float(guidance) if guidance is not None else float(self._wan_config("default_guidance", 5.0))
        flow_shift = float(flow_shift) if flow_shift is not None else float(self._wan_config("flow_shift", 3.0))
        negative_prompt = self._resolve_negative_prompt(negative_prompt)
        reference_image_paths = [Path(path) for path in (reference_image_paths or [])]
        task = "video-to-video" if video_path is not None else "text-to-video"

        scheduler = self._create_scheduler(flow_shift=flow_shift, solver="unipc")
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps.tolist()
        total_steps = len(timesteps)
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
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=guidance > 1.0,
            max_sequence_length=max_sequence_length,
        )
        self._require_tensor_health(prompt_embeds, phase="prompt-encoding", name="prompt_embeds")

        video, mask = self._preprocess_conditions(
            video_path=video_path,
            video_mask_path=video_mask_path,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
        )
        reference_frames = self._preprocess_reference_images(
            reference_image_paths=reference_image_paths,
            height=height,
            width=width,
        )
        num_reference_images = len(reference_frames)
        control = self._prepare_control_latents(
            video=video,
            mask=mask,
            reference_frames=reference_frames,
            masked_region_mode=masked_region_mode,
        )
        del video
        self._require_tensor_health(control, phase="vace-conditioning", name="control_latents")

        mx.random.seed(seed)
        latent_frames = (num_frames - 1) // self.vae.temporal_scale + 1 + num_reference_images
        latents = mx.random.normal(
            (
                1,
                self.vae.z_dim,
                latent_frames,
                height // self.vae.spatial_scale,
                width // self.vae.spatial_scale,
            ),
            dtype=mx.float32,
        )
        conditioning_scales = [float(conditioning_scale)] * len(self.transformer.vace_layers)

        for step_index, timestep in enumerate(timesteps):
            timestep_batch = self._batch_timestep(batch_size=1, timestep=timestep)
            block_context = WanBlockHealthContext(
                step=step_index,
                total_steps=total_steps,
                timestep=timestep,
                denoiser="vace",
                guidance=guidance,
            )
            noise_pred = self.transformer(
                hidden_states=latents.astype(ModelConfig.precision),
                timestep=timestep_batch,
                encoder_hidden_states=prompt_embeds,
                clear_cache_each_block=clear_cache_each_transformer_block,
                block_health_context=block_context,
                control_hidden_states=control,
                control_hidden_states_scale=conditioning_scales,
            )
            if negative_prompt_embeds is not None:
                noise_uncond = self.transformer(
                    hidden_states=latents.astype(ModelConfig.precision),
                    timestep=timestep_batch,
                    encoder_hidden_states=negative_prompt_embeds,
                    clear_cache_each_block=clear_cache_each_transformer_block,
                    block_health_context=block_context,
                    control_hidden_states=control,
                    control_hidden_states_scale=conditioning_scales,
                )
                noise_pred = noise_uncond + guidance * (noise_pred - noise_uncond)
            latents = scheduler.step(noise_pred.astype(mx.float32), timestep, latents, return_dict=False)[0]
            mx.eval(latents)
            if clear_cache_each_step:
                mx.clear_cache()
            self._require_tensor_health(latents, phase="denoise", name=f"latents.step_{step_index}")
            self._emit_progress(
                progress_callback,
                phase="denoise",
                frame=self._progress_frame_for_step(step_index, total_steps, num_frames),
                total_frames=num_frames,
                step=step_index + 1,
                total_steps=total_steps,
                task=task,
                registry=progress_registry,
            )

        # Reference frames are conditioning-only: drop them before decoding, like the reference.
        if num_reference_images > 0:
            latents = latents[:, :, num_reference_images:]
        del control
        if release_denoisers_before_decode:
            self.transformer = None
            gc.collect()
        mx.synchronize()
        mx.clear_cache()
        # VACE keeps its all-at-once decode; the 0089 streamed-decode default covers Wan2_2_TI2V only.
        decoded = self.vae.decode_normalized_latents(latents, clear_cache_each_slice=False)
        mx.eval(decoded)
        self._require_tensor_health(decoded, phase="vae-decode", name="decoded")

        video_artifact = VideoUtil.to_video(
            decoded_latents=decoded,
            fps=fps,
            model_config=self.model_config,
            seed=seed,
            prompt=prompt,
            steps=num_inference_steps,
            guidance=guidance,
            guidance_2=None,
            flow_shift=flow_shift,
            solver="unipc",
            quantization=self.bits,
            generation_time=time.time() - start_time,
            task=task,
            image_path=None,
            video_path=video_path,
            negative_prompt=negative_prompt,
            source_width=width,
            source_height=height,
            requested_width=width,
            requested_height=height,
            lora_paths=getattr(self, "lora_paths", None),
            lora_scales=getattr(self, "lora_scales", None),
            extra_metadata=self._vace_extra_metadata(
                video_mask_path=video_mask_path,
                masked_region_mode=masked_region_mode if video_mask_path is not None else None,
                reference_image_paths=reference_image_paths,
                conditioning_scale=conditioning_scale,
                num_reference_images=num_reference_images,
            ),
            materialize_frames=False,
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
        return video_artifact

    def _validated_vace_spatial_size(self, *, height: int, width: int) -> tuple[int, int]:
        multiple = self.vae.spatial_scale * self.transformer.patch_size[1]
        if height % multiple != 0 or width % multiple != 0:
            raise ValueError(f"Wan VACE requires width/height divisible by {multiple}; got {width}x{height}.")
        return height, width

    def _validated_vace_frame_count(self, num_frames: int) -> int:
        temporal = self.vae.temporal_scale
        if num_frames < 1:
            raise ValueError("Wan VACE requires at least one frame.")
        if (num_frames - 1) % temporal != 0:
            adjusted = max((num_frames - 1) // temporal * temporal + 1, 1)
            print(f"Wan VACE adjusted --frames from {num_frames} to {adjusted} ((frames - 1) must divide {temporal}).")
            return adjusted
        return num_frames

    def _preprocess_conditions(
        self,
        *,
        video_path: Path | str | None,
        video_mask_path: Path | str | None,
        height: int,
        width: int,
        num_frames: int,
        fps: int,
    ) -> tuple[mx.array, mx.array]:
        if video_path is not None:
            clip = VideoUtil.read_video_clip(video_path, max_frames=num_frames, target_fps=float(fps))
            if clip.clip_frame_count < num_frames:
                raise ValueError(
                    f"Wan VACE needs {num_frames} source frames "
                    f"({num_frames / float(fps):.2f}s at {fps} fps), but {video_path} only yielded "
                    f"{clip.clip_frame_count}."
                )
            frames_np = np.empty((1, num_frames, height, width, 3), dtype=np.float32)
            for index, frame in enumerate(clip.frames[:num_frames]):
                frames_np[0, index] = np.asarray(
                    ImageUtil.scale_to_dimensions(frame, target_width=width, target_height=height),
                    dtype=np.float32,
                )
            frames_np = frames_np / 127.5 - 1.0
            video = mx.transpose(mx.array(frames_np), (0, 4, 1, 2, 3))
        else:
            # Reference behavior: no source video means a zero canvas in [-1, 1] space.
            video = mx.zeros((1, 3, num_frames, height, width), dtype=mx.float32)

        if video_mask_path is not None:
            if self._is_video_mask(video_mask_path):
                mask = self._load_video_mask_frames(
                    video_mask_path,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    fps=fps,
                )
            else:
                mask_values = MaskUtil.load_binary_mask(
                    video_mask_path,
                    target_width=width,
                    target_height=height,
                    resampling=Image.Resampling.BOX,
                    alpha_warning_context="Wan VACE mask",
                )
                mask = mx.broadcast_to(
                    mx.array(mask_values)[None, None, None, :, :],
                    (1, 3, num_frames, height, width),
                ).astype(mx.float32)
        else:
            mask = mx.ones_like(video)
        return video, mask

    VIDEO_MASK_SUFFIXES = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".gif")

    @classmethod
    def _is_video_mask(cls, mask_path: Path | str) -> bool:
        return str(mask_path).lower().endswith(cls.VIDEO_MASK_SUFFIXES)

    @staticmethod
    def _load_video_mask_frames(
        mask_path: Path | str,
        *,
        height: int,
        width: int,
        num_frames: int,
        fps: int,
    ) -> mx.array:
        # Per-frame animated masks are the native VACE format: the mask trajectory is what
        # carries the object's motion into the conditioning. Read on the same fps timeline as
        # the source clip so mask frames align 1:1 with video frames.
        clip = VideoUtil.read_video_clip(mask_path, max_frames=num_frames, target_fps=float(fps))
        if clip.clip_frame_count < num_frames:
            raise ValueError(
                f"Wan VACE video mask needs {num_frames} frames "
                f"({num_frames / float(fps):.2f}s at {fps} fps), but {mask_path} only yielded "
                f"{clip.clip_frame_count}."
            )
        mask_np = np.empty((num_frames, height, width), dtype=np.float32)
        for index, frame in enumerate(clip.frames[:num_frames]):
            resized = frame.convert("L").resize((width, height), Image.Resampling.BOX)
            # Same policy as MaskUtil: BOX area averaging, then a 50% binarization threshold.
            mask_np[index] = (np.asarray(resized, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)
        mask = mx.array(mask_np)[None, None, :, :, :]
        return mx.broadcast_to(mask, (1, 3, num_frames, height, width)).astype(mx.float32)

    def _preprocess_reference_images(
        self,
        *,
        reference_image_paths: list[Path],
        height: int,
        width: int,
    ) -> list[mx.array]:
        reference_frames = []
        for path in reference_image_paths:
            with Image.open(path) as image:
                image = image.convert("RGB")
                scale = min(height / image.height, width / image.width)
                new_height, new_width = int(image.height * scale), int(image.width * scale)
                resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
            # White canvas letterboxing, matching the reference (canvas of ones in [-1, 1]).
            canvas = np.ones((height, width, 3), dtype=np.float32)
            top = (height - new_height) // 2
            left = (width - new_width) // 2
            canvas[top : top + new_height, left : left + new_width] = (
                np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
            )
            reference_frames.append(mx.array(canvas).transpose(2, 0, 1)[None, :, None, :, :])
        return reference_frames

    def _prepare_control_latents(
        self,
        *,
        video: mx.array,
        mask: mx.array,
        reference_frames: list[mx.array],
        masked_region_mode: str = "generate",
    ) -> mx.array:
        binary_mask = mx.where(mask > 0.5, 1.0, 0.0).astype(video.dtype)
        if masked_region_mode == "generate":
            # Official VACE inpainting convention (ali-vilab UserGuide): the editable region of
            # the source video is gray-filled (127.5 = 0.0 in [-1, 1]) before encoding, so the
            # reactive branch reads "missing - generate here" instead of "repaint this content".
            video = video * (1 - binary_mask)
        inactive = self.vae.encode_normalized(video * (1 - binary_mask)).astype(mx.float32)
        reactive = self.vae.encode_normalized(video * binary_mask).astype(mx.float32)
        conditioning = mx.concatenate([inactive, reactive], axis=1)

        # Reference prepend order matches the diffusers loop: each ref is prepended in turn, so
        # the LAST reference image ends up as the first latent frame.
        for reference_frame in reference_frames:
            reference_latent = self.vae.encode_normalized(reference_frame).astype(mx.float32)
            reference_latent = mx.concatenate([reference_latent, mx.zeros_like(reference_latent)], axis=1)
            conditioning = mx.concatenate([reference_latent, conditioning], axis=2)

        mask_channels = self._prepare_mask_channels(
            mask=mask,
            latent_frames=inactive.shape[2],
            num_reference_images=len(reference_frames),
        )
        control = mx.concatenate([conditioning, mask_channels], axis=1)
        mx.eval(control)
        return control.astype(ModelConfig.precision)

    def _prepare_mask_channels(
        self,
        *,
        mask: mx.array,
        latent_frames: int,
        num_reference_images: int,
    ) -> mx.array:
        spatial = self.vae.spatial_scale
        num_frames, height, width = mask.shape[2], mask.shape[3], mask.shape[4]
        latent_height = height // spatial
        latent_width = width // spatial
        single = mask[0, 0]  # [F, H, W]
        single = single.reshape(num_frames, latent_height, spatial, latent_width, spatial)
        single = mx.transpose(single, (2, 4, 0, 1, 3)).reshape(
            spatial * spatial, num_frames, latent_height, latent_width
        )
        # torch nearest-exact temporal resampling: source index floor((i + 0.5) * F / f).
        indices = [min(int((i + 0.5) * num_frames / latent_frames), num_frames - 1) for i in range(latent_frames)]
        single = single[:, mx.array(indices), :, :]
        if num_reference_images > 0:
            padding = mx.zeros((single.shape[0], num_reference_images, latent_height, latent_width), dtype=single.dtype)
            single = mx.concatenate([padding, single], axis=1)
        return single[None, :, :, :, :]

    @staticmethod
    def _vace_extra_metadata(
        *,
        video_mask_path: Path | str | None,
        masked_region_mode: str | None,
        reference_image_paths: list[Path],
        conditioning_scale: float,
        num_reference_images: int,
    ) -> dict:
        extra = {
            "vace": True,
            "conditioning_scale": conditioning_scale,
            "num_reference_images": num_reference_images,
        }
        if video_mask_path is not None:
            extra["video_mask_path"] = str(video_mask_path)
        if masked_region_mode is not None:
            extra["masked_region_mode"] = masked_region_mode
        if reference_image_paths:
            extra["reference_image_paths"] = [str(path) for path in reference_image_paths]
        return extra
