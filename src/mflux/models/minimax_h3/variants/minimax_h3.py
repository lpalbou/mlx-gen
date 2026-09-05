"""MiniMax-H3: joint video + stereo audio generation from a structured text prompt.

Follows the diffusers modular pipeline step for step: Qwen3-VL `hidden_states[50]` conditioning, one
packed sequence of text / audio / video rows, a guidance-distilled transformer (one forward per
step), two rectified-flow schedulers (video shift 12, audio shift 3) advanced in lockstep, the
visual VAE decode in ImageNet pixel space and the fp32 audio VAE decode at 32 kHz.
"""

import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import PIL.Image
from mlx import nn
from mlx.utils import tree_flatten

from mflux.callbacks import ProgressEvent
from mflux.models.common.config import ModelConfig
from mflux.models.minimax_h3.latent_creator.h3_layout import (
    AUDIO_CHANNELS,
    FPS,
    KEYFRAME_NOISE_AUG,
    MAX_ASPECT_RATIO,
    MIN_ASPECT_RATIO,
    PIXEL_MEAN,
    PIXEL_STD,
    TEXT_TAG,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    resolve_canvas_size,
    unpack_audio_rows,
    unpatchify_video_rows,
    video_latent_num_frames,
)
from mflux.models.minimax_h3.minimax_h3_initializer import MiniMaxH3Initializer
from mflux.models.minimax_h3.scheduler.minimax_h3_scheduler import MiniMaxH3Scheduler
from mflux.models.minimax_h3.weights.h3_weight_definition import MiniMaxH3WeightDefinition
from mflux.utils.generated_audio import GeneratedAudio
from mflux.utils.generated_video import GeneratedVideo

PROMPT_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
CANVAS_MULTIPLE = 32
DEFAULT_VIDEO_SHIFT = 12.0
DEFAULT_AUDIO_SHIFT = 3.0


def compose_prompt(prompt: str, soundscape: str | None = None, music: str | None = None) -> str:
    """The structured prompt MiniMax-H3 was trained on: labelled sections separated by blank lines.

    A prompt that already carries the field labels is passed through verbatim (the MiniMax
    Context-IR output); a plain description becomes the `integrated_multimodal_description`.
    """
    text = prompt.strip()
    sections = [text] if any(f"{field}:" in text for field in PROMPT_FIELDS) else [f"{PROMPT_FIELDS[0]}: {text}"]
    if soundscape and soundscape.strip():
        sections.append(f"{PROMPT_FIELDS[1]}: {soundscape.strip()}")
    if music and music.strip():
        sections.append(f"{PROMPT_FIELDS[2]}: {music.strip()}")
    return "\n\n".join(sections)


@dataclass(frozen=True)
class H3GenerationPlan:
    width: int
    height: int
    num_frames: int
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int
    num_inference_steps: int
    video_shift: float
    audio_shift: float

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / FPS


class MiniMaxH3(nn.Module):
    RECOMMENDED_FRAMES = 124  # 17 * 7 + 5 frames: 5.17 s at 24 fps
    RECOMMENDED_FPS = FPS

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()
        MiniMaxH3Initializer.init(
            self,
            model_config=model_config or ModelConfig.minimax_h3(),
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    # ------------------------------------------------------------------ public API

    def generate_video(
        self,
        seed: int,
        prompt: str,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int | None = None,
        video_shift: float | None = None,
        audio_shift: float | None = None,
        image_path: str | None = None,
        soundscape: str | None = None,
        music: str | None = None,
        generate_audio: bool = True,
        progress_callback=None,
        release_text_encoder: bool = False,
    ) -> GeneratedVideo:
        if image_path is not None:
            raise NotImplementedError(
                "MiniMax-H3 first-frame conditioning (FL2VA) needs the Qwen3-VL vision tower for the keyframe tokens; "
                "it is not ported yet. Text-to-video-with-audio is available."
            )
        started = time.perf_counter()
        plan = self._plan(width, height, num_frames, num_inference_steps, video_shift, audio_shift)
        full_prompt = compose_prompt(prompt, soundscape, music)
        task = "text-to-video"

        prompt_embeds = self._encode_prompt(full_prompt)
        if release_text_encoder:
            self.release_text_encoder()
        text_length = prompt_embeds.shape[1]
        layout = build_packed_sequence(
            np.full((text_length,), TEXT_TAG, dtype=np.int32),
            plan.num_latent_frames,
            plan.latent_height,
            plan.latent_width,
            plan.num_audio_latents,
            patch_size=self.transformer.patch_size,
            audio_channels=AUDIO_CHANNELS,
        )
        self._emit(progress_callback, phase="start", plan=plan, step=0, seed=seed, task=task)

        # Draw order matches the reference: video latents first, then the audio rows.
        key_video, key_audio = mx.random.split(mx.random.key(seed), 2)
        latent_channels = self.transformer.in_channels
        video_latents = mx.random.normal(
            (1, latent_channels, plan.num_latent_frames, plan.latent_height, plan.latent_width),
            key=key_video,
            dtype=mx.float32,
        )
        video_rows = patchify_video_latents(video_latents, self.transformer.patch_size)
        audio_rows = mx.random.normal(
            (plan.num_audio_latents * AUDIO_CHANNELS, self.transformer.audio_in_channels),
            key=key_audio,
            dtype=mx.float32,
        )

        video_scheduler = MiniMaxH3Scheduler(shift=plan.video_shift)
        audio_scheduler = MiniMaxH3Scheduler(shift=plan.audio_shift)
        video_scheduler.set_timesteps(plan.num_inference_steps)
        audio_scheduler.set_timesteps(plan.num_inference_steps)
        num_condition_video_rows = layout.num_condition_video_rows
        num_condition_audio_rows = layout.num_condition_audio_rows

        total_steps = len(video_scheduler.timesteps)
        for step, (video_t, audio_t) in enumerate(zip(video_scheduler.timesteps, audio_scheduler.timesteps)):
            unique_timesteps, timestep_indices = build_row_timesteps(
                layout,
                float(video_t),
                float(audio_t),
                max(float(video_t), KEYFRAME_NOISE_AUG),
                1.0,
            )
            video_pred, audio_pred = self.transformer(
                hidden_states=video_rows[None],
                audio_hidden_states=audio_rows[None],
                encoder_hidden_states=prompt_embeds,
                timestep=unique_timesteps,
                timestep_indices=timestep_indices,
                token_tags=layout.token_tags,
                position_ids=layout.position_ids,
                video_indices=layout.video_indices,
                audio_indices=layout.audio_indices,
                text_indices=layout.text_indices,
            )
            video_rows[num_condition_video_rows:] = video_scheduler.step(
                video_pred[0, num_condition_video_rows:].astype(mx.float32), step, video_rows[num_condition_video_rows:]
            )
            audio_rows[num_condition_audio_rows:] = audio_scheduler.step(
                audio_pred[0, num_condition_audio_rows:].astype(mx.float32), step, audio_rows[num_condition_audio_rows:]
            )
            mx.eval(video_rows, audio_rows)
            self._emit(
                progress_callback,
                phase="denoising",
                plan=plan,
                step=step + 1,
                seed=seed,
                task=task,
                timestep=float(video_t),
            )

        self._emit(progress_callback, phase="decode", plan=plan, step=total_steps, seed=seed, task=task)
        frames = self._decode_video(video_rows[num_condition_video_rows:], plan)
        audio = self._decode_audio(audio_rows[num_condition_audio_rows:], plan) if generate_audio else None
        self._emit(progress_callback, phase="complete", plan=plan, step=total_steps, seed=seed, task=task)

        return GeneratedVideo(
            frames=frames,
            fps=FPS,
            model_config=self.model_config,
            seed=seed,
            prompt=full_prompt,
            steps=total_steps,
            guidance=None,
            precision=ModelConfig.precision,
            quantization=self.bits,
            generation_time=time.perf_counter() - started,
            height=plan.height,
            width=plan.width,
            task=task,
            image_path=image_path,
            flow_shift=plan.video_shift,
            lora_paths=getattr(self, "lora_paths", None) or None,
            lora_scales=getattr(self, "lora_scales", None) or None,
            extra_metadata={
                "video_shift": plan.video_shift,
                "audio_shift": plan.audio_shift,
                "num_inference_steps": plan.num_inference_steps,
                "duration_seconds": round(plan.duration_seconds, 4),
                "text_tokens": int(text_length),
                "generate_audio": bool(generate_audio),
            },
            audio=audio,
        )

    def release_text_encoder(self) -> None:
        """Drop the conditioner (the largest resident component); cached prompt embeddings stay usable."""
        if getattr(self, "text_encoder", None) is None:
            return
        self.text_encoder = None
        import gc

        gc.collect()
        mx.clear_cache()

    def save_model(self, base_path: str) -> None:
        from mflux.models.common.weights.saving.model_saver import ModelSaver

        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=getattr(
                self, "weight_definition", MiniMaxH3WeightDefinition.for_config(self.model_config)
            ),
        )
        self._copy_runtime_assets(base_path)

    def _copy_runtime_assets(self, base_path: str) -> None:
        """Component configs travel with a prepared package so a reload rebuilds the exact released shapes."""
        import shutil
        from pathlib import Path

        root = Path(getattr(self, "root_path", ""))
        for relative in (
            "model_index.json",
            "vae/config.json",
            "audio_vae/config.json",
            "transformer/config.json",
            "text_encoder/config.json",
        ):
            source = root / relative
            if source.exists():
                target = Path(base_path) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)

    # ------------------------------------------------------------------ planning

    def _plan(
        self,
        width: int | None,
        height: int | None,
        num_frames: int | None,
        num_inference_steps: int | None,
        video_shift: float | None,
        audio_shift: float | None,
    ) -> H3GenerationPlan:
        overrides = self.model_config.transformer_overrides
        if (width is None) != (height is None):
            raise ValueError("width and height must be given together, or neither of them.")
        if width is None:
            if overrides.get("default_width") and overrides.get("default_height"):
                width, height = int(overrides["default_width"]), int(overrides["default_height"])
            else:
                height, width = resolve_canvas_size(16, 9, CANVAS_MULTIPLE)
        if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
            raise ValueError(
                f"MiniMax-H3 width and height must be multiples of {CANVAS_MULTIPLE}, got {width}x{height}."
            )
        if not MIN_ASPECT_RATIO <= width / height <= MAX_ASPECT_RATIO:
            raise ValueError(f"MiniMax-H3 aspect ratio must lie within [1/4, 4], got {width}x{height}.")
        requested_frames = num_frames or int(overrides.get("default_frames", self.RECOMMENDED_FRAMES))
        aligned_frames = align_num_frames(requested_frames)
        duration = aligned_frames / FPS
        if not 5.0 <= duration <= 15.0:
            raise ValueError(
                f"MiniMax-H3 generates 5 to 15 seconds at {FPS} fps: num_frames (rounded up to 17n+5) must lie in "
                f"[124, 362], got {requested_frames} (rounded to {aligned_frames})."
            )
        steps = int(num_inference_steps or overrides.get("default_steps", 50))
        if steps < 1:
            raise ValueError("num_inference_steps must be at least 1.")
        ratio = self.vae.spatial_compression_ratio
        return H3GenerationPlan(
            width=int(width),
            height=int(height),
            num_frames=aligned_frames,
            num_latent_frames=video_latent_num_frames(aligned_frames),
            latent_height=height // ratio,
            latent_width=width // ratio,
            num_audio_latents=audio_latent_num_frames(aligned_frames),
            # `steps` counts transformer evaluations; the scheduler grid has one more point.
            num_inference_steps=steps + 1,
            video_shift=float(
                video_shift if video_shift is not None else overrides.get("default_video_shift", DEFAULT_VIDEO_SHIFT)
            ),
            audio_shift=float(
                audio_shift if audio_shift is not None else overrides.get("default_audio_shift", DEFAULT_AUDIO_SHIFT)
            ),
        )

    # ------------------------------------------------------------------ conditioning

    def _encode_prompt(self, prompt: str) -> mx.array:
        cached = self.prompt_embed_cache.get(prompt)
        if cached is not None:
            return cached
        if getattr(self, "text_encoder", None) is None:
            raise RuntimeError("The MiniMax-H3 text encoder was released; only cached prompts can be generated.")
        tokens = self.tokenizers[MiniMaxH3WeightDefinition.TOKENIZER_NAME].tokenize(prompt)
        input_ids = tokens.input_ids
        if input_ids.shape[1] == 0:
            raise ValueError("MiniMax-H3 needs a non-empty prompt.")
        embeds = self.text_encoder(input_ids).astype(ModelConfig.precision)
        mx.eval(embeds)
        self.prompt_embed_cache[prompt] = embeds
        return embeds

    # ------------------------------------------------------------------ decoding

    def _decode_video(self, video_rows: mx.array, plan: H3GenerationPlan) -> list[PIL.Image.Image]:
        latents = unpatchify_video_rows(
            video_rows,
            self.transformer.in_channels,
            plan.num_latent_frames,
            plan.latent_height,
            plan.latent_width,
            self.transformer.patch_size,
        )
        latents = latents * self.vae.latents_std.reshape(1, -1, 1, 1, 1) + self.vae.latents_mean.reshape(1, -1, 1, 1, 1)
        pixels = self.vae.decode(
            latents.astype(self._component_dtype(self.vae))
        )  # (1, 3, F, H, W), ImageNet-normalized
        pixels = pixels.astype(mx.float32) * mx.array(PIXEL_STD).reshape(1, 3, 1, 1, 1) + mx.array(PIXEL_MEAN).reshape(
            1, 3, 1, 1, 1
        )
        pixels = mx.clip(pixels * 255.0 + 0.5, 0, 255).astype(mx.uint8)[0].transpose(1, 2, 3, 0)  # (F, H, W, 3)
        mx.eval(pixels)
        frames = np.array(pixels)
        return [PIL.Image.fromarray(frame) for frame in frames]

    def _decode_audio(self, audio_rows: mx.array, plan: H3GenerationPlan) -> GeneratedAudio:
        latents = unpack_audio_rows(audio_rows, AUDIO_CHANNELS, plan.num_audio_latents)  # (2, C, A)
        latents = latents * self.audio_vae.latents_std.reshape(1, -1, 1) + self.audio_vae.latents_mean.reshape(1, -1, 1)
        waveform = self.audio_vae.decode(latents.astype(mx.float32))  # (2, 1, samples)
        mx.eval(waveform)
        track = GeneratedAudio(
            waveform=np.array(waveform[:, 0, :], dtype=np.float32), sample_rate=self.audio_vae.sampling_rate
        )
        return track.trimmed(plan.duration_seconds)

    @staticmethod
    def _component_dtype(module: nn.Module) -> mx.Dtype:
        for _, value in tree_flatten(module.parameters()):
            if value.dtype in (mx.float32, mx.bfloat16, mx.float16):
                return value.dtype
        return ModelConfig.precision

    # ------------------------------------------------------------------ progress

    @staticmethod
    def _emit(
        callback, *, phase: str, plan: H3GenerationPlan, step: int, seed: int, task: str, timestep: float | None = None
    ):
        if callback is None:
            return
        callback(
            ProgressEvent(
                phase=phase,
                step=step,
                total_steps=plan.num_inference_steps - 1,
                total_frames=plan.num_frames,
                task=task,
                timestep=timestep,
                seed=seed,
                width=plan.width,
                height=plan.height,
            )
        )
