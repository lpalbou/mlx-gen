"""MiniMax-H3: joint video + stereo audio generation from a structured text prompt.

Follows the diffusers modular pipeline step for step: Qwen3-VL `hidden_states[50]` conditioning, one
packed sequence of text / audio / video rows, a guidance-distilled transformer (one forward per
step), two rectified-flow schedulers (video shift 12, audio shift 3) advanced in lockstep, the
visual VAE decode in ImageNet pixel space and the fp32 audio VAE decode at 32 kHz.
"""

import hashlib
import sys
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
    CANVAS_MAX_PIXELS,
    CANVAS_SHORT_EDGE,
    FPS,
    KEYFRAME_ENCODE_SEED,
    KEYFRAME_NOISE_AUG,
    MAX_ASPECT_RATIO,
    MAX_DURATION_SECONDS,
    MAX_NUM_FRAMES,
    MIN_ASPECT_RATIO,
    MIN_DURATION_SECONDS,
    MIN_NUM_FRAMES,
    PIXEL_MEAN,
    PIXEL_STD,
    TEXT_TAG,
    VIDEO_TAG,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    packed_sequence_length,
    patchify_video_latents,
    resolve_canvas_size,
    unpack_audio_rows,
    unpatchify_video_rows,
    valid_frame_counts,
    video_latent_num_frames,
)
from mflux.models.minimax_h3.minimax_h3_initializer import MiniMaxH3Initializer
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_model import (
    IMAGE_TOKEN_ID,
    VISION_END_TOKEN_ID,
    VISION_START_TOKEN_ID,
)
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_vision_model import preprocess_image
from mflux.models.minimax_h3.scheduler.minimax_h3_scheduler import MiniMaxH3Scheduler
from mflux.models.minimax_h3.weights.h3_weight_definition import MiniMaxH3WeightDefinition
from mflux.utils.generated_audio import GeneratedAudio
from mflux.utils.generated_video import GeneratedVideo
from mflux.utils.runtime_memory import RuntimeMemory

PROMPT_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
CANVAS_MULTIPLE = 32
DEFAULT_VIDEO_SHIFT = 12.0
DEFAULT_AUDIO_SHIFT = 3.0


def prompt_sections_present(prompt: str) -> tuple[str, ...]:
    """The section labels a prompt already carries, in `PROMPT_FIELDS` order.

    A label counts only where it opens a line, which is the format the model was trained on; the
    same words inside a sentence are prose, not a section header.
    """
    lines = [line.lstrip() for line in prompt.strip().splitlines()]
    return tuple(field for field in PROMPT_FIELDS if any(line.startswith(f"{field}:") for line in lines))


def compose_prompt(prompt: str, soundscape: str | None = None, music: str | None = None) -> str:
    """The structured prompt MiniMax-H3 was trained on: labelled sections separated by blank lines.

    A prompt that already carries the field labels is passed through verbatim (the MiniMax
    Context-IR output); a plain description becomes the `integrated_multimodal_description`.
    """
    text = prompt.strip()
    present = prompt_sections_present(text)
    for field, value in ((PROMPT_FIELDS[1], soundscape), (PROMPT_FIELDS[2], music)):
        if value and value.strip() and field in present:
            # Appending would send the model two of the same section, and the contradiction only
            # surfaces after the whole run. Say so now instead.
            option = "--soundscape" if field == PROMPT_FIELDS[1] else "--music"
            raise ValueError(
                f"The prompt already carries a `{field}:` section, so {option} would add a second one. "
                f"Drop {option}, or remove that section from the prompt."
            )
    sections = [text] if present else [f"{PROMPT_FIELDS[0]}: {text}"]
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
    # The caller's own count before the 17n+5 snap; equal to `num_frames` unless the request was off-grid.
    requested_frames: int = 0

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
        started = time.perf_counter()
        keyframe = self._load_keyframe(image_path) if image_path is not None else None
        plan = self._plan(width, height, num_frames, num_inference_steps, video_shift, audio_shift, keyframe=keyframe)
        self._preflight_request(plan)
        full_prompt = compose_prompt(prompt, soundscape, music)
        task = "image-to-video" if keyframe is not None else "text-to-video"
        # The keyframe is stretched onto the canvas (the reference's geometry anchor), then conditions both the text
        # sequence (a `<Picture 1>: ` label plus its vision block) and the packed latent rows.
        keyframes = [self._fit_keyframe(keyframe, plan.width, plan.height)] if keyframe is not None else []

        prompt_embeds, text_token_tags = self._encode_presentation(full_prompt, keyframes)
        if release_text_encoder:
            self.release_text_encoder()
        text_length = prompt_embeds.shape[1]
        layout = build_packed_sequence(
            text_token_tags,
            plan.num_latent_frames,
            plan.latent_height,
            plan.latent_width,
            plan.num_audio_latents,
            patch_size=self.transformer.patch_size,
            audio_channels=AUDIO_CHANNELS,
            keyframe_anchors=("first",) if keyframes else (),
        )
        self._emit(progress_callback, phase="start", plan=plan, step=0, seed=seed, task=task)

        video_scheduler = MiniMaxH3Scheduler(shift=plan.video_shift)
        audio_scheduler = MiniMaxH3Scheduler(shift=plan.audio_shift)
        video_scheduler.set_timesteps(plan.num_inference_steps)
        audio_scheduler.set_timesteps(plan.num_inference_steps)

        # Draw order matches the reference: conditioning noise first, then the video latents, then the audio rows.
        keys = mx.random.split(mx.random.key(seed), 3 if keyframes else 2)
        key_video, key_audio = keys[-2], keys[-1]
        latent_channels = self.transformer.in_channels
        video_latents = mx.random.normal(
            (1, latent_channels, plan.num_latent_frames, plan.latent_height, plan.latent_width),
            key=key_video,
            dtype=mx.float32,
        )
        video_rows = patchify_video_latents(video_latents, self.transformer.patch_size)
        if keyframes:
            condition = self._encode_keyframe_latents(keyframes[0])
            condition_noise = mx.random.normal(condition.shape, key=keys[0], dtype=mx.float32)
            # The anchor is not fully clean: the released model holds it at `t = 0.999` for every step.
            noised = video_scheduler.scale_noise(condition, KEYFRAME_NOISE_AUG, condition_noise)
            video_rows = mx.concatenate(
                [patchify_video_latents(noised, self.transformer.patch_size), video_rows], axis=0
            )
        audio_rows = mx.random.normal(
            (plan.num_audio_latents * AUDIO_CHANNELS, self.transformer.audio_in_channels),
            key=key_audio,
            dtype=mx.float32,
        )

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
                phase="denoise",
                plan=plan,
                step=step + 1,
                seed=seed,
                task=task,
                timestep=float(video_t),
            )

        self._emit(progress_callback, phase="decode", plan=plan, step=total_steps, seed=seed, task=task)
        frames = self._decode_video(video_rows[num_condition_video_rows:], plan)
        audio = self._decode_audio(audio_rows[num_condition_audio_rows:], plan) if generate_audio else None
        # `generated`, not `complete`: the CLI emits `complete` once the file is written and the audio
        # muxed, so a terminal `complete` here would point a host at a path with no file behind it.
        self._emit(progress_callback, phase="generated", plan=plan, step=total_steps, seed=seed, task=task)

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
            source_width=keyframe.width if keyframe is not None else None,
            source_height=keyframe.height if keyframe is not None else None,
            flow_shift=plan.video_shift,
            lora_paths=getattr(self, "lora_paths", None) or None,
            lora_scales=getattr(self, "lora_scales", None) or None,
            extra_metadata={
                "video_shift": plan.video_shift,
                "audio_shift": plan.audio_shift,
                "num_inference_steps": plan.num_inference_steps,
                "duration_seconds": round(plan.duration_seconds, 4),
                # What the caller asked for, so a host can see that an off-grid request was rounded up.
                "requested_frames": plan.requested_frames,
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
        keyframe: PIL.Image.Image | None = None,
    ) -> H3GenerationPlan:
        overrides = self.model_config.transformer_overrides
        if (width is None) != (height is None):
            raise ValueError("width and height must be given together, or neither of them.")
        if width is None:
            if keyframe is not None:
                # MiniMax-H3's own geometry for the keyframe's aspect ratio (768 short edge, or the entry's override).
                height, width = resolve_canvas_size(
                    keyframe.width,
                    keyframe.height,
                    CANVAS_MULTIPLE,
                    int(overrides.get("canvas_short_edge", CANVAS_SHORT_EDGE)),
                    int(overrides.get("canvas_max_pixels", CANVAS_MAX_PIXELS)),
                )
            elif overrides.get("default_width") and overrides.get("default_height"):
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
        if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
            # The ceiling holds for the ALIGNED count, so the largest accepted value is 345, not 362:
            # 346 rounds up to 362, which is 15.083 s and outside the window the model generates.
            raise ValueError(
                f"MiniMax-H3 generates {MIN_DURATION_SECONDS:g} to {MAX_DURATION_SECONDS:g} seconds at {FPS} fps, so "
                f"num_frames (rounded up to the next 17n+5) must lie in [{MIN_NUM_FRAMES}, {MAX_NUM_FRAMES}], got "
                f"{requested_frames} (rounded to {aligned_frames}). Accepted counts: "
                f"{', '.join(str(count) for count in valid_frame_counts())}."
            )
        if aligned_frames != requested_frames:
            # The reference warns here too: a silently lengthened clip is the failure mode of a host
            # slider that steps off the grid. `requested_frames` also lands in the metadata.
            print(
                f"⚠️  MiniMax-H3 frame counts are 17n+5: rounding {requested_frames} up to {aligned_frames} "
                f"({aligned_frames / FPS:.2f} s).",
                file=sys.stderr,
            )
        steps = int(num_inference_steps or overrides.get("default_steps", 50))
        if steps < 1:
            raise ValueError("num_inference_steps must be at least 1.")
        ratio = self.vae.spatial_compression_ratio
        return H3GenerationPlan(
            width=int(width),
            height=int(height),
            num_frames=aligned_frames,
            requested_frames=int(requested_frames),
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
        return self._encode_presentation(prompt, [])[0]

    def _encode_presentation(self, prompt: str, keyframes: list[PIL.Image.Image]) -> tuple[mx.array, np.ndarray]:
        """Tokenize the presentation (`<Picture i>: ` + vision block per keyframe, then the prompt verbatim) and
        return `hidden_states[50]` with MiniMax-H3's per-row modality tags (vision blocks count as video rows)."""
        digest = hashlib.sha1(prompt.encode("utf-8"))
        for frame in keyframes:
            digest.update(str(frame.size).encode())
            digest.update(frame.tobytes())
        cache_key = digest.hexdigest()
        cached = self.prompt_embed_cache.get(cache_key)
        if cached is not None:
            return cached
        if getattr(self, "text_encoder", None) is None:
            raise RuntimeError("The MiniMax-H3 text encoder was released; only cached prompts can be generated.")
        tokenizer = self.tokenizers[MiniMaxH3WeightDefinition.TOKENIZER_NAME].tokenizer
        merge = self.text_encoder.visual.spatial_merge_size
        token_ids: list[int] = []
        tags: list[int] = []
        patches, grids = [], []
        for index, frame in enumerate(keyframes):
            frame_patches, grid = preprocess_image(frame)
            patches.append(frame_patches)
            grids.append(grid)
            label = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
            num_image_tokens = (grid[0] * grid[1] * grid[2]) // (merge * merge)
            vision = [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * num_image_tokens + [VISION_END_TOKEN_ID]
            token_ids += label + vision
            tags += [TEXT_TAG] * len(label) + [VIDEO_TAG] * len(vision)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            raise ValueError("MiniMax-H3 needs a non-empty prompt.")
        token_ids += prompt_ids
        tags += [TEXT_TAG] * len(prompt_ids)
        embeds = self.text_encoder.encode(np.array(token_ids, dtype=np.int32), patches or None, grids or None)
        embeds = embeds.astype(ModelConfig.precision)
        mx.eval(embeds)
        result = (embeds, np.array(tags, dtype=np.int32))
        self.prompt_embed_cache[cache_key] = result
        return result

    @staticmethod
    def _load_keyframe(image_path: str) -> PIL.Image.Image:
        return PIL.Image.open(image_path).convert("RGB")

    @staticmethod
    def _fit_keyframe(keyframe: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
        """The geometry anchor is stretched onto the canvas (PIL LANCZOS), as in the reference pipeline."""
        if keyframe.size == (width, height):
            return keyframe
        return keyframe.resize((width, height), PIL.Image.Resampling.LANCZOS)

    def _encode_keyframe_latents(self, keyframe: PIL.Image.Image) -> mx.array:
        """Normalized condition latents `(1, C, 1, H/16, W/16)`: ImageNet-normalized pixels through the video VAE,
        the posterior *sampled* under a fixed seed and rounded to float16, as the released model conditions."""
        pixels = np.asarray(keyframe, dtype=np.float32) / 255.0
        pixels = (pixels - np.array(PIXEL_MEAN, dtype=np.float32)) / np.array(PIXEL_STD, dtype=np.float32)
        pixels = mx.array(pixels.transpose(2, 0, 1))[None, :, None]  # (1, 3, 1, H, W)
        mean, logvar = self.vae.encode(pixels.astype(self._component_dtype(self.vae)))
        mean, logvar = mean.astype(mx.float32), logvar.astype(mx.float32)
        noise = mx.random.normal(mean.shape, key=mx.random.key(KEYFRAME_ENCODE_SEED), dtype=mx.float32)
        latents = (mean + mx.exp(0.5 * logvar) * noise).astype(mx.float16).astype(mx.float32)
        latents = (latents - self.vae.latents_mean.reshape(1, -1, 1, 1, 1)) / self.vae.latents_std.reshape(
            1, -1, 1, 1, 1
        )
        mx.eval(latents)
        return latents

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

    # Peak process footprint as a function of packed rows, fitted on runs at 960x544 and 1344x768 and
    # confirmed by a third at a different frame count: a 243-frame 960x544 clip and a 124-frame
    # 1344x768 clip differ by 0.5% in rows and were measured at the same MLX peak, 84.8 GiB. Measured
    # at q8 with the default 8 GiB cache and the conditioner resident; a different cache limit moves
    # the footprint by far more than the canvas does.
    PEAK_FIXED_BYTES = 89_410_000_000
    PEAK_BYTES_PER_PACKED_ROW = 271_356

    @classmethod
    def estimate_peak_bytes(cls, width: int, height: int, num_frames: int) -> int:
        """Expected peak process footprint for a request, from its packed row count."""
        return int(
            cls.PEAK_FIXED_BYTES + cls.PEAK_BYTES_PER_PACKED_ROW * packed_sequence_length(width, height, num_frames)
        )

    def _preflight_request(self, plan: H3GenerationPlan) -> None:
        """Report before a long run whose expected peak does not fit, rather than after.

        The expected peak is a measured line through packed rows, not a guarantee: whether a run
        survives also depends on what else is resident, and a request has been killed at 72% of this
        machine's RAM because another application held 7.5 GiB. So this refuses only what cannot fit
        at all and warns about the rest, naming the levers instead of changing the request.
        """
        physical_bytes = RuntimeMemory.total_physical_memory_bytes()
        if physical_bytes <= 0:
            return
        expected = self.estimate_peak_bytes(plan.width, plan.height, plan.num_frames)
        gib = 1024**3
        shape = (
            f"{plan.width}x{plan.height} at {plan.num_frames} frames needs about {expected / gib:.0f} GiB, "
            f"and this machine has {physical_bytes / gib:.0f} GiB"
        )
        levers = (
            "Shorten the clip or use a smaller canvas (both reduce the packed sequence, which is what "
            "the peak follows), pass --release-text-encoder, or lower --mlx-cache-limit-gb."
        )
        if expected >= physical_bytes:
            raise MemoryError(f"MiniMax-H3: {shape}, so the run cannot fit. {levers}")
        if expected > physical_bytes * self.REQUEST_BUDGET_SHARE:
            print(
                f"⚠️  MiniMax-H3: {shape}. That leaves little room for anything else resident, and runs "
                f"this close to the limit have been killed by the OS mid-denoise. {levers}",
                file=sys.stderr,
            )

    # Above this share of physical memory a run is close enough to the limit that other resident
    # processes decide the outcome. The 243-frame measurement that was killed sat at 0.73.
    REQUEST_BUDGET_SHARE = 0.72

    # ------------------------------------------------------------------ progress

    def _emit(
        self,
        callback,
        *,
        phase: str,
        plan: H3GenerationPlan,
        step: int,
        seed: int,
        task: str,
        timestep: float | None = None,
    ):
        """Publish one progress event to the caller's callback and to the model's registry.

        Hosts embedding the runtime subscribe to the registry rather than passing a callback, so a
        run that only invoked the callback was invisible to them for its whole duration.
        """
        registry = getattr(self, "callbacks", None)
        if callback is None and registry is None:
            return
        event = ProgressEvent(
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
        if callback is not None:
            callback(event)
        if registry is not None:
            registry.emit_progress(event)
