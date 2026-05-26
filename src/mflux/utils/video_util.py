import logging
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.common.config import ModelConfig
from mflux.utils.image_util import ImageUtil

log = logging.getLogger(__name__)


class VideoUtil:
    @staticmethod
    def to_video(
        decoded_latents: mx.array,
        fps: int,
        model_config: ModelConfig,
        seed: int,
        prompt: str,
        steps: int,
        guidance: float | None,
        quantization: int,
        generation_time: float,
        task: str = "text-to-video",
        image_path: str | Path | None = None,
        negative_prompt: str | None = None,
    ):
        from mflux.utils.generated_video import GeneratedVideo

        frames = VideoUtil._latents_to_frames(decoded_latents)
        first_frame = frames[0]
        return GeneratedVideo(
            frames=frames,
            fps=fps,
            model_config=model_config,
            seed=seed,
            prompt=prompt,
            steps=steps,
            guidance=guidance,
            precision=ModelConfig.precision,
            quantization=quantization,
            generation_time=generation_time,
            height=first_frame.height,
            width=first_frame.width,
            task=task,
            image_path=image_path,
            negative_prompt=negative_prompt,
        )

    @staticmethod
    def save_video(
        frames: list[PIL.Image.Image],
        path: str | Path,
        fps: int,
        metadata: dict | None = None,
        export_json_metadata: bool = False,
        overwrite: bool = True,
    ) -> Path:
        if not frames:
            raise ValueError("Cannot save a video without frames.")
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")

        file_path = ImageUtil.resolve_output_path(path=path, overwrite=overwrite)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = frames[0].size
        writer = cv2.VideoWriter(
            str(file_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {file_path}")

        try:
            for frame in frames:
                rgb = frame.convert("RGB")
                if rgb.size != (width, height):
                    rgb = rgb.resize((width, height), PIL.Image.Resampling.LANCZOS)
                writer.write(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

        if export_json_metadata and metadata is not None:
            from mflux.utils.generated_video import GeneratedVideo

            GeneratedVideo.save_metadata(file_path, metadata)

        log.info(f"Video saved successfully at: {file_path}")
        return file_path

    @staticmethod
    def extract_frame(path: str | Path, index: int = 0) -> PIL.Image.Image:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video for reading: {path}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {index} from {path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return PIL.Image.fromarray(rgb)
        finally:
            capture.release()

    @staticmethod
    def _latents_to_frames(decoded_latents: mx.array) -> list[PIL.Image.Image]:
        normalized = ImageUtil._denormalize(decoded_latents)
        if normalized.ndim != 5:
            raise ValueError(f"Expected decoded video latents with shape [B, C, F, H, W], got {normalized.shape}")
        video = normalized[0]
        video = mx.transpose(video, (1, 2, 3, 0))
        video = mx.array.astype(video, mx.float32)
        frames_np = np.array(video)
        frames_np = (np.clip(frames_np, 0, 1) * 255).round().astype("uint8")
        return [PIL.Image.fromarray(frame) for frame in frames_np]
