import logging
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.common.config import ModelConfig
from mflux.utils.image_util import ImageUtil
from mflux.utils.tensor_health import TensorHealth
from mflux.utils.video_health import VideoHealth

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
        flow_shift: float | None = None,
        solver: str | None = None,
        guidance_2: float | None = None,
        task: str = "text-to-video",
        image_path: str | Path | None = None,
        negative_prompt: str | None = None,
        source_width: int | None = None,
        source_height: int | None = None,
        requested_width: int | None = None,
        requested_height: int | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        extra_metadata: dict | None = None,
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
            flow_shift=flow_shift,
            solver=solver,
            guidance_2=guidance_2,
            precision=ModelConfig.precision,
            quantization=quantization,
            generation_time=generation_time,
            height=first_frame.height,
            width=first_frame.width,
            task=task,
            image_path=image_path,
            negative_prompt=negative_prompt,
            source_width=source_width,
            source_height=source_height,
            requested_width=requested_width,
            requested_height=requested_height,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def save_video(
        frames: list[PIL.Image.Image],
        path: str | Path,
        fps: int,
        metadata: dict | None = None,
        export_json_metadata: bool = False,
        overwrite: bool = True,
        validate_health: bool = True,
    ) -> Path:
        if not frames:
            raise ValueError("Cannot save a video without frames.")
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")

        file_path = ImageUtil.resolve_output_path(path=path, overwrite=overwrite)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = frames[0].size
        frame_health = None
        file_health = None
        if validate_health:
            frame_health = VideoHealth.validate_frames(
                frames,
                fps=fps,
                expected_width=width,
                expected_height=height,
                strict_visual=True,
            )
        VideoUtil._save_video_with_pyav(frames=frames, file_path=file_path, fps=fps, width=width, height=height)
        if validate_health:
            file_health = VideoHealth.validate_file(
                file_path,
                expected_width=width,
                expected_height=height,
                expected_frames=len(frames),
                expected_fps=fps,
                strict_visual=True,
            )

        if metadata is not None and validate_health:
            metadata = dict(metadata)
            metadata["video_health"] = {
                "frames": asdict(frame_health),
                "file": asdict(file_health),
            }

        VideoUtil._save_metadata(
            file_path=file_path,
            metadata=metadata,
            export_json_metadata=export_json_metadata,
        )

        log.info(f"Video saved successfully at: {file_path}")
        return file_path

    @staticmethod
    def extract_frame(path: str | Path, index: int = 0) -> PIL.Image.Image:
        if index < 0:
            raise ValueError("Frame index must be greater than or equal to zero.")

        import av

        with av.open(str(path)) as container:
            if len(container.streams.video) == 0:
                raise RuntimeError(f"Could not find a video stream in {path}")
            video_stream = container.streams.video[0]
            for frame_number, frame in enumerate(container.decode(video_stream)):
                if frame_number == index:
                    return PIL.Image.fromarray(frame.to_ndarray(format="rgb24"))
        raise RuntimeError(f"Could not read frame {index} from {path}")

    @staticmethod
    def _latents_to_frames(decoded_latents: mx.array) -> list[PIL.Image.Image]:
        if decoded_latents.ndim != 5:
            raise ValueError(f"Expected decoded video latents with shape [B, C, F, H, W], got {decoded_latents.shape}")
        video = decoded_latents[0]
        video = mx.transpose(video, (1, 2, 3, 0))
        frames = []
        for frame_index in range(video.shape[0]):
            frame = mx.array.astype(video[frame_index], mx.float32)
            frame_np = np.array(frame)
            TensorHealth.ensure_finite(
                frame_np,
                name="decoded_video_frame",
                phase="video-frame-conversion",
                frame=frame_index + 1,
                total_frames=video.shape[0],
            )
            frame_np = (np.clip(frame_np / 2 + 0.5, 0, 1) * 255).round().astype("uint8")
            frames.append(PIL.Image.fromarray(frame_np))
        return frames

    @staticmethod
    def _save_video_with_pyav(
        frames: list[PIL.Image.Image],
        file_path: Path,
        fps: int,
        width: int,
        height: int,
    ) -> None:
        import av

        with NamedTemporaryFile(
            suffix=file_path.suffix or ".mp4",
            prefix=f".{file_path.stem}-",
            dir=file_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        should_replace = False
        try:
            container = av.open(str(temp_path), mode="w", options={"movflags": "+faststart"})
            stream = container.add_stream("libx264", rate=fps)
            with container:
                stream.width = width
                stream.height = height
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "18", "preset": "medium"}
                for frame in frames:
                    rgb = frame.convert("RGB")
                    if rgb.size != (width, height):
                        rgb = rgb.resize((width, height), PIL.Image.Resampling.LANCZOS)
                    video_frame = av.VideoFrame.from_ndarray(np.array(rgb), format="rgb24")
                    for packet in stream.encode(video_frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            should_replace = True
        finally:
            if should_replace:
                temp_path.replace(file_path)
            elif temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _save_metadata(file_path: Path, metadata: dict | None, export_json_metadata: bool) -> None:
        if export_json_metadata and metadata is not None:
            from mflux.utils.generated_video import GeneratedVideo

            GeneratedVideo.save_metadata(file_path, metadata)
