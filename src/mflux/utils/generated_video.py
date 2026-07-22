import json
import logging
from collections.abc import Callable, Iterable
from datetime import datetime
from itertools import chain
from pathlib import Path

import mlx.core as mx
import PIL.Image

from mflux.models.common.config import ModelConfig
from mflux.utils.metadata_schema import MetadataSchema
from mflux.utils.runtime_memory import RuntimeMemory
from mflux.utils.version_util import VersionUtil

log = logging.getLogger(__name__)


class GeneratedVideo:
    def __init__(
        self,
        frames: list[PIL.Image.Image] | None,
        fps: int | float,
        model_config: ModelConfig,
        seed: int,
        prompt: str,
        steps: int,
        guidance: float | None,
        precision: mx.Dtype,
        quantization: int,
        generation_time: float,
        height: int,
        width: int,
        task: str = "text-to-video",
        image_path: str | Path | None = None,
        video_path: str | Path | None = None,
        negative_prompt: str | None = None,
        guidance_2: float | None = None,
        flow_shift: float | None = None,
        solver: str | None = None,
        source_width: int | None = None,
        source_height: int | None = None,
        requested_width: int | None = None,
        requested_height: int | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        extra_metadata: dict | None = None,
        frame_batches_factory: Callable[[], Iterable[list[PIL.Image.Image]]] | None = None,
        frame_count: int | None = None,
    ):
        if not frames and frame_batches_factory is None:
            raise ValueError("GeneratedVideo requires frames or a frame batch factory.")
        if frames:
            frame_count = len(frames)
        if frame_count is None or frame_count <= 0:
            raise ValueError("GeneratedVideo requires at least one frame.")
        self._frames = frames
        self._frame_batches_factory = frame_batches_factory
        self._frame_count = frame_count
        self.fps = fps
        self.model_config = model_config
        self.seed = seed
        self.prompt = prompt
        self.steps = steps
        self.guidance = guidance
        self.guidance_2 = guidance_2
        self.flow_shift = flow_shift
        self.solver = solver
        self.precision = precision
        self.quantization = quantization
        self.generation_time = generation_time
        self.height = height
        self.width = width
        self.task = task
        self.image_path = image_path
        self.video_path = video_path
        self.negative_prompt = negative_prompt
        self.source_width = source_width
        self.source_height = source_height
        self.requested_width = requested_width
        self.requested_height = requested_height
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales
        self.extra_metadata = extra_metadata

    @property
    def num_frames(self) -> int:
        return self._frame_count

    @property
    def frames(self) -> list[PIL.Image.Image]:
        if self._frames is None:
            assert self._frame_batches_factory is not None
            self._frames = list(chain.from_iterable(self._frame_batches_factory()))
        return self._frames

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / float(self.fps)

    def save(
        self,
        path: str | Path,
        export_json_metadata: bool = False,
        overwrite: bool = True,
        validate_health: bool = True,
    ) -> Path:
        from mflux.utils.video_util import SourceAudioCopySpec, VideoUtil

        source_audio_copy = None
        if self.task == "video-to-video" and self.video_path is not None:
            source_audio_copy = SourceAudioCopySpec(
                source_video_path=self.video_path,
                clip_start_seconds=0.0,
                clip_duration_seconds=self.duration_seconds,
            )
        metadata = self._get_metadata()
        if not validate_health:
            # Embedded hosts that probe the file themselves can skip the
            # post-save re-decode; the skip is recorded so downstream tooling
            # can tell an unvalidated save from a validated one.
            metadata["health_check"] = "skipped"
        if self._frames is None and self._frame_batches_factory is not None:
            return VideoUtil.save_video_batches(
                frame_batches=self._frame_batches_factory(),
                path=path,
                fps=self.fps,
                metadata=metadata,
                export_json_metadata=export_json_metadata,
                overwrite=overwrite,
                validate_health=validate_health,
                source_audio_copy=source_audio_copy,
            )
        return VideoUtil.save_video(
            frames=self.frames,
            path=path,
            fps=self.fps,
            metadata=metadata,
            export_json_metadata=export_json_metadata,
            overwrite=overwrite,
            validate_health=validate_health,
            source_audio_copy=source_audio_copy,
        )

    def first_frame(self) -> PIL.Image.Image:
        if self._frames is None and self._frame_batches_factory is not None:
            first_batch = next(iter(self._frame_batches_factory()), None)
            if not first_batch:
                raise ValueError("GeneratedVideo frame batch factory returned no frames.")
            return first_batch[0]
        return self.frames[0]

    def _get_metadata(self) -> dict:
        return GeneratedVideo.build_metadata(
            model_config=self.model_config,
            seed=self.seed,
            prompt=self.prompt,
            steps=self.steps,
            guidance=self.guidance,
            guidance_2=self.guidance_2,
            flow_shift=self.flow_shift,
            solver=self.solver,
            precision=self.precision,
            quantization=self.quantization,
            generation_time=self.generation_time,
            height=self.height,
            width=self.width,
            frame_count=self.num_frames,
            fps=self.fps,
            task=self.task,
            image_path=self.image_path,
            video_path=self.video_path,
            negative_prompt=self.negative_prompt,
            source_width=self.source_width,
            source_height=self.source_height,
            requested_width=self.requested_width,
            requested_height=self.requested_height,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            extra_metadata=self.extra_metadata,
        )

    @staticmethod
    def build_metadata(
        *,
        model_config: ModelConfig,
        seed: int,
        prompt: str,
        steps: int,
        guidance: float | None,
        guidance_2: float | None,
        flow_shift: float | None,
        solver: str | None,
        precision: mx.Dtype,
        quantization: int,
        generation_time: float,
        height: int,
        width: int,
        frame_count: int,
        fps: int | float,
        task: str = "text-to-video",
        image_path: str | Path | None = None,
        video_path: str | Path | None = None,
        negative_prompt: str | None = None,
        source_width: int | None = None,
        source_height: int | None = None,
        requested_width: int | None = None,
        requested_height: int | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        extra_metadata: dict | None = None,
    ) -> dict:
        metadata = {
            "metadata_schema_version": MetadataSchema.VERSION,
            "mflux_version": VersionUtil.get_mflux_version(),
            "model": model_config.model_name,
            "base_model": str(model_config.base_model),
            "task": task,
            "seed": seed,
            "steps": steps,
            "guidance": guidance if model_config.supports_guidance else None,
            "guidance_2": guidance_2 if model_config.supports_guidance else None,
            "flow_shift": flow_shift,
            "solver": solver,
            "height": height,
            "width": width,
            "requested_height": requested_height,
            "requested_width": requested_width,
            "source_image_height": source_height,
            "source_image_width": source_width,
            "frames": frame_count,
            "fps": fps,
            "duration_seconds": round(frame_count / float(fps), 3),
            "precision": str(precision),
            "quantize": quantization,
            "generation_time_seconds": round(generation_time, 2),
            "created_at": datetime.now().isoformat(),
            "image_path": str(image_path) if image_path else None,
            "video_path": str(video_path) if video_path else None,
            "prompt": prompt,
            "negative_prompt": negative_prompt if negative_prompt else None,
            "lora_paths": [str(path) for path in lora_paths] if lora_paths else None,
            "lora_scales": [round(scale, 2) for scale in lora_scales] if lora_scales else None,
            "runtime_memory": RuntimeMemory.snapshot("video-metadata").to_metadata(),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    @staticmethod
    def save_metadata(path: str | Path, metadata: dict) -> None:
        metadata_path = Path(path).with_suffix(".metadata.json")
        with open(metadata_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
