import json
import logging
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import PIL.Image

from mflux.models.common.config import ModelConfig
from mflux.utils.version_util import VersionUtil

log = logging.getLogger(__name__)


class GeneratedVideo:
    def __init__(
        self,
        frames: list[PIL.Image.Image],
        fps: int,
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
    ):
        if not frames:
            raise ValueError("GeneratedVideo requires at least one frame.")
        self.frames = frames
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
        return len(self.frames)

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps

    def save(
        self,
        path: str | Path,
        export_json_metadata: bool = False,
        overwrite: bool = True,
        validate_health: bool = True,
    ) -> Path:
        from mflux.utils.video_util import VideoUtil

        return VideoUtil.save_video(
            frames=self.frames,
            path=path,
            fps=self.fps,
            metadata=self._get_metadata(),
            export_json_metadata=export_json_metadata,
            overwrite=overwrite,
            validate_health=validate_health,
        )

    def first_frame(self) -> PIL.Image.Image:
        return self.frames[0]

    def _get_metadata(self) -> dict:
        metadata = {
            "mflux_version": VersionUtil.get_mflux_version(),
            "model": self.model_config.model_name,
            "base_model": str(self.model_config.base_model),
            "task": self.task,
            "seed": self.seed,
            "steps": self.steps,
            "guidance": self.guidance if self.model_config.supports_guidance else None,
            "guidance_2": self.guidance_2 if self.model_config.supports_guidance else None,
            "flow_shift": self.flow_shift,
            "solver": self.solver,
            "height": self.height,
            "width": self.width,
            "requested_height": self.requested_height,
            "requested_width": self.requested_width,
            "source_image_height": self.source_height,
            "source_image_width": self.source_width,
            "frames": self.num_frames,
            "fps": self.fps,
            "duration_seconds": round(self.duration_seconds, 3),
            "precision": str(self.precision),
            "quantize": self.quantization,
            "generation_time_seconds": round(self.generation_time, 2),
            "created_at": datetime.now().isoformat(),
            "image_path": str(self.image_path) if self.image_path else None,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt if self.negative_prompt else None,
            "lora_paths": [str(path) for path in self.lora_paths] if self.lora_paths else None,
            "lora_scales": [round(scale, 2) for scale in self.lora_scales] if self.lora_scales else None,
        }
        if self.extra_metadata:
            metadata.update(self.extra_metadata)
        return metadata

    @staticmethod
    def save_metadata(path: str | Path, metadata: dict) -> None:
        metadata_path = Path(path).with_suffix(".metadata.json")
        with open(metadata_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
