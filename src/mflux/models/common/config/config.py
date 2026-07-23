import logging
import time
from pathlib import Path

import mlx.core as mx
from tqdm import tqdm

from mflux.models.common.config.inference_defaults import default_inference_steps
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers import SCHEDULER_REGISTRY, try_import_external_scheduler
from mflux.models.common.schedulers.linear_scheduler import LinearScheduler
from mflux.utils.dimension_resolver import (
    CANVAS_POLICY_EXACT_RESIZE,
    CANVAS_POLICY_SOURCE_ASPECT,
    DimensionResolver,
)
from mflux.utils.scale_factor import ScaleFactor

logger = logging.getLogger(__name__)


class Config:
    _progress_enabled = True

    def __init__(
        self,
        model_config: ModelConfig,
        num_inference_steps: int | None = None,
        height: int | ScaleFactor | None = 1024,
        width: int | ScaleFactor | None = 1024,
        guidance: float = 4.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        depth_image_path: Path | str | None = None,
        redux_image_paths: list[Path | str] | None = None,
        redux_image_strengths: list[float] | None = None,
        masked_image_path: Path | str | None = None,
        controlnet_strength: float | None = None,
        scheduler: str = "linear",
        canvas_policy: str = CANVAS_POLICY_EXACT_RESIZE,
        resize_mode: str = "resize",
        preserve_image_aspect_ratio: bool = False,
        dimension_multiple: int = 16,
    ):
        if dimension_multiple <= 0:
            raise ValueError("Dimension multiple must be positive.")
        if num_inference_steps is None:
            num_inference_steps = default_inference_steps(model_config, fallback=4)
        if num_inference_steps < 1:
            raise ValueError("Number of inference steps must be greater than zero.")
        effective_canvas_policy = CANVAS_POLICY_SOURCE_ASPECT if preserve_image_aspect_ratio else canvas_policy
        resolved_dimensions = DimensionResolver.resolve_image_canvas(
            width=ScaleFactor.parse("1x") if width is None else width,
            height=ScaleFactor.parse("1x") if height is None else height,
            reference_image_path=image_path,
            canvas_policy=effective_canvas_policy if image_path is not None else CANVAS_POLICY_EXACT_RESIZE,
        )
        width = resolved_dimensions.width
        height = resolved_dimensions.height

        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive.")

        rounded_width = dimension_multiple * (width // dimension_multiple)
        rounded_height = dimension_multiple * (height // dimension_multiple)
        if rounded_width <= 0 or rounded_height <= 0:
            raise ValueError(f"Width and height must be at least {dimension_multiple}px.")
        if rounded_width != width or rounded_height != height:
            logger.warning("Width and height should be multiples of %s. Rounding down.", dimension_multiple)
        if (
            resolved_dimensions.canvas_policy == CANVAS_POLICY_SOURCE_ASPECT
            and (rounded_width != width or rounded_height != height)
            and image_path is not None
        ):
            resolved_dimensions = DimensionResolver.resolve_image_canvas(
                width=resolved_dimensions.requested_width,
                height=resolved_dimensions.requested_height,
                reference_image_path=image_path,
                canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
                multiple=dimension_multiple,
            )
            width = resolved_dimensions.width
            height = resolved_dimensions.height
            rounded_width = width
            rounded_height = height

        self.model_config = model_config
        self._num_inference_steps = num_inference_steps
        self._height = rounded_height
        self._width = rounded_width
        self._guidance = 0.0 if guidance is None else float(guidance)
        self._image_path = Path(image_path) if isinstance(image_path, str) else image_path
        self._image_strength = image_strength
        self._depth_image_path = Path(depth_image_path) if isinstance(depth_image_path, str) else depth_image_path
        self._redux_image_paths = (
            [Path(p) if isinstance(p, str) else p for p in redux_image_paths] if redux_image_paths else None
        )
        self._redux_image_strengths = redux_image_strengths
        self._masked_image_path = Path(masked_image_path) if isinstance(masked_image_path, str) else masked_image_path
        self._controlnet_strength = controlnet_strength
        self._scheduler_str = scheduler
        self._scheduler = None
        self._time_steps = None
        self._canvas_policy = resolved_dimensions.canvas_policy
        # Orthogonal to canvas_policy: the policy picks the canvas, the mode picks
        # how source pixels map onto it (resize | crop | pad).
        self._resize_mode = DimensionResolver.normalize_resize_mode(resize_mode)
        self._requested_width = resolved_dimensions.requested_width
        self._requested_height = resolved_dimensions.requested_height
        self._source_image_width = resolved_dimensions.source_width
        self._source_image_height = resolved_dimensions.source_height

    @staticmethod
    def set_progress_enabled(enabled: bool) -> None:
        Config._progress_enabled = bool(enabled)

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def canvas_policy(self) -> str:
        return self._canvas_policy

    @property
    def resize_mode(self) -> str:
        return self._resize_mode

    @property
    def requested_width(self) -> int | None:
        return self._requested_width

    @property
    def requested_height(self) -> int | None:
        return self._requested_height

    @property
    def source_image_width(self) -> int | None:
        return self._source_image_width

    @property
    def source_image_height(self) -> int | None:
        return self._source_image_height

    @property
    def image_seq_len(self) -> int:
        return (self._height // 16) * (self._width // 16)

    @property
    def guidance(self) -> float:
        return self._guidance

    @property
    def num_inference_steps(self) -> int:
        return self._num_inference_steps

    @property
    def precision(self) -> mx.Dtype:
        return ModelConfig.precision

    @property
    def num_train_steps(self) -> int:
        return self.model_config.num_train_steps

    @property
    def image_path(self) -> Path | None:
        return self._image_path

    @property
    def image_strength(self) -> float | None:
        return self._image_strength

    @property
    def depth_image_path(self) -> Path | None:
        return self._depth_image_path

    @property
    def redux_image_paths(self) -> list[Path] | None:
        return self._redux_image_paths

    @property
    def redux_image_strengths(self) -> list[float] | None:
        return self._redux_image_strengths

    @property
    def masked_image_path(self) -> Path | None:
        return self._masked_image_path

    @property
    def init_time_step(self) -> int:
        is_img2img = (
            self._image_path is not None and
            self._image_strength is not None and
            self._image_strength > 0.0
        )  # fmt: off

        if is_img2img:
            strength = max(0.0, min(1.0, self._image_strength))  # type: ignore
            denoise_steps = min(self._num_inference_steps, max(1, int(self._num_inference_steps * strength)))
            return self._num_inference_steps - denoise_steps
        else:
            return 0

    @property
    def time_steps(self) -> tqdm:
        if self._time_steps is None:
            steps = range(self.init_time_step, self.num_inference_steps)
            self._time_steps = tqdm(steps) if Config._progress_enabled else _PlainTimeSteps(steps)
        return self._time_steps

    @property
    def controlnet_strength(self) -> float | None:
        return self._controlnet_strength

    @property
    def scheduler(self):
        if self._scheduler is not None:
            return self._scheduler

        if self._scheduler_str == "linear":
            self._scheduler = LinearScheduler(self)
        elif (registered_scheduler := SCHEDULER_REGISTRY.get(self._scheduler_str, None)) is not None:
            self._scheduler = registered_scheduler(self)
        elif "." in self._scheduler_str:
            # this raises ValueError if scheduler is not importable
            scheduler_cls = try_import_external_scheduler(self._scheduler_str)
            self._scheduler = scheduler_cls(self)
        else:
            raise NotImplementedError(f"The scheduler {self._scheduler_str!r} is not implemented by mflux.")

        if hasattr(self._scheduler, "set_image_seq_len") and self.model_config.requires_sigma_shift:
            self._scheduler.set_image_seq_len(self.image_seq_len)

        return self._scheduler


class _PlainTimeSteps:
    def __init__(self, steps: range):
        self._steps = steps
        self._started = time.perf_counter()

    def __iter__(self):
        return iter(self._steps)

    @property
    def format_dict(self) -> dict[str, float]:
        return {"elapsed": time.perf_counter() - self._started}
