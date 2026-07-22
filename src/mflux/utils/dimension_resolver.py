import math
from dataclasses import dataclass
from pathlib import Path

from mflux.cli.defaults import defaults as ui_defaults
from mflux.utils.scale_factor import ScaleFactor

# PIL.Image is imported inside the methods that open a reference image: this
# module sits on the task_inference import chain and must not pull the image
# stack (~0.2 s) at import time (0088).

CANVAS_POLICY_SOURCE_ASPECT = "source-aspect"
CANVAS_POLICY_EXACT_RESIZE = "exact-resize"
CANVAS_POLICY_ALIASES = {
    None: CANVAS_POLICY_SOURCE_ASPECT,
    "source": CANVAS_POLICY_SOURCE_ASPECT,
    "source-aspect": CANVAS_POLICY_SOURCE_ASPECT,
    "preserve": CANVAS_POLICY_SOURCE_ASPECT,
    "preserve-aspect": CANVAS_POLICY_SOURCE_ASPECT,
    "exact": CANVAS_POLICY_EXACT_RESIZE,
    "exact-resize": CANVAS_POLICY_EXACT_RESIZE,
}
CANVAS_POLICY_CHOICES = (CANVAS_POLICY_SOURCE_ASPECT, CANVAS_POLICY_EXACT_RESIZE)


@dataclass(frozen=True)
class ResolvedImageDimensions:
    width: int
    height: int
    requested_width: int
    requested_height: int
    source_width: int | None = None
    source_height: int | None = None
    canvas_policy: str = CANVAS_POLICY_EXACT_RESIZE


class DimensionResolver:
    @staticmethod
    def resolve(
        height: int | ScaleFactor,
        width: int | ScaleFactor,
        reference_image_path: Path | str | None = None,
    ) -> tuple[int, int]:
        height_is_scale = isinstance(height, ScaleFactor)
        width_is_scale = isinstance(width, ScaleFactor)

        # If neither dimension uses ScaleFactor, just return as-is
        if not height_is_scale and not width_is_scale:
            return int(width), int(height)

        # ScaleFactor requires a reference image - fall back to defaults if not provided
        if reference_image_path is None:
            resolved_width = ui_defaults.WIDTH if width_is_scale else int(width)
            resolved_height = ui_defaults.HEIGHT if height_is_scale else int(height)
            return resolved_width, resolved_height

        import PIL.Image

        # Open image lazily - PIL.Image.open only reads metadata, not pixel data
        with PIL.Image.open(reference_image_path) as orig_image:
            orig_width, orig_height = orig_image.size

        # Resolve height
        if height_is_scale:
            resolved_height = height.get_scaled_value(orig_height)
        else:
            resolved_height = int(height)

        # Resolve width
        if width_is_scale:
            resolved_width = width.get_scaled_value(orig_width)
        else:
            resolved_width = int(width)

        return resolved_width, resolved_height

    @staticmethod
    def normalize_canvas_policy(canvas_policy: str | None) -> str:
        normalized = CANVAS_POLICY_ALIASES.get(canvas_policy, canvas_policy)
        if normalized not in CANVAS_POLICY_CHOICES:
            choices = ", ".join(CANVAS_POLICY_CHOICES)
            raise ValueError(f"Unsupported canvas policy {canvas_policy!r}. Expected one of: {choices}.")
        return normalized

    @staticmethod
    def resolve_image_canvas(
        height: int | ScaleFactor | None,
        width: int | ScaleFactor | None,
        reference_image_path: Path | str | None = None,
        *,
        canvas_policy: str | None = CANVAS_POLICY_EXACT_RESIZE,
        multiple: int = ui_defaults.DIMENSION_STEP_PIXELS,
    ) -> ResolvedImageDimensions:
        normalized_policy = DimensionResolver.normalize_canvas_policy(canvas_policy)
        if reference_image_path is None or normalized_policy == CANVAS_POLICY_EXACT_RESIZE:
            source_width = None
            source_height = None
            if reference_image_path is not None:
                import PIL.Image

                with PIL.Image.open(reference_image_path) as orig_image:
                    source_width, source_height = orig_image.size
            resolved_width, resolved_height = DimensionResolver.resolve(
                width=ScaleFactor.parse("1x") if width is None else width,
                height=ScaleFactor.parse("1x") if height is None else height,
                reference_image_path=reference_image_path,
            )
            return ResolvedImageDimensions(
                width=resolved_width,
                height=resolved_height,
                requested_width=resolved_width,
                requested_height=resolved_height,
                source_width=source_width,
                source_height=source_height,
                canvas_policy=CANVAS_POLICY_EXACT_RESIZE,
            )

        import PIL.Image

        with PIL.Image.open(reference_image_path) as orig_image:
            source_width, source_height = orig_image.size

        requested_width = DimensionResolver._raw_dimension(width, source_width)
        requested_height = DimensionResolver._raw_dimension(height, source_height)
        width_is_auto = DimensionResolver._is_auto_dimension(width)
        height_is_auto = DimensionResolver._is_auto_dimension(height)
        source_ratio = source_width / source_height

        if width_is_auto and height_is_auto:
            ideal_width = float(source_width)
            ideal_height = float(source_height)
        elif not width_is_auto and height_is_auto:
            ideal_width = requested_width
            ideal_height = ideal_width / source_ratio
        elif width_is_auto and not height_is_auto:
            ideal_height = requested_height
            ideal_width = ideal_height * source_ratio
        else:
            requested_area = requested_width * requested_height
            ideal_height = math.sqrt(requested_area / source_ratio)
            ideal_width = ideal_height * source_ratio

        resolved_width, resolved_height = DimensionResolver._closest_aspect_canvas(
            ideal_width=ideal_width,
            ideal_height=ideal_height,
            source_ratio=source_ratio,
            multiple=multiple,
        )
        return ResolvedImageDimensions(
            width=resolved_width,
            height=resolved_height,
            requested_width=max(1, int(round(requested_width))),
            requested_height=max(1, int(round(requested_height))),
            source_width=source_width,
            source_height=source_height,
            canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
        )

    @staticmethod
    def _is_auto_dimension(value: int | ScaleFactor | None) -> bool:
        if value is None:
            return True
        return isinstance(value, ScaleFactor) and value.value == 1

    @staticmethod
    def _raw_dimension(value: int | ScaleFactor | None, source_dimension: int) -> float:
        if value is None:
            return float(source_dimension)
        if isinstance(value, ScaleFactor):
            return float(value.value * source_dimension)
        return float(value)

    @staticmethod
    def _closest_aspect_canvas(
        *,
        ideal_width: float,
        ideal_height: float,
        source_ratio: float,
        multiple: int,
    ) -> tuple[int, int]:
        if ideal_width <= 0 or ideal_height <= 0:
            raise ValueError("Image width and height must be positive.")
        if multiple <= 0:
            raise ValueError("Dimension multiple must be positive.")

        target_area = ideal_width * ideal_height
        center_h = max(1, int(round(ideal_height / multiple)))
        center_w = max(1, int(round(ideal_width / multiple)))
        span = max(8, int(max(center_h, center_w) * 0.2))
        candidates: set[tuple[int, int]] = set()

        for h_units in range(max(1, center_h - span), center_h + span + 1):
            candidate_height = h_units * multiple
            candidate_width = max(multiple, round(candidate_height * source_ratio / multiple) * multiple)
            candidates.add((int(candidate_width), int(candidate_height)))

        for w_units in range(max(1, center_w - span), center_w + span + 1):
            candidate_width = w_units * multiple
            candidate_height = max(multiple, round(candidate_width / source_ratio / multiple) * multiple)
            candidates.add((int(candidate_width), int(candidate_height)))

        def score(candidate: tuple[int, int]) -> tuple[float, float, float]:
            candidate_width, candidate_height = candidate
            candidate_ratio = candidate_width / candidate_height
            candidate_area = candidate_width * candidate_height
            ratio_error = abs(math.log(candidate_ratio / source_ratio))
            area_error = abs(math.log(candidate_area / target_area))
            shape_error = abs(math.log(candidate_width / ideal_width)) + abs(math.log(candidate_height / ideal_height))
            return area_error, ratio_error, shape_error

        return min(candidates, key=score)
