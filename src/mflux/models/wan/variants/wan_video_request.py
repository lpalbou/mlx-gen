from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mflux.utils.dimension_resolver import DimensionResolver


# eq=False: fields include an mx.array and a dict, so generated equality/hash would raise.
@dataclass(frozen=True, eq=False)
class WanVideoRequest:
    task: str
    is_image_to_video: bool
    is_video_to_video: bool
    height: int
    width: int
    # Mutable by design: generate_video merges source-video metadata into a local copy.
    spatial_metadata: dict
    num_frames: int
    guidance: float
    guidance_2: float | None
    flow_shift: float
    solver: str
    negative_prompt: str
    video_strength: float | None
    video_mask: mx.array | None
    health_check_interval: int | None
    batch_size: int
    canvas_policy: str | None = None
    resize_mode: str = "resize"

    # Executes the generate_video validation/resolution head in its original order, through the
    # model's helpers so existing instance monkeypatches keep working. Must be called at the top
    # of generate_video (validation reads live model state such as released denoisers).
    @staticmethod
    def resolve(
        model,
        *,
        guidance: float | None,
        guidance_2,
        guidance_2_unset_sentinel: object,
        height: int,
        width: int,
        num_frames: int,
        image_path: Path | str | None,
        video_path: Path | str | None,
        video_strength: float | None,
        video_mask_path: Path | str | None,
        flow_shift: float | None,
        solver: str | None,
        negative_prompt: str | None,
        tensor_health_check_interval: int | None,
        canvas_policy: str | None = None,
        resize_mode: str = "resize",
    ) -> "WanVideoRequest":
        health_check_interval = model._validate_tensor_health_check_interval(tensor_health_check_interval)
        if (
            guidance_2 is not guidance_2_unset_sentinel
            and guidance_2 is not None
            and model._wan_config("boundary_ratio", None) is None
        ):
            raise ValueError("guidance_2 is only supported for Wan models with two-transformer boundary routing.")
        is_image_to_video = image_path is not None
        is_video_to_video = video_path is not None
        if is_image_to_video and is_video_to_video:
            raise ValueError("Wan accepts either image_path or video_path, not both.")
        task = "video-to-video" if is_video_to_video else ("image-to-video" if is_image_to_video else "text-to-video")
        if image_path is not None and not model._supports_image_to_video():
            raise ValueError(f"{model.model_config.model_name} does not support image-to-video input.")
        if video_path is not None and not model._supports_video_to_video():
            raise ValueError(f"{model.model_config.model_name} does not support video-to-video input.")
        # Validate the mapping mode up front, before any model work.
        resize_mode = DimensionResolver.normalize_resize_mode(resize_mode)
        resolved_canvas_policy = (
            DimensionResolver.normalize_canvas_policy(canvas_policy) if (image_path or video_path) else None
        )
        model._validate_denoisers_available()
        height, width, spatial_metadata = model._resolve_video_spatial_size(
            height=height,
            width=width,
            image_path=image_path,
            video_path=video_path,
            canvas_policy=canvas_policy,
        )
        num_frames = model._validated_frame_count(num_frames)
        if video_strength is not None and not is_video_to_video:
            raise ValueError("video_strength requires video_path.")
        if video_mask_path is not None and not is_video_to_video:
            raise ValueError("video_mask_path requires video_path.")
        video_strength = model._resolve_video_strength(video_strength) if is_video_to_video else None
        video_mask = (
            model._prepare_video_mask(video_mask_path, height=height, width=width, resize_mode=resize_mode)
            if video_mask_path is not None
            else None
        )
        guidance, guidance_2 = model._resolve_guidance_pair(guidance=guidance, guidance_2=guidance_2)
        model._validate_guidance_values(guidance=guidance, guidance_2=guidance_2)
        flow_shift = model._resolve_flow_shift(flow_shift)
        solver = model._resolve_solver(solver)
        model._validate_video_to_video_solver(is_video_to_video=is_video_to_video, solver=solver)
        negative_prompt = model._resolve_negative_prompt(negative_prompt)
        model._validate_runtime_contract(is_image_to_video=is_image_to_video)
        return WanVideoRequest(
            task=task,
            is_image_to_video=is_image_to_video,
            is_video_to_video=is_video_to_video,
            height=height,
            width=width,
            spatial_metadata=spatial_metadata,
            num_frames=num_frames,
            guidance=guidance,
            guidance_2=guidance_2,
            flow_shift=flow_shift,
            solver=solver,
            negative_prompt=negative_prompt,
            video_strength=video_strength,
            video_mask=video_mask,
            health_check_interval=health_check_interval,
            batch_size=1,
            canvas_policy=resolved_canvas_policy,
            resize_mode=resize_mode,
        )
