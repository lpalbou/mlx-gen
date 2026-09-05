from __future__ import annotations

from dataclasses import dataclass, replace

from mflux.lora_validation_registry import LORA_STATUS_UNSUPPORTED, get_lora_validation_status
from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.utils.dimension_resolver import (
    CANVAS_POLICY_EXACT_RESIZE,
    CANVAS_POLICY_SOURCE_ASPECT,
    RESIZE_MODE_CHOICES,
    RESIZE_MODE_RESIZE,
)
from mflux.utils.exceptions import ModelConfigError

TASK_ALIASES = {
    "txt2img": "text-to-image",
    "img2img": "image-to-image",
    "txt2vid": "text-to-video",
    "t2v": "text-to-video",
    "img2vid": "image-to-video",
    "i2v": "image-to-video",
    "vid2vid": "video-to-video",
    "v2v": "video-to-video",
}

TASK_AUTO = "auto"
TEXT_TO_IMAGE = "text-to-image"
IMAGE_TO_IMAGE = "image-to-image"
EDIT = "edit"
TEXT_TO_VIDEO = "text-to-video"
IMAGE_TO_VIDEO = "image-to-video"
VIDEO_TO_VIDEO = "video-to-video"

PUBLIC_IMAGE_TASKS = {TEXT_TO_IMAGE, IMAGE_TO_IMAGE}
PUBLIC_VIDEO_TASKS = {TEXT_TO_VIDEO, IMAGE_TO_VIDEO, VIDEO_TO_VIDEO}
PUBLIC_TASKS = {*PUBLIC_IMAGE_TASKS, *PUBLIC_VIDEO_TASKS}
IMAGE_TASKS = {*PUBLIC_IMAGE_TASKS, EDIT}
VIDEO_TASKS = PUBLIC_VIDEO_TASKS
VALID_TASKS = {TASK_AUTO, EDIT, *PUBLIC_TASKS}
# v5: additive supports_last_image row field (0097), matching the v4 bump
# convention for supports_video_mask.
# v6: additive supports_context_frames row field (0102), same additive-field
# convention: hosts gate multi-frame context conditioning on this field.
# v7: additive supports_svi row field (0103), same additive-field convention:
# hosts gate SVI 2.0 Pro anchor/motion-latent conditioning on this field.
# v8: reference images are a role separate from primary image inputs (ADR 0007).
# v9: restoration routes are a second top-level row array (`restoration`);
# `capabilities` stays empty for restoration-only families, so an empty
# `capabilities` means "not routable through mlxgen generate", never "unsupported".
# v10: additive outpaint conditioning-canvas fields, same additive-field convention: hosts read
# the fill contract instead of inferring it from adapter filenames.
# v11: additive outpaint_preservation row field, same additive-field convention: hosts read how a
# route keeps the source pixels instead of inferring it from the model family.
# v12: additive outpaint pass fields (outpaint_pass_modes, outpaint_default_passes,
# outpaint_auto_split_corner_ratio), same additive-field convention: hosts read whether a route
# splits a two-deep-axis request into single-axis passes, and at what depth, instead of
# discovering the second pass from the run time. Also additive supports_negative_prompt: hosts
# read whether a route takes --negative-prompt instead of learning it from a backend error (the
# FLUX.2 Klein base rows accept it, the distilled rows do not). The `outpaint_preservation` value
# set changed in the same release: every outpaint route now publishes
# `adaptive-content-aware-source-blend`, and `latent-locked-transition-band-no-postblend` is no
# longer emitted.
CAPABILITIES_SCHEMA_VERSION = 12

# Outpaint conditioning-canvas contract. Outpaint quality is decided by which canvas the source
# is pasted onto before denoising, and that used to be inferred from --lora-paths basenames, so a
# host reading `supports_outpaint: true, supports_lora: true` could not tell that omitting the
# adapter silently downgraded the algorithm. These constants publish the coupling.
#
# The concrete names match OutpaintUtil.create_expanded_canvas(fill_mode=...) and the
# --outpaint-fill choices in mflux.cli.parser.parsers; test_task_inference locks them together.
#
# `auto` is the one non-concrete mode: it names the run-time policy rather than a canvas, so the
# shared outpaint layer and the CLI parser both need the literal. It is declared here, next to the
# fill-mode tuples it belongs to, and imported from there.
OUTPAINT_FILL_AUTO = "auto"
FLUX2_OUTPAINT_FILL_MODES = (OUTPAINT_FILL_AUTO, "edge", "neutral", "solid", "blur")
FLUX2_OUTPAINT_DEFAULT_FILL_MODE = OUTPAINT_FILL_AUTO
# `auto` keeps edge fill while every padded side stays within OutpaintUtil.edge_fill_reach() and
# switches to a blank canvas past it. Edge fill stretches a source border strip across the
# padding, so the bound that matters is the stretch factor, not the padding as a fraction of the
# source: the published profile runs 80% single-side padding at 10.9x and is validated.
# Mirrors OutpaintUtil.EDGE_FILL_MAX_STRETCH. Duplicated rather than imported because
# outpaint_util pulls PIL, which test_import_hygiene keeps off the `import mflux` chain.
# test_task_inference locks the two values together.
FLUX2_OUTPAINT_AUTO_EDGE_FILL_MAX_STRETCH = 12.0
# Measured to materially improve the strict-outpaint route; trained on a pure-green canvas, which
# is why `auto` switches to solid green when it is loaded.
FLUX2_OUTPAINT_RECOMMENDED_LORA = "fal/flux-2-klein-4B-outpaint-lora"
# How a route keeps the source pixels. Every outpaint route now does the same two things: it holds
# the source region in latent space behind a narrow transition band while the added area is
# denoised (the FLUX.2 route through its own lock, the Qwen route through its masked-edit input),
# then pastes the original crop back over the decoded result while the generated source window
# still matches it. The latent lock alone cannot keep the source pixel-exact - the VAE decoder
# couples the whole canvas - so the paste is what returns the original pixels, and it is safe
# because the lock rules out the recomposition the paste would otherwise have to guard against.
# One strategy, one published value; a host that reads this field knows which guarantee it gets
# without loading weights, and it is the same string recorded in generated metadata as
# `outpaint_preservation`. Until 0.33.0 the FLUX.2 rows published
# `latent-locked-transition-band-no-postblend` and never pasted.
OUTPAINT_PRESERVATION_ADAPTIVE_SOURCE_BLEND = "adaptive-content-aware-source-blend"
FLUX2_OUTPAINT_PRESERVATION = OUTPAINT_PRESERVATION_ADAPTIVE_SOURCE_BLEND
QWEN_OUTPAINT_PRESERVATION = OUTPAINT_PRESERVATION_ADAPTIVE_SOURCE_BLEND
# The Qwen edit backend has no --outpaint-fill option and always builds an edge-extended canvas.
QWEN_OUTPAINT_FILL_MODES = ("edge",)
QWEN_OUTPAINT_DEFAULT_FILL_MODE = "edge"
# Release-validation envelope. Every recorded outpaint validation run in this repository (profile
# reframe_outpaint_2026_06_08 and the FLUX.2 Klein base starship profile) uses this one padding
# value on a single 432x240 source, producing a 1040x272 canvas, with edge fill and no adapter.
# Outside this envelope there is no recorded evidence for any fill mode.
OUTPAINT_VALIDATED_PADDING = "5%,80%,5%,60%"
OUTPAINT_VALIDATED_FILL_MODE = "edge"
OUTPAINT_VALIDATED_MAX_CANVAS_PIXELS = 282880
# --outpaint-passes contract. A request that pads a vertical side and a horizontal side deeply
# opens a free corner - new canvas sharing neither a row nor a column with the source - and on the
# routes measured so far the model paints a second copy of the subject into it. Two single-axis
# passes reach the same canvas with no free corner at either pass. `auto` splits when the corner
# is deep enough; "1" and "2" name the count explicitly. The literals are declared here, next to
# the fill-mode tuples, for the same reason `auto` is: the shared outpaint layer and the CLI
# parser both need them without importing PIL.
OUTPAINT_PASSES_AUTO = "auto"
OUTPAINT_PASS_MODES = (OUTPAINT_PASSES_AUTO, "1", "2")
OUTPAINT_DEFAULT_PASSES = OUTPAINT_PASSES_AUTO
# The depth past which `auto` splits: the shallower of the two axis depths, each measured against
# the source dimension it grows along, has to exceed this fraction. Zero on any single-axis request,
# so those never split. Published per route as `outpaint_auto_split_corner_ratio`; None on a route
# that never splits.
#
# Set from recorded runs on the FLUX.2 Klein route with a subject-bearing 432x240 source, 16 steps,
# three seeds per geometry (docs/assets/validation/outpaint-corner-sweep-2026-09-02): see the
# calibration note next to the FLUX.2 row for where the clean and the duplicating runs sit.
OUTPAINT_TWO_DEEP_AXIS_RATIO = 0.30
QWEN_CONTROL_UNION_MODEL = "InstantX/Qwen-Image-ControlNet-Union:diffusion_pytorch_model.safetensors"
QWEN_CONTROL_INPAINT_MODEL = "InstantX/Qwen-Image-ControlNet-Inpainting:diffusion_pytorch_model.safetensors"
# Untrusted inferred identities that earned native masked edit through an exact smoke proof.
QWEN_BASE_NATIVE_INPAINT_EXACT_ROWS = frozenset(
    {
        "AbstractFramework/qwen-image-2512-8bit",
    }
)

I2I_MODE_AUTO = "auto"
MODE_TEXT_ONLY = "text-only"
MODE_LATENT_IMG2IMG = "latent-img2img"
MODE_EDIT_REFERENCE = "edit-reference"
MODE_MULTI_REFERENCE = "multi-reference"
MODE_TEXT_VIDEO = "text-video"
MODE_FIRST_FRAME_I2V = "first-frame-i2v"
MODE_LATENT_VIDEO = "latent-video"
MODE_REFERENCE_VIDEO = "reference-video"
MODE_REFERENCE_VIDEO_EDIT = "reference-video-edit"
# Restoration modes. Deliberately absent from PUBLIC_TASKS/VALID_TASKS: restoration
# reuses the existing public task strings (image-to-image, video-to-video) that the
# routes already emit as progress labels, and it is never routable through
# `mlxgen generate` (ADR 0006 keeps the prompt-driven and promptless surfaces apart).
MODE_RESTORE_IMAGE = "restore-image"
MODE_RESTORE_VIDEO = "restore-video"

# SeedVR2 temporal-chunk request defaults. These mirror the --temporal-chunk-size /
# --temporal-chunk-overlap defaults declared in
# mflux/cli/parser/parsers.py::add_seedvr2_upscale_arguments. They are duplicated here
# only because the parser owns the literals today; hoisting them into SeedVR2Util (next
# to VIDEO_MIN_PRODUCTION_STREAMING_*) and reading them from there is the follow-up that
# removes this second copy.
SEEDVR2_DEFAULT_TEMPORAL_CHUNK_SIZE = 49
SEEDVR2_DEFAULT_TEMPORAL_CHUNK_OVERLAP = 16

I2I_MODE_ALIASES = {
    None: I2I_MODE_AUTO,
    I2I_MODE_AUTO: I2I_MODE_AUTO,
    "latent": MODE_LATENT_IMG2IMG,
    "img2img": MODE_LATENT_IMG2IMG,
    MODE_LATENT_IMG2IMG: MODE_LATENT_IMG2IMG,
    "edit": MODE_EDIT_REFERENCE,
    "edit-conditioned": MODE_EDIT_REFERENCE,
    MODE_EDIT_REFERENCE: MODE_EDIT_REFERENCE,
    "reference": MODE_EDIT_REFERENCE,
    "multi": MODE_MULTI_REFERENCE,
    "multi-reference": MODE_MULTI_REFERENCE,
    MODE_MULTI_REFERENCE: MODE_MULTI_REFERENCE,
}


class TaskInferenceError(ValueError):
    """Raised when model capabilities and requested image inputs cannot resolve to one plan."""


@dataclass(frozen=True)
class GenerationCapability:
    id: str
    public_task: str
    mode: str
    handler_id: str
    min_images: int = 0
    max_images: int | None = 0
    min_videos: int = 0
    max_videos: int | None = 0
    supports_image_strength: bool = False
    supports_video_strength: bool = False
    supports_video_mask: bool = False
    # Wan A14B i2v first+last bracket conditioning (--last-image, 0097).
    supports_last_image: bool = False
    # Wan A14B i2v multi-frame context head conditioning (--context-frames, 0102).
    supports_context_frames: bool = False
    # Wan A14B i2v SVI 2.0 Pro anchor/motion-latent conditioning (--svi-anchor-image, 0103).
    supports_svi: bool = False
    # Whether the route takes --negative-prompt / --negative and uses it (classifier-free
    # guidance against it). False where the backend rejects the option or the weights run no
    # guidance branch: Bonsai, and FLUX.2 Klein distilled weights, which are step-distilled.
    supports_negative_prompt: bool = False
    supports_mask: bool = False
    supports_control_image: bool = False
    supports_control_mask: bool = False
    supports_outpaint: bool = False
    # Whether the route accepts the explicit --outpaint-fill / --outpaint-fill-color options.
    # False with a single-entry outpaint_fill_modes means the fill algorithm is fixed.
    supports_outpaint_fill: bool = False
    # Conditioning-canvas fill algorithms the route accepts. All-empty/None on routes where
    # supports_outpaint is False.
    outpaint_fill_modes: tuple[str, ...] = ()
    # Fill mode that runs when --outpaint-fill is omitted.
    outpaint_default_fill_mode: str | None = None
    # With outpaint_default_fill_mode "auto": the largest bicubic stretch edge fill is allowed to
    # apply to the sampled border strip. Padding deeper than strip * this factor switches auto to
    # a blank canvas. None when the route has no auto policy.
    outpaint_auto_edge_fill_max_stretch: float | None = None
    # Adapter measured to give the best outpaint results on this route. Optional, not required:
    # supports_lora stays the authority on whether LoRA loads at all.
    outpaint_recommended_lora: str | None = None
    # How the route keeps the source pixels of an outpaint run, and the value recorded in the
    # generated artifact's `outpaint_preservation` metadata. None when supports_outpaint is False.
    outpaint_preservation: str | None = None
    # Padding envelope that release validation actually covers, and the fill mode those recorded
    # runs used. Outside this envelope outpaint is supported but unvalidated.
    outpaint_validated_padding: str | None = None
    outpaint_validated_fill_mode: str | None = None
    outpaint_validated_max_canvas_pixels: int | None = None
    # --outpaint-passes values the route accepts, and the one that runs when it is omitted. Empty
    # and None on routes where supports_outpaint is False.
    outpaint_pass_modes: tuple[str, ...] = ()
    outpaint_default_passes: str | None = None
    # With outpaint_default_passes "auto": a request whose shallower axis depth (deepest top or
    # bottom padding over the source height, deepest left or right padding over the source width,
    # the smaller of the two) exceeds this fraction runs as two single-axis passes. None when the
    # route never splits, so a host can predict a second pass before starting the job.
    outpaint_auto_split_corner_ratio: float | None = None
    supports_reframe: bool = False
    supports_lora: bool = False
    control_model: str | None = None
    lora_status: str = "unsupported"
    lora_target_roles: tuple[str, ...] = ()
    lora_validation_profile: str | None = None
    supports_frames: bool = False
    supports_fps: bool = False
    default_for_task: bool = False
    model_override: str | None = None
    canvas_policies: tuple[str, ...] = ()
    default_canvas_policy: str | None = None
    # Source-to-canvas mapping modes the route's handler actually accepts
    # (--resize-mode). Empty on routes with reference-pinned geometry
    # (edit/reference, controlnet, outpaint) and on text-only routes.
    resize_modes: tuple[str, ...] = ()
    primary_image_index: int | None = None
    dimension_multiple: int | None = None
    min_reference_images: int = 0
    max_reference_images: int | None = 0

    def allows_image_count(self, image_count: int) -> bool:
        if image_count < self.min_images:
            return False
        return self.max_images is None or image_count <= self.max_images

    def allows_video_count(self, video_count: int) -> bool:
        if video_count < self.min_videos:
            return False
        return self.max_videos is None or video_count <= self.max_videos

    def allows_reference_image_count(self, reference_image_count: int) -> bool:
        if reference_image_count < self.min_reference_images:
            return False
        return self.max_reference_images is None or reference_image_count <= self.max_reference_images

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "public_task": self.public_task,
            "mode": self.mode,
            "handler_id": self.handler_id,
            "min_images": self.min_images,
            "max_images": self.max_images,
            "min_videos": self.min_videos,
            "max_videos": self.max_videos,
            "min_reference_images": self.min_reference_images,
            "max_reference_images": self.max_reference_images,
            "supports_image_strength": self.supports_image_strength,
            "supports_video_strength": self.supports_video_strength,
            "supports_video_mask": self.supports_video_mask,
            "supports_last_image": self.supports_last_image,
            "supports_context_frames": self.supports_context_frames,
            "supports_svi": self.supports_svi,
            "supports_mask": self.supports_mask,
            "supports_control_image": self.supports_control_image,
            "supports_control_mask": self.supports_control_mask,
            "supports_negative_prompt": self.supports_negative_prompt,
            "supports_outpaint": self.supports_outpaint,
            "supports_outpaint_fill": self.supports_outpaint_fill,
            "outpaint_fill_modes": list(self.outpaint_fill_modes),
            "outpaint_default_fill_mode": self.outpaint_default_fill_mode,
            "outpaint_auto_edge_fill_max_stretch": self.outpaint_auto_edge_fill_max_stretch,
            "outpaint_recommended_lora": self.outpaint_recommended_lora,
            "outpaint_preservation": self.outpaint_preservation,
            "outpaint_validated_padding": self.outpaint_validated_padding,
            "outpaint_validated_fill_mode": self.outpaint_validated_fill_mode,
            "outpaint_validated_max_canvas_pixels": self.outpaint_validated_max_canvas_pixels,
            "outpaint_pass_modes": list(self.outpaint_pass_modes),
            "outpaint_default_passes": self.outpaint_default_passes,
            "outpaint_auto_split_corner_ratio": self.outpaint_auto_split_corner_ratio,
            "supports_reframe": self.supports_reframe,
            "supports_lora": self.supports_lora,
            "control_model": self.control_model,
            "lora_status": self.lora_status,
            "lora_target_roles": list(self.lora_target_roles),
            "lora_validation_profile": self.lora_validation_profile,
            "supports_frames": self.supports_frames,
            "supports_fps": self.supports_fps,
            "default_for_task": self.default_for_task,
            "model_override": self.model_override,
            "canvas_policies": list(self.canvas_policies),
            "default_canvas_policy": self.default_canvas_policy,
            "resize_modes": list(self.resize_modes),
            "primary_image_index": self.primary_image_index,
            "dimension_multiple": self.dimension_multiple,
        }


@dataclass(frozen=True)
class RestorationCapability:
    """One promptless restoration route: what it accepts, and what it refuses.

    A separate row type from :class:`GenerationCapability` on purpose. That dataclass
    carries ~32 generation-only fields (image strength, masks, controlnet, outpaint,
    reframe, LoRA profiles, SVI anchors, canvas policies) which are all meaningless for a
    promptless restorer, and ``resolve_generation_plan`` filters over that exact type.
    Only the two proven cardinality predicates are mirrored, byte-for-byte in semantics,
    so consumer code that already gates on them keeps working.

    Every field here earns its place by one of two rules: some route or CLI currently
    raises on the corresponding option, or the value is a hard numeric limit a caller
    cannot discover without a failed preflight.
    """

    id: str
    public_task: str
    mode: str
    handler_id: str
    command: str = "mlxgen upscale"
    # Cardinality.
    min_images: int = 0
    max_images: int | None = 0
    min_videos: int = 0
    max_videos: int | None = 0
    # Geometry.
    supports_scaling: bool = False
    # None means "any positive factor accepted"; a tuple pins the accepted set.
    scale_factors: tuple[str, ...] | None = None
    supports_short_side_resolution: bool = False
    default_resolution: str = "1x"
    dimension_multiple: int | None = None
    max_canvas_pixels: int | None = None
    # Precision.
    supports_quantization: bool = False
    quantization_bits: tuple[int, ...] = ()
    weight_precision: str | None = None
    # Sampling.
    supports_steps: bool = False
    steps_min: int | None = None
    steps_max: int | None = None
    supports_multi_seed: bool = False
    # Clip window and source frame protocol.
    supports_clip_window: bool = False
    min_frames: int | None = None
    max_frames: int | None = None
    frame_multiple: int | None = None
    frame_remainder: int | None = None
    # Chunking.
    chunk_strategy: str | None = None
    chunk_options_user_settable: bool = False
    chunk_size_default: int | None = None
    chunk_size_min: int | None = None
    chunk_size_multiple: int | None = None
    chunk_size_remainder: int | None = None
    chunk_overlap_default: int | None = None
    chunk_overlap_min: int | None = None
    chunk_overlap_multiple: int | None = None
    # Post-processing and IO.
    supports_softness: bool = False
    supports_vae_tiling: bool = False
    color_correction_modes: tuple[str, ...] = ()
    default_color_correction: str | None = None
    supports_audio_passthrough: bool = False

    @property
    def accepted_media(self) -> tuple[str, ...]:
        """Media kinds this row accepts, derived rather than stored so it cannot drift."""
        media = []
        if self.max_images is None or self.max_images > 0:
            media.append("image")
        if self.max_videos is None or self.max_videos > 0:
            media.append("video")
        return tuple(media)

    def allows_image_count(self, image_count: int) -> bool:
        if image_count < self.min_images:
            return False
        return self.max_images is None or image_count <= self.max_images

    def allows_video_count(self, video_count: int) -> bool:
        if video_count < self.min_videos:
            return False
        return self.max_videos is None or video_count <= self.max_videos

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "public_task": self.public_task,
            "mode": self.mode,
            "handler_id": self.handler_id,
            "command": self.command,
            "accepted_media": list(self.accepted_media),
            "min_images": self.min_images,
            "max_images": self.max_images,
            "min_videos": self.min_videos,
            "max_videos": self.max_videos,
            "supports_scaling": self.supports_scaling,
            "scale_factors": None if self.scale_factors is None else list(self.scale_factors),
            "supports_short_side_resolution": self.supports_short_side_resolution,
            "default_resolution": self.default_resolution,
            "dimension_multiple": self.dimension_multiple,
            "max_canvas_pixels": self.max_canvas_pixels,
            "supports_quantization": self.supports_quantization,
            "quantization_bits": list(self.quantization_bits),
            "weight_precision": self.weight_precision,
            "supports_steps": self.supports_steps,
            "steps_min": self.steps_min,
            "steps_max": self.steps_max,
            "supports_multi_seed": self.supports_multi_seed,
            "supports_clip_window": self.supports_clip_window,
            "min_frames": self.min_frames,
            "max_frames": self.max_frames,
            "frame_multiple": self.frame_multiple,
            "frame_remainder": self.frame_remainder,
            "chunk_strategy": self.chunk_strategy,
            "chunk_options_user_settable": self.chunk_options_user_settable,
            "chunk_size_default": self.chunk_size_default,
            "chunk_size_min": self.chunk_size_min,
            "chunk_size_multiple": self.chunk_size_multiple,
            "chunk_size_remainder": self.chunk_size_remainder,
            "chunk_overlap_default": self.chunk_overlap_default,
            "chunk_overlap_min": self.chunk_overlap_min,
            "chunk_overlap_multiple": self.chunk_overlap_multiple,
            "supports_softness": self.supports_softness,
            "supports_vae_tiling": self.supports_vae_tiling,
            "color_correction_modes": list(self.color_correction_modes),
            "default_color_correction": self.default_color_correction,
            "supports_audio_passthrough": self.supports_audio_passthrough,
        }


@dataclass(frozen=True)
class ModelCapabilities:
    schema_version: int
    family: str
    label: str
    model_name: str | None
    capabilities: tuple[GenerationCapability, ...]
    # Promptless restoration routes (`mlxgen upscale`). Empty for every generation
    # family; `capabilities` is empty for every restoration-only family. The two arrays
    # are disjoint by design and neither emptiness means "unsupported model".
    restoration: tuple[RestorationCapability, ...] = ()

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "label": self.label,
            "model_name": self.model_name,
            "capabilities": [capability.to_dict() for capability in self.capabilities],
            "restoration": [capability.to_dict() for capability in self.restoration],
        }


@dataclass(frozen=True)
class GenerationPlan:
    public_task: str
    mode: str
    capability_id: str
    family: str
    handler_id: str
    image_count: int
    video_count: int = 0
    model_name: str | None = None
    model_override: str | None = None
    canvas_policies: tuple[str, ...] = ()
    default_canvas_policy: str | None = None
    resize_modes: tuple[str, ...] = ()
    primary_image_index: int | None = None
    dimension_multiple: int | None = None
    supports_lora: bool = False
    control_model: str | None = None
    lora_status: str = "unsupported"
    lora_target_roles: tuple[str, ...] = ()
    lora_validation_profile: str | None = None
    reference_image_count: int = 0

    @property
    def task(self) -> str:
        return self.public_task

    def to_dict(self) -> dict:
        return {
            "public_task": self.public_task,
            "task": self.public_task,
            "mode": self.mode,
            "capability_id": self.capability_id,
            "family": self.family,
            "handler_id": self.handler_id,
            "image_count": self.image_count,
            "video_count": self.video_count,
            "reference_image_count": self.reference_image_count,
            "model_name": self.model_name,
            "model_override": self.model_override,
            "canvas_policies": list(self.canvas_policies),
            "default_canvas_policy": self.default_canvas_policy,
            "resize_modes": list(self.resize_modes),
            "primary_image_index": self.primary_image_index,
            "dimension_multiple": self.dimension_multiple,
            "supports_lora": self.supports_lora,
            "control_model": self.control_model,
            "lora_status": self.lora_status,
            "lora_target_roles": list(self.lora_target_roles),
            "lora_validation_profile": self.lora_validation_profile,
        }


@dataclass(frozen=True)
class ResolvedTask:
    task: str
    family: str
    image_count: int
    video_count: int = 0
    model_name: str | None = None
    mode: str | None = None
    capability_id: str | None = None
    handler_id: str | None = None
    reference_image_count: int = 0


@dataclass(frozen=True)
class _ModelIdentity:
    model_config: ModelConfig | None
    aliases: set[str]
    model_name: str | None
    model_key: str
    family: str
    identity_source: str


def normalize_task(task: str | None) -> str:
    normalized = TASK_AUTO if task is None else task
    normalized = TASK_ALIASES.get(normalized, normalized)
    if normalized not in VALID_TASKS:
        valid_tasks = ", ".join(sorted(VALID_TASKS))
        raise TaskInferenceError(f"Unsupported task {task!r}. Expected one of: {valid_tasks}.")
    return normalized


def normalize_i2i_mode(i2i_mode: str | None) -> str:
    normalized = I2I_MODE_ALIASES.get(i2i_mode, i2i_mode)
    if normalized not in {I2I_MODE_AUTO, MODE_LATENT_IMG2IMG, MODE_EDIT_REFERENCE, MODE_MULTI_REFERENCE}:
        valid_modes = ", ".join(["auto", "latent", "edit", "multi-reference"])
        raise TaskInferenceError(f"Unsupported image-to-image mode {i2i_mode!r}. Expected one of: {valid_modes}.")
    return normalized


def get_model_capabilities(
    *,
    model: str | None = None,
    model_config: ModelConfig | None = None,
    family: str | None = None,
    base_model: str | None = None,
) -> ModelCapabilities:
    identity = _resolve_model_identity(model=model, model_config=model_config, family=family, base_model=base_model)
    return _capabilities_for(identity)


def resolve_generation_plan(
    *,
    model: str | None = None,
    model_config: ModelConfig | None = None,
    family: str | None = None,
    base_model: str | None = None,
    image_count: int = 0,
    video_count: int = 0,
    reference_image_count: int = 0,
    task: str | None = TASK_AUTO,
    i2i_mode: str | None = I2I_MODE_AUTO,
    has_image_strength: bool = False,
    has_video_strength: bool = False,
    has_video_mask: bool = False,
    has_mask: bool = False,
    has_control_image: bool = False,
    has_outpaint: bool = False,
    has_reframe: bool = False,
    has_lora: bool = False,
) -> GenerationPlan:
    if image_count < 0:
        raise TaskInferenceError("image_count must be greater than or equal to zero.")
    if video_count < 0:
        raise TaskInferenceError("video_count must be greater than or equal to zero.")
    if reference_image_count < 0:
        raise TaskInferenceError("reference_image_count must be greater than or equal to zero.")
    if image_count > 0 and video_count > 0:
        raise TaskInferenceError(
            "mlxgen generate accepts either input images or input videos for one request, not both."
        )
    if has_image_strength and image_count == 0:
        raise TaskInferenceError("--image-strength requires --image or --image-path.")
    if has_video_strength and video_count == 0:
        raise TaskInferenceError("--video-strength requires --video or --video-path.")
    if has_video_mask and video_count == 0:
        raise TaskInferenceError("--video-mask-path requires --video or --video-path.")
    if has_video_mask and has_mask:
        raise TaskInferenceError("--video-mask-path cannot be combined with --mask-path.")
    if has_mask and image_count == 0:
        raise TaskInferenceError("--mask-path requires --image or --image-path.")
    if has_mask and has_image_strength:
        raise TaskInferenceError(
            "--image-strength cannot be combined with --mask-path; masked inpaint is a separate route from latent image-to-image."
        )
    if has_video_strength and image_count > 0:
        raise TaskInferenceError("--video-strength can only be used with --video or --video-path.")
    if has_mask and video_count > 0:
        raise TaskInferenceError("--mask-path is only supported for image inputs, not source videos.")
    if has_control_image and image_count > 0:
        raise TaskInferenceError(
            "--controlnet-image-path currently targets text-to-image structured control and cannot be combined "
            "with --image or --image-path."
        )
    if has_control_image and video_count > 0:
        raise TaskInferenceError(
            "--controlnet-image-path is only supported for image generation routes, not source videos."
        )
    if has_outpaint and image_count == 0:
        raise TaskInferenceError("--outpaint-padding requires --image or --image-path.")
    if has_outpaint and video_count > 0:
        raise TaskInferenceError("--outpaint-padding is only supported for image edit routes, not source videos.")
    if has_reframe and image_count == 0:
        raise TaskInferenceError("--reframe-padding requires --image or --image-path.")
    if has_reframe and video_count > 0:
        raise TaskInferenceError("--reframe-padding is only supported for image edit routes, not source videos.")

    if model_config is None and model is not None and _is_unsupported_flux2_dev_model(model):
        raise TaskInferenceError(
            "black-forest-labs/FLUX.2-dev is not supported by the current MLX-Gen FLUX.2 runtime. "
            "Use a supported FLUX.2 Klein model, or add a first-class FLUX.2-dev model config and "
            "weight mapping before using FLUX.2-dev LoRAs."
        )

    normalized_task = normalize_task(task)
    normalized_i2i_mode = normalize_i2i_mode(i2i_mode)
    identity = _resolve_model_identity(model=model, model_config=model_config, family=family, base_model=base_model)
    model_capabilities = _capabilities_for(identity)
    if not model_capabilities.capabilities:
        raise TaskInferenceError(
            f"{model_capabilities.label} does not expose unified generation capabilities through mlxgen generate."
        )

    public_task = _requested_public_task(
        model_capabilities=model_capabilities,
        task=normalized_task,
        image_count=image_count,
        video_count=video_count,
    )
    requested_mode = _requested_mode(
        task=normalized_task,
        public_task=public_task,
        image_count=image_count,
        i2i_mode=normalized_i2i_mode,
        has_image_strength=has_image_strength,
    )

    candidates = [
        capability
        for capability in model_capabilities.capabilities
        if (
            capability.public_task == public_task
            and capability.allows_image_count(image_count)
            and capability.allows_video_count(video_count)
            and capability.allows_reference_image_count(reference_image_count)
        )
    ]
    if requested_mode != I2I_MODE_AUTO:
        candidates = [capability for capability in candidates if _mode_matches_request(capability.mode, requested_mode)]

    if has_image_strength:
        candidates = [capability for capability in candidates if capability.supports_image_strength]
        if not candidates:
            raise TaskInferenceError("--image-strength is only supported for latent image-to-image mode.")
    if has_video_strength:
        candidates = [capability for capability in candidates if capability.supports_video_strength]
        if not candidates:
            raise TaskInferenceError(
                "--video-strength is not supported by this model's video-to-video route. "
                "Wan VACE models condition through --video-mask-path/--reference-image/--conditioning-scale "
                "instead of an SDEdit strength; for strength-based edits use Wan2.2-T2V-A14B."
            )
    if has_video_mask:
        candidates = [capability for capability in candidates if capability.supports_video_mask]
        if not candidates:
            raise TaskInferenceError("--video-mask-path is only supported for video-to-video routes with mask support.")
    if has_mask:
        candidates = [capability for capability in candidates if capability.supports_mask]
        if not candidates:
            raise TaskInferenceError("--mask-path is only supported for image-to-image modes with mask support.")
    if has_control_image:
        candidates = [capability for capability in candidates if capability.supports_control_image]
        if not candidates:
            raise TaskInferenceError(
                "--controlnet-image-path is only supported for structured-control modes with control image support."
            )
    if has_outpaint:
        candidates = [capability for capability in candidates if capability.supports_outpaint]
        if not candidates:
            raise TaskInferenceError(
                "--outpaint-padding is only supported for image-to-image modes with outpaint support."
            )
    if has_reframe:
        candidates = [capability for capability in candidates if capability.supports_reframe]
        if not candidates:
            raise TaskInferenceError(
                "--reframe-padding is only supported for image-to-image edit models with generative reframe support."
            )
    if has_lora:
        candidates = [capability for capability in candidates if capability.supports_lora]
        if not candidates:
            raise TaskInferenceError(
                "--lora-paths/--lora-scales are only supported for model families and task modes "
                "with an MLX-Gen LoRA mapping."
            )

    capability = _select_capability(
        model_capabilities=model_capabilities,
        public_task=public_task,
        requested_mode=requested_mode,
        image_count=image_count,
        video_count=video_count,
        reference_image_count=reference_image_count,
        candidates=candidates,
    )
    if (
        public_task == IMAGE_TO_IMAGE
        and capability.mode == MODE_LATENT_IMG2IMG
        and image_count > 0
        and not has_image_strength
    ):
        raise TaskInferenceError("--image-strength is required for latent image-to-image mode.")
    # Masked routes cannot run without a mask; fail here instead of after model load.
    if capability.supports_mask and not has_mask:
        raise TaskInferenceError(
            f"--mask-path is required for the masked edit / inpaint route {capability.id}. "
            "Pass --mask-path, or select a different image-to-image mode."
        )

    return GenerationPlan(
        public_task=capability.public_task,
        mode=capability.mode,
        capability_id=capability.id,
        family=model_capabilities.family,
        handler_id=capability.handler_id,
        image_count=image_count,
        video_count=video_count,
        reference_image_count=reference_image_count,
        model_name=model_capabilities.model_name,
        model_override=capability.model_override,
        canvas_policies=capability.canvas_policies,
        default_canvas_policy=capability.default_canvas_policy,
        resize_modes=capability.resize_modes,
        primary_image_index=capability.primary_image_index,
        dimension_multiple=capability.dimension_multiple,
        supports_lora=capability.supports_lora,
        control_model=capability.control_model,
        lora_status=capability.lora_status,
        lora_target_roles=capability.lora_target_roles,
        lora_validation_profile=capability.lora_validation_profile,
    )


def resolve_task(
    *,
    model: str | None = None,
    model_config: ModelConfig | None = None,
    family: str | None = None,
    base_model: str | None = None,
    image_count: int = 0,
    video_count: int = 0,
    reference_image_count: int = 0,
    task: str | None = TASK_AUTO,
    i2i_mode: str | None = I2I_MODE_AUTO,
    has_image_strength: bool = False,
    has_video_strength: bool = False,
    has_video_mask: bool = False,
    has_mask: bool = False,
    has_control_image: bool = False,
    has_outpaint: bool = False,
    has_reframe: bool = False,
    has_lora: bool = False,
) -> ResolvedTask:
    plan = resolve_generation_plan(
        model=model,
        model_config=model_config,
        family=family,
        base_model=base_model,
        image_count=image_count,
        video_count=video_count,
        reference_image_count=reference_image_count,
        task=task,
        i2i_mode=i2i_mode,
        has_image_strength=has_image_strength,
        has_video_strength=has_video_strength,
        has_video_mask=has_video_mask,
        has_mask=has_mask,
        has_control_image=has_control_image,
        has_outpaint=has_outpaint,
        has_reframe=has_reframe,
        has_lora=has_lora,
    )
    return ResolvedTask(
        task=plan.public_task,
        family=plan.family,
        image_count=plan.image_count,
        video_count=plan.video_count,
        reference_image_count=plan.reference_image_count,
        model_name=plan.model_name,
        mode=plan.mode,
        capability_id=plan.capability_id,
        handler_id=plan.handler_id,
    )


def infer_task(
    *,
    model: str | None = None,
    model_config: ModelConfig | None = None,
    family: str | None = None,
    base_model: str | None = None,
    image_count: int = 0,
    video_count: int = 0,
    reference_image_count: int = 0,
    task: str | None = TASK_AUTO,
    i2i_mode: str | None = I2I_MODE_AUTO,
    has_image_strength: bool = False,
    has_video_strength: bool = False,
    has_video_mask: bool = False,
    has_mask: bool = False,
    has_control_image: bool = False,
    has_outpaint: bool = False,
    has_reframe: bool = False,
    has_lora: bool = False,
) -> str:
    return resolve_task(
        model=model,
        model_config=model_config,
        family=family,
        base_model=base_model,
        image_count=image_count,
        video_count=video_count,
        reference_image_count=reference_image_count,
        task=task,
        i2i_mode=i2i_mode,
        has_image_strength=has_image_strength,
        has_video_strength=has_video_strength,
        has_video_mask=has_video_mask,
        has_mask=has_mask,
        has_control_image=has_control_image,
        has_outpaint=has_outpaint,
        has_reframe=has_reframe,
        has_lora=has_lora,
    ).task


def _resolve_model_identity(
    *,
    model: str | None,
    model_config: ModelConfig | None,
    family: str | None,
    base_model: str | None,
) -> _ModelIdentity:
    if model_config is None and model is not None and _is_unsupported_flux2_dev_model(model):
        raise TaskInferenceError(
            "black-forest-labs/FLUX.2-dev is not supported by the current MLX-Gen FLUX.2 runtime. "
            "Use a supported FLUX.2 Klein model, or add a first-class FLUX.2-dev model config and "
            "weight mapping before using FLUX.2-dev LoRAs."
        )

    identity_source = "provided"
    if model_config is None and model is not None:
        try:
            resolved = ConfigResolution.resolve_with_source(model_name=model, base_model=base_model)
            model_config = resolved.model_config
            identity_source = resolved.identity_source
        except ModelConfigError:
            model_config = None
            identity_source = "family_override_only" if family is not None else "unresolved"
    elif model_config is not None:
        identity_source = _provided_identity_source(model=model, model_config=model_config, base_model=base_model)
    elif family is not None:
        identity_source = "family_override_only"

    if model_config is None and family is not None:
        raise TaskInferenceError(
            f"family={family!r} is not enough to configure model {model!r}. "
            "Pass --base-model with a supported model alias so MLX-Gen can build a trustworthy model config."
        )

    family_aliases = set(model_config.aliases) if model_config is not None else set()
    family_key = _model_key(
        model_config.base_model if model_config is not None else None, *sorted(family_aliases), model
    )
    inferred_family = _infer_family(family_aliases, family_key)
    if family is not None and inferred_family is not None and family != inferred_family:
        raise TaskInferenceError(
            f"family {family!r} conflicts with model {model!r}, which resolves to family {inferred_family!r}."
        )
    resolved_family = family or inferred_family
    if resolved_family is None:
        raise TaskInferenceError(
            f"Could not infer a supported backend from model {model!r}. "
            "Pass family='qwen', 'flux2', 'fibo', 'z-image', 'ernie-image', 'wan', 'bonsai', "
            "'minimax-h3', 'seedvr2', or 'swiftvr'."
        )

    trusted_identity_sources = {"catalog", "explicit_base", "official_prepared", "provided", "provided_derived"}
    aliases = family_aliases if identity_source in trusted_identity_sources else set()
    if model_config is None:
        model_key = _model_key(model)
    elif identity_source in trusted_identity_sources:
        model_key = _model_key(model_config.base_model, *sorted(family_aliases))
    else:
        model_key = ""
    return _ModelIdentity(
        model_config=model_config,
        aliases=aliases,
        model_name=model_config.model_name if model_config is not None else model,
        model_key=model_key,
        family=resolved_family,
        identity_source=identity_source,
    )


def _provided_identity_source(
    *,
    model: str | None,
    model_config: ModelConfig,
    base_model: str | None,
) -> str:
    from mflux.models.common.resolution.config_resolution import ConfigResolution

    if model is None:
        return "provided"
    if model_config.model_name != model:
        return "catalog"
    if base_model is not None:
        return "explicit_base"
    if model_config.base_model is not None:
        return "official_prepared" if ConfigResolution._is_official_prepared_repo_id(model) else "infer_substring"
    return "provided"


def _is_unsupported_flux2_dev_model(model: str) -> bool:
    normalized = model.lower().replace("\\", "/").replace("--", "/")
    return "flux.2-dev" in normalized or "flux.2/dev" in normalized


def _capabilities_for(identity: _ModelIdentity) -> ModelCapabilities:
    capabilities = _family_capabilities(identity)
    # Negative-prompt support is a property of the family and, for FLUX.2, of the weights, so it
    # is stamped here once rather than repeated on every row constructor.
    return replace(
        capabilities,
        capabilities=tuple(
            replace(row, supports_negative_prompt=_supports_negative_prompt(identity))
            for row in capabilities.capabilities
        ),
    )


def _supports_negative_prompt(identity: _ModelIdentity) -> bool:
    family = identity.family
    if family == "bonsai":
        # The backend refuses the option: the model has no guidance branch to steer.
        return False
    if family == "flux2":
        # Base Klein runs true classifier-free guidance and takes a negative prompt with guidance
        # above 1.0; distilled Klein is step-distilled and has no guidance branch on any route.
        return _is_flux2_klein_base(identity.aliases, identity.model_key)
    return family in {"ernie-image", "z-image", "qwen", "fibo", "wan"}


def _family_capabilities(identity: _ModelIdentity) -> ModelCapabilities:
    family = identity.family
    if family == "bonsai":
        return ModelCapabilities(
            schema_version=CAPABILITIES_SCHEMA_VERSION,
            family=family,
            label="Bonsai Image",
            model_name=identity.model_name,
            capabilities=(
                GenerationCapability(
                    id="bonsai.text",
                    public_task=TEXT_TO_IMAGE,
                    mode=MODE_TEXT_ONLY,
                    handler_id="bonsai.generate",
                    default_for_task=True,
                ),
            ),
        )
    if family == "ernie-image":
        return _image_latent_capabilities(
            identity=identity,
            family=family,
            label="ERNIE Image Turbo",
            model_name=identity.model_name,
            handler_id="ernie-image.generate",
            supports_guidance=True,
            supports_lora=True,
        )
    if family == "z-image":
        return _z_image_capabilities(identity)
    if family == "qwen":
        return _qwen_capabilities(identity)
    if family == "flux2":
        return _flux2_capabilities(identity)
    if family == "fibo":
        return _fibo_capabilities(identity)
    if family == "wan":
        return _wan_capabilities(identity)
    if family == "minimax-h3":
        return _minimax_h3_capabilities(identity)
    if family == "seedvr2":
        return _seedvr2_capabilities(identity)
    if family == "swiftvr":
        return _swiftvr_capabilities(identity)
    raise TaskInferenceError(f"Unsupported generation family {family!r}.")


def _seedvr2_label(identity: _ModelIdentity) -> str:
    # Variant identity comes from trusted aliases only, mirroring
    # seedvr2_upscale._seedvr2_variant_name. seedvr2-7b and seedvr2-7b-sharp share the
    # model_name "ByteDance-Seed/SeedVR2-7B", so model_name cannot tell them apart, and a
    # path substring is not evidence (ADR 0070). An unprovable variant degrades to the
    # bare family label; the rows are identical across variants, so nothing is claimed.
    aliases = {alias.lower() for alias in identity.aliases}
    if "seedvr2-7b-sharp" in aliases or "seedvr2-7b-sharp-fp16" in aliases:
        return "SeedVR2 7B Sharp"
    if "seedvr2-7b" in aliases:
        return "SeedVR2 7B"
    if "seedvr2-3b" in aliases or "seedvr2" in aliases:
        return "SeedVR2 3B"
    return "SeedVR2"


def _seedvr2_quantization_bits() -> tuple[int, ...]:
    # The weight applier carries no SeedVR2-specific bit restriction, so the accepted set
    # is exactly the CLI's shared --quantize choices. Imported lazily: `import mflux` must
    # stay free of the CLI defaults chain.
    from mflux.cli.defaults import defaults as ui_defaults

    return tuple(sorted(ui_defaults.QUANTIZE_CHOICES))


def _seedvr2_capabilities(identity: _ModelIdentity) -> ModelCapabilities:
    # Streaming chunk floors are read from the module that enforces them rather than
    # copied, so a retune cannot make the payload lie. Lazy: seedvr2_util pulls PIL.
    from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util

    quantization_bits = _seedvr2_quantization_bits()
    shared = {
        "handler_id": "seedvr2.upscale",
        "supports_scaling": True,
        "scale_factors": None,
        "supports_short_side_resolution": True,
        "max_canvas_pixels": None,
        "supports_quantization": True,
        "quantization_bits": quantization_bits,
        # No astype in the SeedVR2 weight definition: runtime dtype is the on-disk dtype,
        # which is a property of the package, not of the route.
        "weight_precision": None,
        "supports_multi_seed": True,
        "supports_softness": True,
        "color_correction_modes": ("wavelet", "lab", "off"),
        "default_color_correction": "wavelet",
    }
    return ModelCapabilities(
        schema_version=CAPABILITIES_SCHEMA_VERSION,
        family="seedvr2",
        label=_seedvr2_label(identity),
        model_name=identity.model_name,
        capabilities=(),
        restoration=(
            RestorationCapability(
                id="seedvr2.restore-image",
                public_task=IMAGE_TO_IMAGE,
                mode=MODE_RESTORE_IMAGE,
                min_images=1,
                max_images=1,
                min_videos=0,
                max_videos=0,
                default_resolution="384",
                # generate_image enforces only steps >= 1; the "1-4" in --steps help is
                # guidance and stays in the docs (ADR 0003), not in the contract.
                supports_steps=True,
                steps_min=1,
                steps_max=None,
                supports_clip_window=False,
                supports_vae_tiling=True,
                supports_audio_passthrough=False,
                # Any geometry is legal: the route pads to the VAE's multiple internally
                # and crops back, so the output preserves the requested size exactly
                # (`1x` is pixel-identical). A 16 here would tell hosts to refuse legal
                # requests or predict a 16-multiple output that never arrives.
                dimension_multiple=1,
                **shared,
            ),
            RestorationCapability(
                id="seedvr2.restore-video",
                public_task=VIDEO_TO_VIDEO,
                mode=MODE_RESTORE_VIDEO,
                min_images=0,
                max_images=0,
                min_videos=1,
                max_videos=1,
                # Video overrides the 384 parser default when --resolution is omitted.
                default_resolution="1x",
                # Neither restore_video_to_path nor generate_video takes a step count.
                supports_steps=False,
                supports_clip_window=True,
                min_frames=1,
                max_frames=None,
                # The route pads the request via padded_video_frame_count instead of
                # constraining it; only --temporal-chunk-size carries the 4n+1 rule.
                frame_multiple=None,
                frame_remainder=None,
                chunk_strategy="streaming-overlap",
                chunk_options_user_settable=True,
                chunk_size_default=SEEDVR2_DEFAULT_TEMPORAL_CHUNK_SIZE,
                # The floors bite only once chunking actually engages (chunk_size below
                # the clip's frame count); a whole-shot restore is exempt.
                chunk_size_min=SeedVR2Util.VIDEO_MIN_PRODUCTION_STREAMING_CHUNK_FRAMES,
                chunk_size_multiple=4,
                chunk_size_remainder=1,
                chunk_overlap_default=SEEDVR2_DEFAULT_TEMPORAL_CHUNK_OVERLAP,
                chunk_overlap_min=SeedVR2Util.VIDEO_MIN_PRODUCTION_STREAMING_OVERLAP_FRAMES,
                chunk_overlap_multiple=4,
                supports_vae_tiling=False,
                supports_audio_passthrough=True,
                # Video output IS constrained: frames are center-cropped to multiples
                # of 16, matching the official pipeline.
                dimension_multiple=16,
                **shared,
            ),
        ),
    )


def _swiftvr_capabilities(identity: _ModelIdentity) -> ModelCapabilities:
    # Lazy imports: both modules sit behind the SwiftVR package __init__, which builds the
    # model classes. No weights are loaded, but `import mflux` must not pay for it.
    from mflux.models.swiftvr.swiftvr_initializer import SwiftVRInitializer
    from mflux.models.swiftvr.variants.upscale.swiftvr_util import (
        MAX_ANALYSED_CANVAS_PIXELS,
        SPATIAL_PAD_MULTIPLE,
        SwiftVRUtil,
    )

    # An unresolved local checkpoint still gets the protocol constants, which are fixed by
    # the weights rather than by the handle. The catalog entry is the fallback source.
    model_config = identity.model_config or ModelConfig.swiftvr()
    overrides = model_config.transformer_overrides or {}
    if "rope_max_seq_len" not in overrides:
        raise TaskInferenceError(
            f"SwiftVR catalog entry {model_config.model_name!r} is missing transformer_overrides "
            "['rope_max_seq_len'], so MLX-Gen cannot state the longest supported clip. Add the key "
            "rather than letting the capability payload invent a frame ceiling."
        )
    try:
        # The single source for the run defaults; it raises rather than inventing a value.
        runtime_settings = SwiftVRInitializer.runtime_settings(model_config)
    except ValueError as exc:
        raise TaskInferenceError(str(exc)) from exc

    return ModelCapabilities(
        schema_version=CAPABILITIES_SCHEMA_VERSION,
        family="swiftvr",
        label="SwiftVR 5B",
        model_name=identity.model_name,
        capabilities=(),
        restoration=(
            RestorationCapability(
                id="swiftvr.restore-video",
                public_task=VIDEO_TO_VIDEO,
                mode=MODE_RESTORE_VIDEO,
                handler_id="swiftvr.restore",
                min_images=0,
                max_images=0,
                min_videos=1,
                max_videos=1,
                # 1x only. SwiftVRUtil.output_canvas is the enforcement site and owns the
                # actionable message; this row describes, it does not re-implement.
                supports_scaling=False,
                scale_factors=("1x",),
                supports_short_side_resolution=False,
                default_resolution="1x",
                dimension_multiple=SPATIAL_PAD_MULTIPLE,
                max_canvas_pixels=MAX_ANALYSED_CANVAS_PIXELS,
                supports_quantization=False,
                quantization_bits=(),
                weight_precision="bf16",
                supports_steps=False,
                # One forward pass at a fixed timestep: no noise, so no seed axis.
                supports_multi_seed=False,
                supports_clip_window=True,
                min_frames=1,
                max_frames=SwiftVRUtil.max_supported_source_frames(int(overrides["rope_max_seq_len"])),
                frame_multiple=4,
                frame_remainder=1,
                chunk_strategy="fixed-causal",
                # clip_len / dit_overlap stay Python-API parameters; the CLI rejects
                # --temporal-chunk-size and --temporal-chunk-overlap.
                chunk_options_user_settable=False,
                chunk_size_default=runtime_settings.clip_len,
                chunk_size_min=4,
                chunk_size_multiple=4,
                chunk_size_remainder=0,
                chunk_overlap_default=runtime_settings.dit_overlap,
                chunk_overlap_min=0,
                chunk_overlap_multiple=None,
                supports_softness=False,
                supports_vae_tiling=False,
                # The route accepts exactly one mode: the reference pipeline writes the
                # decoder output unchanged, and SwiftVR._assert_supported_options refuses
                # wavelet/lab. Declaring them here would invite a host to offer a flag
                # that only fails after the 5B weight load.
                color_correction_modes=("off",),
                default_color_correction="off",
                supports_audio_passthrough=True,
            ),
        ),
    )


def _ordinary_i2i_canvas_contract() -> dict:
    return {
        "canvas_policies": (CANVAS_POLICY_SOURCE_ASPECT, CANVAS_POLICY_EXACT_RESIZE),
        "default_canvas_policy": CANVAS_POLICY_SOURCE_ASPECT,
        "primary_image_index": 0,
        "dimension_multiple": 16,
    }


def _outpaint_capability_kwargs(
    *,
    supports_outpaint: bool,
    fill_modes: tuple[str, ...] = (),
    default_fill_mode: str | None = None,
    supports_fill_option: bool = False,
    auto_edge_fill_max_stretch: float | None = None,
    recommended_lora: str | None = None,
    preservation: str | None = None,
    validated_padding: str | None = None,
    validated_fill_mode: str | None = None,
    validated_max_canvas_pixels: int | None = None,
    auto_split_corner_ratio: float | None = None,
) -> dict:
    # Mirrors _lora_capability_kwargs: one place decides the whole field group so a route can
    # never advertise half a contract (outpaint supported, fill contract silently missing).
    #
    # The validated envelope and the recommended adapter are per-row arguments rather than
    # constants baked in here: they are claims about recorded evidence for one exact route, and a
    # new row that shares the algorithm does not thereby inherit another row's proof.
    if not supports_outpaint:
        return {
            "supports_outpaint": False,
            "supports_outpaint_fill": False,
            "outpaint_fill_modes": (),
            "outpaint_default_fill_mode": None,
            "outpaint_auto_edge_fill_max_stretch": None,
            "outpaint_recommended_lora": None,
            "outpaint_preservation": None,
            "outpaint_validated_padding": None,
            "outpaint_validated_fill_mode": None,
            "outpaint_validated_max_canvas_pixels": None,
            "outpaint_pass_modes": (),
            "outpaint_default_passes": None,
            "outpaint_auto_split_corner_ratio": None,
        }
    if preservation is None:
        # A route that outpaints without saying how it keeps the source is exactly the half
        # contract this helper exists to prevent: the shared outpaint layer would have to guess
        # the preservation strategy, and a host could not read it at all.
        raise ValueError("An outpaint-capable capability row must declare an outpaint preservation strategy.")
    return {
        "supports_outpaint": True,
        "supports_outpaint_fill": supports_fill_option,
        "outpaint_fill_modes": fill_modes,
        "outpaint_default_fill_mode": default_fill_mode,
        "outpaint_auto_edge_fill_max_stretch": auto_edge_fill_max_stretch,
        "outpaint_recommended_lora": recommended_lora,
        "outpaint_preservation": preservation,
        "outpaint_validated_padding": validated_padding,
        "outpaint_validated_fill_mode": validated_fill_mode,
        "outpaint_validated_max_canvas_pixels": validated_max_canvas_pixels,
        # Every expanded-canvas route runs the shared pass planner, so the accepted values are the
        # same on every row; only the auto-split depth is a per-route claim about measured
        # behaviour, and None is the honest value on a route where the corner has not been measured.
        "outpaint_pass_modes": OUTPAINT_PASS_MODES,
        "outpaint_default_passes": OUTPAINT_DEFAULT_PASSES,
        "outpaint_auto_split_corner_ratio": auto_split_corner_ratio,
    }


def _lora_capability_kwargs(
    *,
    identity: _ModelIdentity,
    capability_id: str,
    supports_lora: bool,
    lora_target_roles: tuple[str, ...] = ("transformer",),
) -> dict:
    if not supports_lora:
        return {
            "supports_lora": False,
            "lora_status": "unsupported",
            "lora_target_roles": (),
            "lora_validation_profile": None,
        }
    status, validation_profile = get_lora_validation_status(
        model=identity.model_name,
        model_config=identity.model_config,
        capability_id=capability_id,
    )
    if status == LORA_STATUS_UNSUPPORTED:
        return {
            "supports_lora": False,
            "lora_status": LORA_STATUS_UNSUPPORTED,
            "lora_target_roles": (),
            "lora_validation_profile": None,
        }
    return {
        "supports_lora": True,
        "lora_status": status,
        "lora_target_roles": lora_target_roles,
        "lora_validation_profile": validation_profile,
    }


def _image_latent_capabilities(
    *,
    identity: _ModelIdentity,
    family: str,
    label: str,
    model_name: str | None,
    handler_id: str,
    supports_guidance: bool,
    supports_lora: bool = False,
) -> ModelCapabilities:
    i2i_canvas = _ordinary_i2i_canvas_contract()
    return ModelCapabilities(
        schema_version=CAPABILITIES_SCHEMA_VERSION,
        family=family,
        label=label,
        model_name=model_name,
        capabilities=(
            GenerationCapability(
                id=f"{family}.text",
                public_task=TEXT_TO_IMAGE,
                mode=MODE_TEXT_ONLY,
                handler_id=handler_id,
                default_for_task=True,
                **_lora_capability_kwargs(
                    identity=identity, capability_id=f"{family}.text", supports_lora=supports_lora
                ),
            ),
            GenerationCapability(
                id=f"{family}.latent",
                public_task=IMAGE_TO_IMAGE,
                mode=MODE_LATENT_IMG2IMG,
                handler_id=handler_id,
                min_images=1,
                max_images=1,
                supports_image_strength=True,
                default_for_task=True,
                resize_modes=RESIZE_MODE_CHOICES,
                **_lora_capability_kwargs(
                    identity=identity, capability_id=f"{family}.latent", supports_lora=supports_lora
                ),
                **i2i_canvas,
            ),
        ),
    )


def _z_image_capabilities(identity: _ModelIdentity) -> ModelCapabilities:
    handler_id = (
        "z-image-turbo.generate" if _is_z_image_turbo(identity.aliases, identity.model_key) else "z-image.generate"
    )
    base = _image_latent_capabilities(
        identity=identity,
        family="z-image",
        label="Z-Image",
        model_name=identity.model_name,
        handler_id=handler_id,
        supports_guidance=True,
        supports_lora=True,
    )
    # Native inpaint is Turbo-only: the 2026-07-15 masked matrix measured reproducible
    # geometry artifacts on non-turbo rows (docs/assets/validation/masked-edit-matrix-2026-07-15),
    # so non-turbo masked edit is not supported for the moment. Spoofed or inferred local
    # names stay fail-closed (0070 hardening).
    is_trusted_turbo = bool(identity.aliases or identity.model_key) and _is_z_image_turbo(
        identity.aliases, identity.model_key
    )
    if not is_trusted_turbo:
        return base
    return ModelCapabilities(
        schema_version=base.schema_version,
        family=base.family,
        label=base.label,
        model_name=base.model_name,
        capabilities=(
            *base.capabilities,
            GenerationCapability(
                id="z-image.inpaint",
                public_task=IMAGE_TO_IMAGE,
                mode=MODE_EDIT_REFERENCE,
                handler_id=handler_id,
                min_images=1,
                max_images=1,
                supports_mask=True,
                # Native inpaint maps source+mask through the shared geometry, so it
                # honors --resize-mode (unlike reference-pinned edit routes).
                resize_modes=RESIZE_MODE_CHOICES,
                **_lora_capability_kwargs(identity=identity, capability_id="z-image.inpaint", supports_lora=True),
                **_ordinary_i2i_canvas_contract(),
            ),
        ),
    )


def _qwen_capabilities(identity: _ModelIdentity) -> ModelCapabilities:
    is_edit_model = _is_qwen_edit(identity.aliases, identity.model_key)
    is_edit_plus_model = _is_qwen_edit_plus(identity.aliases, identity.model_key)
    i2i_canvas = _ordinary_i2i_canvas_contract()
    if is_edit_model:
        capabilities: tuple[GenerationCapability, ...] = (
            GenerationCapability(
                id="qwen.edit",
                public_task=IMAGE_TO_IMAGE,
                mode=MODE_EDIT_REFERENCE,
                handler_id="qwen.edit",
                min_images=1,
                max_images=1,
                default_for_task=True,
                **_lora_capability_kwargs(identity=identity, capability_id="qwen.edit", supports_lora=True),
                **i2i_canvas,
            ),
            GenerationCapability(
                id="qwen.inpaint",
                public_task=IMAGE_TO_IMAGE,
                mode=MODE_EDIT_REFERENCE,
                handler_id="qwen.edit",
                min_images=1,
                max_images=1,
                supports_mask=True,
                **_lora_capability_kwargs(identity=identity, capability_id="qwen.inpaint", supports_lora=True),
                **i2i_canvas,
            ),
            GenerationCapability(
                id="qwen.reframe",
                public_task=IMAGE_TO_IMAGE,
                mode=MODE_EDIT_REFERENCE,
                handler_id="qwen.edit",
                min_images=1,
                max_images=1,
                supports_reframe=True,
                **_lora_capability_kwargs(identity=identity, capability_id="qwen.reframe", supports_lora=True),
                **i2i_canvas,
            ),
            GenerationCapability(
                id="qwen.outpaint",
                public_task=IMAGE_TO_IMAGE,
                mode=MODE_EDIT_REFERENCE,
                handler_id="qwen.edit",
                min_images=1,
                max_images=1,
                **_outpaint_capability_kwargs(
                    supports_outpaint=True,
                    fill_modes=QWEN_OUTPAINT_FILL_MODES,
                    default_fill_mode=QWEN_OUTPAINT_DEFAULT_FILL_MODE,
                    supports_fill_option=False,
                    preservation=QWEN_OUTPAINT_PRESERVATION,
                    validated_padding=OUTPAINT_VALIDATED_PADDING,
                    validated_fill_mode=OUTPAINT_VALIDATED_FILL_MODE,
                    validated_max_canvas_pixels=OUTPAINT_VALIDATED_MAX_CANVAS_PIXELS,
                    auto_split_corner_ratio=OUTPAINT_TWO_DEEP_AXIS_RATIO,
                ),
                **_lora_capability_kwargs(identity=identity, capability_id="qwen.outpaint", supports_lora=True),
                **i2i_canvas,
            ),
        )
        if is_edit_plus_model:
            capabilities += (
                GenerationCapability(
                    id="qwen.multi-reference",
                    public_task=IMAGE_TO_IMAGE,
                    mode=MODE_MULTI_REFERENCE,
                    handler_id="qwen.edit",
                    min_images=2,
                    max_images=None,
                    default_for_task=True,
                    **_lora_capability_kwargs(
                        identity=identity, capability_id="qwen.multi-reference", supports_lora=True
                    ),
                    **i2i_canvas,
                ),
            )
    else:
        capabilities = (
            GenerationCapability(
                id="qwen.text",
                public_task=TEXT_TO_IMAGE,
                mode=MODE_TEXT_ONLY,
                handler_id="qwen.generate",
                default_for_task=True,
                **_lora_capability_kwargs(identity=identity, capability_id="qwen.text", supports_lora=True),
            ),
            *(
                (
                    GenerationCapability(
                        id="qwen.control",
                        public_task=TEXT_TO_IMAGE,
                        mode=MODE_TEXT_ONLY,
                        handler_id="qwen.generate",
                        supports_control_image=True,
                        control_model=QWEN_CONTROL_UNION_MODEL,
                        **_lora_capability_kwargs(
                            identity=identity,
                            capability_id="qwen.control",
                            supports_lora=True,
                        ),
                    ),
                )
                if _supports_qwen_base_control(identity)
                else ()
            ),
            *(
                (
                    GenerationCapability(
                        id="qwen.control-inpaint",
                        public_task=IMAGE_TO_IMAGE,
                        mode=MODE_EDIT_REFERENCE,
                        handler_id="qwen.generate",
                        min_images=1,
                        max_images=1,
                        supports_mask=True,
                        control_model=QWEN_CONTROL_INPAINT_MODEL,
                        **_lora_capability_kwargs(
                            identity=identity,
                            capability_id="qwen.control-inpaint",
                            supports_lora=True,
                        ),
                        **i2i_canvas,
                    ),
                )
                if _supports_qwen_base_control(identity)
                else ()
            ),
            *(
                (
                    GenerationCapability(
                        # Native base-Qwen masked edit (diffusers QwenImageInpaintPipeline port).
                        # Exactly one masked route per row: the exact validated control-inpaint
                        # row keeps its sidecar route, every other trusted base row gets native.
                        id="qwen.base-inpaint",
                        public_task=IMAGE_TO_IMAGE,
                        mode=MODE_EDIT_REFERENCE,
                        handler_id="qwen.generate",
                        min_images=1,
                        max_images=1,
                        supports_mask=True,
                        # Native masked edit maps source+mask through the shared
                        # geometry; the control-inpaint sidecar row stays
                        # reference-pinned and advertises no resize modes.
                        resize_modes=RESIZE_MODE_CHOICES,
                        **_lora_capability_kwargs(
                            identity=identity,
                            capability_id="qwen.base-inpaint",
                            supports_lora=True,
                        ),
                        **i2i_canvas,
                    ),
                )
                if _supports_qwen_base_native_inpaint(identity)
                else ()
            ),
            GenerationCapability(
                id="qwen.latent",
                public_task=IMAGE_TO_IMAGE,
                mode=MODE_LATENT_IMG2IMG,
                handler_id="qwen.generate",
                min_images=1,
                max_images=1,
                supports_image_strength=True,
                default_for_task=True,
                resize_modes=RESIZE_MODE_CHOICES,
                **_lora_capability_kwargs(identity=identity, capability_id="qwen.latent", supports_lora=True),
                **i2i_canvas,
            ),
        )
    return ModelCapabilities(
        schema_version=CAPABILITIES_SCHEMA_VERSION,
        family=identity.family,
        label=_qwen_label(identity),
        model_name=identity.model_name,
        capabilities=capabilities,
    )


def _flux2_capabilities(identity: _ModelIdentity) -> ModelCapabilities:
    i2i_canvas = _ordinary_i2i_canvas_contract()
    is_base_model = _is_flux2_klein_base(identity.aliases, identity.model_key)
    if identity.identity_source == "explicit_base":
        return ModelCapabilities(
            schema_version=CAPABILITIES_SCHEMA_VERSION,
            family=identity.family,
            label="FLUX.2",
            model_name=identity.model_name,
            capabilities=(
                GenerationCapability(
                    id="flux2.text",
                    public_task=TEXT_TO_IMAGE,
                    mode=MODE_TEXT_ONLY,
                    handler_id="flux2.generate",
                    default_for_task=True,
                    **_lora_capability_kwargs(identity=identity, capability_id="flux2.text", supports_lora=True),
                ),
                GenerationCapability(
                    id="flux2.edit",
                    public_task=IMAGE_TO_IMAGE,
                    mode=MODE_EDIT_REFERENCE,
                    handler_id="flux2.edit",
                    min_images=1,
                    max_images=1,
                    default_for_task=True,
                    **_lora_capability_kwargs(identity=identity, capability_id="flux2.edit", supports_lora=True),
                    **i2i_canvas,
                ),
            ),
        )
    return ModelCapabilities(
        schema_version=CAPABILITIES_SCHEMA_VERSION,
        family=identity.family,
        label="FLUX.2",
        model_name=identity.model_name,
        capabilities=(
            GenerationCapability(
                id="flux2.text",
                public_task=TEXT_TO_IMAGE,
                mode=MODE_TEXT_ONLY,
                handler_id="flux2.generate",
                default_for_task=True,
                **_lora_capability_kwargs(identity=identity, capability_id="flux2.text", supports_lora=True),
            ),
            GenerationCapability(
                id="flux2.latent",
                public_task=IMAGE_TO_IMAGE,
                mode=MODE_LATENT_IMG2IMG,
                handler_id="flux2.generate",
                min_images=1,
                max_images=1,
                supports_image_strength=True,
                resize_modes=RESIZE_MODE_CHOICES,
                **_lora_capability_kwargs(identity=identity, capability_id="flux2.latent", supports_lora=True),
                **i2i_canvas,
            ),
            GenerationCapability(
                id="flux2.edit",
                public_task=IMAGE_TO_IMAGE,
                mode=MODE_EDIT_REFERENCE,
                handler_id="flux2.edit",
                min_images=1,
                max_images=1,
                default_for_task=True,
                **_lora_capability_kwargs(identity=identity, capability_id="flux2.edit", supports_lora=True),
                **i2i_canvas,
            ),
            GenerationCapability(
                # Masked edit follows the diffusers Flux2KleinInpaintPipeline. The unified route
                # takes one source image; extra masked-area reference images stay on the backend
                # command and Python API, mirroring the narrow qwen.inpaint contract here.
                id="flux2.inpaint",
                public_task=IMAGE_TO_IMAGE,
                mode=MODE_EDIT_REFERENCE,
                handler_id="flux2.edit",
                min_images=1,
                max_images=1,
                supports_mask=True,
                **_lora_capability_kwargs(identity=identity, capability_id="flux2.inpaint", supports_lora=True),
                **i2i_canvas,
            ),
            *_flux2_canvas_expansion_capabilities(
                identity=identity,
                is_base_model=is_base_model,
                i2i_canvas=i2i_canvas,
            ),
            GenerationCapability(
                id="flux2.multi-reference",
                public_task=IMAGE_TO_IMAGE,
                mode=MODE_MULTI_REFERENCE,
                handler_id="flux2.edit",
                min_images=2,
                max_images=None,
                default_for_task=True,
                **_lora_capability_kwargs(identity=identity, capability_id="flux2.multi-reference", supports_lora=True),
                **i2i_canvas,
            ),
        ),
    )


def _flux2_canvas_expansion_capabilities(
    *,
    identity: _ModelIdentity,
    is_base_model: bool,
    i2i_canvas: dict,
) -> tuple[GenerationCapability, ...]:
    """The FLUX.2 Klein expanded-canvas rows: strict outpaint, plus reframe on distilled weights.

    Both weight families outpaint through the same latent-locked route: `Flux2KleinOutpaint`
    takes any FLUX.2 model config, and the preservation strategy is a property of the route, not
    of the weights. They differ in guidance (base runs true CFG, distilled is step-distilled and
    stays at 1.0) and in the evidence behind them, which is why the recommended adapter and the
    validated envelope are declared per row instead of shared.

    Distilled weights additionally keep the generative reframe row, which is allowed to recompose
    the source rather than preserve it. Outpaint and reframe are different guarantees, so they are
    two coexisting rows rather than one row whose identity flips with the weights.
    """
    outpaint = GenerationCapability(
        id="flux2.outpaint",
        public_task=IMAGE_TO_IMAGE,
        mode=MODE_EDIT_REFERENCE,
        handler_id="flux2.edit",
        min_images=1,
        max_images=1,
        **_outpaint_capability_kwargs(
            supports_outpaint=True,
            fill_modes=FLUX2_OUTPAINT_FILL_MODES,
            default_fill_mode=FLUX2_OUTPAINT_DEFAULT_FILL_MODE,
            supports_fill_option=True,
            auto_edge_fill_max_stretch=FLUX2_OUTPAINT_AUTO_EDGE_FILL_MAX_STRETCH,
            # The green-canvas adapter is trained on FLUX.2 Klein base 4B and its A/B proof is a
            # base row; there is no distilled measurement, so distilled rows recommend nothing.
            recommended_lora=FLUX2_OUTPAINT_RECOMMENDED_LORA if is_base_model else None,
            preservation=FLUX2_OUTPAINT_PRESERVATION,
            # Both families have a recorded row inside this envelope: base Klein through the
            # 2026-06-10 starship profile, distilled Klein 4B/9B q8 through the 2026-09-01
            # latent-lock profile, which ran the same padding, canvas and edge fill.
            validated_padding=OUTPAINT_VALIDATED_PADDING,
            validated_fill_mode=OUTPAINT_VALIDATED_FILL_MODE,
            validated_max_canvas_pixels=OUTPAINT_VALIDATED_MAX_CANVAS_PIXELS,
            # Measured to duplicate the subject in a deep free corner on distilled 9B (3/3 seeds
            # at 0.59) and to be clean as two single-axis passes; `auto` splits past the ratio.
            auto_split_corner_ratio=OUTPAINT_TWO_DEEP_AXIS_RATIO,
        ),
        **_lora_capability_kwargs(identity=identity, capability_id="flux2.outpaint", supports_lora=True),
        **i2i_canvas,
    )
    if is_base_model:
        return (outpaint,)
    reframe = GenerationCapability(
        id="flux2.reframe",
        public_task=IMAGE_TO_IMAGE,
        mode=MODE_EDIT_REFERENCE,
        handler_id="flux2.edit",
        min_images=1,
        max_images=1,
        supports_reframe=True,
        **_outpaint_capability_kwargs(supports_outpaint=False),
        **_lora_capability_kwargs(identity=identity, capability_id="flux2.reframe", supports_lora=True),
        **i2i_canvas,
    )
    return (reframe, outpaint)


def _fibo_capabilities(identity: _ModelIdentity) -> ModelCapabilities:
    is_edit_model = _is_fibo_edit(identity.aliases, identity.model_key)
    if is_edit_model:
        capabilities = ()
    else:
        capabilities = (
            GenerationCapability(
                id="fibo.text",
                public_task=TEXT_TO_IMAGE,
                mode=MODE_TEXT_ONLY,
                handler_id="fibo.generate",
                default_for_task=True,
            ),
        )
    return ModelCapabilities(
        schema_version=CAPABILITIES_SCHEMA_VERSION,
        family=identity.family,
        label="FIBO",
        model_name=identity.model_name,
        capabilities=capabilities,
    )


def _minimax_h3_capabilities(identity: _ModelIdentity) -> ModelCapabilities:
    # Guidance-distilled joint video + audio model: one text-to-video route (the audio track rides along),
    # no negative prompt, LoRA on the single transformer (the lightx2v Turbo adapters). First-frame
    # conditioning lands with the Qwen3-VL vision tower.
    is_turbo = bool(identity.model_config is not None and identity.model_config.transformer_overrides.get("turbo_lora"))
    return ModelCapabilities(
        schema_version=CAPABILITIES_SCHEMA_VERSION,
        family=identity.family,
        label="MiniMax-H3 Turbo" if is_turbo else "MiniMax-H3",
        model_name=identity.model_name,
        capabilities=(
            GenerationCapability(
                id="minimax-h3.text-video",
                public_task=TEXT_TO_VIDEO,
                mode=MODE_TEXT_VIDEO,
                handler_id="minimax-h3.generate",
                supports_frames=True,
                supports_fps=False,
                default_for_task=True,
                dimension_multiple=32,
                **_lora_capability_kwargs(
                    identity=identity,
                    capability_id="minimax-h3.text-video",
                    supports_lora=True,
                    lora_target_roles=("transformer",),
                ),
            ),
            # First-frame conditioning (the reference's FL2VA): the keyframe is stretched onto a canvas
            # resolved from its own aspect ratio and conditions both the latent rows and the text sequence.
            GenerationCapability(
                id="minimax-h3.first-frame",
                public_task=IMAGE_TO_VIDEO,
                mode=MODE_FIRST_FRAME_I2V,
                handler_id="minimax-h3.generate",
                min_images=1,
                max_images=1,
                supports_frames=True,
                supports_fps=False,
                default_for_task=True,
                dimension_multiple=32,
                canvas_policies=(CANVAS_POLICY_SOURCE_ASPECT,),
                default_canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
                resize_modes=(RESIZE_MODE_RESIZE,),
                **_lora_capability_kwargs(
                    identity=identity,
                    capability_id="minimax-h3.first-frame",
                    supports_lora=True,
                    lora_target_roles=("transformer",),
                ),
            ),
        ),
    )


def _wan_capabilities(identity: _ModelIdentity) -> ModelCapabilities:
    if identity.model_config is None:
        raise TaskInferenceError(
            "Cannot infer a supported Wan model config. "
            "Use an exact supported Wan repo or a local prepared folder whose name includes a specific Wan alias."
        )
    declared_task = identity.model_config.transformer_overrides.get("task")
    supports_image_to_video = bool(identity.model_config.transformer_overrides.get("supports_image_to_video", True))
    supports_video_to_video = bool(identity.model_config.transformer_overrides.get("supports_video_to_video", False))
    is_vace = bool(identity.model_config.transformer_overrides.get("supports_vace", False))
    is_bernini = bool(identity.model_config.transformer_overrides.get("supports_bernini_renderer", False))
    supports_lora = not is_bernini
    lora_target_roles = (
        ("high_noise_transformer", "low_noise_transformer")
        if bool(identity.model_config.transformer_overrides.get("has_transformer_2", False))
        else ("transformer",)
    )
    if is_bernini:
        bernini_capabilities = [
            GenerationCapability(
                id="bernini.reference-video",
                public_task=TEXT_TO_VIDEO,
                mode=MODE_REFERENCE_VIDEO,
                handler_id="wan.generate",
                min_reference_images=1,
                max_reference_images=8,
                supports_frames=True,
                supports_fps=True,
                default_for_task=True,
                canvas_policies=(CANVAS_POLICY_EXACT_RESIZE,),
                default_canvas_policy=CANVAS_POLICY_EXACT_RESIZE,
                resize_modes=(RESIZE_MODE_RESIZE,),
            ),
            GenerationCapability(
                id="bernini.video-edit",
                public_task=VIDEO_TO_VIDEO,
                mode=MODE_LATENT_VIDEO,
                handler_id="wan.generate",
                min_videos=1,
                max_videos=1,
                supports_frames=True,
                supports_fps=True,
                default_for_task=True,
                canvas_policies=(CANVAS_POLICY_SOURCE_ASPECT,),
                default_canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
                resize_modes=(RESIZE_MODE_RESIZE,),
            ),
            GenerationCapability(
                id="bernini.reference-video-edit",
                public_task=VIDEO_TO_VIDEO,
                mode=MODE_REFERENCE_VIDEO_EDIT,
                handler_id="wan.generate",
                min_videos=1,
                max_videos=1,
                min_reference_images=1,
                max_reference_images=8,
                supports_frames=True,
                supports_fps=True,
                canvas_policies=(CANVAS_POLICY_SOURCE_ASPECT,),
                default_canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
                resize_modes=(RESIZE_MODE_RESIZE,),
            ),
        ]
        return ModelCapabilities(
            schema_version=CAPABILITIES_SCHEMA_VERSION,
            family=identity.family,
            label="Bernini-R 1.3B",
            model_name=identity.model_name,
            capabilities=tuple(bernini_capabilities),
        )
    capabilities: list[GenerationCapability] = []
    if declared_task in {TEXT_TO_VIDEO, "text-image-to-video", None}:
        capabilities.append(
            GenerationCapability(
                id="wan.text-video",
                public_task=TEXT_TO_VIDEO,
                mode=MODE_TEXT_VIDEO,
                handler_id="wan.generate",
                max_reference_images=None if is_vace else 0,
                supports_frames=True,
                supports_fps=True,
                default_for_task=True,
                **_lora_capability_kwargs(
                    identity=identity,
                    capability_id="wan.text-video",
                    supports_lora=supports_lora,
                    lora_target_roles=lora_target_roles,
                ),
            )
        )
        if supports_video_to_video:
            capabilities.append(
                GenerationCapability(
                    id="wan.video-video",
                    public_task=VIDEO_TO_VIDEO,
                    mode=MODE_LATENT_VIDEO,
                    handler_id="wan.generate",
                    min_videos=1,
                    max_videos=1,
                    max_reference_images=None if is_vace else 0,
                    # VACE conditions via masks/references, not an SDEdit strength warm start.
                    supports_video_strength=not is_vace,
                    supports_video_mask=True,
                    supports_frames=True,
                    supports_fps=True,
                    default_for_task=True,
                    # VACE requires the exact validated canvas; plain v2v can opt in
                    # to a source-ratio canvas derived from the clip.
                    canvas_policies=(CANVAS_POLICY_EXACT_RESIZE,)
                    if is_vace
                    else (CANVAS_POLICY_EXACT_RESIZE, CANVAS_POLICY_SOURCE_ASPECT),
                    default_canvas_policy=CANVAS_POLICY_EXACT_RESIZE,
                    resize_modes=RESIZE_MODE_CHOICES,
                    **_lora_capability_kwargs(
                        identity=identity,
                        capability_id="wan.video-video",
                        supports_lora=supports_lora,
                        lora_target_roles=lora_target_roles,
                    ),
                )
            )
    if (declared_task in {IMAGE_TO_VIDEO, "text-image-to-video", None}) and supports_image_to_video:
        # First+last bracket conditioning (0097) exists only on the A14B
        # 36-channel concat i2v path; the 5B expand-timesteps path has no
        # last-frame slot.
        uses_expanded_timesteps = bool(identity.model_config.transformer_overrides.get("expand_timesteps", True))
        has_transformer_2 = bool(identity.model_config.transformer_overrides.get("has_transformer_2", False))
        capabilities.append(
            GenerationCapability(
                id="wan.first-frame",
                public_task=IMAGE_TO_VIDEO,
                mode=MODE_FIRST_FRAME_I2V,
                handler_id="wan.generate",
                min_images=1,
                max_images=1,
                supports_frames=True,
                supports_fps=True,
                supports_last_image=not uses_expanded_timesteps and not is_vace,
                supports_context_frames=not uses_expanded_timesteps and not is_vace,
                # SVI 2.0 Pro (0103) needs the dual-expert A14B conditioning
                # stream AND the high/low LoRA pair slots.
                supports_svi=not uses_expanded_timesteps and not is_vace and has_transformer_2,
                default_for_task=True,
                canvas_policies=(CANVAS_POLICY_SOURCE_ASPECT, CANVAS_POLICY_EXACT_RESIZE),
                default_canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
                resize_modes=RESIZE_MODE_CHOICES,
                **_lora_capability_kwargs(
                    identity=identity,
                    capability_id="wan.first-frame",
                    supports_lora=supports_lora,
                    lora_target_roles=lora_target_roles,
                ),
            )
        )
    if not capabilities:
        raise TaskInferenceError(f"Unsupported Wan2.2 model task contract: {declared_task!r}.")
    return ModelCapabilities(
        schema_version=CAPABILITIES_SCHEMA_VERSION,
        family=identity.family,
        label="Wan2.2",
        model_name=identity.model_name,
        capabilities=tuple(capabilities),
    )


def _requested_public_task(
    *,
    model_capabilities: ModelCapabilities,
    task: str,
    image_count: int,
    video_count: int,
) -> str:
    if task == EDIT:
        if model_capabilities.family == "wan":
            raise TaskInferenceError(f"{model_capabilities.label} supports video generation tasks, not {task}.")
        return IMAGE_TO_IMAGE
    if task != TASK_AUTO:
        return task
    if video_count:
        return VIDEO_TO_VIDEO
    public_tasks = {capability.public_task for capability in model_capabilities.capabilities}
    if public_tasks.issubset(PUBLIC_VIDEO_TASKS):
        if image_count:
            return IMAGE_TO_VIDEO if IMAGE_TO_VIDEO in public_tasks else TEXT_TO_VIDEO
        return TEXT_TO_VIDEO if TEXT_TO_VIDEO in public_tasks else IMAGE_TO_VIDEO
    if public_tasks == {IMAGE_TO_IMAGE}:
        return IMAGE_TO_IMAGE
    return IMAGE_TO_IMAGE if image_count else TEXT_TO_IMAGE


def _requested_mode(
    *,
    task: str,
    public_task: str,
    image_count: int,
    i2i_mode: str,
    has_image_strength: bool,
) -> str:
    if public_task != IMAGE_TO_IMAGE:
        if i2i_mode != I2I_MODE_AUTO:
            raise TaskInferenceError("--i2i-mode can only be used with image-to-image generation.")
        return I2I_MODE_AUTO

    requested_mode = i2i_mode
    if task == EDIT and requested_mode == I2I_MODE_AUTO:
        requested_mode = MODE_MULTI_REFERENCE if image_count > 1 else MODE_EDIT_REFERENCE
    elif requested_mode == I2I_MODE_AUTO and image_count > 1:
        requested_mode = MODE_MULTI_REFERENCE

    if has_image_strength:
        if requested_mode not in {I2I_MODE_AUTO, MODE_LATENT_IMG2IMG}:
            raise TaskInferenceError("--image-strength is only supported for latent image-to-image mode.")
        requested_mode = MODE_LATENT_IMG2IMG
    return requested_mode


def _mode_matches_request(capability_mode: str, requested_mode: str) -> bool:
    return capability_mode == requested_mode


def _select_capability(
    *,
    model_capabilities: ModelCapabilities,
    public_task: str,
    requested_mode: str,
    image_count: int,
    video_count: int,
    reference_image_count: int,
    candidates: list[GenerationCapability],
) -> GenerationCapability:
    if candidates:
        defaults = [capability for capability in candidates if capability.default_for_task]
        if requested_mode == I2I_MODE_AUTO and len(defaults) == 1:
            return defaults[0]
        if len(candidates) == 1:
            return candidates[0]
        if requested_mode != I2I_MODE_AUTO and len(candidates) > 1:
            return candidates[0]
        modes = ", ".join(sorted({capability.mode for capability in candidates}))
        raise TaskInferenceError(
            f"{model_capabilities.label} image-to-image request is ambiguous; choose --i2i-mode. "
            f"Available modes: {modes}."
        )

    _raise_no_capability(
        model_capabilities=model_capabilities,
        public_task=public_task,
        requested_mode=requested_mode,
        image_count=image_count,
        video_count=video_count,
        reference_image_count=reference_image_count,
    )


def _raise_no_capability(
    *,
    model_capabilities: ModelCapabilities,
    public_task: str,
    requested_mode: str,
    image_count: int,
    video_count: int,
    reference_image_count: int,
) -> None:
    label = model_capabilities.label
    if not model_capabilities.capabilities:
        raise TaskInferenceError(f"{label} does not expose unified generation capabilities through mlxgen generate.")
    if public_task in VIDEO_TASKS and not any(
        cap.public_task in VIDEO_TASKS for cap in model_capabilities.capabilities
    ):
        raise TaskInferenceError(f"{label} supports image generation tasks, not {public_task}.")
    if public_task in PUBLIC_IMAGE_TASKS and not any(
        cap.public_task in PUBLIC_IMAGE_TASKS for cap in model_capabilities.capabilities
    ):
        raise TaskInferenceError(f"{label} supports video generation tasks, not {public_task}.")
    if reference_image_count and not any(
        capability.public_task == public_task and capability.allows_reference_image_count(reference_image_count)
        for capability in model_capabilities.capabilities
    ):
        raise TaskInferenceError(
            f"{label} does not support {reference_image_count} --reference-image input(s) for {public_task}."
        )
    if public_task == IMAGE_TO_IMAGE and not any(
        cap.public_task == IMAGE_TO_IMAGE for cap in model_capabilities.capabilities
    ):
        raise TaskInferenceError(f"{label} supports text-to-image only; image-to-image/edit is not supported.")
    if public_task == TEXT_TO_IMAGE and image_count:
        raise TaskInferenceError(f"{label} text-to-image cannot be combined with --image or --images.")
    if public_task == IMAGE_TO_IMAGE and image_count == 0:
        raise TaskInferenceError(f"{label} image-to-image requires --image or --image-path.")
    if public_task == IMAGE_TO_IMAGE and requested_mode == MODE_MULTI_REFERENCE:
        raise TaskInferenceError(f"{label} does not support multi-reference image-to-image generation.")
    if public_task == IMAGE_TO_IMAGE and requested_mode == MODE_EDIT_REFERENCE:
        raise TaskInferenceError(f"{label} does not support edit-reference image-to-image generation.")
    if public_task == IMAGE_TO_IMAGE and requested_mode == MODE_LATENT_IMG2IMG:
        raise TaskInferenceError(f"{label} does not support latent image-to-image generation.")
    if public_task == IMAGE_TO_IMAGE:
        raise TaskInferenceError(f"{label} accepts at most one input image for image-to-image generation.")
    if public_task == TEXT_TO_VIDEO and image_count:
        raise TaskInferenceError(f"This {label} text-to-video model does not accept input images.")
    if public_task == IMAGE_TO_VIDEO and image_count == 0:
        raise TaskInferenceError(f"This {label} image-to-video model requires --image or --image-path.")
    if public_task == IMAGE_TO_VIDEO and image_count > 1:
        raise TaskInferenceError(f"{label} image-to-video accepts exactly one input image.")
    if public_task == IMAGE_TO_VIDEO:
        raise TaskInferenceError(f"This {label} text-to-video model does not accept input images.")
    if public_task == VIDEO_TO_VIDEO and not any(
        cap.public_task == VIDEO_TO_VIDEO for cap in model_capabilities.capabilities
    ):
        raise TaskInferenceError(f"{label} does not support video-to-video latent editing.")
    if public_task == VIDEO_TO_VIDEO and video_count == 0:
        raise TaskInferenceError(f"{label} video-to-video requires --video or --video-path.")
    if public_task == VIDEO_TO_VIDEO and video_count > 1:
        raise TaskInferenceError(f"{label} video-to-video accepts exactly one input video.")
    if public_task == VIDEO_TO_VIDEO:
        raise TaskInferenceError(f"{label} does not support the requested video-to-video route.")
    raise TaskInferenceError(f"{label} does not support {public_task}.")


def _infer_family(aliases: set[str], model_key: str) -> str | None:
    if _is_bonsai(aliases, model_key):
        return "bonsai"
    if _is_qwen(aliases, model_key):
        return "qwen"
    if _is_flux2(aliases, model_key):
        return "flux2"
    if _is_fibo(aliases, model_key):
        return "fibo"
    if _is_z_image(aliases, model_key):
        return "z-image"
    if _is_ernie(aliases, model_key):
        return "ernie-image"
    # Restoration families are matched BEFORE _is_wan. SwiftVR runs on a Wan2.2-TI2V-5B
    # backbone and _is_wan matches any "wan" substring, so a checkpoint whose handle
    # mentions wan would otherwise be handed a generation contract for a restorer.
    if _is_seedvr2(aliases, model_key):
        return "seedvr2"
    if _is_swiftvr(aliases, model_key):
        return "swiftvr"
    if _is_minimax_h3(aliases, model_key):
        return "minimax-h3"
    if _is_wan(aliases, model_key):
        return "wan"
    return None


def _model_key(*parts: str | None) -> str:
    return " ".join(part for part in parts if part).lower().replace("_", "-")


def _has_alias(aliases: set[str], *needles: str) -> bool:
    return bool(aliases.intersection(needles))


def _is_qwen(aliases: set[str], model_key: str) -> bool:
    return (
        _has_alias(
            aliases,
            "qwen-image",
            "qwen-image-edit",
            "qwen-image-edit-2509",
            "qwen-image-edit-2511",
        )
        or "qwen" in model_key
    )


def _is_qwen_edit(aliases: set[str], model_key: str) -> bool:
    return _has_alias(
        aliases,
        "qwen-image-edit",
        "qwen-image-edit-2509",
        "qwen-image-edit-2511",
    ) or ("qwen" in model_key and "edit" in model_key)


def _is_qwen_edit_plus(aliases: set[str], model_key: str) -> bool:
    return _has_alias(
        aliases,
        "qwen-image-edit-2509",
        "qwen-edit-2509",
        "qwen-edit-plus",
        "qwen-edit-plus-2509",
        "qwen-image-edit-2511",
        "qwen-edit-2511",
    ) or any(
        token in model_key
        for token in (
            "qwen-image-edit-2509",
            "qwen-edit-2509",
            "qwen-edit-plus",
            "qwen-image-edit-2511",
            "qwen-edit-2511",
        )
    )


def _is_qwen_edit_2511(aliases: set[str], model_key: str) -> bool:
    return _has_alias(aliases, "qwen-image-edit-2511", "qwen-edit-2511") or any(
        token in model_key for token in ("qwen-image-edit-2511", "qwen-edit-2511")
    )


def _supports_qwen_base_control(identity: _ModelIdentity) -> bool:
    if _is_qwen_edit(identity.aliases, identity.model_key):
        return False
    return identity.model_name == "AbstractFramework/qwen-image-8bit"


def _supports_qwen_base_native_inpaint(identity: _ModelIdentity) -> bool:
    # Exactly one masked route per row: the exact validated control-inpaint row keeps the
    # sidecar route; native inpaint covers the other base rows. Untrusted inferred
    # identities stay fail-closed except exact proven rows (0070 hardening).
    if _supports_qwen_base_control(identity):
        return False
    if _is_qwen_edit(identity.aliases, identity.model_key):
        return False
    if identity.aliases or identity.model_key:
        return True
    return identity.model_name in QWEN_BASE_NATIVE_INPAINT_EXACT_ROWS


def _qwen_label(identity: _ModelIdentity) -> str:
    if not _is_qwen_edit(identity.aliases, identity.model_key):
        return "Qwen Image"
    if _is_qwen_edit_2511(identity.aliases, identity.model_key):
        return "Qwen Image Edit 2511"
    if _has_alias(identity.aliases, "qwen-image-edit-2509", "qwen-edit-2509", "qwen-edit-plus") or any(
        token in identity.model_key for token in ("qwen-image-edit-2509", "qwen-edit-2509", "qwen-edit-plus")
    ):
        return "Qwen Image Edit 2509"
    return "Qwen Image Edit"


def _is_flux2(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("flux2") or alias.startswith("klein") for alias in aliases) or any(
        token in model_key for token in ("flux2", "flux.2", "klein")
    )


def _is_flux2_klein_base(aliases: set[str], model_key: str) -> bool:
    return any("klein-base" in alias or "flux2-base" in alias or "flux.2-klein-base" in alias for alias in aliases) or (
        "klein-base" in model_key or "flux2-base" in model_key or "flux.2-klein-base" in model_key
    )


def _is_bonsai(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("bonsai") for alias in aliases) or "bonsai" in model_key


def _is_fibo(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("fibo") for alias in aliases) or "fibo" in model_key


def _is_fibo_edit(aliases: set[str], model_key: str) -> bool:
    return _has_alias(aliases, "fibo-edit", "fibo-edit-rmbg") or ("fibo" in model_key and "edit" in model_key)


def _is_z_image(aliases: set[str], model_key: str) -> bool:
    return _has_alias(aliases, "z-image", "z-image-turbo") or "z-image" in model_key or "zimage" in model_key


def _is_z_image_turbo(aliases: set[str], model_key: str) -> bool:
    return _has_alias(aliases, "z-image-turbo") or (
        ("z-image" in model_key or "zimage" in model_key) and "turbo" in model_key
    )


def _is_ernie(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("ernie") for alias in aliases) or "ernie" in model_key


def _is_wan(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("wan") for alias in aliases) or "wan" in model_key


def _is_minimax_h3(aliases: set[str], model_key: str) -> bool:
    return any(alias.startswith("minimax-h3") for alias in aliases) or "minimax-h3" in model_key


def _is_seedvr2(aliases: set[str], model_key: str) -> bool:
    return any(alias.lower().startswith("seedvr2") for alias in aliases) or "seedvr2" in model_key


def _is_swiftvr(aliases: set[str], model_key: str) -> bool:
    return any(alias.lower().startswith("swiftvr") for alias in aliases) or "swiftvr" in model_key
