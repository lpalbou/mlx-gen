from __future__ import annotations

from dataclasses import dataclass

from mflux.lora_validation_registry import LORA_STATUS_UNSUPPORTED, get_lora_validation_status
from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.utils.dimension_resolver import CANVAS_POLICY_EXACT_RESIZE, CANVAS_POLICY_SOURCE_ASPECT
from mflux.utils.exceptions import ModelConfigError

TASK_ALIASES = {
    "txt2img": "text-to-image",
    "img2img": "image-to-image",
    "txt2vid": "text-to-video",
    "t2v": "text-to-video",
    "img2vid": "image-to-video",
    "i2v": "image-to-video",
}

TASK_AUTO = "auto"
TEXT_TO_IMAGE = "text-to-image"
IMAGE_TO_IMAGE = "image-to-image"
EDIT = "edit"
TEXT_TO_VIDEO = "text-to-video"
IMAGE_TO_VIDEO = "image-to-video"

PUBLIC_IMAGE_TASKS = {TEXT_TO_IMAGE, IMAGE_TO_IMAGE}
PUBLIC_VIDEO_TASKS = {TEXT_TO_VIDEO, IMAGE_TO_VIDEO}
PUBLIC_TASKS = {*PUBLIC_IMAGE_TASKS, *PUBLIC_VIDEO_TASKS}
IMAGE_TASKS = {*PUBLIC_IMAGE_TASKS, EDIT}
VIDEO_TASKS = PUBLIC_VIDEO_TASKS
VALID_TASKS = {TASK_AUTO, EDIT, *PUBLIC_TASKS}
CAPABILITIES_SCHEMA_VERSION = 3
QWEN_CONTROL_UNION_MODEL = (
    "InstantX/Qwen-Image-ControlNet-Union:"
    "diffusion_pytorch_model.safetensors"
)
QWEN_CONTROL_INPAINT_MODEL = (
    "InstantX/Qwen-Image-ControlNet-Inpainting:"
    "diffusion_pytorch_model.safetensors"
)

I2I_MODE_AUTO = "auto"
MODE_TEXT_ONLY = "text-only"
MODE_LATENT_IMG2IMG = "latent-img2img"
MODE_EDIT_REFERENCE = "edit-reference"
MODE_MULTI_REFERENCE = "multi-reference"
MODE_TEXT_VIDEO = "text-video"
MODE_FIRST_FRAME_I2V = "first-frame-i2v"

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
    supports_image_strength: bool = False
    supports_mask: bool = False
    supports_control_image: bool = False
    supports_control_mask: bool = False
    supports_outpaint: bool = False
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
    primary_image_index: int | None = None
    dimension_multiple: int | None = None

    def allows_image_count(self, image_count: int) -> bool:
        if image_count < self.min_images:
            return False
        return self.max_images is None or image_count <= self.max_images

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "public_task": self.public_task,
            "mode": self.mode,
            "handler_id": self.handler_id,
            "min_images": self.min_images,
            "max_images": self.max_images,
            "supports_image_strength": self.supports_image_strength,
            "supports_mask": self.supports_mask,
            "supports_control_image": self.supports_control_image,
            "supports_control_mask": self.supports_control_mask,
            "supports_outpaint": self.supports_outpaint,
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
            "primary_image_index": self.primary_image_index,
            "dimension_multiple": self.dimension_multiple,
        }


@dataclass(frozen=True)
class ModelCapabilities:
    schema_version: int
    family: str
    label: str
    model_name: str | None
    capabilities: tuple[GenerationCapability, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "label": self.label,
            "model_name": self.model_name,
            "capabilities": [capability.to_dict() for capability in self.capabilities],
        }


@dataclass(frozen=True)
class GenerationPlan:
    public_task: str
    mode: str
    capability_id: str
    family: str
    handler_id: str
    image_count: int
    model_name: str | None = None
    model_override: str | None = None
    canvas_policies: tuple[str, ...] = ()
    default_canvas_policy: str | None = None
    primary_image_index: int | None = None
    dimension_multiple: int | None = None
    supports_lora: bool = False
    control_model: str | None = None
    lora_status: str = "unsupported"
    lora_target_roles: tuple[str, ...] = ()
    lora_validation_profile: str | None = None

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
            "model_name": self.model_name,
            "model_override": self.model_override,
            "canvas_policies": list(self.canvas_policies),
            "default_canvas_policy": self.default_canvas_policy,
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
    model_name: str | None = None
    mode: str | None = None
    capability_id: str | None = None
    handler_id: str | None = None


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
    task: str | None = TASK_AUTO,
    i2i_mode: str | None = I2I_MODE_AUTO,
    has_image_strength: bool = False,
    has_mask: bool = False,
    has_control_image: bool = False,
    has_outpaint: bool = False,
    has_reframe: bool = False,
    has_lora: bool = False,
) -> GenerationPlan:
    if image_count < 0:
        raise TaskInferenceError("image_count must be greater than or equal to zero.")
    if has_image_strength and image_count == 0:
        raise TaskInferenceError("--image-strength requires --image or --image-path.")
    if has_mask and image_count == 0:
        raise TaskInferenceError("--mask-path requires --image or --image-path.")
    if has_mask and has_image_strength:
        raise TaskInferenceError(
            "--image-strength cannot be combined with --mask-path; masked inpaint is a separate route from latent image-to-image."
        )
    if has_control_image and image_count > 0:
        raise TaskInferenceError(
            "--controlnet-image-path currently targets text-to-image structured control and cannot be combined "
            "with --image or --image-path."
        )
    if has_outpaint and image_count == 0:
        raise TaskInferenceError("--outpaint-padding requires --image or --image-path.")
    if has_reframe and image_count == 0:
        raise TaskInferenceError("--reframe-padding requires --image or --image-path.")

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
        if capability.public_task == public_task and capability.allows_image_count(image_count)
    ]
    if requested_mode != I2I_MODE_AUTO:
        candidates = [capability for capability in candidates if _mode_matches_request(capability.mode, requested_mode)]

    if has_image_strength:
        candidates = [capability for capability in candidates if capability.supports_image_strength]
        if not candidates:
            raise TaskInferenceError("--image-strength is only supported for latent image-to-image mode.")
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
        candidates=candidates,
    )
    if (
        public_task == IMAGE_TO_IMAGE
        and capability.mode == MODE_LATENT_IMG2IMG
        and image_count > 0
        and not has_image_strength
    ):
        raise TaskInferenceError("--image-strength is required for latent image-to-image mode.")

    return GenerationPlan(
        public_task=capability.public_task,
        mode=capability.mode,
        capability_id=capability.id,
        family=model_capabilities.family,
        handler_id=capability.handler_id,
        image_count=image_count,
        model_name=model_capabilities.model_name,
        model_override=capability.model_override,
        canvas_policies=capability.canvas_policies,
        default_canvas_policy=capability.default_canvas_policy,
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
    task: str | None = TASK_AUTO,
    i2i_mode: str | None = I2I_MODE_AUTO,
    has_image_strength: bool = False,
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
        task=task,
        i2i_mode=i2i_mode,
        has_image_strength=has_image_strength,
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
    task: str | None = TASK_AUTO,
    i2i_mode: str | None = I2I_MODE_AUTO,
    has_image_strength: bool = False,
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
        task=task,
        i2i_mode=i2i_mode,
        has_image_strength=has_image_strength,
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
    family_key = _model_key(model_config.base_model if model_config is not None else None, *sorted(family_aliases), model)
    inferred_family = _infer_family(family_aliases, family_key)
    if family is not None and inferred_family is not None and family != inferred_family:
        raise TaskInferenceError(
            f"family {family!r} conflicts with model {model!r}, which resolves to family {inferred_family!r}."
        )
    resolved_family = family or inferred_family
    if resolved_family is None:
        raise TaskInferenceError(
            f"Could not infer a supported backend from model {model!r}. "
            "Pass family='qwen', 'flux2', 'fibo', 'z-image', 'ernie-image', 'wan', or 'bonsai'."
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
    raise TaskInferenceError(f"Unsupported generation family {family!r}.")


def _ordinary_i2i_canvas_contract() -> dict:
    return {
        "canvas_policies": (CANVAS_POLICY_SOURCE_ASPECT, CANVAS_POLICY_EXACT_RESIZE),
        "default_canvas_policy": CANVAS_POLICY_SOURCE_ASPECT,
        "primary_image_index": 0,
        "dimension_multiple": 16,
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
    if not _is_z_image_turbo(identity.aliases, identity.model_key):
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
                supports_outpaint=True,
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
            GenerationCapability(
                id="qwen.latent",
                public_task=IMAGE_TO_IMAGE,
                mode=MODE_LATENT_IMG2IMG,
                handler_id="qwen.generate",
                min_images=1,
                max_images=1,
                supports_image_strength=True,
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
                id="flux2.outpaint" if is_base_model else "flux2.reframe",
                public_task=IMAGE_TO_IMAGE,
                mode=MODE_EDIT_REFERENCE,
                handler_id="flux2.edit",
                min_images=1,
                max_images=1,
                supports_outpaint=is_base_model,
                supports_reframe=not is_base_model,
                **_lora_capability_kwargs(
                    identity=identity,
                    capability_id="flux2.outpaint" if is_base_model else "flux2.reframe",
                    supports_lora=True,
                ),
                **i2i_canvas,
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


def _wan_capabilities(identity: _ModelIdentity) -> ModelCapabilities:
    if identity.model_config is None:
        raise TaskInferenceError(
            "Cannot infer a supported Wan model config. "
            "Use an exact supported Wan repo or a local prepared folder whose name includes a specific Wan alias."
        )
    declared_task = identity.model_config.transformer_overrides.get("task")
    supports_image_to_video = bool(identity.model_config.transformer_overrides.get("supports_image_to_video", True))
    supports_lora = True
    lora_target_roles = (
        ("high_noise_transformer", "low_noise_transformer")
        if bool(identity.model_config.transformer_overrides.get("has_transformer_2", False))
        else ("transformer",)
    )
    capabilities: list[GenerationCapability] = []
    if declared_task in {TEXT_TO_VIDEO, "text-image-to-video", None}:
        capabilities.append(
            GenerationCapability(
                id="wan.text-video",
                public_task=TEXT_TO_VIDEO,
                mode=MODE_TEXT_VIDEO,
                handler_id="wan.generate",
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
    if (declared_task in {IMAGE_TO_VIDEO, "text-image-to-video", None}) and supports_image_to_video:
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
                default_for_task=True,
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
) -> str:
    if task == EDIT:
        if model_capabilities.family == "wan":
            raise TaskInferenceError(f"{model_capabilities.label} supports video generation tasks, not {task}.")
        return IMAGE_TO_IMAGE
    if task != TASK_AUTO:
        return task
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
    )


def _raise_no_capability(
    *,
    model_capabilities: ModelCapabilities,
    public_task: str,
    requested_mode: str,
    image_count: int,
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
