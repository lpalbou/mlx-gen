from __future__ import annotations

from dataclasses import dataclass

from mflux.lora_validation_registry import (
    ERNIE_TURBO_Q8_ANIME_STYLE_LATENT_PROFILE_ID,
    ERNIE_TURBO_Q8_ANIME_STYLE_PROFILE_ID,
    FLUX2_KLEIN9B_Q8_CONSISTENCY_EDIT_PROFILE_ID,
    FLUX2_KLEIN9B_Q8_MULTI_REFERENCE_MIGRATION_PROFILE_ID,
    FLUX2_KLEIN_BASE4B_Q8_OUTPAINT_PROFILE_ID,
    QWEN2509_Q8_SINGLE_EDIT_MULTI_ANGLE_PROFILE_ID,
    QWEN2511_Q8_INPAINT_LIGHTNING_PROFILE_ID,
    QWEN2511_Q8_MULTI_REFERENCE_MULTI_ANGLE_PROFILE_ID,
    QWEN2511_Q8_OUTPAINT_MULTI_ANGLE_PROFILE_ID,
    QWEN2511_Q8_REFRAME_MULTI_ANGLE_PROFILE_ID,
    QWEN2511_Q8_SINGLE_EDIT_MULTI_ANGLE_PROFILE_ID,
    QWEN2512_Q8_PIXEL_ART_PROFILE_ID,
    QWEN_EDIT_Q8_GHIBLI_PROFILE_ID,
    QWEN_Q8_CONTROL_INPAINT_LIGHTNING_PROFILE_ID,
    QWEN_Q8_CONTROL_LIGHTNING_PROFILE_ID,
    QWEN_Q8_LATENT_REALISM_PROFILE_ID,
    QWEN_Q8_REALISM_PROFILE_ID,
    WAN_A14B_Q8_FOLLOWCAM_T2V_PROFILE_ID,
    WAN_A14B_Q8_LIGHTNING_V2V_PROFILE_ID,
    WAN_A14B_Q8_LIGHTX2V_4STEP_I2V_PROFILE_ID,
    WAN_A14B_Q8_LIGHTX2V_4STEP_T2V_PROFILE_ID,
    WAN_A14B_Q8_ORBIT_I2V_PROFILE_ID,
    WAN_TI2V5B_Q8_CRUSHIT_I2V_PROFILE_ID,
    WAN_TI2V5B_Q8_HSTORIC_T2V_PROFILE_ID,
    ZIMAGE_Q8_CHILDRENS_DRAWINGS_LATENT_PROFILE_ID,
    ZIMAGE_Q8_TECHNICALLYCOLOR_PROFILE_ID,
)

STATUS_PASS = "PASS"
STATUS_PARTIAL = "PARTIAL"
STATUS_FAIL = "FAIL"
STATUS_STALE = "STALE"
STATUS_NA = "N/A"
STATUS_UNREVIEWED = "UNREVIEWED"
STATUS_NOT_AVAILABLE = "NOT_AVAILABLE"

_STATUS_RANK = {
    STATUS_FAIL: 0,
    STATUS_STALE: 1,
    STATUS_UNREVIEWED: 2,
    STATUS_PARTIAL: 3,
    STATUS_PASS: 4,
    STATUS_NA: 5,
}

I2I_EDIT_5X4_PROFILE_ID = "i2i_edit_5x4_2026_06_05"
REFRAME_OUTPAINT_PROFILE_ID = "reframe_outpaint_2026_06_08"
FLUX2_KLEIN_BASE_STARSHIP_PROFILE_ID = "flux2_klein_base_starship_2026_06_10"

CANONICAL_SOURCE = "docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png"
QWEN2511_PARITY_DIR = "docs/assets/validation/qwen-edit-2511-parity-2026-06-06"
REFRAME_OUTPAINT_DIR = "docs/assets/validation/reframe-outpaint-2026-06-08"
REFRAME_OUTPAINT_SOURCE = f"{REFRAME_OUTPAINT_DIR}/source-b-cropped-starship.png"
FLUX2_KLEIN_BASE_STARSHIP_DIR = "docs/assets/validation/flux2-klein-base-starship-2026-06-10"
FLUX2_KLEIN_BASE_STARSHIP_SOURCE = REFRAME_OUTPAINT_SOURCE
LORA_VALIDATION_DIR = "docs/assets/validation/lora-2026-06-11"
WAN_LORA_VALIDATION_DIR = "docs/assets/validation/wan-lora-2026-06-11"
LIGHTX2V_WAN_4STEP_VALIDATION_DIR = "docs/assets/validation/lightx2v-wan-4step-2026-06-12"
LIGHTNING_V2V_VALIDATION_DIR = "docs/assets/validation/lightning-v2v-2026-07-04"
QWEN_INPAINT_VALIDATION_DIR = "docs/assets/validation/qwen-inpaint-2026-06-15"
QWEN_CONTROL_VALIDATION_DIR = "docs/assets/validation/qwen-control-2026-06-15"
QWEN_CONTROL_INPAINT_VALIDATION_DIR = "docs/assets/validation/qwen-control-inpaint-2026-06-21"
ZIMAGE_INPAINT_VALIDATION_DIR = "docs/assets/validation/zimage-inpaint-2026-06-21"
ZIMAGE_INPAINT_PROFILE_ID = "zimage_inpaint_2026_06_21"
ZIMAGE_LATENT_LORA_VALIDATION_DIR = "docs/assets/validation/zimage-latent-lora-2026-06-24"
LORA_ROUTE_EXPANSION_VALIDATION_DIR = "docs/assets/validation/lora-route-expansion-2026-06-22"


@dataclass(frozen=True)
class ValidationRecord:
    profile_id: str
    model: str
    family: str
    package_variant: str
    step: str
    step_label: str
    public_task: str
    mode: str
    status: str
    artifact_path: str | None
    source_images: tuple[str, ...]
    prompt: str
    reviewer_notes: str
    evidence_date: str = "2026-06-05"
    evidence_type: str = "manual_visual_review"

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "model": self.model,
            "family": self.family,
            "package_variant": self.package_variant,
            "step": self.step,
            "step_label": self.step_label,
            "public_task": self.public_task,
            "mode": self.mode,
            "status": self.status,
            "artifact_path": self.artifact_path,
            "source_images": list(self.source_images),
            "prompt": self.prompt,
            "reviewer_notes": self.reviewer_notes,
            "evidence_date": self.evidence_date,
            "evidence_type": self.evidence_type,
        }


@dataclass(frozen=True)
class ValidationProfile:
    id: str
    title: str
    canonical_source: str
    description: str
    records: tuple[ValidationRecord, ...]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "canonical_source": self.canonical_source,
            "description": self.description,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class ModelValidation:
    profile_id: str
    model: str
    status: str
    records: tuple[ValidationRecord, ...]

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "model": self.model,
            "status": self.status,
            "records": [record.to_dict() for record in self.records],
        }


PROMPTS = {
    "B": (
        "Make this same spaceship in the snow look like polished cinematic science-fiction concept art at blue hour. "
        "Preserve the exact camera angle, ship position, snowy canyon, and overall layout. Sharpen hull panels and "
        "add cold blue shadows; no crash, no damage."
    ),
    "C": (
        "Edit the source into the same spaceship after a hard landing in the snow. Preserve the same camera angle, "
        "spaceship identity, rear engines, canyon cliffs, and wide framing. The ship must remain solid and sharp, "
        "but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, "
        "and a thin smoke plume. No blur, no mesh, no dissolve."
    ),
    "D": (
        "Turn the source into a clean graphite pencil sketch of the same hard-landed spaceship scene. Preserve the "
        "spaceship identity, snowy canyon layout, tilted hull, bent landing struts, disturbed snow, debris, and smoke "
        "plume. White paper background, precise line art, no color fill, no blur."
    ),
    "E": (
        "Use the first image as the pencil crash structure and the second image as the cinematic lighting and color "
        "reference. Produce one coherent image of the same hard-landed spaceship in the snow: graphite sketch lines "
        "with subtle cinematic blue-hour shading, stable canyon layout, solid spaceship, visible crash debris, no blur, "
        "no text."
    ),
}

STEP_LABELS = {
    "B": "cinematic reference",
    "C": "crash from source",
    "D": "pencil sketch",
    "E": "multi-reference composition",
}

_MATRIX_DIR = "docs/assets/validation/i2i-edit-5x4-2026-06-05"


def list_validation_profiles() -> tuple[ValidationProfile, ...]:
    return (
        _i2i_edit_profile(),
        _reframe_outpaint_profile(),
        _flux2_klein_base_starship_profile(),
        _zimage_inpaint_profile(),
        *_lora_profiles(),
    )


def get_validation_profile(profile_id: str = I2I_EDIT_5X4_PROFILE_ID) -> ValidationProfile:
    for profile in list_validation_profiles():
        if profile.id == profile_id:
            return profile
    raise KeyError(f"Unknown validation profile {profile_id!r}.")


def get_model_validation(model: str, profile_id: str = I2I_EDIT_5X4_PROFILE_ID) -> ModelValidation:
    profile = get_validation_profile(profile_id)
    exact_keys, fallback_keys = _candidate_model_keys(model)
    records = tuple(record for record in profile.records if _normalize_model_key(record.model) in exact_keys)
    if not records and fallback_keys:
        records = tuple(record for record in profile.records if _normalize_model_key(record.model) in fallback_keys)
    return ModelValidation(
        profile_id=profile.id,
        model=model,
        status=_aggregate_status(records),
        records=records,
    )


def default_validation_profile_id_for_model(model: str) -> str:
    for profile in list_validation_profiles():
        if get_model_validation(model, profile_id=profile.id).records:
            return profile.id
    return I2I_EDIT_5X4_PROFILE_ID


def _aggregate_status(records: tuple[ValidationRecord, ...]) -> str:
    if not records:
        return STATUS_NOT_AVAILABLE
    return min((record.status for record in records), key=lambda status: _STATUS_RANK.get(status, -1))


def _normalize_model_key(model: str) -> str:
    return model.lower().replace("_", "-").rstrip("/")


def _candidate_model_keys(model: str) -> tuple[set[str], set[str]]:
    exact_keys = {_normalize_model_key(model)}
    fallback_keys: set[str] = set()
    try:
        from mflux.models.common.config import ModelConfig
        from mflux.utils.exceptions import ModelConfigError

        model_config = ModelConfig.from_name(model)
    except (ModelConfigError, ValueError):
        return exact_keys, fallback_keys

    exact_keys.add(_normalize_model_key(model_config.model_name))
    if model_config.base_model:
        fallback_keys.add(_normalize_model_key(model_config.base_model))
    return exact_keys, fallback_keys


def _i2i_edit_profile() -> ValidationProfile:
    return ValidationProfile(
        id=I2I_EDIT_5X4_PROFILE_ID,
        title="I2I Edit 5x4 Spaceship Snow Validation",
        canonical_source=CANONICAL_SOURCE,
        description=(
            "Manual visual QA for the spaceship-in-snow I2I profile. The profile separates route support from "
            "release validation and records exact model/package status for edit-reference, latent-img2img, and "
            "multi-reference cells used by the 5x4 contact sheets."
        ),
        records=tuple(_records()),
    )


def _reframe_outpaint_profile() -> ValidationProfile:
    return ValidationProfile(
        id=REFRAME_OUTPAINT_PROFILE_ID,
        title="Reframe And Outpaint Spaceship Validation",
        canonical_source=REFRAME_OUTPAINT_SOURCE,
        description=(
            "Manual visual QA for single-image edit-reference reframe and canvas expansion workflows. "
            "Qwen Image Edit rows remain current for reframe and outpaint. Distilled FLUX.2 Klein 4B/9B "
            "reframe rows remain current, but their 2026-06-08 outpaint artifacts are retained as stale "
            "historical evidence because current strict FLUX.2 outpaint now requires a base Klein model."
        ),
        records=tuple(_reframe_outpaint_records()),
    )


def _flux2_klein_base_starship_profile() -> ValidationProfile:
    return ValidationProfile(
        id=FLUX2_KLEIN_BASE_STARSHIP_PROFILE_ID,
        title="FLUX.2 Klein Base Starship Source Validation",
        canonical_source=FLUX2_KLEIN_BASE_STARSHIP_SOURCE,
        description=(
            "Manual visual QA for the base FLUX.2 Klein source-model starship profile. This profile covers "
            "latent img2img, edit-reference, multi-reference, and strict outpaint on the same cropped "
            "starship source. Published evidence here is source-model only; prepared base q8/q4 packages "
            "share the route surface through capabilities, but their starship contact sheets are still pending."
        ),
        records=tuple(_flux2_klein_base_starship_records()),
    )


def _lora_profiles() -> tuple[ValidationProfile, ...]:
    return (
        _single_record_profile(
            _lora_record(
                profile_id=QWEN_EDIT_Q8_GHIBLI_PROFILE_ID,
                model="AbstractFramework/qwen-image-edit-8bit",
                family="Qwen Image Edit",
                package_variant="q8 prepared",
                public_task="image-to-image",
                mode="edit-reference",
                artifact_path=f"{LORA_VALIDATION_DIR}/qwen_edit_q8_ghibli_trials_contact_sheet.png",
                source_images=(CANONICAL_SOURCE,),
                prompt=(
                    "ghibli style. Transform the source into a whimsical hand-painted animated film "
                    "frame with soft brushwork, warm pastel sky, painterly snow, and gentle storybook "
                    "lighting. Preserve the same spaceship, snowy canyon, wide framing, and overall layout."
                ),
                reviewer_notes=(
                    "PASS on the accepted 2026-06-11 same-seed Ghibli-style A/B proof. The adapter "
                    "matched 1680/1680 tensors, applied 840 targets, and produces a visible style shift "
                    "while keeping the edit route stable."
                ),
            ),
            title="Qwen Image Edit q8 Ghibli-style LoRA Validation",
            canonical_source=CANONICAL_SOURCE,
            description="Accepted single-image edit LoRA proof for the original Qwen Image Edit q8 route.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=QWEN_Q8_CONTROL_LIGHTNING_PROFILE_ID,
                model="AbstractFramework/qwen-image-8bit",
                family="Qwen Image",
                package_variant="q8 prepared",
                public_task="text-to-image",
                mode="text-only",
                artifact_path=f"{QWEN_CONTROL_VALIDATION_DIR}/qwen_q8_control_lightning_contact_sheet.png",
                source_images=(
                    "InstantX/Qwen-Image-ControlNet-Union:conds/canny.png",
                    "InstantX/Qwen-Image-ControlNet-Union:conds/pose.png",
                ),
                prompt=(
                    "Two-condition structured-control proof. Condition A uses the InstantX canny control "
                    "image with the pagoda illustration prompt. Condition B uses the InstantX pose control "
                    "image with the seated portrait prompt. Each row keeps the same prompt, seed, and "
                    "LightX2V 4-step Lightning adapter between the no-control and control columns."
                ),
                reviewer_notes=(
                    "PASS on the 2026-06-15 base q8 structured-control proof. The accepted row uses "
                    "InstantX/Qwen-Image-ControlNet-Union with lightx2v/Qwen-Image-Lightning V2.0 in 4 "
                    "steps. In both the canny pagoda and pose portrait rows, the control image materially "
                    "changes layout while the no-control baseline stays on the same prompt/seed/LoRA."
                ),
                evidence_date="2026-06-15",
            ),
            title="Qwen Image q8 Structured Control Lightning Validation",
            canonical_source="InstantX/Qwen-Image-ControlNet-Union:conds/canny.png",
            description="Exact structured-control LoRA proof for the base Qwen Image q8 route.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=QWEN_Q8_CONTROL_INPAINT_LIGHTNING_PROFILE_ID,
                model="AbstractFramework/qwen-image-8bit",
                family="Qwen Image",
                package_variant="q8 prepared",
                public_task="image-to-image",
                mode="edit-reference",
                artifact_path=f"{QWEN_CONTROL_INPAINT_VALIDATION_DIR}/qwen_control_inpaint_contact_sheet.png",
                source_images=(
                    "docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png",
                    "docs/assets/examples/spaceship-snow/03_i2i_crash_snow.png",
                ),
                prompt=(
                    "Two-condition base-Qwen control-inpaint proof. Condition A intensifies the masked engine area "
                    "into brighter plasma thrusters. Condition B repairs the masked hull and cockpit while keeping "
                    "the rest of the crash scene stable. Both rows keep the same source, mask, prompt, seed, and "
                    "Qwen Lightning adapter between the existing masked-edit comparison and the new base-Qwen route."
                ),
                reviewer_notes=(
                    "PASS on the 2026-06-21 base-Qwen control-inpaint proof. The accepted row uses the exact "
                    "InstantX inpainting ControlNet sidecar with lightx2v/Qwen-Image-Lightning on the prepared q8 "
                    "base package. The published sheet shows acceptable localized engine and repair edits on the "
                    "same source/mask pairs used for the existing Qwen edit masked route."
                ),
                evidence_date="2026-06-21",
            ),
            title="Qwen Image q8 Control-Inpaint Lightning Validation",
            canonical_source="docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png",
            description="Exact base-Qwen control-inpaint LoRA proof for the prepared q8 route.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=QWEN2511_Q8_SINGLE_EDIT_MULTI_ANGLE_PROFILE_ID,
                model="AbstractFramework/qwen-image-edit-2511-8bit",
                family="Qwen Image Edit 2511",
                package_variant="q8 prepared",
                public_task="image-to-image",
                mode="edit-reference",
                artifact_path="docs/assets/validation/lora-2026-06-08/qwen2511-q8-multi-angle-lora-ab-contact-sheet.png",
                source_images=(CANONICAL_SOURCE,),
                prompt="Use the source spaceship as the same object. <sks> back view low-angle shot wide shot.",
                reviewer_notes="PASS on the 2026-06-08 multi-angle A/B proof for the q8 edit row.",
            ),
            title="Qwen Image Edit 2511 q8 Multi-Angle LoRA Validation",
            canonical_source=CANONICAL_SOURCE,
            description="Exact single-image edit LoRA proof for Qwen Image Edit 2511 q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=QWEN2511_Q8_INPAINT_LIGHTNING_PROFILE_ID,
                model="AbstractFramework/qwen-image-edit-2511-8bit",
                family="Qwen Image Edit 2511",
                package_variant="q8 prepared",
                public_task="image-to-image",
                mode="edit-reference",
                artifact_path=f"{QWEN_INPAINT_VALIDATION_DIR}/qwen2511_q8_inpaint_lightning_contact_sheet.png",
                source_images=(
                    "docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png",
                    "docs/assets/examples/spaceship-snow/03_i2i_crash_snow.png",
                ),
                prompt=(
                    "Two-condition masked edit proof. Condition A: intensify the masked engines into brighter "
                    "blue plasma thrusters while preserving the rest of the image. Condition B: repair the "
                    "masked hull and cockpit while preserving the damaged snow scene outside the mask."
                ),
                reviewer_notes=(
                    "PASS on the 2026-06-15 same-seed masked-edit proof. The accepted q8 row uses "
                    "Qwen Image Edit 2511 with --mask-path and the LightX2V 4-step Lightning adapter. "
                    "The public proof publishes two conditions: localized engine enhancement and localized "
                    "crash repair, each against the regular 20-step q8 edit path."
                ),
                evidence_date="2026-06-15",
            ),
            title="Qwen Image Edit 2511 q8 Masked Edit Lightning Validation",
            canonical_source="docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png",
            description="Exact masked-edit LoRA proof for Qwen Image Edit 2511 q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=QWEN2509_Q8_SINGLE_EDIT_MULTI_ANGLE_PROFILE_ID,
                model="AbstractFramework/qwen-image-edit-2509-8bit",
                family="Qwen Image Edit 2509",
                package_variant="q8 prepared",
                public_task="image-to-image",
                mode="edit-reference",
                artifact_path=f"{LORA_VALIDATION_DIR}/qwen2509_q8_multi_angle_ab_contact_sheet.png",
                source_images=(CANONICAL_SOURCE,),
                prompt="Move the camera to the right and keep the same spaceship design and wide scene.",
                reviewer_notes="PASS on the validated Lightning-style 8-step q8 edit proof.",
            ),
            title="Qwen Image Edit 2509 q8 Multi-Angle LoRA Validation",
            canonical_source=CANONICAL_SOURCE,
            description="Exact single-image edit LoRA proof for Qwen Image Edit 2509 q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=QWEN_Q8_REALISM_PROFILE_ID,
                model="AbstractFramework/qwen-image-8bit",
                family="Qwen Image",
                package_variant="q8 prepared",
                public_task="text-to-image",
                mode="text-only",
                artifact_path=f"{LORA_ROUTE_EXPANSION_VALIDATION_DIR}/qwen_q8_text_realism_ab_contact_sheet.png",
                source_images=(),
                prompt=(
                    "Realism portrait of a young woman of African descent standing in a sunlit park, arms crossed, "
                    "dramatic natural lighting, three-quarter view, delicate jewelry, loose shoulder-length curls, "
                    "natural skin texture, environmental portrait photography."
                ),
                reviewer_notes=(
                    "PASS on the exact base-Qwen q8 text-to-image proof. The same-seed realism adapter shifts the "
                    "baseline into a tighter, more photographic portrait without route drift."
                ),
                evidence_date="2026-06-22",
            ),
            title="Qwen Image q8 Realism LoRA Validation",
            canonical_source=f"{LORA_ROUTE_EXPANSION_VALIDATION_DIR}/qwen_q8_text_realism_ab_contact_sheet.png",
            description="Exact text-to-image LoRA proof for base Qwen Image q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=QWEN_Q8_LATENT_REALISM_PROFILE_ID,
                model="AbstractFramework/qwen-image-8bit",
                family="Qwen Image",
                package_variant="q8 prepared",
                public_task="image-to-image",
                mode="latent-img2img",
                artifact_path=f"{LORA_ROUTE_EXPANSION_VALIDATION_DIR}/qwen_q8_latent_studio_cfg_auto_contact_sheet.png",
                source_images=(
                    f"{LORA_ROUTE_EXPANSION_VALIDATION_DIR}/qwen_q8_latent_source_portrait_illustration.png",
                ),
                prompt=(
                    "Studio Realism, photorealistic portrait of the same young woman of African descent standing "
                    "in the same sunlit park with arms crossed, the same loose shoulder-length curls, the same "
                    "pendant necklace, and the same sleeveless taupe dress. Preserve the same pose, framing, and "
                    "background layout. Natural skin texture, realistic hair strands, subtle outdoor depth of "
                    "field, no text."
                ),
                reviewer_notes=(
                    "PASS on the exact base-Qwen q8 latent img2img proof. The accepted row uses "
                    "prithivMLmods/Qwen-Image-Studio-Realism with the upstream-style blank negative-prompt CFG "
                    "contract, preserves the same pose and park layout, and produces a visibly more photographic portrait."
                ),
                evidence_date="2026-06-22",
            ),
            title="Qwen Image q8 Latent Studio Realism LoRA Validation",
            canonical_source=f"{LORA_ROUTE_EXPANSION_VALIDATION_DIR}/qwen_q8_latent_source_portrait_illustration.png",
            description="Exact latent img2img LoRA proof for base Qwen Image q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=QWEN2511_Q8_REFRAME_MULTI_ANGLE_PROFILE_ID,
                model="AbstractFramework/qwen-image-edit-2511-8bit",
                family="Qwen Image Edit 2511",
                package_variant="q8 prepared",
                public_task="image-to-image",
                mode="edit-reference",
                artifact_path=f"{LORA_ROUTE_EXPANSION_VALIDATION_DIR}/qwen2511_q8_reframe_multi_angle_exact_contact_sheet.png",
                source_images=("docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png",),
                prompt=(
                    "Use the expanded canvas to reveal the same spaceship as a back view low-angle wide shot. "
                    "<sks> back view low-angle shot wide shot. Keep the same silver starship identity, snowy canyon "
                    "environment, and coherent wide framing. Show the rear engines and tail from behind, keep the "
                    "scene sharp, and do not add a second ship, text, or border."
                ),
                reviewer_notes=(
                    "PASS on the exact q8 reframe A/B proof. The accepted row uses "
                    "lightx2v/Qwen-Image-Edit-2511-Lightning plus fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA "
                    "at scale 1.25, and it produces a materially stronger rear-view reframing than the Lightning-only baseline."
                ),
                evidence_date="2026-06-22",
            ),
            title="Qwen Image Edit 2511 q8 Reframe Multi-Angle LoRA Validation",
            canonical_source="docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png",
            description="Exact generative reframe LoRA proof for Qwen Image Edit 2511 q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=QWEN2511_Q8_OUTPAINT_MULTI_ANGLE_PROFILE_ID,
                model="AbstractFramework/qwen-image-edit-2511-8bit",
                family="Qwen Image Edit 2511",
                package_variant="q8 prepared",
                public_task="image-to-image",
                mode="edit-reference",
                artifact_path=f"{LORA_ROUTE_EXPANSION_VALIDATION_DIR}/qwen2511_q8_outpaint_multiangle_exact_contact_sheet.png",
                source_images=("docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png",),
                prompt=(
                    "Outpaint this cropped starship image into a back view low-angle wide shot of the same spacecraft "
                    "in the snowy canyon. <sks> back view low-angle shot wide shot. Keep the same silver starship "
                    "identity, snowy canyon environment, and coherent wide framing. Reveal the rear engines and tail "
                    "in the new space, keep it sharp, and do not add a second ship, text, or border."
                ),
                reviewer_notes=(
                    "PASS on the exact q8 outpaint A/B proof. The accepted row uses "
                    "lightx2v/Qwen-Image-Edit-2511-Lightning plus fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA "
                    "and produces a cleaner rear-view canvas extension than the Lightning-only baseline."
                ),
                evidence_date="2026-06-22",
            ),
            title="Qwen Image Edit 2511 q8 Outpaint Multi-Angle LoRA Validation",
            canonical_source="docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png",
            description="Exact canvas outpaint LoRA proof for Qwen Image Edit 2511 q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=QWEN2511_Q8_MULTI_REFERENCE_MULTI_ANGLE_PROFILE_ID,
                model="AbstractFramework/qwen-image-edit-2511-8bit",
                family="Qwen Image Edit 2511",
                package_variant="q8 prepared",
                public_task="image-to-image",
                mode="multi-reference",
                artifact_path=f"{LORA_ROUTE_EXPANSION_VALIDATION_DIR}/qwen2511_q8_multi_reference_multiangle_exact_contact_sheet.png",
                source_images=(
                    "docs/assets/validation/qwen-edit-2511-parity-2026-06-06/qwen2511-source-pencil.png",
                    "docs/assets/validation/qwen-edit-2511-parity-2026-06-06/qwen2511-source-crash.png",
                ),
                prompt=(
                    "Use the first image as the graphite pencil sketch style reference and the second image as the "
                    "hard-landing crash content reference. <sks> back view low-angle shot wide shot. Produce one "
                    "coherent wide image of the same spaceship crashed in the snowy canyon from behind at a low "
                    "camera angle: graphite pencil outlines on white paper, visible tilted hull, disturbed snow, "
                    "broken ice chunks, scrape trail, and a thin smoke plume. Preserve the spaceship identity and "
                    "canyon layout. No blur, no colored photo, no text."
                ),
                reviewer_notes=(
                    "PASS on the exact q8 multi-reference A/B proof. The accepted row uses "
                    "lightx2v/Qwen-Image-Edit-2511-Lightning plus fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA "
                    "and shifts the composition to a clearer rear-view while keeping the two-reference pencil-crash contract coherent."
                ),
                evidence_date="2026-06-22",
            ),
            title="Qwen Image Edit 2511 q8 Multi-Reference Multi-Angle LoRA Validation",
            canonical_source="docs/assets/validation/qwen-edit-2511-parity-2026-06-06/qwen2511-source-pencil.png",
            description="Exact multi-reference LoRA proof for Qwen Image Edit 2511 q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=QWEN2512_Q8_PIXEL_ART_PROFILE_ID,
                model="AbstractFramework/qwen-image-2512-8bit",
                family="Qwen Image 2512",
                package_variant="q8 prepared",
                public_task="text-to-image",
                mode="text-only",
                artifact_path=f"{LORA_VALIDATION_DIR}/qwen2512_q8_pixel_art_ab_contact_sheet.png",
                source_images=(),
                prompt="Pixel Art, a pixelated image of a space astronaut floating in zero gravity.",
                reviewer_notes="PASS on the q8 pixel-art A/B proof.",
            ),
            title="Qwen Image 2512 q8 Pixel-Art LoRA Validation",
            canonical_source=f"{LORA_VALIDATION_DIR}/qwen2512_q8_pixel_art_ab_contact_sheet.png",
            description="Exact text-to-image LoRA proof for Qwen Image 2512 q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=ZIMAGE_Q8_TECHNICALLYCOLOR_PROFILE_ID,
                model="AbstractFramework/z-image-turbo-8bit",
                family="Z-Image Turbo",
                package_variant="q8 prepared",
                public_task="text-to-image",
                mode="text-only",
                artifact_path=f"{LORA_VALIDATION_DIR}/zimage_q8_technically_color_ab_contact_sheet.png",
                source_images=(),
                prompt="t3chnic4lly vibrant 1960s close-up portrait by a lake.",
                reviewer_notes="PASS on the q8 Technically Color A/B proof.",
            ),
            title="Z-Image Turbo q8 Technically Color LoRA Validation",
            canonical_source=f"{LORA_VALIDATION_DIR}/zimage_q8_technically_color_ab_contact_sheet.png",
            description="Exact text-to-image LoRA proof for Z-Image Turbo q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=ZIMAGE_Q8_CHILDRENS_DRAWINGS_LATENT_PROFILE_ID,
                model="AbstractFramework/z-image-turbo-8bit",
                family="Z-Image Turbo",
                package_variant="q8 prepared",
                public_task="image-to-image",
                mode="latent-img2img",
                artifact_path=f"{ZIMAGE_LATENT_LORA_VALIDATION_DIR}/zimage_q8_latent_childdraw_contact_sheet.png",
                source_images=(CANONICAL_SOURCE,),
                prompt=(
                    "Turn this same spaceship in the snow into a childs wax-crayon drawing on white paper. "
                    "Preserve the exact camera angle, ship position, snowy canyon layout, and single ship "
                    "silhouette. Use thick uneven crayon lines, simple childlike shapes, and flat hand-colored fills."
                ),
                reviewer_notes=(
                    "PASS on the exact q8 latent img2img A/B proof. The accepted row keeps the same snow-canyon "
                    "layout and ship silhouette while the children's-drawing adapter adds a clear hand-drawn "
                    "crayon treatment."
                ),
                evidence_date="2026-06-24",
            ),
            title="Z-Image Turbo q8 Latent Children's-Drawings LoRA Validation",
            canonical_source=CANONICAL_SOURCE,
            description="Exact latent img2img LoRA proof for Z-Image Turbo q8 with a same-source style-transfer A/B.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=FLUX2_KLEIN9B_Q8_CONSISTENCY_EDIT_PROFILE_ID,
                model="AbstractFramework/flux.2-klein-9b-8bit",
                family="FLUX.2 Klein 9B",
                package_variant="q8 prepared",
                public_task="image-to-image",
                mode="edit-reference",
                artifact_path=f"{LORA_VALIDATION_DIR}/flux2_klein9b_q8_consistency_ab_contact_sheet.png",
                source_images=(CANONICAL_SOURCE,),
                prompt="Edit the source into the same spaceship after a hard landing in the snow at blue hour.",
                reviewer_notes="PASS on the q8 consistency-edit A/B proof.",
            ),
            title="FLUX.2 Klein 9B q8 Consistency LoRA Validation",
            canonical_source=CANONICAL_SOURCE,
            description="Exact single-image edit LoRA proof for FLUX.2 Klein 9B q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=FLUX2_KLEIN9B_Q8_MULTI_REFERENCE_MIGRATION_PROFILE_ID,
                model="AbstractFramework/flux.2-klein-9b-8bit",
                family="FLUX.2 Klein 9B",
                package_variant="q8 prepared",
                public_task="image-to-image",
                mode="multi-reference",
                artifact_path=f"{LORA_ROUTE_EXPANSION_VALIDATION_DIR}/flux2_klein9b_q8_multiref_exact_contact_sheet.png",
                source_images=(
                    "docs/assets/validation/i2i-edit-5x4-2026-06-05/reference-inputs/flux2_klein_9b_8bit_d_pencil_crash.png",
                    "docs/assets/validation/i2i-edit-5x4-2026-06-05/reference-inputs/flux2_klein_9b_8bit_b_cinematic.png",
                ),
                prompt=(
                    "Use the first image as the pencil-crash structure and the second image as the cinematic lighting "
                    "and material reference. Produce one coherent image of the same hard-landed spaceship in the snow "
                    "with graphite sketch lines, subtle metallic blue-hour shading, stable canyon layout, and visible "
                    "crash debris. Preserve the same front-left crop and do not add extra ships or text."
                ),
                reviewer_notes=(
                    "PASS on the exact FLUX.2 Klein 9B q8 multi-reference A/B proof. "
                    "dx8152/Flux2-Klein-9B-Migration preserves the two-reference crash composition while producing a visibly stronger migration toward the intended sketched crash treatment."
                ),
                evidence_date="2026-06-22",
            ),
            title="FLUX.2 Klein 9B q8 Multi-Reference Migration LoRA Validation",
            canonical_source="docs/assets/validation/i2i-edit-5x4-2026-06-05/reference-inputs/flux2_klein_9b_8bit_d_pencil_crash.png",
            description="Exact multi-reference LoRA proof for FLUX.2 Klein 9B q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=FLUX2_KLEIN_BASE4B_Q8_OUTPAINT_PROFILE_ID,
                model="AbstractFramework/flux.2-klein-base-4b-8bit",
                family="FLUX.2 Klein Base 4B",
                package_variant="q8 prepared",
                public_task="image-to-image",
                mode="edit-reference",
                artifact_path=f"{LORA_ROUTE_EXPANSION_VALIDATION_DIR}/flux2_klein_base4b_q8_outpaint_route_exact_contact_sheet.png",
                source_images=("docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png",),
                prompt="Fill the green spaces according to the image",
                reviewer_notes=(
                    "PASS on the exact FLUX.2 Klein base 4B q8 strict-outpaint proof. "
                    "fal/flux-2-klein-4B-outpaint-lora applies cleanly after normalized key matching and improves the exact green-canvas outpaint route over the no-LoRA baseline at the accepted seed."
                ),
                evidence_date="2026-06-22",
            ),
            title="FLUX.2 Klein Base 4B q8 Outpaint LoRA Validation",
            canonical_source="docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png",
            description="Exact strict-outpaint LoRA proof for FLUX.2 Klein base 4B q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=ERNIE_TURBO_Q8_ANIME_STYLE_PROFILE_ID,
                model="AbstractFramework/ernie-image-turbo-8bit",
                family="ERNIE Image Turbo",
                package_variant="q8 prepared",
                public_task="text-to-image",
                mode="text-only",
                artifact_path=f"{LORA_VALIDATION_DIR}/ernie_turbo_q8_anime_style_ab_contact_sheet.png",
                source_images=(),
                prompt="elusarca anime style, a young woman with silver hair and a red trench coat.",
                reviewer_notes="PASS on the q8 anime-style A/B proof.",
            ),
            title="ERNIE Image Turbo q8 Anime-Style LoRA Validation",
            canonical_source=f"{LORA_VALIDATION_DIR}/ernie_turbo_q8_anime_style_ab_contact_sheet.png",
            description="Exact text-to-image LoRA proof for ERNIE Image Turbo q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=ERNIE_TURBO_Q8_ANIME_STYLE_LATENT_PROFILE_ID,
                model="AbstractFramework/ernie-image-turbo-8bit",
                family="ERNIE Image Turbo",
                package_variant="q8 prepared",
                public_task="image-to-image",
                mode="latent-img2img",
                artifact_path=f"{LORA_ROUTE_EXPANSION_VALIDATION_DIR}/ernie_turbo_q8_latent_anime_style_ab_contact_sheet.png",
                source_images=(f"{LORA_ROUTE_EXPANSION_VALIDATION_DIR}/qwen_q8_text_realism_no_lora.png",),
                prompt=(
                    "elusarca anime style, portrait of a young woman of African descent standing in a sunlit park, "
                    "arms crossed, delicate jewelry, loose shoulder-length curls, preserve the same pose and park "
                    "layout, polished anime illustration, crisp linework, soft luminous skin, cinematic outdoor light."
                ),
                reviewer_notes=(
                    "PASS on the q8 latent img2img proof. The same-source same-seed pair preserves the portrait setup "
                    "while the adapter pushes the face, hair, and shading into a clearly illustrated anime treatment."
                ),
                evidence_date="2026-06-22",
            ),
            title="ERNIE Image Turbo q8 Latent Anime-Style LoRA Validation",
            canonical_source=f"{LORA_ROUTE_EXPANSION_VALIDATION_DIR}/qwen_q8_text_realism_no_lora.png",
            description="Exact latent img2img LoRA proof for ERNIE Image Turbo q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=WAN_TI2V5B_Q8_HSTORIC_T2V_PROFILE_ID,
                model="AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit",
                family="Wan2.2 TI2V-5B",
                package_variant="q8 prepared",
                public_task="text-to-video",
                mode="text-only",
                artifact_path=f"{WAN_LORA_VALIDATION_DIR}/ti2v_t2v_hstoric_ab_contact_sheet.jpg",
                source_images=(),
                prompt="HST style HD film, early 1900s, autochrome, analog cinema. A horse-drawn carriage crossing a snowy town square at dusk.",
                reviewer_notes="PASS on the q8 TI2V-5B text-to-video LoRA proof.",
            ),
            title="Wan2.2 TI2V-5B q8 Text-to-Video LoRA Validation",
            canonical_source=f"{WAN_LORA_VALIDATION_DIR}/ti2v_t2v_hstoric_ab_contact_sheet.jpg",
            description="Exact text-to-video LoRA proof for Wan2.2 TI2V-5B q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=WAN_TI2V5B_Q8_CRUSHIT_I2V_PROFILE_ID,
                model="AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit",
                family="Wan2.2 TI2V-5B",
                package_variant="q8 prepared",
                public_task="image-to-video",
                mode="first-frame-i2v",
                artifact_path=f"{WAN_LORA_VALIDATION_DIR}/ti2v_i2v_crushit_ab_contact_sheet.jpg",
                source_images=(),
                prompt="crush it. An invisible hydraulic press crushes the centered aluminum soda can flat on the clean studio floor.",
                reviewer_notes="PASS on the q8 TI2V-5B first-frame image-to-video LoRA proof.",
            ),
            title="Wan2.2 TI2V-5B q8 First-Frame Image-to-Video LoRA Validation",
            canonical_source=f"{WAN_LORA_VALIDATION_DIR}/ti2v_i2v_crushit_ab_contact_sheet.jpg",
            description="Exact first-frame image-to-video LoRA proof for Wan2.2 TI2V-5B q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=WAN_A14B_Q8_FOLLOWCAM_T2V_PROFILE_ID,
                model="AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
                family="Wan2.2 T2V-A14B",
                package_variant="q8 prepared",
                public_task="text-to-video",
                mode="text-only",
                artifact_path=f"{WAN_LORA_VALIDATION_DIR}/a14b_t2v_followcam_ab_contact_sheet.jpg",
                source_images=(),
                prompt="FollowCam. A continuous wide-angle shot as we follow a rider on horseback galloping through a foggy meadow at dawn.",
                reviewer_notes="PASS on the q8 T2V-A14B text-to-video LoRA proof.",
            ),
            title="Wan2.2 T2V-A14B q8 Text-to-Video LoRA Validation",
            canonical_source=f"{WAN_LORA_VALIDATION_DIR}/a14b_t2v_followcam_ab_contact_sheet.jpg",
            description="Exact text-to-video LoRA proof for Wan2.2 T2V-A14B q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=WAN_A14B_Q8_LIGHTX2V_4STEP_T2V_PROFILE_ID,
                model="AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
                family="Wan2.2 T2V-A14B",
                package_variant="q8 prepared",
                public_task="text-to-video",
                mode="text-only",
                artifact_path=f"{LIGHTX2V_WAN_4STEP_VALIDATION_DIR}/a14b_t2v_lightx2v_4step_ab_contact_sheet.jpg",
                source_images=(),
                prompt=(
                    "A cinematic wide-angle movie shot of a massive futuristic starship taking off from a "
                    "frozen tundra. The ship features sleek dark metallic armor. Two massive warp nacelles "
                    "pulse with bright blue plasma. Violent snow squalls whip around the hull. The camera "
                    "slowly tilts up as the thrusters ignite and massive clouds of snow blast away from the "
                    "launch pad. Photorealistic, highly detailed, dramatic lighting."
                ),
                reviewer_notes=(
                    "PASS on the q8 LightX2V Lightning 4-step A/B proof. The 4-step no-LoRA baseline is a "
                    "muddy silhouette while the paired high-noise/low-noise Lightning files produce a stable "
                    "starship takeoff shot."
                ),
                evidence_date="2026-06-12",
            ),
            title="Wan2.2 T2V-A14B q8 LightX2V 4-Step Validation",
            canonical_source=f"{LIGHTX2V_WAN_4STEP_VALIDATION_DIR}/a14b_t2v_lightx2v_4step_ab_contact_sheet.jpg",
            description="Exact 4-step LightX2V Lightning text-to-video proof for Wan2.2 T2V-A14B q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=WAN_A14B_Q8_LIGHTNING_V2V_PROFILE_ID,
                model="AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
                family="Wan2.2 T2V-A14B",
                package_variant="q8 prepared",
                public_task="video-to-video",
                mode="latent-video",
                artifact_path=f"{LIGHTNING_V2V_VALIDATION_DIR}/lightning_v2v_matrix_comparison.png",
                source_images=(),
                prompt=(
                    "A realistic wide shot of a woman giving a talk on a conference stage. Keep the exact same "
                    "stage, wooden podium with microphones, large bright presentation screen with the same logo, "
                    "conference banner, warm stage lighting, camera framing, and natural speaking gestures."
                ),
                reviewer_notes=(
                    "PASS on the bounded Lightning video-to-video matrix: Seko-V1.1 across two seeds and two "
                    "clips (conference 480x832x25f, ship 448x256x17f), Seko-V2.0 one seed, plus the masked "
                    "combination whose preserved regions measured 1.89 mean delta at the 1.92 H.264 re-encode "
                    "floor. On-grid recipe: steps 4, video_strength 0.75, guidance 1/1, flow_shift 5, unipc."
                ),
                evidence_date="2026-07-04",
            ),
            title="Wan2.2 T2V-A14B q8 Lightning Video-to-Video Validation",
            canonical_source=f"{LIGHTNING_V2V_VALIDATION_DIR}/lightning_v2v_matrix_comparison.png",
            description="Bounded Lightning 4-step video-to-video matrix proof for Wan2.2 T2V-A14B q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=WAN_A14B_Q8_ORBIT_I2V_PROFILE_ID,
                model="AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit",
                family="Wan2.2 I2V-A14B",
                package_variant="q8 prepared",
                public_task="image-to-video",
                mode="first-frame-i2v",
                artifact_path=f"{WAN_LORA_VALIDATION_DIR}/a14b_i2v_orbit_spaceship_ab_contact_sheet.jpg",
                source_images=(CANONICAL_SOURCE,),
                prompt="orbit 360 around the landed silver spaceship in the snowy canyon.",
                reviewer_notes="PASS on the q8 I2V-A14B first-frame image-to-video LoRA proof.",
            ),
            title="Wan2.2 I2V-A14B q8 First-Frame Image-to-Video LoRA Validation",
            canonical_source=CANONICAL_SOURCE,
            description="Exact first-frame image-to-video LoRA proof for Wan2.2 I2V-A14B q8.",
        ),
        _single_record_profile(
            _lora_record(
                profile_id=WAN_A14B_Q8_LIGHTX2V_4STEP_I2V_PROFILE_ID,
                model="AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit",
                family="Wan2.2 I2V-A14B",
                package_variant="q8 prepared",
                public_task="image-to-video",
                mode="first-frame-i2v",
                artifact_path=f"{LIGHTX2V_WAN_4STEP_VALIDATION_DIR}/a14b_i2v_lightx2v_4step_ab_contact_sheet.jpg",
                source_images=(CANONICAL_SOURCE,),
                prompt=(
                    "Starting from the input image, the silver spaceship powers up and lifts off from the "
                    "frozen ground. Blue engines brighten, snow blasts outward, vapor rolls under the hull, "
                    "and the camera holds the same wide icy canyon framing while the ship rises smoothly."
                ),
                reviewer_notes=(
                    "PASS on the q8 LightX2V Lightning 4-step A/B proof. The 4-step no-LoRA baseline stays "
                    "mostly static and foggy, while the paired Lightning files preserve the source layout and "
                    "produce a clear lift-off."
                ),
                evidence_date="2026-06-12",
            ),
            title="Wan2.2 I2V-A14B q8 LightX2V 4-Step Validation",
            canonical_source=CANONICAL_SOURCE,
            description="Exact 4-step LightX2V Lightning first-frame image-to-video proof for Wan2.2 I2V-A14B q8.",
        ),
    )


def _single_record_profile(
    record: ValidationRecord,
    *,
    title: str,
    canonical_source: str,
    description: str,
) -> ValidationProfile:
    return ValidationProfile(
        id=record.profile_id,
        title=title,
        canonical_source=canonical_source,
        description=description,
        records=(record,),
    )


def _lora_record(
    *,
    profile_id: str,
    model: str,
    family: str,
    package_variant: str,
    public_task: str,
    mode: str,
    artifact_path: str,
    source_images: tuple[str, ...],
    prompt: str,
    reviewer_notes: str,
    evidence_date: str = "2026-06-11",
) -> ValidationRecord:
    return ValidationRecord(
        profile_id=profile_id,
        model=model,
        family=family,
        package_variant=package_variant,
        step="L",
        step_label="LoRA A/B validation",
        public_task=public_task,
        mode=mode,
        status=STATUS_PASS,
        artifact_path=artifact_path,
        source_images=source_images,
        prompt=prompt,
        reviewer_notes=reviewer_notes,
        evidence_date=evidence_date,
    )


def _zimage_inpaint_profile() -> ValidationProfile:
    return ValidationProfile(
        id=ZIMAGE_INPAINT_PROFILE_ID,
        title="Z-Image Turbo q8 Native Inpaint Validation",
        canonical_source="docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png",
        description=(
            "Manual visual QA for the first narrow Z-Image Turbo native inpaint proof. The public validation "
            "surface compares the accepted masked engine edit against a same-prompt same-seed latent img2img "
            "baseline so the route difference is visible."
        ),
        records=(
            ValidationRecord(
                profile_id=ZIMAGE_INPAINT_PROFILE_ID,
                model="AbstractFramework/z-image-turbo-8bit",
                family="Z-Image Turbo",
                package_variant="q8 prepared",
                step="ENGINE",
                step_label="engine localized inpaint",
                public_task="image-to-image",
                mode="edit-reference",
                status=STATUS_PASS,
                artifact_path=f"{ZIMAGE_INPAINT_VALIDATION_DIR}/zimage_inpaint_contact_sheet.png",
                source_images=("docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png",),
                prompt=(
                    "Keep the same silver spaceship, icy canyon, and sunrise lighting. Only inside the masked "
                    "engine area, intensify both blue engines into brighter plasma thrusters, add dense blue "
                    "glow and snow vapor around the thrusters, and preserve the rest of the image unchanged."
                ),
                reviewer_notes=(
                    "PASS on the 2026-06-21 narrow native-inpaint proof. The accepted public row is the exact "
                    "AbstractFramework/z-image-turbo-8bit package on the engine mask example, with a published "
                    "same-prompt same-seed latent baseline and a masked-area crop sheet."
                ),
                evidence_date="2026-06-21",
            ),
        ),
    )


def _records() -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    records.extend(_fibo_records())
    records.extend(
        _flux2_records(
            family="FLUX.2 Klein 4B",
            source_model="black-forest-labs/FLUX.2-klein-4B",
            source_slug="flux2_klein_4b_source",
            q8_model="AbstractFramework/flux.2-klein-4b-8bit",
            q8_slug="flux2_klein_4b_8bit",
            q4_model="AbstractFramework/flux.2-klein-4b-4bit",
            q4_slug="flux2_klein_4b_4bit",
        )
    )
    records.extend(
        _flux2_records(
            family="FLUX.2 Klein 9B",
            source_model="black-forest-labs/FLUX.2-klein-9B",
            source_slug="flux2_klein_9b_source",
            q8_model="AbstractFramework/flux.2-klein-9b-8bit",
            q8_slug="flux2_klein_9b_8bit",
            q4_model="AbstractFramework/flux.2-klein-9b-4bit",
            q4_slug="flux2_klein_9b_4bit",
        )
    )
    records.extend(_qwen2509_records())
    records.extend(_qwen2511_records())
    return records


QWEN_REFRAME_PROMPT = (
    "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic "
    "silver starship in the snowy alien plain, including the nose, full hull, both engines, landing "
    "legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not "
    "crop the ship. Keep it sharp and centered in a coherent wide frame."
)

QWEN_OUTPAINT_PROMPT = (
    "Outpaint this close cropped starship image into a much wider realistic shot of the full "
    "spacecraft in the snowy canyon. Keep the existing central spacecraft surface consistent, and "
    "complete the missing nose, full hull, tail, engines, snow field, and ice cliffs in the newly "
    "added space. The entire ship must fit inside the final wide frame with empty snow visible "
    "around it. Preserve the same lighting and camera angle. No text, no frame, no border, no "
    "duplicate ship."
)

QWEN2511_OUTPAINT_PROMPT = (
    "Outpaint this close cropped image into a wider realistic snowy canyon shot while keeping the "
    "same compact pod-like silver starship design from the source. Complete the missing nose, "
    "rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly "
    "added space. The final ship must remain a compact rounded spacecraft, not an airplane, with no "
    "large wings. Preserve the same lighting and camera angle. No text, no frame, no border, no "
    "duplicate ship."
)

FLUX_REFRAME_PROMPT = (
    "Generatively reframe this close-up into a wider establishing shot. Reveal the entire "
    "futuristic silver starship in the snowy alien plain, including the nose, full hull, both "
    "engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and "
    "material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame. No "
    "duplicated spacecraft, no text, no border."
)

FLUX9_REFRAME_PROMPT = (
    "Zoom out from the source image into a wider snowy canyon view while keeping the exact same "
    "visible spacecraft design: a smooth silver sci-fi hull seen from the side, pointed nose on "
    "the left, one large circular black side engine intake, rounded metal body, short landing legs, "
    "and snowy canyon background. Use the larger canvas to reveal the missing rear, tail, full "
    "hull, surrounding snow, and ice cliffs. Keep the original side-view camera angle. Do not "
    "redesign it as an airplane, do not add long wings, propellers, or a front-facing cockpit "
    "aircraft view. No duplicate ship, no text, no border."
)

FLUX_OUTPAINT_PROMPT = (
    "Outpaint this close cropped starship image into a much wider realistic shot of the full "
    "spacecraft in the snowy canyon. Keep the existing compact silver spacecraft consistent, "
    "complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and "
    "ice cliffs in the newly added space. The entire ship must fit inside the final wide frame. No "
    "duplicated spacecraft, no repeated mountains, no text, no border."
)

BASE_STARSHIP_LATENT_PROMPT = (
    "Transform this exact close-up starship crop into a clearly darker polar night scene with a "
    "visible teal aurora over the peaks. Preserve the exact crop, camera angle, fuselage shape, "
    "engines, snow field, and ice cliffs. Make the sky distinctly navy, deepen the snow shadows, "
    "and add obvious cyan aurora reflections across the silver metal. No damage, no extra ships, "
    "no text."
)

BASE_STARSHIP_DAMAGE_PROMPT = (
    "Edit this same close-up starship into a hard-landed damaged version. Preserve the exact crop, "
    "camera angle, fuselage shape, cockpit, engine positions, snow field, and ice cliffs. Add "
    "scraped metal, bent panels, soot near the intakes, a thin smoke plume, disturbed snow, and a "
    "shallow impact groove. Keep the ship sharp and coherent. No extra ships, no blur, no text."
)

BASE_STARSHIP_SKETCH_PROMPT = (
    "Turn this same close-up starship scene into a clean graphite pencil sketch. Preserve the exact "
    "crop, camera angle, fuselage shape, cockpit, engines, snow field, and ice cliffs. Use white "
    "paper, precise line art, subtle shading, no color fill, no blur, no text."
)

BASE_STARSHIP_MULTIREF_9B_PROMPT = (
    "Use the first image as the structural line-art reference and the second image as the lighting "
    "and material reference. Produce one coherent close-up of the same starship scene: graphite "
    "line art with cool aurora metallic reflections and darker polar-night shading, the same crop, "
    "same fuselage and engines, same snow field and ice cliffs, no extra ships, no text."
)

BASE_STARSHIP_MULTIREF_4B_PROMPT = (
    "Use the first image as the structural graphite sketch reference and the second image as the "
    "metallic material and lighting reference. Produce one coherent close-up of the same starship "
    "scene. Preserve the exact crop including the nose, cockpit edge, front engine, snow field, "
    "and ice cliffs. Keep clean graphite lines with subtle metallic shading. No extra ships, no text."
)


def _reframe_outpaint_records() -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    specs = (
        (
            "Qwen Image Edit",
            QWEN_REFRAME_PROMPT,
            QWEN_OUTPAINT_PROMPT,
            8201,
            8212,
            20,
            4,
            "25%,50%,25%,50%",
            "5%,80%,5%,60%",
            (
                (
                    "source",
                    "Qwen/Qwen-Image-Edit",
                    "qwen_edit_source_reframe_b.png",
                    "qwen_edit_source_outpaint_b_wide.png",
                ),
                (
                    "q8 prepared",
                    "AbstractFramework/qwen-image-edit-8bit",
                    "qwen_edit_q8_reframe_b.png",
                    "qwen_edit_q8_outpaint_b.png",
                ),
                (
                    "q4 prepared",
                    "AbstractFramework/qwen-image-edit-4bit",
                    "qwen_edit_q4_reframe_b.png",
                    "qwen_edit_q4_outpaint_b.png",
                ),
            ),
        ),
        (
            "Qwen Image Edit 2509",
            QWEN_REFRAME_PROMPT,
            QWEN_OUTPAINT_PROMPT,
            8301,
            8312,
            20,
            4,
            "25%,50%,25%,50%",
            "5%,80%,5%,60%",
            (
                (
                    "source",
                    "Qwen/Qwen-Image-Edit-2509",
                    "qwen2509_source_reframe_b.png",
                    "qwen2509_source_outpaint_b.png",
                ),
                (
                    "q8 prepared",
                    "AbstractFramework/qwen-image-edit-2509-8bit",
                    "qwen2509_q8_reframe_b.png",
                    "qwen2509_q8_outpaint_b.png",
                ),
                (
                    "q4 prepared",
                    "AbstractFramework/qwen-image-edit-2509-4bit",
                    "qwen2509_q4_reframe_b.png",
                    "qwen2509_q4_outpaint_b.png",
                ),
            ),
        ),
        (
            "Qwen Image Edit 2511",
            QWEN_REFRAME_PROMPT,
            QWEN2511_OUTPAINT_PROMPT,
            8401,
            8413,
            20,
            4,
            "25%,50%,25%,50%",
            "5%,80%,5%,60%",
            (
                (
                    "source",
                    "Qwen/Qwen-Image-Edit-2511",
                    "qwen2511_source_reframe_b.png",
                    "qwen2511_source_outpaint_b_retry_compact.png",
                ),
                (
                    "q8 prepared",
                    "AbstractFramework/qwen-image-edit-2511-8bit",
                    "qwen2511_q8_reframe_b.png",
                    "qwen2511_q8_outpaint_b.png",
                ),
                (
                    "q4 prepared",
                    "AbstractFramework/qwen-image-edit-2511-4bit",
                    "qwen2511_q4_reframe_b.png",
                    "qwen2511_q4_outpaint_b.png",
                ),
            ),
        ),
        (
            "FLUX.2 Klein 4B",
            FLUX_REFRAME_PROMPT,
            FLUX_OUTPAINT_PROMPT,
            8501,
            8512,
            16,
            1,
            "25%,50%,25%,50%",
            "5%,80%,5%,60%",
            (
                (
                    "source",
                    "black-forest-labs/FLUX.2-klein-4B",
                    "flux2_4b_source_reframe_b.png",
                    "flux2_4b_source_outpaint_b.png",
                ),
                (
                    "q8 prepared",
                    "AbstractFramework/flux.2-klein-4b-8bit",
                    "flux2_4b_q8_reframe_b.png",
                    "flux2_4b_q8_outpaint_b.png",
                ),
                (
                    "q4 prepared",
                    "AbstractFramework/flux.2-klein-4b-4bit",
                    "flux2_4b_q4_reframe_b.png",
                    "flux2_4b_q4_outpaint_b.png",
                ),
            ),
        ),
        (
            "FLUX.2 Klein 9B",
            FLUX9_REFRAME_PROMPT,
            FLUX_OUTPAINT_PROMPT,
            8604,
            8612,
            16,
            1,
            "25%,80%,25%,60%",
            "5%,80%,5%,60%",
            (
                (
                    "source",
                    "black-forest-labs/FLUX.2-klein-9B",
                    "flux2_9b_source_reframe_b_wide_anchors.png",
                    "flux2_9b_source_outpaint_b.png",
                ),
                (
                    "q8 prepared",
                    "AbstractFramework/flux.2-klein-9b-8bit",
                    "flux2_9b_q8_reframe_b.png",
                    "flux2_9b_q8_outpaint_b.png",
                ),
                (
                    "q4 prepared",
                    "AbstractFramework/flux.2-klein-9b-4bit",
                    "flux2_9b_q4_reframe_b.png",
                    "flux2_9b_q4_outpaint_b.png",
                ),
            ),
        ),
    )
    for (
        family,
        reframe_prompt,
        outpaint_prompt,
        reframe_seed,
        outpaint_seed,
        steps,
        guidance,
        reframe_padding,
        outpaint_padding,
        variants,
    ) in specs:
        for package_variant, model, reframe_output, outpaint_output in variants:
            is_distilled_flux = "FLUX.2 Klein" in family and "base" not in model.lower()
            records.append(
                _reframe_outpaint_record(
                    model=model,
                    family=family,
                    package_variant=package_variant,
                    step="RF",
                    step_label="generative reframe",
                    artifact_file=reframe_output,
                    prompt=reframe_prompt,
                    reviewer_notes=(
                        f"PASS at padding {reframe_padding}, seed {reframe_seed}, {steps} steps, guidance {guidance}."
                    ),
                )
            )
            records.append(
                _reframe_outpaint_record(
                    model=model,
                    family=family,
                    package_variant=package_variant,
                    step="OP",
                    step_label="canvas-guided outpaint",
                    artifact_file=outpaint_output,
                    prompt=outpaint_prompt,
                    reviewer_notes=(
                        (
                            f"STALE historical artifact at padding {outpaint_padding}, seed {outpaint_seed}, {steps} steps, guidance {guidance}. "
                            "This distilled FLUX.2 canvas expansion visibly preserved a source box and is no longer exposed as current strict outpaint support."
                        )
                        if is_distilled_flux
                        else (
                            f"PASS at padding {outpaint_padding}, seed {outpaint_seed}, {steps} steps, guidance {guidance}. "
                            "This is a generative canvas expansion, not a native masked fill/inpaint run."
                        )
                    ),
                    status=STATUS_STALE if is_distilled_flux else STATUS_PASS,
                )
            )
    return records


def _flux2_klein_base_starship_records() -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    records.extend(
        _flux2_klein_base_model_records(
            family="FLUX.2 Klein Base 9B",
            model="black-forest-labs/FLUX.2-klein-base-9B",
            slug="base9b_source",
            latent_prompt=BASE_STARSHIP_LATENT_PROMPT,
            multiref_prompt=BASE_STARSHIP_MULTIREF_9B_PROMPT,
            multiref_source_images=(
                f"{FLUX2_KLEIN_BASE_STARSHIP_DIR}/base9b_source_d_sketch.png",
                f"{FLUX2_KLEIN_BASE_STARSHIP_DIR}/base9b_source_b_latent_dusk.png",
            ),
            multiref_status=STATUS_PASS,
            multiref_notes=(
                "PASS at 20 steps, guidance 1.5, seed 8614. Combines the sketch structure and aurora "
                "lighting reference into one coherent close-up."
            ),
            outpaint_notes=(
                "PASS at padding 5%,80%,5%,60%, 20 steps, guidance 4, seed 8612. The source crop stays "
                "visually stable without a pasted source rectangle."
            ),
        )
    )
    records.extend(
        _flux2_klein_base_model_records(
            family="FLUX.2 Klein Base 4B",
            model="black-forest-labs/FLUX.2-klein-base-4B",
            slug="base4b_source",
            latent_prompt=BASE_STARSHIP_LATENT_PROMPT,
            multiref_prompt=BASE_STARSHIP_MULTIREF_4B_PROMPT,
            multiref_source_images=(
                f"{FLUX2_KLEIN_BASE_STARSHIP_DIR}/base4b_source_d_sketch.png",
                FLUX2_KLEIN_BASE_STARSHIP_SOURCE,
            ),
            multiref_status=STATUS_PARTIAL,
            multiref_notes=(
                "PARTIAL at 20 steps, guidance 2.0, seed 8614. The composition is coherent, but it drifts "
                "toward the nose/front-engine crop more than requested."
            ),
            outpaint_notes=(
                "PASS at padding 5%,80%,5%,60%, 20 steps, guidance 4, seed 8612. Extends to a full-ship "
                "wide frame without a visible pasted source rectangle."
            ),
        )
    )
    return records


def _reframe_outpaint_record(
    *,
    model: str,
    family: str,
    package_variant: str,
    step: str,
    step_label: str,
    artifact_file: str,
    prompt: str,
    reviewer_notes: str,
    status: str = STATUS_PASS,
) -> ValidationRecord:
    return ValidationRecord(
        profile_id=REFRAME_OUTPAINT_PROFILE_ID,
        model=model,
        family=family,
        package_variant=package_variant,
        step=step,
        step_label=step_label,
        public_task="image-to-image",
        mode="edit-reference",
        status=status,
        artifact_path=f"{REFRAME_OUTPAINT_DIR}/{artifact_file}",
        source_images=(REFRAME_OUTPAINT_SOURCE,),
        prompt=prompt,
        reviewer_notes=reviewer_notes,
        evidence_date="2026-06-08",
    )


def _flux2_klein_base_model_records(
    *,
    family: str,
    model: str,
    slug: str,
    latent_prompt: str,
    multiref_prompt: str,
    multiref_source_images: tuple[str, ...],
    multiref_status: str,
    multiref_notes: str,
    outpaint_notes: str,
) -> list[ValidationRecord]:
    return [
        _flux2_klein_base_record(
            model=model,
            family=family,
            step="B",
            step_label="latent aurora restyle",
            mode="latent-img2img",
            status=STATUS_PASS,
            artifact_file=f"{slug}_b_latent_dusk.png",
            source_images=(FLUX2_KLEIN_BASE_STARSHIP_SOURCE,),
            prompt=latent_prompt,
            reviewer_notes="PASS at 20 steps, guidance 3.0, seed 8611. Preserves the crop while producing a clearly darker aurora/night restyle.",
        ),
        _flux2_klein_base_record(
            model=model,
            family=family,
            step="C",
            step_label="damage edit",
            mode="edit-reference",
            status=STATUS_PASS,
            artifact_file=f"{slug}_c_damage.png",
            source_images=(FLUX2_KLEIN_BASE_STARSHIP_SOURCE,),
            prompt=BASE_STARSHIP_DAMAGE_PROMPT,
            reviewer_notes="PASS at 20 steps, guidance 1.5, seed 8612. Adds coherent damage while holding the single-ship scene together.",
        ),
        _flux2_klein_base_record(
            model=model,
            family=family,
            step="D",
            step_label="graphite sketch edit",
            mode="edit-reference",
            status=STATUS_PASS,
            artifact_file=f"{slug}_d_sketch.png",
            source_images=(FLUX2_KLEIN_BASE_STARSHIP_SOURCE,),
            prompt=BASE_STARSHIP_SKETCH_PROMPT,
            reviewer_notes="PASS at 20 steps, guidance 1.5, seed 8613. Produces stable line art close to the source geometry.",
        ),
        _flux2_klein_base_record(
            model=model,
            family=family,
            step="E",
            step_label="multi-reference composition",
            mode="multi-reference",
            status=multiref_status,
            artifact_file=f"{slug}_e_multiref.png",
            source_images=multiref_source_images,
            prompt=multiref_prompt,
            reviewer_notes=multiref_notes,
        ),
        _flux2_klein_base_record(
            model=model,
            family=family,
            step="F",
            step_label="strict outpaint",
            mode="edit-reference",
            status=STATUS_PASS,
            artifact_file=f"{slug}_f_outpaint.png",
            source_images=(FLUX2_KLEIN_BASE_STARSHIP_SOURCE,),
            prompt=QWEN_OUTPAINT_PROMPT,
            reviewer_notes=outpaint_notes,
        ),
    ]


def _flux2_klein_base_record(
    *,
    model: str,
    family: str,
    step: str,
    step_label: str,
    mode: str,
    status: str,
    artifact_file: str,
    source_images: tuple[str, ...],
    prompt: str,
    reviewer_notes: str,
) -> ValidationRecord:
    return ValidationRecord(
        profile_id=FLUX2_KLEIN_BASE_STARSHIP_PROFILE_ID,
        model=model,
        family=family,
        package_variant="source",
        step=step,
        step_label=step_label,
        public_task="image-to-image",
        mode=mode,
        status=status,
        artifact_path=f"{FLUX2_KLEIN_BASE_STARSHIP_DIR}/{artifact_file}",
        source_images=source_images,
        prompt=prompt,
        reviewer_notes=reviewer_notes,
        evidence_date="2026-06-10",
    )


def _source_images(step: str, slug: str | None = None) -> tuple[str, ...]:
    if step != "E":
        return (CANONICAL_SOURCE,)
    if slug is None:
        raise ValueError("Multi-reference validation records require a source slug.")
    return (
        _reference_input_path(f"{slug}_d_pencil_crash.png"),
        _reference_input_path(f"{slug}_b_cinematic.png"),
    )


def _reference_input_path(file_name: str) -> str:
    return f"{_MATRIX_DIR}/reference-inputs/{file_name}"


def _record(
    *,
    model: str,
    family: str,
    package_variant: str,
    step: str,
    mode: str,
    status: str,
    artifact_path: str | None,
    source_images: tuple[str, ...],
    reviewer_notes: str,
    prompt: str | None = None,
    step_label: str | None = None,
    evidence_date: str = "2026-06-05",
) -> ValidationRecord:
    return ValidationRecord(
        profile_id=I2I_EDIT_5X4_PROFILE_ID,
        model=model,
        family=family,
        package_variant=package_variant,
        step=step,
        step_label=step_label or STEP_LABELS[step],
        public_task="image-to-image",
        mode=mode,
        status=status,
        artifact_path=artifact_path,
        source_images=source_images,
        prompt=prompt or PROMPTS[step],
        reviewer_notes=reviewer_notes,
        evidence_date=evidence_date,
    )


def _fibo_records() -> list[ValidationRecord]:
    artifact_path = _matrix_path("fibo-edit-variant-matrix.jpg")
    return [
        _record(
            model="briaai/Fibo-Edit",
            family="FIBO Edit",
            package_variant="source",
            step="D",
            mode="edit-reference",
            status=STATUS_FAIL,
            artifact_path=artifact_path,
            source_images=(CANONICAL_SOURCE,),
            reviewer_notes="Source route preserves some spaceship structure but is overexposed and does not satisfy the crash/sketch edit.",
        ),
        _record(
            model="briaai/Fibo-Edit",
            family="FIBO Edit",
            package_variant="source",
            step="C",
            mode="edit-reference",
            status=STATUS_FAIL,
            artifact_path=artifact_path,
            source_images=(CANONICAL_SOURCE,),
            reviewer_notes="Hard-landing edit collapses or loses the ship.",
        ),
        _record(
            model="models/fibo-edit-bf16",
            family="FIBO Edit",
            package_variant="BF16 prepared",
            step="D",
            mode="edit-reference",
            status=STATUS_FAIL,
            artifact_path=artifact_path,
            source_images=(CANONICAL_SOURCE,),
            reviewer_notes="Current local BF16 prepared folder has the required final bias, but full validation still fails before release-quality output.",
        ),
        _record(
            model="models/fibo-edit-bf16",
            family="FIBO Edit",
            package_variant="BF16 prepared",
            step="C",
            mode="edit-reference",
            status=STATUS_FAIL,
            artifact_path=artifact_path,
            source_images=(CANONICAL_SOURCE,),
            reviewer_notes="Current local BF16 prepared folder has the required final bias, but hard-landing validation still fails.",
        ),
        _record(
            model="models/fibo-edit-8bit",
            family="FIBO Edit",
            package_variant="q8 prepared",
            step="D",
            mode="edit-reference",
            status=STATUS_FAIL,
            artifact_path=artifact_path,
            source_images=(CANONICAL_SOURCE,),
            reviewer_notes="Current local q8 prepared folder keeps sensitive paths unquantized, but full validation still fails before release-quality output.",
        ),
        _record(
            model="models/fibo-edit-8bit",
            family="FIBO Edit",
            package_variant="q8 prepared",
            step="C",
            mode="edit-reference",
            status=STATUS_FAIL,
            artifact_path=artifact_path,
            source_images=(CANONICAL_SOURCE,),
            reviewer_notes="Current local q8 prepared folder keeps sensitive paths unquantized, but hard-landing validation still fails.",
        ),
    ]


def _flux2_records(
    *,
    family: str,
    source_model: str,
    source_slug: str,
    q8_model: str,
    q8_slug: str,
    q4_model: str,
    q4_slug: str,
) -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    artifact_path = _matrix_path(f"{source_slug.rsplit('_', 1)[0].replace('_', '-')}-variant-matrix.jpg")
    specs = (
        ("source", source_model, source_slug),
        ("q8 prepared", q8_model, q8_slug),
        ("q4 prepared", q4_model, q4_slug),
    )
    for package_variant, model, _slug in specs:
        for step, suffix, mode, notes in (
            ("B", "cinematic", "latent-img2img", "Preserves spaceship and scene while adding cinematic polish."),
            (
                "D",
                "pencil_crash",
                "edit-reference",
                "Clean pencil hard-landing sketch with recognizable ship and crash/smoke cues.",
            ),
            (
                "C",
                "crash",
                "edit-reference",
                "Solid hard-landing edit with smoke/snow disruption and preserved spaceship identity.",
            ),
            (
                "E",
                "composition",
                "multi-reference",
                "Uses pencil/crash and cinematic references coherently; preserves hard-landing scene.",
            ),
        ):
            records.append(
                _record(
                    model=model,
                    family=family,
                    package_variant=package_variant,
                    step=step,
                    mode=mode,
                    status=STATUS_PASS,
                    artifact_path=artifact_path,
                    source_images=_source_images(step, _slug),
                    reviewer_notes=notes,
                )
            )
    return records


def _qwen2509_records() -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    artifact_path = _matrix_path("qwen-image-edit-2509-variant-matrix.jpg")
    specs = (
        ("source", "Qwen/Qwen-Image-Edit-2509", "qwen_edit_2509_source"),
        ("q8 prepared", "AbstractFramework/qwen-image-edit-2509-8bit", "qwen_edit_2509_8bit"),
        ("q4 prepared", "AbstractFramework/qwen-image-edit-2509-4bit", "qwen_edit_2509_4bit"),
    )
    for package_variant, model, _slug in specs:
        for step, suffix, status, notes in (
            ("B", "cinematic", STATUS_PASS, "Preserves spaceship and scene while adding cinematic polish."),
            (
                "D",
                "pencil_crash",
                STATUS_PASS,
                "Clean pencil hard-landing sketch with recognizable ship and crash/smoke cues.",
            ),
            (
                "C",
                "crash",
                STATUS_PASS,
                "Solid hard-landing edit with smoke/snow disruption and preserved spaceship identity.",
            ),
            (
                "E",
                "composition",
                STATUS_PARTIAL if package_variant == "q4 prepared" else STATUS_PASS,
                "Preserves sketch/crash structure but weakly applies the color reference."
                if package_variant == "q4 prepared"
                else "Uses pencil/crash and cinematic references coherently; preserves hard-landing scene.",
            ),
        ):
            records.append(
                _record(
                    model=model,
                    family="Qwen Image Edit 2509",
                    package_variant=package_variant,
                    step=step,
                    mode="multi-reference" if step == "E" else "edit-reference",
                    status=status,
                    artifact_path=artifact_path,
                    source_images=_source_images(step, _slug),
                    reviewer_notes=notes,
                )
            )
    return records


def _qwen2511_records() -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    artifact_path = f"{QWEN2511_PARITY_DIR}/qwen-image-edit-2511-source-q8-q4-parity.jpg"
    specs = (
        ("source", "Qwen/Qwen-Image-Edit-2511", "source"),
        ("q8 prepared", "AbstractFramework/qwen-image-edit-2511-8bit", "q8"),
        ("q4 prepared", "AbstractFramework/qwen-image-edit-2511-4bit", "q4"),
    )
    qwen2511_prompts = {
        "B": (
            "Convert the source image into a clean graphite pencil sketch on white paper. Preserve the same wide "
            "camera framing, the same spaceship shape, the icy canyon background, and the rear engines. Use thin "
            "gray pencil outlines with light hand shading only. The final image must clearly look like a hand "
            "drawn pencil sketch, not a blurred photo."
        ),
        "C": (
            "Edit the source into the same spaceship after a hard landing in the snow at dusk. Preserve the same "
            "wide camera angle, spaceship identity, rear engines, canyon cliffs, and framing. The ship must remain "
            "solid and sharp, but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a "
            "shallow scrape trail, and a thin smoke plume. Use blue-hour dusk lighting. No blur, no mesh, no "
            "dissolve."
        ),
        "E": (
            "Use the first image as the graphite pencil sketch style reference and the second image as the "
            "hard-landing crash content reference. Produce one coherent wide image of the same spaceship crashed "
            "in the snowy canyon: graphite pencil outlines on white paper, visible tilted hull, disturbed snow, "
            "broken ice chunks, scrape trail, and a thin smoke plume. Preserve the spaceship identity and canyon "
            "layout. No blur, no colored photo, no text."
        ),
    }
    qwen2511_q4_prompts = {
        "C": (
            "Wide establishing shot of the same spaceship after a hard landing in the snow at dusk. Preserve the "
            "original wide camera angle, full spaceship fully visible inside the frame, rear engines visible, "
            "canyon cliffs visible on both left and right sides, and snowy ground foreground. Show a tilted hull, "
            "bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, and a thin smoke "
            "plume. Use blue-hour dusk lighting. Keep the ship solid and sharp."
        ),
        "E": (
            "Use the first image as the graphite pencil sketch style reference and the second image as the "
            "hard-landing crash content reference. Produce one coherent wide image of the same spaceship crashed "
            "in the snowy canyon: graphite pencil outlines on white paper, visible tilted hull, disturbed snow, "
            "broken ice chunks, scrape trail, and a thin smoke plume. Preserve the spaceship identity, full wide "
            "framing, and canyon layout. No blur, no colored photo, no close-up, no cropped spaceship, no text."
        ),
    }
    labels = {"B": "pencil sketch", "C": "crash from source", "E": "multi-reference composition"}
    notes = {
        "B": "Clean pencil sketch that preserves the source spaceship and canyon layout.",
        "C": "Hard-landing edit with visible dusk lighting, smoke, snow disruption, and preserved spaceship identity.",
        "E": "Composition uses the pencil style and hard-landing reference coherently.",
    }
    for package_variant, model, slug in specs:
        for step in ("B", "C", "E"):
            prompt = qwen2511_prompts[step]
            reviewer_notes = notes[step]
            if package_variant == "q4 prepared" and step in qwen2511_q4_prompts:
                prompt = qwen2511_q4_prompts[step]
                reviewer_notes += " The q4 row used an explicit crop-avoidance negative prompt; see the command log."
            records.append(
                _record(
                    model=model,
                    family="Qwen Image Edit 2511",
                    package_variant=package_variant,
                    step=step,
                    mode="multi-reference" if step == "E" else "edit-reference",
                    status=STATUS_PASS,
                    artifact_path=artifact_path,
                    source_images=_qwen2511_source_images(slug, step),
                    reviewer_notes=reviewer_notes,
                    prompt=prompt,
                    step_label=labels[step],
                    evidence_date="2026-06-06",
                )
            )
    return records


def _qwen2511_source_images(slug: str, step: str) -> tuple[str, ...]:
    if step != "E":
        return (CANONICAL_SOURCE,)
    return (
        f"{QWEN2511_PARITY_DIR}/qwen2511-{slug}-pencil.png",
        f"{QWEN2511_PARITY_DIR}/qwen2511-{slug}-crash.png",
    )


def _matrix_path(file_name: str) -> str:
    return f"{_MATRIX_DIR}/{file_name}"
