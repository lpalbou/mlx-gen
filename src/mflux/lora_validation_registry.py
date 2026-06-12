from __future__ import annotations

from dataclasses import dataclass

from mflux.models.common.config import ModelConfig

LORA_STATUS_UNSUPPORTED = "unsupported"
LORA_STATUS_MAPPED_UNVALIDATED = "mapped-unvalidated"
LORA_STATUS_VALIDATED = "validated"

QWEN2511_Q8_SINGLE_EDIT_MULTI_ANGLE_PROFILE_ID = "lora_qwen2511_q8_single_edit_multi_angle_2026_06_08"
QWEN2509_Q8_SINGLE_EDIT_MULTI_ANGLE_PROFILE_ID = "lora_qwen2509_q8_single_edit_multi_angle_2026_06_11"
QWEN_EDIT_Q8_GHIBLI_PROFILE_ID = "lora_qwen_edit_q8_ghibli_edit_2026_06_11"
QWEN2512_Q8_PIXEL_ART_PROFILE_ID = "lora_qwen2512_q8_pixel_art_t2i_2026_06_11"
ZIMAGE_Q8_TECHNICALLYCOLOR_PROFILE_ID = "lora_zimage_q8_technically_color_t2i_2026_06_11"
FLUX2_KLEIN9B_Q8_CONSISTENCY_EDIT_PROFILE_ID = "lora_flux2_klein9b_q8_consistency_edit_2026_06_11"
ERNIE_TURBO_Q8_ANIME_STYLE_PROFILE_ID = "lora_ernie_turbo_q8_anime_style_t2i_2026_06_11"
WAN_TI2V5B_Q8_HSTORIC_T2V_PROFILE_ID = "lora_wan_ti2v5b_q8_hstoric_t2v_2026_06_11"
WAN_TI2V5B_Q8_CRUSHIT_I2V_PROFILE_ID = "lora_wan_ti2v5b_q8_crushit_i2v_2026_06_11"
WAN_A14B_Q8_FOLLOWCAM_T2V_PROFILE_ID = "lora_wan_a14b_q8_followcam_t2v_2026_06_11"
WAN_A14B_Q8_ORBIT_I2V_PROFILE_ID = "lora_wan_a14b_q8_orbit_i2v_2026_06_11"
WAN_A14B_Q8_LIGHTX2V_4STEP_T2V_PROFILE_ID = "lora_wan_a14b_q8_lightx2v_4step_t2v_2026_06_12"
WAN_A14B_Q8_LIGHTX2V_4STEP_I2V_PROFILE_ID = "lora_wan_a14b_q8_lightx2v_4step_i2v_2026_06_12"


@dataclass(frozen=True)
class LoRAValidationRecord:
    model: str
    capability_id: str
    validation_profile: str


_RECORDS: tuple[LoRAValidationRecord, ...] = (
    LoRAValidationRecord(
        model="AbstractFramework/qwen-image-edit-2511-8bit",
        capability_id="qwen.edit",
        validation_profile=QWEN2511_Q8_SINGLE_EDIT_MULTI_ANGLE_PROFILE_ID,
    ),
    LoRAValidationRecord(
        model="AbstractFramework/qwen-image-edit-2509-8bit",
        capability_id="qwen.edit",
        validation_profile=QWEN2509_Q8_SINGLE_EDIT_MULTI_ANGLE_PROFILE_ID,
    ),
    LoRAValidationRecord(
        model="AbstractFramework/qwen-image-edit-8bit",
        capability_id="qwen.edit",
        validation_profile=QWEN_EDIT_Q8_GHIBLI_PROFILE_ID,
    ),
    LoRAValidationRecord(
        model="AbstractFramework/qwen-image-2512-8bit",
        capability_id="qwen.text",
        validation_profile=QWEN2512_Q8_PIXEL_ART_PROFILE_ID,
    ),
    LoRAValidationRecord(
        model="AbstractFramework/z-image-turbo-8bit",
        capability_id="z-image.text",
        validation_profile=ZIMAGE_Q8_TECHNICALLYCOLOR_PROFILE_ID,
    ),
    LoRAValidationRecord(
        model="AbstractFramework/flux.2-klein-9b-8bit",
        capability_id="flux2.edit",
        validation_profile=FLUX2_KLEIN9B_Q8_CONSISTENCY_EDIT_PROFILE_ID,
    ),
    LoRAValidationRecord(
        model="AbstractFramework/ernie-image-turbo-8bit",
        capability_id="ernie-image.text",
        validation_profile=ERNIE_TURBO_Q8_ANIME_STYLE_PROFILE_ID,
    ),
    LoRAValidationRecord(
        model="AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit",
        capability_id="wan.text-video",
        validation_profile=WAN_TI2V5B_Q8_HSTORIC_T2V_PROFILE_ID,
    ),
    LoRAValidationRecord(
        model="AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit",
        capability_id="wan.first-frame",
        validation_profile=WAN_TI2V5B_Q8_CRUSHIT_I2V_PROFILE_ID,
    ),
    LoRAValidationRecord(
        model="AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
        capability_id="wan.text-video",
        validation_profile=WAN_A14B_Q8_LIGHTX2V_4STEP_T2V_PROFILE_ID,
    ),
    LoRAValidationRecord(
        model="AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit",
        capability_id="wan.first-frame",
        validation_profile=WAN_A14B_Q8_LIGHTX2V_4STEP_I2V_PROFILE_ID,
    ),
)


def get_lora_validation_status(
    *,
    model: str | None,
    model_config: ModelConfig | None,
    capability_id: str,
) -> tuple[str, str | None]:
    candidate_keys = _candidate_model_keys(model=model, model_config=model_config)
    for record in _RECORDS:
        if _normalize(record.model) in candidate_keys and record.capability_id == capability_id:
            return LORA_STATUS_VALIDATED, record.validation_profile
    return LORA_STATUS_MAPPED_UNVALIDATED, None


def _candidate_model_keys(
    *,
    model: str | None,
    model_config: ModelConfig | None,
) -> set[str]:
    keys = set()
    if model:
        keys.add(_normalize(model))
    if model_config is not None and model_config.model_name:
        keys.add(_normalize(model_config.model_name))
    return keys


def _normalize(value: str) -> str:
    return value.lower().replace("_", "-").rstrip("/")
