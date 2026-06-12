from pathlib import Path

import pytest

import mlxgen
from mflux.task_inference import (
    CANVAS_POLICY_EXACT_RESIZE,
    CANVAS_POLICY_SOURCE_ASPECT,
    MODE_EDIT_REFERENCE,
    MODE_FIRST_FRAME_I2V,
    MODE_LATENT_IMG2IMG,
    MODE_MULTI_REFERENCE,
    MODE_TEXT_ONLY,
    TaskInferenceError,
)


def test_public_resolver_uses_image_presence_for_image_models():
    assert mlxgen.infer_task(model="flux2-klein-4b") == "text-to-image"
    assert mlxgen.infer_task(model="flux2-klein-4b", image_count=1) == "image-to-image"
    assert mlxgen.infer_task(model="flux2-klein-4b", image_count=2) == "image-to-image"


def test_generation_plan_exposes_internal_mode_for_flux2():
    assert mlxgen.resolve_generation_plan(model="flux2-klein-4b").mode == MODE_TEXT_ONLY
    assert mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=1).mode == MODE_EDIT_REFERENCE
    assert (
        mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=1, has_image_strength=True).mode
        == MODE_LATENT_IMG2IMG
    )
    assert mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=2).mode == MODE_MULTI_REFERENCE


def test_generation_plan_requires_strength_for_qwen_base_latent_i2i():
    with pytest.raises(TaskInferenceError, match="image-strength is required"):
        mlxgen.resolve_generation_plan(model="qwen-image", image_count=1)

    latent = mlxgen.resolve_generation_plan(model="qwen-image", image_count=1, has_image_strength=True)
    assert latent.mode == MODE_LATENT_IMG2IMG
    assert latent.model_override is None

    with pytest.raises(TaskInferenceError, match="does not support edit-reference"):
        mlxgen.resolve_generation_plan(model="qwen-image", image_count=1, i2i_mode="edit")


def test_base_fibo_no_longer_advertises_unvalidated_latent_i2i():
    capabilities = mlxgen.get_model_capabilities(model="fibo")
    assert MODE_LATENT_IMG2IMG not in {capability.mode for capability in capabilities.capabilities}

    with pytest.raises(TaskInferenceError, match="supports text-to-image only"):
        mlxgen.resolve_generation_plan(model="fibo", image_count=1)

    with pytest.raises(TaskInferenceError, match="supports text-to-image only"):
        mlxgen.resolve_generation_plan(model="fibo", image_count=1, i2i_mode="latent")


def test_public_resolver_uses_wan_model_capability():
    assert mlxgen.infer_task(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers") == "text-to-video"
    assert mlxgen.infer_task(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers", image_count=1) == "image-to-video"
    assert mlxgen.infer_task(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers", image_count=1) == "image-to-video"
    assert (
        mlxgen.resolve_generation_plan(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers", image_count=1).mode
        == MODE_FIRST_FRAME_I2V
    )


def test_public_resolver_rejects_wan_fixed_task_contradictions():
    with pytest.raises(TaskInferenceError, match="text-to-video model does not accept input images"):
        mlxgen.infer_task(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers", image_count=1)

    with pytest.raises(TaskInferenceError, match="image-to-video model requires --image"):
        mlxgen.infer_task(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers")


def test_public_resolver_rejects_generic_wan_names_without_specific_config():
    with pytest.raises(TaskInferenceError, match="Cannot infer a supported Wan model config"):
        mlxgen.infer_task(model="models/my-wan-video-folder", image_count=1)


def test_public_resolver_rejects_explicit_task_image_contradictions():
    with pytest.raises(TaskInferenceError, match="image-to-image requires --image"):
        mlxgen.infer_task(model="Qwen/Qwen-Image", task="image-to-image")

    with pytest.raises(TaskInferenceError, match="text-to-image cannot be combined with --image"):
        mlxgen.infer_task(model="Qwen/Qwen-Image", task="text-to-image", image_count=1)


def test_task_edit_is_compatibility_alias_for_image_to_image_mode():
    plan = mlxgen.resolve_generation_plan(model="flux2-klein-4b", task="edit", image_count=1)

    assert plan.public_task == "image-to-image"
    assert plan.mode == MODE_EDIT_REFERENCE


def test_edit_only_models_require_an_image_in_auto_mode():
    with pytest.raises(TaskInferenceError, match="image-to-image requires --image"):
        mlxgen.resolve_generation_plan(model="qwen-image-edit")


def test_qwen_edit_versions_expose_distinct_reference_modes():
    regular_capabilities = mlxgen.get_model_capabilities(model="qwen-image-edit")
    edit_2509_capabilities = mlxgen.get_model_capabilities(model="qwen-image-edit-2509")
    edit_2511_capabilities = mlxgen.get_model_capabilities(model="qwen-image-edit-2511")
    regular_modes = {capability.mode for capability in regular_capabilities.capabilities}
    edit_2509_modes = {capability.mode for capability in edit_2509_capabilities.capabilities}
    edit_2511_modes = {capability.mode for capability in edit_2511_capabilities.capabilities}

    assert regular_capabilities.label == "Qwen Image Edit"
    assert edit_2509_capabilities.label == "Qwen Image Edit 2509"
    assert edit_2511_capabilities.label == "Qwen Image Edit 2511"
    assert regular_modes == {MODE_EDIT_REFERENCE}
    assert edit_2509_modes == {MODE_EDIT_REFERENCE, MODE_MULTI_REFERENCE}
    assert edit_2511_modes == {MODE_EDIT_REFERENCE, MODE_MULTI_REFERENCE}

    with pytest.raises(TaskInferenceError, match="does not support multi-reference"):
        mlxgen.resolve_generation_plan(model="qwen-image-edit", image_count=2)
    assert mlxgen.resolve_generation_plan(model="qwen-image-edit-2509", image_count=2).mode == MODE_MULTI_REFERENCE


def test_qwen_prepared_edit_versions_inherit_distinct_reference_modes():
    regular_modes = {
        capability.mode
        for capability in mlxgen.get_model_capabilities(model="AbstractFramework/qwen-image-edit-8bit").capabilities
    }
    edit_2509_modes = {
        capability.mode
        for capability in mlxgen.get_model_capabilities(
            model="AbstractFramework/qwen-image-edit-2509-8bit"
        ).capabilities
    }

    assert regular_modes == {MODE_EDIT_REFERENCE}
    assert edit_2509_modes == {MODE_EDIT_REFERENCE, MODE_MULTI_REFERENCE}


def test_image_strength_is_rejected_for_edit_reference_mode():
    with pytest.raises(TaskInferenceError, match="image-strength is only supported for latent"):
        mlxgen.resolve_generation_plan(
            model="flux2-klein-4b",
            image_count=1,
            i2i_mode="edit",
            has_image_strength=True,
        )

    with pytest.raises(TaskInferenceError, match="image-strength is only supported for latent"):
        mlxgen.resolve_generation_plan(
            model="qwen-image-edit",
            image_count=1,
            has_image_strength=True,
        )


def test_mask_and_outpaint_options_are_checked_against_capabilities():
    with pytest.raises(TaskInferenceError, match="does not expose unified generation capabilities"):
        mlxgen.resolve_generation_plan(model="fibo-edit", image_count=1, has_mask=True)

    with pytest.raises(TaskInferenceError, match="mask-path is only supported"):
        mlxgen.resolve_generation_plan(model="fibo", image_count=1, has_mask=True)

    with pytest.raises(TaskInferenceError, match="mask-path is only supported"):
        mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=1, has_mask=True)

    flux2_outpaint = mlxgen.resolve_generation_plan(model="flux2-klein-base-4b", image_count=1, has_outpaint=True)
    qwen_outpaint = mlxgen.resolve_generation_plan(
        model="qwen-image-edit-2511",
        image_count=1,
        has_outpaint=True,
    )
    qwen_base_outpaint = mlxgen.resolve_generation_plan(
        model="qwen-image-edit",
        image_count=1,
        has_outpaint=True,
    )
    qwen_2509_outpaint = mlxgen.resolve_generation_plan(
        model="qwen-image-edit-2509",
        image_count=1,
        has_outpaint=True,
    )
    assert flux2_outpaint.capability_id == "flux2.outpaint"
    assert qwen_outpaint.capability_id == "qwen.outpaint"
    assert qwen_base_outpaint.capability_id == "qwen.outpaint"
    assert qwen_2509_outpaint.capability_id == "qwen.outpaint"

    with pytest.raises(TaskInferenceError, match="outpaint-padding is only supported"):
        mlxgen.resolve_generation_plan(model="z-image-turbo", image_count=1, has_outpaint=True)

    with pytest.raises(TaskInferenceError, match="outpaint-padding is only supported"):
        mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=1, has_outpaint=True)


def test_reframe_option_is_limited_to_validated_edit_capabilities():
    flux2 = mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=1, has_reframe=True)
    qwen = mlxgen.resolve_generation_plan(model="qwen-image-edit-2511", image_count=1, has_reframe=True)
    qwen_base = mlxgen.resolve_generation_plan(model="qwen-image-edit", image_count=1, has_reframe=True)
    qwen_2509 = mlxgen.resolve_generation_plan(model="qwen-image-edit-2509", image_count=1, has_reframe=True)

    assert flux2.mode == MODE_EDIT_REFERENCE
    assert flux2.capability_id == "flux2.reframe"
    assert qwen.mode == MODE_EDIT_REFERENCE
    assert qwen.capability_id == "qwen.reframe"
    assert qwen_base.mode == MODE_EDIT_REFERENCE
    assert qwen_base.capability_id == "qwen.reframe"
    assert qwen_2509.mode == MODE_EDIT_REFERENCE
    assert qwen_2509.capability_id == "qwen.reframe"

    with pytest.raises(TaskInferenceError, match="reframe-padding is only supported"):
        mlxgen.resolve_generation_plan(model="qwen-image", image_count=1, has_reframe=True)

    with pytest.raises(TaskInferenceError, match="reframe-padding is only supported"):
        mlxgen.resolve_generation_plan(model="z-image-turbo", image_count=1, has_reframe=True)

    with pytest.raises(TaskInferenceError, match="reframe-padding is only supported"):
        mlxgen.resolve_generation_plan(model="flux2-klein-base-4b", image_count=1, has_reframe=True)

    with pytest.raises(TaskInferenceError, match="reframe-padding is only supported"):
        mlxgen.resolve_generation_plan(model="ernie-image-turbo", image_count=1, has_reframe=True)

    with pytest.raises(TaskInferenceError, match="does not expose unified generation capabilities"):
        mlxgen.resolve_generation_plan(model="fibo-edit", image_count=1, has_reframe=True)


def test_model_capabilities_are_publicly_inspectable():
    capabilities = mlxgen.get_model_capabilities(model="flux2-klein-4b")

    assert capabilities.schema_version == 2
    assert capabilities.family == "flux2"
    assert {capability.mode for capability in capabilities.capabilities} >= {
        MODE_TEXT_ONLY,
        MODE_LATENT_IMG2IMG,
        MODE_EDIT_REFERENCE,
        MODE_MULTI_REFERENCE,
    }
    latent = next(capability for capability in capabilities.capabilities if capability.mode == MODE_LATENT_IMG2IMG)
    assert latent.default_canvas_policy == CANVAS_POLICY_SOURCE_ASPECT
    assert latent.canvas_policies == (CANVAS_POLICY_SOURCE_ASPECT, CANVAS_POLICY_EXACT_RESIZE)
    assert latent.primary_image_index == 0
    assert latent.dimension_multiple == 16
    edit = next(capability for capability in capabilities.capabilities if capability.id == "flux2.edit")
    reframe = next(capability for capability in capabilities.capabilities if capability.id == "flux2.reframe")
    assert edit.supports_outpaint is False
    assert edit.supports_reframe is False
    assert edit.supports_lora is True
    assert edit.lora_status == "mapped-unvalidated"
    assert edit.lora_target_roles == ("transformer",)
    assert reframe.supports_reframe is True
    assert reframe.lora_status == "mapped-unvalidated"

    base_capabilities = mlxgen.get_model_capabilities(model="flux2-klein-base-4b")
    base_edit = next(capability for capability in base_capabilities.capabilities if capability.id == "flux2.edit")
    base_outpaint = next(
        capability for capability in base_capabilities.capabilities if capability.id == "flux2.outpaint"
    )
    assert base_edit.supports_reframe is False
    assert base_edit.supports_outpaint is False
    assert base_outpaint.supports_reframe is False
    assert base_outpaint.supports_outpaint is True


def test_qwen_2511_q8_single_edit_lora_status_is_exact():
    capabilities = mlxgen.get_model_capabilities(model="AbstractFramework/qwen-image-edit-2511-8bit")

    edit = next(capability for capability in capabilities.capabilities if capability.id == "qwen.edit")
    reframe = next(capability for capability in capabilities.capabilities if capability.id == "qwen.reframe")
    outpaint = next(capability for capability in capabilities.capabilities if capability.id == "qwen.outpaint")
    multi = next(capability for capability in capabilities.capabilities if capability.id == "qwen.multi-reference")

    assert edit.lora_status == "validated"
    assert edit.lora_validation_profile == "lora_qwen2511_q8_single_edit_multi_angle_2026_06_08"
    assert reframe.lora_status == "mapped-unvalidated"
    assert outpaint.lora_status == "mapped-unvalidated"
    assert multi.lora_status == "mapped-unvalidated"


def test_qwen_2509_q8_single_edit_lora_status_is_exact():
    capabilities = mlxgen.get_model_capabilities(model="AbstractFramework/qwen-image-edit-2509-8bit")

    edit = next(capability for capability in capabilities.capabilities if capability.id == "qwen.edit")
    reframe = next(capability for capability in capabilities.capabilities if capability.id == "qwen.reframe")
    multi = next(capability for capability in capabilities.capabilities if capability.id == "qwen.multi-reference")

    assert edit.lora_status == "validated"
    assert edit.lora_validation_profile == "lora_qwen2509_q8_single_edit_multi_angle_2026_06_11"
    assert reframe.lora_status == "mapped-unvalidated"
    assert multi.lora_status == "mapped-unvalidated"


def test_qwen_2512_q8_text_lora_status_is_exact():
    capabilities = mlxgen.get_model_capabilities(model="AbstractFramework/qwen-image-2512-8bit")

    text = next(capability for capability in capabilities.capabilities if capability.id == "qwen.text")
    latent = next(capability for capability in capabilities.capabilities if capability.id == "qwen.latent")

    assert text.lora_status == "validated"
    assert text.lora_validation_profile == "lora_qwen2512_q8_pixel_art_t2i_2026_06_11"
    assert latent.lora_status == "mapped-unvalidated"
    assert latent.lora_validation_profile is None


def test_original_qwen_edit_q8_single_edit_lora_status_is_exact():
    original_edit = mlxgen.get_model_capabilities(model="AbstractFramework/qwen-image-edit-8bit")
    original_edit_row = next(capability for capability in original_edit.capabilities if capability.id == "qwen.edit")
    assert original_edit_row.lora_status == "validated"
    assert original_edit_row.lora_validation_profile == "lora_qwen_edit_q8_ghibli_edit_2026_06_11"


def test_unvalidated_base_qwen_rows_remain_mapped_unvalidated():
    base_qwen = mlxgen.get_model_capabilities(model="AbstractFramework/qwen-image-8bit")
    base_text = next(capability for capability in base_qwen.capabilities if capability.id == "qwen.text")
    base_latent = next(capability for capability in base_qwen.capabilities if capability.id == "qwen.latent")
    assert base_text.lora_status == "mapped-unvalidated"
    assert base_text.lora_validation_profile is None
    assert base_latent.lora_status == "mapped-unvalidated"
    assert base_latent.lora_validation_profile is None


def test_zimage_q8_text_lora_status_is_exact():
    capabilities = mlxgen.get_model_capabilities(model="AbstractFramework/z-image-turbo-8bit")

    text = next(capability for capability in capabilities.capabilities if capability.id == "z-image.text")
    latent = next(capability for capability in capabilities.capabilities if capability.id == "z-image.latent")

    assert text.lora_status == "validated"
    assert text.lora_validation_profile == "lora_zimage_q8_technically_color_t2i_2026_06_11"
    assert latent.lora_status == "mapped-unvalidated"


def test_flux2_klein9b_q8_edit_lora_status_is_exact():
    capabilities = mlxgen.get_model_capabilities(model="AbstractFramework/flux.2-klein-9b-8bit")

    edit = next(capability for capability in capabilities.capabilities if capability.id == "flux2.edit")
    multi = next(capability for capability in capabilities.capabilities if capability.id == "flux2.multi-reference")

    assert edit.lora_status == "validated"
    assert edit.lora_validation_profile == "lora_flux2_klein9b_q8_consistency_edit_2026_06_11"
    assert multi.lora_status == "mapped-unvalidated"


def test_ernie_turbo_q8_text_lora_status_is_exact():
    capabilities = mlxgen.get_model_capabilities(model="AbstractFramework/ernie-image-turbo-8bit")

    text = next(capability for capability in capabilities.capabilities if capability.id == "ernie-image.text")
    latent = next(capability for capability in capabilities.capabilities if capability.id == "ernie-image.latent")

    assert text.supports_lora is True
    assert text.lora_status == "validated"
    assert text.lora_validation_profile == "lora_ernie_turbo_q8_anime_style_t2i_2026_06_11"
    assert latent.supports_lora is True
    assert latent.lora_status == "mapped-unvalidated"
    assert latent.lora_validation_profile is None


def test_lora_requests_are_checked_against_route_capabilities():
    flux2 = mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=1, has_lora=True)
    qwen = mlxgen.resolve_generation_plan(model="qwen-image-edit-2511", image_count=1, has_lora=True)
    z_image = mlxgen.resolve_generation_plan(model="z-image-turbo", has_lora=True)
    ernie = mlxgen.resolve_generation_plan(model="ernie-image-turbo", has_lora=True)
    wan_ti2v = mlxgen.resolve_generation_plan(model="wan2.2-ti2v-5b", has_lora=True)
    wan_a14b = mlxgen.resolve_generation_plan(model="wan2.2-t2v-a14b", has_lora=True)

    assert flux2.capability_id == "flux2.edit"
    assert flux2.supports_lora is True
    assert qwen.capability_id == "qwen.edit"
    assert qwen.supports_lora is True
    assert z_image.capability_id == "z-image.text"
    assert z_image.supports_lora is True
    assert ernie.capability_id == "ernie-image.text"
    assert ernie.supports_lora is True
    assert wan_ti2v.capability_id == "wan.text-video"
    assert wan_ti2v.supports_lora is True
    assert wan_ti2v.lora_target_roles == ("transformer",)
    assert wan_a14b.capability_id == "wan.text-video"
    assert wan_a14b.supports_lora is True
    assert wan_a14b.lora_target_roles == ("high_noise_transformer", "low_noise_transformer")

    with pytest.raises(TaskInferenceError, match="LoRA mapping"):
        mlxgen.resolve_generation_plan(model="bonsai-image-ternary", has_lora=True)


def test_wan_i2v_capability_surfaces_lora_support_and_roles():
    capabilities = mlxgen.get_model_capabilities(model="wan2.2-i2v-a14b")

    first_frame = next(capability for capability in capabilities.capabilities if capability.id == "wan.first-frame")

    assert first_frame.supports_lora is True
    assert first_frame.lora_status == "mapped-unvalidated"
    assert first_frame.lora_target_roles == ("high_noise_transformer", "low_noise_transformer")


def test_wan_ti2v_q8_text_video_lora_status_is_exact():
    capabilities = mlxgen.get_model_capabilities(model="AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit")

    text_video = next(capability for capability in capabilities.capabilities if capability.id == "wan.text-video")
    first_frame = next(capability for capability in capabilities.capabilities if capability.id == "wan.first-frame")

    assert text_video.supports_lora is True
    assert text_video.lora_status == "validated"
    assert text_video.lora_validation_profile == "lora_wan_ti2v5b_q8_hstoric_t2v_2026_06_11"
    assert text_video.lora_target_roles == ("transformer",)
    assert first_frame.supports_lora is True
    assert first_frame.lora_status == "validated"
    assert first_frame.lora_validation_profile == "lora_wan_ti2v5b_q8_crushit_i2v_2026_06_11"
    assert first_frame.lora_target_roles == ("transformer",)


def test_wan_a14b_q8_rows_are_exactly_validated():
    t2v_capabilities = mlxgen.get_model_capabilities(model="AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit")
    i2v_capabilities = mlxgen.get_model_capabilities(model="AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit")

    text_video = next(capability for capability in t2v_capabilities.capabilities if capability.id == "wan.text-video")
    first_frame = next(capability for capability in i2v_capabilities.capabilities if capability.id == "wan.first-frame")

    assert text_video.supports_lora is True
    assert text_video.lora_status == "validated"
    assert text_video.lora_validation_profile == "lora_wan_a14b_q8_lightx2v_4step_t2v_2026_06_12"
    assert text_video.lora_target_roles == ("high_noise_transformer", "low_noise_transformer")
    assert first_frame.supports_lora is True
    assert first_frame.lora_status == "validated"
    assert first_frame.lora_validation_profile == "lora_wan_a14b_q8_lightx2v_4step_i2v_2026_06_12"
    assert first_frame.lora_target_roles == ("high_noise_transformer", "low_noise_transformer")


def test_flux2_dev_handle_is_not_inferred_as_flux1_dev():
    with pytest.raises(TaskInferenceError, match="FLUX.2-dev is not supported"):
        mlxgen.resolve_generation_plan(model="black-forest-labs/FLUX.2-dev", has_lora=True)


def test_fibo_edit_exposes_no_public_generation_capabilities():
    capabilities = mlxgen.get_model_capabilities(model="fibo-edit")
    assert capabilities.capabilities == ()

    with pytest.raises(TaskInferenceError, match="does not expose unified generation capabilities"):
        mlxgen.resolve_generation_plan(model="fibo-edit", image_count=1)

    with pytest.raises(TaskInferenceError, match="does not expose unified generation capabilities"):
        mlxgen.resolve_generation_plan(model="fibo-edit", image_count=1, has_image_strength=True)

    assert mlxgen.get_model_capabilities(model="fibo-edit-rmbg").capabilities == ()


def test_model_validation_is_separate_from_route_capabilities():
    capabilities = mlxgen.get_model_capabilities(model="briaai/Fibo-Edit")
    assert capabilities.capabilities == ()

    validation = mlxgen.get_model_validation("briaai/Fibo-Edit")
    assert validation.status == "FAIL"
    assert {record.mode for record in validation.records} == {MODE_EDIT_REFERENCE}

    alias_validation = mlxgen.get_model_validation("fibo-edit")
    assert alias_validation.status == "FAIL"
    assert {record.model for record in alias_validation.records} == {"briaai/Fibo-Edit"}


def test_validation_registry_reports_variant_specific_statuses():
    qwen2509_q8 = mlxgen.get_model_validation("AbstractFramework/qwen-image-edit-2509-8bit")
    assert qwen2509_q8.status == "PASS"
    assert {record.step for record in qwen2509_q8.records} == {"B", "C", "D", "E"}
    assert {record.model for record in qwen2509_q8.records} == {"AbstractFramework/qwen-image-edit-2509-8bit"}
    for record in qwen2509_q8.records:
        assert record.artifact_path is not None
        assert Path(record.artifact_path).exists()
        if record.mode == MODE_MULTI_REFERENCE:
            assert len(record.source_images) >= 2
        for source_image in record.source_images:
            assert Path(source_image).exists()

    qwen2509_q4 = mlxgen.get_model_validation("AbstractFramework/qwen-image-edit-2509-4bit")
    assert qwen2509_q4.status == "PARTIAL"
    assert {record.model for record in qwen2509_q4.records} == {"AbstractFramework/qwen-image-edit-2509-4bit"}
    assert next(record for record in qwen2509_q4.records if record.step == "E").status == "PARTIAL"

    qwen2511_q8 = mlxgen.get_model_validation("AbstractFramework/qwen-image-edit-2511-8bit")
    assert qwen2511_q8.status == "PASS"
    assert {record.step for record in qwen2511_q8.records} == {"B", "C", "E"}
    assert {record.model for record in qwen2511_q8.records} == {"AbstractFramework/qwen-image-edit-2511-8bit"}
    assert next(record for record in qwen2511_q8.records if record.step == "E").status == "PASS"

    qwen2511_q4 = mlxgen.get_model_validation("AbstractFramework/qwen-image-edit-2511-4bit")
    assert qwen2511_q4.status == "PASS"
    assert {record.step for record in qwen2511_q4.records} == {"B", "C", "E"}
    assert {record.model for record in qwen2511_q4.records} == {"AbstractFramework/qwen-image-edit-2511-4bit"}
    assert next(record for record in qwen2511_q4.records if record.step == "E").status == "PASS"


def test_lora_validation_profiles_are_resolvable():
    cases = [
        (
            "AbstractFramework/qwen-image-edit-8bit",
            "lora_qwen_edit_q8_ghibli_edit_2026_06_11",
            MODE_EDIT_REFERENCE,
        ),
        (
            "AbstractFramework/qwen-image-edit-2511-8bit",
            "lora_qwen2511_q8_single_edit_multi_angle_2026_06_08",
            MODE_EDIT_REFERENCE,
        ),
        (
            "AbstractFramework/qwen-image-edit-2509-8bit",
            "lora_qwen2509_q8_single_edit_multi_angle_2026_06_11",
            MODE_EDIT_REFERENCE,
        ),
        (
            "AbstractFramework/qwen-image-2512-8bit",
            "lora_qwen2512_q8_pixel_art_t2i_2026_06_11",
            MODE_TEXT_ONLY,
        ),
        (
            "AbstractFramework/ernie-image-turbo-8bit",
            "lora_ernie_turbo_q8_anime_style_t2i_2026_06_11",
            MODE_TEXT_ONLY,
        ),
        (
            "AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit",
            "lora_wan_a14b_q8_lightx2v_4step_i2v_2026_06_12",
            MODE_FIRST_FRAME_I2V,
        ),
        (
            "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
            "lora_wan_a14b_q8_lightx2v_4step_t2v_2026_06_12",
            MODE_TEXT_ONLY,
        ),
    ]

    for model, profile_id, expected_mode in cases:
        validation = mlxgen.get_model_validation(model, profile_id=profile_id)
        assert validation.status == "PASS"
        assert len(validation.records) == 1
        assert validation.records[0].mode == expected_mode
        assert validation.records[0].artifact_path is not None
        assert Path(validation.records[0].artifact_path).exists()
        for source_image in validation.records[0].source_images:
            assert Path(source_image).exists()


def test_reframe_outpaint_validation_profile_reports_supported_variants():
    profile = mlxgen.get_validation_profile(mlxgen.REFRAME_OUTPAINT_PROFILE_ID)
    assert (
        profile.canonical_source == "docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png"
    )
    assert len(profile.records) == 30
    for record in profile.records:
        assert Path(record.artifact_path).exists()
        assert record.source_images == (profile.canonical_source,)
        assert Path(record.source_images[0]).exists()
        model_name = record.model.lower()
        if record.step == "OP" and "flux.2-klein-" in model_name and "base" not in model_name:
            with pytest.raises(TaskInferenceError, match="outpaint-padding is only supported"):
                mlxgen.resolve_generation_plan(
                    model=record.model,
                    image_count=1,
                    has_reframe=False,
                    has_outpaint=True,
                )
            continue
        plan = mlxgen.resolve_generation_plan(
            model=record.model,
            image_count=1,
            has_reframe=record.step == "RF",
            has_outpaint=record.step == "OP",
        )
        assert plan.mode == MODE_EDIT_REFERENCE

    qwen2509_q4 = mlxgen.get_model_validation(
        "AbstractFramework/qwen-image-edit-2509-4bit",
        profile_id=mlxgen.REFRAME_OUTPAINT_PROFILE_ID,
    )
    assert qwen2509_q4.status == "PASS"
    assert {record.step for record in qwen2509_q4.records} == {"RF", "OP"}
    assert {record.mode for record in qwen2509_q4.records} == {MODE_EDIT_REFERENCE}

    flux9_source = mlxgen.get_model_validation(
        "black-forest-labs/FLUX.2-klein-9B",
        profile_id=mlxgen.REFRAME_OUTPAINT_PROFILE_ID,
    )
    assert flux9_source.status == "STALE"
    assert next(record for record in flux9_source.records if record.step == "RF").artifact_path.endswith(
        "flux2_9b_source_reframe_b_wide_anchors.png"
    )


def test_flux2_klein_base_starship_profile_reports_source_model_statuses():
    profile = mlxgen.get_validation_profile(mlxgen.FLUX2_KLEIN_BASE_STARSHIP_PROFILE_ID)
    assert (
        profile.canonical_source == "docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png"
    )
    assert len(profile.records) == 10
    for record in profile.records:
        assert Path(record.artifact_path).exists()
        for source_image in record.source_images:
            assert Path(source_image).exists()
        if record.step == "F":
            plan = mlxgen.resolve_generation_plan(
                model=record.model,
                image_count=1,
                has_outpaint=True,
            )
        elif record.step == "E":
            plan = mlxgen.resolve_generation_plan(model=record.model, image_count=2)
        elif record.step == "B":
            plan = mlxgen.resolve_generation_plan(
                model=record.model,
                image_count=1,
                has_image_strength=True,
            )
        else:
            plan = mlxgen.resolve_generation_plan(model=record.model, image_count=1)
        assert plan.mode == record.mode

    base9b = mlxgen.get_model_validation(
        "black-forest-labs/FLUX.2-klein-base-9B",
        profile_id=mlxgen.FLUX2_KLEIN_BASE_STARSHIP_PROFILE_ID,
    )
    assert base9b.status == "PASS"
    assert {record.step for record in base9b.records} == {"B", "C", "D", "E", "F"}

    base4b = mlxgen.get_model_validation(
        "black-forest-labs/FLUX.2-klein-base-4B",
        profile_id=mlxgen.FLUX2_KLEIN_BASE_STARSHIP_PROFILE_ID,
    )
    assert base4b.status == "PARTIAL"
    assert next(record for record in base4b.records if record.step == "E").status == "PARTIAL"


def test_multi_reference_validation_records_list_reference_inputs():
    profile = mlxgen.get_validation_profile()

    for record in profile.records:
        if record.mode != MODE_MULTI_REFERENCE:
            continue
        assert len(record.source_images) >= 2
        for source_image in record.source_images:
            assert Path(source_image).exists()
