from mflux.release.validation_registry import (
    MASKED_EDIT_MATRIX_PROFILE_ID,
    default_validation_profile_id_for_model,
    get_model_validation,
)


def test_default_profile_prefers_exact_row_evidence_over_base_model_fallback():
    # The 8bit package row must not default to the source checkpoint's masked-matrix records
    # (which only match through the base_model fallback); its own exact evidence wins.
    profile_id = default_validation_profile_id_for_model("AbstractFramework/qwen-image-8bit")
    validation = get_model_validation("AbstractFramework/qwen-image-8bit", profile_id=profile_id)
    assert profile_id != MASKED_EDIT_MATRIX_PROFILE_ID
    assert validation.records
    assert {record.model for record in validation.records} == {"AbstractFramework/qwen-image-8bit"}


def test_default_profile_uses_exact_masked_matrix_rows():
    for model in [
        "Qwen/Qwen-Image",
        "AbstractFramework/qwen-image-4bit",
        "AbstractFramework/z-image-8bit",
    ]:
        assert default_validation_profile_id_for_model(model) == MASKED_EDIT_MATRIX_PROFILE_ID, model


def test_base_model_fallback_still_serves_source_repacks_without_exact_rows():
    # No profile holds exact rows for the never-published bf16 repack; it inherits the source
    # checkpoint's records through the documented base-model fallback.
    profile_id = default_validation_profile_id_for_model("AbstractFramework/qwen-image-bf16")
    validation = get_model_validation("AbstractFramework/qwen-image-bf16", profile_id=profile_id)
    assert validation.records
    assert {record.model for record in validation.records} == {"Qwen/Qwen-Image"}
