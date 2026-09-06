import mlx.core as mx
import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.wan.variants.wan_bernini import BerniniRenderer
from mflux.models.wan.weights import WanWeightDefinition
from mflux.task_inference import TaskInferenceError, get_model_capabilities, resolve_generation_plan
from mflux.utils.exceptions import ModelConfigError


def test_bernini_catalog_config_uses_factored_wan_components():
    config = ModelConfig.from_name("bernini-r-1.3b")

    assert config.model_name == "ByteDance/Bernini-R-1.3B-Diffusers"
    assert config.custom_transformer_model == config.model_name
    assert config.transformer_overrides["component_base_model"] == config.model_name
    assert config.transformer_overrides["supports_bernini_renderer"] is True
    assert config.transformer_overrides["flow_shift"] == 5.0
    assert config.transformer_overrides["unipc_flow_sigma_schedule"] == "diffusers-0.35.2"
    assert config.transformer_overrides["expected_renderer_config"]["shift"] == 3.0
    assert config.transformer_overrides["num_layers"] == 30
    assert config.transformer_overrides["num_attention_heads"] == 12
    assert config.transformer_overrides["default_width"] == 848
    assert config.transformer_overrides["default_reference_guidance"] == 4.5
    assert config.transformer_overrides["default_source_guidance"] == 1.25
    assert config.transformer_overrides["default_apg_eta"] == 0.5


def test_bernini_transformer_preserves_official_module_precision_policy():
    component = WanWeightDefinition.for_config(ModelConfig.bernini_r_1_3b()).get_components()[0]
    assert component.precision == mx.bfloat16
    assert component.precision_override is not None

    weights = {
        "patch_embedding.weight": mx.ones((1,), dtype=mx.float32),
        "patch_embedding.bias": mx.ones((1,), dtype=mx.float32),
        "condition_embedder.time_embedder.linear_1.weight": mx.ones((1,), dtype=mx.float32),
        "condition_embedder.time_embedder.linear_2.bias": mx.ones((1,), dtype=mx.float32),
        "condition_embedder.time_proj.weight": mx.ones((1,), dtype=mx.float32),
        "condition_embedder.time_proj.bias": mx.ones((1,), dtype=mx.float32),
        "condition_embedder.text_embedder.linear_1.weight": mx.ones((1,), dtype=mx.float32),
        "blocks.0.scale_shift_table": mx.ones((1,), dtype=mx.float32),
        "blocks.0.norm2.weight": mx.ones((1,), dtype=mx.float32),
        "blocks.0.norm2.bias": mx.ones((1,), dtype=mx.float32),
        "blocks.0.attn1.norm_q.weight": mx.ones((1,), dtype=mx.float32),
        "blocks.0.attn1.norm_k.weight": mx.ones((1,), dtype=mx.float32),
        "norm_out.weight": mx.ones((1,), dtype=mx.float32),
        "blocks.0.attn1.to_q.weight": mx.ones((1,), dtype=mx.float32),
        "scale_shift_table": mx.ones((1,), dtype=mx.float32),
    }
    converted = WeightLoader._convert_precision(
        weights,
        component.precision,
        precision_override=component.precision_override,
    )

    fp32 = {
        "condition_embedder.time_embedder.linear_1.weight",
        "condition_embedder.time_embedder.linear_2.bias",
        "blocks.0.scale_shift_table",
        "blocks.0.norm2.weight",
        "blocks.0.norm2.bias",
        "scale_shift_table",
    }
    assert {key for key, value in converted.items() if value.dtype == mx.float32} == fp32
    assert {key for key, value in converted.items() if value.dtype == mx.bfloat16} == set(weights) - fp32


def test_ordinary_wan_transformers_keep_blanket_runtime_precision():
    component = WanWeightDefinition.for_config(ModelConfig.wan2_2_ti2v_5b()).get_components()[0]

    assert component.precision == mx.bfloat16
    assert component.precision_override is None

@pytest.mark.parametrize(
    "model_name",
    [
        "ByteDance/Bernini-R",
        "third-party/Bernini-R-A14B",
        "third-party/Bernini-R-1.3B-custom",
    ],
)
def test_bernini_does_not_infer_unvalidated_renderer_repositories(model_name):
    with pytest.raises(ModelConfigError, match="Cannot infer base_model"):
        ModelConfig.from_name(model_name)


@pytest.mark.parametrize("bits", [3, 4, 5, 6, 8])
def test_bernini_rejects_unvalidated_low_bit_quantization(bits):
    with pytest.raises(ValueError, match="BF16 inference only"):
        BerniniRenderer(quantize=bits)


def test_bernini_capabilities_expose_reference_role_separately():
    capabilities = get_model_capabilities(model="ByteDance/Bernini-R-1.3B-Diffusers")
    rows = {row.id: row for row in capabilities.capabilities}

    assert capabilities.schema_version == 15
    assert capabilities.family == "wan"
    assert capabilities.label == "Bernini-R 1.3B"
    assert rows["bernini.reference-video"].public_task == "text-to-video"
    assert rows["bernini.reference-video"].min_images == 0
    assert rows["bernini.reference-video"].min_reference_images == 1
    assert rows["bernini.reference-video"].max_reference_images == 8
    assert rows["bernini.reference-video"].canvas_policies == ("exact-resize",)
    assert rows["bernini.reference-video"].default_canvas_policy == "exact-resize"
    assert rows["bernini.reference-video"].resize_modes == ("resize",)
    assert rows["bernini.reference-video-edit"].min_videos == 1
    assert rows["bernini.reference-video-edit"].min_reference_images == 1
    assert rows["bernini.reference-video-edit"].max_reference_images == 8
    assert rows["bernini.video-edit"].canvas_policies == ("source-aspect",)
    assert rows["bernini.video-edit"].default_canvas_policy == "source-aspect"
    assert rows["bernini.video-edit"].resize_modes == ("resize",)
    assert rows["bernini.reference-video-edit"].canvas_policies == ("source-aspect",)
    assert rows["bernini.reference-video-edit"].resize_modes == ("resize",)


def test_bernini_reference_to_video_plan_is_not_first_frame_i2v():
    plan = resolve_generation_plan(model="bernini-r-1.3b", reference_image_count=2)

    assert plan.public_task == "text-to-video"
    assert plan.mode == "reference-video"
    assert plan.image_count == 0
    assert plan.reference_image_count == 2


def test_bernini_reference_guided_video_edit_plan_accepts_mixed_roles():
    plan = resolve_generation_plan(model="bernini-r-1.3b", video_count=1, reference_image_count=2)

    assert plan.public_task == "video-to-video"
    assert plan.mode == "reference-video-edit"
    assert plan.video_count == 1
    assert plan.reference_image_count == 2


def test_bernini_plain_video_edit_selects_non_reference_mode():
    plan = resolve_generation_plan(model="bernini-r-1.3b", video_count=1)

    assert plan.public_task == "video-to-video"
    assert plan.mode == "latent-video"


def test_bernini_reference_to_video_requires_a_reference():
    with pytest.raises(TaskInferenceError, match="does not support text-to-video"):
        resolve_generation_plan(model="bernini-r-1.3b")


def test_bernini_reference_count_is_bounded_by_the_official_eight_reference_case():
    with pytest.raises(TaskInferenceError, match="9 --reference-image"):
        resolve_generation_plan(model="bernini-r-1.3b", reference_image_count=9)


def test_reference_images_fail_closed_on_ordinary_wan():
    with pytest.raises(TaskInferenceError, match="--reference-image"):
        resolve_generation_plan(model="wan2.2-t2v-a14b", reference_image_count=1)


def test_primary_image_and_video_rejection_remains_with_reference_role():
    with pytest.raises(TaskInferenceError, match="either input images or input videos"):
        resolve_generation_plan(
            model="bernini-r-1.3b",
            image_count=1,
            video_count=1,
            reference_image_count=1,
        )


def test_vace_capabilities_now_report_reference_images():
    capabilities = get_model_capabilities(model="wan2.1-vace-1.3b")
    rows = {row.id: row for row in capabilities.capabilities}

    assert rows["wan.text-video"].max_reference_images is None
    assert rows["wan.video-video"].max_reference_images is None
    plan = resolve_generation_plan(model="wan2.1-vace-1.3b", video_count=1, reference_image_count=2)
    assert plan.reference_image_count == 2
