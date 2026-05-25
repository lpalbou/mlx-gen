from types import SimpleNamespace

from mflux.models.common.weights.saving.model_card_saver import ModelCardSaver
from mflux.models.common.weights.saving.model_saver import ModelSaver


class QwenImageEdit:
    model_config = SimpleNamespace(
        model_name="Qwen/Qwen-Image-Edit-2511",
        base_model=None,
        aliases=["qwen-image-edit", "qwen-edit"],
    )


class ZImageTurbo:
    model_config = SimpleNamespace(
        model_name="Tongyi-MAI/Z-Image-Turbo",
        base_model=None,
        aliases=["z-image-turbo"],
    )


class EmptyWeightDefinition:
    @staticmethod
    def get_tokenizers():
        return []

    @staticmethod
    def get_components():
        return []


def test_model_saver_writes_model_card(tmp_path):
    ModelSaver.save_model(QwenImageEdit(), 4, str(tmp_path), EmptyWeightDefinition)

    assert (tmp_path / "README.md").exists()


def test_model_card_for_qwen_q4_documents_mixed_policy(tmp_path):
    ModelCardSaver.save_model_card(str(tmp_path / "qwen-image-edit-2511-4bit"), QwenImageEdit(), 4)

    card = (tmp_path / "qwen-image-edit-2511-4bit" / "README.md").read_text()

    assert "base_model: Qwen/Qwen-Image-Edit-2511" in card
    assert "pipeline_tag: image-to-image" in card
    assert "- mixed-q4" in card
    assert "- mixed-q4-q8" in card
    assert "q8 for Qwen `*.img_mod_linear` transformer modulation layers" in card
    assert "mlxgen download --model lpalbou/qwen-image-edit-2511-4bit" in card
    assert "not a Diffusers or Transformers `from_pretrained()` checkpoint" in card
    assert "https://github.com/filipstrand/mflux" in card
    assert "https://github.com/lpalbou/mlx-gen" in card
    assert "https://huggingface.co/lpalbou" in card
    assert "Recommended Hugging Face collection: AbstractFramework / mlx-gen" in card


def test_model_card_for_q8_keeps_standard_quantization_wording(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "z-image-turbo-8bit"), ZImageTurbo(), 8)

    assert "pipeline_tag: text-to-image" in card
    assert "- 8-bit" in card
    assert "- mixed-q4" not in card
    assert "This is an MLX q8 checkpoint" in card
    assert "Qwen-specific mixed q4/q8 policy only applies" not in card


def test_model_card_for_qwen_q8_explains_q4_policy_is_not_used(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "qwen-image-edit-2511-8bit"), QwenImageEdit(), 8)

    assert "This is an MLX q8 checkpoint" in card
    assert "Qwen-specific mixed q4/q8 policy only applies" in card
