from types import SimpleNamespace

from mflux.models.common.weights.saving.model_card_saver import ModelCardSaver
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.utils.version_util import VersionUtil


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


class ErnieImageTurbo:
    model_config = SimpleNamespace(
        model_name="baidu/ERNIE-Image-Turbo",
        base_model=None,
        aliases=["ernie-image-turbo"],
    )


class Flux2Klein4B:
    model_config = SimpleNamespace(
        model_name="black-forest-labs/FLUX.2-klein-4B",
        base_model=None,
        aliases=["flux2-klein-4b", "klein-4b"],
    )


class Flux2Klein9B:
    model_config = SimpleNamespace(
        model_name="black-forest-labs/FLUX.2-klein-9B",
        base_model=None,
        aliases=["flux2-klein-9b", "klein-9b"],
    )


class Flux2KleinBase9B:
    model_config = SimpleNamespace(
        model_name="black-forest-labs/FLUX.2-klein-base-9B",
        base_model=None,
        aliases=["flux2-klein-base-9b", "klein-base-9b"],
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
    assert "https://github.com/lpalbou/mlx-gen/blob/main/docs/quantization.md" in card
    assert "python -m pip install -U mlx-gen" in card
    assert "mlxgen download --model AbstractFramework/qwen-image-edit-2511-4bit" in card
    assert "not a Diffusers or Transformers `from_pretrained()` checkpoint" in card
    assert f"Generated with `mlx-gen {VersionUtil.get_mflux_version()}`" in card
    assert "https://github.com/filipstrand/mflux" in card
    assert "https://github.com/lpalbou/mlx-gen" in card
    assert "https://huggingface.co/lpalbou" in card
    assert "Recommended Hugging Face collection" not in card
    assert "mflux-save" not in card
    assert "mflux-*" not in card


def test_model_card_for_q8_keeps_standard_quantization_wording(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "z-image-turbo-8bit"), ZImageTurbo(), 8)

    assert "pipeline_tag: text-to-image" in card
    assert "license: apache-2.0" in card
    assert "- 8-bit" in card
    assert "- mixed-q4" not in card
    assert "This is an MLX q8 checkpoint" in card
    assert "Qwen-specific mixed q4/q8 policy only applies" not in card
    assert "--steps 8" in card
    assert "--guidance 0" in card


def test_model_card_for_ernie_documents_bf16_usage(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "ernie-image-turbo"), ErnieImageTurbo(), None)

    assert "base_model: baidu/ERNIE-Image-Turbo" in card
    assert "license: apache-2.0" in card
    assert "- ernie-image-turbo" in card
    assert "This checkpoint stores MLX-Gen weights without an explicit quantization level." in card
    assert "--width 512" in card
    assert "--height 512" in card
    assert "--steps 8" in card
    assert "--guidance 1" in card
    assert "Prepared and contributed by" in card
    assert "Quantized and contributed by" not in card


def test_model_card_for_qwen_q8_explains_q4_policy_is_not_used(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "qwen-image-edit-2511-8bit"), QwenImageEdit(), 8)

    assert "This is an MLX q8 checkpoint" in card
    assert "Qwen-specific mixed q4/q8 policy only applies" in card
    assert "with `--quantize 4`" in card
    assert "https://github.com/lpalbou/mlx-gen/blob/main/docs/quantization.md" in card


def test_model_card_for_flux2_4b_uses_apache_license(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "flux.2-klein-4b-4bit"), Flux2Klein4B(), 4)

    assert "license: apache-2.0" in card
    assert "FLUX Non-Commercial License" not in card
    assert "This quantized derivative follows the Apache 2.0 license of the source model." in card


def test_model_card_for_flux2_9b_marks_non_commercial_gated_derivative(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "flux.2-klein-9b-4bit"), Flux2Klein9B(), 4)

    assert "license: other" in card
    assert "license_name: flux-non-commercial-license" in card
    assert "license_link: https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md" in card
    assert "extra_gated_prompt:" in card
    assert "Acceptable Use Policy" in card
    assert "This checkpoint is a quantized derivative of a gated" in card
    assert "Host this derivative as a gated Hugging Face repository" in card


def test_model_card_for_flux2_base_9b_links_base_license(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "flux.2-klein-base-9b-8bit"), Flux2KleinBase9B(), 8)

    assert "license_name: flux-non-commercial-license" in card
    assert "license_link: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/blob/main/LICENSE.md" in card
