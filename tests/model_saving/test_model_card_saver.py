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


class Wan2_2_TI2V:
    model_config = SimpleNamespace(
        model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        base_model=None,
        aliases=["wan2.2-ti2v-5b"],
    )


class Wan2_2_T2V_A14B:
    model_config = SimpleNamespace(
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        base_model=None,
        aliases=["wan2.2-t2v-a14b"],
    )


class Wan2_2_I2V_A14B:
    model_config = SimpleNamespace(
        model_name="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        base_model=None,
        aliases=["wan2.2-i2v-a14b"],
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


class FIBO:
    model_config = SimpleNamespace(
        model_name="briaai/FIBO",
        base_model=None,
        aliases=["fibo"],
    )


class SeedVR2:
    model_config = SimpleNamespace(
        model_name="ByteDance-Seed/SeedVR2-3B",
        base_model=None,
        aliases=["seedvr2-3b", "seedvr2"],
    )


class SeedVR2_7B:
    model_config = SimpleNamespace(
        model_name="ByteDance-Seed/SeedVR2-7B",
        base_model=None,
        aliases=["seedvr2-7b"],
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


def test_model_card_for_fibo_q8_documents_mixed_bf16_policy(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "fibo-8bit"), FIBO(), 8)

    assert "base_model: briaai/FIBO" in card
    assert "license: other" in card
    assert "license_name: bria-fibo" in card
    assert "- mixed-q8-bf16" in card
    assert "mixed q8/BF16 checkpoint for base FIBO text-to-image generation" in card
    assert "q8 for quantizable FIBO transformer and text-encoder linears" in card
    assert "BF16 for the FIBO VAE" in card
    assert "FIBO conditioning, timestep, caption-projection, normalization" in card


def test_model_card_for_fibo_q4_documents_mixed_bf16_policy(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "fibo-4bit"), FIBO(), 4)

    assert "base_model: briaai/FIBO" in card
    assert "license: other" in card
    assert "license_name: bria-fibo" in card
    assert "- mixed-q4-bf16" in card
    assert "- mixed-q4-q8" not in card
    assert "mixed q4/BF16 checkpoint for base FIBO text-to-image generation" in card
    assert "q4 for quantizable FIBO transformer and text-encoder linears" in card
    assert "BF16 for the FIBO VAE" in card


def test_model_card_for_seedvr2_q8_uses_upscale_command(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "seedvr2-3b-8bit"), SeedVR2(), 8)

    assert "base_model: ByteDance-Seed/SeedVR2-3B" in card
    assert "license: apache-2.0" in card
    assert "pipeline_tag: image-to-image" in card
    assert "- seedvr2" in card
    assert "- image-upscaling" in card
    assert "SeedVR2 3B image super-resolution" in card
    assert "q8 for quantizable SeedVR2 transformer linears and VAE attention linears" in card
    assert "mlxgen upscale" in card
    assert "mflux-upscale-seedvr2" not in card
    assert "mlxgen generate" not in card
    assert "mlxgen download --model AbstractFramework/seedvr2-3b-8bit" in card
    assert "--resolution 2x" in card


def test_model_card_for_seedvr2_7b_q4_uses_7b_wording(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "seedvr2-7b-4bit"), SeedVR2_7B(), 4)

    assert "base_model: ByteDance-Seed/SeedVR2-7B" in card
    assert "SeedVR2 7B image super-resolution" in card
    assert "SeedVR2 3B image super-resolution" not in card
    assert "mlxgen download --model AbstractFramework/seedvr2-7b-4bit" in card
    assert "mlxgen upscale" in card


def test_model_card_for_ernie_documents_bf16_usage(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "ernie-image-turbo"), ErnieImageTurbo(), None)

    assert "base_model: baidu/ERNIE-Image-Turbo" in card
    assert "license: apache-2.0" in card
    assert "- ernie-image-turbo" in card
    assert "This checkpoint stores MLX-Gen ERNIE Image Turbo generation weights" in card
    assert "Prompt Enhancer files are not bundled" in card
    assert "--width 512" in card
    assert "--height 512" in card
    assert "--steps 8" in card
    assert "--guidance 1" in card
    assert "Prepared and contributed by" in card
    assert "Quantized and contributed by" not in card


def test_model_card_for_ernie_q4_documents_mixed_policy(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "ernie-image-turbo-4bit"), ErnieImageTurbo(), 4)

    assert "This is an MLX mixed q4/q8 checkpoint for ERNIE Image Turbo." in card
    assert "- mixed-q4" in card
    assert "- mixed-q4-q8" in card
    assert "q4 for ERNIE transformer Q/K attention projections and feed-forward modules" in card
    assert "q8 for ERNIE transformer V/O attention projections" in card
    assert "q8 for Mistral3 text-encoder and Prompt Enhancer linears" in card
    assert "q8 for quantizable ERNIE VAE attention modules" in card
    assert "Prompt Enhancer files are not bundled" in card


def test_model_card_for_ernie_q8_mentions_q4_uses_mixed_policy(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "ernie-image-turbo-8bit"), ErnieImageTurbo(), 8)

    assert "This is an MLX q8 checkpoint" in card
    assert "q8 for quantizable ERNIE transformer modules" in card
    assert "q8 for quantizable ERNIE text-encoder modules" in card
    assert "q8 for quantizable ERNIE VAE attention modules" in card
    assert "ERNIE q4 uses a model-specific mixed q4/q8 policy" in card
    assert "https://github.com/lpalbou/mlx-gen/blob/main/docs/quantization.md" in card
    assert "Prompt Enhancer files are not bundled" in card


def test_model_card_for_wan_q8_uses_video_metadata_and_usage(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "wan2.2-ti2v-5b-diffusers-8bit"), Wan2_2_TI2V(), 8)

    assert "base_model: Wan-AI/Wan2.2-TI2V-5B-Diffusers" in card
    assert "license: apache-2.0" in card
    assert "pipeline_tag: text-to-video" in card
    assert "- video-generation" in card
    assert "- image-to-video" in card
    assert "- mixed-q8-bf16" in card
    assert "mixed q8/BF16 package" in card
    assert "q8 for quantizable Wan transformer attention and feed-forward modules" in card
    assert "BF16 for the Wan VAE" in card
    assert "BF16 for Wan transformer conditioning/output projection linears" in card
    assert "the UMT5 text encoder" in card
    assert "Wan q4 quality and any possible mixed q4/q8 policy are still under validation" in card
    assert "--task text-to-video" not in card
    assert "--frames 121" in card
    assert "--fps 24" in card
    assert "--output video.mp4" in card
    assert "--output image.png" not in card


def test_model_card_for_wan_bf16_documents_prepared_runtime_precision(tmp_path):
    card = ModelCardSaver.render_model_card(
        str(tmp_path / "wan2.2-t2v-a14b-diffusers-bf16"),
        Wan2_2_T2V_A14B(),
        None,
    )

    assert "base_model: Wan-AI/Wan2.2-T2V-A14B-Diffusers" in card
    assert "This prepared derivative follows the Apache 2.0 license of the source model." in card
    assert "MLX quantization tensors" not in card
    assert "without an explicit quantization level" in card
    assert "loads transformer and VAE weights at BF16 runtime precision" in card
    assert "The UMT5 text encoder is preserved" in card
    assert "recommended full-size Wan A14B shape" in card
    assert "separate short low-RAM validation profiles" in card
    assert "`Max RSS` can under-report Apple unified-memory/Metal pressure" in card
    assert "--frames 81" in card
    assert "--guidance-2 3" in card
    assert "Prepared and contributed by" in card


def test_model_card_for_wan_a14b_q8_uses_validation_sized_usage(tmp_path):
    card = ModelCardSaver.render_model_card(
        str(tmp_path / "wan2.2-t2v-a14b-diffusers-8bit"),
        Wan2_2_T2V_A14B(),
        8,
    )

    assert "intentionally validation-sized" in card
    assert "full-size `1280x720`, 81-frame, 40-step readiness" in card
    assert "short low-RAM profiles" in card
    assert "Apple M5 Max with 128 GiB unified memory" in card
    assert "`MLX Peak` is only the MLX allocator high-water mark" in card
    assert "--width 384" in card
    assert "--height 224" in card
    assert "--frames 33" in card
    assert "--steps 12" in card
    assert "--fps 8" in card
    assert "--low-ram" in card
    assert "--metadata" in card
    assert "--width 1280" not in card
    assert "--frames 81" not in card
    assert "- image-to-video" not in card


def test_model_card_for_wan_t2v_a14b_ignores_shared_ti2v_runtime_class_name(tmp_path):
    class Wan2_2_TI2V:
        model_config = Wan2_2_T2V_A14B.model_config

    card = ModelCardSaver.render_model_card(
        str(tmp_path / "wan2.2-t2v-a14b-diffusers-8bit"),
        Wan2_2_TI2V(),
        8,
    )

    assert "pipeline_tag: text-to-video" in card
    assert "- text-to-video" in card
    assert "- image-to-video" not in card
    assert "--task text-to-video" not in card


def test_model_card_for_wan_i2v_a14b_uses_image_to_video_metadata(tmp_path):
    card = ModelCardSaver.render_model_card(
        str(tmp_path / "wan2.2-i2v-a14b-diffusers-8bit"),
        Wan2_2_I2V_A14B(),
        8,
    )

    assert "pipeline_tag: image-to-video" in card
    assert "- image-to-video" in card
    assert "- text-to-video" not in card
    assert "--task image-to-video" not in card
    assert "--image input.png" in card
    assert "--width 384" in card
    assert "--height 384" in card
    assert "--low-ram" in card
    assert "--frames 33" in card
    assert "--guidance 3.5" in card
    assert "--guidance-2 3.5" in card


def test_model_card_for_wan_i2v_a14b_ignores_shared_ti2v_runtime_class_name(tmp_path):
    class Wan2_2_TI2V:
        model_config = Wan2_2_I2V_A14B.model_config

    card = ModelCardSaver.render_model_card(
        str(tmp_path / "wan2.2-i2v-a14b-diffusers-8bit"),
        Wan2_2_TI2V(),
        8,
    )

    assert "pipeline_tag: image-to-video" in card
    assert "- image-to-video" in card
    assert "- text-to-video" not in card
    assert "--task image-to-video" not in card


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
