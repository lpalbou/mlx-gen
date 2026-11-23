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


class MiniMaxH3:
    model_config = SimpleNamespace(
        model_name="MiniMaxAI/MiniMax-H3",
        base_model=None,
        aliases=["minimax-h3", "minimaxai/minimax-h3", "h3"],
    )


def test_model_card_for_minimax_h3_q8_is_a_video_card_under_the_community_license(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "minimax-h3-8bit"), MiniMaxH3(), 8)

    assert "base_model: MiniMaxAI/MiniMax-H3" in card
    assert "pipeline_tag: text-to-video" in card
    assert "license: other" in card and "license: apache-2.0" not in card
    assert "license_name: minimax-h3-community-license-agreement" in card
    assert "license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE" in card
    assert "extra_gated_prompt:" in card
    for tag in ("- 8-bit", "- mixed-q8-bf16", "- minimax-h3", "- image-to-video", "- audio-video-generation"):
        assert tag in card
    assert "- mixed-precision" not in card
    # The territory exclusion and the files the license requires are stated, not left to the link.
    assert "the European Union, the United Kingdom, the Republic of Korea and the United States of America" in card
    assert "`LICENSE`" in card and "`NOTICE`" in card
    assert "The weight files here are not the original release" in card
    assert "mlxgen download --model AbstractFramework/minimax-h3-8bit --base-model minimax-h3-turbo-544p" in card
    assert "  --base-model minimax-h3-turbo-544p \\" in card
    assert "--soundscape" in card and "--output fox.mp4" in card
    assert "Do not pass `--quantize`" in card
    assert "--steps 20" not in card and "image.png" not in card


def test_minimax_h3_card_states_the_widths_the_policy_assigns(tmp_path):
    from mflux.models.minimax_h3.weights.h3_weight_definition import MiniMaxH3WeightDefinition as Definition

    for bits in Definition.QUANTIZATION_BITS:
        card = ModelCardSaver.render_model_card(str(tmp_path / f"minimax-h3-{bits}bit"), MiniMaxH3(), bits)
        transformer_bits = Definition.component_bits("transformer", bits)
        conditioner_bits = Definition.component_bits("text_encoder", bits)
        assert f"- q{transformer_bits} for the transformer attention and feed-forward linears" in card
        assert f"- q{conditioner_bits} for the Qwen3-VL conditioner's 50 decoder layers" in card
        assert "As released: the conditioner's vision tower" in card and "`adaln_proj`" in card

    # The package is named 4-bit and its transformer is not: the card has to say so.
    mixed = ModelCardSaver.render_model_card(str(tmp_path / "minimax-h3-4bit"), MiniMaxH3(), 4)
    assert "- 4-bit" in mixed and "- mixed-precision" in mixed and "- mixed-q8-bf16" not in mixed
    assert "a 4-bit conditioner and a 6-bit transformer" in mixed
    assert "The transformer is 6-bit, not 4-bit." in mixed
    assert "mlxgen download --model AbstractFramework/minimax-h3-4bit --base-model minimax-h3-turbo-544p" in mixed


def test_minimax_h3_cards_show_every_width_with_its_modes_and_the_machines_they_ran_on(tmp_path):
    from mflux.task_inference import (
        MINIMAX_H3_MEASURED_RUNS,
        MINIMAX_H3_PACKAGE_WEIGHT_BYTES,
        MINIMAX_H3_SMALLEST_TESTED_MACHINE,
    )

    assert MINIMAX_H3_SMALLEST_TESTED_MACHINE == {
        None: {"default": 96, "low_ram": 64, "stream_blocks": 32},
        8: {"default": 48, "low_ram": 36, "stream_blocks": 24},
        4: {"low_ram": 32},
    }
    gib = 1024**3
    footprints = {
        (run.quantize, run.low_ram, run.stream_blocks): f"{run.footprint_bytes / gib:.1f} GiB"
        for run in MINIMAX_H3_MEASURED_RUNS
        if (run.width, run.height) == (960, 544)
    }

    def row(bits, name, marker):
        tested = MINIMAX_H3_SMALLEST_TESTED_MACHINE[bits]
        cells = []
        for mode, key in (("default", (False, False)), ("low_ram", (True, False)), ("stream_blocks", (True, True))):
            peak = footprints.get((bits, *key))
            ran_on = f", ran on {tested[mode]} GB" if mode in tested else ""
            cells.append(f"{peak}{ran_on}" if peak else "not measured")
        disk = f"{MINIMAX_H3_PACKAGE_WEIGHT_BYTES[bits] / gib:.1f} GiB"
        return f"| `AbstractFramework/minimax-h3-{name}`{marker} | {disk} | " + " | ".join(cells) + " |"

    names = {None: "bf16", 8: "8bit", 4: "4bit"}
    for bits, name in names.items():
        card = ModelCardSaver.render_model_card(str(tmp_path / f"minimax-h3-{name}"), MiniMaxH3(), bits)
        # Every card shows the three packages, so a reader on the wrong one for their Mac sees the others.
        for other, other_name in names.items():
            assert row(other, other_name, " (this package)" if other == bits else "") in card
        assert "| Package | Disk | Default | `--low-ram` | `--stream-blocks --low-ram` |" in card
        assert "The smaller machines were tested on that Mac with the difference in RAM locked away" in card
        assert "The three modes produce the same clip." in card
        assert "This repository is gated" in card and "hf auth login" in card
        # The command is the default mode; the two options are named for a Mac with less memory.
        assert "--low-ram \\" not in card and "--stream-blocks \\" not in card
        assert "add `--low-ram`, and then `--stream-blocks`" in card
    eight = ModelCardSaver.render_model_card(str(tmp_path / "minimax-h3-8bit"), MiniMaxH3(), 8)
    assert "37.0 GiB, ran on 48 GB | 26.3 GiB, ran on 36 GB | 11.1 GiB, ran on 24 GB" in eight
    assert "54.4 GiB, ran on 96 GB | 44.0 GiB, ran on 64 GB | 11.7 GiB, ran on 32 GB" in eight


def test_the_16_bit_card_describes_the_released_weights(tmp_path):
    card = ModelCardSaver.render_model_card(str(tmp_path / "minimax-h3-bf16"), MiniMaxH3(), None)
    assert "- bf16" in card and "- 8-bit" not in card and "- mixed-q8-bf16" not in card
    assert "holds the MiniMax-H3 weights as released" in card and "without quantization" in card
    assert "mlxgen download --model AbstractFramework/minimax-h3-bf16 --base-model minimax-h3-turbo-544p" in card
    assert "prepared derivative" in card
    assert "This repository is an unquantized derivative of MiniMaxAI/MiniMax-H3" in card
    assert "docs/minimax-h3-memory.md" in card

    # The quantized cards say how far each width is from these weights.
    eight = ModelCardSaver.render_model_card(str(tmp_path / "minimax-h3-8bit"), MiniMaxH3(), 8)
    four = ModelCardSaver.render_model_card(str(tmp_path / "minimax-h3-4bit"), MiniMaxH3(), 4)
    assert "This repository is a quantized derivative of MiniMaxAI/MiniMax-H3" in eight
    assert "within 1.2 to 2.0 times the 16-bit model's own numerical noise" in eight
    assert "takes the nearest level of the grid that is stored" in eight
    assert "2.4 to 5.3 times the 16-bit model's own numerical noise" in four and "80 to 88%" in four


def test_card_version_lines_name_the_release_that_wrote_the_weights(tmp_path):
    import mlx.core as mx

    package = tmp_path / "minimax-h3-8bit"
    (package / "transformer").mkdir(parents=True)
    mx.save_safetensors(
        str(package / "transformer" / "0.safetensors"), {"w": mx.zeros((2, 2))}, {"mflux_version": "0.39.0"}
    )
    card = ModelCardSaver.render_model_card(str(package), MiniMaxH3(), 8)
    # The floor is the release whose memory figures the card quotes, whatever wrote the weights.
    assert "Requires `mlx-gen >= 0.40.0`:" in card and "Generated with `mlx-gen 0.39.0`." in card

    current = VersionUtil.get_mflux_version()
    fresh = ModelCardSaver.render_model_card(str(tmp_path / "not-written-yet"), MiniMaxH3(), 8)
    assert f"Generated with `mlx-gen {current}`." in fresh

    other = ModelCardSaver.render_model_card(str(package), QwenImageEdit(), 4)
    assert "Requires `mlx-gen >= 0.39.0`." in other and "Generated with `mlx-gen 0.39.0`." in other


def test_minimax_h3_notice_carries_the_prescribed_text_and_the_modification():
    notice = ModelCardSaver.render_minimax_h3_notice("MiniMaxAI/MiniMax-H3", 4)
    assert notice.startswith(
        "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, "
        "Copyright © 2026 MiniMax. All Rights Reserved."
    )
    assert "Modification notice" in notice and "quantized (4-bit package)" in notice
    assert f"mlx-gen {VersionUtil.get_mflux_version()}" in notice
    assert "quantized" not in ModelCardSaver.render_minimax_h3_notice("MiniMaxAI/MiniMax-H3", None)
