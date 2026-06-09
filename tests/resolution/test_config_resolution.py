import pytest

from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.utils.exceptions import InvalidBaseModel, ModelConfigError


class TestConfigResolutionExactMatch:
    @pytest.mark.fast
    def test_exact_alias_match(self):
        config = ConfigResolution.resolve(model_name="schnell")

        assert config.model_name == "black-forest-labs/FLUX.1-schnell"
        assert "schnell" in config.aliases

    @pytest.mark.fast
    def test_exact_alias_match_dev(self):
        config = ConfigResolution.resolve(model_name="dev")

        assert config.model_name == "black-forest-labs/FLUX.1-dev"

    @pytest.mark.fast
    def test_exact_alias_match_fibo(self):
        config = ConfigResolution.resolve(model_name="fibo")

        assert config.model_name == "briaai/FIBO"

    @pytest.mark.fast
    def test_exact_hf_name_match(self):
        config = ConfigResolution.resolve(model_name="black-forest-labs/FLUX.1-schnell")

        assert config.model_name == "black-forest-labs/FLUX.1-schnell"

    @pytest.mark.fast
    def test_exact_hf_name_match_ernie_image_turbo(self):
        config = ConfigResolution.resolve(model_name="baidu/ERNIE-Image-Turbo")

        assert config.model_name == "baidu/ERNIE-Image-Turbo"
        assert "ernie-image-turbo" in config.aliases

    @pytest.mark.fast
    def test_exact_hf_name_match_wan2_2_ti2v_5b(self):
        config = ConfigResolution.resolve(model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers")

        assert config.model_name == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        assert "wan2.2-ti2v-5b" in config.aliases

    @pytest.mark.fast
    def test_exact_hf_name_match_wan2_2_t2v_a14b(self):
        config = ConfigResolution.resolve(model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers")

        assert config.model_name == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        assert config.base_model is None
        assert config.transformer_overrides["in_channels"] == 16
        assert config.transformer_overrides["vae_config"]["z_dim"] == 16

    @pytest.mark.fast
    def test_exact_hf_name_match_wan2_2_i2v_a14b(self):
        config = ConfigResolution.resolve(model_name="Wan-AI/Wan2.2-I2V-A14B-Diffusers")

        assert config.model_name == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        assert config.base_model is None
        assert config.transformer_overrides["in_channels"] == 36
        assert config.transformer_overrides["vae_config"]["z_dim"] == 16


class TestConfigResolutionExplicitBase:
    @pytest.mark.fast
    def test_explicit_base_model(self):
        config = ConfigResolution.resolve(model_name="my-custom-model", base_model="schnell")

        assert config.model_name == "my-custom-model"
        assert config.base_model == "black-forest-labs/FLUX.1-schnell"
        assert config.max_sequence_length == 256  # schnell's value

    @pytest.mark.fast
    def test_explicit_base_model_dev(self):
        config = ConfigResolution.resolve(model_name="org/my-finetune", base_model="dev")

        assert config.model_name == "org/my-finetune"
        assert config.base_model == "black-forest-labs/FLUX.1-dev"
        assert config.supports_guidance is True  # dev's value

    @pytest.mark.fast
    def test_invalid_base_model_raises(self):
        with pytest.raises(InvalidBaseModel):
            ConfigResolution.resolve(model_name="whatever", base_model="invalid-base")


class TestConfigResolutionInferSubstring:
    @pytest.mark.fast
    def test_infer_from_schnell_substring(self):
        config = ConfigResolution.resolve(model_name="my-schnell-finetune")

        assert config.model_name == "my-schnell-finetune"
        assert config.base_model == "black-forest-labs/FLUX.1-schnell"

    @pytest.mark.fast
    def test_infer_from_dev_substring(self):
        config = ConfigResolution.resolve(model_name="dev-lora-something")

        assert config.model_name == "dev-lora-something"
        assert config.base_model == "black-forest-labs/FLUX.1-dev"

    @pytest.mark.fast
    def test_infer_case_insensitive(self):
        config = ConfigResolution.resolve(model_name="MY-SCHNELL-MODEL")

        assert config.base_model == "black-forest-labs/FLUX.1-schnell"

    @pytest.mark.fast
    def test_longer_alias_preferred(self):
        # "dev-kontext" is longer than "dev", should match dev-kontext if present
        config = ConfigResolution.resolve(model_name="my-dev-kontext-model")

        assert config.base_model == "black-forest-labs/FLUX.1-Kontext-dev"

    @pytest.mark.fast
    def test_inferred_config_preserves_text_encoder_overrides(self):
        config = ConfigResolution.resolve(model_name="/models/local-flux2-klein-9b-q4")

        assert config.base_model == "black-forest-labs/FLUX.2-klein-9B"
        assert config.transformer_overrides["num_attention_heads"] == 32
        assert config.text_encoder_overrides["hidden_size"] == 4096

    @pytest.mark.fast
    def test_inferred_config_preserves_scheduler_shift_settings(self):
        config = ConfigResolution.resolve(model_name="Qwen/Qwen-Image-Edit-2511")

        assert config.model_name == "Qwen/Qwen-Image-Edit-2511"
        assert config.sigma_max_shift == 0.9
        assert config.sigma_max_seq_len == 8192
        assert config.sigma_shift_terminal == 0.02

    @pytest.mark.fast
    def test_infers_qwen_image_edit_2511_from_prepared_folder(self):
        config = ConfigResolution.resolve(model_name="AbstractFramework/qwen-image-edit-2511-4bit")

        assert config.base_model == "Qwen/Qwen-Image-Edit-2511"
        assert config.model_name == "AbstractFramework/qwen-image-edit-2511-4bit"

    @pytest.mark.fast
    def test_qwen_edit_versions_are_distinct(self):
        regular = ConfigResolution.resolve(model_name="qwen-image-edit")
        edit_2509 = ConfigResolution.resolve(model_name="qwen-image-edit-2509")
        edit_2511 = ConfigResolution.resolve(model_name="qwen-image-edit-2511")

        assert regular.model_name == "Qwen/Qwen-Image-Edit"
        assert regular.transformer_overrides == {}
        assert edit_2509.model_name == "Qwen/Qwen-Image-Edit-2509"
        assert edit_2509.transformer_overrides == {"qwen_edit_plus": True}
        assert edit_2511.model_name == "Qwen/Qwen-Image-Edit-2511"
        assert edit_2511.transformer_overrides["qwen_edit_plus"] is True
        assert edit_2511.transformer_overrides["zero_cond_t"] is True

    @pytest.mark.fast
    def test_qwen_prepared_edit_versions_infer_by_longest_alias(self):
        regular = ConfigResolution.resolve(model_name="AbstractFramework/qwen-image-edit-8bit")
        edit_2509 = ConfigResolution.resolve(model_name="AbstractFramework/qwen-image-edit-2509-8bit")
        edit_2511 = ConfigResolution.resolve(model_name="AbstractFramework/qwen-image-edit-2511-8bit")

        assert regular.base_model == "Qwen/Qwen-Image-Edit"
        assert regular.transformer_overrides == {}
        assert edit_2509.base_model == "Qwen/Qwen-Image-Edit-2509"
        assert edit_2509.transformer_overrides == {"qwen_edit_plus": True}
        assert edit_2511.base_model == "Qwen/Qwen-Image-Edit-2511"
        assert edit_2511.transformer_overrides["zero_cond_t"] is True

    @pytest.mark.fast
    def test_infers_ernie_image_turbo_from_local_path(self):
        config = ConfigResolution.resolve(model_name="/models/ernie-image-turbo-8bit")

        assert config.base_model == "baidu/ERNIE-Image-Turbo"
        assert config.supports_guidance is True

    @pytest.mark.fast
    def test_infers_wan2_2_ti2v_from_local_path(self):
        config = ConfigResolution.resolve(model_name="/models/wan2.2-ti2v-5b-q8")

        assert config.base_model == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        assert config.transformer_overrides["expand_timesteps"] is True

    @pytest.mark.fast
    def test_infers_wan2_2_t2v_a14b_from_local_path(self):
        config = ConfigResolution.resolve(model_name="/models/wan2.2-t2v-a14b-q8")

        assert config.base_model == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        assert config.transformer_overrides["expand_timesteps"] is False
        assert config.transformer_overrides["in_channels"] == 16


class TestConfigResolutionError:
    @pytest.mark.fast
    def test_unknown_model_without_base_raises(self):
        with pytest.raises(ModelConfigError) as exc_info:
            ConfigResolution.resolve(model_name="totally-unknown-model")

        assert "Cannot infer" in str(exc_info.value)

    @pytest.mark.fast
    def test_flux2_dev_is_not_inferred_from_flux1_dev_alias(self):
        with pytest.raises(ModelConfigError, match="FLUX.2-dev is not supported"):
            ConfigResolution.resolve(model_name="black-forest-labs/FLUX.2-dev")

    @pytest.mark.fast
    def test_unknown_wan_repo_does_not_infer_ti2v_from_generic_wan(self):
        with pytest.raises(ModelConfigError):
            ConfigResolution.resolve(model_name="Wan-AI/Wan2.2-Unknown-14B-Diffusers")

    @pytest.mark.fast
    def test_generic_wan_alias_is_ambiguous(self):
        with pytest.raises(ModelConfigError):
            ConfigResolution.resolve(model_name="wan")

    @pytest.mark.fast
    def test_generic_local_wan_video_folder_does_not_infer_ti2v(self):
        with pytest.raises(ModelConfigError):
            ConfigResolution.resolve(model_name="models/my-wan-video-folder")


class TestConfigResolutionRules:
    @pytest.mark.fast
    def test_exact_match_takes_priority(self):
        # "schnell" is both an exact alias AND would match substring
        config = ConfigResolution.resolve(model_name="schnell")

        # Should return the exact config, not create a new one
        assert config.model_name == "black-forest-labs/FLUX.1-schnell"

    @pytest.mark.fast
    def test_explicit_base_overrides_inference(self):
        # Model name contains "schnell" but explicit base is "dev"
        config = ConfigResolution.resolve(model_name="schnell-style-dev", base_model="dev")

        assert config.base_model == "black-forest-labs/FLUX.1-dev"
