from types import SimpleNamespace

import numpy as np

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.wan.variants.wan2_2_ti2v import Wan2_2_TI2V
from mflux.models.wan.wan_text_encoder_loader import WanTextEncoderLoader


def test_ordinary_wan_has_no_bernini_precision_policy():
    assert WanTextEncoderLoader.precision_policy_id(ModelConfig.wan2_2_ti2v_5b()) is None


def test_bernini_precision_policy_invalidates_only_bernini_prompt_cache_key():
    text_inputs = {
        "input_ids": np.array([[1, 2]], dtype=np.int64),
        "attention_mask": np.array([[1, 1]], dtype=np.int64),
    }

    def model(config):
        return SimpleNamespace(
            _prompt_embed_store=SimpleNamespace(enabled=True),
            _prompt_embed_fingerprint="same-encoder",
            root_path=None,
            model_config=config,
        )

    ordinary = Wan2_2_TI2V._prompt_embed_disk_key(
        model(ModelConfig.wan2_2_ti2v_5b()),
        text_inputs=text_inputs,
        max_sequence_length=2,
    )
    bernini = Wan2_2_TI2V._prompt_embed_disk_key(
        model(ModelConfig.bernini_r_1_3b()),
        text_inputs=text_inputs,
        max_sequence_length=2,
    )

    assert ordinary != bernini
    assert WanTextEncoderLoader.precision_policy_id(ModelConfig.bernini_r_1_3b()) == (
        WanTextEncoderLoader.BERNINI_PRECISION_POLICY_ID
    )
