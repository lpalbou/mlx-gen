from pathlib import Path

from mflux.models.wan.wan_text_encoder import WanTextEncoder


class WanTextEncoderLoader:
    BERNINI_PRECISION_POLICY_ID = "bernini-umt5-official-v2"
    # Part of every prompt-embedding cache key, so embeddings encoded another way (0.40.1 and earlier: the whole
    # encoder in bfloat16 on the CPU) never match and are never served for this encoder.
    ENCODE_POLICY_ID = "umt5-float32-block-streamed-v1"

    @staticmethod
    def load(*, text_encoder_path: Path) -> WanTextEncoder:
        return WanTextEncoder(text_encoder_path)

    @classmethod
    def precision_policy_id(cls, model_config) -> str | None:
        if model_config is not None and bool(
            model_config.transformer_overrides.get("supports_bernini_renderer", False)
        ):
            return cls.BERNINI_PRECISION_POLICY_ID
        return None
