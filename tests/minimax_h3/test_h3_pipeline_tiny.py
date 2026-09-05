"""End-to-end `MiniMaxH3.generate_video` on tiny random modules: layout, schedulers, both decoders, audio mux."""

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.minimax_h3.model.h3_audio_vae.h3_audio_vae import H3AudioVAE
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_text_model import Qwen3VLTextModel
from mflux.models.minimax_h3.model.h3_transformer.h3_transformer import MiniMaxH3Transformer
from mflux.models.minimax_h3.model.h3_video_vae.h3_video_vae import H3VideoVAE
from mflux.models.minimax_h3.variants.minimax_h3 import MiniMaxH3
from mflux.models.minimax_h3.weights.h3_weight_definition import MiniMaxH3WeightDefinition


class _StubTokenizer:
    def tokenize(self, prompt: str):
        ids = [(ord(c) % 90) + 3 for c in prompt[:40]]
        return SimpleNamespace(input_ids=mx.array([ids], dtype=mx.int32))


def _tiny_model() -> MiniMaxH3:
    model = MiniMaxH3.__new__(MiniMaxH3)
    nn.Module.__init__(model)
    model.model_config = ModelConfig.minimax_h3()
    model.bits = None
    model.prompt_embed_cache = {}
    model.lora_paths, model.lora_scales = [], []
    model.tokenizers = {MiniMaxH3WeightDefinition.TOKENIZER_NAME: _StubTokenizer()}
    model.text_encoder = Qwen3VLTextModel(
        vocab_size=128,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        mrope_section=(2, 1, 1),
    )
    model.transformer = MiniMaxH3Transformer(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=64,
        text_dim=16,
        freq_dim=32,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    model.vae = H3VideoVAE(
        block_out_channels=(32,) * 6,
        layers_per_block=1,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=16,
        latents_mean=[0.0] * 24,
        latents_std=[1.0] * 24,
    )
    model.audio_vae = H3AudioVAE(
        encoder_dim=8,
        encoder_rates=(2, 4, 4, 5, 5),
        latent_dim=8,
        latent_channels=32,
        num_attention_heads=2,
        decoder_dim=128,
        decoder_rates=(5, 5, 2, 2, 2, 2, 2),
        decoder_kernel_sizes=(9, 9, 4, 4, 4, 4, 4),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1,),),
        sampling_rate=32000,
        latents_mean=[0.0] * 32,
        latents_std=[1.0] * 32,
    )
    mx.eval(model.parameters())
    return model


@pytest.mark.fast
def test_generate_video_produces_aligned_frames_and_stereo_audio(tmp_path):
    model = _tiny_model()
    events = []
    video = model.generate_video(
        seed=1,
        prompt="A fox",
        soundscape="wind",
        num_frames=120,
        width=96,
        height=64,
        num_inference_steps=2,
        progress_callback=events.append,
    )
    assert video.num_frames == 124 and video.width == 96 and video.height == 64 and video.fps == 24
    assert video.frames[0].size == (96, 64)
    assert video.audio.channels == 2 and video.audio.sample_rate == 32000
    assert abs(video.audio.duration_seconds - 124 / 24) < 1e-3
    assert [event.phase for event in events] == ["start", "denoising", "denoising", "decode", "complete"]
    assert video.steps == 2 and video.extra_metadata["num_inference_steps"] == 3
    assert video.prompt.startswith("integrated_multimodal_description: A fox")

    out = video.save(tmp_path / "clip.mp4", export_json_metadata=True, validate_health=False)
    metadata = json.loads((tmp_path / "clip.metadata.json").read_text())
    assert out.exists() and metadata["audio_present"] is True and metadata["audio_source"] == "generated"
    assert metadata["audio_muxed"] is True or (tmp_path / "clip.wav").exists()


@pytest.mark.fast
def test_generate_video_rejects_bad_canvas_and_first_frame_for_now():
    model = _tiny_model()
    with pytest.raises(ValueError, match="multiples of 32"):
        model.generate_video(seed=1, prompt="x", width=100, height=64, num_frames=124, num_inference_steps=1)
    with pytest.raises(ValueError, match="5 to 15 seconds"):
        model.generate_video(seed=1, prompt="x", width=96, height=64, num_frames=50, num_inference_steps=1)
    with pytest.raises(NotImplementedError):
        model.generate_video(seed=1, prompt="x", image_path="frame.png")
