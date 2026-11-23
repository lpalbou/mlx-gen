"""What MiniMax-H3 keeps out of memory: the conditioner, the AdaLN projections and the VAEs until first use."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.weights.loading.streaming_weight_loader import StreamingWeightLoader
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.minimax_h3.minimax_h3_initializer import MiniMaxH3Initializer
from mflux.models.minimax_h3.model.h3_text_encoder.h3_conditioner import H3Conditioner
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_model import IMAGE_TOKEN_ID, Qwen3VLModel
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_text_model import Qwen3VLTextModel
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_vision_model import Qwen3VLVisionModel, preprocess_image
from mflux.models.minimax_h3.model.h3_transformer.h3_block_stream import H3BlockStream
from mflux.models.minimax_h3.model.h3_transformer.h3_modulation_tables import H3ModulationTables
from mflux.models.minimax_h3.model.h3_transformer.h3_transformer import MiniMaxH3Transformer
from mflux.models.minimax_h3.weights.h3_weight_definition import MiniMaxH3WeightDefinition
from tests.minimax_h3.test_h3_pipeline_tiny import _tiny_model


def _tiny_conditioner() -> Qwen3VLModel:
    return Qwen3VLModel(
        language_model=Qwen3VLTextModel(
            vocab_size=151936,
            hidden_size=64,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            intermediate_size=64,
            mrope_section=(2, 1, 1),
        ),
        visual=Qwen3VLVisionModel(
            hidden_size=32,
            depth=2,
            num_heads=4,
            intermediate_size=64,
            num_position_embeddings=16,
            out_hidden_size=64,
            deepstack_visual_indexes=(0, 1),
        ),
    )


def _tiny_transformer(num_layers: int = 2) -> MiniMaxH3Transformer:
    return MiniMaxH3Transformer(
        num_attention_heads=2,
        attention_head_dim=32,
        hidden_size=64,
        num_layers=num_layers,
        num_refiner_layers=1,
        ffn_dim=64,
        text_dim=16,
        freq_dim=32,
        time_embed_hidden_dim=32,
        time_embed_dim=64,
        rope_freq_dim=2,
    )


class _Definition:
    # The two streamed components, as a prepared package lays them out.
    quantization_predicate = staticmethod(MiniMaxH3WeightDefinition.quantization_predicate)

    @staticmethod
    def get_components() -> list[ComponentDefinition]:
        return [
            ComponentDefinition(name="text_encoder", hf_subdir="text_encoder", loading_mode="multi_glob"),
            ComponentDefinition(name="transformer", hf_subdir="transformer", loading_mode="multi_glob"),
        ]


def _quantized(module: nn.Module, bits: int) -> nn.Module:
    def predicate(path: str, candidate):
        weight = getattr(candidate, "weight", None)
        return isinstance(candidate, nn.Linear) and weight is not None and weight.shape[-1] % 64 == 0

    nn.quantize(module, bits=bits, class_predicate=predicate)
    mx.eval(module.parameters())
    return module


@pytest.mark.fast
@pytest.mark.parametrize("bits", [None, 8])
def test_a_streamed_encode_equals_the_conditioner_held_in_memory(tmp_path, bits, monkeypatch):
    monkeypatch.setattr(H3Conditioner, "build", staticmethod(_tiny_conditioner))
    reference = _tiny_conditioner()
    mx.eval(reference.parameters())
    if bits:
        _quantized(reference.language_model, bits)
    ModelSaver._save_weights(str(tmp_path), bits, reference, "text_encoder")
    conditioner = H3Conditioner(tmp_path, _Definition, None)

    token_ids = np.array([11, 12, 13, 14, 15], dtype=np.int32)
    assert mx.array_equal(conditioner.encode(token_ids), reference.encode(token_ids)).item()

    # With a keyframe: the vision tower and its DeepStack features are read too.
    import PIL.Image

    patches, grid = preprocess_image(PIL.Image.new("RGB", (64, 64), (200, 30, 30)))
    image_tokens = (grid[0] * grid[1] * grid[2]) // (reference.visual.spatial_merge_size**2)
    with_image = np.array([7] + [IMAGE_TOKEN_ID] * image_tokens + [9, 10], dtype=np.int32)
    streamed = conditioner.encode(with_image, [patches], [grid])
    assert mx.array_equal(streamed, reference.encode(with_image, [patches], [grid])).item()
    # Each encode reads the weights again, so a second prompt gets the same model.
    assert mx.array_equal(conditioner.encode(token_ids), reference.encode(token_ids)).item()
    assert conditioner.stored_bits() == bits and conditioner.spatial_merge_size == 2


@pytest.mark.fast
def test_released_layers_are_dropped_as_they_run():
    model = _tiny_conditioner()
    mx.eval(model.parameters())
    token_ids = np.array([3, 4, 5], dtype=np.int32)
    expected = model.encode(token_ids)
    assert mx.array_equal(model.encode(token_ids, release_layers=True), expected).item()
    assert model.language_model.layers == [None, None, None]


@pytest.mark.fast
@pytest.mark.parametrize("bits", [None, 8, 6])
def test_modulation_tables_equal_the_projection_they_replace(tmp_path, bits):
    transformer = _tiny_transformer()
    mx.eval(transformer.parameters())
    if bits:
        for block in transformer.transformer_blocks:
            _quantized(block.adaln_proj, bits)
    ModelSaver._save_weights(str(tmp_path), bits, transformer, "transformer")
    component = _Definition.get_components()[1]
    tables = H3ModulationTables(tmp_path / "transformer", component, None)

    # Two steps with different numbers of distinct timesteps, as a real schedule has.
    timesteps = [mx.array([1.0], dtype=mx.float32), mx.array([0.25, 0.5, 0.999], dtype=mx.float32)]
    computed = tables.for_schedule(transformer, timesteps)
    assert len(computed) == 2 and len(computed[0]) == len(transformer.transformer_blocks)
    for step, timestep in enumerate(timesteps):
        embedding = transformer.timestep_embedding(timestep)
        for block_index, block in enumerate(transformer.transformer_blocks):
            for table, expected in zip(computed[step][block_index], block.adaln_proj(embedding)):
                assert table.shape == (3 * timestep.shape[0], 64) and mx.array_equal(table, expected).item()
    # The same schedule is served from memory; another one is computed and the oldest of three is dropped.
    assert tables.for_schedule(transformer, timesteps) is computed
    tables.for_schedule(transformer, [mx.array([0.1], dtype=mx.float32)])
    tables.for_schedule(transformer, [mx.array([0.2], dtype=mx.float32)])
    assert tables.for_schedule(transformer, timesteps) is not computed


@pytest.mark.fast
def test_a_source_snapshot_gets_tables_at_its_load_time_width(tmp_path):
    transformer = _tiny_transformer(num_layers=1)
    mx.eval(transformer.parameters())
    (tmp_path / "transformer").mkdir()
    # A source snapshot: plain safetensors without the package metadata.
    mx.save_safetensors(
        str(tmp_path / "transformer" / "model.safetensors"), dict(tree_flatten(transformer.parameters()))
    )
    component = _Definition.get_components()[1]
    component.precision = mx.bfloat16
    timesteps = [mx.array([0.3, 0.7], dtype=mx.float32)]
    block = transformer.transformer_blocks[0]
    weight, bias = block.adaln_proj.linear.weight.astype(mx.bfloat16), block.adaln_proj.linear.bias.astype(mx.bfloat16)

    block.adaln_proj.linear.weight, block.adaln_proj.linear.bias = weight, bias
    unquantized = H3ModulationTables(tmp_path / "transformer", component, None).for_schedule(transformer, timesteps)
    embedding = transformer.timestep_embedding(timesteps[0])
    for table, expected in zip(unquantized[0][0], block.adaln_proj(embedding)):
        assert mx.array_equal(table, expected).item()

    quantized = H3ModulationTables(tmp_path / "transformer", component, 6).for_schedule(transformer, timesteps)
    _quantized(block.adaln_proj, 6)
    for table, expected in zip(quantized[0][0], block.adaln_proj(embedding)):
        assert mx.array_equal(table, expected).item()


@pytest.mark.fast
def test_a_clip_is_the_same_with_tables_as_with_the_projections_in_memory(tmp_path):
    model = _tiny_model()
    clip = {"seed": 3, "prompt": "A fox", "num_frames": 124, "width": 96, "height": 64, "num_inference_steps": 2}
    reference = model.generate_video(**clip)
    assert reference.extra_metadata["modulation_tables"] is False

    ModelSaver._save_weights(str(tmp_path), None, model.transformer, "transformer")
    model.modulation_tables = H3ModulationTables(tmp_path / "transformer", _Definition.get_components()[1], None)
    for block in model.transformer.transformer_blocks:
        block.adaln_proj = None
    video = model.generate_video(**clip)
    assert video.extra_metadata["modulation_tables"] is True
    assert [frame.tobytes() for frame in video.frames] == [frame.tobytes() for frame in reference.frames]
    assert np.array_equal(video.audio.waveform, reference.audio.waveform)


@pytest.mark.fast
def test_the_loader_skips_tensors_and_leaves_the_rest_unread(tmp_path):
    transformer = _tiny_transformer(num_layers=1)
    mx.eval(transformer.parameters())
    ModelSaver._save_weights(str(tmp_path), None, transformer, "transformer")
    component = _Definition.get_components()[1]

    target = _tiny_transformer(num_layers=1)
    target.transformer_blocks[0].adaln_proj = None
    StreamingWeightLoader.load_into(
        target, tmp_path, component, None, keep=MiniMaxH3Initializer._is_resident_transformer_key, evaluate=False
    )
    loaded = dict(tree_flatten(target.parameters()))
    assert not any(H3ModulationTables.holds(key) for key in loaded)
    assert mx.array_equal(loaded["proj_in.weight"], transformer.proj_in.weight).item()

    # A model that still has the projections refuses a package read without them.
    with pytest.raises(ValueError, match="missing"):
        StreamingWeightLoader.load_into(
            _tiny_transformer(num_layers=1),
            tmp_path,
            component,
            None,
            keep=MiniMaxH3Initializer._is_resident_transformer_key,
        )

    shards = sorted(Path(tmp_path / "transformer").glob("*.safetensors"))
    everything = StreamingWeightLoader.tensor_bytes(shards)
    resident = StreamingWeightLoader.tensor_bytes(shards, MiniMaxH3Initializer._is_resident_transformer_key)
    projection = transformer.transformer_blocks[0].adaln_proj.linear
    assert everything == sum(value.nbytes for _, value in tree_flatten(transformer.parameters()))
    assert everything - resident == projection.weight.nbytes + projection.bias.nbytes


@pytest.mark.fast
def test_a_released_transformer_is_read_again_for_the_next_clip(monkeypatch):
    model = _tiny_model()
    transformer = model.transformer
    reloads = []

    def reload(target) -> None:
        reloads.append(target)
        target.transformer = transformer

    monkeypatch.setattr(MiniMaxH3Initializer, "load_transformer", staticmethod(reload))
    clip = {"seed": 3, "prompt": "A fox", "num_frames": 124, "width": 96, "height": 64, "num_inference_steps": 1}
    first = model.generate_video(**clip, release_transformer=True)
    assert model.transformer is None and first.extra_metadata["transformer_released"] is True and not reloads
    second = model.generate_video(**clip)
    assert reloads == [model] and model.transformer is transformer
    assert [frame.tobytes() for frame in second.frames] == [frame.tobytes() for frame in first.frames]
    # Kept for hosts that call it; the conditioner is never held.
    model.release_text_encoder()


@pytest.mark.fast
@pytest.mark.parametrize("with_adapter", [False, True])
def test_streamed_blocks_give_the_clip_of_the_transformer_held_in_memory(tmp_path, with_adapter):
    from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear

    model = _tiny_model()
    ModelSaver._save_weights(str(tmp_path), None, model.transformer, "transformer")
    block = model.transformer.transformer_blocks[0]
    if with_adapter:
        # An adapter wraps the linear it trains; the stream has to reach the linear underneath.
        block.attn.to_q = LoRALinear.from_linear(block.attn.to_q, r=2, scale=0.5)
        mx.eval(block.attn.to_q.parameters())
    clip = {"seed": 5, "prompt": "A fox", "num_frames": 124, "width": 96, "height": 64, "num_inference_steps": 2}
    reference = model.generate_video(**clip)
    assert reference.extra_metadata["transformer_blocks_streamed"] is False

    component = _Definition.get_components()[1]
    model.modulation_tables = H3ModulationTables(tmp_path / "transformer", component, None)
    model.block_stream = H3BlockStream(
        tmp_path / "transformer", component, MiniMaxH3Initializer._is_resident_transformer_key
    )
    block.adaln_proj = None
    # Every block tensor is replaced by a pending read: nothing the reference run used is left in the block.
    held = {id(value) for _, value in tree_flatten(block.parameters())}
    model.block_stream.rebind(0, block)
    rebound = dict(tree_flatten(block.parameters()))
    adapter_factors = {key for key in rebound if "lora_" in key}
    assert all((id(value) in held) == (key in adapter_factors) for key, value in rebound.items())
    assert bool(adapter_factors) == with_adapter

    video = model.generate_video(**clip, release_transformer=True)
    assert video.extra_metadata["transformer_blocks_streamed"] is True
    # Streaming leaves nothing to free before the decode.
    assert video.extra_metadata["transformer_released"] is False and model.transformer is not None
    assert [frame.tobytes() for frame in video.frames] == [frame.tobytes() for frame in reference.frames]
    assert np.array_equal(video.audio.waveform, reference.audio.waveform)


@pytest.mark.fast
def test_blocks_stream_only_from_weights_stored_as_they_run(tmp_path):
    (tmp_path / "package").mkdir()
    mx.save_safetensors(str(tmp_path / "package" / "0.safetensors"), {"w": mx.zeros((1,))}, {"mflux_version": "test"})
    (tmp_path / "source").mkdir()
    mx.save_safetensors(str(tmp_path / "source" / "model.safetensors"), {"w": mx.zeros((1,))})
    assert H3BlockStream.streams_as_stored(tmp_path / "package", None)
    assert H3BlockStream.streams_as_stored(tmp_path / "package", 8)
    assert H3BlockStream.streams_as_stored(tmp_path / "source", None)
    # A source snapshot quantized at load would be quantized again at every step.
    assert not H3BlockStream.streams_as_stored(tmp_path / "source", 8)


@pytest.mark.fast
def test_weights_not_read_yet_refuse_a_file_that_changed(tmp_path):
    model = _tiny_model()
    ModelSaver._save_weights(str(tmp_path), None, model.transformer, "transformer")
    component = _Definition.get_components()[1]
    stream = H3BlockStream(tmp_path / "transformer", component)
    tables = H3ModulationTables(tmp_path / "transformer", component, None)
    block = model.transformer.transformer_blocks[0]
    stream.rebind(0, block)
    assert stream.bytes_per_step == sum(value.nbytes for _, value in tree_flatten(block.parameters()))

    # A package written again in place: the pending reads would take the new bytes without an error.
    ModelSaver._save_weights(str(tmp_path), None, _tiny_transformer(num_layers=1), "transformer")
    with pytest.raises(RuntimeError, match="changed on disk after the model was loaded"):
        stream.rebind(0, block)
    with pytest.raises(RuntimeError, match="changed on disk after the model was loaded"):
        tables.for_schedule(model.transformer, [mx.array([0.5], dtype=mx.float32)])

    # The VAEs are checked the same way, and at every use: after a first decode the video encoder and, for
    # clips without sound, the audio decoder are still unread. Checked directly: a whole clip would evaluate
    # the block's pending reads against the rewritten file.
    model.decoder_signatures = stream.signatures
    for deferred_weight_bytes in (1, 0):
        model.deferred_weight_bytes = deferred_weight_bytes
        with pytest.raises(RuntimeError, match="changed on disk after the model was loaded"):
            model._require_decoder_files()
    model.decoder_signatures = {}
    model._require_decoder_files()


@pytest.mark.fast
def test_every_use_of_a_vae_checks_its_files(monkeypatch):
    model = _tiny_model()
    checks = []
    monkeypatch.setattr(
        StreamingWeightLoader, "require_unchanged", staticmethod(lambda signatures: checks.append(signatures))
    )
    clip = {"seed": 1, "prompt": "A fox", "num_frames": 124, "width": 96, "height": 64, "num_inference_steps": 1}
    model.generate_video(**clip)
    # The video decode and the audio decode of the first clip, and both again for the second.
    assert len(checks) == 2
    model.generate_video(**clip)
    assert len(checks) == 4
    model.generate_video(**clip, generate_audio=False)
    assert len(checks) == 5


@pytest.mark.fast
def test_a_stream_request_that_cannot_be_met_says_why(tmp_path, capsys):
    from types import SimpleNamespace

    (tmp_path / "transformer").mkdir()
    mx.save_safetensors(str(tmp_path / "transformer" / "model.safetensors"), {"w": mx.zeros((1,))})
    source = SimpleNamespace(root_path=tmp_path, quantize_arg=8, resident_modulation=False)
    assert MiniMaxH3Initializer._can_stream_blocks(source) is False
    assert "quantized again at every step" in capsys.readouterr().err
    adapter = SimpleNamespace(root_path=tmp_path, quantize_arg=None, resident_modulation=True)
    assert MiniMaxH3Initializer._can_stream_blocks(adapter) is False
    assert "an adapter trains the AdaLN projections" in capsys.readouterr().err
    unquantized = SimpleNamespace(root_path=tmp_path, quantize_arg=None, resident_modulation=False)
    assert MiniMaxH3Initializer._can_stream_blocks(unquantized) is True and capsys.readouterr().err == ""


@pytest.mark.fast
def test_adapters_that_train_the_block_projections_are_recognized_in_every_key_layout(tmp_path):
    layouts = {
        "diffusers": "transformer_blocks.0.adaln_proj.linear.lora_A.weight",
        "original": "diffusion_model.blocks.0.adaln_proj.linear.lora_down.weight",
        "kohya": "lora_unet_blocks_0_adaln_proj_linear.lora_down.weight",
    }
    for name, key in layouts.items():
        path = tmp_path / f"{name}.safetensors"
        mx.save_safetensors(str(path), {key: mx.zeros((1, 1))})
        assert MiniMaxH3Initializer.adapter_targets_modulation(str(path)), name
    # The released Turbo adapters train attention and feed-forward only; the output modulation is always held.
    for name, key in {
        "turbo": "transformer_blocks.0.attn.to_q.lora_A.weight",
        "final": "diffusion_model.final_layer.adaln_proj.linear.lora_down.weight",
    }.items():
        path = tmp_path / f"{name}.safetensors"
        mx.save_safetensors(str(path), {key: mx.zeros((1, 1))})
        assert not MiniMaxH3Initializer.adapter_targets_modulation(str(path)), name


@pytest.mark.fast
def test_a_package_is_written_one_component_at_a_time_with_the_shared_layout(tmp_path):
    from types import SimpleNamespace

    transformer = _tiny_transformer(num_layers=1)
    mx.eval(transformer.parameters())
    definition = SimpleNamespace(get_tokenizers=lambda: [], get_components=lambda: [_Definition.get_components()[1]])
    model = SimpleNamespace(
        transformer=transformer, model_config=SimpleNamespace(model_name="tiny", base_model=None, aliases=["tiny"])
    )
    loaded = []

    def load(component):
        loaded.append(component.name)
        return transformer

    ModelSaver.save_model(model, None, str(tmp_path / "whole"), definition)
    ModelSaver.save_components(model, None, str(tmp_path / "streamed"), definition, load)
    assert loaded == ["transformer"]
    whole = sorted((tmp_path / "whole" / "transformer").iterdir())
    streamed = sorted((tmp_path / "streamed" / "transformer").iterdir())
    assert [file.name for file in whole] == [file.name for file in streamed]
    assert all(a.read_bytes() == b.read_bytes() for a, b in zip(whole, streamed))
