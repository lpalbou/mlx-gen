import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.wan.model.wan_transformer.wan_transformer import WanTransformer
from mflux.models.wan.variants.wan_vace import WanVace
from mflux.models.wan.weights.wan_weight_definition import WanWeightDefinition
from mflux.task_inference import get_model_capabilities


def _tiny_vace_transformer() -> WanTransformer:
    return WanTransformer(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=16,
        out_channels=16,
        ffn_dim=32,
        num_layers=4,
        text_dim=16,
        vace_layers=[0, 2],
        vace_in_channels=96,
    )


def test_vace_transformer_forward_shapes_and_guards():
    transformer = _tiny_vace_transformer()
    hidden = mx.random.normal((1, 16, 3, 8, 8))
    control = mx.random.normal((1, 96, 3, 8, 8))
    embeds = mx.random.normal((1, 512, 16))
    timestep = mx.array([500])

    out = transformer(hidden, timestep, embeds, control_hidden_states=control, control_hidden_states_scale=[1.0, 0.5])
    assert out.shape == (1, 16, 3, 8, 8)

    with pytest.raises(ValueError, match="control_hidden_states"):
        transformer(hidden, timestep, embeds)
    with pytest.raises(ValueError, match="control_hidden_states_scale"):
        transformer(hidden, timestep, embeds, control_hidden_states=control, control_hidden_states_scale=[1.0])

    plain = WanTransformer(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=16,
        out_channels=16,
        ffn_dim=32,
        num_layers=2,
        text_dim=16,
    )
    with pytest.raises(ValueError, match="control_hidden_states"):
        plain(hidden, timestep, embeds, control_hidden_states=control)


def test_vace_control_zero_scale_matches_reference_free_backbone():
    # With every hint scaled to zero, the VACE branch must not change the main stream.
    transformer = _tiny_vace_transformer()
    hidden = mx.random.normal((1, 16, 3, 8, 8))
    control = mx.random.normal((1, 96, 3, 8, 8))
    embeds = mx.random.normal((1, 512, 16))
    timestep = mx.array([500])

    zero_scaled = transformer(
        hidden, timestep, embeds, control_hidden_states=control, control_hidden_states_scale=[0.0, 0.0]
    )
    transformer.vace_layers = None  # disable the vace branch entirely
    plain = transformer(hidden, timestep, embeds)
    assert bool(mx.allclose(zero_scaled, plain, atol=1e-5))


def _torch_nearest_exact_indices(source: int, target: int) -> list[int]:
    return [min(int((i + 0.5) * source / target), source - 1) for i in range(target)]


@pytest.mark.parametrize(("frames", "latent_frames"), [(17, 5), (9, 3), (33, 9), (1, 1)])
def test_vace_mask_channels_match_reference_formula(frames, latent_frames):
    # Reproduces the diffusers prepare_masks math (8x8 rearrange + nearest-exact temporal
    # resample) with a random continuous time-varying mask; the port must match bit-exactly.
    rng = np.random.default_rng(7)
    height, width = 32, 48
    mask_np = rng.random((frames, height, width)).astype(np.float32)

    model = WanVace.__new__(WanVace)
    model.vae = SimpleNamespace(spatial_scale=8)
    mask = mx.array(mask_np)[None, None, :, :, :]
    mask = mx.broadcast_to(mask, (1, 3, frames, height, width))

    for num_refs in (0, 1, 2):
        ours = model._prepare_mask_channels(mask=mask, latent_frames=latent_frames, num_reference_images=num_refs)
        assert ours.shape == (1, 64, latent_frames + num_refs, height // 8, width // 8)

        expected = mask_np.reshape(frames, height // 8, 8, width // 8, 8)
        expected = expected.transpose(2, 4, 0, 1, 3).reshape(64, frames, height // 8, width // 8)
        indices = _torch_nearest_exact_indices(frames, latent_frames)
        expected = expected[:, indices]
        if num_refs:
            expected = np.concatenate(
                [np.zeros((64, num_refs, height // 8, width // 8), dtype=np.float32), expected], axis=1
            )
        assert np.array_equal(np.asarray(ours[0]), expected)


def test_vace_masked_region_generate_mode_blanks_reactive_branch():
    # Official VACE inpainting: the editable region is gray-filled (0.0 in [-1,1]) before
    # encoding, so the reactive branch must see all-gray frames, not the source content.
    model = WanVace.__new__(WanVace)
    seen = []

    def fake_encode(pixels):
        seen.append(np.asarray(pixels.astype(mx.float32)))
        return mx.zeros((1, 16, 3, 4, 6))

    model.vae = SimpleNamespace(encode_normalized=fake_encode, spatial_scale=8)
    video = mx.ones((1, 3, 9, 32, 48))  # bright source everywhere
    mask = mx.zeros((1, 3, 9, 32, 48))
    mask[:, :, :, :, 24:] = 1.0  # right half editable

    model._prepare_control_latents(video=video, mask=mask, reference_frames=[], masked_region_mode="generate")
    inactive_input, reactive_input = seen[0], seen[1]
    assert float(np.abs(reactive_input).max()) == 0.0  # all-gray reactive branch
    assert float(np.abs(inactive_input[..., :24]).max()) == 1.0  # background kept
    assert float(np.abs(inactive_input[..., 24:]).max()) == 0.0

    seen.clear()
    model._prepare_control_latents(video=video, mask=mask, reference_frames=[], masked_region_mode="repaint")
    reactive_repaint = seen[1]
    assert float(np.abs(reactive_repaint[..., 24:]).max()) == 1.0  # source kept in editable region


def test_vace_rejects_unknown_masked_region_mode():
    model = WanVace.__new__(WanVace)
    with pytest.raises(ValueError, match="generate.*repaint|repaint.*generate"):
        model.generate_video(seed=1, prompt="x", masked_region_mode="inpaint")


def test_wan_cli_vace_masked_region_flag_and_metadata_replay(tmp_path):
    from mflux.models.wan.cli import wan_generate

    args = wan_generate._parser().parse_args(
        ["--model", "Wan-AI/Wan2.1-VACE-1.3B-diffusers", "--vace-masked-region", "repaint"]
    )
    assert args.vace_masked_region == "repaint"

    metadata_path = tmp_path / "prior.json"
    metadata_path.write_text(json.dumps({"prompt": "replay", "masked_region_mode": "repaint"}))
    args = wan_generate._parser().parse_args(
        ["--model", "Wan-AI/Wan2.1-VACE-1.3B-diffusers", "--config-from-metadata", str(metadata_path)]
    )
    provided = wan_generate._apply_metadata_defaults(args)
    assert args.vace_masked_region == "repaint"
    assert "--vace-masked-region" in provided


def test_vace_reference_prepend_order_puts_last_reference_first(monkeypatch):
    model = WanVace.__new__(WanVace)
    encoded = {}

    def fake_encode(pixels):
        marker = float(np.asarray(pixels.astype(mx.float32)).flat[0])
        encoded[marker] = True
        return mx.full((1, 16, pixels.shape[2] if pixels.ndim == 5 else 1, 4, 6), marker)

    model.vae = SimpleNamespace(encode_normalized=fake_encode, spatial_scale=8)
    video = mx.zeros((1, 3, 9, 32, 48))
    mask = mx.ones_like(video)
    ref_a = mx.full((1, 3, 1, 32, 48), 0.25)
    ref_b = mx.full((1, 3, 1, 32, 48), 0.75)

    control = model._prepare_control_latents(video=video, mask=mask, reference_frames=[ref_a, ref_b])
    frame_markers = [float(np.asarray(control[0, 0, i].astype(mx.float32)).flat[0]) for i in range(3)]
    # Diffusers prepends each reference in turn: the LAST reference ends up as frame 0.
    assert frame_markers[0] == pytest.approx(0.75)
    assert frame_markers[1] == pytest.approx(0.25)


def test_vace_model_config_and_capabilities():
    config = ModelConfig.from_name("wan2.1-vace-1.3b")
    assert config.model_name == "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    overrides = config.transformer_overrides
    assert overrides["supports_vace"] is True
    assert overrides["has_transformer_2"] is False
    assert overrides["vace_layers"] == list(range(0, 30, 2))
    assert overrides["vace_in_channels"] == 96

    caps = get_model_capabilities(model="Wan-AI/Wan2.1-VACE-1.3B-diffusers")
    by_id = {capability.id: capability for capability in caps.capabilities}
    assert by_id["wan.video-video"].supports_video_strength is False
    assert by_id["wan.video-video"].supports_video_mask is True
    assert "wan.text-video" in by_id
    # VACE requires the exact validated canvas (source-aspect is rejected at
    # generate_video), but honors all source-to-canvas resize modes.
    assert by_id["wan.video-video"].canvas_policies == ("exact-resize",)
    assert by_id["wan.video-video"].default_canvas_policy == "exact-resize"
    assert by_id["wan.video-video"].resize_modes == ("resize", "crop", "pad")
    assert by_id["wan.text-video"].canvas_policies == ()
    assert by_id["wan.text-video"].resize_modes == ()


def test_vace_weight_mapping_covers_checkpoint_vace_keys():
    config = ModelConfig.from_name("wan2.1-vace-1.3b")
    definition = WanWeightDefinition(config)
    transformer_component = definition.components()[0]
    mapping = transformer_component.mapping_getter()
    from_patterns = {pattern for target in mapping for pattern in target.from_pattern}
    assert "vace_patch_embedding.weight" in from_patterns
    assert "vace_blocks.0.proj_in.weight" in from_patterns
    assert "vace_blocks.14.proj_out.weight" in from_patterns
    assert "vace_blocks.14.attn2.norm_k.weight" in from_patterns
    assert "vace_blocks.15.proj_out.weight" not in from_patterns


def test_vace_runtime_rejects_unsupported_options():
    model = WanVace.__new__(WanVace)
    with pytest.raises(ValueError, match="guidance_2"):
        model.generate_video(seed=1, prompt="x", guidance_2=3.0)
    with pytest.raises(ValueError, match="video_strength"):
        model.generate_video(seed=1, prompt="x", video_strength=0.7)
    with pytest.raises(ValueError, match="image_path"):
        model.generate_video(seed=1, prompt="x", image_path="frame.png")
    with pytest.raises(ValueError, match="unipc"):
        model.generate_video(seed=1, prompt="x", solver="euler")


def test_wan_cli_rejects_reference_image_on_non_vace_models(monkeypatch, tmp_path, capsys):
    from mflux.models.wan.cli import wan_generate

    reference = tmp_path / "ref.png"
    Image.new("RGB", (8, 8), "white").save(reference)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--prompt",
            "a ship",
            "--reference-image",
            str(reference),
            "--output",
            str(tmp_path / "out.mp4"),
        ],
    )
    with pytest.raises(SystemExit):
        wan_generate.main()
    assert "requires a Wan VACE model" in capsys.readouterr().err


def test_wan_cli_rejects_video_strength_on_vace_model(monkeypatch, tmp_path, capsys):
    from mflux.models.wan.cli import wan_generate

    video = tmp_path / "input.mp4"
    video.write_bytes(b"mp4")
    monkeypatch.setattr(
        "mflux.utils.video_util.VideoUtil.inspect_video",
        staticmethod(lambda path: SimpleNamespace(source_frame_count=81, source_duration_seconds=5.0, fps=16.0)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "--prompt",
            "a ship",
            "--video-path",
            str(video),
            "--video-strength",
            "0.7",
            "--output",
            str(tmp_path / "out.mp4"),
        ],
    )
    with pytest.raises(SystemExit):
        wan_generate.main()
    assert "not supported on Wan VACE" in capsys.readouterr().err


def test_wan_cli_vace_metadata_replay_restores_reference_fields(tmp_path):
    from mflux.models.wan.cli import wan_generate

    metadata_path = tmp_path / "prior.json"
    metadata_path.write_text(
        json.dumps(
            {
                "prompt": "replay",
                "reference_image_paths": ["ref_a.png", "ref_b.png"],
                "conditioning_scale": 0.8,
            }
        )
    )
    args = wan_generate._parser().parse_args(
        [
            "--model",
            "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "--config-from-metadata",
            str(metadata_path),
        ]
    )
    provided = wan_generate._apply_metadata_defaults(args)

    assert args.reference_image_paths == ["ref_a.png", "ref_b.png"]
    assert args.conditioning_scale == 0.8
    assert "--reference-image" in provided
    assert "--conditioning-scale" in provided


def test_python_runtime_selects_wan_vace_for_vace_configs():
    from mflux.python_runtime import resolve_generation_runtime

    runtime_plan = resolve_generation_runtime(model="wan2.1-vace-1.3b", video_count=1, task="video-to-video")
    assert runtime_plan.runtime_id == "wan-vace"
    assert runtime_plan._definition.import_path == "mflux.models.wan.variants.wan_vace.WanVace"

    a14b = resolve_generation_runtime(model="wan2.2-t2v-a14b", video_count=1, task="video-to-video")
    assert a14b._definition.import_path == "mflux.models.wan.variants.wan2_2_ti2v.Wan2_2_TI2V"


def test_plan_rejects_video_strength_for_vace_and_routes_reference_only_to_t2v():
    from mflux.task_inference import TaskInferenceError, resolve_generation_plan

    with pytest.raises(TaskInferenceError, match="video-strength|strength"):
        resolve_generation_plan(
            model="wan2.1-vace-1.3b",
            video_count=1,
            has_video_strength=True,
            task="video-to-video",
        )
    plan = resolve_generation_plan(model="wan2.1-vace-1.3b", task="auto")
    assert plan.capability_id == "wan.text-video"


def test_vace_frame_count_rounding():
    model = WanVace.__new__(WanVace)
    model.vae = SimpleNamespace(temporal_scale=4)
    assert model._validated_vace_frame_count(81) == 81
    assert model._validated_vace_frame_count(18) == 17
    assert model._validated_vace_frame_count(2) == 1
    with pytest.raises(ValueError, match="at least one frame"):
        model._validated_vace_frame_count(0)


def test_vace_preprocess_conditions_no_video_zeros_path():
    model = WanVace.__new__(WanVace)
    video, mask = model._preprocess_conditions(
        video_path=None,
        video_mask_path=None,
        height=48,
        width=80,
        num_frames=9,
        fps=16,
    )
    assert video.shape == (1, 3, 9, 48, 80)
    assert mask.shape == (1, 3, 9, 48, 80)
    assert float(mx.abs(video).max()) == 0.0
    assert float(mask.min()) == 1.0


def test_vace_released_denoiser_raises_clean_error():
    model = WanVace.__new__(WanVace)
    model.transformer = None
    with pytest.raises(ValueError, match="released"):
        model.generate_video(seed=1, prompt="x")


def test_vace_wrong_checkpoint_transformer_raises_actionable_error():
    model = WanVace.__new__(WanVace)
    model.transformer = SimpleNamespace(vace_layers=None)
    with pytest.raises(ValueError, match="wan2.1-vace-1.3b"):
        model.generate_video(seed=1, prompt="x")


def test_wan_cli_rejects_vace_mask_without_video(monkeypatch, tmp_path, capsys):
    from mflux.models.wan.cli import wan_generate

    mask = tmp_path / "mask.png"
    Image.new("L", (8, 8), 255).save(mask)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "--prompt",
            "a ship",
            "--video-mask-path",
            str(mask),
            "--output",
            str(tmp_path / "out.mp4"),
        ],
    )
    with pytest.raises(SystemExit):
        wan_generate.main()
    err = capsys.readouterr().err
    assert "--video-mask-path requires --video-path" in err


def test_vace_generated_metadata_fields():
    extra = WanVace._vace_extra_metadata(
        video_mask_path="mask.png",
        masked_region_mode="generate",
        reference_image_paths=[Path("a.png")],
        conditioning_scale=0.9,
        num_reference_images=1,
    )
    assert extra["vace"] is True
    assert extra["conditioning_scale"] == 0.9
    assert extra["reference_image_paths"] == ["a.png"]
    assert extra["video_mask_path"] == "mask.png"
    assert extra["masked_region_mode"] == "generate"
