import io
import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.callbacks import ProgressEvent
from mflux.cli.runtime_events import CliRuntimeEventStream
from mflux.models.wan.variants.wan2_2_ti2v import Wan2_2_TI2V
from mflux.models.wan.variants.wan_vace import WanVace


def _resolution_model(patch=(1, 4, 4), spatial_scale=8) -> Wan2_2_TI2V:
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.transformer = SimpleNamespace(patch_size=patch)
    model.vae = SimpleNamespace(spatial_scale=spatial_scale)
    return model


def test_wan_i2v_default_canvas_policy_keeps_source_aspect_resolution(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (320, 240), "white").save(image_path)
    model = _resolution_model()

    height, width, metadata = model._resolve_video_spatial_size(
        height=240, width=432, image_path=image_path, video_path=None
    )

    assert (height, width) == (288, 384)
    assert metadata == {"source_width": 320, "source_height": 240, "requested_width": 432, "requested_height": 240}


def test_wan_i2v_exact_resize_honors_requested_canvas_and_prints_resolution(tmp_path, capsys):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (320, 240), "white").save(image_path)
    model = _resolution_model()

    height, width, metadata = model._resolve_video_spatial_size(
        height=240, width=432, image_path=image_path, video_path=None, canvas_policy="exact-resize"
    )

    # The requested canvas is honored after the usual multiple snap (32px here).
    assert (height, width) == (256, 448)
    assert metadata == {"source_width": 320, "source_height": 240, "requested_width": 432, "requested_height": 240}
    output = capsys.readouterr().out
    assert "exact-resize" in output
    assert "(256, 448)" in output


def test_wan_i2v_exact_resize_keeps_multiple_aligned_request_untouched(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (320, 240), "white").save(image_path)
    model = _resolution_model()

    height, width, _ = model._resolve_video_spatial_size(
        height=256, width=448, image_path=image_path, video_path=None, canvas_policy="exact-resize"
    )

    assert (height, width) == (256, 448)


def test_wan_i2v_rejects_unknown_canvas_policy(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (320, 240), "white").save(image_path)
    model = _resolution_model()

    with pytest.raises(ValueError, match="Unsupported canvas policy"):
        model._resolve_video_spatial_size(
            height=240, width=432, image_path=image_path, video_path=None, canvas_policy="stretchy"
        )


def test_wan_t2v_rejects_source_aspect_and_invalid_canvas_policy_loudly():
    # Cycle-3 review regression: the no-source branch used to skip normalization,
    # silently accepting garbage and the contradictory source-aspect request.
    model = _resolution_model()

    with pytest.raises(ValueError, match="Unsupported canvas policy"):
        model._resolve_video_spatial_size(
            height=256, width=448, image_path=None, video_path=None, canvas_policy="banana"
        )
    with pytest.raises(ValueError, match="source-aspect.*requires a source input"):
        model._resolve_video_spatial_size(
            height=256, width=448, image_path=None, video_path=None, canvas_policy="source-aspect"
        )


def test_wan_t2v_accepts_exact_resize_as_redundant_but_consistent():
    # exact-resize names what text-to-video already does; it must not error and must
    # resolve exactly like the default.
    model = _resolution_model()

    default_result = model._resolve_video_spatial_size(height=240, width=432, image_path=None, video_path=None)
    exact_result = model._resolve_video_spatial_size(
        height=240, width=432, image_path=None, video_path=None, canvas_policy="exact-resize"
    )

    assert default_result == exact_result == (256, 448, {})


def test_wan_v2v_default_keeps_requested_canvas_and_source_aspect_derives_from_video(monkeypatch):
    model = _resolution_model()
    monkeypatch.setattr(
        "mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.inspect_video",
        lambda path: SimpleNamespace(source_width=320, source_height=240),
    )

    default_height, default_width, _ = model._resolve_video_spatial_size(
        height=240, width=432, image_path=None, video_path="input.mp4"
    )
    aspect_height, aspect_width, _ = model._resolve_video_spatial_size(
        height=240, width=432, image_path=None, video_path="input.mp4", canvas_policy="source-aspect"
    )

    assert (default_height, default_width) == (256, 448)
    assert (aspect_height, aspect_width) == (288, 384)
    assert aspect_width / aspect_height == pytest.approx(320 / 240)


def test_wan_first_frame_condition_threads_resize_mode_to_pixel_mapping(monkeypatch, tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (128, 64), "white").save(image_path)
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    captured = {}
    model.vae = SimpleNamespace(encode_normalized=lambda pixels: mx.zeros((1, 4, 1, 2, 2), dtype=mx.float32))

    def capture_scale(image, target_width, target_height, resize_mode="resize", fill_color=(0, 0, 0)):
        captured["scale"] = {"resize_mode": resize_mode, "width": target_width, "height": target_height}
        return Image.new("RGB", (target_width, target_height))

    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.ImageUtil.scale_to_dimensions", capture_scale)

    model._load_first_frame_condition(image_path=image_path, height=64, width=96, resize_mode="pad")

    assert captured["scale"] == {"resize_mode": "pad", "width": 96, "height": 64}


def test_wan_condition_caches_key_on_resize_mode(monkeypatch, tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.image_condition_cache = {}
    loads = []

    def fake_load(self, *, image_path, height, width, resize_mode="resize"):
        loads.append(resize_mode)
        return mx.array([float(len(loads))], dtype=mx.float32)

    monkeypatch.setattr(Wan2_2_TI2V, "_load_first_frame_condition", fake_load)

    stretched = model._encode_first_frame_condition(image_path=image_path, height=64, width=96)
    padded = model._encode_first_frame_condition(image_path=image_path, height=64, width=96, resize_mode="pad")
    padded_again = model._encode_first_frame_condition(image_path=image_path, height=64, width=96, resize_mode="pad")

    # Different mapping modes must never share cached tensors; identical requests must.
    assert loads == ["resize", "pad"]
    assert float(stretched.item()) != float(padded.item())
    np.testing.assert_array_equal(np.array(padded), np.array(padded_again))


def test_wan_video_mask_uses_same_mapping_geometry_as_source(tmp_path):
    mask_path = tmp_path / "mask.png"
    Image.new("L", (128, 64), 255).save(mask_path)
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.vae = SimpleNamespace(spatial_scale=8)

    mask = model._prepare_video_mask(mask_path, height=64, width=64, resize_mode="pad")

    mask_grid = np.array(mask)[0, 0, 0]
    assert mask_grid.shape == (8, 8)
    # 128x64 letterboxed into the 8x8 latent grid: content rows 2..5, bars preserved (0).
    assert mask_grid[:2].max() == 0.0
    assert mask_grid[6:].max() == 0.0
    assert mask_grid[2:6].min() == 1.0


def test_wan_ratio_mismatch_warning_fires_only_when_stretching(capsys):
    for resize_mode, expected in (("resize", True), ("crop", False), ("pad", False)):
        Wan2_2_TI2V._warn_source_ratio_mismatch(
            source_width=320,
            source_height=240,
            width=448,
            height=256,
            surface="Wan video-to-video",
            resize_mode=resize_mode,
        )
        output = capsys.readouterr().out
        assert ("stretches source frames" in output) is expected, resize_mode

    # Matching ratios stay silent even when stretching.
    Wan2_2_TI2V._warn_source_ratio_mismatch(
        source_width=320, source_height=240, width=640, height=480, surface="Wan video-to-video", resize_mode="resize"
    )
    assert "stretches" not in capsys.readouterr().out


def test_vace_preprocess_warns_on_source_ratio_mismatch(monkeypatch, capsys):
    model = WanVace.__new__(WanVace)
    frames = [Image.new("RGB", (320, 240), "white") for _ in range(5)]
    monkeypatch.setattr(
        "mflux.models.wan.variants.wan_vace.VideoUtil.read_video_clip",
        lambda path, max_frames, target_fps: SimpleNamespace(
            frames=frames, clip_frame_count=5, source_width=320, source_height=240
        ),
    )

    video, mask = model._preprocess_conditions(
        video_path="input.mp4",
        video_mask_path=None,
        height=48,
        width=80,
        num_frames=5,
        fps=16,
    )

    assert video.shape == (1, 3, 5, 48, 80)
    assert "stretches source frames" in capsys.readouterr().out

    # Aspect-preserving mapping silences the distortion warning.
    model._preprocess_conditions(
        video_path="input.mp4",
        video_mask_path=None,
        height=48,
        width=80,
        num_frames=5,
        fps=16,
        resize_mode="pad",
    )
    assert "stretches source frames" not in capsys.readouterr().out


def test_vace_rejects_source_aspect_canvas_policy():
    model = WanVace.__new__(WanVace)

    with pytest.raises(ValueError, match="exact-resize"):
        model.generate_video(seed=1, prompt="x", canvas_policy="source-aspect")


def test_wan_start_progress_event_carries_resolved_dimensions():
    events = []

    Wan2_2_TI2V._emit_progress(
        events.append,
        phase="start",
        frame=0,
        total_frames=17,
        step=0,
        total_steps=4,
        task="image-to-video",
        width=448,
        height=256,
    )

    assert events == [
        ProgressEvent(
            phase="start",
            frame=0,
            total_frames=17,
            step=0,
            total_steps=4,
            task="image-to-video",
            width=448,
            height=256,
        )
    ]


def test_runtime_event_stream_emits_width_and_height_from_progress_events():
    stream = io.StringIO()
    events = CliRuntimeEventStream(enabled=True, command="mlxgen generate", model="wan", seed=7, stream=stream)

    events.handle_progress(
        ProgressEvent(phase="start", frame=0, total_frames=17, step=0, total_steps=4, width=448, height=256)
    )

    payload = json.loads(stream.getvalue().strip())
    assert payload["phase"] == "start"
    assert payload["width"] == 448
    assert payload["height"] == 256


def _run_wan_cli(monkeypatch, tmp_path, extra_argv):
    import sys

    from mflux.models.wan.cli import wan_generate

    observed = {}
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    class FakeVideo:
        def save(self, **kwargs):
            return tmp_path / "out.mp4"

    class FakeWan:
        def __init__(self, **kwargs):
            pass

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "Wan2_2_TI2V", FakeWan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model", "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "--prompt", "a city timelapse",
            "--width", "128", "--height", "128", "--frames", "5",
            "--steps", "2", "--seed", "123",
            "--image-path", str(image_path),
            "--output", str(tmp_path / "out.mp4"),
            "--no-progress",
            *extra_argv,
        ],
    )
    wan_generate.main()
    return observed["generate"]


def test_wan_cli_defaults_leave_canvas_policy_and_resize_mode_untouched(monkeypatch, tmp_path):
    kwargs = _run_wan_cli(monkeypatch, tmp_path, [])

    # None = per-route default (i2v source-aspect); "resize" = today's stretch mapping.
    assert kwargs["canvas_policy"] is None
    assert kwargs["resize_mode"] == "resize"


def test_wan_cli_forwards_canvas_policy_and_resize_mode(monkeypatch, tmp_path):
    kwargs = _run_wan_cli(monkeypatch, tmp_path, ["--canvas-policy", "exact-resize", "--resize-mode", "pad"])

    assert kwargs["canvas_policy"] == "exact-resize"
    assert kwargs["resize_mode"] == "pad"


def test_wan_cli_replays_canvas_policy_and_resize_mode_from_metadata(monkeypatch, tmp_path):
    metadata_path = tmp_path / "prior.metadata.json"
    metadata_path.write_text(json.dumps({"canvas_policy": "exact-resize", "resize_mode": "crop"}))

    kwargs = _run_wan_cli(monkeypatch, tmp_path, ["--config-from-metadata", str(metadata_path)])

    assert kwargs["canvas_policy"] == "exact-resize"
    assert kwargs["resize_mode"] == "crop"
