import json

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.seedvr2.variants.upscale import seedvr2 as seedvr2_module
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util, StreamedVideoChunk
from mflux.utils.scale_factor import ScaleFactor
from mflux.utils.video_util import AudioCopyResult, DecodedVideoClip, VideoUtil


def _solid_frame(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (16, 16), color)


@pytest.mark.fast
def test_seedvr2_noise_byte_override_is_internal_benchmark_only(monkeypatch):
    monkeypatch.setenv("MFLUX_INTERNAL_SEEDVR2_MAX_GLOBAL_NOISE_BYTES", "0")
    monkeypatch.delenv("MFLUX_INTERNAL_MEMORY_BENCHMARK_MODE", raising=False)
    monkeypatch.delenv("MFLUX_INTERNAL_MEMORY_BENCHMARK_FLAGS", raising=False)

    assert seedvr2_module.SeedVR2StreamedVideoNoiseProvider.max_global_noise_bytes() is None

    monkeypatch.setenv("MFLUX_INTERNAL_MEMORY_BENCHMARK_MODE", "1")
    monkeypatch.setenv("MFLUX_INTERNAL_MEMORY_BENCHMARK_FLAGS", "seedvr2_max_global_noise_bytes")

    assert seedvr2_module.SeedVR2StreamedVideoNoiseProvider.max_global_noise_bytes() == 0


def _iter_fake_chunk_clips(frames: list[Image.Image], windows: list[tuple[int, int]], audio_present: bool):
    frame_count = len(frames)
    for start_frame, end_frame in windows:
        yield DecodedVideoClip(
            frames=frames[start_frame:end_frame],
            fps=12.0,
            source_width=16,
            source_height=16,
            source_frame_count=frame_count,
            source_duration_seconds=frame_count / 12.0,
            audio_present=audio_present,
            clip_start_frame=start_frame,
            clip_frame_count=end_frame - start_frame,
        )


@pytest.mark.fast
def test_seedvr2_streamed_video_chunk_plan_uses_uniform_real_windows_for_77_8_profile():
    chunks = SeedVR2Util.plan_streamed_video_chunks(frame_count=149, chunk_size=77, overlap=8)

    assert chunks == [
        StreamedVideoChunk(
            input_start_frame=0,
            input_end_frame=77,
            trim_leading_context_frames=0,
            output_frame_count=68,
            target_input_frame_count=77,
        ),
        StreamedVideoChunk(
            input_start_frame=60,
            input_end_frame=137,
            trim_leading_context_frames=8,
            output_frame_count=68,
            target_input_frame_count=77,
        ),
        StreamedVideoChunk(
            input_start_frame=72,
            input_end_frame=149,
            trim_leading_context_frames=64,
            output_frame_count=13,
            target_input_frame_count=77,
        ),
    ]


@pytest.mark.fast
def test_seedvr2_wavelet_color_reconstruction_supports_video_tensors():
    content = mx.zeros((1, 3, 2, 8, 8), dtype=mx.float32)
    style = mx.ones((1, 3, 2, 8, 8), dtype=mx.float32)

    corrected = SeedVR2Util.apply_color_correction(content, style, mode="wavelet")

    assert corrected.shape == content.shape


@pytest.mark.fast
def test_seedvr2_resize_and_soften_keeps_native_1x_frames_unchanged():
    frame = Image.new("RGB", (320, 240), (10, 20, 30))

    resized, true_h, true_w = SeedVR2Util._resize_and_soften(
        image=frame,
        resolution=ScaleFactor(1),
        softness=0.0,
    )

    assert resized.size == frame.size
    assert (true_h, true_w) == (240, 320)
    assert np.array_equal(np.asarray(resized), np.asarray(frame))


@pytest.mark.fast
def test_vae_util_preserves_temporal_axis_when_requested():
    class FakeVAE:
        @staticmethod
        def encode(image):
            return mx.zeros((1, 16, 1, 4, 4), dtype=mx.float32)

        @staticmethod
        def decode(latent):
            return mx.zeros((1, 3, 1, 32, 32), dtype=mx.float32)

    encoded = VAEUtil.encode(FakeVAE(), mx.zeros((1, 3, 1, 32, 32), dtype=mx.float32), preserve_temporal_axis=True)
    decoded = VAEUtil.decode(FakeVAE(), mx.zeros((1, 16, 1, 4, 4), dtype=mx.float32), preserve_temporal_axis=True)

    assert encoded.shape == (1, 16, 1, 4, 4)
    assert decoded.shape == (1, 3, 1, 32, 32)


@pytest.mark.fast
def test_seedvr2_restore_video_to_path_records_chunk_metadata(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "restored.mp4"

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = object()

    frames = [_solid_frame(((index * 5) % 256, (index * 5) % 256, (index * 5) % 256)) for index in range(49)]
    chunk_calls: list[int] = []

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=frames[:1],
                fps=12.0,
                source_width=16,
                source_height=16,
                source_frame_count=len(frames),
                source_duration_seconds=len(frames) / 12.0,
                audio_present=True,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "iter_video_frame_windows",
        staticmethod(lambda path, start_frame=0, windows=None: _iter_fake_chunk_clips(frames, windows or [], True)),
    )

    def fake_restore_video_frames(
        self,
        *,
        seed,
        frames,
        resolution,
        softness,
        color_correction_mode,
        enforce_memory_budget=True,
        noise_latents=None,
        emit_progress=True,
    ):
        del self, seed, resolution, softness, color_correction_mode, enforce_memory_budget, noise_latents, emit_progress
        chunk_calls.append(len(frames))
        return list(frames), 16, 16, len(frames)

    monkeypatch.setattr(seedvr2_module.SeedVR2, "_restore_video_frames", fake_restore_video_frames)
    monkeypatch.setattr(seedvr2_module.VideoUtil, "_latents_to_frames", staticmethod(lambda decoded: decoded))
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "copy_source_audio_to_video",
        staticmethod(
            lambda **kwargs: AudioCopyResult(
                audio_present=True,
                audio_copied=True,
                copy_mode="ffmpeg_copy_video_aac_audio",
                reason=None,
            )
        ),
    )

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    file_path = model.restore_video_to_path(
        seed=42,
        video_path=source,
        resolution=256,
        softness=0.0,
        output_path=output,
        export_json_metadata=True,
        temporal_chunk_size=29,
        temporal_chunk_overlap=8,
        color_correction_mode="wavelet",
    )

    assert file_path.exists()
    metadata = json.loads(file_path.with_suffix(".metadata.json").read_text())
    assert metadata["frames"] == 49
    assert metadata["audio_present"] is True
    assert metadata["audio_copied"] is True
    assert metadata["audio_copy_mode"] == "ffmpeg_copy_video_aac_audio"
    assert metadata["audio_copy_reason"] is None
    assert metadata["source_video_fps"] == 12.0
    assert metadata["source_clip_actual_start_seconds"] == 0.0
    assert metadata["video_health"]["file"]["fps"] == pytest.approx(12.0, abs=0.1)
    assert metadata["temporal_chunk_size"] == 29
    assert metadata["temporal_chunk_overlap"] == 8
    assert metadata["temporal_chunk_count"] == 3
    assert metadata["temporal_chunk_plan"] == [
        {
            "input_start_frame": 0,
            "input_end_frame": 29,
            "input_frame_count": 29,
            "target_input_frame_count": 29,
            "trim_leading_context_frames": 0,
            "output_frame_count": 20,
        },
        {
            "input_start_frame": 12,
            "input_end_frame": 41,
            "input_frame_count": 29,
            "target_input_frame_count": 29,
            "trim_leading_context_frames": 8,
            "output_frame_count": 20,
        },
        {
            "input_start_frame": 20,
            "input_end_frame": 49,
            "input_frame_count": 29,
            "target_input_frame_count": 29,
            "trim_leading_context_frames": 20,
            "output_frame_count": 9,
        },
    ]
    assert metadata["color_correction_mode"] == "wavelet"
    assert metadata["processed_chunk_input_frames_total"] == 87
    assert chunk_calls == [29, 29, 29]


@pytest.mark.fast
def test_seedvr2_restore_video_to_path_uses_clip_global_noise_slices(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "restored.mp4"

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = object()

    frames = [_solid_frame(((index * 5) % 256, (index * 5) % 256, (index * 5) % 256)) for index in range(49)]
    seen_noise_frames: list[list[float]] = []

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=frames[:1],
                fps=12.0,
                source_width=16,
                source_height=16,
                source_frame_count=len(frames),
                source_duration_seconds=len(frames) / 12.0,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "iter_video_frame_windows",
        staticmethod(lambda path, start_frame=0, windows=None: _iter_fake_chunk_clips(frames, windows or [], False)),
    )

    def fake_noise_latents(seed, height, width, num_frames=1, batch_size=1, latent_channels=16):
        del seed
        values = mx.arange(num_frames, dtype=mx.float32).reshape(1, 1, num_frames, 1, 1)
        return mx.broadcast_to(values, (batch_size, latent_channels, num_frames, height, width))

    def fake_restore_video_frames(
        self,
        *,
        seed,
        frames,
        resolution,
        softness,
        color_correction_mode,
        enforce_memory_budget=True,
        noise_latents=None,
        emit_progress=True,
    ):
        del self, seed, resolution, softness, color_correction_mode, enforce_memory_budget, emit_progress
        assert noise_latents is not None
        seen_noise_frames.append([float(value) for value in np.array(noise_latents[0, 0, :, 0, 0])])
        return [frames[0]] * len(frames), 16, 16, len(frames)

    monkeypatch.setattr(seedvr2_module.SeedVR2LatentCreator, "create_noise_latents", staticmethod(fake_noise_latents))
    monkeypatch.setattr(seedvr2_module.SeedVR2, "_restore_video_frames", fake_restore_video_frames)
    monkeypatch.setattr(seedvr2_module.VideoUtil, "_latents_to_frames", staticmethod(lambda decoded: decoded))

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    model.restore_video_to_path(
        seed=42,
        video_path=source,
        resolution=256,
        softness=0.0,
        output_path=output,
        export_json_metadata=False,
        temporal_chunk_size=29,
        temporal_chunk_overlap=8,
    )

    assert seen_noise_frames == [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ]


@pytest.mark.fast
def test_seedvr2_restore_video_to_path_rejects_unsafe_tiny_temporal_chunks(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "restored.mp4"

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = object()

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=12.0,
                source_width=16,
                source_height=16,
                source_frame_count=149,
                source_duration_seconds=149 / 12.0,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )

    monkeypatch.setenv("MFLUX_INTERNAL_MEMORY_BENCHMARK_MODE", "1")
    monkeypatch.setenv("MFLUX_INTERNAL_MEMORY_BENCHMARK_FLAGS", "seedvr2_tiny_temporal_chunks")

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_7b())
    with pytest.raises(ValueError, match="refuses temporal chunks smaller"):
        model.restore_video_to_path(
            seed=42,
            video_path=source,
            resolution=256,
            softness=0.0,
            output_path=output,
            export_json_metadata=False,
            temporal_chunk_size=13,
            temporal_chunk_overlap=4,
        )

    assert not output.exists()


@pytest.mark.fast
def test_seedvr2_restore_video_to_path_saved_mp4_keeps_expected_frame_count(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "restored.mp4"

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = object()

    frames = [_solid_frame((index * 20, index * 20, index * 20)) for index in range(13)]
    original_read_video_clip = VideoUtil.read_video_clip

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=frames[:1],
                fps=12.0,
                source_width=16,
                source_height=16,
                source_frame_count=13,
                source_duration_seconds=13 / 12.0,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "iter_video_frame_windows",
        staticmethod(lambda path, start_frame=0, windows=None: _iter_fake_chunk_clips(frames, windows or [], False)),
    )
    monkeypatch.setattr(seedvr2_module.SeedVR2, "_restore_video_frames", lambda self, **kwargs: (list(kwargs["frames"]), 16, 16, len(kwargs["frames"])))
    monkeypatch.setattr(seedvr2_module.VideoUtil, "_latents_to_frames", staticmethod(lambda decoded: decoded))

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    file_path = model.restore_video_to_path(
        seed=42,
        video_path=source,
        resolution=256,
        softness=0.0,
        output_path=output,
        export_json_metadata=False,
        temporal_chunk_size=13,
        temporal_chunk_overlap=0,
    )

    clip = original_read_video_clip(file_path)
    means = [float(np.asarray(frame.convert("L"), dtype=np.float32).mean()) for frame in clip.frames]
    expected = [float(np.asarray(frame.convert("L"), dtype=np.float32).mean()) for frame in frames]

    assert clip.clip_frame_count == 13
    assert clip.fps == pytest.approx(12.0, abs=0.1)
    assert len(means) == len(expected)
    for observed, target in zip(means, expected):
        assert observed == pytest.approx(target, abs=3.0)


@pytest.mark.fast
def test_seedvr2_restore_video_to_path_requires_audio_copy_by_default(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "restored.mp4"

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = object()

    frames = [_solid_frame((index * 20, index * 20, index * 20)) for index in range(13)]

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=frames[:1],
                fps=12.0,
                source_width=16,
                source_height=16,
                source_frame_count=13,
                source_duration_seconds=13 / 12.0,
                audio_present=True,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "iter_video_frame_windows",
        staticmethod(lambda path, start_frame=0, windows=None: _iter_fake_chunk_clips(frames, windows or [], True)),
    )
    monkeypatch.setattr(seedvr2_module.SeedVR2, "_restore_video_frames", lambda self, **kwargs: (list(kwargs["frames"]), 16, 16, len(kwargs["frames"])))
    monkeypatch.setattr(seedvr2_module.VideoUtil, "_latents_to_frames", staticmethod(lambda decoded: decoded))
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "copy_source_audio_to_video",
        staticmethod(
            lambda **kwargs: AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason="ffmpeg_not_found",
            )
        ),
    )

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    all_events = []
    video_events = []

    unsub_all = model.callbacks.subscribe_progress(lambda event: all_events.append((event.phase, event.task)))
    unsub_video = model.callbacks.subscribe_progress(
        lambda event: video_events.append((event.phase, event.task)),
        task="video-to-video",
    )
    with pytest.raises(RuntimeError, match="could not preserve it safely"):
        try:
            model.restore_video_to_path(
                seed=42,
                video_path=source,
                resolution=256,
                softness=0.0,
                output_path=output,
                export_json_metadata=True,
                temporal_chunk_size=13,
                temporal_chunk_overlap=0,
            )
        finally:
            unsub_video()
            unsub_all()

    assert not output.exists()
    assert not output.with_suffix(".metadata.json").exists()
    assert all_events == [
        ("start", "video-to-video"),
        ("denoise", "video-to-video"),
        ("failed", "video-to-video"),
    ]
    assert video_events == all_events


@pytest.mark.fast
def test_seedvr2_restore_video_to_path_allows_drop_audio_opt_out(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "restored.mp4"

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = object()

    frames = [_solid_frame((index * 20, index * 20, index * 20)) for index in range(13)]

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=frames[:1],
                fps=12.0,
                source_width=16,
                source_height=16,
                source_frame_count=13,
                source_duration_seconds=13 / 12.0,
                audio_present=True,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "iter_video_frame_windows",
        staticmethod(lambda path, start_frame=0, windows=None: _iter_fake_chunk_clips(frames, windows or [], True)),
    )
    monkeypatch.setattr(seedvr2_module.SeedVR2, "_restore_video_frames", lambda self, **kwargs: (list(kwargs["frames"]), 16, 16, len(kwargs["frames"])))
    monkeypatch.setattr(seedvr2_module.VideoUtil, "_latents_to_frames", staticmethod(lambda decoded: decoded))
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "copy_source_audio_to_video",
        staticmethod(lambda **kwargs: (_ for _ in ()).throw(AssertionError("audio helper should not run when drop_audio=True"))),
    )

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    file_path = model.restore_video_to_path(
        seed=42,
        video_path=source,
        resolution=256,
        softness=0.0,
        output_path=output,
        export_json_metadata=True,
        temporal_chunk_size=13,
        temporal_chunk_overlap=0,
        drop_audio=True,
    )

    metadata = json.loads(file_path.with_suffix(".metadata.json").read_text())
    assert metadata["audio_present"] is True
    assert metadata["audio_copied"] is False
    assert metadata["audio_copy_mode"] is None
    assert metadata["audio_copy_reason"] == "drop_audio_requested"


@pytest.mark.fast
def test_seedvr2_restore_video_to_path_cleans_temp_file_on_failure(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "restored.mp4"

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = object()

    frames = [_solid_frame(((index * 5) % 256, (index * 5) % 256, (index * 5) % 256)) for index in range(49)]
    call_count = {"count": 0}

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=frames[:1],
                fps=12.0,
                source_width=16,
                source_height=16,
                source_frame_count=len(frames),
                source_duration_seconds=len(frames) / 12.0,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "iter_video_frame_windows",
        staticmethod(lambda path, start_frame=0, windows=None: _iter_fake_chunk_clips(frames, windows or [], False)),
    )

    def fake_restore(self, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 2:
            raise RuntimeError("boom")
        return list(kwargs["frames"]), 16, 16, len(kwargs["frames"])

    monkeypatch.setattr(seedvr2_module.SeedVR2, "_restore_video_frames", fake_restore)
    monkeypatch.setattr(seedvr2_module.VideoUtil, "_latents_to_frames", staticmethod(lambda decoded: decoded))

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    with pytest.raises(RuntimeError, match="boom"):
        model.restore_video_to_path(
            seed=42,
            video_path=source,
            resolution=256,
            softness=0.0,
            output_path=output,
            export_json_metadata=False,
            temporal_chunk_size=29,
            temporal_chunk_overlap=8,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".restored-*")) == []


@pytest.mark.fast
def test_seedvr2_restore_video_to_path_cleans_final_file_on_postwrite_validation_failure(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "restored.mp4"

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = object()

    frames = [_solid_frame((index * 20, index * 20, index * 20)) for index in range(13)]

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=frames[:1],
                fps=12.0,
                source_width=16,
                source_height=16,
                source_frame_count=13,
                source_duration_seconds=13 / 12.0,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "iter_video_frame_windows",
        staticmethod(lambda path, start_frame=0, windows=None: _iter_fake_chunk_clips(frames, windows or [], False)),
    )
    monkeypatch.setattr(seedvr2_module.SeedVR2, "_restore_video_frames", lambda self, **kwargs: (list(kwargs["frames"]), 16, 16, len(kwargs["frames"])))
    monkeypatch.setattr(seedvr2_module.VideoUtil, "_latents_to_frames", staticmethod(lambda decoded: decoded))
    monkeypatch.setattr(seedvr2_module.VideoHealth, "validate_file", staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad video"))))

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    all_events = []
    video_events = []

    unsub_all = model.callbacks.subscribe_progress(lambda event: all_events.append((event.phase, event.task)))
    unsub_video = model.callbacks.subscribe_progress(
        lambda event: video_events.append((event.phase, event.task)),
        task="video-to-video",
    )
    with pytest.raises(RuntimeError, match="bad video"):
        try:
            model.restore_video_to_path(
                seed=42,
                video_path=source,
                resolution=256,
                softness=0.0,
                output_path=output,
                export_json_metadata=True,
                temporal_chunk_size=13,
                temporal_chunk_overlap=0,
            )
        finally:
            unsub_video()
            unsub_all()

    assert not output.exists()
    assert not output.with_suffix(".metadata.json").exists()
    assert all_events == [
        ("start", "video-to-video"),
        ("denoise", "video-to-video"),
        ("failed", "video-to-video"),
    ]
    assert video_events == all_events


@pytest.mark.fast
def test_seedvr2_restore_video_frames_trims_temporal_padding(monkeypatch):
    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = lambda txt, vid, timestep: mx.zeros_like(vid[:, :16])

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "preprocess_video_frames",
        staticmethod(lambda frames, resolution, softness: (mx.zeros((1, 3, 6, 32, 32), dtype=mx.float32), 32, 32)),
    )
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "encode",
        staticmethod(
            lambda vae, image, tiling_config, preserve_temporal_axis=False: mx.zeros(
                (1, 16, image.shape[2], 4, 4), dtype=mx.float32
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "decode",
        staticmethod(
            lambda vae, latent, tiling_config, preserve_temporal_axis=False: mx.zeros(
                (1, 3, latent.shape[2], 32, 32), dtype=mx.float32
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2TextEmbeddings,
        "load_positive",
        staticmethod(lambda: mx.zeros((1, 1, 5120), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "apply_color_correction",
        staticmethod(lambda content, style, mode="lab": content),
    )
    monkeypatch.setattr(seedvr2_module.SeedVR2, "_assert_video_restore_memory_budget", lambda self, **kwargs: None)

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    decoded, _, _, padded_frames = model._restore_video_frames(
        seed=42,
        frames=[_solid_frame((0, 0, 0)) for _ in range(6)],
        resolution=256,
        softness=0.0,
        color_correction_mode="wavelet",
    )

    assert decoded.shape[2] == 6
    assert padded_frames == 9


@pytest.mark.fast
def test_seedvr2_generate_video_rejects_multi_frame_in_memory_restore_before_full_decode(monkeypatch):
    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = object()

    calls: list[int | None] = []

    def fake_read_video_clip(path, start_seconds=0.0, max_frames=None):
        calls.append(max_frames)
        return DecodedVideoClip(
            frames=[_solid_frame((0, 0, 0))],
            fps=12.0,
            source_width=320,
            source_height=240,
            source_frame_count=120,
            source_duration_seconds=10.0,
            audio_present=False,
            clip_start_frame=0,
            clip_frame_count=1,
        )

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(seedvr2_module.VideoUtil, "read_video_clip", staticmethod(fake_read_video_clip))
    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_7b())
    with pytest.raises(ValueError, match="limited to single-frame in-memory use"):
        model.generate_video(
            seed=42,
            video_path="source.mp4",
            resolution=ScaleFactor(1),
            max_frames=40,
        )

    assert calls == [1]


@pytest.mark.fast
def test_seedvr2_restore_video_frames_keeps_single_frame_temporal_axis(monkeypatch):
    captured: dict[str, tuple[int, ...]] = {}

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = lambda txt, vid, timestep: mx.zeros_like(vid[:, :16])

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "preprocess_video_frames",
        staticmethod(lambda frames, resolution, softness: (mx.zeros((1, 3, 1, 32, 32), dtype=mx.float32), 32, 32)),
    )

    def fake_encode(vae, image, tiling_config, preserve_temporal_axis=False):
        captured["encode_input_shape"] = tuple(image.shape)
        captured["encode_preserve_temporal_axis"] = preserve_temporal_axis
        return mx.zeros((1, 16, 1, 4, 4), dtype=mx.float32)

    def fake_decode(vae, latent, tiling_config, preserve_temporal_axis=False):
        captured["decode_input_shape"] = tuple(latent.shape)
        captured["decode_preserve_temporal_axis"] = preserve_temporal_axis
        return mx.zeros((1, 3, 1, 32, 32), dtype=mx.float32)

    monkeypatch.setattr(seedvr2_module.VAEUtil, "encode", staticmethod(fake_encode))
    monkeypatch.setattr(seedvr2_module.VAEUtil, "decode", staticmethod(fake_decode))
    monkeypatch.setattr(
        seedvr2_module.SeedVR2TextEmbeddings,
        "load_positive",
        staticmethod(lambda: mx.zeros((1, 1, 5120), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "apply_color_correction",
        staticmethod(lambda content, style, mode="lab": content),
    )
    monkeypatch.setattr(seedvr2_module.SeedVR2, "_assert_video_restore_memory_budget", lambda self, **kwargs: None)

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    decoded, _, _, padded_frames = model._restore_video_frames(
        seed=42,
        frames=[_solid_frame((0, 0, 0))],
        resolution=256,
        softness=0.0,
        color_correction_mode="wavelet",
    )

    assert captured["encode_input_shape"] == (1, 3, 1, 32, 32)
    assert captured["encode_preserve_temporal_axis"] is True
    assert captured["decode_input_shape"] == (1, 16, 1, 4, 4)
    assert captured["decode_preserve_temporal_axis"] is True
    assert decoded.shape == (1, 3, 1, 32, 32)
    assert padded_frames == 1


@pytest.mark.fast
def test_seedvr2_restore_video_frames_disables_tiled_encode_for_video(monkeypatch):
    captured: dict[str, object] = {}

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = lambda txt, vid, timestep: mx.zeros_like(vid[:, :16])

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "preprocess_video_frames",
        staticmethod(lambda frames, resolution, softness: (mx.zeros((1, 3, 6, 32, 32), dtype=mx.float32), 32, 32)),
    )

    def fake_encode(vae, image, tiling_config, preserve_temporal_axis=False):
        captured["vae_encode_tiled"] = tiling_config.vae_encode_tiled if tiling_config is not None else None
        return mx.zeros((1, 16, image.shape[2], 4, 4), dtype=mx.float32)

    monkeypatch.setattr(seedvr2_module.VAEUtil, "encode", staticmethod(fake_encode))
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "decode",
        staticmethod(
            lambda vae, latent, tiling_config, preserve_temporal_axis=False: mx.zeros(
                (1, 3, latent.shape[2], 32, 32), dtype=mx.float32
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2TextEmbeddings,
        "load_positive",
        staticmethod(lambda: mx.zeros((1, 1, 5120), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "apply_color_correction",
        staticmethod(lambda content, style, mode="lab": content),
    )
    monkeypatch.setattr(seedvr2_module.SeedVR2, "_assert_video_restore_memory_budget", lambda self, **kwargs: None)

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    model._restore_video_frames(
        seed=42,
        frames=[_solid_frame((0, 0, 0)) for _ in range(6)],
        resolution=256,
        softness=0.0,
        color_correction_mode="wavelet",
    )

    assert captured["vae_encode_tiled"] is False


@pytest.mark.fast
def test_seedvr2_restore_video_frames_disables_decode_tiling_for_small_video(monkeypatch):
    captured: dict[str, object] = {}

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = lambda txt, vid, timestep: mx.zeros_like(vid[:, :16])

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "preprocess_video_frames",
        staticmethod(lambda frames, resolution, softness: (mx.zeros((1, 3, 6, 32, 32), dtype=mx.float32), 32, 32)),
    )

    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "encode",
        staticmethod(lambda vae, image, tiling_config, preserve_temporal_axis=False: mx.zeros((1, 16, image.shape[2], 4, 4), dtype=mx.float32)),
    )

    def fake_decode(vae, latent, tiling_config, preserve_temporal_axis=False):
        captured["vae_decode_tiles_per_dim"] = tiling_config.vae_decode_tiles_per_dim if tiling_config is not None else None
        return mx.zeros((1, 3, latent.shape[2], 32, 32), dtype=mx.float32)

    monkeypatch.setattr(seedvr2_module.VAEUtil, "decode", staticmethod(fake_decode))
    monkeypatch.setattr(
        seedvr2_module.SeedVR2TextEmbeddings,
        "load_positive",
        staticmethod(lambda: mx.zeros((1, 1, 5120), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "apply_color_correction",
        staticmethod(lambda content, style, mode="lab": content),
    )

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    model._restore_video_frames(
        seed=42,
        frames=[_solid_frame((0, 0, 0)) for _ in range(6)],
        resolution=256,
        softness=0.0,
        color_correction_mode="wavelet",
    )

    assert captured["vae_decode_tiles_per_dim"] == 0


@pytest.mark.fast
def test_seedvr2_restore_video_frames_keeps_decode_tiling_for_large_video(monkeypatch):
    captured: dict[str, object] = {}

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = lambda txt, vid, timestep: mx.zeros_like(vid[:, :16])

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "preprocess_video_frames",
        staticmethod(
            lambda frames, resolution, softness: (mx.zeros((1, 3, 6, 1536, 1024), dtype=mx.float32), 1536, 1024)
        ),
    )

    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "encode",
        staticmethod(lambda vae, image, tiling_config, preserve_temporal_axis=False: mx.zeros((1, 16, image.shape[2], 4, 4), dtype=mx.float32)),
    )

    def fake_decode(vae, latent, tiling_config, preserve_temporal_axis=False):
        captured["vae_decode_tiles_per_dim"] = tiling_config.vae_decode_tiles_per_dim if tiling_config is not None else None
        return mx.zeros((1, 3, latent.shape[2], 1536, 1024), dtype=mx.float32)

    monkeypatch.setattr(seedvr2_module.VAEUtil, "decode", staticmethod(fake_decode))
    monkeypatch.setattr(
        seedvr2_module.SeedVR2TextEmbeddings,
        "load_positive",
        staticmethod(lambda: mx.zeros((1, 1, 5120), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "apply_color_correction",
        staticmethod(lambda content, style, mode="lab": content),
    )
    monkeypatch.setattr(seedvr2_module.SeedVR2, "_assert_video_restore_memory_budget", lambda self, **kwargs: None)

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    model._restore_video_frames(
        seed=42,
        frames=[_solid_frame((0, 0, 0)) for _ in range(6)],
        resolution=1536,
        softness=0.0,
        color_correction_mode="wavelet",
    )

    assert captured["vae_decode_tiles_per_dim"] == 8


@pytest.mark.fast
def test_video_util_iter_video_frame_windows_streams_overlapping_windows(tmp_path):
    source = tmp_path / "source.mp4"
    frames = [_solid_frame((index * 25, index * 25, index * 25)) for index in range(8)]

    VideoUtil.save_video(
        frames=frames,
        path=source,
        fps=12.0,
        metadata=None,
        export_json_metadata=False,
        overwrite=True,
        validate_health=False,
    )

    windows = list(VideoUtil.iter_video_frame_windows(source, windows=[(0, 5), (3, 8)]))
    means = [
        [float(np.asarray(frame.convert("L"), dtype=np.float32).mean()) for frame in clip.frames]
        for clip in windows
    ]
    expected_windows = [
        [float(np.asarray(frame.convert("L"), dtype=np.float32).mean()) for frame in frames[0:5]],
        [float(np.asarray(frame.convert("L"), dtype=np.float32).mean()) for frame in frames[3:8]],
    ]

    assert [clip.clip_start_frame for clip in windows] == [0, 3]
    assert [clip.clip_frame_count for clip in windows] == [5, 5]
    for observed_window, expected_window in zip(means, expected_windows):
        assert len(observed_window) == len(expected_window)
        for observed, target in zip(observed_window, expected_window):
            assert observed == pytest.approx(target, abs=3.0)
