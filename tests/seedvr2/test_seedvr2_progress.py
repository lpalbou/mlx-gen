import mlx.core as mx
from PIL import Image

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.seedvr2.variants.upscale import seedvr2 as seedvr2_module
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.models.seedvr2.variants.upscale.seedvr2_util import StreamedVideoChunk
from mflux.utils.video_util import DecodedVideoClip


def _build_stub_seedvr2():
    model = SeedVR2.__new__(SeedVR2)
    model.model_config = ModelConfig.seedvr2_3b()
    model.vae = object()
    model.bits = 8
    model.callbacks = CallbackRegistry()
    model.transformer = lambda txt, vid, timestep: mx.zeros_like(vid[:, :16])
    model._effective_tiling_config = lambda **kwargs: None
    model._assert_video_restore_memory_budget = lambda **kwargs: None
    model._assert_post_chunk_memory_health = lambda **kwargs: None
    model._seedvr2_metadata = lambda: {}
    return model


def _decoded_video_clip(frames):
    return DecodedVideoClip(
        frames=frames,
        fps=12.0,
        source_width=32,
        source_height=32,
        source_frame_count=len(frames),
        source_duration_seconds=len(frames) / 12.0,
        audio_present=False,
        clip_start_frame=0,
        clip_frame_count=len(frames),
    )


def test_seedvr2_image_upscale_progress_uses_image_to_image(monkeypatch, tmp_path):
    model = _build_stub_seedvr2()
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), color=(20, 40, 60)).save(source)

    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "preprocess_image",
        staticmethod(lambda **kwargs: (mx.zeros((1, 3, 32, 32), dtype=mx.float32), 32, 32)),
    )
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "encode",
        staticmethod(lambda **kwargs: mx.zeros((1, 16, 4, 4), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2LatentCreator,
        "create_condition",
        staticmethod(lambda encoded_latent: mx.zeros_like(encoded_latent)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2LatentCreator,
        "create_noise_latents",
        staticmethod(lambda **kwargs: mx.zeros((1, 16, 4, 4), dtype=mx.float32)),
    )
    monkeypatch.setattr(seedvr2_module.SeedVR2TextEmbeddings, "load_positive", staticmethod(lambda: object()))
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "decode",
        staticmethod(lambda **kwargs: mx.zeros((1, 3, 32, 32), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "apply_color_correction",
        staticmethod(lambda decoded, style, mode: decoded),
    )
    monkeypatch.setattr(seedvr2_module.MetadataReader, "read_all_metadata", staticmethod(lambda image_path: None))
    monkeypatch.setattr(seedvr2_module.ImageUtil, "to_image", staticmethod(lambda **kwargs: kwargs))

    all_events = []
    img2img_events = []
    text_events = []
    model.callbacks.subscribe_progress(all_events.append)
    model.callbacks.subscribe_progress(img2img_events.append, task="image-to-image")
    model.callbacks.subscribe_progress(text_events.append, task="text-to-image")

    model.generate_image(seed=1, image_path=source, resolution=512)

    assert [event.phase for event in all_events] == ["start", "denoise", "complete"]
    assert [event.task for event in all_events] == ["image-to-image"] * 3
    assert img2img_events == all_events
    assert text_events == []


def test_seedvr2_restore_video_progress_uses_video_to_video(monkeypatch):
    model = _build_stub_seedvr2()

    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "preprocess_video_frames",
        staticmethod(lambda **kwargs: (mx.zeros((1, 3, 1, 32, 32), dtype=mx.float32), 32, 32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "pad_video_frames",
        staticmethod(lambda video: (video, int(video.shape[2]))),
    )
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "encode",
        staticmethod(lambda **kwargs: mx.zeros((1, 16, 1, 4, 4), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2LatentCreator,
        "create_condition",
        staticmethod(lambda encoded_latent: mx.zeros_like(encoded_latent)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2LatentCreator,
        "create_noise_latents",
        staticmethod(lambda **kwargs: mx.zeros((1, 16, 1, 4, 4), dtype=mx.float32)),
    )
    monkeypatch.setattr(seedvr2_module.SeedVR2TextEmbeddings, "load_positive", staticmethod(lambda: object()))
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "decode",
        staticmethod(lambda **kwargs: mx.zeros((1, 3, 1, 32, 32), dtype=mx.float32)),
    )

    all_events = []
    v2v_events = []
    text_events = []
    model.callbacks.subscribe_progress(all_events.append)
    model.callbacks.subscribe_progress(v2v_events.append, task="video-to-video")
    model.callbacks.subscribe_progress(text_events.append, task="text-to-image")

    model._restore_video_frames(
        seed=1,
        frames=[Image.new("RGB", (32, 32), color=(20, 40, 60))],
        resolution=512,
        softness=0.0,
        color_correction_mode="off",
        enforce_memory_budget=False,
    )

    assert [event.phase for event in all_events] == ["start", "denoise", "complete"]
    assert [event.task for event in all_events] == ["video-to-video"] * 3
    assert v2v_events == all_events
    assert text_events == []


def test_seedvr2_generate_video_completes_after_video_finalize(monkeypatch):
    model = _build_stub_seedvr2()
    clip = _decoded_video_clip([Image.new("RGB", (32, 32), color=(20, 40, 60))])

    monkeypatch.setattr(seedvr2_module.VideoUtil, "read_video_clip", staticmethod(lambda **kwargs: clip))
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "preprocess_video_frames",
        staticmethod(lambda **kwargs: (mx.zeros((1, 3, 1, 32, 32), dtype=mx.float32), 32, 32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "pad_video_frames",
        staticmethod(lambda video: (video, int(video.shape[2]))),
    )
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "encode",
        staticmethod(lambda **kwargs: mx.zeros((1, 16, 1, 4, 4), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2LatentCreator,
        "create_condition",
        staticmethod(lambda encoded_latent: mx.zeros_like(encoded_latent)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2LatentCreator,
        "create_noise_latents",
        staticmethod(lambda **kwargs: mx.zeros((1, 16, 1, 4, 4), dtype=mx.float32)),
    )
    monkeypatch.setattr(seedvr2_module.SeedVR2TextEmbeddings, "load_positive", staticmethod(lambda: object()))
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "decode",
        staticmethod(lambda **kwargs: mx.zeros((1, 3, 1, 32, 32), dtype=mx.float32)),
    )

    finalized = {"done": False}

    def fake_to_video(**kwargs):
        finalized["done"] = True
        return kwargs

    monkeypatch.setattr(seedvr2_module.VideoUtil, "to_video", staticmethod(fake_to_video))

    all_events = []
    complete_checks = []

    def record(event):
        all_events.append(event)
        if event.phase == "complete":
            complete_checks.append(finalized["done"])

    model.callbacks.subscribe_progress(record)

    model.generate_video(seed=1, video_path="input.mp4", resolution=512)

    assert [event.phase for event in all_events] == ["start", "denoise", "complete"]
    assert [event.task for event in all_events] == ["video-to-video"] * 3
    assert complete_checks == [True]


def test_seedvr2_streamed_restore_emits_one_terminal_complete(monkeypatch, tmp_path):
    model = _build_stub_seedvr2()
    probe = _decoded_video_clip([Image.new("RGB", (32, 32), color=(20, 40, 60))])
    chunk_a = _decoded_video_clip([Image.new("RGB", (32, 32), color=(20, 40, 60))])
    chunk_b = _decoded_video_clip([Image.new("RGB", (32, 32), color=(30, 50, 70))])
    chunk_plan = [
        StreamedVideoChunk(0, 1, 0, 1, 1),
        StreamedVideoChunk(1, 2, 0, 1, 1),
    ]

    class FakeNoiseProvider:
        def slice(self, **kwargs):
            return mx.zeros((1, 16, 1, 4, 4), dtype=mx.float32)

        def metadata(self):
            return {}

    class FakeWriter:
        def __init__(self, **kwargs):
            self.file_path = tmp_path / "restored.mp4"
            self.file_path.write_bytes(b"video")

        def write_frames(self, frames):
            return None

        def close(self):
            return self.file_path

        def abort(self):
            return None

    monkeypatch.setattr(seedvr2_module.VideoUtil, "read_video_clip", staticmethod(lambda **kwargs: probe))
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "iter_video_frame_windows",
        staticmethod(lambda *args, **kwargs: iter([chunk_a, chunk_b])),
    )
    monkeypatch.setattr(seedvr2_module.SeedVR2Util, "streamed_video_temporal_quality_error", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(seedvr2_module.SeedVR2Util, "plan_streamed_video_chunks", staticmethod(lambda **kwargs: chunk_plan))
    monkeypatch.setattr(seedvr2_module.SeedVR2Util, "padded_video_frame_count", staticmethod(lambda frame_count: frame_count))
    monkeypatch.setattr(seedvr2_module.VideoStreamWriter, "__init__", FakeWriter.__init__)
    monkeypatch.setattr(seedvr2_module.VideoStreamWriter, "write_frames", FakeWriter.write_frames)
    monkeypatch.setattr(seedvr2_module.VideoStreamWriter, "close", FakeWriter.close)
    monkeypatch.setattr(seedvr2_module.VideoStreamWriter, "abort", FakeWriter.abort)
    monkeypatch.setattr(model, "_build_streamed_video_noise_provider", lambda **kwargs: FakeNoiseProvider())
    monkeypatch.setattr(
        model,
        "_restore_video_frames",
        lambda **kwargs: ([Image.new("RGB", (32, 32), color=(0, 255, 0))], 32, 32, 1),
    )
    monkeypatch.setattr(seedvr2_module.GeneratedVideo, "build_metadata", staticmethod(lambda **kwargs: {}))

    all_events = []
    model.callbacks.subscribe_progress(all_events.append)

    output_path = model.restore_video_to_path(
        seed=1,
        video_path="input.mp4",
        resolution=512,
        output_path=tmp_path / "out.mp4",
        validate_health=False,
    )

    assert output_path.exists()
    assert [event.phase for event in all_events] == ["start", "denoise", "denoise", "complete"]
    assert [event.task for event in all_events] == ["video-to-video"] * 4


def test_seedvr2_streamed_restore_writer_failure_emits_failed(monkeypatch, tmp_path):
    model = _build_stub_seedvr2()
    probe = _decoded_video_clip([Image.new("RGB", (32, 32), color=(20, 40, 60))])
    chunk = _decoded_video_clip([Image.new("RGB", (32, 32), color=(20, 40, 60))])
    chunk_plan = [StreamedVideoChunk(0, 1, 0, 1, 1)]

    class FakeNoiseProvider:
        def slice(self, **kwargs):
            return mx.zeros((1, 16, 1, 4, 4), dtype=mx.float32)

        def metadata(self):
            return {}

    class FailingWriter:
        def __init__(self, **kwargs):
            self.file_path = tmp_path / "restored.mp4"

        def write_frames(self, frames):
            raise RuntimeError("writer failed")

        def abort(self):
            return None

    monkeypatch.setattr(seedvr2_module.VideoUtil, "read_video_clip", staticmethod(lambda **kwargs: probe))
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "iter_video_frame_windows",
        staticmethod(lambda *args, **kwargs: iter([chunk])),
    )
    monkeypatch.setattr(seedvr2_module.SeedVR2Util, "streamed_video_temporal_quality_error", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(seedvr2_module.SeedVR2Util, "plan_streamed_video_chunks", staticmethod(lambda **kwargs: chunk_plan))
    monkeypatch.setattr(seedvr2_module.VideoStreamWriter, "__init__", FailingWriter.__init__)
    monkeypatch.setattr(seedvr2_module.VideoStreamWriter, "write_frames", FailingWriter.write_frames)
    monkeypatch.setattr(seedvr2_module.VideoStreamWriter, "abort", FailingWriter.abort)
    monkeypatch.setattr(model, "_build_streamed_video_noise_provider", lambda **kwargs: FakeNoiseProvider())
    monkeypatch.setattr(
        model,
        "_restore_video_frames",
        lambda **kwargs: ([Image.new("RGB", (32, 32), color=(0, 255, 0))], 32, 32, 1),
    )

    all_events = []
    model.callbacks.subscribe_progress(all_events.append)

    try:
        model.restore_video_to_path(
            seed=1,
            video_path="input.mp4",
            resolution=512,
            output_path=tmp_path / "out.mp4",
            validate_health=False,
        )
    except RuntimeError as exc:
        assert str(exc) == "writer failed"
    else:
        raise AssertionError("expected streamed restore failure")

    assert [event.phase for event in all_events] == ["start", "failed"]
    assert [event.task for event in all_events] == ["video-to-video"] * 2
