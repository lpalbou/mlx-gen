import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.download_policy import DownloadRequiredError
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.seedvr2.cli import seedvr2_upscale
from mflux.models.seedvr2.cli.seedvr2_upscale import (
    _aligned_chunk_overlap,
    _aligned_chunk_size,
    _expand_image_paths,
    _expand_video_paths,
    _plan_seedvr2_video_restore,
    _resolve_seedvr2_model,
    _seedvr2_memory_profile,
)
from mflux.models.seedvr2.seedvr2_initializer import SeedVR2Initializer
from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util
from mflux.utils.scale_factor import ScaleFactor
from mflux.utils.video_util import DecodedVideoClip


def _create_seedvr2_upscale_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Upscale an image using SeedVR2")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_seedvr2_upscale_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def seedvr2_upscale_parser() -> CommandLineParser:
    return _create_seedvr2_upscale_parser()


@pytest.fixture
def seedvr2_upscale_minimal_argv() -> list[str]:
    return ["mflux-upscale-seedvr2", "--image-path", "image.png"]


@pytest.mark.fast
def test_seedvr2_resolution_integer(seedvr2_upscale_parser, seedvr2_upscale_minimal_argv):
    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--resolution", "512"]):
        args = seedvr2_upscale_parser.parse_args()
        assert args.resolution == 512


@pytest.mark.fast
def test_seedvr2_resolution_scale_factor(seedvr2_upscale_parser, seedvr2_upscale_minimal_argv):
    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--resolution", "2x"]):
        args = seedvr2_upscale_parser.parse_args()
        assert isinstance(args.resolution, ScaleFactor)
        assert args.resolution.value == 2


@pytest.mark.fast
def test_seedvr2_resolution_auto(seedvr2_upscale_parser, seedvr2_upscale_minimal_argv):
    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--resolution", "auto"]):
        args = seedvr2_upscale_parser.parse_args()
        assert isinstance(args.resolution, ScaleFactor)
        assert args.resolution.value == 1


@pytest.mark.fast
def test_seedvr2_multiple_images_and_seeds(seedvr2_upscale_parser):
    argv = [
        "mflux-upscale-seedvr2",
        "--image-path",
        "img1.png",
        "img2.png",
        "--seed",
        "42",
        "43",
    ]
    with patch("sys.argv", argv):
        args = seedvr2_upscale_parser.parse_args()
        assert args.image_path == [Path("img1.png"), Path("img2.png")]
        assert args.seed == [42, 43]
        # Verify output pattern is updated for multiple seeds
        assert "{seed}" in args.output


@pytest.mark.fast
def test_seedvr2_multiple_video_seeds_use_video_output_pattern(seedvr2_upscale_parser):
    argv = [
        "mflux-upscale-seedvr2",
        "--video-path",
        "clip.mp4",
        "--seed",
        "42",
        "43",
    ]
    with patch("sys.argv", argv):
        args = seedvr2_upscale_parser.parse_args()
        assert args.video_path == [Path("clip.mp4")]
        assert args.seed == [42, 43]
        assert args.output == "video_seed_{seed}.mp4"


@pytest.mark.fast
def test_seedvr2_multiple_videos_use_input_name_output_pattern(seedvr2_upscale_parser):
    argv = [
        "mflux-upscale-seedvr2",
        "--video-path",
        "clip_a.mp4",
        "clip_b.mp4",
    ]
    with patch("sys.argv", argv):
        args = seedvr2_upscale_parser.parse_args()
        assert args.video_path == [Path("clip_a.mp4"), Path("clip_b.mp4")]
        assert len(args.seed) == 1
        assert isinstance(args.seed[0], int)
        assert args.output == "video_{input_name}.mp4"


@pytest.mark.fast
def test_seedvr2_multiple_videos_and_seeds_use_unique_video_output_pattern(seedvr2_upscale_parser):
    argv = [
        "mflux-upscale-seedvr2",
        "--video-path",
        "clip_a.mp4",
        "clip_b.mp4",
        "--seed",
        "42",
        "43",
    ]
    with patch("sys.argv", argv):
        args = seedvr2_upscale_parser.parse_args()
        assert args.video_path == [Path("clip_a.mp4"), Path("clip_b.mp4")]
        assert args.seed == [42, 43]
        assert args.output == "video_seed_{seed}_{input_name}.mp4"


@pytest.mark.fast
def test_seedvr2_auto_seeds_follow_shared_multi_output_contract(seedvr2_upscale_parser, monkeypatch):
    monkeypatch.setattr("mflux.cli.seed_values.random.sample", lambda population, k: [101, 202])
    argv = [
        "mflux-upscale-seedvr2",
        "--image-path",
        "image.png",
        "--auto-seeds",
        "2",
    ]
    with patch("sys.argv", argv):
        args = seedvr2_upscale_parser.parse_args()
        assert args.seed == [101, 202]
        assert args.output == "image_seed_{seed}.png"


@pytest.mark.fast
def test_seedvr2_duplicate_seeds_are_rejected(seedvr2_upscale_parser):
    argv = [
        "mflux-upscale-seedvr2",
        "--image-path",
        "image.png",
        "--seed",
        "7",
        "7",
    ]
    with patch("sys.argv", argv):
        with pytest.raises(SystemExit):
            seedvr2_upscale_parser.parse_args()


@pytest.mark.fast
def test_seedvr2_video_path_and_bounded_clip_args(seedvr2_upscale_parser):
    argv = [
        "mflux-upscale-seedvr2",
        "--video-path",
        "clip.mp4",
        "--start-seconds",
        "1.25",
        "--max-frames",
        "17",
        "--temporal-chunk-size",
        "49",
        "--temporal-chunk-overlap",
        "16",
        "--color-correction",
        "wavelet",
    ]
    with patch("sys.argv", argv):
        args = seedvr2_upscale_parser.parse_args()
        assert args.video_path == [Path("clip.mp4")]
        assert args.start_seconds == 1.25
        assert args.max_frames == 17
        assert args.drop_audio is False
        assert args.temporal_chunk_size == 49
        assert args.temporal_chunk_overlap == 16
        assert args.color_correction == "wavelet"


@pytest.mark.fast
def test_seedvr2_requires_exactly_one_input_kind(seedvr2_upscale_parser):
    with patch("sys.argv", ["mflux-upscale-seedvr2"]):
        with pytest.raises(SystemExit):
            seedvr2_upscale_parser.parse_args()

    with patch("sys.argv", ["mflux-upscale-seedvr2", "--image-path", "image.png", "--video-path", "clip.mp4"]):
        with pytest.raises(SystemExit):
            seedvr2_upscale_parser.parse_args()


@pytest.mark.fast
def test_seedvr2_quantize_choices(seedvr2_upscale_parser, seedvr2_upscale_minimal_argv):
    # Valid choices
    for q in ["4", "8"]:
        with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--quantize", q]):
            args = seedvr2_upscale_parser.parse_args()
            assert args.quantize == int(q)

    # Invalid choice
    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--quantize", "16"]):
        with pytest.raises(SystemExit):
            seedvr2_upscale_parser.parse_args()


@pytest.mark.fast
def test_seedvr2_model_arg(seedvr2_upscale_parser, seedvr2_upscale_minimal_argv):
    # Test with --model
    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--model", "some/path"]):
        args = seedvr2_upscale_parser.parse_args()
        assert args.model == "some/path"
        assert args.model_path == "some/path"

    # Test with -m
    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["-m", "another/path"]):
        args = seedvr2_upscale_parser.parse_args()
        assert args.model == "another/path"
        assert args.model_path == "another/path"


@pytest.mark.fast
def test_seedvr2_softness(seedvr2_upscale_parser, seedvr2_upscale_minimal_argv):
    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--softness", "0.5"]):
        args = seedvr2_upscale_parser.parse_args()
        assert args.softness == 0.5


@pytest.mark.fast
def test_seedvr2_vae_tiling_flag(seedvr2_upscale_parser, seedvr2_upscale_minimal_argv):
    with patch("sys.argv", seedvr2_upscale_minimal_argv):
        args = seedvr2_upscale_parser.parse_args()
        assert args.vae_tiling is False

    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--vae-tiling"]):
        args = seedvr2_upscale_parser.parse_args()
        assert args.vae_tiling is True


@pytest.mark.fast
def test_seedvr2_main_passes_metadata_flag_to_save(monkeypatch, tmp_path):
    image_path = tmp_path / "source.png"
    image_path.touch()
    output_path = tmp_path / "upscaled.png"
    saved: dict[str, object] = {}

    class FakeResult:
        def save(self, path, export_json_metadata=False, overwrite=True, embed_metadata=False):
            saved["path"] = path
            saved["export_json_metadata"] = export_json_metadata
            saved["overwrite"] = overwrite

    class FakeSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            saved["quantize"] = quantize
            saved["model_path"] = model_path
            saved["model_config"] = model_config.model_name
            self.model_config = model_config
            self.tiling_config = None

        def generate_image(self, *, seed, image_path, resolution, softness, color_correction_mode):
            saved["seed"] = seed
            saved["image_path"] = image_path
            saved["resolution"] = resolution
            saved["softness"] = softness
            saved["color_correction_mode"] = color_correction_mode
            return FakeResult()

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", FakeSeedVR2)
    monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--image-path",
            str(image_path),
            "--resolution",
            "2x",
            "--seed",
            "123",
            "--metadata",
            "--output",
            str(output_path),
        ],
    )

    seedvr2_upscale.main()

    assert saved["path"] == output_path
    assert saved["export_json_metadata"] is True
    assert saved["overwrite"] is True
    assert saved["color_correction_mode"] == "wavelet"


@pytest.mark.fast
def test_seedvr2_main_routes_video_inputs_to_restore_video_to_path(monkeypatch, tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.touch()
    output_path = tmp_path / "restored.mp4"
    saved: dict[str, object] = {}

    class FakeSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            self.model_config = model_config
            self.tiling_config = None

        def restore_video_to_path(
            self,
            *,
            seed,
            video_path,
            resolution,
            softness,
            start_seconds,
            max_frames,
            output_path,
            export_json_metadata,
            overwrite,
            temporal_chunk_size,
            temporal_chunk_overlap,
            color_correction_mode,
            drop_audio,
            restore_metadata,
            enforce_memory_budget,
            validate_health=True,
        ):
            saved["seed"] = seed
            saved["video_path"] = video_path
            saved["resolution"] = resolution
            saved["softness"] = softness
            saved["start_seconds"] = start_seconds
            saved["max_frames"] = max_frames
            saved["path"] = output_path
            saved["export_json_metadata"] = export_json_metadata
            saved["overwrite"] = overwrite
            saved["temporal_chunk_size"] = temporal_chunk_size
            saved["temporal_chunk_overlap"] = temporal_chunk_overlap
            saved["color_correction_mode"] = color_correction_mode
            saved["drop_audio"] = drop_audio
            saved["restore_metadata"] = restore_metadata
            saved["validate_health"] = validate_health
            return output_path

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", FakeSeedVR2)
    monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=64,
                source_height=48,
                source_frame_count=200,
                source_duration_seconds=4.004,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_path),
            "--start-seconds",
            "1.5",
            "--max-frames",
            "130",
            "--temporal-chunk-size",
            "49",
            "--temporal-chunk-overlap",
            "16",
            "--color-correction",
            "off",
            "--metadata",
            "--output",
            str(output_path),
        ],
    )

    seedvr2_upscale.main()

    assert saved["path"] == output_path
    assert saved["video_path"] == video_path
    assert saved["start_seconds"] == 1.5
    assert saved["max_frames"] == 130
    assert saved["drop_audio"] is False
    assert saved["temporal_chunk_size"] == 49
    assert saved["temporal_chunk_overlap"] == 16
    assert saved["color_correction_mode"] == "off"
    # The post-save health re-decode stays ON by default (0087).
    assert saved["validate_health"] is True


@pytest.mark.fast
def test_seedvr2_main_routes_small_video_inputs_to_restore_video_to_path(monkeypatch, tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.touch()
    output_path = tmp_path / "restored.mp4"
    saved: dict[str, object] = {}

    class FakeSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            self.model_config = model_config
            self.tiling_config = None

        def restore_video_to_path(
            self,
            *,
            seed,
            video_path,
            resolution,
            softness,
            start_seconds,
            max_frames,
            output_path,
            export_json_metadata,
            overwrite,
            temporal_chunk_size,
            temporal_chunk_overlap,
            color_correction_mode,
            drop_audio,
            restore_metadata,
            enforce_memory_budget,
            validate_health=True,
        ):
            saved["seed"] = seed
            saved["video_path"] = video_path
            saved["resolution"] = resolution
            saved["softness"] = softness
            saved["start_seconds"] = start_seconds
            saved["max_frames"] = max_frames
            saved["color_correction_mode"] = color_correction_mode
            saved["path"] = output_path
            saved["export_json_metadata"] = export_json_metadata
            saved["overwrite"] = overwrite
            saved["temporal_chunk_size"] = temporal_chunk_size
            saved["temporal_chunk_overlap"] = temporal_chunk_overlap
            saved["drop_audio"] = drop_audio
            saved["restore_metadata"] = restore_metadata
            return output_path

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", FakeSeedVR2)
    monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=64,
                source_height=48,
                source_frame_count=100,
                source_duration_seconds=4.004,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_path),
            "--max-frames",
            "100",
            "--color-correction",
            "wavelet",
            "--metadata",
            "--output",
            str(output_path),
        ],
    )

    seedvr2_upscale.main()

    assert saved["video_path"] == video_path
    assert saved["max_frames"] == 100
    assert saved["color_correction_mode"] == "wavelet"
    assert saved["path"] == output_path
    assert str(saved["resolution"]) == "1x"
    assert saved["drop_audio"] is False
    assert saved["restore_metadata"]["restore_mode"] == "streaming"


@pytest.mark.fast
def test_seedvr2_main_routes_drop_audio_opt_out(monkeypatch, tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.touch()
    output_path = tmp_path / "restored.mp4"
    saved: dict[str, object] = {}

    class FakeSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            self.model_config = model_config
            self.tiling_config = None

        def restore_video_to_path(
            self,
            *,
            seed,
            video_path,
            resolution,
            softness,
            start_seconds,
            max_frames,
            output_path,
            export_json_metadata,
            overwrite,
            temporal_chunk_size,
            temporal_chunk_overlap,
            color_correction_mode,
            drop_audio,
            restore_metadata,
            enforce_memory_budget,
            validate_health=True,
        ):
            saved["drop_audio"] = drop_audio
            saved["restore_metadata"] = restore_metadata
            return output_path

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", FakeSeedVR2)
    monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=25.0,
                source_width=64,
                source_height=48,
                source_frame_count=250,
                source_duration_seconds=10.0,
                audio_present=True,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_path),
            "--drop-audio",
            "--output",
            str(output_path),
        ],
    )

    seedvr2_upscale.main()

    assert saved["drop_audio"] is True
    assert saved["restore_metadata"]["drop_audio_requested"] is True


@pytest.mark.fast
def test_seedvr2_main_disables_runtime_budget_enforcement_for_unsafe_video_override(monkeypatch, tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.touch()
    output_path = tmp_path / "restored.mp4"
    saved: dict[str, object] = {}

    class FakeSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            self.model_config = model_config
            self.tiling_config = None

        def restore_video_to_path(
            self,
            *,
            seed,
            video_path,
            resolution,
            softness,
            start_seconds,
            max_frames,
            output_path,
            export_json_metadata,
            overwrite,
            temporal_chunk_size,
            temporal_chunk_overlap,
            color_correction_mode,
            drop_audio,
            restore_metadata,
            enforce_memory_budget,
            validate_health=True,
        ):
            saved["enforce_memory_budget"] = enforce_memory_budget
            saved["temporal_chunk_size"] = temporal_chunk_size
            saved["temporal_chunk_overlap"] = temporal_chunk_overlap
            saved["drop_audio"] = drop_audio
            return output_path

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", FakeSeedVR2)
    monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=64,
                source_height=48,
                source_frame_count=14,
                source_duration_seconds=14 / 29.97,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_path),
            "--max-frames",
            "14",
            "--temporal-chunk-size",
            "49",
            "--temporal-chunk-overlap",
            "16",
            "--force-unsafe-video-memory",
            "--output",
            str(output_path),
        ],
    )

    seedvr2_upscale.main()

    assert saved["enforce_memory_budget"] is False
    assert saved["drop_audio"] is False
    assert saved["temporal_chunk_size"] == 17
    assert saved["temporal_chunk_overlap"] == 0


@pytest.mark.fast
def test_seedvr2_main_preserves_zero_effective_overlap(monkeypatch, tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.touch()
    output_path = tmp_path / "restored.mp4"
    saved: dict[str, object] = {}

    class FakeSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            self.model_config = model_config
            self.tiling_config = None

        def restore_video_to_path(
            self,
            *,
            seed,
            video_path,
            resolution,
            softness,
            start_seconds,
            max_frames,
            output_path,
            export_json_metadata,
            overwrite,
            temporal_chunk_size,
            temporal_chunk_overlap,
            color_correction_mode,
            drop_audio,
            restore_metadata,
            enforce_memory_budget,
            validate_health=True,
        ):
            saved["temporal_chunk_size"] = temporal_chunk_size
            saved["temporal_chunk_overlap"] = temporal_chunk_overlap
            saved["drop_audio"] = drop_audio
            return output_path

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", FakeSeedVR2)
    monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=64,
                source_height=48,
                source_frame_count=13,
                source_duration_seconds=13 / 29.97,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_path),
            "--max-frames",
            "13",
            "--force-unsafe-video-memory",
            "--output",
            str(output_path),
        ],
    )

    seedvr2_upscale.main()

    assert saved["temporal_chunk_size"] == 13
    assert saved["temporal_chunk_overlap"] == 0
    assert saved["drop_audio"] is False


@pytest.mark.fast
def test_seedvr2_main_rejects_tiny_unsafe_streaming_chunk_profiles(monkeypatch, tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.touch()

    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=64,
                source_height=48,
                source_frame_count=400,
                source_duration_seconds=400 / 29.97,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_path),
            "--temporal-chunk-size",
            "5",
            "--temporal-chunk-overlap",
            "1",
            "--force-unsafe-video-memory",
        ],
    )
    monkeypatch.setenv("MFLUX_INTERNAL_MEMORY_BENCHMARK_MODE", "1")
    monkeypatch.setenv("MFLUX_INTERNAL_MEMORY_BENCHMARK_FLAGS", "seedvr2_tiny_temporal_chunks")

    with pytest.raises(SystemExit):
        seedvr2_upscale.main()


@pytest.mark.fast
def test_seedvr2_main_rejects_unsafe_13_frame_streaming_profile(monkeypatch, tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.touch()

    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=64,
                source_height=48,
                source_frame_count=149,
                source_duration_seconds=149 / 29.97,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_path),
            "--temporal-chunk-size",
            "13",
            "--temporal-chunk-overlap",
            "4",
            "--force-unsafe-video-memory",
        ],
    )

    with pytest.raises(SystemExit):
        seedvr2_upscale.main()


@pytest.mark.fast
def test_seedvr2_main_rejects_unaligned_explicit_streaming_chunk_size(monkeypatch, tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.touch()

    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=64,
                source_height=48,
                source_frame_count=149,
                source_duration_seconds=149 / 29.97,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_path),
            "--temporal-chunk-size",
            "48",
            "--temporal-chunk-overlap",
            "8",
            "--force-unsafe-video-memory",
        ],
    )

    with pytest.raises(SystemExit):
        seedvr2_upscale.main()


@pytest.mark.fast
def test_seedvr2_main_preserves_explicit_streaming_overlap(monkeypatch, tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.touch()
    output_path = tmp_path / "restored.mp4"
    saved: dict[str, object] = {}

    class FakeSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            self.model_config = model_config
            self.tiling_config = None

        def restore_video_to_path(
            self,
            *,
            seed,
            video_path,
            resolution,
            softness,
            start_seconds,
            max_frames,
            output_path,
            export_json_metadata,
            overwrite,
            temporal_chunk_size,
            temporal_chunk_overlap,
            color_correction_mode,
            drop_audio,
            restore_metadata,
            enforce_memory_budget,
            validate_health=True,
        ):
            saved["temporal_chunk_size"] = temporal_chunk_size
            saved["temporal_chunk_overlap"] = temporal_chunk_overlap
            return output_path

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", FakeSeedVR2)
    monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=64,
                source_height=48,
                source_frame_count=149,
                source_duration_seconds=149 / 29.97,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_path),
            "--temporal-chunk-size",
            "29",
            "--temporal-chunk-overlap",
            "16",
            "--force-unsafe-video-memory",
            "--output",
            str(output_path),
        ],
    )

    seedvr2_upscale.main()

    assert saved["temporal_chunk_size"] == 29
    assert saved["temporal_chunk_overlap"] == 16


@pytest.mark.fast
def test_seedvr2_main_rejects_vae_tiling_for_video_input(monkeypatch, tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_path),
            "--vae-tiling",
        ],
    )

    with pytest.raises(SystemExit):
        seedvr2_upscale.main()


@pytest.mark.fast
def test_seedvr2_main_keeps_transformer_in_low_ram_for_video_input(monkeypatch, tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.touch()
    output_path = tmp_path / "restored.mp4"
    saver = type("Saver", (), {"keep_transformer": False, "memory_stats": lambda self: "mem"})()

    class FakeSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            self.model_config = model_config
            self.tiling_config = None

        def restore_video_to_path(
            self,
            *,
            seed,
            video_path,
            resolution,
            softness,
            start_seconds,
            max_frames,
            output_path,
            export_json_metadata,
            overwrite,
            temporal_chunk_size,
            temporal_chunk_overlap,
            color_correction_mode,
            drop_audio,
            restore_metadata,
            enforce_memory_budget,
            validate_health=True,
        ):
            assert drop_audio is False
            return output_path

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", FakeSeedVR2)
    monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", lambda **kwargs: saver)
    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=64,
                source_height=48,
                source_frame_count=100,
                source_duration_seconds=4.004,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_path),
            "--max-frames",
            "100",
            "--low-ram",
            "--output",
            str(output_path),
        ],
    )

    seedvr2_upscale.main()

    assert saver.keep_transformer is True


@pytest.mark.fast
def test_seedvr2_main_writes_failure_manifest_for_video_errors(monkeypatch, tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.touch()
    output_path = tmp_path / "restored.mp4"

    class FakeSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            self.model_config = model_config
            self.tiling_config = None

        def restore_video_to_path(
            self,
            *,
            seed,
            video_path,
            resolution,
            softness,
            start_seconds,
            max_frames,
            output_path,
            export_json_metadata,
            overwrite,
            temporal_chunk_size,
            temporal_chunk_overlap,
            color_correction_mode,
            drop_audio,
            restore_metadata,
            enforce_memory_budget,
            validate_health=True,
        ):
            raise RuntimeError("boom")

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", FakeSeedVR2)
    monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=64,
                source_height=48,
                source_frame_count=14,
                source_duration_seconds=14 / 29.97,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_path),
            "--max-frames",
            "14",
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(RuntimeError, match="boom"):
        seedvr2_upscale.main()

    failure = json.loads(output_path.with_suffix(".failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["run"]["restore_plan"]["restore_mode"] == "streaming"


@pytest.mark.fast
def test_seedvr2_main_rejects_bounded_clip_args_for_image_input(monkeypatch, tmp_path):
    image_path = tmp_path / "source.png"
    image_path.touch()

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("SeedVR2 should not be constructed when parser validation should fail")

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", fail_if_loaded)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--image-path",
            str(image_path),
            "--max-frames",
            "5",
        ],
    )

    with pytest.raises(SystemExit):
        seedvr2_upscale.main()


@pytest.mark.fast
def test_seedvr2_main_enables_vae_tiling_when_requested(monkeypatch, tmp_path):
    image_path = tmp_path / "source.png"
    image_path.touch()
    output_path = tmp_path / "upscaled.png"
    saved: dict[str, object] = {}

    class FakeResult:
        def save(self, path, export_json_metadata=False, overwrite=True, embed_metadata=False):
            pass

    class FakeSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            self.model_config = model_config
            self.tiling_config = None
            saved["model"] = self

        def generate_image(self, *, seed, image_path, resolution, softness, color_correction_mode):
            saved["tiling_config"] = self.tiling_config
            saved["color_correction_mode"] = color_correction_mode
            return FakeResult()

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", FakeSeedVR2)
    monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--image-path",
            str(image_path),
            "--vae-tiling",
            "--output",
            str(output_path),
        ],
    )

    seedvr2_upscale.main()

    assert saved["tiling_config"].vae_encode_tiled is True
    assert saved["tiling_config"].vae_encode_tile_size == 768
    assert saved["tiling_config"].vae_encode_tile_overlap == 128
    assert saved["tiling_config"].vae_decode_tiles_per_dim == 8
    assert saved["color_correction_mode"] == "wavelet"


@pytest.mark.fast
def test_seedvr2_image_low_ram_preserves_default_tiling_without_flag(monkeypatch, tmp_path):
    image_path = tmp_path / "source.png"
    image_path.touch()
    output_path = tmp_path / "upscaled.png"
    saved: dict[str, object] = {}

    class FakeResult:
        def save(self, path, export_json_metadata=False, overwrite=True, embed_metadata=False):
            pass

    class FakeSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            self.model_config = model_config
            self.tiling_config = TilingConfig(vae_encode_tiled=False, vae_decode_tiles_per_dim=0)

        def generate_image(self, *, seed, image_path, resolution, softness, color_correction_mode):
            saved["tiling_config"] = self.tiling_config
            return FakeResult()

    def fake_register_callbacks(*, args, model, latent_creator, **kwargs):
        saved["callback_low_ram"] = args.low_ram
        model.tiling_config = TilingConfig()
        return None

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", FakeSeedVR2)
    monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", fake_register_callbacks)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--image-path",
            str(image_path),
            "--low-ram",
            "--output",
            str(output_path),
        ],
    )

    seedvr2_upscale.main()

    assert saved["tiling_config"].vae_encode_tiled is False
    assert saved["tiling_config"].vae_decode_tiles_per_dim == 0
    assert saved["callback_low_ram"] is False


@pytest.mark.fast
def test_seedvr2_main_rejects_unknown_hf_handle_before_loading(monkeypatch, tmp_path, capsys):
    image_path = tmp_path / "source.png"
    image_path.touch()

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("SeedVR2 should not be constructed for an unsupported model handle")

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", fail_if_loaded)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--model",
            "AbstractFramework/not-seedvr2",
            "--image-path",
            str(image_path),
        ],
    )

    with pytest.raises(SystemExit):
        seedvr2_upscale.main()

    assert "Unsupported SeedVR2 model handle" in capsys.readouterr().err


@pytest.mark.fast
def test_seedvr2_main_reports_missing_package_without_traceback(monkeypatch, tmp_path, capsys):
    image_path = tmp_path / "source.png"
    image_path.touch()

    class MissingPackageSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            raise DownloadRequiredError("AbstractFramework/seedvr2-7b-8bit")

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", MissingPackageSeedVR2)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--model",
            "AbstractFramework/seedvr2-7b-8bit",
            "--image-path",
            str(image_path),
        ],
    )

    with pytest.raises(SystemExit):
        seedvr2_upscale.main()

    error_output = capsys.readouterr().err
    assert "MLX-Gen will not download model files during generation" in error_output
    assert "mlxgen download --model AbstractFramework/seedvr2-7b-8bit" in error_output
    assert "Traceback" not in error_output


@pytest.mark.fast
def test_seedvr2_expands_directory_image_paths(tmp_path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.JPG"
    not_image = tmp_path / "note.txt"
    subdir = tmp_path / "subdir"

    image_a.touch()
    image_b.touch()
    not_image.touch()
    subdir.mkdir()
    (subdir / "c.png").touch()

    assert _expand_image_paths([tmp_path]) == [image_a, image_b]


@pytest.mark.fast
def test_seedvr2_expands_directories_and_files_in_order(tmp_path):
    dir_images = tmp_path / "images"
    dir_images.mkdir()
    dir_image = dir_images / "dir.png"
    dir_image.touch()

    file_image = tmp_path / "file.webp"
    file_image.touch()

    assert _expand_image_paths([dir_images, file_image]) == [dir_image, file_image]


@pytest.mark.fast
def test_seedvr2_expands_directory_video_paths(tmp_path):
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.MOV"
    not_video = tmp_path / "note.txt"

    video_a.touch()
    video_b.touch()
    not_video.touch()

    assert _expand_video_paths([tmp_path]) == [video_a, video_b]


@pytest.mark.fast
def test_seedvr2_video_plan_rejects_safe_enlargement_profile():
    source_probe = DecodedVideoClip(
        frames=[],
        fps=29.97,
        source_width=320,
        source_height=240,
        source_frame_count=130,
        source_duration_seconds=4.337,
        audio_present=False,
        clip_start_frame=0,
        clip_frame_count=1,
    )

    plan = _plan_seedvr2_video_restore(
        model_config=ModelConfig.seedvr2_3b(),
        source_probe=source_probe,
        requested_frames=130,
        resolution=ScaleFactor(2),
        temporal_chunk_size=49,
        temporal_chunk_overlap=16,
        chunk_size_was_explicit=True,
        chunk_overlap_was_explicit=True,
        low_ram_requested=True,
        cache_limit_gb=8.0,
        force_unsafe_memory_profile=False,
    )

    assert plan.restore_mode == "streaming"
    assert plan.warnings
    assert "source-size restoration" in plan.warnings[0]


@pytest.mark.fast
def test_seedvr2_video_plan_requires_low_ram_for_7b_video():
    source_probe = DecodedVideoClip(
        frames=[],
        fps=29.97,
        source_width=320,
        source_height=240,
        source_frame_count=6,
        source_duration_seconds=0.2,
        audio_present=False,
        clip_start_frame=0,
        clip_frame_count=1,
    )

    plan = _plan_seedvr2_video_restore(
        model_config=ModelConfig.seedvr2_7b(),
        source_probe=source_probe,
        requested_frames=6,
        resolution=720,
        temporal_chunk_size=49,
        temporal_chunk_overlap=16,
        chunk_size_was_explicit=True,
        chunk_overlap_was_explicit=True,
        low_ram_requested=False,
        cache_limit_gb=8.0,
        force_unsafe_memory_profile=False,
    )

    assert plan.low_ram_required is True
    assert plan.warnings


@pytest.mark.fast
def test_seedvr2_video_plan_reserves_resident_weight_headroom_for_7b_chunks(monkeypatch):
    source_probe = DecodedVideoClip(
        frames=[],
        fps=29.97,
        source_width=320,
        source_height=240,
        source_frame_count=2908,
        source_duration_seconds=97.03,
        audio_present=False,
        clip_start_frame=0,
        clip_frame_count=1,
    )
    resident_weight_bytes = 34 * (1000**3)

    monkeypatch.setattr(
        SeedVR2Initializer,
        "estimate_resident_weight_bytes",
        staticmethod(lambda *args, **kwargs: resident_weight_bytes),
    )
    monkeypatch.setattr(SeedVR2Util, "_host_total_memory_bytes", staticmethod(lambda: 128 * (1000**3)))
    monkeypatch.setattr(SeedVR2Util, "_host_available_memory_bytes", staticmethod(lambda: 121 * (1000**3)))

    plan = _plan_seedvr2_video_restore(
        model_config=ModelConfig.seedvr2_7b(),
        source_probe=source_probe,
        requested_frames=2908,
        resolution=ScaleFactor(1),
        temporal_chunk_size=49,
        temporal_chunk_overlap=16,
        chunk_size_was_explicit=True,
        chunk_overlap_was_explicit=True,
        low_ram_requested=True,
        cache_limit_gb=8.0,
        force_unsafe_memory_profile=False,
    )

    inner_dim, text_attention_mode = _seedvr2_memory_profile(ModelConfig.seedvr2_7b())
    estimated_bytes = SeedVR2Util.estimate_video_restore_working_set_bytes(
        frame_count=SeedVR2Util.padded_video_frame_count(plan.effective_chunk_size or 1),
        height=240,
        width=320,
        inner_dim=inner_dim,
        text_attention_mode=text_attention_mode,
    )
    budget_bytes = SeedVR2Util.host_safe_video_memory_budget_bytes(reserve_bytes=resident_weight_bytes)

    assert plan.effective_chunk_size == 49
    assert plan.effective_chunk_overlap == 16
    assert estimated_bytes <= budget_bytes


@pytest.mark.fast
def test_seedvr2_video_plan_unsafe_default_prefers_whole_shot_when_chunk_size_is_implicit():
    source_probe = DecodedVideoClip(
        frames=[],
        fps=29.97,
        source_width=320,
        source_height=240,
        source_frame_count=149,
        source_duration_seconds=149 / 29.97,
        audio_present=False,
        clip_start_frame=0,
        clip_frame_count=1,
    )

    plan = _plan_seedvr2_video_restore(
        model_config=ModelConfig.seedvr2_7b(),
        source_probe=source_probe,
        requested_frames=149,
        resolution=ScaleFactor(1),
        temporal_chunk_size=49,
        temporal_chunk_overlap=16,
        chunk_size_was_explicit=False,
        chunk_overlap_was_explicit=False,
        low_ram_requested=True,
        cache_limit_gb=8.0,
        force_unsafe_memory_profile=True,
    )

    assert plan.effective_chunk_size == 149
    assert plan.effective_chunk_overlap == 0


@pytest.mark.fast
def test_seedvr2_main_rejects_unsafe_safe_profile_before_model_load(monkeypatch, tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.touch()

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("SeedVR2 should not be constructed for an unsupported safe video profile")

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", fail_if_loaded)
    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=320,
                source_height=240,
                source_frame_count=130,
                source_duration_seconds=4.337,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_path),
            "--resolution",
            "2x",
        ],
    )

    with pytest.raises(SystemExit):
        seedvr2_upscale.main()


@pytest.mark.fast
def test_seedvr2_main_reloads_model_per_video_seed(monkeypatch, tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.touch()
    output_path = tmp_path / "restored.mp4"
    constructed: list[int] = []

    class FakeSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            self.model_config = model_config
            constructed.append(1)
            self.tiling_config = None

        def restore_video_to_path(
            self,
            *,
            seed,
            video_path,
            resolution,
            softness,
            start_seconds,
            max_frames,
            output_path,
            export_json_metadata,
            overwrite,
            temporal_chunk_size,
            temporal_chunk_overlap,
            color_correction_mode,
            drop_audio,
            restore_metadata,
            enforce_memory_budget,
            validate_health=True,
        ):
            assert drop_audio is False
            return output_path

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", FakeSeedVR2)
    monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=64,
                source_height=48,
                source_frame_count=14,
                source_duration_seconds=14 / 29.97,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_path),
            "--seed",
            "1",
            "2",
            "--output",
            str(output_path),
        ],
    )

    seedvr2_upscale.main()

    assert len(constructed) == 2


@pytest.mark.fast
def test_seedvr2_main_uses_unique_output_paths_for_multiple_video_inputs_and_seeds(monkeypatch, tmp_path):
    video_a = tmp_path / "clip_a.mp4"
    video_b = tmp_path / "clip_b.mp4"
    video_a.touch()
    video_b.touch()
    observed_paths: list[str] = []

    class FakePlan:
        warnings: list[str] = []

    monkeypatch.setattr(
        seedvr2_upscale, "_resolve_seedvr2_model", lambda *args, **kwargs: (ModelConfig.seedvr2_3b(), None)
    )
    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=64,
                source_height=48,
                source_frame_count=14,
                source_duration_seconds=14 / 29.97,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(seedvr2_upscale, "_plan_seedvr2_video_restore", lambda **kwargs: FakePlan())
    monkeypatch.setattr(seedvr2_upscale, "_apply_cache_limit_if_needed", lambda args: None)

    def fake_run_video_with_fresh_model(**kwargs):
        observed_paths.append(
            kwargs["output_pattern"].format(seed=kwargs["seed"], input_name=kwargs["video_path"].stem)
        )

    monkeypatch.setattr(seedvr2_upscale, "_run_video_with_fresh_model", fake_run_video_with_fresh_model)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_a),
            str(video_b),
            "--seed",
            "1",
            "2",
        ],
    )

    seedvr2_upscale.main()

    assert observed_paths == [
        "video_seed_1_clip_a.mp4",
        "video_seed_2_clip_a.mp4",
        "video_seed_1_clip_b.mp4",
        "video_seed_2_clip_b.mp4",
    ]


@pytest.mark.fast
def test_seedvr2_main_uses_unique_output_paths_for_expanded_video_directory(monkeypatch, tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "clip_a.mp4").touch()
    (video_dir / "clip_b.mp4").touch()
    observed_paths: list[str] = []

    class FakePlan:
        warnings: list[str] = []

    monkeypatch.setattr(
        seedvr2_upscale, "_resolve_seedvr2_model", lambda *args, **kwargs: (ModelConfig.seedvr2_3b(), None)
    )
    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=64,
                source_height=48,
                source_frame_count=14,
                source_duration_seconds=14 / 29.97,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(seedvr2_upscale, "_plan_seedvr2_video_restore", lambda **kwargs: FakePlan())
    monkeypatch.setattr(seedvr2_upscale, "_apply_cache_limit_if_needed", lambda args: None)

    def fake_run_video_with_fresh_model(**kwargs):
        observed_paths.append(
            kwargs["output_pattern"].format(seed=kwargs["seed"], input_name=kwargs["video_path"].stem)
        )

    monkeypatch.setattr(seedvr2_upscale, "_run_video_with_fresh_model", fake_run_video_with_fresh_model)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(video_dir),
            "--seed",
            "1",
            "2",
        ],
    )

    seedvr2_upscale.main()

    assert observed_paths == [
        "video_seed_1_clip_a.mp4",
        "video_seed_2_clip_a.mp4",
        "video_seed_1_clip_b.mp4",
        "video_seed_2_clip_b.mp4",
    ]


@pytest.mark.fast
def test_seedvr2_rejects_same_stem_video_collisions_with_replace_true(monkeypatch, tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "frame.mp4").touch()
    (second_dir / "frame.mp4").touch()

    class FakePlan:
        warnings: list[str] = []

    monkeypatch.setattr(
        seedvr2_upscale, "_resolve_seedvr2_model", lambda *args, **kwargs: (ModelConfig.seedvr2_3b(), None)
    )
    monkeypatch.setattr(
        seedvr2_upscale.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=[],
                fps=29.97,
                source_width=64,
                source_height=48,
                source_frame_count=14,
                source_duration_seconds=14 / 29.97,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(seedvr2_upscale, "_plan_seedvr2_video_restore", lambda **kwargs: FakePlan())

    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-upscale-seedvr2",
            "--video-path",
            str(first_dir),
            str(second_dir),
        ],
    )

    with pytest.raises(SystemExit):
        seedvr2_upscale.main()


@pytest.mark.fast
def test_aligned_chunk_size_prefers_one_mod_four():
    assert _aligned_chunk_size(49) == 49
    assert _aligned_chunk_size(14) == 13
    assert _aligned_chunk_size(2) == 1


@pytest.mark.fast
def test_aligned_chunk_overlap_returns_stride_aligned_values_only():
    assert _aligned_chunk_overlap(29, 8) == 8
    assert _aligned_chunk_overlap(9, 4) == 0


@pytest.mark.fast
def test_seedvr2_model_resolution_defaults_to_3b():
    model_config, model_path = _resolve_seedvr2_model(model_arg=None, model_path=None)
    assert "seedvr2-3b" in model_config.aliases
    assert model_config.model_name == "ByteDance-Seed/SeedVR2-3B"
    assert model_path is None


@pytest.mark.fast
def test_seedvr2_model_resolution_supports_7b_alias():
    model_config, model_path = _resolve_seedvr2_model(model_arg="seedvr2-7b", model_path="seedvr2-7b")
    assert "seedvr2-7b" in model_config.aliases
    assert model_config.model_name == "ByteDance-Seed/SeedVR2-7B"
    assert model_path is None


@pytest.mark.fast
def test_seedvr2_model_resolution_supports_7b_sharp_alias():
    model_config, model_path = _resolve_seedvr2_model(model_arg="seedvr2-7b-sharp", model_path="seedvr2-7b-sharp")
    assert "seedvr2-7b-sharp" in model_config.aliases
    assert model_config.model_name == "ByteDance-Seed/SeedVR2-7B"
    assert model_path is None


@pytest.mark.fast
def test_seedvr2_model_resolution_infers_7b_from_custom_path():
    model_config, model_path = _resolve_seedvr2_model(
        model_arg="/tmp/seedvr2_ema_7b_fp16.safetensors",
        model_path="/tmp/seedvr2_ema_7b_fp16.safetensors",
    )
    assert "seedvr2-7b" in model_config.aliases
    assert model_path == "/tmp/seedvr2_ema_7b_fp16.safetensors"


@pytest.mark.fast
def test_seedvr2_model_resolution_detects_7b_directory(tmp_path):
    model_dir = tmp_path / "seedvr2"
    model_dir.mkdir()
    (model_dir / "seedvr2_ema_7b_fp16.safetensors").touch()
    model_config, model_path = _resolve_seedvr2_model(
        model_arg=str(model_dir),
        model_path=str(model_dir),
    )
    assert "seedvr2-7b" in model_config.aliases
    assert model_path == str(model_dir)


@pytest.mark.fast
def test_seedvr2_model_resolution_detects_official_7b_directory(tmp_path):
    model_dir = tmp_path / "seedvr2-official-7b"
    model_dir.mkdir()
    (model_dir / "seedvr2_ema_7b.pth").touch()

    model_config, model_path = _resolve_seedvr2_model(
        model_arg=str(model_dir),
        model_path=str(model_dir),
    )

    assert "seedvr2-7b" in model_config.aliases
    assert model_path == str(model_dir)


@pytest.mark.fast
def test_seedvr2_model_resolution_detects_official_7b_sharp_directory(tmp_path):
    model_dir = tmp_path / "seedvr2-official-7b-sharp"
    model_dir.mkdir()
    (model_dir / "seedvr2_ema_7b_sharp.pth").touch()

    model_config, model_path = _resolve_seedvr2_model(
        model_arg=str(model_dir),
        model_path=str(model_dir),
    )

    assert "seedvr2-7b-sharp" in model_config.aliases
    assert model_path == str(model_dir)


@pytest.mark.fast
def test_seedvr2_model_resolution_prefers_local_3b_when_directory_contains_only_3b(tmp_path):
    parent = tmp_path / "contains-7b-in-parent-name"
    parent.mkdir()
    model_dir = parent / "seedvr2"
    model_dir.mkdir()
    (model_dir / "seedvr2_ema_3b_fp16.safetensors").touch()

    model_config, model_path = _resolve_seedvr2_model(
        model_arg=str(model_dir),
        model_path=str(model_dir),
    )
    assert "seedvr2-3b" in model_config.aliases
    assert model_path == str(model_dir)


@pytest.mark.fast
def test_seedvr2_model_resolution_preserves_official_bytedance_handle():
    model_config, model_path = _resolve_seedvr2_model(
        model_arg="ByteDance-Seed/SeedVR2-3B",
        model_path=None,
    )

    assert "seedvr2-3b" in model_config.aliases
    assert model_path == "ByteDance-Seed/SeedVR2-3B"


@pytest.mark.fast
def test_seedvr2_model_resolution_preserves_official_bytedance_7b_handle():
    model_config, model_path = _resolve_seedvr2_model(
        model_arg="ByteDance-Seed/SeedVR2-7B",
        model_path=None,
    )

    assert "seedvr2-7b" in model_config.aliases
    assert model_path == "ByteDance-Seed/SeedVR2-7B"


@pytest.mark.fast
def test_seedvr2_model_resolution_preserves_abstractframework_3b_package_handle():
    model_config, model_path = _resolve_seedvr2_model(
        model_arg="AbstractFramework/seedvr2-3b-8bit",
        model_path=None,
    )

    assert "seedvr2-3b" in model_config.aliases
    assert model_path == "AbstractFramework/seedvr2-3b-8bit"


@pytest.mark.fast
def test_seedvr2_model_resolution_preserves_abstractframework_7b_package_handle():
    model_config, model_path = _resolve_seedvr2_model(
        model_arg="AbstractFramework/seedvr2-7b-8bit",
        model_path=None,
    )

    assert "seedvr2-7b" in model_config.aliases
    assert model_path == "AbstractFramework/seedvr2-7b-8bit"


@pytest.mark.fast
def test_seedvr2_model_resolution_keeps_legacy_numz_handle_explicit():
    model_config, model_path = _resolve_seedvr2_model(
        model_arg="numz/SeedVR2_comfyUI",
        model_path=None,
    )

    assert "seedvr2-3b" in model_config.aliases
    assert model_path == "numz/SeedVR2_comfyUI"


@pytest.mark.fast
def test_seedvr2_model_resolution_rejects_unknown_hf_handle():
    with pytest.raises(ValueError, match="Unsupported SeedVR2 model handle"):
        _resolve_seedvr2_model(model_arg="AbstractFramework/not-seedvr2", model_path=None)


@pytest.mark.fast
def test_seedvr2_model_resolution_detects_official_3b_directory(tmp_path):
    model_dir = tmp_path / "seedvr2-official"
    model_dir.mkdir()
    (model_dir / "seedvr2_ema_3b.pth").touch()

    model_config, model_path = _resolve_seedvr2_model(
        model_arg=str(model_dir),
        model_path=str(model_dir),
    )

    assert "seedvr2-3b" in model_config.aliases
    assert model_path == str(model_dir)
