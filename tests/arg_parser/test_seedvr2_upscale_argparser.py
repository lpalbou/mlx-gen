from pathlib import Path
from unittest.mock import patch

import pytest

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.download_policy import DownloadRequiredError
from mflux.models.seedvr2.cli import seedvr2_upscale
from mflux.models.seedvr2.cli.seedvr2_upscale import _expand_image_paths, _resolve_seedvr2_model
from mflux.utils.scale_factor import ScaleFactor


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
        def save(self, path, export_json_metadata=False, overwrite=True):
            saved["path"] = path
            saved["export_json_metadata"] = export_json_metadata
            saved["overwrite"] = overwrite

    class FakeSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            saved["quantize"] = quantize
            saved["model_path"] = model_path
            saved["model_config"] = model_config.model_name
            self.tiling_config = None

        def generate_image(self, *, seed, image_path, resolution, softness):
            saved["seed"] = seed
            saved["image_path"] = image_path
            saved["resolution"] = resolution
            saved["softness"] = softness
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

    assert saved["path"] == str(output_path)
    assert saved["export_json_metadata"] is True
    assert saved["overwrite"] is True


@pytest.mark.fast
def test_seedvr2_main_enables_vae_tiling_when_requested(monkeypatch, tmp_path):
    image_path = tmp_path / "source.png"
    image_path.touch()
    output_path = tmp_path / "upscaled.png"
    saved: dict[str, object] = {}

    class FakeResult:
        def save(self, path, export_json_metadata=False, overwrite=True):
            pass

    class FakeSeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            self.tiling_config = None
            saved["model"] = self

        def generate_image(self, *, seed, image_path, resolution, softness):
            saved["tiling_config"] = self.tiling_config
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
    assert saved["tiling_config"].vae_decode_tiles_per_dim == 8


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
