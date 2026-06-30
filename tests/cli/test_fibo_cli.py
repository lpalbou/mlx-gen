from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.fibo.cli import fibo_edit


def test_fibo_edit_missing_prompt_is_parser_error(tmp_path, capsys):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(source)

    with patch(
        "sys.argv",
        [
            "mflux-generate-fibo-edit",
            "--image-path",
            str(source),
        ],
    ):
        with pytest.raises(SystemExit):
            fibo_edit.main()

    assert "requires an edit instruction" in capsys.readouterr().err


def test_fibo_edit_multi_seed_matte_output_gets_seed_suffix(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(source)

    class FakeResult:
        def __init__(self):
            self.image = Image.new("RGB", (64, 64), (240, 240, 240))

        def save_with_image(self, **kwargs):
            Path(kwargs["path"]).write_bytes(b"rgba")

        def save(self, **kwargs):
            Path(kwargs["path"]).write_bytes(b"matte")

    class FakeFiboEdit:
        def __init__(self, **kwargs):
            self.model_config = kwargs["model_config"]

        def generate_image(self, **kwargs):
            return FakeResult()

    monkeypatch.setattr(fibo_edit, "FIBOEdit", FakeFiboEdit)
    monkeypatch.setattr(fibo_edit.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        fibo_edit.CliRuntimeEventStream,
        "subscribe_model",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        fibo_edit,
        "_resolve_fibo_edit_model_config",
        lambda parser, args: ModelConfig.from_name("fibo-edit-rmbg"),
    )
    monkeypatch.setattr(fibo_edit, "_json_prompt_for_edit", lambda args, model_config: "{}")
    monkeypatch.setattr(
        fibo_edit.FiboEditUtil,
        "build_rgba_composite_image",
        staticmethod(lambda image_path, generated: Image.new("RGBA", (64, 64), (255, 255, 255, 255))),
    )

    with patch(
        "sys.argv",
        [
            "mflux-generate-fibo-edit",
            "--model",
            "fibo-edit-rmbg",
            "--image-path",
            str(source),
            "--prompt",
            "remove the background",
            "--seed",
            "11",
            "22",
            "--output",
            str(tmp_path / "edited.png"),
            "--matte-output",
            str(tmp_path / "matte.png"),
        ],
    ):
        fibo_edit.main()

    assert (tmp_path / "edited_seed_11.png").read_bytes() == b"rgba"
    assert (tmp_path / "edited_seed_22.png").read_bytes() == b"rgba"
    assert (tmp_path / "matte_seed_11.png").read_bytes() == b"matte"
    assert (tmp_path / "matte_seed_22.png").read_bytes() == b"matte"


def test_fibo_edit_rejects_matte_output_aliasing_primary_output(monkeypatch, tmp_path, capsys):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(source)
    shared_output = tmp_path / "shared.png"

    class FakeResult:
        def __init__(self):
            self.image = Image.new("RGB", (64, 64), (240, 240, 240))

        def save_with_image(self, **kwargs):
            Path(kwargs["path"]).write_bytes(b"rgba")

        def save(self, **kwargs):
            Path(kwargs["path"]).write_bytes(b"matte")

    class FakeFiboEdit:
        def __init__(self, **kwargs):
            self.model_config = kwargs["model_config"]

        def generate_image(self, **kwargs):
            return FakeResult()

    monkeypatch.setattr(fibo_edit, "FIBOEdit", FakeFiboEdit)
    monkeypatch.setattr(fibo_edit.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        fibo_edit.CliRuntimeEventStream,
        "subscribe_model",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        fibo_edit,
        "_resolve_fibo_edit_model_config",
        lambda parser, args: ModelConfig.from_name("fibo-edit-rmbg"),
    )
    monkeypatch.setattr(fibo_edit, "_json_prompt_for_edit", lambda args, model_config: "{}")
    monkeypatch.setattr(
        fibo_edit.FiboEditUtil,
        "build_rgba_composite_image",
        staticmethod(lambda image_path, generated: Image.new("RGBA", (64, 64), (255, 255, 255, 255))),
    )

    with patch(
        "sys.argv",
        [
            "mflux-generate-fibo-edit",
            "--model",
            "fibo-edit-rmbg",
            "--image-path",
            str(source),
            "--prompt",
            "remove the background",
            "--seed",
            "11",
            "--output",
            str(shared_output),
            "--matte-output",
            str(shared_output),
        ],
    ):
        with pytest.raises(SystemExit) as exc:
            fibo_edit.main()

    assert exc.value.code == 1
    assert shared_output.read_bytes() == b"rgba"
    captured = capsys.readouterr()
    assert "same path as --output" in captured.out + captured.err
