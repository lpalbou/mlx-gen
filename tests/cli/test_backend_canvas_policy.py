from pathlib import Path

import pytest
from PIL import Image

from mflux.models.qwen.cli import qwen_image_generate
from mflux.utils.scale_factor import ScaleFactor


class _SavedImage:
    def save(self, path, export_json_metadata=False, overwrite=True, embed_metadata=False):
        Path(path).touch()


def test_latent_backend_cli_defers_auto_dimension_to_config(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGB", (432, 240)).save(source)
    output = tmp_path / "out.png"
    captured = {}

    class FakeQwen:
        def __init__(self, **kwargs):
            pass

        def generate_image(self, **kwargs):
            captured.update(kwargs)
            return _SavedImage()

    monkeypatch.setattr(qwen_image_generate, "QwenImage", FakeQwen)
    monkeypatch.setattr(qwen_image_generate.PromptUtil, "read_prompt", lambda args: "test")
    monkeypatch.setattr(qwen_image_generate.PromptUtil, "read_negative_prompt", lambda args: "")
    monkeypatch.setattr(qwen_image_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-generate-qwen",
            "--model",
            "qwen-image",
            "--prompt",
            "test",
            "--image-path",
            str(source),
            "--width",
            "432",
            "--height",
            "auto",
            "--image-strength",
            "0.45",
            "--output",
            str(output),
        ],
    )

    qwen_image_generate.main()

    assert captured["width"] == 432
    assert isinstance(captured["height"], ScaleFactor)
    assert captured["height"].value == 1
    assert captured["canvas_policy"] == "source-aspect"
    assert captured["resize_mode"] == "resize"


def test_latent_backend_cli_forwards_resize_mode(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGB", (432, 240)).save(source)
    output = tmp_path / "out.png"
    captured = {}

    class FakeQwen:
        def __init__(self, **kwargs):
            pass

        def generate_image(self, **kwargs):
            captured.update(kwargs)
            return _SavedImage()

    monkeypatch.setattr(qwen_image_generate, "QwenImage", FakeQwen)
    monkeypatch.setattr(qwen_image_generate.PromptUtil, "read_prompt", lambda args: "test")
    monkeypatch.setattr(qwen_image_generate.PromptUtil, "read_negative_prompt", lambda args: "")
    monkeypatch.setattr(qwen_image_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-generate-qwen",
            "--model",
            "qwen-image",
            "--prompt",
            "test",
            "--image-path",
            str(source),
            "--image-strength",
            "0.45",
            "--canvas-policy",
            "exact-resize",
            "--resize-mode",
            "crop",
            "--output",
            str(output),
        ],
    )

    qwen_image_generate.main()

    assert captured["canvas_policy"] == "exact-resize"
    assert captured["resize_mode"] == "crop"


def test_latent_backend_cli_replays_canvas_policy_and_resize_mode_from_metadata(monkeypatch, tmp_path: Path):
    # Cycle-3 review regression: image metadata records canvas_policy/resize_mode,
    # so a faithful -C replay must reproduce the same geometry (crop != resize).
    import json

    source = tmp_path / "source.png"
    Image.new("RGB", (432, 240)).save(source)
    metadata_path = tmp_path / "prior.json"
    metadata_path.write_text(
        json.dumps(
            {
                "prompt": "test",
                "image_path": str(source),
                "image_strength": 0.45,
                "canvas_policy": "exact-resize",
                "resize_mode": "crop",
            }
        )
    )
    captured = {}

    class FakeQwen:
        def __init__(self, **kwargs):
            pass

        def generate_image(self, **kwargs):
            captured.update(kwargs)
            return _SavedImage()

    monkeypatch.setattr(qwen_image_generate, "QwenImage", FakeQwen)
    monkeypatch.setattr(qwen_image_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-generate-qwen",
            "--model",
            "qwen-image",
            "--config-from-metadata",
            str(metadata_path),
            "--output",
            str(tmp_path / "out.png"),
        ],
    )

    qwen_image_generate.main()

    assert captured["canvas_policy"] == "exact-resize"
    assert captured["resize_mode"] == "crop"

    # An explicit flag still beats the replayed value.
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-generate-qwen",
            "--model",
            "qwen-image",
            "--config-from-metadata",
            str(metadata_path),
            "--resize-mode",
            "pad",
            "--output",
            str(tmp_path / "out2.png"),
        ],
    )

    qwen_image_generate.main()

    assert captured["resize_mode"] == "pad"


def test_structured_control_route_rejects_non_default_resize_mode(monkeypatch, tmp_path: Path, capsys):
    control = tmp_path / "control.png"
    Image.new("RGB", (64, 64)).save(control)

    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-generate-qwen",
            "--model",
            # The exact validated structured-control row; base qwen-image rejects
            # --controlnet-image-path before the resize-mode guard is reached.
            "AbstractFramework/qwen-image-8bit",
            "--prompt",
            "test",
            "--controlnet-image-path",
            str(control),
            "--resize-mode",
            "crop",
            "--output",
            str(tmp_path / "out.png"),
        ],
    )

    with pytest.raises(SystemExit):
        qwen_image_generate.main()

    assert "--resize-mode is not supported" in capsys.readouterr().err


def test_edit_route_parser_rejects_resize_mode_flag(monkeypatch, tmp_path: Path, capsys):
    # Edit/reference conditioning keeps reference-pinned resize geometry: the flag is
    # deliberately not defined on that parser, so argparse fails loudly instead of
    # silently ignoring the request.
    from mflux.models.qwen.cli import qwen_image_edit_generate

    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64)).save(source)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-generate-qwen-edit",
            "--model",
            "qwen-image-edit",
            "--prompt",
            "test",
            "--image-paths",
            str(source),
            "--resize-mode",
            "crop",
            "--output",
            str(tmp_path / "out.png"),
        ],
    )

    with pytest.raises(SystemExit):
        qwen_image_edit_generate.main()

    assert "unrecognized arguments: --resize-mode" in capsys.readouterr().err
