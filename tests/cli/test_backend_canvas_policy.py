from pathlib import Path

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
