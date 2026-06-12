import importlib
import json
import sys
from pathlib import Path

import pytest
import toml
from PIL import Image, ImageChops

import mlxgen
from mflux.cli import mlx_gen
from mflux.cli.mlx_gen import RouterInvocation
from mflux.models.common.download_policy import downloads_enabled


def test_generate_help_renders_padding_examples(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mlxgen", "generate", "--help"])

    with pytest.raises(SystemExit) as exc:
        mlx_gen.main()

    assert exc.value.code == 0
    help_output = capsys.readouterr().out
    assert "--reframe-padding" in help_output
    assert "0,25%,0,25%" in help_output


@pytest.mark.parametrize(
    "module_name,argv0",
    [
        ("mflux.models.flux2.cli.flux2_edit_generate", "mflux-generate-flux2-edit"),
        ("mflux.models.qwen.cli.qwen_image_edit_generate", "mflux-generate-qwen-edit"),
    ],
)
def test_backend_edit_help_renders_canvas_expansion_options(module_name, argv0, monkeypatch, capsys):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [argv0, "--help"])

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 0
    help_output = capsys.readouterr().out
    assert "--reframe-padding" in help_output
    assert "--outpaint-padding" in help_output
    assert "adaptive" in help_output
    assert "source blend" in help_output


def test_flux2_edit_backend_rejects_distilled_outpaint_before_model_load(tmp_path, monkeypatch, capsys):
    from mflux.models.flux2.cli import flux2_edit_generate

    source = tmp_path / "source.png"
    Image.new("RGB", (128, 64), color="white").save(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-flux2-edit",
            "--model",
            "flux2-klein-4b",
            "--image-paths",
            str(source),
            "--outpaint-padding",
            "0,25%,0,25%",
            "--prompt",
            "extend",
        ],
    )

    with pytest.raises(SystemExit):
        flux2_edit_generate.main()

    assert "requires a FLUX.2 Klein base model" in capsys.readouterr().err


def test_qwen_base_single_image_requires_latent_strength(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "Qwen/Qwen-Image",
                "--image",
                "input.png",
                "--prompt",
                "make it cinematic",
            ]
        )

    assert "image-strength is required for latent image-to-image mode" in capsys.readouterr().err


def test_routes_qwen_single_image_with_strength_to_latent_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "Qwen/Qwen-Image",
            "--image",
            "input.png",
            "--image-strength",
            "0.4",
            "--prompt",
            "make it cinematic",
        ]
    )

    assert invocation.target_name == "mflux-generate-qwen"
    assert invocation.argv == [
        "mflux-generate-qwen",
        "--model",
        "Qwen/Qwen-Image",
        "--image-path",
        "input.png",
        "--image-strength",
        "0.4",
        "--prompt",
        "make it cinematic",
    ]


def test_qwen_backend_defaults_to_flow_match_scheduler(monkeypatch):
    from mflux.models.qwen.cli import qwen_image_generate

    observed = {}

    class FakeImage:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeQwenImage:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_image(self, **kwargs):
            observed["generate"] = kwargs
            return FakeImage()

    monkeypatch.setattr(qwen_image_generate, "QwenImage", FakeQwenImage)
    monkeypatch.setattr(qwen_image_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-qwen",
            "--model",
            "qwen-image",
            "--prompt",
            "a spaceship in the snow",
            "--output",
            "out.png",
        ],
    )

    qwen_image_generate.main()

    assert observed["generate"]["scheduler"] == "flow_match_euler_discrete"


def test_qwen_edit_backend_outpaint_preserves_source_region(monkeypatch, tmp_path):
    from mflux.models.qwen.cli import qwen_image_edit_generate

    source = tmp_path / "source.png"
    output = tmp_path / "out.png"
    Image.new("RGB", (12, 8), color=(20, 40, 60)).save(source)
    observed = {}

    class FakeImage:
        def __init__(self, image):
            self.image = image
            self.image_paths = None

        def save(self, **kwargs):
            observed["save"] = kwargs
            self.image.save(kwargs["path"])

    class FakeQwenEdit:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_image(self, **kwargs):
            observed["generate"] = kwargs
            return FakeImage(Image.open(kwargs["image_paths"][0]).convert("RGB"))

    monkeypatch.setattr(qwen_image_edit_generate, "QwenImageEdit", FakeQwenEdit)
    monkeypatch.setattr(qwen_image_edit_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-qwen-edit",
            "--model",
            "qwen-image-edit-2511",
            "--image-paths",
            str(source),
            "--outpaint-padding",
            "2,4,2,4",
            "--prompt",
            "extend the icy canyon",
            "--output",
            str(output),
        ],
    )

    qwen_image_edit_generate.main()

    assert observed["generate"]["width"] == 32
    assert observed["generate"]["height"] == 16
    assert observed["generate"]["canvas_policy"] == "exact-resize"
    assert observed["generate"]["image_path"] == str(source)
    assert Path(observed["generate"]["image_paths"][0]).name == "outpaint_canvas.png"
    generated = Image.open(output).convert("RGB")
    source_interior = Image.open(source).convert("RGB").crop((1, 1, 11, 7))
    output_interior = generated.crop((5, 3, 15, 9))
    assert ImageChops.difference(output_interior, source_interior).getbbox() is None


def test_qwen_edit_backend_reframe_uses_expanded_canvas(monkeypatch, tmp_path):
    from mflux.models.qwen.cli import qwen_image_edit_generate

    source = tmp_path / "source.png"
    output = tmp_path / "out.png"
    Image.new("RGB", (12, 8), color=(20, 40, 60)).save(source)
    observed = {}

    class FakeImage:
        def __init__(self, image):
            self.image = image
            self.image_paths = None

        def save(self, **kwargs):
            observed["save"] = kwargs
            observed["extra_metadata"] = dict(getattr(self, "extra_metadata", {}))
            self.image.save(kwargs["path"])

    class FakeQwenEdit:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_image(self, **kwargs):
            observed["generate"] = kwargs
            canvas = Image.open(kwargs["image_paths"][0]).convert("RGB")
            observed["canvas_region"] = canvas.crop((0, 2, 12, 10)).copy()
            return FakeImage(Image.new("RGB", (kwargs["width"], kwargs["height"]), color=(0, 255, 0)))

    monkeypatch.setattr(qwen_image_edit_generate, "QwenImageEdit", FakeQwenEdit)
    monkeypatch.setattr(qwen_image_edit_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-qwen-edit",
            "--model",
            "qwen-image-edit-2511",
            "--image-paths",
            str(source),
            "--reframe-padding",
            "2,4,2,0",
            "--prompt",
            "zoom out to reveal more canyon",
            "--output",
            str(output),
        ],
    )

    qwen_image_edit_generate.main()

    assert observed["generate"]["width"] == 16
    assert observed["generate"]["height"] == 16
    assert observed["generate"]["canvas_policy"] == "exact-resize"
    assert Path(observed["generate"]["image_paths"][0]).name == "reframe_canvas.png"
    assert ImageChops.difference(observed["canvas_region"], Image.open(source).convert("RGB")).getbbox() is None
    assert observed["extra_metadata"]["reframe_padding"] == "2,4,2,0"
    assert observed["extra_metadata"]["reframe_source_paste_left"] == 0
    assert observed["extra_metadata"]["reframe_source_paste_top"] == 2


def test_qwen_edit_backend_canvas_options_reject_explicit_size(monkeypatch, tmp_path, capsys):
    from mflux.models.qwen.cli import qwen_image_edit_generate

    source = tmp_path / "source.png"
    Image.new("RGB", (12, 8), color=(20, 40, 60)).save(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-qwen-edit",
            "--model",
            "qwen-image-edit-2511",
            "--image-paths",
            str(source),
            "--outpaint-padding",
            "2,4,2,4",
            "--width",
            "512",
            "--prompt",
            "extend the icy canyon",
            "--output",
            str(tmp_path / "out.png"),
        ],
    )

    with pytest.raises(SystemExit):
        qwen_image_edit_generate.main()

    assert "computes --width and --height" in capsys.readouterr().err


def test_qwen_base_multiple_images_requires_edit_model(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "qwen",
                "--images",
                "input.png",
                "style.png",
                "--prompt",
                "apply the second image style",
            ]
        )

    assert "does not support multi-reference image-to-image generation" in capsys.readouterr().err


def test_routes_qwen_edit_model_to_edit_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "AbstractFramework/qwen-image-edit-2511-8bit",
            "--image",
            "input.png",
            "--prompt",
            "turn the room into a pencil sketch",
        ]
    )

    assert invocation.target_name == "mflux-generate-qwen-edit"
    assert invocation.argv == [
        "mflux-generate-qwen-edit",
        "--model",
        "AbstractFramework/qwen-image-edit-2511-8bit",
        "--image-paths",
        "input.png",
        "--prompt",
        "turn the room into a pencil sketch",
    ]


def test_routes_generate_subcommand_form():
    invocation = mlx_gen._resolve_invocation(
        mlx_gen._normalize_command(
            [
                "generate",
                "--model",
                "z-image-turbo",
                "--prompt",
                "a puffin standing on a cliff",
            ]
        )
    )

    assert invocation.target_name == "mflux-generate-z-image-turbo"
    assert invocation.argv == [
        "mflux-generate-z-image-turbo",
        "--model",
        "z-image-turbo",
        "--prompt",
        "a puffin standing on a cliff",
    ]


def test_upscale_subcommand_routes_to_seedvr2_command(monkeypatch):
    from mflux.models.seedvr2.cli import seedvr2_upscale

    observed = {}

    def fake_upscale_main():
        observed["argv"] = list(sys.argv)

    monkeypatch.setattr(seedvr2_upscale, "main", fake_upscale_main)

    mlx_gen._run_model_command(
        [
            "upscale",
            "--model",
            "AbstractFramework/seedvr2-7b-8bit",
            "--image-path",
            "input.png",
            "--resolution",
            "2x",
        ]
    )

    assert observed["argv"] == [
        "mlxgen upscale",
        "--model",
        "AbstractFramework/seedvr2-7b-8bit",
        "--image-path",
        "input.png",
        "--resolution",
        "2x",
    ]


def test_upscale_subcommand_does_not_enable_downloads(monkeypatch):
    from mflux.models.seedvr2.cli import seedvr2_upscale

    observed = {}

    def fake_upscale_main():
        observed["downloads_enabled"] = downloads_enabled()

    monkeypatch.setattr(seedvr2_upscale, "main", fake_upscale_main)

    mlx_gen._run_model_command(["upscale", "--image-path", "input.png"])

    assert observed["downloads_enabled"] is False


def test_capabilities_command_reports_model_modes(capsys):
    mlx_gen._show_capabilities(["--model", "flux2-klein-4b"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["family"] == "flux2"
    modes = {capability["mode"] for capability in payload["capabilities"]}
    assert {"text-only", "latent-img2img", "edit-reference", "multi-reference"}.issubset(modes)
    edit = next(capability for capability in payload["capabilities"] if capability["id"] == "flux2.edit")
    reframe = next(capability for capability in payload["capabilities"] if capability["id"] == "flux2.reframe")
    assert edit["supports_lora"] is True
    assert edit["lora_status"] == "mapped-unvalidated"
    assert edit["supports_reframe"] is False
    assert reframe["supports_reframe"] is True


def test_capabilities_command_accepts_base_model_for_local_paths(capsys):
    mlx_gen._show_capabilities(["--model", "../models/local-flux2-folder", "--base-model", "flux2-klein-4b"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["family"] == "flux2"
    assert payload["model_name"] == "../models/local-flux2-folder"


def test_unified_router_rejects_lora_scales_without_paths(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "flux2-klein-4b",
                "--prompt",
                "test",
                "--lora-scales",
                "0.9",
            ]
        )

    assert "--lora-scales requires --lora-paths" in capsys.readouterr().err


def test_unified_router_rejects_lora_for_unsupported_family(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "bonsai-image-ternary",
                "--prompt",
                "test",
                "--lora-paths",
                "adapter.safetensors",
            ]
        )

    assert "LoRA mapping" in capsys.readouterr().err


def test_unified_router_rejects_flux2_dev_lora_for_flux2_klein(monkeypatch, capsys):
    from mflux.models.common.lora.lora_compatibility import LoRACompatibility

    monkeypatch.setattr(
        LoRACompatibility,
        "_cached_base_models",
        staticmethod(lambda repo_id: ("black-forest-labs/FLUX.2-dev",)),
    )

    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "flux2-klein-4b",
                "--prompt",
                "<sks> back view eye-level shot medium shot",
                "--lora-paths",
                "lovis93/Flux-2-Multi-Angles-LoRA-v2:flux-multi-angles-v2-72poses-comfy.safetensors",
            ]
        )

    error = capsys.readouterr().err
    assert "targets black-forest-labs/FLUX.2-dev" in error
    assert "FLUX.2 Klein" in error


def test_validation_command_reports_model_specific_status(capsys):
    mlx_gen._show_validation(["--model", "briaai/Fibo-Edit"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "briaai/Fibo-Edit"
    assert payload["status"] == "FAIL"
    assert {record["mode"] for record in payload["records"]} == {"edit-reference"}


def test_validation_command_lists_profiles(capsys):
    mlx_gen._show_validation(["--list"])

    payload = json.loads(capsys.readouterr().out)
    profile_ids = [profile["id"] for profile in payload["profiles"]]
    assert profile_ids[:3] == [
        "i2i_edit_5x4_2026_06_05",
        "reframe_outpaint_2026_06_08",
        "flux2_klein_base_starship_2026_06_10",
    ]
    assert "lora_qwen_edit_q8_ghibli_edit_2026_06_11" in profile_ids
    assert "lora_wan_a14b_q8_lightx2v_4step_i2v_2026_06_12" in profile_ids
    assert "lora_wan_a14b_q8_lightx2v_4step_t2v_2026_06_12" in profile_ids


def test_validation_command_reports_reframe_outpaint_profile(capsys):
    mlx_gen._show_validation(
        [
            "--profile",
            "reframe_outpaint_2026_06_08",
            "--model",
            "AbstractFramework/qwen-image-edit-2511-8bit",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "AbstractFramework/qwen-image-edit-2511-8bit"
    assert payload["status"] == "PASS"
    assert {record["step"] for record in payload["records"]} == {"RF", "OP"}
    assert {record["mode"] for record in payload["records"]} == {"edit-reference"}


def test_generate_with_path_reports_prepare_command(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "generate",
                "--model",
                "black-forest-labs/FLUX.2-klein-4B",
                "-q",
                "4",
                "--path",
                "models/flux.2-klein-4b-4bit",
            ]
        )

    error_output = capsys.readouterr().err
    assert "--path prepares a local model folder" in error_output
    assert (
        "mlxgen prepare --model black-forest-labs/FLUX.2-klein-4B --path models/flux.2-klein-4b-4bit --quantize 4"
    ) in error_output
    assert "--output" in error_output


def test_routes_flux2_with_image_to_image_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "AbstractFramework/flux.2-klein-4b-8bit",
            "--image",
            "input.png",
            "--prompt",
            "add sunglasses",
        ]
    )

    assert invocation.target_name == "mflux-generate-flux2-edit"
    assert invocation.argv == [
        "mflux-generate-flux2-edit",
        "--model",
        "AbstractFramework/flux.2-klein-4b-8bit",
        "--image-paths",
        "input.png",
        "--prompt",
        "add sunglasses",
    ]


def test_flux2_edit_backend_outpaint_preserves_source_region(monkeypatch, tmp_path):
    from mflux.models.flux2.cli import flux2_edit_generate

    source = tmp_path / "source.png"
    output = tmp_path / "out.png"
    Image.new("RGB", (12, 8), color=(80, 20, 10)).save(source)
    observed = {}

    class FakeImage:
        def __init__(self, image):
            self.image = image
            self.image_path = None
            self.image_paths = None

        def save(self, **kwargs):
            observed["save"] = kwargs
            self.image.save(kwargs["path"])

    class FakeFlux2Outpaint:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_image(self, **kwargs):
            observed["generate"] = kwargs
            return FakeImage(Image.open(kwargs["canvas"].canvas_path).convert("RGB"))

    monkeypatch.setattr(flux2_edit_generate, "Flux2KleinOutpaint", FakeFlux2Outpaint)
    monkeypatch.setattr(flux2_edit_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-flux2-edit",
            "--model",
            "flux2-klein-base-4b",
            "--image-paths",
            str(source),
            "--outpaint-padding",
            "2,4,2,4",
            "--prompt",
            "extend the icy canyon",
            "--output",
            str(output),
        ],
    )

    flux2_edit_generate.main()

    assert observed["generate"]["canvas"].target_width == 32
    assert observed["generate"]["canvas"].target_height == 16
    assert Path(observed["generate"]["canvas"].canvas_path).name == "outpaint_canvas.png"
    generated = Image.open(output).convert("RGB")
    source_interior = Image.open(source).convert("RGB").crop((1, 1, 11, 7))
    output_interior = generated.crop((5, 3, 15, 9))
    assert ImageChops.difference(output_interior, source_interior).getbbox() is None


def test_flux2_edit_backend_reframe_uses_expanded_canvas(monkeypatch, tmp_path):
    from mflux.models.flux2.cli import flux2_edit_generate

    source = tmp_path / "source.png"
    output = tmp_path / "out.png"
    Image.new("RGB", (12, 8), color=(80, 20, 10)).save(source)
    observed = {}

    class FakeImage:
        def __init__(self, image):
            self.image = image
            self.image_path = None
            self.image_paths = None

        def save(self, **kwargs):
            observed["save"] = kwargs
            observed["extra_metadata"] = dict(getattr(self, "extra_metadata", {}))
            self.image.save(kwargs["path"])

    class FakeFlux2Edit:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_image(self, **kwargs):
            observed["generate"] = kwargs
            canvas = Image.open(kwargs["image_paths"][0]).convert("RGB")
            observed["canvas_region"] = canvas.crop((0, 2, 12, 10)).copy()
            return FakeImage(Image.new("RGB", (kwargs["width"], kwargs["height"]), color=(0, 255, 0)))

    monkeypatch.setattr(flux2_edit_generate, "Flux2KleinEdit", FakeFlux2Edit)
    monkeypatch.setattr(flux2_edit_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-flux2-edit",
            "--model",
            "flux2-klein-4b",
            "--image-paths",
            str(source),
            "--reframe-padding",
            "2,4,2,0",
            "--prompt",
            "zoom out to reveal more canyon",
            "--output",
            str(output),
        ],
    )

    flux2_edit_generate.main()

    assert observed["generate"]["width"] == 16
    assert observed["generate"]["height"] == 16
    assert observed["generate"]["canvas_policy"] == "exact-resize"
    assert Path(observed["generate"]["image_paths"][0]).name == "reframe_canvas.png"
    assert ImageChops.difference(observed["canvas_region"], Image.open(source).convert("RGB")).getbbox() is None
    assert observed["extra_metadata"]["reframe_padding"] == "2,4,2,0"
    assert observed["extra_metadata"]["reframe_source_paste_left"] == 0
    assert observed["extra_metadata"]["reframe_source_paste_top"] == 2


def test_flux2_edit_backend_canvas_options_reject_explicit_size(monkeypatch, tmp_path, capsys):
    from mflux.models.flux2.cli import flux2_edit_generate

    source = tmp_path / "source.png"
    Image.new("RGB", (12, 8), color=(80, 20, 10)).save(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-flux2-edit",
            "--model",
            "flux2-klein-base-4b",
            "--image-paths",
            str(source),
            "--outpaint-padding",
            "2,4,2,4",
            "--width",
            "512",
            "--prompt",
            "extend the icy canyon",
            "--output",
            str(tmp_path / "out.png"),
        ],
    )

    with pytest.raises(SystemExit):
        flux2_edit_generate.main()

    assert "computes --width and --height" in capsys.readouterr().err


def test_routes_flux2_explicit_edit_with_image_to_edit_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "AbstractFramework/flux.2-klein-4b-8bit",
            "--task",
            "edit",
            "--image",
            "input.png",
            "--prompt",
            "add sunglasses",
        ]
    )

    assert invocation.target_name == "mflux-generate-flux2-edit"
    assert invocation.argv == [
        "mflux-generate-flux2-edit",
        "--model",
        "AbstractFramework/flux.2-klein-4b-8bit",
        "--image-paths",
        "input.png",
        "--prompt",
        "add sunglasses",
    ]


def test_routes_flux2_multiple_images_to_edit_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "AbstractFramework/flux.2-klein-4b-8bit",
            "--images",
            "input.png",
            "style.png",
            "--prompt",
            "combine the source and style",
        ]
    )

    assert invocation.target_name == "mflux-generate-flux2-edit"
    assert invocation.argv == [
        "mflux-generate-flux2-edit",
        "--model",
        "AbstractFramework/flux.2-klein-4b-8bit",
        "--image-paths",
        "input.png",
        "style.png",
        "--prompt",
        "combine the source and style",
    ]


def test_routes_flux2_with_image_strength_to_image_to_image_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "AbstractFramework/flux.2-klein-4b-8bit",
            "--image",
            "input.png",
            "--image-strength",
            "0.4",
            "--prompt",
            "make it cinematic",
        ]
    )

    assert invocation.target_name == "mflux-generate-flux2"
    assert invocation.argv == [
        "mflux-generate-flux2",
        "--model",
        "AbstractFramework/flux.2-klein-4b-8bit",
        "--image-path",
        "input.png",
        "--image-strength",
        "0.4",
        "--prompt",
        "make it cinematic",
    ]


def test_flux2_metadata_image_strength_routes_to_latent_i2i(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": "AbstractFramework/flux.2-klein-4b-8bit",
                "image_path": "input.png",
                "image_strength": 0.4,
                "prompt": "make it cinematic",
            }
        )
    )

    invocation = mlx_gen._resolve_invocation(["--config-from-metadata", str(metadata_path)])

    assert invocation.target_name == "mflux-generate-flux2"
    assert invocation.argv == [
        "mflux-generate-flux2",
        "--model",
        "AbstractFramework/flux.2-klein-4b-8bit",
        "--image-path",
        "input.png",
        "--config-from-metadata",
        str(metadata_path),
    ]


def test_routes_flux2_explicit_latent_i2i_mode_to_image_to_image_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "AbstractFramework/flux.2-klein-4b-8bit",
            "--image",
            "input.png",
            "--i2i-mode",
            "latent",
            "--image-strength",
            "0.4",
            "--prompt",
            "make it cinematic",
        ]
    )

    assert invocation.target_name == "mflux-generate-flux2"
    assert invocation.argv == [
        "mflux-generate-flux2",
        "--model",
        "AbstractFramework/flux.2-klein-4b-8bit",
        "--image-path",
        "input.png",
        "--image-strength",
        "0.4",
        "--prompt",
        "make it cinematic",
    ]


def test_latent_i2i_requires_image_strength(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "AbstractFramework/flux.2-klein-4b-8bit",
                "--image",
                "input.png",
                "--i2i-mode",
                "latent",
                "--prompt",
                "make it cinematic",
            ]
        )

    assert "image-strength is required for latent image-to-image mode" in capsys.readouterr().err


def test_routes_canvas_policy_to_selected_i2i_backend():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "AbstractFramework/flux.2-klein-4b-8bit",
            "--image",
            "input.png",
            "--i2i-mode",
            "latent",
            "--image-strength",
            "0.4",
            "--canvas-policy",
            "exact-resize",
            "--prompt",
            "make it cinematic",
        ]
    )

    assert invocation.target_name == "mflux-generate-flux2"
    assert "--canvas-policy" in invocation.argv
    assert invocation.argv[invocation.argv.index("--canvas-policy") + 1] == "exact-resize"


def test_image_strength_is_rejected_for_flux2_edit_mode(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "AbstractFramework/flux.2-klein-4b-8bit",
                "--image",
                "input.png",
                "--i2i-mode",
                "edit",
                "--image-strength",
                "0.4",
                "--prompt",
                "make it cinematic",
            ]
        )

    assert "image-strength is only supported for latent image-to-image mode" in capsys.readouterr().err


def test_mask_path_is_rejected_for_flux2_edit_mode(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "AbstractFramework/flux.2-klein-4b-8bit",
                "--image",
                "input.png",
                "--mask-path",
                "mask.png",
                "--prompt",
                "make it cinematic",
            ]
        )

    assert "mask-path is only supported" in capsys.readouterr().err


def test_outpaint_padding_routes_flux2_edit():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "AbstractFramework/flux.2-klein-base-9b-8bit",
            "--image",
            "input.png",
            "--outpaint-padding",
            "0,25%,0,25%",
            "--prompt",
            "extend the room",
        ]
    )

    assert invocation.target_name == "mflux-generate-flux2-edit"
    assert invocation.argv == [
        "mflux-generate-flux2-edit",
        "--model",
        "AbstractFramework/flux.2-klein-base-9b-8bit",
        "--image-paths",
        "input.png",
        "--prompt",
        "extend the room",
        "--outpaint-padding",
        "0,25%,0,25%",
    ]


def test_outpaint_padding_routes_qwen_edit():
    for model in [
        "AbstractFramework/qwen-image-edit-8bit",
        "AbstractFramework/qwen-image-edit-2509-8bit",
        "AbstractFramework/qwen-image-edit-2511-8bit",
    ]:
        invocation = mlx_gen._resolve_invocation(
            [
                "--model",
                model,
                "--image",
                "input.png",
                "--image-outpaint-padding",
                "25%,0,25%,0",
                "--prompt",
                "extend the room",
            ]
        )

        assert invocation.target_name == "mflux-generate-qwen-edit"
        assert invocation.argv == [
            "mflux-generate-qwen-edit",
            "--model",
            model,
            "--image-paths",
            "input.png",
            "--prompt",
            "extend the room",
            "--outpaint-padding",
            "25%,0,25%,0",
        ]


def test_outpaint_padding_is_rejected_for_latent_only_models(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "z-image-turbo",
                "--image",
                "input.png",
                "--outpaint-padding",
                "0,25%,0,25%",
                "--prompt",
                "extend the room",
            ]
        )

    assert "outpaint-padding is only supported" in capsys.readouterr().err


def test_outpaint_padding_rejects_conflicting_canvas_options(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "flux2-klein-base-4b",
                "--image",
                "input.png",
                "--outpaint-padding",
                "0,25%,0,25%",
                "--height",
                "512",
                "--prompt",
                "extend the room",
            ]
        )

    assert "computes --width and --height" in capsys.readouterr().err


def test_reframe_padding_routes_flux2_edit_with_computed_canvas(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (100, 50), color="white").save(source)

    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "AbstractFramework/flux.2-klein-4b-8bit",
            "--image",
            str(source),
            "--reframe-padding",
            "0,50%,0,0",
            "--prompt",
            "zoom out to reveal more background",
        ]
    )

    assert invocation.target_name == "mflux-generate-flux2-edit"
    assert invocation.argv == [
        "mflux-generate-flux2-edit",
        "--model",
        "AbstractFramework/flux.2-klein-4b-8bit",
        "--image-paths",
        str(source),
        "--prompt",
        "zoom out to reveal more background",
        "--reframe-padding",
        "0,50%,0,0",
    ]


def test_reframe_padding_routes_qwen_edit_with_computed_canvas(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (128, 64), color="white").save(source)

    for model in [
        "AbstractFramework/qwen-image-edit-8bit",
        "AbstractFramework/qwen-image-edit-2509-8bit",
        "AbstractFramework/qwen-image-edit-2511-8bit",
    ]:
        invocation = mlx_gen._resolve_invocation(
            [
                "--model",
                model,
                "--image",
                str(source),
                "--reframe-padding",
                "25%,0,25%,0",
                "--prompt",
                "zoom out vertically",
            ]
        )

        assert invocation.target_name == "mflux-generate-qwen-edit"
        assert invocation.argv == [
            "mflux-generate-qwen-edit",
            "--model",
            model,
            "--image-paths",
            str(source),
            "--prompt",
            "zoom out vertically",
            "--reframe-padding",
            "25%,0,25%,0",
        ]


def test_reframe_outpaint_validation_profile_records_route_to_supported_backends(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (128, 64), color="white").save(source)
    profile = mlxgen.get_validation_profile(mlxgen.REFRAME_OUTPAINT_PROFILE_ID)

    expected_targets = {
        "FLUX.2 Klein 4B": "mflux-generate-flux2-edit",
        "FLUX.2 Klein 9B": "mflux-generate-flux2-edit",
        "Qwen Image Edit": "mflux-generate-qwen-edit",
        "Qwen Image Edit 2509": "mflux-generate-qwen-edit",
        "Qwen Image Edit 2511": "mflux-generate-qwen-edit",
    }
    for record in profile.records:
        option = "--reframe-padding" if record.step == "RF" else "--outpaint-padding"
        if record.step == "OP" and record.family in {"FLUX.2 Klein 4B", "FLUX.2 Klein 9B"}:
            with pytest.raises(SystemExit):
                mlx_gen._resolve_invocation(
                    [
                        "--model",
                        record.model,
                        "--image",
                        str(source),
                        option,
                        "0,25%,0,25%",
                        "--prompt",
                        "expand the image",
                    ]
                )
            continue
        invocation = mlx_gen._resolve_invocation(
            [
                "--model",
                record.model,
                "--image",
                str(source),
                option,
                "0,25%,0,25%",
                "--prompt",
                "expand the image",
            ]
        )

        assert invocation.target_name == expected_targets[record.family]
        assert option in invocation.argv


def test_reframe_padding_is_rejected_for_latent_only_models(tmp_path, capsys):
    source = tmp_path / "source.png"
    Image.new("RGB", (128, 64), color="white").save(source)

    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "z-image-turbo",
                "--image",
                str(source),
                "--reframe-padding",
                "0,25%,0,25%",
                "--prompt",
                "zoom out",
            ]
        )

    assert "reframe-padding is only supported" in capsys.readouterr().err


def test_flux2_klein_base_accepts_outpaint(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (128, 64), color="white").save(source)

    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "flux2-klein-base-4b",
            "--image",
            str(source),
            "--outpaint-padding",
            "0,25%,0,25%",
            "--prompt",
            "extend the room",
        ]
    )

    assert invocation.target_name == "mflux-generate-flux2-edit"
    assert "--outpaint-padding" in invocation.argv


def test_flux2_klein_base_rejects_reframe(tmp_path, capsys):
    source = tmp_path / "source.png"
    Image.new("RGB", (128, 64), color="white").save(source)

    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "flux2-klein-base-4b",
                "--image",
                str(source),
                "--reframe-padding",
                "0,25%,0,25%",
                "--prompt",
                "zoom out",
            ]
        )

    assert "reframe-padding is only supported" in capsys.readouterr().err


def test_reframe_padding_rejects_conflicting_canvas_options(tmp_path, capsys):
    source = tmp_path / "source.png"
    Image.new("RGB", (128, 64), color="white").save(source)

    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "flux2-klein-4b",
                "--image",
                str(source),
                "--reframe-padding",
                "0,25%,0,25%",
                "--width",
                "512",
                "--prompt",
                "zoom out",
            ]
        )

    assert "computes --width and --height" in capsys.readouterr().err


def test_reframe_padding_rejects_noop_and_negative_values(tmp_path, capsys):
    source = tmp_path / "source.png"
    Image.new("RGB", (128, 64), color="white").save(source)

    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "flux2-klein-4b",
                "--image",
                str(source),
                "--reframe-padding",
                "0,0,0,0",
                "--prompt",
                "zoom out",
            ]
        )
    assert "must add pixels on at least one side" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "flux2-klein-4b",
                "--image",
                str(source),
                "--reframe-padding=-1,0,0,0",
                "--prompt",
                "zoom out",
            ]
        )
    assert "must be zero or positive" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "flux2-klein-4b",
                "--image",
                str(source),
                "--reframe-padding=abc%,0,0,0",
                "--prompt",
                "zoom out",
            ]
        )
    assert "Invalid padding value: abc%" in capsys.readouterr().err


def test_reframe_padding_rejects_multi_reference_inputs(tmp_path, capsys):
    source = tmp_path / "source.png"
    reference = tmp_path / "reference.png"
    Image.new("RGB", (128, 64), color="white").save(source)
    Image.new("RGB", (128, 64), color="black").save(reference)

    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "flux2-klein-4b",
                "--image",
                str(source),
                "--image",
                str(reference),
                "--reframe-padding",
                "0,25%,0,25%",
                "--prompt",
                "zoom out",
            ]
        )

    assert "reframe-padding is only supported" in capsys.readouterr().err


def test_routes_flux2_without_image_to_text_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "flux2-klein-9b",
            "--prompt",
            "a glass building in soft morning light",
        ]
    )

    assert invocation.target_name == "mflux-generate-flux2"
    assert invocation.argv == [
        "mflux-generate-flux2",
        "--model",
        "flux2-klein-9b",
        "--prompt",
        "a glass building in soft morning light",
    ]


def test_routes_bonsai_to_text_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "prism-ml/bonsai-image-ternary-4B-mlx-2bit",
            "--prompt",
            "a bonsai tree in a ceramic studio",
        ]
    )

    assert invocation.target_name == "mflux-generate-bonsai"
    assert invocation.argv == [
        "mflux-generate-bonsai",
        "--model",
        "prism-ml/bonsai-image-ternary-4B-mlx-2bit",
        "--prompt",
        "a bonsai tree in a ceramic studio",
    ]


def test_bonsai_rejects_image_inputs(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "bonsai-image-ternary",
                "--image",
                "input.png",
                "--prompt",
                "make it cinematic",
            ]
        )

    assert "Bonsai Image supports text-to-image only" in capsys.readouterr().err


def test_qwen_image_to_image_task_requires_an_image(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "Qwen/Qwen-Image",
                "--task",
                "image-to-image",
                "--prompt",
                "make it cinematic",
            ]
        )

    assert "image-to-image requires --image" in capsys.readouterr().err


def test_qwen_text_to_image_rejects_image(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "Qwen/Qwen-Image",
                "--task",
                "text-to-image",
                "--image",
                "input.png",
                "--prompt",
                "make it cinematic",
            ]
        )

    assert "text-to-image cannot be combined with --image" in capsys.readouterr().err


def test_task_edit_does_not_silently_swap_qwen_base_to_edit_model(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "Qwen/Qwen-Image",
                "--task",
                "edit",
                "--images",
                "input.png",
                "style.png",
                "--prompt",
                "apply the style from the second image",
            ]
        )

    assert "does not support multi-reference image-to-image generation" in capsys.readouterr().err


def test_family_override_routes_unidentifiable_local_path():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "../models/local-edit-checkpoint",
            "--family",
            "qwen",
            "--base-model",
            "qwen-image-edit-2511",
            "--task",
            "edit",
            "--image",
            "input.png",
            "--prompt",
            "replace the chair",
        ]
    )

    assert invocation.target_name == "mflux-generate-qwen-edit"
    assert invocation.argv == [
        "mflux-generate-qwen-edit",
        "--model",
        "../models/local-edit-checkpoint",
        "--base-model",
        "qwen-image-edit-2511",
        "--image-paths",
        "input.png",
        "--prompt",
        "replace the chair",
    ]


def test_family_override_flux2_local_path_requires_base_model(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "../models/local-flux2-folder",
                "--family",
                "flux2",
                "--prompt",
                "hello",
            ]
        )

    assert "Pass --base-model" in capsys.readouterr().err


def test_family_override_flux2_local_path_forwards_base_model():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "../models/local-flux2-folder",
            "--family",
            "flux2",
            "--base-model",
            "flux2-klein-4b",
            "--prompt",
            "hello",
        ]
    )

    assert invocation.target_name == "mflux-generate-flux2"
    assert invocation.argv == [
        "mflux-generate-flux2",
        "--model",
        "../models/local-flux2-folder",
        "--base-model",
        "flux2-klein-4b",
        "--prompt",
        "hello",
    ]


def test_family_override_rejects_conflicting_known_model(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "qwen-image",
                "--family",
                "flux2",
                "--prompt",
                "hello",
            ]
        )

    assert "conflicts with model" in capsys.readouterr().err


def test_family_override_fibo_local_path_requires_base_model(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "../models/local-image-folder",
                "--family",
                "fibo",
                "--prompt",
                "hello",
            ]
        )

    assert "Pass --base-model" in capsys.readouterr().err


def test_family_override_fibo_local_path_forwards_base_model():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "../models/local-fibo-folder",
            "--family",
            "fibo",
            "--base-model",
            "fibo",
            "--prompt",
            "hello",
        ]
    )

    assert invocation.target_name == "mflux-generate-fibo"
    assert invocation.argv == [
        "mflux-generate-fibo",
        "--model",
        "../models/local-fibo-folder",
        "--base-model",
        "fibo",
        "--prompt",
        "hello",
    ]


def test_unified_router_rejects_fibo_edit_until_visual_parity_passes(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "briaai/Fibo-Edit",
                "--image",
                "input.png",
                "--prompt",
                '{"short_description":"ship","edit_instruction":"Turn the ship into a pencil sketch"}',
            ]
        )

    assert "does not expose unified generation capabilities" in capsys.readouterr().err


def test_fibo_edit_mask_path_is_not_advertised_by_unified_router(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "briaai/Fibo-Edit",
                "--image",
                "input.png",
                "--mask-path",
                "mask.png",
                "--prompt",
                "replace the selected object",
            ]
        )

    assert "does not expose unified generation capabilities" in capsys.readouterr().err


def test_fibo_edit_masked_image_path_alias_is_not_advertised_by_unified_router(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "briaai/Fibo-Edit",
                "--image",
                "input.png",
                "--masked-image-path",
                "mask.png",
                "--prompt",
                "replace the selected object",
            ]
        )

    assert "does not expose unified generation capabilities" in capsys.readouterr().err


def test_metadata_can_supply_model_and_images(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": "AbstractFramework/qwen-image-edit-2511-8bit",
                "image_paths": ["input.png", "style.png"],
                "prompt": "apply the second image style",
            }
        )
    )

    invocation = mlx_gen._resolve_invocation(["--config-from-metadata", str(metadata_path)])

    assert invocation.target_name == "mflux-generate-qwen-edit"
    assert invocation.argv == [
        "mflux-generate-qwen-edit",
        "--model",
        "AbstractFramework/qwen-image-edit-2511-8bit",
        "--image-paths",
        "input.png",
        "style.png",
        "--config-from-metadata",
        str(metadata_path),
    ]


def test_edit_model_requires_input_image():
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(["--model", "qwen-image-edit", "--prompt", "change the scene"])


def test_unknown_model_requires_supported_family():
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(["--model", "unknown/model", "--prompt", "hello"])


def test_routes_ernie_image_turbo_to_ernie_generation():
    invocation = mlx_gen._resolve_invocation(["--model", "baidu/ERNIE-Image-Turbo", "--prompt", "hello"])

    assert invocation.target_name == "mflux-generate-ernie-image"
    assert invocation.argv == [
        "mflux-generate-ernie-image",
        "--model",
        "baidu/ERNIE-Image-Turbo",
        "--prompt",
        "hello",
    ]


def test_ernie_family_override_local_folder_requires_base_model(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "../models/custom-folder",
                "--family",
                "ernie-image",
                "--prompt",
                "hello",
            ]
        )

    assert "--family ernie-image is not enough to configure model path" in capsys.readouterr().err


def test_ernie_family_override_routes_local_folder():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "../models/custom-ernie-folder",
            "--family",
            "ernie-image",
            "--base-model",
            "ernie-image-turbo",
            "--prompt",
            "hello",
        ]
    )

    assert invocation.target_name == "mflux-generate-ernie-image"
    assert invocation.argv == [
        "mflux-generate-ernie-image",
        "--model",
        "../models/custom-ernie-folder",
        "--base-model",
        "ernie-image-turbo",
        "--prompt",
        "hello",
    ]


def test_ernie_family_override_routes_local_folder_with_image():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "../models/custom-folder",
            "--family",
            "ernie-image",
            "--base-model",
            "ernie-image-turbo",
            "--image",
            "input.png",
            "--image-strength",
            "0.4",
            "--prompt",
            "make it cinematic",
        ]
    )

    assert invocation.target_name == "mflux-generate-ernie-image"
    assert invocation.argv == [
        "mflux-generate-ernie-image",
        "--model",
        "../models/custom-folder",
        "--base-model",
        "ernie-image-turbo",
        "--image-path",
        "input.png",
        "--image-strength",
        "0.4",
        "--prompt",
        "make it cinematic",
    ]


def test_routes_ernie_image_input_to_image_to_image_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "baidu/ERNIE-Image-Turbo",
            "--image",
            "input.png",
            "--image-strength",
            "0.4",
            "--prompt",
            "make it cinematic",
        ]
    )

    assert invocation.target_name == "mflux-generate-ernie-image"
    assert invocation.argv == [
        "mflux-generate-ernie-image",
        "--model",
        "baidu/ERNIE-Image-Turbo",
        "--image-path",
        "input.png",
        "--image-strength",
        "0.4",
        "--prompt",
        "make it cinematic",
    ]


def test_ernie_image_to_image_task_requires_an_image(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "baidu/ERNIE-Image-Turbo",
                "--task",
                "image-to-image",
                "--prompt",
                "make it cinematic",
            ]
        )

    assert "image-to-image requires --image" in capsys.readouterr().err


def test_ernie_rejects_multiple_image_inputs(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "baidu/ERNIE-Image-Turbo",
                "--images",
                "input.png",
                "style.png",
                "--prompt",
                "make it cinematic",
            ]
        )

    assert "does not support multi-reference image-to-image generation" in capsys.readouterr().err


def test_ernie_rejects_edit_task(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "baidu/ERNIE-Image-Turbo",
                "--task",
                "edit",
                "--image",
                "input.png",
                "--prompt",
                "replace the mug",
            ]
        )

    assert "does not support edit-reference image-to-image generation" in capsys.readouterr().err


def test_ernie_cli_passes_prompt_enhancer_options(monkeypatch):
    from mflux.models.ernie_image.cli import ernie_image_generate

    observed = {}

    class FakeImage:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeErnieImageTurbo:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_image(self, **kwargs):
            observed["generate"] = kwargs
            return FakeImage()

    monkeypatch.setattr(ernie_image_generate, "ErnieImageTurbo", FakeErnieImageTurbo)
    monkeypatch.setattr(ernie_image_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-ernie-image",
            "--model",
            "baidu/ERNIE-Image-Turbo",
            "--prompt",
            "hello",
            "--image-path",
            "input.png",
            "--image-strength",
            "0.6",
            "--width",
            "512",
            "--height",
            "512",
            "--use-prompt-enhancer",
            "--prompt-enhancer-temperature",
            "0.7",
            "--prompt-enhancer-top-p",
            "0.9",
            "--prompt-enhancer-max-new-tokens",
            "12",
        ],
    )

    ernie_image_generate.main()

    assert observed["generate"]["use_pe"] is True
    assert observed["generate"]["image_path"].as_posix() == "input.png"
    assert observed["generate"]["image_strength"] == 0.6
    assert observed["generate"]["pe_temperature"] == 0.7
    assert observed["generate"]["pe_top_p"] == 0.9
    assert observed["generate"]["pe_max_new_tokens"] == 12


def test_ernie_cli_uses_selected_prepared_model_config(monkeypatch):
    from mflux.models.ernie_image.cli import ernie_image_generate

    observed = {}

    class FakeImage:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeErnieImageTurbo:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_image(self, **kwargs):
            observed["generate"] = kwargs
            return FakeImage()

    monkeypatch.setattr(ernie_image_generate, "ErnieImageTurbo", FakeErnieImageTurbo)
    monkeypatch.setattr(ernie_image_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-ernie-image",
            "--model",
            "AbstractFramework/ernie-image-turbo-8bit",
            "--prompt",
            "make it cinematic",
            "--image-path",
            "input.png",
            "--image-strength",
            "0.4",
            "--output",
            "out.png",
        ],
    )

    ernie_image_generate.main()

    model_config = observed["init"]["model_config"]
    assert model_config.model_name == "AbstractFramework/ernie-image-turbo-8bit"
    assert model_config.base_model == "baidu/ERNIE-Image-Turbo"
    assert observed["init"]["model_path"] is None


def test_z_image_turbo_cli_uses_selected_prepared_model_config(monkeypatch):
    from mflux.models.z_image.cli import z_image_turbo_generate

    observed = {}

    class FakeImage:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeZImage:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_image(self, **kwargs):
            observed["generate"] = kwargs
            return FakeImage()

    monkeypatch.setattr(z_image_turbo_generate, "ZImage", FakeZImage)
    monkeypatch.setattr(z_image_turbo_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-z-image-turbo",
            "--model",
            "AbstractFramework/z-image-turbo-8bit",
            "--prompt",
            "make it cinematic",
            "--image-path",
            "input.png",
            "--image-strength",
            "0.4",
            "--output",
            "out.png",
        ],
    )

    z_image_turbo_generate.main()

    model_config = observed["init"]["model_config"]
    assert model_config.model_name == "AbstractFramework/z-image-turbo-8bit"
    assert model_config.base_model == "Tongyi-MAI/Z-Image-Turbo"
    assert observed["init"]["model_path"] is None


def test_qwen_edit_backend_accepts_local_path_with_2511_base_model(monkeypatch):
    from mflux.models.qwen.cli import qwen_image_edit_generate

    observed = {}

    class FakeImage:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeQwenImageEdit:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_image(self, **kwargs):
            observed["generate"] = kwargs
            return FakeImage()

    monkeypatch.setattr(qwen_image_edit_generate, "QwenImageEdit", FakeQwenImageEdit)
    monkeypatch.setattr(qwen_image_edit_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-qwen-edit",
            "--model",
            "../models/local-qwen-edit",
            "--base-model",
            "qwen-image-edit-2511",
            "--prompt",
            "replace the chair",
            "--image-paths",
            "input.png",
            "style.png",
            "--output",
            "out.png",
        ],
    )

    qwen_image_edit_generate.main()

    model_config = observed["init"]["model_config"]
    assert model_config.model_name == "../models/local-qwen-edit"
    assert model_config.base_model == "Qwen/Qwen-Image-Edit-2511"
    assert observed["init"]["model_path"] == "../models/local-qwen-edit"
    assert observed["generate"]["image_paths"] == ["input.png", "style.png"]
    assert observed["generate"]["negative_prompt"] is None
    assert observed["generate"]["guidance"] == 4.0
    assert observed["generate"]["scheduler"] == "flow_match_euler_discrete"
    assert observed["generate"]["num_inference_steps"] == 40
    assert observed["save"]["path"] == Path("out.png")


def test_qwen_edit_backend_preserves_explicit_scheduler(monkeypatch):
    from mflux.models.qwen.cli import qwen_image_edit_generate

    observed = {}

    class FakeImage:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeQwenImageEdit:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_image(self, **kwargs):
            observed["generate"] = kwargs
            return FakeImage()

    monkeypatch.setattr(qwen_image_edit_generate, "QwenImageEdit", FakeQwenImageEdit)
    monkeypatch.setattr(qwen_image_edit_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-qwen-edit",
            "--model",
            "qwen-image-edit",
            "--prompt",
            "replace the chair",
            "--image-paths",
            "input.png",
            "--scheduler",
            "linear",
            "--output",
            "out.png",
        ],
    )

    qwen_image_edit_generate.main()

    assert observed["generate"]["scheduler"] == "linear"


def test_qwen_edit_backend_accepts_negative_alias(monkeypatch):
    from mflux.models.qwen.cli import qwen_image_edit_generate

    observed = {}

    class FakeImage:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeQwenImageEdit:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_image(self, **kwargs):
            observed["generate"] = kwargs
            return FakeImage()

    monkeypatch.setattr(qwen_image_edit_generate, "QwenImageEdit", FakeQwenImageEdit)
    monkeypatch.setattr(qwen_image_edit_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-qwen-edit",
            "--model",
            "qwen-image-edit",
            "--prompt",
            "replace the chair",
            "--image-paths",
            "input.png",
            "--negative",
            "cropped, blurry",
            "--output",
            "out.png",
        ],
    )

    qwen_image_edit_generate.main()

    assert observed["generate"]["negative_prompt"] == "cropped, blurry"


def test_qwen_edit_backend_rejects_multi_reference_without_edit_plus(monkeypatch, capsys):
    from mflux.models.qwen.cli import qwen_image_edit_generate

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-qwen-edit",
            "--model",
            "qwen-image-edit",
            "--prompt",
            "combine these references",
            "--image-paths",
            "input.png",
            "style.png",
            "--output",
            "out.png",
        ],
    )

    with pytest.raises(SystemExit):
        qwen_image_edit_generate.main()

    assert "Multiple Qwen edit reference images require an Edit-Plus model" in capsys.readouterr().err


def test_routes_wan_text_to_video_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "--prompt",
            "a city timelapse",
            "--frames",
            "5",
            "--fps",
            "8",
            "--flow-shift",
            "3",
        ]
    )

    assert invocation.target_name == "mlxgen-generate-wan"
    assert invocation.argv == [
        "mlxgen-generate-wan",
        "--model",
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "--prompt",
        "a city timelapse",
        "--frames",
        "5",
        "--fps",
        "8",
        "--flow-shift",
        "3",
    ]


def test_routes_wan_image_to_video_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "wan2.2-ti2v-5b",
            "--image",
            "input.png",
            "--prompt",
            "make the room slowly brighten",
        ]
    )

    assert invocation.target_name == "mlxgen-generate-wan"
    assert "--image-path" in invocation.argv
    assert "input.png" in invocation.argv


def test_wan_i2v_a14b_requires_image_before_model_load(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "--prompt",
                "make a video",
            ]
        )

    assert "image-to-video model requires --image" in capsys.readouterr().err


def test_wan_t2v_a14b_rejects_image_before_model_load(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "--image",
                "input.png",
                "--prompt",
                "make a video",
            ]
        )

    assert "text-to-video model does not accept input images" in capsys.readouterr().err


def test_wan_router_rejects_generic_local_wan_name_before_backend_load(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "models/my-wan-video-folder",
                "--image",
                "input.png",
                "--prompt",
                "make a video",
            ]
        )

    assert "Cannot infer a supported Wan model config" in capsys.readouterr().err


def test_wan_rejects_multiple_images(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "wan2.2-ti2v-5b",
                "--images",
                "input.png",
                "style.png",
                "--prompt",
                "make a video",
            ]
        )

    assert "accepts exactly one input image" in capsys.readouterr().err


def test_wan_cli_generates_video_and_respects_replace(monkeypatch, tmp_path):
    from mflux.models.wan.cli import wan_generate

    observed = {}
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"fake")

    class FakeVideo:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeWan:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "Wan2_2_TI2V", FakeWan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "--prompt",
            "a city timelapse",
            "--width",
            "128",
            "--height",
            "128",
            "--frames",
            "5",
            "--fps",
            "8",
            "--steps",
            "2",
            "--guidance-2",
            "1.5",
            "--flow-shift",
            "3",
            "--solver",
            "euler",
            "--seed",
            "123",
            "--image-path",
            str(image_path),
            "--replace",
            "false",
            "--output",
            "out.mp4",
        ],
    )

    wan_generate.main()

    assert observed["init"]["quantize"] is None
    assert observed["init"]["model_config"] is wan_generate.ModelConfig.wan2_2_i2v_a14b()
    assert observed["generate"]["prompt"] == "a city timelapse"
    assert observed["generate"]["width"] == 128
    assert observed["generate"]["height"] == 128
    assert observed["generate"]["num_frames"] == 5
    assert observed["generate"]["fps"] == 8
    assert observed["generate"]["num_inference_steps"] == 2
    assert observed["generate"]["guidance_2"] == 1.5
    assert observed["generate"]["flow_shift"] == 3.0
    assert observed["generate"]["solver"] == "euler"
    assert observed["generate"]["seed"] == 123
    assert observed["generate"]["image_path"] == str(image_path)
    assert callable(observed["generate"]["progress_callback"])
    assert observed["generate"]["release_inactive_denoiser"] is True
    assert observed["generate"]["clear_cache_each_step"] is False
    assert observed["generate"]["clear_cache_each_transformer_block"] is False
    assert observed["generate"]["tensor_health_check_interval"] is None
    assert observed["save"]["path"] == "out.mp4"
    assert observed["save"]["overwrite"] is False


def test_wan_cli_can_disable_progress(monkeypatch):
    from mflux.models.wan.cli import wan_generate

    observed = {}

    class FakeVideo:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeWan:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "Wan2_2_TI2V", FakeWan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "--prompt",
            "a city timelapse",
            "--seed",
            "123",
            "--no-progress",
        ],
    )

    wan_generate.main()

    assert observed["generate"]["progress_callback"] is None


def test_wan_cli_rejects_disabled_tensor_health_checks(monkeypatch):
    from mflux.models.wan.cli import wan_generate

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "--prompt",
            "a city timelapse",
            "--seed",
            "123",
            "--tensor-health-check-interval",
            "0",
        ],
    )

    with pytest.raises(SystemExit):
        wan_generate.main()


def test_wan_cli_writes_failure_manifest(monkeypatch, tmp_path):
    from mflux.models.wan.cli import wan_generate

    output_path = tmp_path / "failed.mp4"

    class FakeWan:
        def __init__(self, **kwargs):
            pass

        def generate_video(self, **kwargs):
            raise ValueError("synthetic tensor failure")

    monkeypatch.setattr(wan_generate, "Wan2_2_TI2V", FakeWan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "--prompt",
            "a city timelapse",
            "--seed",
            "123",
            "--output",
            str(output_path),
            "--failure-diagnostics",
            "--no-progress",
        ],
    )

    with pytest.raises(SystemExit):
        wan_generate.main()

    manifest = json.loads(output_path.with_suffix(".failure.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "ValueError"
    assert manifest["error"] == "synthetic tensor failure"
    assert manifest["run"]["prompt"] == "a city timelapse"
    assert manifest["run"]["seed"] == 123
    assert manifest["run"]["output"] == str(output_path)
    assert manifest["run"]["failure_diagnostics"] is True
    assert "runtime_diagnostics" in manifest
    assert "mlx_peak_memory_bytes" in manifest["runtime_diagnostics"]


def test_routes_wan_failure_diagnostics_to_backend():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--prompt",
            "a city timelapse",
            "--failure-diagnostics",
        ]
    )

    assert invocation.target_name == "mlxgen-generate-wan"
    assert "--failure-diagnostics" in invocation.argv


def test_wan_cli_low_ram_releases_denoisers_before_decode_and_sets_cache_limit(monkeypatch):
    from mflux.models.wan.cli import wan_generate

    observed = {"cache_limit": None, "cache_cleared": False, "peak_reset": False}

    class FakeVideo:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeWan:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "Wan2_2_TI2V", FakeWan)
    monkeypatch.setattr(wan_generate.mx, "set_cache_limit", lambda value: observed.update(cache_limit=value))
    monkeypatch.setattr(wan_generate.mx, "clear_cache", lambda: observed.update(cache_cleared=True))
    monkeypatch.setattr(wan_generate.mx, "reset_peak_memory", lambda: observed.update(peak_reset=True))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "--prompt",
            "a city timelapse",
            "--seed",
            "123",
            "--image-path",
            "input.png",
            "--low-ram",
            "--mlx-cache-limit-gb",
            "2.5",
            "--no-progress",
        ],
    )
    monkeypatch.setattr(wan_generate.Path, "exists", lambda self: True)

    wan_generate.main()

    assert observed["cache_limit"] == int(2.5 * (1000**3))
    assert observed["cache_cleared"] is True
    assert observed["peak_reset"] is True
    assert observed["generate"]["release_inactive_denoiser"] is True
    assert observed["generate"]["release_denoisers_before_decode"] is True
    assert observed["generate"]["clear_cache_each_step"] is True
    assert observed["generate"]["clear_cache_each_transformer_block"] is True


def test_wan_cli_cache_limit_without_low_ram_sets_allocator_only(monkeypatch):
    from mflux.models.wan.cli import wan_generate

    observed = {"cache_limit": None, "cache_cleared": False, "peak_reset": False}

    class FakeVideo:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeWan:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "Wan2_2_TI2V", FakeWan)
    monkeypatch.setattr(wan_generate.mx, "set_cache_limit", lambda value: observed.update(cache_limit=value))
    monkeypatch.setattr(wan_generate.mx, "clear_cache", lambda: observed.update(cache_cleared=True))
    monkeypatch.setattr(wan_generate.mx, "reset_peak_memory", lambda: observed.update(peak_reset=True))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--prompt",
            "a city timelapse",
            "--seed",
            "123",
            "--mlx-cache-limit-gb",
            "2.5",
            "--no-progress",
        ],
    )

    wan_generate.main()

    assert observed["cache_limit"] == int(2.5 * (1000**3))
    assert observed["cache_cleared"] is True
    assert observed["peak_reset"] is True
    assert observed["generate"]["release_inactive_denoiser"] is True
    assert observed["generate"]["release_denoisers_before_decode"] is False
    assert observed["generate"]["clear_cache_each_step"] is False
    assert observed["generate"]["clear_cache_each_transformer_block"] is False


def test_wan_cli_low_ram_defaults_cache_limit(monkeypatch):
    from mflux.models.wan.cli import wan_generate

    observed = {}

    class FakeVideo:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeWan:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "Wan2_2_TI2V", FakeWan)
    monkeypatch.setattr(wan_generate.mx, "set_cache_limit", lambda value: observed.update(cache_limit=value))
    monkeypatch.setattr(wan_generate.mx, "clear_cache", lambda: None)
    monkeypatch.setattr(wan_generate.mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "--prompt",
            "a city timelapse",
            "--seed",
            "123",
            "--image-path",
            "input.png",
            "--low-ram",
            "--no-progress",
        ],
    )
    monkeypatch.setattr(wan_generate.Path, "exists", lambda self: True)

    wan_generate.main()

    assert observed["cache_limit"] == 1000**3


def test_wan_cli_multiple_seeds_keep_denoisers_for_reuse(monkeypatch):
    from mflux.models.wan.cli import wan_generate

    observed = {"generate": []}

    class FakeVideo:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeWan:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_video(self, **kwargs):
            observed["generate"].append(kwargs)
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "Wan2_2_TI2V", FakeWan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "--prompt",
            "a city timelapse",
            "--seed",
            "123",
            "456",
            "--image-path",
            "input.png",
            "--no-progress",
        ],
    )
    monkeypatch.setattr(wan_generate.Path, "exists", lambda self: True)

    wan_generate.main()

    assert len(observed["generate"]) == 2
    assert all(call["release_inactive_denoiser"] is False for call in observed["generate"])
    assert all(call["release_denoisers_before_decode"] is False for call in observed["generate"])
    assert all(call["clear_cache_each_transformer_block"] is False for call in observed["generate"])


def test_wan_cli_progress_advances_by_denoise_steps(monkeypatch):
    from mflux.callbacks import ProgressEvent
    from mflux.models.wan.cli import wan_generate

    bars = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.updates = []
            self.postfixes = []
            self.closed = False
            bars.append(self)

        def update(self, delta):
            self.updates.append(delta)

        def set_postfix_str(self, text):
            self.postfixes.append(text)

        def close(self):
            self.closed = True

        @staticmethod
        def set_lock(lock):
            pass

    monkeypatch.setattr(wan_generate, "tqdm", FakeTqdm)
    wan_generate._WanCliProgress._lock_configured = False
    try:
        progress = wan_generate._WanCliProgress(enabled=True)
        progress(ProgressEvent(phase="start", frame=0, total_frames=81, step=0, total_steps=20))
        progress(ProgressEvent(phase="denoise", frame=24, total_frames=81, step=6, total_steps=20))
        progress(ProgressEvent(phase="denoise", frame=28, total_frames=81, step=7, total_steps=20))
        progress(ProgressEvent(phase="generated", frame=81, total_frames=81, step=20, total_steps=20))
        assert bars[0].closed is False
        progress(ProgressEvent(phase="save", frame=81, total_frames=81, step=20, total_steps=20))
        assert bars[0].closed is False
        progress(ProgressEvent(phase="complete", frame=81, total_frames=81, step=20, total_steps=20))
    finally:
        wan_generate._WanCliProgress._lock_configured = False

    assert bars[0].kwargs == {"total": 20, "desc": "Denoising video", "unit": "step"}
    assert bars[0].updates == [6, 1, 13]
    assert bars[0].postfixes[-1] == "complete; 81 frames"
    assert bars[0].closed is True


def test_wan_progress_uses_thread_only_tqdm_lock(monkeypatch):
    from mflux.models.wan.cli import wan_generate

    locks = []
    monkeypatch.setattr(wan_generate.tqdm, "set_lock", locks.append)
    wan_generate._WanCliProgress._lock_configured = False
    try:
        wan_generate._WanCliProgress(enabled=True)
    finally:
        wan_generate._WanCliProgress._lock_configured = False

    assert len(locks) == 1
    assert hasattr(locks[0], "acquire")
    assert hasattr(locks[0], "release")


def test_wan_cli_applies_a14b_defaults(monkeypatch):
    from mflux.models.wan.cli import wan_generate

    observed = {}

    class FakeVideo:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeWan:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "Wan2_2_TI2V", FakeWan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--prompt",
            "a cinematic wave",
            "--seed",
            "123",
            "--no-progress",
        ],
    )

    wan_generate.main()

    assert observed["init"]["model_config"] is wan_generate.ModelConfig.wan2_2_t2v_a14b()
    assert observed["init"]["model_path"] is None
    assert observed["generate"]["width"] == 1280
    assert observed["generate"]["height"] == 720
    assert observed["generate"]["num_frames"] == 81
    assert observed["generate"]["fps"] == 16
    assert observed["generate"]["num_inference_steps"] == 40
    assert observed["generate"]["guidance"] == 4.0
    assert observed["generate"]["guidance_2"] == 3.0
    assert "低质量" in observed["generate"]["negative_prompt"]


def test_wan_cli_applies_ti2v_5b_defaults(monkeypatch):
    from mflux.models.wan.cli import wan_generate

    observed = {}

    class FakeVideo:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeWan:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "Wan2_2_TI2V", FakeWan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "--prompt",
            "a city timelapse",
            "--seed",
            "123",
            "--no-progress",
        ],
    )

    wan_generate.main()

    assert observed["init"]["model_config"] is wan_generate.ModelConfig.wan2_2_ti2v_5b()
    assert observed["generate"]["width"] == 1280
    assert observed["generate"]["height"] == 704
    assert observed["generate"]["num_frames"] == 121
    assert observed["generate"]["fps"] == 24
    assert observed["generate"]["num_inference_steps"] == 50
    assert observed["generate"]["guidance"] == 5.0
    assert observed["generate"]["guidance_2"] is None
    assert "低质量" in observed["generate"]["negative_prompt"]


def test_wan_cli_forwards_lora_args_to_model_init(monkeypatch):
    from mflux.models.wan.cli import wan_generate

    observed = {}

    class FakeVideo:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeWan:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "Wan2_2_TI2V", FakeWan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--prompt",
            "a cinematic wave",
            "--seed",
            "123",
            "--lora-paths",
            "high.safetensors",
            "low.safetensors",
            "--lora-scales",
            "1.0",
            "0.8",
            "--lora-target-roles",
            "high_noise_transformer",
            "low_noise_transformer",
            "--no-progress",
        ],
    )

    wan_generate.main()

    assert observed["init"]["lora_paths"] == ["high.safetensors", "low.safetensors"]
    assert observed["init"]["lora_scales"] == [1.0, 0.8]
    assert observed["init"]["lora_target_roles"] == ["high_noise_transformer", "low_noise_transformer"]


def test_router_forwards_wan_lora_target_roles():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "wan2.2-t2v-a14b",
            "--prompt",
            "a cinematic wave",
            "--lora-paths",
            "high.safetensors",
            "low.safetensors",
            "--lora-target-roles",
            "high_noise_transformer",
            "low_noise_transformer",
        ]
    )

    assert invocation.target_name == "mlxgen-generate-wan"
    assert "--lora-target-roles" in invocation.argv
    assert invocation.argv[-2:] == ["high_noise_transformer", "low_noise_transformer"]


def test_wan_cli_explicit_guidance_keeps_guidance_2_diffusers_default(monkeypatch):
    from mflux.models.wan.cli import wan_generate

    observed = {}

    class FakeVideo:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeWan:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "Wan2_2_TI2V", FakeWan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--prompt",
            "a cinematic wave",
            "--guidance",
            "4.5",
            "--seed",
            "123",
            "--no-progress",
        ],
    )

    wan_generate.main()

    assert observed["generate"]["guidance"] == 4.5
    assert observed["generate"]["guidance_2"] is None


def test_wan_cli_negative_alias_disables_default_negative_prompt(monkeypatch):
    from mflux.models.wan.cli import wan_generate

    observed = {}

    class FakeVideo:
        def save(self, **kwargs):
            observed["save"] = kwargs

    class FakeWan:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "Wan2_2_TI2V", FakeWan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "--prompt",
            "a city timelapse",
            "--negative",
            "",
            "--seed",
            "123",
            "--no-progress",
        ],
    )

    wan_generate.main()

    assert observed["generate"]["negative_prompt"] == ""


def test_wan_cli_rejects_unrecognized_remote_wan_repo():
    from mflux.models.wan.cli import wan_generate

    with pytest.raises(wan_generate.ModelConfigError, match="Cannot infer a supported Wan model config"):
        wan_generate._resolve_model("Wan-AI/Wan2.2-Unknown-14B-Diffusers")


def test_wan_cli_rejects_generic_local_wan_inference():
    from mflux.models.wan.cli import wan_generate

    with pytest.raises(wan_generate.ModelConfigError, match="Cannot infer a supported Wan model config"):
        wan_generate._resolve_model("models/my-wan-video-folder")


def test_wan_cli_accepts_specific_local_wan_alias():
    from mflux.models.wan.cli import wan_generate

    model_config, model_path = wan_generate._resolve_model("models/wan2.2-ti2v-5b-8bit")

    assert model_config.base_model == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert model_path == "models/wan2.2-ti2v-5b-8bit"


def test_wan_router_target_has_console_scripts():
    pyproject = toml.load(Path(__file__).parents[2] / "pyproject.toml")
    scripts = pyproject["project"]["scripts"]

    assert scripts["mlxgen-generate-wan"] == "mflux.models.wan.cli.wan_generate:main"
    assert scripts["mflux-generate-wan"] == "mflux.models.wan.cli.wan_generate:main"


def test_main_without_args_prints_top_level_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mlxgen"])

    mlx_gen.main()

    output = capsys.readouterr().out
    assert "usage: mlxgen" in output
    assert "mlxgen generate" in output
    assert "mlxgen upscale" in output
    assert "mlxgen capabilities" in output
    assert "mlxgen download" in output
    assert "mlxgen prepare" in output


def test_main_restores_sys_argv(monkeypatch):
    observed = []
    original_argv = ["mlx-gen", "--model", "qwen-image", "--prompt", "x"]
    routed_argv = ["mflux-generate-qwen", "--model", "qwen-image", "--prompt", "x"]

    def fake_main():
        observed.append(sys.argv[:])

    monkeypatch.setattr(
        mlx_gen,
        "_resolve_invocation",
        lambda argv: RouterInvocation("mflux-generate-qwen", fake_main, routed_argv),
    )

    monkeypatch.setattr(sys, "argv", original_argv[:])
    mlx_gen.main()

    assert observed == [routed_argv]
    assert sys.argv == original_argv


def test_main_prints_download_hint_without_traceback(monkeypatch, capsys):
    def fake_main():
        raise FileNotFoundError("MLX-Gen will not download model files during generation.")

    monkeypatch.setattr(
        mlx_gen,
        "_resolve_invocation",
        lambda argv: RouterInvocation("mflux-generate-qwen", fake_main, ["mflux-generate-qwen"]),
    )
    monkeypatch.setattr(sys, "argv", ["mlxgen", "generate", "--model", "Qwen/Qwen-Image"])

    with pytest.raises(SystemExit) as exc_info:
        mlx_gen.main()

    assert exc_info.value.code == 1
    assert "MLX-Gen will not download model files during generation" in capsys.readouterr().out


def test_download_command_enables_downloads_temporarily(monkeypatch, capsys):
    calls = []

    def fake_snapshot_download(*, repo_id, allow_patterns):
        calls.append((repo_id, allow_patterns, downloads_enabled()))
        return "/tmp/hf-cache/snapshot"

    monkeypatch.setattr(mlx_gen, "snapshot_download", fake_snapshot_download)

    mlx_gen._download_model(["--model", "Qwen/Qwen-Image"])

    assert calls
    assert calls[0][0] == "Qwen/Qwen-Image"
    assert calls[0][1] is not None
    assert calls[0][2] is True
    assert downloads_enabled() is False
    assert "mlxgen generate --model" in capsys.readouterr().out


def test_download_command_uses_ernie_source_patterns(monkeypatch):
    calls = []

    def fake_snapshot_download(*, repo_id, allow_patterns):
        calls.append((repo_id, allow_patterns, downloads_enabled()))
        return "/tmp/hf-cache/ernie"

    monkeypatch.setattr(mlx_gen, "snapshot_download", fake_snapshot_download)

    mlx_gen._download_model(["--model", "baidu/ERNIE-Image-Turbo"])

    assert calls[0][0] == "baidu/ERNIE-Image-Turbo"
    assert calls[0][2] is True
    assert {
        "README.md",
        "tokenizer/*",
        "text_encoder/*.safetensors",
        "transformer/*.safetensors",
        "vae/*.safetensors",
    }.issubset(set(calls[0][1]))
    assert "pe/*.safetensors" not in calls[0][1]


def test_download_command_uses_a14b_wan_patterns(monkeypatch):
    calls = []

    def fake_snapshot_download(*, repo_id, allow_patterns):
        calls.append((repo_id, allow_patterns, downloads_enabled()))
        return "/tmp/hf-cache/wan-a14b"

    monkeypatch.setattr(mlx_gen, "snapshot_download", fake_snapshot_download)

    mlx_gen._download_model(["--model", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"])

    assert calls[0][0] == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert calls[0][2] is True
    assert "transformer_2/*.safetensors" in calls[0][1]
    assert "transformer_2/*.json" in calls[0][1]


def test_prepare_command_routes_to_save_with_downloads_enabled(monkeypatch):
    observed = []

    def fake_save_main():
        observed.append((sys.argv[:], downloads_enabled()))

    monkeypatch.setattr("mflux.models.common.cli.save.main", fake_save_main)

    mlx_gen._prepare_model(["--model", "Qwen/Qwen-Image", "--path", "../models/qwen-image-8bit", "-q", "8"])

    assert observed == [
        (
            ["mlxgen prepare", "--model", "Qwen/Qwen-Image", "--path", "../models/qwen-image-8bit", "-q", "8"],
            True,
        )
    ]
    assert downloads_enabled() is False


def test_prepare_ernie_routes_to_save_with_downloads_enabled(monkeypatch, tmp_path):
    observed = []

    def fake_save_main():
        observed.append((sys.argv[:], downloads_enabled()))

    monkeypatch.setattr("mflux.models.common.cli.save.main", fake_save_main)

    mlx_gen._prepare_model(["--model", "baidu/ERNIE-Image-Turbo", "--path", str(tmp_path / "ernie-image-turbo")])

    assert observed == [
        (
            ["mlxgen prepare", "--model", "baidu/ERNIE-Image-Turbo", "--path", str(tmp_path / "ernie-image-turbo")],
            True,
        )
    ]
    assert downloads_enabled() is False


def test_depth_pro_download_command_enables_direct_url_download(monkeypatch, capsys, tmp_path):
    calls = []

    def fake_download_from_url(url, component_name):
        calls.append((url, component_name, downloads_enabled()))
        return tmp_path / component_name / "depth_pro.pt"

    monkeypatch.setattr(
        "mflux.models.common.weights.loading.weight_loader.WeightLoader._download_from_url", fake_download_from_url
    )

    mlx_gen._download_model(["--model", "depth-pro"])

    assert calls == [
        (
            "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt",
            "depth_pro",
            True,
        )
    ]
    assert downloads_enabled() is False
    assert "Downloaded Depth Pro weights" in capsys.readouterr().out
