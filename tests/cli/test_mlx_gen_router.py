import json
import sys

import pytest

from mflux.cli import mlx_gen
from mflux.cli.mlx_gen import RouterInvocation
from mflux.models.common.download_policy import downloads_enabled


def test_routes_qwen_image_to_text_or_image_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "Qwen/Qwen-Image",
            "--image",
            "input.png",
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
        "--prompt",
        "make it cinematic",
    ]


def test_routes_qwen_multiple_images_to_edit_generation():
    invocation = mlx_gen._resolve_invocation(
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

    assert invocation.target_name == "mflux-generate-qwen-edit"
    assert invocation.argv == [
        "mflux-generate-qwen-edit",
        "--model",
        "qwen-image-edit",
        "--image-paths",
        "input.png",
        "style.png",
        "--prompt",
        "apply the second image style",
    ]


def test_routes_qwen_edit_model_to_edit_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "lpalbou/qwen-image-edit-2511-4bit",
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
        "lpalbou/qwen-image-edit-2511-4bit",
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


def test_routes_flux2_with_image_to_edit_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "lpalbou/flux2-klein-4b-4bit",
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
        "lpalbou/flux2-klein-4b-4bit",
        "--image-paths",
        "input.png",
        "--prompt",
        "add sunglasses",
    ]


def test_routes_flux2_with_image_strength_to_image_to_image_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "lpalbou/flux2-klein-4b-4bit",
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
        "lpalbou/flux2-klein-4b-4bit",
        "--image-path",
        "input.png",
        "--image-strength",
        "0.4",
        "--prompt",
        "make it cinematic",
    ]


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


def test_task_edit_can_select_qwen_edit_from_qwen_base_model():
    invocation = mlx_gen._resolve_invocation(
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

    assert invocation.target_name == "mflux-generate-qwen-edit"
    assert invocation.argv == [
        "mflux-generate-qwen-edit",
        "--model",
        "qwen-image-edit",
        "--image-paths",
        "input.png",
        "style.png",
        "--prompt",
        "apply the style from the second image",
    ]


def test_family_override_routes_unidentifiable_local_path():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "../models/local-edit-checkpoint",
            "--family",
            "qwen",
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
        "--image-paths",
        "input.png",
        "--prompt",
        "replace the chair",
    ]


def test_metadata_can_supply_model_and_images(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": "lpalbou/qwen-image-edit-2511-4bit",
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
        "lpalbou/qwen-image-edit-2511-4bit",
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


def test_main_without_args_prints_top_level_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mlxgen"])

    mlx_gen.main()

    output = capsys.readouterr().out
    assert "usage: mlxgen" in output
    assert "mlxgen generate" in output
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
