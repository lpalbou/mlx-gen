import json
import sys
from pathlib import Path

import pytest
import toml

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


def test_ernie_family_override_routes_local_folder():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "../models/custom-ernie-folder",
            "--family",
            "ernie-image",
            "--prompt",
            "hello",
        ]
    )

    assert invocation.target_name == "mflux-generate-ernie-image"
    assert invocation.argv == [
        "mflux-generate-ernie-image",
        "--model",
        "../models/custom-ernie-folder",
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
            "--image",
            "input.png",
            "--prompt",
            "make it cinematic",
        ]
    )

    assert invocation.target_name == "mflux-generate-ernie-image"
    assert invocation.argv == [
        "mflux-generate-ernie-image",
        "--model",
        "../models/custom-folder",
        "--image-path",
        "input.png",
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

    assert "supports only one input image" in capsys.readouterr().err


def test_ernie_rejects_edit_task(capsys):
    with pytest.raises(SystemExit):
        mlx_gen._resolve_invocation(
            [
                "--model",
                "baidu/ERNIE-Image-Turbo",
                "--task",
                "edit",
                "--prompt",
                "replace the mug",
            ]
        )

    assert "Multi-image edit is not supported" in capsys.readouterr().err


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


def test_routes_wan_text_to_video_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "--task",
            "text-to-video",
            "--prompt",
            "a city timelapse",
            "--frames",
            "5",
            "--fps",
            "8",
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
    ]


def test_routes_wan_image_to_video_generation():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            "wan2.2-ti2v-5b",
            "--task",
            "image-to-video",
            "--image",
            "input.png",
            "--prompt",
            "make the room slowly brighten",
        ]
    )

    assert invocation.target_name == "mlxgen-generate-wan"
    assert "--image-path" in invocation.argv
    assert "input.png" in invocation.argv


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
    assert observed["generate"]["seed"] == 123
    assert observed["generate"]["image_path"] == str(image_path)
    assert callable(observed["generate"]["progress_callback"])
    assert observed["generate"]["release_inactive_denoiser"] is True
    assert observed["generate"]["clear_cache_each_step"] is False
    assert observed["generate"]["tensor_health_check_interval"] == 1
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
        "LICENSE",
        "README.md",
        "model_index.json",
        "scheduler/*.json",
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
