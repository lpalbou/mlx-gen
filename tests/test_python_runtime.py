import importlib
import types
from pathlib import Path

import pytest

from mflux import (
    load_generation_model,
    load_generation_model_for_plan,
    resolve_generation_plan,
    resolve_generation_runtime,
    resolve_generation_runtime_for_plan,
)
from mflux.callbacks import ProgressEvent
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.task_inference import GenerationPlan


def test_resolve_generation_runtime_selects_qwen_controlnet(monkeypatch):
    created = {}
    original_import = importlib.import_module

    class FakeQwenControlNet:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

    def fake_import(name, package=None):
        if name == "mflux.models.qwen.variants.controlnet.qwen_image_controlnet":
            return types.SimpleNamespace(QwenImageControlNet=FakeQwenControlNet)
        return original_import(name, package)

    monkeypatch.setattr("mflux.python_runtime.importlib.import_module", fake_import)

    runtime = resolve_generation_runtime(model="AbstractFramework/qwen-image-8bit", has_control_image=True)
    model = runtime.load(quantize=8, model_path="prepared/qwen-image")

    assert isinstance(model, FakeQwenControlNet)
    assert runtime.runtime_id == "qwen.controlnet"
    assert runtime.plan.control_model is not None
    assert created["kwargs"]["controlnet_model"] == runtime.plan.control_model
    assert created["kwargs"]["quantize"] == 8
    assert created["kwargs"]["model_path"] == "prepared/qwen-image"
    assert "prepared/qwen-image" in runtime.cache_key(model_path="prepared/qwen-image")


def test_resolve_generation_runtime_for_plan_selects_flux2_outpaint(monkeypatch):
    created = {}
    original_import = importlib.import_module

    class FakeFlux2Outpaint:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

    def fake_import(name, package=None):
        if name == "mflux.models.flux2.variants.edit.flux2_klein_outpaint":
            return types.SimpleNamespace(Flux2KleinOutpaint=FakeFlux2Outpaint)
        return original_import(name, package)

    monkeypatch.setattr("mflux.python_runtime.importlib.import_module", fake_import)

    plan = resolve_generation_plan(model="flux2-klein-base-4b", image_count=1, has_outpaint=True)
    runtime = resolve_generation_runtime_for_plan(plan=plan)
    model = runtime.load()

    assert isinstance(model, FakeFlux2Outpaint)
    assert runtime.runtime_id == "flux2.klein-outpaint"
    assert created["kwargs"]["model_config"].model_name == runtime.model_config.model_name


def test_resolve_generation_runtime_for_plan_selects_flux2_inpaint(monkeypatch):
    created = {}
    original_import = importlib.import_module

    class FakeFlux2Inpaint:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

    def fake_import(name, package=None):
        if name == "mflux.models.flux2.variants.edit.flux2_klein_inpaint":
            return types.SimpleNamespace(Flux2KleinInpaint=FakeFlux2Inpaint)
        return original_import(name, package)

    monkeypatch.setattr("mflux.python_runtime.importlib.import_module", fake_import)

    plan = resolve_generation_plan(model="flux2-klein-4b", image_count=1, has_mask=True)
    runtime = resolve_generation_runtime_for_plan(plan=plan)
    model = runtime.load()

    assert isinstance(model, FakeFlux2Inpaint)
    assert plan.capability_id == "flux2.inpaint"
    assert runtime.runtime_id == "flux2.klein-inpaint"
    assert created["kwargs"]["model_config"].model_name == runtime.model_config.model_name


def test_load_generation_model_returns_runtime_metadata_for_wan(monkeypatch):
    created = {}
    original_import = importlib.import_module

    class FakeWan:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

    def fake_import(name, package=None):
        if name == "mflux.models.wan.variants.wan2_2_ti2v":
            return types.SimpleNamespace(Wan2_2_TI2V=FakeWan)
        return original_import(name, package)

    monkeypatch.setattr("mflux.python_runtime.importlib.import_module", fake_import)

    loaded = load_generation_model(
        model="wan2.2-i2v-a14b",
        image_count=1,
        quantize=8,
        model_path="prepared/wan-a14b",
        lora_target_roles=["high_noise_transformer"],
    )

    assert isinstance(loaded.model, FakeWan)
    assert loaded.runtime_id == "wan2.2-ti2v"
    assert created["kwargs"]["lora_target_roles"] == ["high_noise_transformer"]
    assert "prepared/wan-a14b" in loaded.cache_key


def test_resolve_generation_runtime_loads_z_image_turbo_from_package_alias(monkeypatch):
    created = {}
    original_import = importlib.import_module

    class FakeZImageTurbo:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

    def fake_import(name, package=None):
        if name == "mflux.models.z_image.variants":
            return types.SimpleNamespace(ZImageTurbo=FakeZImageTurbo)
        return original_import(name, package)

    monkeypatch.setattr("mflux.python_runtime.importlib.import_module", fake_import)

    runtime = resolve_generation_runtime(model="z-image-turbo")
    model = runtime.load(model_path="prepared/z-image-turbo", quantize=8)

    assert isinstance(model, FakeZImageTurbo)
    assert runtime.runtime_id == "z-image-turbo"
    assert created["kwargs"]["model_path"] == "prepared/z-image-turbo"
    assert created["kwargs"]["quantize"] == 8


def test_non_wan_runtime_rejects_lora_target_roles():
    runtime = resolve_generation_runtime(model="z-image-turbo")

    with pytest.raises(ValueError, match="only supported for Wan runtimes"):
        runtime.load(lora_target_roles=["transformer"])


def test_loaded_generation_model_rejects_duplicate_seeds(monkeypatch):
    original_import = importlib.import_module

    class FakeZImageTurbo:
        def __init__(self, **kwargs):
            self.callbacks = CallbackRegistry()

    def fake_import(name, package=None):
        if name == "mflux.models.z_image.variants":
            return types.SimpleNamespace(ZImageTurbo=FakeZImageTurbo)
        return original_import(name, package)

    monkeypatch.setattr("mflux.python_runtime.importlib.import_module", fake_import)

    loaded = load_generation_model(model="z-image-turbo")

    with pytest.raises(ValueError, match="Duplicate seeds"):
        loaded.generate_outputs(
            seeds=[101, 101],
            prompt="test prompt",
            width=16,
            height=16,
            guidance=1.0,
            num_inference_steps=1,
        )


def test_loaded_generation_model_reuses_one_model_instance_across_serial_seeds(monkeypatch, tmp_path):
    created = {"instances": 0, "seeds": [], "saved_paths": []}
    original_import = importlib.import_module

    class FakeArtifact:
        def __init__(self, seed: int):
            self.seed = seed
            self.task = "text-to-image"

        def save(self, path, overwrite=True, **kwargs):
            created["saved_paths"].append((self.seed, Path(path), overwrite, dict(kwargs)))
            Path(path).write_text(f"seed={self.seed}")
            return Path(path)

    class FakeQwenImage:
        def __init__(self, **kwargs):
            created["instances"] += 1
            self.callbacks = CallbackRegistry()

        def generate_image(self, **kwargs):
            seed = kwargs["seed"]
            created["seeds"].append(seed)
            self.callbacks.emit_progress(ProgressEvent(task="text-to-image", phase="start", step=0, total_steps=2))
            self.callbacks.emit_progress(ProgressEvent(task="text-to-image", phase="denoise", step=1, total_steps=2))
            self.callbacks.emit_progress(ProgressEvent(task="text-to-image", phase="complete", step=2, total_steps=2))
            return FakeArtifact(seed)

    def fake_import(name, package=None):
        if name == "mflux.models.qwen.variants.txt2img.qwen_image":
            return types.SimpleNamespace(QwenImage=FakeQwenImage)
        return original_import(name, package)

    monkeypatch.setattr("mflux.python_runtime.importlib.import_module", fake_import)

    loaded = load_generation_model(model="qwen-image")
    progress_events = []
    results = loaded.generate_outputs(
        seeds=[101, 202],
        prompt="test prompt",
        width=16,
        height=16,
        guidance=1.0,
        num_inference_steps=2,
        output=tmp_path / "image.png",
        progress_callback=progress_events.append,
        save_kwargs={"export_json_metadata": False, "embed_metadata": False},
    )

    assert created["instances"] == 1
    assert created["seeds"] == [101, 202]
    assert [result.seed for result in results] == [101, 202]
    assert [result.output_path.name for result in results] == ["image_seed_101.png", "image_seed_202.png"]
    assert [result.saved_path.name for result in results] == ["image_seed_101.png", "image_seed_202.png"]
    assert [path.read_text() for _, path, _, _ in created["saved_paths"]] == ["seed=101", "seed=202"]
    assert [(seed, path.name) for seed, path, _, _ in created["saved_paths"]] == [
        (101, "image_seed_101.png"),
        (202, "image_seed_202.png"),
    ]
    assert [(event.seed, event.phase) for event in progress_events] == [
        (101, "start"),
        (101, "denoise"),
        (101, "generated"),
        (101, "save"),
        (101, "complete"),
        (202, "start"),
        (202, "denoise"),
        (202, "generated"),
        (202, "save"),
        (202, "complete"),
    ]


def test_loaded_generation_model_routes_video_to_video_to_generate_video(monkeypatch, tmp_path):
    created = {}
    original_import = importlib.import_module

    class FakeArtifact:
        task = "video-to-video"

        def save(self, path, overwrite=True, **kwargs):
            created["save"] = {"path": Path(path), "overwrite": overwrite, "kwargs": dict(kwargs)}
            Path(path).write_text("video")
            return Path(path)

    class FakeWan:
        def __init__(self, **kwargs):
            created["init"] = kwargs
            self.callbacks = CallbackRegistry()

        def generate_video(self, **kwargs):
            created["generate"] = kwargs
            self.callbacks.emit_progress(ProgressEvent(task="video-to-video", phase="start", step=0, total_steps=3))
            self.callbacks.emit_progress(ProgressEvent(task="video-to-video", phase="denoise", step=1, total_steps=3))
            self.callbacks.emit_progress(ProgressEvent(task="video-to-video", phase="complete", step=3, total_steps=3))
            return FakeArtifact()

    def fake_import(name, package=None):
        if name == "mflux.models.wan.variants.wan2_2_ti2v":
            return types.SimpleNamespace(Wan2_2_TI2V=FakeWan)
        return original_import(name, package)

    monkeypatch.setattr("mflux.python_runtime.importlib.import_module", fake_import)

    loaded = load_generation_model_for_plan(
        plan=GenerationPlan(
            public_task="video-to-video",
            mode="latent-video",
            capability_id="wan.video-video",
            family="wan",
            handler_id="wan.generate",
            image_count=0,
            video_count=1,
            model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        )
    )
    progress_events = []
    result = loaded.generate_output(
        seed=777,
        prompt="replace the ship hull",
        video_path="input.mp4",
        video_strength=0.6,
        video_mask_path="mask.png",
        output=tmp_path / "video.mp4",
        progress_callback=progress_events.append,
    )

    assert created["generate"]["video_path"] == "input.mp4"
    assert created["generate"]["video_strength"] == 0.6
    assert created["generate"]["video_mask_path"] == "mask.png"
    assert result.task == "video-to-video"
    assert result.saved_path == tmp_path / "video.mp4"
    assert [event.phase for event in progress_events] == ["start", "denoise", "generated", "save", "complete"]
    assert [event.item_index for event in progress_events] == [1, 1, 1, 1, 1]
    assert all(event.item_count == 1 for event in progress_events)
    assert [event.step for event in progress_events] == [0, 1, 3, 3, 3]
    assert progress_events[0].output_path == str(tmp_path / "video.mp4")
    assert progress_events[-1].output_path == str(tmp_path / "video.mp4")


def test_loaded_generation_model_preserves_existing_image_output_when_overwrite_false(monkeypatch, tmp_path):
    original_import = importlib.import_module

    class FakeArtifact:
        task = "text-to-image"

        def save(self, path, overwrite=True, **kwargs):
            Path(path).write_text("new-image")
            return Path(path)

    class FakeQwenImage:
        def __init__(self, **kwargs):
            self.callbacks = CallbackRegistry()

        def generate_image(self, **kwargs):
            return FakeArtifact()

    def fake_import(name, package=None):
        if name == "mflux.models.qwen.variants.txt2img.qwen_image":
            return types.SimpleNamespace(QwenImage=FakeQwenImage)
        return original_import(name, package)

    monkeypatch.setattr("mflux.python_runtime.importlib.import_module", fake_import)

    occupied_path = tmp_path / "image.png"
    occupied_path.write_text("existing-image")

    loaded = load_generation_model(model="qwen-image")
    result = loaded.generate_output(
        seed=404,
        prompt="test prompt",
        width=16,
        height=16,
        guidance=1.0,
        num_inference_steps=1,
        output=occupied_path,
        overwrite=False,
        save_kwargs={"export_json_metadata": False, "embed_metadata": False},
    )

    assert occupied_path.read_text() == "existing-image"
    assert result.output_path == tmp_path / "image_1.png"
    assert result.saved_path == tmp_path / "image_1.png"
    assert result.saved_path.read_text() == "new-image"


def test_loaded_generation_model_resolves_per_seed_collisions_when_overwrite_false(monkeypatch, tmp_path):
    original_import = importlib.import_module

    class FakeArtifact:
        def __init__(self, seed: int):
            self.seed = seed
            self.task = "text-to-image"

        def save(self, path, overwrite=True, **kwargs):
            Path(path).write_text(f"seed={self.seed}")
            return Path(path)

    class FakeQwenImage:
        def __init__(self, **kwargs):
            self.callbacks = CallbackRegistry()

        def generate_image(self, **kwargs):
            return FakeArtifact(kwargs["seed"])

    def fake_import(name, package=None):
        if name == "mflux.models.qwen.variants.txt2img.qwen_image":
            return types.SimpleNamespace(QwenImage=FakeQwenImage)
        return original_import(name, package)

    monkeypatch.setattr("mflux.python_runtime.importlib.import_module", fake_import)

    occupied_path = tmp_path / "image_seed_101.png"
    occupied_path.write_text("existing-seed-101")

    loaded = load_generation_model(model="qwen-image")
    results = loaded.generate_outputs(
        seeds=[101, 202],
        prompt="test prompt",
        width=16,
        height=16,
        guidance=1.0,
        num_inference_steps=1,
        output=tmp_path / "image.png",
        overwrite=False,
        save_kwargs={"export_json_metadata": False, "embed_metadata": False},
    )

    assert occupied_path.read_text() == "existing-seed-101"
    assert [result.output_path.name for result in results] == ["image_seed_101_1.png", "image_seed_202.png"]
    assert [result.saved_path.name for result in results] == ["image_seed_101_1.png", "image_seed_202.png"]
    assert [result.saved_path.read_text() for result in results] == ["seed=101", "seed=202"]


def test_loaded_generation_model_generate_output_dispatches_video_and_uses_save_contract(monkeypatch, tmp_path):
    created = {"instances": 0, "kwargs": None, "save_kwargs": None}
    original_import = importlib.import_module

    class FakeVideo:
        def __init__(self, seed: int):
            self.seed = seed
            self.task = "text-to-video"

        def save(self, path, overwrite=True, **kwargs):
            created["save_kwargs"] = {"path": Path(path), "overwrite": overwrite, **kwargs}
            Path(path).write_text(f"video-seed={self.seed}")
            return Path(path)

    class FakeWan:
        def __init__(self, **kwargs):
            created["instances"] += 1
            self.callbacks = CallbackRegistry()

        def generate_video(self, **kwargs):
            created["kwargs"] = dict(kwargs)
            self.callbacks.emit_progress(
                ProgressEvent(
                    task="text-to-video",
                    phase="start",
                    frame=0,
                    total_frames=9,
                    step=0,
                    total_steps=1,
                )
            )
            self.callbacks.emit_progress(
                ProgressEvent(
                    task="text-to-video",
                    phase="complete",
                    frame=9,
                    total_frames=9,
                    step=1,
                    total_steps=1,
                )
            )
            return FakeVideo(kwargs["seed"])

    def fake_import(name, package=None):
        if name == "mflux.models.wan.variants.wan2_2_ti2v":
            return types.SimpleNamespace(Wan2_2_TI2V=FakeWan)
        return original_import(name, package)

    monkeypatch.setattr("mflux.python_runtime.importlib.import_module", fake_import)

    loaded = load_generation_model(model="wan2.2-ti2v-5b")
    result = loaded.generate_output(
        seed=303,
        prompt="test prompt",
        width=64,
        height=64,
        num_frames=9,
        fps=8,
        guidance=1.0,
        num_inference_steps=1,
        output=tmp_path / "video.mp4",
        save_kwargs={"export_json_metadata": False, "validate_health": False},
    )

    assert created["instances"] == 1
    assert created["kwargs"]["seed"] == 303
    assert created["kwargs"]["prompt"] == "test prompt"
    assert created["kwargs"]["num_frames"] == 9
    assert result.task == "text-to-video"
    assert result.output_path == tmp_path / "video.mp4"
    assert result.saved_path == tmp_path / "video.mp4"
    assert created["save_kwargs"] == {
        "path": tmp_path / "video.mp4",
        "overwrite": True,
        "export_json_metadata": False,
        "validate_health": False,
    }


def test_loaded_generation_model_preserves_existing_video_output_when_overwrite_false(monkeypatch, tmp_path):
    original_import = importlib.import_module

    class FakeVideo:
        task = "text-to-video"

        def save(self, path, overwrite=True, **kwargs):
            output_path = Path(path)
            output_path.write_text("new-video")
            if kwargs.get("export_json_metadata"):
                output_path.with_suffix(".metadata.json").write_text('{"seed": 505}')
            return output_path

    class FakeWan:
        def __init__(self, **kwargs):
            self.callbacks = CallbackRegistry()

        def generate_video(self, **kwargs):
            return FakeVideo()

    def fake_import(name, package=None):
        if name == "mflux.models.wan.variants.wan2_2_ti2v":
            return types.SimpleNamespace(Wan2_2_TI2V=FakeWan)
        return original_import(name, package)

    monkeypatch.setattr("mflux.python_runtime.importlib.import_module", fake_import)

    occupied_path = tmp_path / "video.mp4"
    occupied_path.write_text("existing-video")

    loaded = load_generation_model(model="wan2.2-ti2v-5b")
    result = loaded.generate_output(
        seed=505,
        prompt="test prompt",
        width=64,
        height=64,
        num_frames=9,
        fps=8,
        guidance=1.0,
        num_inference_steps=1,
        output=occupied_path,
        overwrite=False,
        save_kwargs={"export_json_metadata": True, "validate_health": False},
    )

    assert occupied_path.read_text() == "existing-video"
    assert result.output_path == tmp_path / "video_1.mp4"
    assert result.saved_path == tmp_path / "video_1.mp4"
    assert result.saved_path.read_text() == "new-video"
    assert (tmp_path / "video_1.metadata.json").read_text() == '{"seed": 505}'
