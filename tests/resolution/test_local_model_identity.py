from pathlib import Path

import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS, ModelConfig
from mflux.models.common.lora.lora_compatibility import LoRACompatibility
from mflux.models.common.lora.mapping.lora_loader import LoRAApplicationError
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
from mflux.models.seedvr2.cli.seedvr2_upscale import _resolve_seedvr2_model
from mflux.models.wan.cli import wan_generate
from mflux.task_inference import TaskInferenceError, get_model_capabilities
from mflux.utils.exceptions import ModelConfigError

# A parent made only of words no catalog alias contains, so its presence changes nothing on any release.
NEUTRAL_PARENT = "/opt/neutral-store/checkpoints"
BASE_MODELS = sorted((m for m in AVAILABLE_MODELS.values() if m.base_model is None), key=lambda m: m.priority)
KNOWN_NAMES = sorted(
    {name for base in BASE_MODELS for name in (base.model_name, *base.aliases, *base.inference_aliases)}
)


class TableExpectation:
    @staticmethod
    def exact(name: str) -> ModelConfig | None:
        return next((b for b in BASE_MODELS if name == b.model_name or name in b.aliases), None)

    @staticmethod
    def inferred(text: str) -> ModelConfig | None:
        lowered = text.lower()
        matches = [(b, a) for b in BASE_MODELS for a in b.inference_aliases if a and a.lower() in lowered]
        if not matches:
            return None
        return sorted(matches, key=lambda match: (-len(match[1]), match[0].priority))[0][0]


def _model_dir(root: Path, relative: str) -> str:
    folder = root / relative
    folder.mkdir(parents=True)
    return str(folder)


# The table of the local-model-name specification shared with AbstractVision, row for row. T is a temporary directory,
# C a temporary Hugging Face hub directory; `cwd` is where the row runs (never the real current directory).
SPEC_ROWS = [
    ("1", "flux2-klein-4b", None, None, "flux2-klein-4b"),
    ("2", "black-forest-labs/FLUX.2-klein-4B", None, None, "black-forest-labs/FLUX.2-klein-4B"),
    ("3-file", "org/repo:file.safetensors", None, None, "org/repo:file.safetensors"),
    ("3-rev", "org/repo@rev", None, None, "org/repo@rev"),
    ("4", "black-forest-labs/FLUX.2-klein-4B/transformer", None, None, "black-forest-labs/FLUX.2-klein-4B/transformer"),
    ("5-empty", "", None, None, ""),
    ("5-blank", "   ", None, None, "   "),
    ("6", "{T}/qwen-image-edit_q4/klein-source", "dir", None, "klein-source"),
    ("7", "{T}/qwen/flux2-klein-4b/", "dir", None, "flux2-klein-4b"),
    ("8", "{T}/wan-models/model-q4.gguf", "file", None, "model-q4.gguf"),
    ("9", "{C}/models--Tongyi-MAI--Z-Image-Turbo/snapshots/abc7b123", "dir", None, "Tongyi-MAI/Z-Image-Turbo"),
    (
        "10",
        "{C}/models--Tongyi-MAI--Z-Image-Turbo/snapshots/abc7b123/transformer",
        "dir",
        None,
        "Tongyi-MAI/Z-Image-Turbo/transformer",
    ),
    (
        "11",
        "{C}/models--numz--SeedVR2_comfyUI/snapshots/r1/seedvr2_ema_7b_fp16.safetensors",
        "file",
        None,
        "numz/SeedVR2_comfyUI/seedvr2_ema_7b_fp16.safetensors",
    ),
    ("12", "{C}/models--gpt2/snapshots/abc", "dir", None, "gpt2"),
    ("13", "{C}/models--org--name", "dir", None, "org/name"),
    ("14", "./qwen-image_q4", "dir:{T}/qwen-image_q4", "{T}", "qwen-image_q4"),
    ("15", "~/x/flux2-klein-4b", None, None, "flux2-klein-4b"),
    ("16", ".", "dir:{T}/z-image-turbo", "{T}/z-image-turbo", "z-image-turbo"),
    ("17", "{T}/link-name", "link:{T}/qwen/real", None, "link-name"),
    ("18", "{T}/models--fake/flux", "dir", None, "flux"),
    (
        "19",
        "{C}/models--user--my-loras/snapshots/abc/flux2-klein-4b-style.safetensors",
        "file",
        None,
        "user/my-loras/flux2-klein-4b-style.safetensors",
    ),
    ("20", "{T}/qwen/snapshots/abc/flux2-klein-4b", "dir", None, "flux2-klein-4b"),
]


class SpecRow:
    @staticmethod
    def prepare(tmp_path: Path, monkeypatch, text: str, entry: str | None, cwd: str | None) -> str:
        roots = {"{T}": str(tmp_path / "T"), "{C}": str(tmp_path / "C" / "hub")}

        def place(value: str) -> str:
            for key, root in roots.items():
                value = value.replace(key, root)
            return value

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        text = place(text)
        if entry is not None:
            kind, _, target = entry.partition(":")
            target_path = Path(place(target) if target else text)
            if kind == "file":
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(b"")
            else:
                target_path.mkdir(parents=True, exist_ok=True)
            if kind == "link":
                Path(text).symlink_to(target_path, target_is_directory=True)
        neutral = tmp_path / "cwd"
        neutral.mkdir(exist_ok=True)
        monkeypatch.chdir(place(cwd) if cwd is not None else neutral)
        return text


@pytest.mark.fast
@pytest.mark.parametrize(("row", "text", "entry", "cwd", "expected"), SPEC_ROWS, ids=[row[0] for row in SPEC_ROWS])
def test_identifying_name_follows_the_shared_specification(tmp_path, monkeypatch, row, text, entry, cwd, expected):
    model = SpecRow.prepare(tmp_path, monkeypatch, text, entry, cwd)

    name = ConfigResolution.identifying_name(model)

    assert name == expected
    assert ConfigResolution.identifying_name(name) == name


@pytest.mark.fast
def test_relative_path_on_disk_is_named_by_its_own_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "qwen-image" / "flux2-klein-4b").mkdir(parents=True)

    assert ConfigResolution.identifying_name("qwen-image/flux2-klein-4b") == "flux2-klein-4b"


@pytest.mark.fast
@pytest.mark.parametrize(
    ("relative", "expected_base", "expected_family"),
    [
        ("qwen/flux2-klein-4b", "black-forest-labs/FLUX.2-klein-4B", "flux2"),
        ("wan-models/z-image-turbo", "Tongyi-MAI/Z-Image-Turbo", "z-image"),
        ("models_q4/flux2-klein-4b-8bit", "black-forest-labs/FLUX.2-klein-4B", "flux2"),
        ("qwen-image-edit/hub/models--Tongyi-MAI--Z-Image/snapshots/0123abcd", "Tongyi-MAI/Z-Image", "z-image"),
        ("wan2.2-t2v-a14b/fibo-edit", "briaai/Fibo-Edit", "fibo"),
    ],
)
def test_parent_directory_never_names_the_model(tmp_path, relative, expected_base, expected_family):
    model = _model_dir(tmp_path, relative)

    assert ConfigResolution.resolve(model_name=model).base_model == expected_base
    capabilities = get_model_capabilities(model=model)
    assert capabilities.family == expected_family
    assert capabilities.model_name == model


@pytest.mark.fast
@pytest.mark.parametrize("parent", ["qwen-image", "flux2-klein-9b", "dev", "wan2.2-ti2v-5b"])
def test_a_folder_that_names_nothing_asks_for_its_base_model(tmp_path, parent):
    model = _model_dir(tmp_path, f"{parent}/my-checkpoint")

    with pytest.raises(ModelConfigError, match="Cannot infer base_model"):
        ConfigResolution.resolve(model_name=model)
    with pytest.raises(TaskInferenceError, match="Could not infer a supported backend"):
        get_model_capabilities(model=model)

    config = ConfigResolution.resolve(model_name=model, base_model="flux2-klein-4b")
    assert config.base_model == "black-forest-labs/FLUX.2-klein-4B"
    assert config.model_name == model


@pytest.mark.fast
def test_flux2_base_weights_are_not_named_by_a_parent(tmp_path):
    distilled = ConfigResolution.resolve(model_name=_model_dir(tmp_path, "klein-base-checkpoints/flux2-klein-4b"))
    base = ConfigResolution.resolve(model_name=_model_dir(tmp_path, "checkpoints/flux2-klein-base-4b"))

    assert not _Flux2KleinEditHelpers.is_base_model(distilled)
    assert _Flux2KleinEditHelpers.is_base_model(base)


@pytest.mark.fast
def test_lora_adapter_class_is_not_named_by_a_parent(tmp_path):
    adapter_dir = tmp_path / "qwen-image-edit-2511"
    adapter_dir.mkdir()
    adapter = adapter_dir / "style.safetensors"
    adapter.write_bytes(b"")
    named_adapter = adapter_dir / "flux2-klein-4b-style.safetensors"
    named_adapter.write_bytes(b"")

    assert LoRACompatibility._model_class(str(adapter)) is None
    assert LoRACompatibility._model_class(str(named_adapter)) == "flux2-klein-4b"
    LoRACompatibility.validate_for_model_config(
        model_config=ModelConfig.from_name("flux2-klein-4b"), lora_paths=[str(adapter)]
    )


@pytest.mark.fast
def test_seedvr2_width_is_not_named_by_a_parent(tmp_path):
    model = Path(_model_dir(tmp_path, "seedvr2-7b-sharp/restorer"))
    (model / "transformer").mkdir()
    (model / "transformer" / "model.safetensors.index.json").write_text("{}")

    config, model_path = _resolve_seedvr2_model(str(model), None)

    assert config.aliases == ModelConfig.seedvr2_3b().aliases
    assert model_path == str(model)


@pytest.mark.fast
def test_wan_specific_alias_is_not_read_from_a_parent(tmp_path):
    model = _model_dir(tmp_path, "wan2.2-t2v-a14b/wan-custom")

    with pytest.raises(ModelConfigError, match="specific Wan alias"):
        wan_generate._resolve_model(model)


@pytest.mark.fast
@pytest.mark.parametrize("name", KNOWN_NAMES)
def test_every_known_name_resolves_as_the_table_says(name):
    exact = TableExpectation.exact(name)
    if ConfigResolution._is_unsupported_flux2_dev(name):
        pytest.skip("FLUX.2-dev is refused by name")

    bare = ConfigResolution.resolve(model_name=name)
    expected_bare = exact or TableExpectation.inferred(name)
    assert expected_bare is not None
    assert bare.aliases == expected_bare.aliases
    assert bare.model_name == (expected_bare.model_name if exact is not None else name)

    # The path form is inferred from the whole string on 0.40.2 and from the own folder now: with a neutral parent
    # both readings are the same entry of the table, so a named folder resolves exactly as before.
    path_form = f"{NEUTRAL_PARENT}/{name}"
    expected_path = TableExpectation.inferred(path_form)
    assert expected_path is TableExpectation.inferred(Path(path_form).name)
    if expected_path is None:
        with pytest.raises(ModelConfigError):
            ConfigResolution.resolve(model_name=path_form)
        return
    resolved = ConfigResolution.resolve(model_name=path_form)
    assert resolved.aliases == expected_path.aliases
    assert resolved.model_name == path_form


@pytest.mark.fast
def test_neutral_parent_names_nothing():
    assert TableExpectation.inferred(NEUTRAL_PARENT) is None


@pytest.mark.fast
def test_prepare_backend_is_not_named_by_a_parent(tmp_path):
    from mflux.models.common.cli import save
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    flux2 = ConfigResolution.resolve(model_name=_model_dir(tmp_path, "qwen-image-edit/flux2-klein-4b"))
    qwen = ConfigResolution.resolve(
        model_name=_model_dir(tmp_path, "qwen-image-edit_q4/custom-qwen-folder"), base_model="qwen-image"
    )

    assert save._model_class_for_config(flux2) is Flux2Klein
    assert save._model_class_for_config(qwen) is QwenImage


class TestReviewFindings:
    @staticmethod
    def _hub_file(tmp_path: Path, relative: str) -> str:
        target = tmp_path / "hub" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
        return str(target)

    @pytest.mark.fast
    def test_1_a_single_file_seedvr2_checkpoint_in_a_cache_keeps_its_width(self, tmp_path):
        seven = self._hub_file(tmp_path, "models--numz--SeedVR2_comfyUI/snapshots/r1/seedvr2_ema_7b_fp16.safetensors")
        sharp = self._hub_file(
            tmp_path, "models--numz--SeedVR2_comfyUI/snapshots/r1/seedvr2_ema_7b_sharp_fp16.safetensors"
        )

        assert _resolve_seedvr2_model(seven, None)[0].aliases == ModelConfig.seedvr2_7b().aliases
        assert _resolve_seedvr2_model(sharp, None)[0].aliases == ModelConfig.seedvr2_7b_sharp().aliases

    @pytest.mark.fast
    def test_2_a_lora_file_in_a_cache_is_classed_by_repository_and_file_name(self, tmp_path):
        adapter = self._hub_file(tmp_path, "models--user--my-loras/snapshots/abc/flux2-klein-4b-style.safetensors")

        assert LoRACompatibility._model_class(adapter) == "flux2-klein-4b"
        with pytest.raises(LoRAApplicationError):
            LoRACompatibility.validate_for_model_config(
                model_config=ModelConfig.from_name("qwen-image"), lora_paths=[adapter]
            )

    @pytest.mark.fast
    def test_3_empty_text_is_never_named_by_the_current_directory(self, tmp_path, monkeypatch):
        current = tmp_path / "qwen-image"
        current.mkdir()
        monkeypatch.chdir(current)

        with pytest.raises(ModelConfigError, match="Cannot infer base_model from $"):
            ConfigResolution.resolve(model_name="")
        with pytest.raises(TaskInferenceError, match="Could not infer a supported backend from model ''"):
            get_model_capabilities(model="")
        assert ConfigResolution.resolve(model_name=".").base_model == "Qwen/Qwen-Image"

    @pytest.mark.fast
    def test_4_fibo_low_ram_guard_is_not_named_by_a_parent(self, tmp_path, monkeypatch):
        from mflux.cli.parser.parsers import CommandLineParser

        model = _model_dir(tmp_path, "fibo-experiments/flux2-klein-4b")
        parser = CommandLineParser(description="naming")
        parser.add_general_arguments()
        parser.add_model_arguments(require_model_arg=False)
        parser.add_image_generator_arguments(supports_metadata_config=True)
        parser.add_output_arguments()
        argv = ["mflux-generate-flux2", "--model", model, "--prompt", "a red cube", "--seed", "1", "2", "--low-ram"]
        monkeypatch.setattr("sys.argv", argv)

        assert parser.parse_args().model == model

    @pytest.mark.fast
    def test_5_a_sharded_seedvr2_snapshot_is_not_sized_by_its_revision(self, tmp_path):
        snapshot = tmp_path / "hub" / "models--AbstractFramework--seedvr2-3b-8bit" / "snapshots" / "a17b0c2d"
        (snapshot / "transformer").mkdir(parents=True)
        (snapshot / "transformer" / "model.safetensors.index.json").write_text("{}")

        assert _resolve_seedvr2_model(str(snapshot), None)[0].aliases == ModelConfig.seedvr2_3b().aliases

    @pytest.mark.fast
    @pytest.mark.parametrize(
        ("model", "expected_base"),
        [
            ("black-forest-labs/FLUX.2-klein-4B/transformer", "black-forest-labs/FLUX.2-klein-4B"),
            ("Wan-AI/Wan2.2-T2V-A14B/high_noise_model", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
        ],
    )
    def test_6_a_relative_text_not_on_disk_is_not_a_local_path(self, tmp_path, monkeypatch, model, expected_base):
        monkeypatch.chdir(tmp_path)

        assert ConfigResolution.resolve(model_name=model).base_model == expected_base

    @pytest.mark.fast
    @pytest.mark.parametrize("relative", ["flux2-klein-4b/8bit", "z-image-turbo/mlx", "wan2.2-t2v-a14b/transformer"])
    def test_7_an_unidentified_local_folder_says_what_to_pass(self, tmp_path, relative):
        model = _model_dir(tmp_path, relative)
        own = Path(relative).name

        with pytest.raises(ModelConfigError) as config_error:
            ConfigResolution.resolve(model_name=model)
        with pytest.raises(TaskInferenceError) as capability_error:
            get_model_capabilities(model=model)

        expected = f"a local folder is identified by its own name ('{own}'), not by a parent directory: pass "
        assert str(config_error.value) == f"Cannot infer base_model from {model}: {expected}base_model=<alias>"
        assert str(capability_error.value).startswith(f"Could not infer a supported backend from model {model!r}: ")
        assert f"{expected}--base-model <alias>" in str(capability_error.value)

    @pytest.mark.fast
    def test_7_a_name_that_is_not_a_path_keeps_its_message(self):
        with pytest.raises(ModelConfigError, match=r"^Cannot infer base_model from my-unknown-model$"):
            ConfigResolution.resolve(model_name="my-unknown-model")
        with pytest.raises(TaskInferenceError, match=r"Pass family='qwen'"):
            get_model_capabilities(model="my-unknown-model")
