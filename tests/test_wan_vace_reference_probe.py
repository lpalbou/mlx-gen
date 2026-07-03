import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "wan_vace_reference_probe.py"


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("wan_vace_reference_probe", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_portrait_mask_prioritizes_hair_region_over_face_region():
    module = _load_probe_module()
    mask = module.WanVaceReferenceProbe._build_portrait_mask(384, 384)
    assert mask.size == (384, 384)
    assert mask.mode == "L"
    assert mask.getpixel((192, 28)) > mask.getpixel((192, 160))
    assert mask.getpixel((120, 56)) > mask.getpixel((192, 160))


def test_ship_mask_tracks_motion_upward_and_right():
    module = _load_probe_module()
    first = np.array(
        module.WanVaceReferenceProbe._build_ship_mask_frame(
            frame_index=0,
            total_frames=17,
            width=448,
            height=256,
        )
    )
    last = np.array(
        module.WanVaceReferenceProbe._build_ship_mask_frame(
            frame_index=16,
            total_frames=17,
            width=448,
            height=256,
        )
    )
    first_y, first_x = np.nonzero(first > 32)
    last_y, last_x = np.nonzero(last > 32)
    assert last_x.mean() > first_x.mean() + 80
    assert last_y.mean() < first_y.mean() - 40


def test_prepare_case_inputs_writes_manifest_and_artifacts(tmp_path):
    module = _load_probe_module()
    case = module.WanVaceReferenceProbe.case_presets()["portrait_hair_eyes"]
    manifest_path = module.WanVaceReferenceProbe._prepare_case_inputs(case=case, output_dir=tmp_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["kind"] == "upstream_reference_only"
    assert Path(manifest["source_video_path"]).exists()
    assert Path(manifest["source_contact_sheet_path"]).exists()
    assert Path(manifest["mask_contact_sheet_path"]).exists()
    mask_paths = sorted(Path(manifest["mask_dir"]).glob("mask_*.png"))
    assert len(mask_paths) == case.num_frames


def test_reference_images_for_case_prefers_explicit_portrait_reference(tmp_path):
    module = _load_probe_module()
    case = module.WanVaceReferenceProbe.case_presets()["portrait_hair_eyes"]
    reference_path = tmp_path / "portrait_reference.png"
    Image.new("RGB", (32, 32), (240, 220, 180)).save(reference_path)
    source_frame = Image.new("RGB", (32, 32), (10, 20, 30))

    references = module.WanVaceReferenceProbe._reference_images_for_case(
        case=case,
        manifest={"portrait_reference_image": str(reference_path)},
        source_frames=[source_frame],
    )

    assert references is not None
    assert len(references) == 1
    assert references[0].size == (32, 32)
    assert references[0].getpixel((0, 0)) == (240, 220, 180)


def test_masked_source_video_replaces_edit_region_with_mid_gray():
    module = _load_probe_module()
    source = Image.new("RGB", (4, 4), (10, 20, 30))
    mask = Image.new("L", (4, 4), 0)
    mask.putpixel((1, 1), 255)
    mask.putpixel((2, 2), 255)

    frames = module.WanVaceReferenceProbe._build_masked_source_video(
        source_frames=[source],
        mask_frames=[mask],
    )

    output = np.array(frames[0])
    assert tuple(output[0, 0]) == (10, 20, 30)
    assert tuple(output[1, 1]) == (127, 127, 127)
    assert tuple(output[2, 2]) == (127, 127, 127)


def test_reference_device_auto_prefers_cpu_on_darwin(monkeypatch):
    module = _load_probe_module()

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeMps:
        @staticmethod
        def is_available():
            return True

    class FakeBackends:
        mps = FakeMps()

    class FakeTorch:
        cuda = FakeCuda()
        backends = FakeBackends()

    monkeypatch.setenv("MFLUX_WAN_VACE_DEVICE", "auto")
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")

    assert module.WanVaceReferenceProbe._reference_device(FakeTorch) == "cpu"


def test_reference_device_rejects_unavailable_mps(monkeypatch):
    module = _load_probe_module()

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeMps:
        @staticmethod
        def is_available():
            return False

    class FakeBackends:
        mps = FakeMps()

    class FakeTorch:
        cuda = FakeCuda()
        backends = FakeBackends()

    monkeypatch.setenv("MFLUX_WAN_VACE_DEVICE", "mps")

    with pytest.raises(ValueError, match="MPS is not available"):
        module.WanVaceReferenceProbe._reference_device(FakeTorch)
