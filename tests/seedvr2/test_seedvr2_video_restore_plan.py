import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from PIL import Image

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.seedvr2 import SeedVR2VideoRestorePlan, SeedVR2VideoRestoreRefused
from mflux.models.seedvr2.cli import seedvr2_upscale
from mflux.models.seedvr2.seedvr2_initializer import SeedVR2Initializer
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util
from mflux.utils.scale_factor import ScaleFactor
from mflux.utils.video_util import DecodedVideoClip, VideoUtil

GB = 1000**3
CONFIGS = {"3b": ModelConfig.seedvr2_3b, "7b": ModelConfig.seedvr2_7b, "7b-sharp": ModelConfig.seedvr2_7b_sharp}
RESIDENT_WEIGHT_BYTES = {"3b": 15 * GB, "7b": 34 * GB, "7b-sharp": 34 * GB}

SCALE_REFUSAL = (
    "SeedVR2 safe video mode only supports source-size restoration or smaller output. "
    "Use 1x or override with --force-unsafe-video-memory."
)
MEMORY_REFUSAL = (
    "This SeedVR2 video chunk request exceeds the supported safe memory profile. "
    "Use --force-unsafe-video-memory to run it explicitly, or reduce source size, resolution, or clip length."
)
FRAME_SIZE_REFUSAL_1080P = (
    "SeedVR2 video restore cannot fit a 1920x1072 output frame in this machine's video memory budget (1.0 GB): "
    "the VAE works on whole frames, so this size needs about 8.4 GB whatever the temporal chunk size. "
    "The largest output that fits is 640x368 (235520 pixels per frame): use --resolution 368 or smaller, "
    "or --force-unsafe-video-memory to run this size anyway."
)
CHUNK_FLOOR_REFUSAL = (
    "SeedVR2 video restore refuses temporal chunks smaller than 29 frames because short independent chunks "
    "can distort object continuity. Use whole-shot restore, reduce the clip/resolution, or use a larger "
    "--temporal-chunk-size."
)

# What a caller passes: (variant, (width, height, source_frame_count, fps, clip_start_frame), max_frames,
# resolution, temporal_chunk_size, temporal_chunk_overlap, force_unsafe_memory_profile, available_gb).
# A source_frame_count of None makes the probe report only a 4 s duration.
CASES = {
    "small_whole_shot": ("3b", (320, 192, 17, 24.0, 0), None, None, None, None, False, 64),
    "hd_default_chunking": ("3b", (1920, 1080, 300, 30.0, 0), None, None, None, None, False, 64),
    "explicit_49_16": ("3b", (640, 360, 130, 29.97, 0), None, "1x", 49, 16, False, 64),
    "clip_window": ("3b", (640, 360, 100, 25.0, 30), 40, None, None, None, False, 64),
    "duration_only_probe": ("3b", (640, 360, None, 25.0, 0), None, None, None, None, False, 64),
    "int_resolution_source_size": ("3b", (640, 360, 60, 25.0, 0), None, 360, None, None, False, 64),
    "downscale_half": ("3b", (1280, 720, 90, 25.0, 0), None, "0.5x", None, None, False, 64),
    "seven_b_long": ("7b", (320, 240, 2908, 29.97, 0), None, None, 49, 16, False, 121),
    "seven_b_sharp": ("7b-sharp", (854, 480, 80, 24.0, 0), None, None, None, None, False, 64),
    "refuse_scale_2x": ("3b", (320, 240, 130, 29.97, 0), None, "2x", 49, 16, False, 64),
    "refuse_scale_int": ("3b", (640, 360, 60, 25.0, 0), None, 720, None, None, False, 64),
    "refuse_beyond_memory": ("3b", (1920, 1080, 300, 30.0, 0), None, None, 97, 16, False, 16),
    "refuse_chunk_beyond_memory": ("3b", (640, 360, 300, 30.0, 0), None, None, 97, 16, False, 16),
    "refuse_scale_and_memory": ("3b", (960, 540, 300, 30.0, 0), None, "2x", 97, 16, False, 16),
    "refuse_temporal_quality": ("3b", (640, 360, 100, 25.0, 0), None, None, 17, None, False, 64),
    "force_scale_2x": ("3b", (320, 240, 130, 29.97, 0), None, "2x", None, None, True, 64),
    "force_beyond_memory": ("3b", (1920, 1080, 300, 30.0, 0), None, None, 97, 16, True, 16),
    "force_temporal_quality": ("3b", (640, 360, 100, 25.0, 0), None, None, 17, None, True, 64),
    "error_chunk_not_4n_plus_1": ("3b", (640, 360, 100, 25.0, 0), None, None, 50, None, False, 64),
    "error_overlap_not_multiple_of_4": ("3b", (640, 360, 100, 25.0, 0), None, None, 49, 10, False, 64),
}

# Recorded once from the 0.40.1 planner: `_plan_seedvr2_video_restore` and `_requested_video_frame_count`
# of `git show main:src/mflux/models/seedvr2/cli/seedvr2_upscale.py` (main = 0.40.1), fed each case
# exactly as its `main()` fed it (low_ram_requested=True, 49/16 marked not explicit when omitted), with
# host memory (128 GB total, `available_gb` available) and resident weights pinned as in `_pin_host`.
# (variant, requested_frames, target_height, target_width, route_reason, effective_chunk_size,
#  effective_chunk_overlap, chunk_frame_limit, chunk_pixel_volume, risk_level); every one of these plans
# had restore_mode "streaming", no warnings and low_ram_required True.
EXPECTED_0401_PLANS = {
    "small_whole_shot": ("3b", 17, 192, 320, "safe_streaming", 17, 0, 17, 1044480, "low"),
    "hd_default_chunking": ("3b", 300, 1072, 1920, "safe_streaming", 85, 16, 85, 174950400, "low"),
    "explicit_49_16": ("3b", 130, 352, 640, "safe_streaming", 49, 16, 130, 11038720, "low"),
    "clip_window": ("3b", 40, 352, 640, "safe_streaming", 41, 0, 40, 9236480, "low"),
    "duration_only_probe": ("3b", 100, 352, 640, "safe_streaming", 101, 0, 100, 22753280, "low"),
    "int_resolution_source_size": ("3b", 60, 352, 640, "safe_streaming", 61, 0, 60, 13742080, "low"),
    "downscale_half": ("3b", 90, 352, 640, "safe_streaming", 93, 0, 90, 20951040, "low"),
    "seven_b_long": ("7b", 2908, 240, 320, "safe_streaming", 49, 16, 2597, 3763200, "medium"),
    "seven_b_sharp": ("7b-sharp", 80, 480, 848, "safe_streaming", 81, 0, 80, 32970240, "medium"),
    "force_scale_2x": ("3b", 130, 480, 640, "unsafe_override", 133, 0, 130, 40857600, "medium"),
    "force_beyond_memory": ("3b", 300, 1072, 1920, "unsafe_override", 97, 16, 1, 199649280, "medium"),
}

# Same recording: the 0.40.1 CLI exited with `parser.error(plan.warnings[0])` for these.
# (reason, message, number of warnings the 0.40.1 plan carried)
EXPECTED_0401_REFUSALS = {
    "refuse_scale_2x": (SeedVR2VideoRestoreRefused.SCALE_ABOVE_SAFE_PROFILE, SCALE_REFUSAL, 1),
    "refuse_scale_int": (SeedVR2VideoRestoreRefused.SCALE_ABOVE_SAFE_PROFILE, SCALE_REFUSAL, 1),
    # 0.40.1 gave MEMORY_REFUSAL here. Even the smallest chunk the profile allows does not fit this
    # frame size, so the plan now says that the frame size is what does not fit (same reason).
    "refuse_beyond_memory": (SeedVR2VideoRestoreRefused.BEYOND_MEMORY_PROFILE, FRAME_SIZE_REFUSAL_1080P, 1),
    "refuse_chunk_beyond_memory": (SeedVR2VideoRestoreRefused.BEYOND_MEMORY_PROFILE, MEMORY_REFUSAL, 1),
    "refuse_scale_and_memory": (SeedVR2VideoRestoreRefused.SCALE_ABOVE_SAFE_PROFILE, SCALE_REFUSAL, 2),
    "refuse_temporal_quality": (SeedVR2VideoRestoreRefused.BELOW_TEMPORAL_QUALITY_FLOOR, CHUNK_FLOOR_REFUSAL, 1),
}

# Same recording: the 0.40.1 planner raised ValueError with these messages.
EXPECTED_0401_ERRORS = {
    "force_temporal_quality": CHUNK_FLOOR_REFUSAL,
    "error_chunk_not_4n_plus_1": (
        "--temporal-chunk-size must satisfy 4n+1 for SeedVR2 video restore. "
        "Got 50; use an explicit aligned value such as 53."
    ),
    "error_overlap_not_multiple_of_4": "--temporal-chunk-overlap must be a multiple of 4 for SeedVR2 video restore.",
}


@pytest.fixture(autouse=True)
def _without_the_vae_term(monkeypatch):
    # These tests pin the planner's policy and the CLI's plumbing as recorded from 0.40.1, whose
    # estimate had no VAE term. The VAE term and its refusals are tested in
    # test_seedvr2_video_vae_memory.py.
    monkeypatch.setattr(SeedVR2Util, "estimate_video_vae_working_set_bytes", staticmethod(lambda **kwargs: 0))


def _probe(width: int, height: int, frame_count: int | None, fps: float, clip_start_frame: int) -> DecodedVideoClip:
    return DecodedVideoClip(
        frames=[],
        fps=fps,
        source_width=width,
        source_height=height,
        source_frame_count=frame_count,
        source_duration_seconds=None if frame_count is not None else 4.0,
        audio_present=True,
        clip_start_frame=clip_start_frame,
        clip_frame_count=1,
    )


def _pin_host(
    monkeypatch,
    *,
    variant: str,
    available_gb: int,
    probe: DecodedVideoClip,
    probe_calls: list,
    estimate_calls: list | None = None,
) -> None:
    monkeypatch.setattr(SeedVR2Util, "_host_total_memory_bytes", staticmethod(lambda: 128 * GB))
    monkeypatch.setattr(SeedVR2Util, "_host_available_memory_bytes", staticmethod(lambda: available_gb * GB))

    def estimate_resident_weight_bytes(model_config, model_path=None):
        if estimate_calls is not None:
            estimate_calls.append(model_path)
        return RESIDENT_WEIGHT_BYTES[variant]

    monkeypatch.setattr(
        SeedVR2Initializer, "estimate_resident_weight_bytes", staticmethod(estimate_resident_weight_bytes)
    )

    def read_video_clip(path, start_seconds=0.0, max_frames=None):
        probe_calls.append((path, start_seconds, max_frames))
        return probe

    monkeypatch.setattr(VideoUtil, "read_video_clip", staticmethod(read_video_clip))


def _plan_case(monkeypatch, case_id: str) -> SeedVR2VideoRestorePlan:
    variant, probe_spec, max_frames, resolution, chunk, overlap, force, available_gb = CASES[case_id]
    _pin_host(monkeypatch, variant=variant, available_gb=available_gb, probe=_probe(*probe_spec), probe_calls=[])
    return SeedVR2VideoRestorePlan.for_video(
        "clip.mp4",
        model_config=CONFIGS[variant](),
        max_frames=max_frames,
        resolution=resolution,
        temporal_chunk_size=chunk,
        temporal_chunk_overlap=overlap,
        force_unsafe_memory_profile=force,
    )


@pytest.mark.fast
@pytest.mark.parametrize("case_id", sorted(EXPECTED_0401_PLANS))
def test_plan_matches_the_0401_planner(monkeypatch, case_id):
    plan = _plan_case(monkeypatch, case_id)

    assert (
        plan.variant,
        plan.requested_frames,
        plan.target_height,
        plan.target_width,
        plan.route_reason,
        plan.effective_chunk_size,
        plan.effective_chunk_overlap,
        plan.chunk_frame_limit,
        plan.chunk_pixel_volume,
        plan.risk_level,
    ) == EXPECTED_0401_PLANS[case_id]
    assert plan.restore_mode == "streaming"
    assert plan.warnings == ()
    assert plan.low_ram_required is True


@pytest.mark.fast
@pytest.mark.parametrize("case_id", sorted(EXPECTED_0401_REFUSALS))
def test_refusal_carries_the_0401_cli_error_and_a_reason(monkeypatch, case_id):
    reason, message, warning_count = EXPECTED_0401_REFUSALS[case_id]

    with pytest.raises(SeedVR2VideoRestoreRefused) as refused:
        _plan_case(monkeypatch, case_id)

    assert isinstance(refused.value, ValueError)
    assert str(refused.value) == message
    assert refused.value.reason == reason
    assert refused.value.warnings[0] == message
    assert len(refused.value.warnings) == warning_count


@pytest.mark.fast
@pytest.mark.parametrize("case_id", sorted(EXPECTED_0401_ERRORS))
def test_argument_errors_keep_the_0401_messages(monkeypatch, case_id):
    with pytest.raises(ValueError) as error:
        _plan_case(monkeypatch, case_id)

    assert str(error.value) == EXPECTED_0401_ERRORS[case_id]


@pytest.mark.fast
def test_force_switch_does_not_lift_the_temporal_quality_floor(monkeypatch):
    with pytest.raises(SeedVR2VideoRestoreRefused) as refused:
        _plan_case(monkeypatch, "force_temporal_quality")

    assert refused.value.reason == SeedVR2VideoRestoreRefused.BELOW_TEMPORAL_QUALITY_FLOOR


@pytest.mark.fast
@pytest.mark.parametrize(
    "refused_case, forced_case",
    [("refuse_scale_2x", "force_scale_2x"), ("refuse_beyond_memory", "force_beyond_memory")],
)
def test_force_switch_returns_the_plan_for_memory_refusals(monkeypatch, refused_case, forced_case):
    with pytest.raises(SeedVR2VideoRestoreRefused):
        _plan_case(monkeypatch, refused_case)

    plan = _plan_case(monkeypatch, forced_case)

    assert plan.route_reason == "unsafe_override"
    assert plan.enforce_memory_budget is False
    assert plan.warnings == ()


@pytest.mark.fast
def test_plan_carries_what_a_host_applies(monkeypatch):
    probe_calls: list = []
    _pin_host(
        monkeypatch, variant="3b", available_gb=64, probe=_probe(640, 360, 100, 25.0, 25), probe_calls=probe_calls
    )

    plan = SeedVR2VideoRestorePlan.for_video("clip.mp4", start_seconds=1.0, max_frames=40)

    assert probe_calls == [("clip.mp4", 1.0, 1)]
    assert plan.resolution == ScaleFactor(1)
    assert (plan.start_seconds, plan.max_frames, plan.requested_frames) == (1.0, 40, 40)
    assert (plan.source_width, plan.source_height, plan.audio_present) == (640, 360, True)
    assert (plan.effective_chunk_size, plan.effective_chunk_overlap) == (41, 0)
    assert plan.cache_limit_bytes == SeedVR2VideoRestorePlan.CACHE_LIMIT_BYTES == 8 * GB
    assert plan.low_ram_required is True
    assert plan.keep_transformer is True
    assert plan.enforce_memory_budget is True
    with pytest.raises(AttributeError):
        plan.effective_chunk_size = 49


@pytest.mark.fast
@pytest.mark.parametrize(
    "requested, resolved",
    [(None, ScaleFactor(1)), ("1x", ScaleFactor(1)), ("0.5x", ScaleFactor(0.5)), ("360", 360), (360, 360)],
)
def test_resolution_accepts_short_edge_or_scale_factor(monkeypatch, requested, resolved):
    _pin_host(monkeypatch, variant="3b", available_gb=64, probe=_probe(640, 360, 17, 25.0, 0), probe_calls=[])

    assert SeedVR2VideoRestorePlan.for_video("clip.mp4", resolution=requested).resolution == resolved


# `restore_video_to_path` kwargs the 0.40.1 CLI built for these command lines, recorded once by running
# `main()` of `git show main:src/mflux/models/seedvr2/cli/seedvr2_upscale.py` with the stubs below (a
# 640x360, 130-frame, 25 fps probe with audio; 128 GB host with 64 GB available; 15 GB resident weights).
BASE_0401_RESTORE_METADATA = {
    "restore_mode": "streaming",
    "restore_mode_reason": "safe_streaming",
    "low_ram_requested": True,
    "low_ram_effective": True,
    "mlx_cache_limit_gb": 8.0,
    "requested_temporal_chunk_size": 49,
    "requested_temporal_chunk_overlap": 16,
    "effective_temporal_chunk_size": 133,
    "effective_temporal_chunk_overlap": 0,
    "seedvr2_risk_level": "low",
    "seedvr2_target_height": 352,
    "seedvr2_target_width": 640,
    "drop_audio_requested": False,
}
BASE_0401_RESTORE_KWARGS = {
    "seed": 7,
    "resolution": ScaleFactor(1),
    "softness": 0.0,
    "start_seconds": 0.0,
    "max_frames": None,
    "output_path": Path("video.mp4"),
    "export_json_metadata": False,
    "overwrite": True,
    "temporal_chunk_size": 133,
    "temporal_chunk_overlap": 0,
    "color_correction_mode": "wavelet",
    "drop_audio": False,
    "enforce_memory_budget": True,
    "validate_health": True,
}
CLI_0401_CALLS = {
    "defaults": (["--seed", "7"], {}, {}),
    "explicit_everything": (
        [
            "--seed",
            "7",
            "--resolution",
            "1x",
            "--temporal-chunk-size",
            "49",
            "--temporal-chunk-overlap",
            "16",
            "--start-seconds",
            "0.5",
            "--max-frames",
            "60",
            "--softness",
            "0.2",
            "--color-correction",
            "lab",
            "--drop-audio",
            "--metadata",
            "--output",
            "out.mp4",
        ],
        {
            "color_correction_mode": "lab",
            "drop_audio": True,
            "export_json_metadata": True,
            "max_frames": 60,
            "output_path": Path("out.mp4"),
            "softness": 0.2,
            "start_seconds": 0.5,
            "temporal_chunk_size": 49,
            "temporal_chunk_overlap": 16,
        },
        {"drop_audio_requested": True, "effective_temporal_chunk_size": 49, "effective_temporal_chunk_overlap": 16},
    ),
    "force_2x_cache_12": (
        [
            "--seed",
            "3",
            "--resolution",
            "2x",
            "--force-unsafe-video-memory",
            "--mlx-cache-limit-gb",
            "12",
            "--no-validate-health",
        ],
        {"seed": 3, "resolution": ScaleFactor(2), "enforce_memory_budget": False, "validate_health": False},
        {
            "mlx_cache_limit_gb": 12.0,
            "restore_mode_reason": "unsafe_override",
            "seedvr2_risk_level": "medium",
            "seedvr2_target_height": 720,
            "seedvr2_target_width": 1280,
        },
    ),
    "seven_b_clamped_cache": (
        ["--seed", "1", "--model", "seedvr2-7b", "--mlx-cache-limit-gb", "20"],
        {"seed": 1},
        {"seedvr2_risk_level": "medium"},
    ),
    "int_resolution": (["--seed", "1", "--resolution", "360", "--low-ram"], {"seed": 1, "resolution": 360}, {}),
}


class _RecordingSeedVR2:
    calls: list = []

    def __init__(self, *, quantize, model_path, model_config):
        self.model_config = model_config
        self.tiling_config = None

    def restore_video_to_path(self, **kwargs):
        _RecordingSeedVR2.calls.append(kwargs)
        if kwargs["seed"] == 13:
            raise RuntimeError("boom")
        return kwargs["output_path"]

    def generate_image(self, **kwargs):
        _RecordingSeedVR2.calls.append(kwargs)
        raise _StopAfterImageCall


class _StopAfterImageCall(Exception):
    pass


def _run_cli(
    monkeypatch, tmp_path, argv: list[str], *, image: bool = False, estimate_calls: list | None = None
) -> list[dict]:
    clip = tmp_path / "clip.mp4"
    clip.touch()
    frame = tmp_path / "frame.png"
    frame.touch()
    monkeypatch.chdir(tmp_path)
    _pin_host(
        monkeypatch,
        variant="3b",
        available_gb=64,
        probe=_probe(640, 360, 130, 25.0, 0),
        probe_calls=[],
        estimate_calls=estimate_calls,
    )
    _RecordingSeedVR2.calls = []
    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", _RecordingSeedVR2)
    monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", lambda **kwargs: None)
    inputs = ["--image-path", str(frame)] if image else ["--video-path", str(clip)]
    monkeypatch.setattr("sys.argv", ["mflux-upscale-seedvr2", *inputs, *argv])
    try:
        seedvr2_upscale.main()
    except _StopAfterImageCall:
        pass
    for kwargs in _RecordingSeedVR2.calls:
        assert kwargs.pop("image_path" if image else "video_path") == (frame if image else clip)
    return _RecordingSeedVR2.calls


@pytest.mark.fast
@pytest.mark.parametrize("command", sorted(CLI_0401_CALLS))
def test_cli_builds_the_0401_restore_call(monkeypatch, tmp_path, command):
    argv, kwargs_overrides, metadata_overrides = CLI_0401_CALLS[command]

    calls = _run_cli(monkeypatch, tmp_path, argv)

    expected = {
        **BASE_0401_RESTORE_KWARGS,
        **kwargs_overrides,
        "restore_metadata": {**BASE_0401_RESTORE_METADATA, **metadata_overrides},
    }
    assert calls == [expected]


@pytest.mark.fast
def test_cli_failure_manifest_keeps_the_0401_plan_keys(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        _run_cli(monkeypatch, tmp_path, ["--seed", "13"])

    restore_plan = json.loads((tmp_path / "video.failure.json").read_text())["run"]["restore_plan"]
    assert {
        "variant": "3b",
        "requested_frames": 130,
        "target_height": 352,
        "target_width": 640,
        "restore_mode": "streaming",
        "route_reason": "safe_streaming",
        "effective_chunk_size": 133,
        "effective_chunk_overlap": 0,
        "chunk_frame_limit": 130,
        "chunk_pixel_volume": 29962240,
        "low_ram_required": True,
        "low_ram_effective": True,
        "cache_limit_gb": 8.0,
        "risk_level": "low",
        "warnings": [],
    }.items() <= restore_plan.items()


# 0.40.1 decided that an option was given by matching argv strings, which missed `--opt=value`,
# abbreviations and `-r2x`: `--temporal-chunk-size=97` was ignored and `--resolution=2x` on a video
# ran at 1x. Every spelling argparse accepts must now make the call its spaced spelling makes.
SPELLINGS = [
    (
        ["--temporal-chunk-size", "49", "--temporal-chunk-overlap", "8"],
        ["--temporal-chunk-size", "49", "--temporal-chunk-overlap=8"],
    ),
    (
        ["--temporal-chunk-size", "49", "--temporal-chunk-overlap", "8"],
        ["--temporal-chunk-size", "49", "--temporal-chunk-ov", "8"],
    ),
    (["--temporal-chunk-size", "97"], ["--temporal-chunk-size=97"]),
    (["--temporal-chunk-size", "97"], ["--temporal-chunk-s", "97"]),
    (["--resolution", "2x", "--force-unsafe-video-memory"], ["--resolution=2x", "--force-unsafe-video-memory"]),
    (["--resolution", "2x", "--force-unsafe-video-memory"], ["-r2x", "--force-unsafe-video-memory"]),
]


@pytest.mark.fast
@pytest.mark.parametrize("spaced, other", SPELLINGS)
def test_cli_honours_every_spelling_of_a_planning_option(monkeypatch, tmp_path, spaced, other):
    assert _run_cli(monkeypatch, tmp_path, ["--seed", "7", *other]) == _run_cli(
        monkeypatch, tmp_path, ["--seed", "7", *spaced]
    )


@pytest.mark.fast
def test_cli_honours_temporal_chunk_overlap_with_equals(monkeypatch, tmp_path):
    (call,) = _run_cli(monkeypatch, tmp_path, ["--temporal-chunk-size", "49", "--temporal-chunk-overlap=8"])

    assert (call["temporal_chunk_size"], call["temporal_chunk_overlap"]) == (49, 8)


@pytest.mark.fast
def test_cli_validates_temporal_chunk_overlap_with_equals_as_typed(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _run_cli(monkeypatch, tmp_path, ["--temporal-chunk-size", "49", "--temporal-chunk-overlap=10"])

    assert exit_info.value.code == 2
    assert "--temporal-chunk-overlap must be a multiple of 4" in capsys.readouterr().err


@pytest.mark.fast
def test_cli_honours_temporal_chunk_size_with_equals(monkeypatch, tmp_path):
    (call,) = _run_cli(monkeypatch, tmp_path, ["--temporal-chunk-size=97"])

    assert (call["temporal_chunk_size"], call["temporal_chunk_overlap"]) == (97, 16)


@pytest.mark.fast
@pytest.mark.parametrize("spelling", [["--resolution=2x"], ["--resol", "2x"], ["-r2x"]])
def test_cli_refuses_a_video_scale_above_the_safe_profile_in_every_spelling(monkeypatch, tmp_path, capsys, spelling):
    with pytest.raises(SystemExit) as exit_info:
        _run_cli(monkeypatch, tmp_path, spelling)

    assert exit_info.value.code == 2
    assert capsys.readouterr().err.rstrip().endswith(f"error: {SCALE_REFUSAL}")
    assert _RecordingSeedVR2.calls == []


@pytest.mark.fast
def test_cli_honours_resolution_with_equals_on_an_image(monkeypatch, tmp_path):
    (call,) = _run_cli(monkeypatch, tmp_path, ["--seed", "5", "--resolution=2x"], image=True)

    assert call["resolution"] == ScaleFactor(2)


@pytest.mark.fast
def test_cli_image_resolution_defaults_to_384(monkeypatch, tmp_path):
    (call,) = _run_cli(monkeypatch, tmp_path, ["--seed", "5"], image=True)

    assert call["resolution"] == 384


@pytest.mark.fast
def test_cli_help_still_states_the_defaults(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["mflux-upscale-seedvr2", "--help"])

    with pytest.raises(SystemExit):
        seedvr2_upscale.main()

    help_text = " ".join(capsys.readouterr().out.split())
    assert "For video, omitting --resolution defaults to 1x." in help_text
    assert "such as 45 or 49. Default: 49." in help_text
    assert "not an output crossfade. Default: 16." in help_text


@pytest.mark.fast
def test_resident_weights_are_not_reserved_twice(monkeypatch):
    # On real weights (SeedVR2 3B, 7.28 GB) the available figure dropped by 7.84 GB once the
    # model was loaded and evaluated: a plan made after the load must not reserve them again.
    hd = _probe(1920, 1080, 300, 30.0, 0)
    _pin_host(monkeypatch, variant="3b", available_gb=64, probe=hd, probe_calls=[])
    before_load = SeedVR2VideoRestorePlan.for_video("clip.mp4")

    estimate_calls: list = []
    _pin_host(monkeypatch, variant="3b", available_gb=64 - 15, probe=hd, probe_calls=[], estimate_calls=estimate_calls)
    loaded = SeedVR2VideoRestorePlan.for_video("clip.mp4", weights_resident=True)
    reserved_twice = SeedVR2VideoRestorePlan.for_video("clip.mp4")

    assert loaded == before_load
    assert estimate_calls == [None], "weights_resident=True does not estimate the weights"
    assert reserved_twice.effective_chunk_size < before_load.effective_chunk_size


@pytest.mark.fast
def test_plan_forwards_model_path_to_the_weight_estimate(monkeypatch):
    estimate_calls: list = []
    _pin_host(
        monkeypatch,
        variant="3b",
        available_gb=64,
        probe=_probe(640, 360, 17, 25.0, 0),
        probe_calls=[],
        estimate_calls=estimate_calls,
    )

    SeedVR2VideoRestorePlan.for_video("clip.mp4", model_path="/models/seedvr2-3b-8bit")

    assert estimate_calls == ["/models/seedvr2-3b-8bit"]


@pytest.mark.fast
@pytest.mark.parametrize(
    "frames, chunk_size, overlap, message",
    [
        (1, 0, None, "--temporal-chunk-size must be greater than zero."),
        (1, -5, None, "--temporal-chunk-size must be greater than zero."),
        (40, None, -4, "--temporal-chunk-overlap must be greater than or equal to zero."),
        (40, 49, 60, "--temporal-chunk-overlap must be smaller than --temporal-chunk-size."),
        (40, 9, None, "--temporal-chunk-overlap must be smaller than --temporal-chunk-size."),
    ],
)
def test_plan_refuses_the_chunk_shapes_the_cli_refuses(monkeypatch, frames, chunk_size, overlap, message):
    _pin_host(monkeypatch, variant="3b", available_gb=64, probe=_probe(640, 360, frames, 25.0, 0), probe_calls=[])

    with pytest.raises(ValueError) as error:
        SeedVR2VideoRestorePlan.for_video("clip.mp4", temporal_chunk_size=chunk_size, temporal_chunk_overlap=overlap)

    assert str(error.value) == message
    assert not isinstance(error.value, SeedVR2VideoRestoreRefused)


@pytest.mark.fast
def test_cli_forwards_the_model_path_to_the_plan(monkeypatch, tmp_path):
    estimate_calls: list = []

    _run_cli(monkeypatch, tmp_path, ["--seed", "7", "--model", "numz/SeedVR2_comfyUI"], estimate_calls=estimate_calls)

    assert estimate_calls == ["numz/SeedVR2_comfyUI"]


@pytest.mark.fast
@pytest.mark.parametrize("overlap_args, overlap", [([], 8), (["--temporal-chunk-overlap", "16"], 16)])
def test_cli_caps_the_default_overlap_but_keeps_a_typed_one(monkeypatch, tmp_path, overlap_args, overlap):
    (call,) = _run_cli(monkeypatch, tmp_path, ["--temporal-chunk-size", "33", *overlap_args])

    assert (call["temporal_chunk_size"], call["temporal_chunk_overlap"]) == (33, overlap)


@pytest.mark.fast
@pytest.mark.parametrize(
    "argv, cache_limit, audio_note",
    [
        (
            [],
            "8.0",
            "SeedVR2 note: source audio detected; the CLI will preserve the matching source audio segment. "
            "If that cannot be proven safe, the run fails. Pass --drop-audio to allow a silent output intentionally.",
        ),
        (
            ["--mlx-cache-limit-gb", "4", "--drop-audio"],
            "4.0",
            "SeedVR2 note: source audio detected; --drop-audio was requested, so the restored MP4 will stay silent.",
        ),
    ],
)
def test_cli_prints_the_cache_limit_in_force_and_the_audio_note(
    monkeypatch, tmp_path, capsys, argv, cache_limit, audio_note
):
    _run_cli(monkeypatch, tmp_path, ["--seed", "7", *argv])

    lines = capsys.readouterr().out.splitlines()
    assert (
        "SeedVR2 video preflight: model=3b source=640x360 target=640x352 frames=130 mode=streaming "
        f"reason=safe_streaming low_ram=True cache_limit_gb={cache_limit} video=clip.mp4"
    ) in lines
    assert audio_note in lines


@pytest.mark.fast
def test_cli_failure_manifest_states_the_cache_limit_in_force_once(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        _run_cli(monkeypatch, tmp_path, ["--seed", "13", "--mlx-cache-limit-gb", "4"])

    restore_plan = json.loads((tmp_path / "video.failure.json").read_text())["run"]["restore_plan"]
    assert restore_plan["cache_limit_gb"] == 4.0
    assert "cache_limit_bytes" not in restore_plan


# (variant, resident weight bytes): the 3B and 7B source weights, and the 3B 8-bit package measured
# on real weights (7.28 GB).
RUNTIME_CASES = [("3b", 15 * GB), ("7b", 34 * GB), ("3b", 7_284_343_622)]
HD_CLIP = (1920, 1080, 300, 30.0, 0)


def _plan_before_load_then_pin_loaded(monkeypatch, variant: str, weight_bytes: int, available_gb: int):
    # The CLI plans before the load; once the model is loaded, available memory is lower by its weights.
    monkeypatch.setattr(SeedVR2Util, "_host_total_memory_bytes", staticmethod(lambda: 128 * GB))
    monkeypatch.setattr(SeedVR2Util, "_host_available_memory_bytes", staticmethod(lambda: available_gb * GB))
    monkeypatch.setattr(
        SeedVR2Initializer,
        "estimate_resident_weight_bytes",
        staticmethod(lambda model_config, model_path=None: weight_bytes),
    )
    monkeypatch.setattr(VideoUtil, "read_video_clip", staticmethod(lambda *args, **kwargs: _probe(*HD_CLIP)))
    plan = SeedVR2VideoRestorePlan.for_video("clip.mp4", model_config=CONFIGS[variant]())
    monkeypatch.setattr(
        SeedVR2Util, "_host_available_memory_bytes", staticmethod(lambda: available_gb * GB - weight_bytes)
    )
    loaded_model = SimpleNamespace(
        model_config=CONFIGS[variant](), seedvr2_resident_weight_bytes=weight_bytes, seedvr2_weights_read=True
    )
    return plan, loaded_model


def _runtime_admits(loaded_model, plan: SeedVR2VideoRestorePlan, frame_count: int) -> bool:
    try:
        SeedVR2._assert_video_restore_memory_budget(
            loaded_model, frame_count=frame_count, true_height=plan.target_height, true_width=plan.target_width
        )
    except ValueError:
        return False
    return True


@pytest.mark.fast
@pytest.mark.parametrize("variant, weight_bytes", RUNTIME_CASES, ids=["3b", "7b", "3b-8bit"])
def test_runtime_check_admits_what_the_plan_admits_and_refuses_above_it(monkeypatch, variant, weight_bytes):
    plan, loaded_model = _plan_before_load_then_pin_loaded(monkeypatch, variant, weight_bytes, available_gb=64)
    assert 1 <= plan.chunk_frame_limit < plan.requested_frames, "the memory limit binds in this case"

    assert _runtime_admits(loaded_model, plan, plan.chunk_frame_limit)
    assert not _runtime_admits(loaded_model, plan, plan.chunk_frame_limit + 1)


@pytest.mark.fast
def test_runtime_check_of_0401_refused_a_chunk_its_plan_admitted(monkeypatch):
    plan, _ = _plan_before_load_then_pin_loaded(monkeypatch, "3b", 15 * GB, available_gb=64)
    inner_dim, text_attention_mode = SeedVR2VideoRestorePlan._memory_profile(ModelConfig.seedvr2_3b())
    planned_chunk_bytes = SeedVR2Util.estimate_video_restore_working_set_bytes(
        frame_count=SeedVR2Util.padded_video_frame_count(plan.chunk_frame_limit),
        height=plan.target_height,
        width=plan.target_width,
        inner_dim=inner_dim,
        text_attention_mode=text_attention_mode,
    )

    assert planned_chunk_bytes > SeedVR2Util.host_safe_video_memory_budget_bytes(reserve_bytes=15 * GB)
    assert planned_chunk_bytes <= SeedVR2Util.video_restore_budget_bytes(weights_resident=True)


@pytest.mark.fast
@pytest.mark.parametrize("variant, weight_bytes", RUNTIME_CASES, ids=["3b", "7b", "3b-8bit"])
def test_post_chunk_peak_check_uses_the_same_budget(monkeypatch, variant, weight_bytes):
    plan, loaded_model = _plan_before_load_then_pin_loaded(monkeypatch, variant, weight_bytes, available_gb=64)
    budget = SeedVR2Util.video_restore_budget_bytes(weights_resident=True)
    monkeypatch.setattr(SeedVR2Util, "mlx_active_memory_bytes", staticmethod(lambda: weight_bytes))
    monkeypatch.setattr(SeedVR2Util, "mlx_cache_memory_bytes", staticmethod(lambda: 0))

    def check(working_peak_bytes: int) -> None:
        monkeypatch.setattr(
            SeedVR2Util, "mlx_peak_memory_bytes", staticmethod(lambda: weight_bytes + working_peak_bytes)
        )
        SeedVR2._assert_post_chunk_memory_health(
            loaded_model,
            true_height=plan.target_height,
            true_width=plan.target_width,
            frame_count=plan.chunk_frame_limit,
            enforce_peak_budget=True,
        )

    check(budget)
    with pytest.raises(RuntimeError, match="peak memory budget"):
        check(budget + GB)


@pytest.mark.fast
def test_runtime_budget_is_unchanged_where_0401_already_agreed(monkeypatch):
    # With memory to spare both formulas reach the 48 GB ceiling, so 0.40.1's decisions stand.
    monkeypatch.setattr(SeedVR2Util, "_host_total_memory_bytes", staticmethod(lambda: 256 * GB))
    monkeypatch.setattr(SeedVR2Util, "_host_available_memory_bytes", staticmethod(lambda: 215 * GB))

    assert (
        SeedVR2Util.video_restore_budget_bytes(weights_resident=True)
        == SeedVR2Util.host_safe_video_memory_budget_bytes(reserve_bytes=15 * GB)
        == SeedVR2Util.VIDEO_SAFE_MEMORY_CEILING_BYTES
    )


# Host states for a 1920x1072 clip on a 128 GB Mac: available memory before the weights are read,
# and the weights still to be read.
FIRST_CHUNK_HOSTS = [("3b", 15, 64), ("7b", 34, 64), ("3b", 15, 24)]


def _largest_admitted(model, *, width: int, height: int) -> int:
    admitted = 0
    for frames in range(1, 400):
        try:
            SeedVR2._assert_video_restore_memory_budget(model, frame_count=frames, true_height=height, true_width=width)
        except ValueError:
            continue
        admitted = frames
    return admitted


@pytest.mark.fast
@pytest.mark.parametrize("variant, weight_gb, available_gb", FIRST_CHUNK_HOSTS, ids=["3b-64", "7b-64", "3b-24"])
def test_a_fresh_model_first_chunk_gets_the_plans_limit(monkeypatch, variant, weight_gb, available_gb):
    _pin_host(
        monkeypatch, variant=variant, available_gb=available_gb, probe=_probe(1920, 1080, 400, 30.0, 0), probe_calls=[]
    )
    monkeypatch.setattr(
        SeedVR2Initializer,
        "estimate_resident_weight_bytes",
        staticmethod(lambda model_config, model_path=None: weight_gb * GB),
    )
    plan_limit = SeedVR2VideoRestorePlan.for_video(
        "clip.mp4", model_config=CONFIGS[variant](), force_unsafe_memory_profile=True
    ).chunk_frame_limit
    fresh = SimpleNamespace(
        model_config=CONFIGS[variant](), seedvr2_resident_weight_bytes=weight_gb * GB, seedvr2_weights_read=False
    )

    assert _largest_admitted(fresh, width=1920, height=1072) == plan_limit

    # After the first chunk the weights have been read: available memory no longer contains them,
    # and the resident budget admits the same chunk.
    monkeypatch.setattr(
        SeedVR2Util, "_host_available_memory_bytes", staticmethod(lambda: (available_gb - weight_gb) * GB)
    )
    read = SimpleNamespace(
        model_config=CONFIGS[variant](), seedvr2_resident_weight_bytes=weight_gb * GB, seedvr2_weights_read=True
    )
    assert _largest_admitted(read, width=1920, height=1072) == plan_limit


@pytest.mark.fast
def test_a_restored_chunk_records_that_the_weights_were_read(monkeypatch):
    from mflux.callbacks.callback_registry import CallbackRegistry
    from mflux.models.common.vae.tiling_config import TilingConfig
    from mflux.models.seedvr2.variants.upscale import seedvr2 as seedvr2_module

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = lambda txt, vid, timestep: mx.zeros_like(vid[:, :16])
        model.seedvr2_resident_weight_bytes = 0
        model.seedvr2_weights_read = False

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        SeedVR2Util,
        "preprocess_video_frames",
        staticmethod(lambda frames, resolution, softness: (mx.zeros((1, 3, 5, 32, 32), dtype=mx.float32), 32, 32)),
    )
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "encode",
        staticmethod(lambda vae, image, tiling_config, preserve_temporal_axis=False: mx.zeros((1, 16, 2, 4, 4))),
    )
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "decode",
        staticmethod(lambda vae, latent, tiling_config, preserve_temporal_axis=False: mx.zeros((1, 3, 5, 32, 32))),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2TextEmbeddings, "load_positive", staticmethod(lambda: mx.zeros((1, 1, 5120)))
    )
    monkeypatch.setattr(SeedVR2Util, "apply_color_correction", staticmethod(lambda content, style, mode="lab": content))
    model = SeedVR2(model_config=ModelConfig.seedvr2_3b())
    assert model.seedvr2_weights_read is False

    model._restore_video_frames(
        seed=42,
        frames=[Image.new("RGB", (32, 32)) for _ in range(5)],
        resolution=32,
        softness=0.0,
        color_correction_mode="wavelet",
        enforce_memory_budget=False,
    )

    assert model.seedvr2_weights_read is True
