import mlx.core as mx
import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.seedvr2 import SeedVR2VideoRestorePlan, SeedVR2VideoRestoreRefused
from mflux.models.seedvr2.cli import seedvr2_upscale
from mflux.models.seedvr2.seedvr2_initializer import SeedVR2Initializer
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util
from mflux.utils.video_util import DecodedVideoClip, VideoUtil

GB = 1000**3

# Working memory of one restored chunk (MLX peak minus the 7.28 GB of resident weights, the largest over the run),
# SeedVR2 3B (numz/SeedVR2_comfyUI), MLX 0.32.5, 2026-11-26, with CausalConv3d summing its taps per frame: the full
# public path (SeedVR2VideoRestorePlan, MemorySaver from the plan, restore_video_to_path with the plan's values) on
# a synthetic clip, seed 42, 1x, one fresh process per point, run as one chunk of the given padded length. 3B only:
# the 7B variants use the same VAE, but their working set was not measured.
# Columns: output width, output height, padded chunk frames, working GB.
MEASURED_MLX_VERSION = "0.32.5"
MEASURED_WORKING_SETS = [
    (64, 64, 5, 0.64),
    (64, 64, 17, 0.65),
    (96, 64, 5, 0.66),
    (96, 64, 17, 0.67),
    (160, 96, 5, 1.24),
    (160, 96, 17, 1.26),
    (144, 112, 5, 1.3),
    (144, 112, 17, 1.32),
    (128, 128, 5, 0.6),
    (128, 128, 17, 0.69),
    (320, 192, 1, 0.71),
    (320, 192, 5, 1.44),
    (320, 192, 9, 1.92),
    (320, 192, 17, 1.93),
    (320, 192, 29, 1.94),
    (320, 192, 33, 1.94),
    (320, 192, 49, 1.95),
    (320, 192, 97, 1.99),
    (480, 272, 1, 1.4),
    (480, 272, 9, 4.07),
    (480, 272, 17, 4.08),
    (480, 272, 49, 4.14),
    (480, 272, 97, 4.22),
    (640, 352, 1, 2.4),
    (640, 352, 5, 5.25),
    (640, 352, 9, 7.01),
    (640, 352, 17, 7.03),
    (640, 352, 29, 7.07),
    (640, 352, 33, 7.08),
    (640, 352, 37, 7.09),
    (640, 352, 41, 7.1),
    (640, 352, 45, 7.12),
    (640, 352, 49, 7.13),
    (640, 352, 97, 7.27),
    (960, 528, 17, 15.8),
    (1280, 720, 1, 9.76),
    (1280, 720, 17, 28.72),
    (1280, 720, 29, 28.86),
    (1280, 720, 49, 29.1),
]

# The plan's answer for a 1920x1080 clip on a 48 GB budget.
LARGEST_FIT_1080P = "The largest output that fits is 1552x880 (1365760 pixels per frame): use --resolution 880"
SUGGESTED_RESOLUTION_1080P = 880
SUGGESTED_SIZE_1080P = (1552, 880)


def _estimate_gb(width: int, height: int, frames: int) -> float:
    return (
        SeedVR2VideoRestorePlan._working_set_bytes(
            ModelConfig.seedvr2_3b(), frame_count=frames, height=height, width=width
        )
        / GB
    )


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".")[:3])


@pytest.mark.fast
def test_installed_mlx_is_the_one_the_vae_memory_model_was_measured_on():
    low, high = SeedVR2Util.VIDEO_VAE_MEMORY_MLX_RANGE
    assert low <= _version(mx.__version__) < high, (
        f"MLX {mx.__version__} is outside the range the SeedVR2 video VAE memory model was measured on "
        f"({'.'.join(map(str, low))} to below {'.'.join(map(str, high))}, table taken on {MEASURED_MLX_VERSION}). "
        "Re-measure the working sets on this MLX and update VIDEO_VAE_MEMORY_BUCKETS before widening the range."
    )


@pytest.mark.fast
@pytest.mark.parametrize("width, height, frames, measured_gb", MEASURED_WORKING_SETS)
def test_estimate_covers_every_measured_working_set_within_thirty_percent(width, height, frames, measured_gb):
    estimate = _estimate_gb(width, height, frames)

    assert measured_gb <= estimate
    # Below 320x192 the per-pixel rows were not measured and the small-frame floor applies; there the estimate is
    # only required to cover the measurement.
    if width * height >= SeedVR2Util.VIDEO_VAE_SMALL_FRAME_FLOORS[-1][0]:
        assert estimate <= 1.3 * measured_gb


@pytest.mark.fast
def test_the_small_frame_floors_cover_every_small_frame_and_admit_the_tiny_ones_that_fit_1_gb():
    # Below 16,384 output pixels the decoder's half-resolution layers keep MLX's single call: 160x96 and 144x112
    # need more than 128x128. The floors are fitted to these points, not a flat value: 64x64, 96x64 and 128x128
    # work in under 1.0 GB, the minimum budget, and their estimate stays under it.
    small = [
        (w, h, f, gb)
        for w, h, f, gb in MEASURED_WORKING_SETS
        if w * h < SeedVR2Util.VIDEO_VAE_SMALL_FRAME_FLOORS[-1][0]
    ]
    assert {(w, h) for w, h, _, _ in small} >= {(64, 64), (96, 64), (160, 96), (144, 112), (128, 128)}
    for width, height, frames, gb in small:
        assert _estimate_gb(width, height, frames) >= gb
        if gb < 0.95:
            assert _estimate_gb(width, height, frames) <= 1.0


@pytest.mark.fast
def test_the_memory_table_is_the_measured_one():
    # Rows and boundaries come from the measurements above (29 / 33 / 49 / 97 frames); change them only with new
    # measurements.
    assert SeedVR2Util.VIDEO_VAE_MEMORY_BUCKETS == (
        (1, 12_000),
        (5, 25_000),
        (29, 32_000),
        (49, 30_000),
        (None, 28_000),
    )
    assert SeedVR2Util.VIDEO_VAE_SMALL_FRAME_FLOORS == ((16_384, 450_000_000, 56_000), (61_440, 300_000_000, 25_000))


def _probe(width: int, height: int, frames: int) -> DecodedVideoClip:
    return DecodedVideoClip(
        frames=[],
        fps=24.0,
        source_width=width,
        source_height=height,
        source_frame_count=frames,
        source_duration_seconds=None,
        audio_present=False,
        clip_start_frame=0,
        clip_frame_count=1,
    )


def _pin_host(monkeypatch, *, probe: DecodedVideoClip, available_gb: int = 64, weight_gb: int = 15) -> None:
    monkeypatch.setattr(SeedVR2Util, "_host_total_memory_bytes", staticmethod(lambda: 128 * GB))
    monkeypatch.setattr(SeedVR2Util, "_host_available_memory_bytes", staticmethod(lambda: available_gb * GB))
    monkeypatch.setattr(
        SeedVR2Initializer,
        "estimate_resident_weight_bytes",
        staticmethod(lambda model_config, model_path=None: weight_gb * GB),
    )
    monkeypatch.setattr(VideoUtil, "read_video_clip", staticmethod(lambda *args, **kwargs: probe))


@pytest.mark.fast
@pytest.mark.parametrize("width, height, target", [(640, 360, (640, 352)), (1280, 720, (1280, 720))])
def test_frames_up_to_720p_are_admitted_on_the_48_gb_budget(monkeypatch, width, height, target):
    # Budget min(48, (128 - 15) / 2) = 48 GB. Before the per-frame convolution taps, both were refused.
    _pin_host(monkeypatch, probe=_probe(width, height, 17), available_gb=128)

    plan = SeedVR2VideoRestorePlan.for_video("clip.mp4")

    assert (plan.target_width, plan.target_height) == target
    assert (plan.effective_chunk_size, plan.effective_chunk_overlap, plan.route_reason) == (17, 0, "safe_streaming")
    assert plan.warnings == ()


@pytest.mark.fast
def test_a_long_clip_runs_as_one_chunk_when_it_fits(monkeypatch):
    _pin_host(monkeypatch, probe=_probe(480, 272, 97), available_gb=128)

    plan = SeedVR2VideoRestorePlan.for_video("clip.mp4")

    assert (plan.effective_chunk_size, plan.effective_chunk_overlap, plan.chunk_frame_limit) == (97, 0, 97)


@pytest.mark.fast
def test_a_frame_too_large_for_the_budget_is_refused_whatever_the_chunk(monkeypatch):
    _pin_host(monkeypatch, probe=_probe(1920, 1080, 17), available_gb=128)

    with pytest.raises(SeedVR2VideoRestoreRefused) as refused:
        SeedVR2VideoRestorePlan.for_video("clip.mp4")

    assert refused.value.reason == SeedVR2VideoRestoreRefused.BEYOND_MEMORY_PROFILE
    message = str(refused.value)
    assert message.startswith(
        "SeedVR2 video restore cannot fit a 1920x1072 output frame in this machine's video memory budget (48.0 GB)"
    )
    assert "whatever the temporal chunk size" in message
    assert LARGEST_FIT_1080P in message
    assert "--force-unsafe-video-memory" in message
    for chunk in (29, 49):
        with pytest.raises(SeedVR2VideoRestoreRefused):
            SeedVR2VideoRestorePlan.for_video("clip.mp4", temporal_chunk_size=chunk)


@pytest.mark.fast
def test_the_suggested_resolution_runs(monkeypatch):
    _pin_host(monkeypatch, probe=_probe(1920, 1080, 17), available_gb=128)

    plan = SeedVR2VideoRestorePlan.for_video("clip.mp4", resolution=SUGGESTED_RESOLUTION_1080P)

    assert (plan.target_width, plan.target_height) == SUGGESTED_SIZE_1080P
    assert _estimate_gb(*SUGGESTED_SIZE_1080P, plan.effective_chunk_size) <= 48


@pytest.mark.fast
def test_force_unsafe_memory_profile_keeps_its_meaning(monkeypatch):
    _pin_host(monkeypatch, probe=_probe(1920, 1080, 17), available_gb=128)

    plan = SeedVR2VideoRestorePlan.for_video("clip.mp4", force_unsafe_memory_profile=True)

    assert plan.route_reason == "unsafe_override"
    assert plan.enforce_memory_budget is False
    assert (plan.target_width, plan.target_height, plan.effective_chunk_size) == (1920, 1072, 17)


@pytest.mark.fast
def test_a_frame_that_fits_keeps_the_plan_it_had(monkeypatch):
    _pin_host(monkeypatch, probe=_probe(320, 192, 17))

    plan = SeedVR2VideoRestorePlan.for_video("clip.mp4")

    assert (plan.effective_chunk_size, plan.effective_chunk_overlap, plan.route_reason) == (17, 0, "safe_streaming")


@pytest.mark.fast
def test_the_cli_refuses_a_frame_too_large_before_any_load(monkeypatch, tmp_path, capsys):
    clip = tmp_path / "clip.mp4"
    clip.touch()
    _pin_host(monkeypatch, probe=_probe(1920, 1080, 17), available_gb=128)

    def no_load(*args, **kwargs):
        raise AssertionError("the model must not be loaded for a refused plan")

    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", no_load)
    monkeypatch.setattr("sys.argv", ["mflux-upscale-seedvr2", "--video-path", str(clip)])

    with pytest.raises(SystemExit) as exit_info:
        seedvr2_upscale.main()

    assert exit_info.value.code == 2
    last_line = capsys.readouterr().err.rstrip().splitlines()[-1]
    assert last_line.startswith(
        "mflux-upscale-seedvr2: error: SeedVR2 video restore cannot fit a 1920x1072 output frame"
    )


@pytest.mark.fast
def test_the_runtime_check_counts_the_vae(monkeypatch):
    # Budget (64 - 15) / 2 = 24.5 GB with the weights already read: 640x352 fits, 1920x1072 does not.
    _pin_host(monkeypatch, probe=_probe(640, 360, 17), available_gb=64 - 15)
    loaded_model = type("Loaded", (), {"model_config": ModelConfig.seedvr2_3b()})()

    with pytest.raises(ValueError, match="exceeds the supported host-safe memory budget"):
        SeedVR2._assert_video_restore_memory_budget(loaded_model, frame_count=17, true_height=1072, true_width=1920)
    SeedVR2._assert_video_restore_memory_budget(loaded_model, frame_count=17, true_height=352, true_width=640)
    SeedVR2._assert_video_restore_memory_budget(loaded_model, frame_count=17, true_height=192, true_width=320)


@pytest.mark.fast
def test_the_image_path_does_not_use_the_video_estimate(monkeypatch, tmp_path):
    image = tmp_path / "frame.png"
    image.touch()

    def video_estimate_called(**kwargs):
        raise AssertionError("the image route must not use the video working-set estimate")

    class _StopAfterImageCall(Exception):
        pass

    class _ImageOnlySeedVR2:
        def __init__(self, *, quantize, model_path, model_config):
            self.model_config = model_config
            self.tiling_config = None

        def generate_image(self, **kwargs):
            raise _StopAfterImageCall

    monkeypatch.setattr(SeedVR2Util, "estimate_video_restore_working_set_bytes", staticmethod(video_estimate_called))
    monkeypatch.setattr(seedvr2_upscale, "SeedVR2", _ImageOnlySeedVR2)
    monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr("sys.argv", ["mflux-upscale-seedvr2", "--image-path", str(image), "--resolution", "2x"])

    with pytest.raises(_StopAfterImageCall):
        seedvr2_upscale.main()
