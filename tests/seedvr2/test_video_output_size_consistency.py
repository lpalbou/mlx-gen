"""Every SeedVR2 video size estimate must match the frames the VAE actually receives.

Three code paths independently computed the restored video geometry: the CLI preflight and
memory plan, the runtime's progress/noise/memory helper, and the preprocessing that builds
the tensor. They drifted. On a 480x360 source at 1x the estimators said 468x352 and 464x352
while preprocessing produced 480x352, so the streamed noise provider allocated latents 58
wide against an encode 60 wide and every streamed restore of such a source died with
"SeedVR2 streamed video noise slice shape mismatch".

Preprocessing is ground truth: it is what the VAE encodes. These tests pin the estimators to
it across sizes that are and are not already multiples of 16.
"""

import pytest
from PIL import Image

from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util
from mflux.models.seedvr2.variants.upscale.seedvr2_video_restore_plan import SeedVR2VideoRestorePlan
from mflux.utils.scale_factor import ScaleFactor
from mflux.utils.video_util import DecodedVideoClip, VideoUtil

# 480x360 is the shape that failed in the field: the width is already a multiple of 16 but
# the height is not, so any independent rounding drifts on one axis only.
SOURCE_SIZES = [(480, 360), (1920, 1080), (320, 240), (640, 480), (1280, 704), (854, 480)]


def _actual_preprocessed_size(width: int, height: int, resolution) -> tuple[int, int]:
    _, true_height, true_width = SeedVR2Util.preprocess_video_frames(
        [Image.new("RGB", (width, height))], resolution=resolution
    )
    return true_height, true_width


def _planned_size(monkeypatch, width: int, height: int, resolution) -> tuple[int, int]:
    probe = DecodedVideoClip(
        frames=[],
        fps=24.0,
        source_width=width,
        source_height=height,
        source_frame_count=17,
        source_duration_seconds=17 / 24.0,
        audio_present=False,
        clip_start_frame=0,
        clip_frame_count=1,
    )
    monkeypatch.setattr(VideoUtil, "read_video_clip", staticmethod(lambda *args, **kwargs: probe))
    plan = SeedVR2VideoRestorePlan.for_video("source.mp4", resolution=resolution, force_unsafe_memory_profile=True)
    return plan.target_height, plan.target_width


@pytest.mark.parametrize("width,height", SOURCE_SIZES)
def test_runtime_estimate_matches_preprocessing(width, height):
    resolution = ScaleFactor(value=1)
    assert SeedVR2._estimate_output_size(
        source_width=width, source_height=height, resolution=resolution
    ) == _actual_preprocessed_size(width, height, resolution)


@pytest.mark.parametrize("width,height", SOURCE_SIZES)
def test_plan_estimate_matches_preprocessing(monkeypatch, width, height):
    resolution = ScaleFactor(value=1)
    assert _planned_size(monkeypatch, width, height, resolution) == _actual_preprocessed_size(width, height, resolution)


@pytest.mark.parametrize("scale", [1, 2])
def test_estimates_agree_with_each_other_across_scales(monkeypatch, scale):
    resolution = ScaleFactor(value=scale)
    for width, height in SOURCE_SIZES:
        runtime = SeedVR2._estimate_output_size(source_width=width, source_height=height, resolution=resolution)
        planned = _planned_size(monkeypatch, width, height, resolution)
        assert runtime == planned, f"{width}x{height} at {scale}x: runtime {runtime} vs plan {planned}"


def test_estimated_size_yields_latent_dimensions_the_encoder_will_produce():
    # The concrete regression: latent width must be 60, not 58, for a 480x360 source at 1x.
    resolution = ScaleFactor(value=1)
    height, width = SeedVR2._estimate_output_size(source_width=480, source_height=360, resolution=resolution)
    assert (width // SeedVR2Util.LATENT_SPATIAL_SCALE, height // SeedVR2Util.LATENT_SPATIAL_SCALE) == (60, 44)
