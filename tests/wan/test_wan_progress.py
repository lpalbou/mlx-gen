from mflux.callbacks import ProgressEvent
from mflux.models.wan.variants import Wan2_2_TI2V


def test_wan_progress_event_exposes_fraction():
    event = ProgressEvent(phase="denoise", frame=5, total_frames=20, step=2, total_steps=10)

    assert event.progress == 0.2
    assert event.step_progress == 0.2
    assert event.frame_progress == 0.25


def test_wan_progress_frame_for_step_is_frame_based_and_leaves_final_frame_for_completion():
    frames = [
        Wan2_2_TI2V._progress_frame_for_step(step_index=step, total_steps=50, total_frames=121) for step in range(50)
    ]

    assert frames[0] == 2
    assert frames[-1] == 120
    assert frames == sorted(frames)
    assert max(frames) == 120


def test_wan_progress_frame_maps_user_reported_a14b_boundary():
    assert Wan2_2_TI2V._progress_frame_for_step(step_index=5, total_steps=20, total_frames=81) == 24


def test_wan_emit_progress_callback_receives_structured_event():
    events = []

    Wan2_2_TI2V._emit_progress(
        events.append,
        phase="complete",
        frame=121,
        total_frames=121,
        step=50,
        total_steps=50,
    )

    assert events == [ProgressEvent(phase="complete", frame=121, total_frames=121, step=50, total_steps=50)]


def test_wan_emit_progress_notifies_registry_and_direct_callback():
    class Registry:
        def __init__(self):
            self.events = []

        def emit_progress(self, event):
            self.events.append(event)

    registry = Registry()
    direct_events = []

    Wan2_2_TI2V._emit_progress(
        direct_events.append,
        registry=registry,
        task="image-to-video",
        phase="denoise",
        frame=24,
        total_frames=81,
        step=6,
        total_steps=20,
    )

    expected = ProgressEvent(
        task="image-to-video",
        phase="denoise",
        frame=24,
        total_frames=81,
        step=6,
        total_steps=20,
    )
    assert direct_events == [expected]
    assert registry.events == [expected]
