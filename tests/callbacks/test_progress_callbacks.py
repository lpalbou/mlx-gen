from mflux.callbacks import (
    ProgressEvent as PublicProgressEvent,
    generation_context,
)
from mflux.callbacks.callback_registry import CallbackRegistry


class FakeConfig:
    def __init__(
        self,
        *,
        num_inference_steps: int = 4,
        init_time_step: int = 0,
        image_path=None,
        image_strength=None,
    ):
        self.num_inference_steps = num_inference_steps
        self.init_time_step = init_time_step
        self.image_path = image_path
        self.image_strength = image_strength


def test_public_progress_event_export_matches_shared_event():
    from mflux.callbacks.progress import ProgressEvent

    assert PublicProgressEvent is ProgressEvent


def test_image_progress_subscription_receives_lightweight_lifecycle_events():
    registry = CallbackRegistry()
    events = []
    registry.subscribe_progress(events.append)

    ctx = registry.start(seed=7, prompt="prompt", config=FakeConfig(num_inference_steps=4))
    ctx.before_loop(latents=object())
    ctx.in_loop(0, latents=object(), time_steps=object())
    ctx.in_loop(1, latents=object(), time_steps=object())
    ctx.after_loop(latents=object())
    ctx.complete()

    assert [event.phase for event in events] == ["start", "denoise", "denoise", "complete"]
    assert [event.step for event in events] == [0, 1, 2, 4]
    assert all(event.total_steps == 4 for event in events)
    assert all(event.task == "text-to-image" for event in events)
    assert events[1].progress == 0.25
    assert events[1].frame is None
    assert events[1].frame_progress is None


def test_image_progress_infers_image_to_image_and_filters_subscribers():
    registry = CallbackRegistry()
    all_events = []
    img2img_events = []
    text_events = []
    registry.subscribe_progress(all_events.append)
    registry.subscribe_progress(img2img_events.append, task="image-to-image")
    registry.subscribe_progress(text_events.append, task="text-to-image")

    ctx = registry.start(
        seed=7,
        prompt="prompt",
        config=FakeConfig(num_inference_steps=5, init_time_step=2, image_path="input.png", image_strength=0.6),
    )
    ctx.before_loop(latents=object())
    ctx.in_loop(2, latents=object(), time_steps=object())
    ctx.after_loop(latents=object())
    ctx.complete()

    assert [event.phase for event in all_events] == ["start", "denoise", "complete"]
    assert [event.task for event in all_events] == ["image-to-image"] * 3
    assert [event.total_steps for event in all_events] == [3, 3, 3]
    assert [event.step for event in all_events] == [0, 1, 3]
    assert img2img_events == all_events
    assert text_events == []


def test_image_progress_reports_text_to_image_when_image_strength_is_nonpositive():
    registry = CallbackRegistry()
    all_events = []
    img2img_events = []
    text_events = []
    registry.subscribe_progress(all_events.append)
    registry.subscribe_progress(img2img_events.append, task="image-to-image")
    registry.subscribe_progress(text_events.append, task="text-to-image")

    ctx = registry.start(
        seed=7,
        prompt="prompt",
        config=FakeConfig(num_inference_steps=5, image_path="input.png", image_strength=0.0),
    )
    ctx.before_loop(latents=object())
    ctx.after_loop(latents=object())
    ctx.complete()

    assert [event.task for event in all_events] == ["text-to-image", "text-to-image"]
    assert text_events == all_events
    assert img2img_events == []


def test_image_progress_uses_explicit_task_for_edit_conditioned_generation():
    registry = CallbackRegistry()
    all_events = []
    img2img_events = []
    text_events = []
    registry.subscribe_progress(all_events.append)
    registry.subscribe_progress(img2img_events.append, task="image-to-image")
    registry.subscribe_progress(text_events.append, task="text-to-image")

    ctx = registry.start(
        seed=7,
        prompt="prompt",
        config=FakeConfig(num_inference_steps=4, image_path="input.png", image_strength=None),
        task="image-to-image",
    )
    ctx.before_loop(latents=object())
    ctx.after_loop(latents=object())
    ctx.complete()

    assert [event.task for event in all_events] == ["image-to-image", "image-to-image"]
    assert img2img_events == all_events
    assert text_events == []


def test_progress_subscribers_receive_distinct_terminal_sequences_for_serial_reuse():
    registry = CallbackRegistry()
    events = []
    registry.subscribe_progress(events.append)

    first = registry.start(seed=101, prompt="first", config=FakeConfig(num_inference_steps=3))
    first.before_loop(latents=object())
    first.in_loop(0, latents=object(), time_steps=object())
    first.after_loop(latents=object())
    first.complete()

    second = registry.start(seed=202, prompt="second", config=FakeConfig(num_inference_steps=2))
    second.before_loop(latents=object())
    second.in_loop(0, latents=object(), time_steps=object())
    second.after_loop(latents=object())
    second.complete()

    assert [event.phase for event in events] == [
        "start",
        "denoise",
        "complete",
        "start",
        "denoise",
        "complete",
    ]
    assert [event.step for event in events] == [0, 1, 3, 0, 1, 2]


def test_progress_subscription_can_be_removed():
    registry = CallbackRegistry()
    events = []
    unsubscribe = registry.subscribe_progress(events.append)
    unsubscribe()

    ctx = registry.start(seed=7, prompt="prompt", config=FakeConfig())
    ctx.before_loop(latents=object())
    ctx.after_loop(latents=object())
    ctx.complete()

    assert events == []


def test_image_progress_listener_evaluates_latents_before_denoise_event(monkeypatch):
    registry = CallbackRegistry()
    order = []
    registry.subscribe_progress(lambda event: order.append(event.phase))
    monkeypatch.setattr(generation_context.mx, "eval", lambda latents: order.append("eval"))

    ctx = registry.start(seed=7, prompt="prompt", config=FakeConfig(num_inference_steps=2))
    ctx.before_loop(latents=object())
    ctx.in_loop(0, latents=object(), time_steps=object())

    assert order == ["start", "eval", "denoise"]


def test_image_progress_does_not_force_eval_without_listener(monkeypatch):
    registry = CallbackRegistry()
    eval_calls = []
    monkeypatch.setattr(generation_context.mx, "eval", lambda latents: eval_calls.append(latents))

    ctx = registry.start(seed=7, prompt="prompt", config=FakeConfig(num_inference_steps=2))
    ctx.before_loop(latents=object())
    ctx.in_loop(0, latents=object(), time_steps=object())

    assert eval_calls == []


def test_after_loop_is_not_terminal_and_failed_is_terminal():
    registry = CallbackRegistry()
    events = []
    registry.subscribe_progress(events.append)

    ctx = registry.start(seed=7, prompt="prompt", config=FakeConfig(num_inference_steps=2))
    ctx.before_loop(latents=object())
    ctx.after_loop(latents=object())
    ctx.failed()
    ctx.complete()
    ctx.failed()

    assert [event.phase for event in events] == ["start", "failed"]


def test_terminal_events_are_mutually_exclusive():
    registry = CallbackRegistry()
    events = []
    registry.subscribe_progress(events.append)

    ctx = registry.start(seed=7, prompt="prompt", config=FakeConfig(num_inference_steps=2))
    ctx.before_loop(latents=object())
    ctx.complete()
    ctx.interruption(0, latents=object(), time_steps=object())
    ctx.failed()

    assert [event.phase for event in events] == ["start", "complete"]


def test_after_loop_exception_emits_failed_terminal():
    registry = CallbackRegistry()
    events = []
    registry.subscribe_progress(events.append)

    class RaisingAfterLoop:
        @staticmethod
        def call_after_loop(**kwargs):
            raise RuntimeError("after loop failed")

    registry.register(RaisingAfterLoop())

    ctx = registry.start(seed=7, prompt="prompt", config=FakeConfig(num_inference_steps=2))
    ctx.before_loop(latents=object())

    try:
        ctx.after_loop(latents=object())
    except RuntimeError as exc:
        assert str(exc) == "after loop failed"
    else:
        raise AssertionError("expected after_loop failure")

    assert [event.phase for event in events] == ["start", "failed"]
