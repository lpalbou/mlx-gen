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

    assert [event.phase for event in all_events] == ["start", "denoise", "complete"]
    assert [event.task for event in all_events] == ["image-to-image"] * 3
    assert [event.total_steps for event in all_events] == [3, 3, 3]
    assert [event.step for event in all_events] == [0, 1, 3]
    assert img2img_events == all_events
    assert text_events == []


def test_progress_subscription_can_be_removed():
    registry = CallbackRegistry()
    events = []
    unsubscribe = registry.subscribe_progress(events.append)
    unsubscribe()

    ctx = registry.start(seed=7, prompt="prompt", config=FakeConfig())
    ctx.before_loop(latents=object())
    ctx.after_loop(latents=object())

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
