from types import SimpleNamespace

import mlx.core as mx
from PIL import Image

from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.utils.image_util import ImageUtil


class _FakeGeneratedImage:
    def __init__(self, color: str):
        self.image = Image.new("RGB", (4, 4), color=color)

    def save(self, path, export_json_metadata=False):
        del export_json_metadata
        self.image.save(path)


def test_stepwise_handler_resets_retained_images_between_outputs(monkeypatch, tmp_path):
    composites: list[int] = []
    colors = iter(("red", "blue", "green"))

    class FakeLatentCreator:
        @staticmethod
        def unpack_latents(latents, height, width):
            del height, width
            return latents

    class FakeVAE:
        @staticmethod
        def decode(latents):
            return latents

    class FakeModel:
        bits = 8
        lora_paths = []
        lora_scales = []
        vae = FakeVAE()

    def fake_to_image(**kwargs):
        del kwargs
        return _FakeGeneratedImage(next(colors))

    def fake_composite(images):
        composites.append(len(images))
        return Image.new("RGB", (4 * len(images), 4), color="white")

    monkeypatch.setattr(ImageUtil, "to_image", fake_to_image)
    monkeypatch.setattr(ImageUtil, "to_composite_pil_images", fake_composite)

    handler = StepwiseHandler(
        model=FakeModel(),
        output_dir=str(tmp_path),
        latent_creator=FakeLatentCreator,
    )
    config = SimpleNamespace(height=4, width=4, num_inference_steps=3, init_time_step=0)
    time_steps = SimpleNamespace(format_dict={"elapsed": 0.0})
    latents = mx.zeros((1, 1, 1, 1))

    handler.call_before_loop(seed=1, prompt="first", latents=latents, config=config)
    handler.call_in_loop(t=0, seed=1, prompt="first", latents=latents, config=config, time_steps=time_steps)

    assert composites[-1] == 2
    with Image.open(tmp_path / "seed_1_composite.png") as composite:
        assert composite.size == (8, 4)

    handler.call_before_loop(seed=2, prompt="second", latents=latents, config=config)

    assert composites[-1] == 1
    assert len(handler.step_wise_images) == 1
    with Image.open(tmp_path / "seed_2_composite.png") as composite:
        assert composite.size == (4, 4)
