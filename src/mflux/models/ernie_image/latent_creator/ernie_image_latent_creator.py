import mlx.core as mx

from mflux.models.common.config import ModelConfig


class ErnieImageLatentCreator:
    @staticmethod
    def create_noise(seed: int, height: int, width: int, batch_size: int = 1) -> mx.array:
        return mx.random.normal(
            shape=[
                batch_size,
                128,
                height // 16,
                width // 16,
            ],
            key=mx.random.key(seed),
        ).astype(ModelConfig.precision)

    @staticmethod
    def pack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        return latents

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        return latents
