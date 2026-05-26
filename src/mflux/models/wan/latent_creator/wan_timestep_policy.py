import mlx.core as mx


class WanTimestepPolicy:
    @staticmethod
    def expand_for_text_to_video(
        *,
        latent_shape: tuple[int, int, int, int, int],
        timestep: int | float | mx.array,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ) -> mx.array:
        batch_size, _, frames, height, width = latent_shape
        mask = mx.ones((1, 1, frames, height, width), dtype=mx.float32)
        return WanTimestepPolicy.expand_from_mask(mask=mask, batch_size=batch_size, timestep=timestep, patch_size=patch_size)

    @staticmethod
    def first_frame_mask(
        *,
        latent_shape: tuple[int, int, int, int, int],
    ) -> mx.array:
        _, _, frames, height, width = latent_shape
        tail = mx.ones((1, 1, max(frames - 1, 0), height, width), dtype=mx.float32)
        first = mx.zeros((1, 1, 1, height, width), dtype=mx.float32)
        return mx.concatenate([first, tail], axis=2)

    @staticmethod
    def apply_first_frame_condition(
        *,
        latents: mx.array,
        condition: mx.array,
        first_frame_mask: mx.array,
    ) -> mx.array:
        return (1 - first_frame_mask) * condition + first_frame_mask * latents

    @staticmethod
    def expand_from_mask(
        *,
        mask: mx.array,
        batch_size: int,
        timestep: int | float | mx.array,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ) -> mx.array:
        _, patch_height, patch_width = patch_size
        WanTimestepPolicy._validate_mask(mask)
        scalar = mx.array(timestep, dtype=mx.float32)
        timestep_tokens = (mask[0, 0, :, ::patch_height, ::patch_width] * scalar).reshape(-1)
        return mx.broadcast_to(timestep_tokens.reshape(1, -1), (batch_size, timestep_tokens.shape[0]))

    @staticmethod
    def _validate_mask(mask: mx.array) -> tuple[int, int, int, int, int]:
        if mask.ndim != 5:
            raise ValueError(f"Wan timestep masks must have shape [B, C, F, H, W], got {mask.shape}")
        return mask.shape
