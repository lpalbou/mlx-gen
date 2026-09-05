"""Qwen3-VL conditioner for MiniMax-H3: the vision tower plus the first 50 decoder layers.

`encode` takes a tokenized presentation (text with optional `<|image_pad|>` blocks) and returns
`hidden_states[50]`, the pre-norm output of decoder layer 49, exactly as the reference pipeline reads it.
"""

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_text_model import Qwen3VLTextModel
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_vision_model import Qwen3VLVisionModel

IMAGE_TOKEN_ID = 151655
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653


def rope_index(
    token_ids: np.ndarray, image_token_id: int, image_grids: list[tuple[int, int, int]], merge_size: int
) -> np.ndarray:
    """Qwen3-VL `get_rope_index` for one sequence: `(3, L)` positions, text runs shared across the three axes,
    image runs laid out on a (t, h, w) grid, every run starting where the previous one ended."""
    is_image = token_ids == image_token_id
    positions = np.zeros((3, len(token_ids)), dtype=np.int64)
    current = 0
    grids = iter(image_grids)
    start = 0
    while start < len(token_ids):
        end = start
        while end < len(token_ids) and is_image[end] == is_image[start]:
            end += 1
        if not is_image[start]:
            positions[:, start:end] = np.arange(end - start) + current
            current += end - start
        else:
            t, h, w = next(grids)
            grid_t, grid_h, grid_w = t, h // merge_size, w // merge_size
            tt, hh, ww = np.meshgrid(np.arange(grid_t), np.arange(grid_h), np.arange(grid_w), indexing="ij")
            positions[0, start:end] = tt.reshape(-1) + current
            positions[1, start:end] = hh.reshape(-1) + current
            positions[2, start:end] = ww.reshape(-1) + current
            current += max(grid_h, grid_w)
        start = end
    return positions


class Qwen3VLModel(nn.Module):
    def __init__(self, language_model: Qwen3VLTextModel | None = None, visual: Qwen3VLVisionModel | None = None):
        super().__init__()
        self.language_model = language_model or Qwen3VLTextModel()
        self.visual = visual or Qwen3VLVisionModel()

    def encode(
        self,
        token_ids: np.ndarray,
        image_patches: list[np.ndarray] | None = None,
        image_grids: list[tuple[int, int, int]] | None = None,
        image_token_id: int = IMAGE_TOKEN_ID,
    ) -> mx.array:
        """`(1, L, hidden)` hidden state after the last loaded decoder layer, with image tokens filled in."""
        ids = mx.array(np.asarray(token_ids, dtype=np.int32))[None]
        embeds = self.language_model.embed_tokens(ids)
        deepstack = None
        visual_positions = None
        if image_patches:
            merged, features = [], []
            for patches, grid in zip(image_patches, image_grids):
                tokens, deep = self.visual(mx.array(patches), grid)
                merged.append(tokens)
                features.append(deep)
            image_embeds = mx.concatenate(merged, axis=0)
            deepstack = [mx.concatenate([f[stage] for f in features], axis=0) for stage in range(len(features[0]))]
            visual_positions = mx.array(np.flatnonzero(np.asarray(token_ids) == image_token_id).astype(np.int32))
            if int(visual_positions.shape[0]) != int(image_embeds.shape[0]):
                raise ValueError(
                    f"{int(visual_positions.shape[0])} image tokens but {int(image_embeds.shape[0])} image features."
                )
            embeds[:, visual_positions, :] = image_embeds.astype(embeds.dtype)[None]
        positions = rope_index(
            np.asarray(token_ids), image_token_id, list(image_grids or []), self.visual.spatial_merge_size
        )
        return self.language_model(
            ids,
            position_ids=mx.array(positions.astype(np.int32))[:, None, :],
            inputs_embeds=embeds,
            deepstack_visual_embeds=deepstack,
            visual_positions=visual_positions,
        )
