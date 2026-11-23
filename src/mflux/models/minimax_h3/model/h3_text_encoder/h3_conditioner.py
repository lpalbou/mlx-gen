import gc
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mflux.models.common.weights.loading.streaming_weight_loader import StreamingWeightLoader
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_model import Qwen3VLModel
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_text_model import Qwen3VLTextModel
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_vision_model import Qwen3VLVisionModel
from mflux.models.minimax_h3.weights.h3_weight_mapping import TEXT_ENCODER_NUM_LAYERS, MiniMaxH3WeightMapping


class H3Conditioner:
    # The Qwen3-VL conditioner, read from disk for the length of one encode. Its layers are loaded, run and
    # released one at a time, so the largest component after the transformer never stands in memory: an
    # encode holds one layer and the token embeddings, plus the vision tower when there is a keyframe.
    COMPONENT_NAME = "text_encoder"

    def __init__(self, root_path: Path, weight_definition, quantize: int | None):
        self.root_path = Path(root_path)
        self.weight_definition = weight_definition
        self.quantize = quantize
        self.spatial_merge_size = Qwen3VLVisionModel.SPATIAL_MERGE_SIZE

    def encode(
        self,
        token_ids: np.ndarray,
        image_patches: list[np.ndarray] | None = None,
        image_grids: list[tuple[int, int, int]] | None = None,
    ) -> mx.array:
        model = H3Conditioner.build()
        StreamingWeightLoader.load_into(
            model=model,
            component_root=self.root_path,
            component=self.component(),
            quantize_arg=self.quantize,
            quantization_predicate=self.weight_definition.quantization_predicate,
            evaluate=False,
        )
        embeds = model.encode(token_ids, image_patches, image_grids, release_layers=True)
        mx.eval(embeds)
        del model
        gc.collect()
        mx.clear_cache()
        return embeds

    def component(self) -> ComponentDefinition:
        component = next(
            component
            for component in self.weight_definition.get_components()
            if component.name == H3Conditioner.COMPONENT_NAME
        )
        component.weight_files = H3Conditioner.needed_shards(self.root_path / component.hf_subdir)
        return component

    def stored_bits(self) -> int | None:
        # The width a prepared package records for the conditioner; None for a source snapshot or bf16.
        return StreamingWeightLoader.stored_bits(self.root_path / H3Conditioner.COMPONENT_NAME)

    @staticmethod
    def build() -> Qwen3VLModel:
        return Qwen3VLModel(language_model=Qwen3VLTextModel(num_hidden_layers=TEXT_ENCODER_NUM_LAYERS))

    @staticmethod
    def needed_shards(text_encoder_root: Path) -> list[str] | None:
        # Only the shards holding the embeddings and decoder layers below 50; the rest of Qwen3-VL-32B is never read.
        index_path = text_encoder_root / "model.safetensors.index.json"
        if not index_path.exists():
            return None
        weight_map = json.loads(index_path.read_text()).get("weight_map")
        if not weight_map:
            return None
        files = sorted(
            {file for key, file in weight_map.items() if MiniMaxH3WeightMapping.text_encoder_key_is_needed(key)}
        )
        return files or None
