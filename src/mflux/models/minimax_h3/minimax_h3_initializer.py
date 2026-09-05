import gc
import json
from pathlib import Path

import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.lora_resolution import LoraResolution
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.streaming_weight_loader import StreamingWeightLoader
from mflux.models.minimax_h3.model.h3_audio_vae.h3_audio_vae import H3AudioVAE
from mflux.models.minimax_h3.model.h3_precision import disable_tf32
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_model import Qwen3VLModel
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_text_model import Qwen3VLTextModel
from mflux.models.minimax_h3.model.h3_transformer.h3_transformer import MiniMaxH3Transformer
from mflux.models.minimax_h3.model.h3_video_vae.h3_video_vae import H3VideoVAE
from mflux.models.minimax_h3.weights.h3_audio_vae_weights import audio_vae_kwargs
from mflux.models.minimax_h3.weights.h3_lora_mapping import MiniMaxH3LoRAMapping
from mflux.models.minimax_h3.weights.h3_video_vae_weights import video_vae_kwargs
from mflux.models.minimax_h3.weights.h3_weight_definition import MiniMaxH3WeightDefinition
from mflux.models.minimax_h3.weights.h3_weight_mapping import TEXT_ENCODER_NUM_LAYERS, MiniMaxH3WeightMapping

# PEFT adapters carry `lora_alpha` in the safetensors metadata, not as tensors; the lightx2v Turbo
# adapters are trained with alpha 8 at rank 128 (an effective 1/16 scale on `B @ A`).
DEFAULT_LORA_ALPHA = 8.0


class MiniMaxH3Initializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        disable_tf32()
        weight_definition = MiniMaxH3WeightDefinition.for_config(model_config)
        source = model_path if model_path else model_config.model_name
        root_path = PathResolution.resolve(path=source, patterns=weight_definition.get_download_patterns())
        if root_path is None:
            raise FileNotFoundError(f"MiniMax-H3 model source {source!r} did not resolve to a model root.")
        root_path = Path(root_path)

        model.model_config = model_config
        model.root_path = root_path
        model.weight_definition = weight_definition
        model.callbacks = CallbackRegistry()
        model.quantize_arg = quantize
        model.prompt_embed_cache = {}
        model.tokenizers = TokenizerLoader.load_all(
            definitions=weight_definition.get_tokenizers(), model_path=str(root_path)
        )

        MiniMaxH3Initializer._init_models(model, root_path)
        model.bits = MiniMaxH3Initializer._load_weights(model, root_path, quantize, weight_definition)
        MiniMaxH3Initializer._apply_lora(model, lora_paths=lora_paths, lora_scales=lora_scales)

    @staticmethod
    def _init_models(model, root_path: Path) -> None:
        # Every constructor defaults to the released configuration; the component config.json files refine
        # them when present (they are not part of a prepared MLX-Gen package).
        model.text_encoder = Qwen3VLModel(language_model=Qwen3VLTextModel(num_hidden_layers=TEXT_ENCODER_NUM_LAYERS))
        model.transformer = MiniMaxH3Transformer()
        vae_config = MiniMaxH3Initializer._read_json(root_path / "vae" / "config.json")
        model.vae = H3VideoVAE(**video_vae_kwargs(vae_config)) if vae_config else H3VideoVAE()
        audio_config = MiniMaxH3Initializer._read_json(root_path / "audio_vae" / "config.json")
        model.audio_vae = H3AudioVAE(**audio_vae_kwargs(audio_config)) if audio_config else H3AudioVAE()

    @staticmethod
    def _load_weights(model, root_path: Path, quantize: int | None, weight_definition: MiniMaxH3WeightDefinition):
        bits = None
        bits_resolved = False
        for component in weight_definition.get_components():
            if component.name == "text_encoder":
                component.weight_files = MiniMaxH3Initializer._text_encoder_shards(root_path / component.hf_subdir)
            post_transform = MiniMaxH3WeightMapping.convert_audio_vae_state if component.name == "audio_vae" else None
            component_bits = StreamingWeightLoader.load_into(
                model=getattr(model, component.name),
                component_root=root_path,
                component=component,
                quantize_arg=quantize,
                quantization_predicate=weight_definition.quantization_predicate,
                post_transform=post_transform,
            )
            if component.skip_quantization:
                pass
            elif not bits_resolved:
                bits, bits_resolved = component_bits, True
            elif component_bits != bits:
                raise ValueError(
                    f"MiniMax-H3 component quantization mismatch: {component.name} resolved to {component_bits}, "
                    f"but earlier components resolved to {bits}."
                )
            gc.collect()
            mx.clear_cache()
        return bits

    @staticmethod
    def _text_encoder_shards(text_encoder_root: Path) -> list[str] | None:
        """Only the shards holding the embeddings and decoder layers below 50; the rest of Qwen3-VL-32B is never read."""
        index = MiniMaxH3Initializer._read_json(text_encoder_root / "model.safetensors.index.json")
        weight_map = (index or {}).get("weight_map")
        if not weight_map:
            return None
        files = sorted(
            {file for key, file in weight_map.items() if MiniMaxH3WeightMapping.text_encoder_key_is_needed(key)}
        )
        return files or None

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        turbo_lora = model.model_config.transformer_overrides.get("turbo_lora")
        if not lora_paths and turbo_lora:
            lora_paths = [turbo_lora]
        if not lora_paths:
            model.lora_paths, model.lora_scales, model.lora_report = [], [], None
            return
        resolved_paths = LoraResolution.resolve_paths(lora_paths)
        user_scales = LoraResolution.resolve_scales(lora_scales, len(resolved_paths))
        effective_scales = [
            scale * MiniMaxH3Initializer._peft_alpha_scale(path) for path, scale in zip(resolved_paths, user_scales)
        ]
        result = LoRALoader.load_and_apply_lora_detailed(
            lora_mapping=MiniMaxH3LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=resolved_paths,
            lora_scales=effective_scales,
        )
        model.lora_paths = list(resolved_paths)
        model.lora_scales = list(user_scales)
        model.lora_report = result

    @staticmethod
    def _peft_alpha_scale(path: str) -> float:
        """`alpha / rank` of a PEFT adapter, read from the safetensors metadata (alpha) and any `lora_A` (rank)."""
        weights, metadata = mx.load(path, return_metadata=True)
        alpha = float((metadata or {}).get("alpha", DEFAULT_LORA_ALPHA))
        rank = next(
            (int(v.shape[0]) for k, v in weights.items() if k.endswith(("lora_A.weight", "lora_A.default.weight"))),
            None,
        )
        if rank is None:
            return 1.0
        return alpha / rank

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.exists():
            return None
        return json.loads(path.read_text())
