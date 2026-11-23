import gc
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.lora_resolution import LoraResolution
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.streaming_weight_loader import StreamingWeightLoader
from mflux.models.minimax_h3.model.h3_audio_vae.h3_audio_vae import H3AudioVAE
from mflux.models.minimax_h3.model.h3_precision import disable_tf32
from mflux.models.minimax_h3.model.h3_text_encoder.h3_conditioner import H3Conditioner
from mflux.models.minimax_h3.model.h3_transformer.h3_block_stream import H3BlockStream
from mflux.models.minimax_h3.model.h3_transformer.h3_modulation_tables import H3ModulationTables
from mflux.models.minimax_h3.model.h3_transformer.h3_transformer import MiniMaxH3Transformer
from mflux.models.minimax_h3.model.h3_video_vae.h3_video_vae import H3VideoVAE
from mflux.models.minimax_h3.weights.h3_audio_vae_weights import audio_vae_kwargs
from mflux.models.minimax_h3.weights.h3_lora_mapping import MiniMaxH3LoRAMapping
from mflux.models.minimax_h3.weights.h3_video_vae_weights import video_vae_kwargs
from mflux.models.minimax_h3.weights.h3_weight_definition import MiniMaxH3WeightDefinition
from mflux.models.minimax_h3.weights.h3_weight_mapping import MiniMaxH3WeightMapping
from mflux.utils.runtime_memory import RuntimeMemory

_LORA_DOWN_KEY = re.compile(r"\.lora_(A|down)(\.[^.]+)?\.weight$")


class MiniMaxH3Initializer:
    # The markers MiniMax-H3 adds to the Qwen vocabulary: dialogue tags, the cutoff, lyrics and captions.
    MARKER_TOKENS = (
        "<d>",
        "</d>",
        "<|cutoff|>",
        "<|lyrics_start|>",
        "<|lyrics_end|>",
        "<|caption_start|>",
        "<|caption_end|>",
    )
    # Above this share of physical memory the weights alone leave no room for the rest of a run: activations,
    # the MLX cache and the OS want roughly 18 GiB, which is 14% of a 128 GiB machine.
    WEIGHT_BUDGET_SHARE = 0.85
    TRANSFORMER = "transformer"
    DECODERS = ("vae", "audio_vae")
    # A quantized group of 64 weights also stores a bf16 scale and a bf16 bias: half a bit per weight.
    QUANTIZATION_OVERHEAD_BITS = 0.5

    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        stream_blocks: bool = False,
    ) -> None:
        disable_tf32()
        MiniMaxH3WeightDefinition.require_supported_bits(quantize)
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

        MiniMaxH3Initializer._require_marker_tokens(model.tokenizers, root_path)
        # What stands in memory is the transformer without its AdaLN projections. The conditioner is read
        # for the length of an encode, the projections for one pass per schedule, each VAE at its first use.
        model.conditioner = H3Conditioner(root_path, weight_definition, quantize)
        model.lora_paths, model.lora_scales = MiniMaxH3Initializer._resolve_adapters(
            model_config, lora_paths, lora_scales
        )
        model.lora_application_result = None
        model.resident_modulation = any(
            MiniMaxH3Initializer.adapter_targets_modulation(path) for path in model.lora_paths
        )
        model.modulation_tables = None
        model.stream_blocks = stream_blocks and MiniMaxH3Initializer._can_stream_blocks(model)
        MiniMaxH3Initializer._preflight_memory(
            root_path, weight_definition, quantize, model.resident_modulation, model.stream_blocks
        )
        MiniMaxH3Initializer.load_transformer(model)
        MiniMaxH3Initializer._init_decoders(model)

    @staticmethod
    def load_transformer(model) -> None:
        root_path, definition, quantize = model.root_path, model.weight_definition, model.quantize_arg
        component = MiniMaxH3Initializer._component(definition, MiniMaxH3Initializer.TRANSFORMER)
        transformer = MiniMaxH3Transformer()
        keep = None
        if not model.resident_modulation:
            for block in transformer.transformer_blocks:
                block.adaln_proj = None
            keep = MiniMaxH3Initializer._is_resident_transformer_key
            table_bits = None
            if quantize is not None:
                table_bits = MiniMaxH3WeightDefinition.component_bits(MiniMaxH3Initializer.TRANSFORMER, quantize)
            # A transformer read again after a release keeps the tables its earlier clips computed.
            if model.modulation_tables is None:
                model.modulation_tables = H3ModulationTables(root_path / component.hf_subdir, component, table_bits)
        model.bits = StreamingWeightLoader.load_into(
            model=transformer,
            component_root=root_path,
            component=component,
            quantize_arg=quantize,
            quantization_predicate=definition.quantization_predicate,
            keep=keep,
            evaluate=not model.stream_blocks,
        )
        model.block_stream = None
        if model.stream_blocks:
            model.block_stream = H3BlockStream(root_path / component.hf_subdir, component, keep)
        conditioner_bits = model.conditioner.stored_bits()
        if conditioner_bits is not None and conditioner_bits != model.bits:
            raise ValueError(
                f"MiniMax-H3 component quantization mismatch: the transformer resolved to {model.bits}, "
                f"but the conditioner is stored at {conditioner_bits}."
            )
        model.transformer = transformer
        model.transformer_geometry = (transformer.patch_size, transformer.in_channels)
        gc.collect()
        mx.clear_cache()
        MiniMaxH3Initializer._apply_lora(model)

    @staticmethod
    def load_component(root_path: Path, definition, name: str, quantize: int | None) -> nn.Module:
        # One whole component held in memory, AdaLN projections included: what a package is written from.
        component = MiniMaxH3Initializer._component(definition, name)
        if name == H3Conditioner.COMPONENT_NAME:
            module = H3Conditioner.build()
            component.weight_files = H3Conditioner.needed_shards(root_path / component.hf_subdir)
        elif name == MiniMaxH3Initializer.TRANSFORMER:
            module = MiniMaxH3Transformer()
        else:
            module = MiniMaxH3Initializer._build_decoder(root_path, name)
        StreamingWeightLoader.load_into(
            model=module,
            component_root=root_path,
            component=component,
            quantize_arg=quantize,
            quantization_predicate=definition.quantization_predicate,
            post_transform=MiniMaxH3Initializer._post_transform(name),
        )
        return module

    @staticmethod
    def _can_stream_blocks(model) -> bool:
        # Possible when the file holds each block as the model computes with it, and no adapter needs the
        # AdaLN projections in memory next to the blocks. A request that cannot be met is reported, not dropped.
        reason = None
        if model.resident_modulation:
            reason = "an adapter trains the AdaLN projections, which keeps the transformer in memory"
        elif not H3BlockStream.streams_as_stored(
            model.root_path / MiniMaxH3Initializer.TRANSFORMER, model.quantize_arg
        ):
            reason = (
                "a source snapshot quantized at load would be quantized again at every step; "
                "stream from a prepared package, or load the snapshot without --quantize"
            )
        if reason is None:
            return True
        print(f"⚠️  MiniMax-H3: the transformer blocks are not streamed from disk: {reason}.", file=sys.stderr)
        return False

    @staticmethod
    def _is_resident_transformer_key(key: str) -> bool:
        return not H3ModulationTables.holds(key)

    @staticmethod
    def _is_outside_the_blocks(key: str) -> bool:
        return not key.startswith(H3BlockStream.PREFIX)

    @staticmethod
    def _component(definition, name: str):
        return next(component for component in definition.get_components() if component.name == name)

    @staticmethod
    def _post_transform(name: str):
        return MiniMaxH3WeightMapping.convert_audio_vae_state if name == "audio_vae" else None

    @staticmethod
    def _require_marker_tokens(tokenizers: dict, root_path: Path) -> None:
        # A tokenizer that splits these markers still runs, and every dialogue prompt is then conditioned on
        # tokens the model never saw in that role. A re-serialized tokenizer can lose them: they are declared
        # in tokenizer_config.json only.
        tokenizer = tokenizers[MiniMaxH3WeightDefinition.TOKENIZER_NAME].tokenizer
        split = [
            marker
            for marker in MiniMaxH3Initializer.MARKER_TOKENS
            if len(tokenizer(marker, add_special_tokens=False)["input_ids"]) != 1
        ]
        if split:
            raise ValueError(
                f"The tokenizer in {str(root_path)!r} splits {', '.join(split)} into several tokens, so dialogue "
                "prompts would be conditioned incorrectly. Prepare the package again with "
                "`mlxgen prepare --model minimax-h3 --quantize <bits> --path <folder>`."
            )

    @staticmethod
    def _preflight_memory(
        root_path: Path,
        weight_definition,
        quantize: int | None,
        resident_modulation: bool = False,
        stream_blocks: bool = False,
    ) -> None:
        # Unquantized, the released weights do not fit a 128 GiB machine, and without this the OS kills the
        # process partway through the load with nothing said. Reports and stops; never quantizes on the caller's
        # behalf (ADR 0002).
        physical_bytes = RuntimeMemory.total_physical_memory_bytes()
        if physical_bytes <= 0:
            # Unknown physical memory: a preflight that guesses would refuse runs that fit.
            return
        weight_bytes = MiniMaxH3Initializer._estimate_resident_weight_bytes(
            root_path, weight_definition, quantize, resident_modulation, stream_blocks
        )
        if weight_bytes <= 0 or weight_bytes <= physical_bytes * MiniMaxH3Initializer.WEIGHT_BUDGET_SHARE:
            return
        gib = 1024**3
        widths = " or ".join(f"--quantize {bits}" for bits in sorted(MiniMaxH3WeightDefinition.QUANTIZATION_BITS))
        raise MemoryError(
            f"MiniMax-H3 needs about {weight_bytes / gib:.0f} GiB of resident weights from {str(root_path)!r}, and "
            f"this machine has {physical_bytes / gib:.0f} GiB. The weights alone would leave no room for the run. "
            f"Pass --stream-blocks to read the transformer from disk at every step instead of holding it; "
            f"or, from the source snapshot, pass {widths}; or load a smaller prepared package."
        )

    @staticmethod
    def _estimate_resident_weight_bytes(
        root_path: Path,
        weight_definition,
        quantize: int | None,
        resident_modulation: bool = False,
        stream_blocks: bool = False,
    ) -> int:
        total = 0
        for component in weight_definition.get_components():
            if component.name == H3Conditioner.COMPONENT_NAME:
                # Read for the length of an encode, one layer at a time; never part of what stays resident.
                continue
            component_root = root_path / component.hf_subdir
            if not component_root.exists():
                return 0
            names = component.weight_files
            paths = [component_root / name for name in names] if names else list(component_root.glob("*.safetensors"))
            paths = [path for path in paths if path.is_file()]
            keep = None
            if component.name == MiniMaxH3Initializer.TRANSFORMER and stream_blocks:
                keep = MiniMaxH3Initializer._is_outside_the_blocks
            elif component.name == MiniMaxH3Initializer.TRANSFORMER and not resident_modulation:
                keep = MiniMaxH3Initializer._is_resident_transformer_key
            component_bytes = StreamingWeightLoader.tensor_bytes(paths, keep)
            # A prepared package is already at its stored widths; only a bf16 source shrinks under --quantize.
            is_source = not StreamingWeightLoader.is_prepared(component_root)
            if quantize is not None and is_source and not component.skip_quantization:
                bits = MiniMaxH3WeightDefinition.component_bits(component.name, quantize)
                component_bytes = int(component_bytes * (bits + MiniMaxH3Initializer.QUANTIZATION_OVERHEAD_BITS) / 16)
            total += component_bytes
        return total

    @staticmethod
    def _init_decoders(model) -> None:
        # Bound to their files and read at first use: the video decoder at the first decode, its encoder only
        # for a keyframe. `deferred_weight_bytes` is what that first use will add to the footprint.
        model.deferred_weight_bytes = 0
        model.decoder_signatures = {}
        for name in MiniMaxH3Initializer.DECODERS:
            component = MiniMaxH3Initializer._component(model.weight_definition, name)
            files, _ = StreamingWeightLoader.component_files(model.root_path / component.hf_subdir, component)
            model.decoder_signatures.update(StreamingWeightLoader.file_signatures(files))
            module = MiniMaxH3Initializer._build_decoder(model.root_path, name)
            StreamingWeightLoader.load_into(
                model=module,
                component_root=model.root_path,
                component=component,
                quantize_arg=model.quantize_arg,
                quantization_predicate=model.weight_definition.quantization_predicate,
                post_transform=MiniMaxH3Initializer._post_transform(name),
                evaluate=False,
            )
            setattr(model, name, module)
            model.deferred_weight_bytes += sum(value.nbytes for _, value in tree_flatten(module.parameters()))

    @staticmethod
    def _build_decoder(root_path: Path, name: str):
        # Every constructor defaults to the released configuration; the component config.json files refine
        # them when present.
        config = MiniMaxH3Initializer._read_json(root_path / name / "config.json")
        if name == "vae":
            return H3VideoVAE(**video_vae_kwargs(config)) if config else H3VideoVAE()
        return H3AudioVAE(**audio_vae_kwargs(config)) if config else H3AudioVAE()

    @staticmethod
    def _resolve_adapters(
        model_config: ModelConfig, lora_paths: list[str] | None, lora_scales: list[float] | None
    ) -> tuple[list[str], list[float]]:
        turbo_lora = model_config.transformer_overrides.get("turbo_lora")
        if not lora_paths and turbo_lora:
            lora_paths = [turbo_lora]
        if not lora_paths:
            return [], []
        resolved_paths = LoraResolution.resolve_paths(lora_paths)
        return list(resolved_paths), list(LoraResolution.resolve_scales(lora_scales, len(resolved_paths)))

    @staticmethod
    def adapter_targets_modulation(path: str) -> bool:
        # An adapter that trains the per-block AdaLN projections needs them in memory to be applied to.
        # `final_layer.adaln_proj` is the output modulation, which is always resident.
        return any("adaln_proj" in key and "final_layer" not in key for key in mx.load(path))

    @staticmethod
    def _apply_lora(model) -> None:
        resolved_paths, user_scales = model.lora_paths, model.lora_scales
        if not resolved_paths:
            model.lora_application_result = None
            return
        effective_scales = [
            scale * MiniMaxH3Initializer.adapter_scale(path) for path, scale in zip(resolved_paths, user_scales)
        ]
        result = LoRALoader.load_and_apply_lora_detailed(
            lora_mapping=MiniMaxH3LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=resolved_paths,
            lora_scales=effective_scales,
            state_dict_transform=MiniMaxH3LoRAMapping.normalize_state_dict,
        )
        model.lora_application_result = result

    @staticmethod
    def adapter_scale(path: str) -> float:
        """The `alpha / rank` factor an adapter file was trained with, beyond the user's `--lora-scales`.

        PEFT files carry `alpha` in the safetensors metadata (the lightx2v Turbo adapters: alpha 8 at
        rank 128). A PEFT file without it runs at alpha == rank, which is how ai-toolkit exports (it
        drops alpha on save) and how ComfyUI and diffusers then load them; kohya-style files carry
        per-module `.alpha` tensors, which the loader folds into each target itself.
        """
        weights, metadata = mx.load(path, return_metadata=True)
        MiniMaxH3LoRAMapping.reject_unsupported_layout(weights.keys())
        if any(key.endswith(".alpha") for key in weights):
            return 1.0
        alpha = (metadata or {}).get("alpha")
        rank = next((int(v.shape[0]) for k, v in weights.items() if _LORA_DOWN_KEY.search(k)), None)
        if alpha is None or rank is None:
            return 1.0
        return float(alpha) / rank

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.exists():
            return None
        return json.loads(path.read_text())
