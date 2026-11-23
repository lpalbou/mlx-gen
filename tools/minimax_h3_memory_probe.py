import argparse
import ctypes
import gc
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import PIL.Image
from minimax_h3_quant_probe import TRANSFORMER_INPUT_KEYS, H3QuantProbe, ResidentConditioner
from mlx import nn
from mlx.utils import tree_unflatten

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.streaming_weight_loader import StreamingWeightLoader
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.minimax_h3.minimax_h3_initializer import MiniMaxH3Initializer
from mflux.models.minimax_h3.model.h3_audio_vae.h3_audio_vae import H3AudioVAE
from mflux.models.minimax_h3.model.h3_precision import disable_tf32, linear_input_dtype
from mflux.models.minimax_h3.model.h3_transformer.h3_transformer import MiniMaxH3Transformer
from mflux.models.minimax_h3.model.h3_video_vae.h3_video_vae import H3VideoVAE
from mflux.models.minimax_h3.variants.minimax_h3 import MiniMaxH3, compose_prompt
from mflux.models.minimax_h3.weights.h3_audio_vae_weights import audio_vae_kwargs
from mflux.models.minimax_h3.weights.h3_video_vae_weights import video_vae_kwargs
from mflux.models.minimax_h3.weights.h3_weight_definition import MiniMaxH3WeightDefinition
from mflux.models.minimax_h3.weights.h3_weight_mapping import MiniMaxH3WeightMapping

GIB = 1024**3
ADALN_FRAGMENT = ".adaln_proj.linear."


class FootprintSampler:
    # Polls the Darwin physical footprint of this process, so a phase has its own peak.
    class _RUsage(ctypes.Structure):
        _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(f"field_{index}", ctypes.c_uint64) for index in range(36)]

    FOOTPRINT_FIELD = "field_7"

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.peak_bytes = 0
        self._libproc = ctypes.CDLL("libproc.dylib")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def current(self) -> int:
        info = FootprintSampler._RUsage()
        if self._libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
            return 0
        return int(getattr(info, FootprintSampler.FOOTPRINT_FIELD))

    def lifetime_peak(self) -> int:
        info = FootprintSampler._RUsage()
        self._libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info))
        return int(getattr(info, "field_28"))

    def take_peak(self) -> int:
        peak = max(self.peak_bytes, self.current())
        self.peak_bytes = 0
        return peak

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_bytes = max(self.peak_bytes, self.current())


class PhaseLog:
    def __init__(self):
        self.sampler = FootprintSampler()
        self.phases: dict[str, dict] = {}
        self._started = time.perf_counter()
        mx.reset_peak_memory()
        self.sampler.take_peak()

    def close(self, name: str) -> dict:
        entry = {
            "seconds": round(time.perf_counter() - self._started, 1),
            "mlx_peak_gib": round(mx.get_peak_memory() / GIB, 2),
            "mlx_active_gib": round(mx.get_active_memory() / GIB, 2),
            "footprint_peak_gib": round(self.sampler.take_peak() / GIB, 2),
            "footprint_end_gib": round(self.sampler.current() / GIB, 2),
        }
        self.phases[name] = entry
        print(f"phase {name}: {json.dumps(entry)}", flush=True)
        self._started = time.perf_counter()
        mx.reset_peak_memory()
        return entry


class ScheduleComplete(Exception):
    pass


class ScheduleRecorder:
    # Stands in for the transformer during a dry run and keeps the distinct timesteps of each step.
    def __init__(self, transformer):
        self.timesteps: list[mx.array] = []
        self.patch_size = transformer.patch_size
        self.in_channels = transformer.in_channels
        self.audio_in_channels = transformer.audio_in_channels

    def __call__(self, **kwargs):
        self.timesteps.append(mx.array(kwargs["timestep"]))
        return mx.zeros_like(kwargs["hidden_states"]), mx.zeros_like(kwargs["audio_hidden_states"])


class StepClock:
    def __init__(self):
        self.step = 0


class TableAdaLN(nn.Module):
    # One block's modulation tables for every step of a schedule, in place of its projection weights.
    def __init__(self, tables: list[tuple[mx.array, ...]], tembs: list[mx.array] | None, clock: StepClock):
        super().__init__()
        self.tables = tables
        self.tembs = tembs
        self.clock = clock

    def __call__(self, temb: mx.array) -> tuple[mx.array, ...]:
        step = self.clock.step
        if self.tembs is not None and not mx.array_equal(temb, self.tembs[step]).item():
            raise ValueError(f"The AdaLN tables of step {step} were computed for another timestep embedding.")
        return self.tables[step]


class AdaLNTables:
    @staticmethod
    def weight_index(transformer_root: Path) -> dict[str, Path]:
        index = {}
        for file in sorted(transformer_root.glob("*.safetensors")):
            for key in mx.load(str(file)):
                if ADALN_FRAGMENT in f"{key}":
                    index[key] = file
        return index

    @staticmethod
    def schedule(model: MiniMaxH3, clip: dict) -> list[mx.array]:
        transformer = model.transformer
        recorder = ScheduleRecorder(transformer)

        def stop_at_decode(event) -> None:
            if event.phase == "decode":
                raise ScheduleComplete

        model.transformer = recorder
        try:
            model.generate_video(**clip, progress_callback=stop_at_decode)
        except ScheduleComplete:
            pass
        finally:
            model.transformer = transformer
        return recorder.timesteps

    @staticmethod
    def timestep_embeddings(transformer: MiniMaxH3Transformer, timesteps: list[mx.array]) -> list[mx.array]:
        embedder = transformer.time_embedder
        tembs = [embedder(transformer.time_proj(t).astype(linear_input_dtype(embedder.linear_1))) for t in timesteps]
        mx.eval(*tembs)
        return tembs

    @staticmethod
    def compute(modules: list, index: dict[str, Path], tembs: list[mx.array]) -> list[list[tuple[mx.array, ...]]]:
        # Each block's projection is read from disk, applied to every step's embedding and released.
        result = []
        for block_index, module in enumerate(modules):
            base = getattr(module.linear, "linear", module.linear)
            prefix = f"transformer_blocks.{block_index}.adaln_proj.linear"
            weight = mx.load(str(index[f"{prefix}.weight"]))[f"{prefix}.weight"].astype(ModelConfig.precision)
            bias = mx.load(str(index[f"{prefix}.bias"]))[f"{prefix}.bias"].astype(ModelConfig.precision)
            placeholder = {key: getattr(base, key) for key in ("weight", "bias", "scales", "biases") if key in base}
            if isinstance(base, nn.QuantizedLinear):
                base.weight, base.scales, base.biases = mx.quantize(weight, group_size=base.group_size, bits=base.bits)
            else:
                base.weight = weight
            base.bias = bias
            tables = [module(temb) for temb in tembs]
            mx.eval(*[table for step in tables for table in step])
            for key, value in placeholder.items():
                setattr(base, key, value)
            del weight, bias
            result.append(tables)
            if block_index % 5 == 4:
                gc.collect()
                mx.clear_cache()
        gc.collect()
        mx.clear_cache()
        return result


class StreamedBlock:
    # Reads one transformer block's weights for the length of its call.
    def __init__(self, streamer, index: int, block):
        self.streamer = streamer
        self.index = index
        self.block = block

    def __call__(self, packed, temb, adaln_indices, rotary_emb, modulation=None):
        bound = self.streamer.bind(self.index, self.block)
        try:
            packed = self.block(packed, temb, adaln_indices, rotary_emb)
            mx.eval(packed)
        finally:
            for module, leaf, placeholder in bound:
                setattr(module, leaf, placeholder)
        return packed


class BlockStreamer:
    BLOCK_PREFIX = "transformer_blocks."

    def __init__(self, transformer_root: Path, component, skip_adaln: bool, convert: bool):
        self.component = component
        self.convert = convert
        self.skip_adaln = skip_adaln
        self.leaf_counts: dict[int, int] = {}
        self.files: dict[int, dict[Path, list[str]]] = {}
        for file in sorted(transformer_root.glob("*.safetensors")):
            for key in mx.load(str(file)):
                if not key.startswith(BlockStreamer.BLOCK_PREFIX) or (skip_adaln and ADALN_FRAGMENT in key):
                    continue
                block_index = int(key.split(".")[1])
                self.files.setdefault(block_index, {}).setdefault(file, []).append(key)

    def install(self, transformer) -> None:
        transformer.transformer_blocks = [
            StreamedBlock(self, index, block) for index, block in enumerate(transformer.transformer_blocks)
        ]

    def bind(self, index: int, block) -> list:
        bound = []
        for file, keys in self.files[index].items():
            stored = mx.load(str(file))
            raw = {key: stored[key] for key in keys}
            if self.convert:
                raw = WeightLoader._convert_precision(
                    raw, self.component.precision, precision_override=self.component.precision_override
                )
            for key, tensor in raw.items():
                module, leaf = BlockStreamer.target(block, key.split(".")[2:])
                bound.append((module, leaf, getattr(module, leaf)))
                setattr(module, leaf, tensor)
        expected = self.leaf_counts.get(index)
        if expected is None:
            expected = self.leaf_counts[index] = BlockStreamer.count_base_leaves(block, self.skip_adaln)
        if len(bound) != expected:
            raise ValueError(f"Block {index}: bound {len(bound)} tensors, the block has {expected}.")
        return bound

    @staticmethod
    def count_base_leaves(block, skip_adaln: bool) -> int:
        # Every array the block owns except adapter factors and, with tables, the AdaLN projection.
        from mlx.utils import tree_flatten

        return sum(
            1
            for key, _ in tree_flatten(block.parameters())
            if "lora_" not in key and not (skip_adaln and "adaln_proj" in key)
        )

    @staticmethod
    def target(block, parts: list[str]):
        module = block
        for part in parts[:-1]:
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        # The same unwrapping as `linear_input_dtype`: a single adapter wraps `linear`, fused ones `base_linear`.
        module = getattr(module, "linear", getattr(module, "base_linear", module))
        return module, parts[-1]


class ReleasingLayer:
    # Runs one conditioner layer, then drops it: its weights are read from disk for this call only.
    def __init__(self, layer):
        self.layer = layer

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = self.layer(x, cos, sin)
        mx.eval(x)
        self.layer = None
        gc.collect()
        mx.clear_cache()
        return x


class H3MemoryProbe:
    @staticmethod
    def encode_streamed(args) -> None:
        disable_tf32()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        root = H3QuantProbe._source_root(args.source)
        log = PhaseLog()
        tokenizers = TokenizerLoader.load_all(
            definitions=MiniMaxH3WeightDefinition.get_tokenizers(), model_path=str(root)
        )
        for label, prompt, keyframes in H3MemoryProbe._presentations(args):
            # The loader's evaluations are skipped, so every weight stays a pending file read.
            evaluate = mx.eval
            mx.eval = lambda *arrays, **kwargs: None
            try:
                encoder = H3QuantProbe._load_text_encoder(root, args.variant)
            finally:
                mx.eval = evaluate
            encoder.language_model.layers = [ReleasingLayer(layer) for layer in encoder.language_model.layers]
            holder = SimpleNamespace(
                prompt_embed_cache={}, conditioner=ResidentConditioner(encoder), tokenizers=tokenizers
            )
            embeds, tags = MiniMaxH3._encode_presentation(holder, compose_prompt(prompt), keyframes)
            mx.save_safetensors(
                str(output_dir / f"embeds_{args.tag}_{label}.safetensors"), {"embeds": embeds, "tags": mx.array(tags)}
            )
            del encoder, holder
            gc.collect()
            mx.clear_cache()
            log.close(f"encode_{label}")
        H3MemoryProbe._write_report(output_dir, f"encode_{args.tag}", dict(log.phases))

    @staticmethod
    def encode(args) -> None:
        disable_tf32()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        root = H3QuantProbe._source_root(args.source)
        log = PhaseLog()
        tokenizers = TokenizerLoader.load_all(
            definitions=MiniMaxH3WeightDefinition.get_tokenizers(), model_path=str(root)
        )
        encoder = H3QuantProbe._load_text_encoder(root, args.variant)
        weights_gib = round(H3QuantProbe._parameter_bytes(encoder) / GIB, 2)
        log.close("conditioner_load")
        holder = SimpleNamespace(prompt_embed_cache={}, conditioner=ResidentConditioner(encoder), tokenizers=tokenizers)
        for label, prompt, keyframes in H3MemoryProbe._presentations(args):
            embeds, tags = MiniMaxH3._encode_presentation(holder, compose_prompt(prompt), keyframes)
            mx.save_safetensors(
                str(output_dir / f"embeds_{args.tag}_{label}.safetensors"), {"embeds": embeds, "tags": mx.array(tags)}
            )
            log.close(f"encode_{label}")
        H3MemoryProbe._write_report(output_dir, f"encode_{args.tag}", {"weights_gib": weights_gib, **log.phases})

    @staticmethod
    def clips(args) -> None:
        disable_tf32()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log = PhaseLog()
        model, adaln_modules, adaln_index = H3MemoryProbe._load_denoiser(args)
        report = {}
        if not args.stream_blocks:
            report["transformer_weights_gib"] = H3MemoryProbe._resident_gib(model.transformer, args.tables)
        log.close("denoiser_load")
        clock = StepClock()
        presentations = H3MemoryProbe._presentations(args)
        for label, prompt, keyframes in presentations:
            clip = {
                "seed": args.seed,
                "prompt": prompt,
                "width": args.width,
                "height": args.height,
                "num_frames": args.frames,
                "image_path": args.image_path if label == args.image_label else None,
            }
            for embeds_tag in args.embeds:
                H3MemoryProbe._inject(model, output_dir, embeds_tag, label, compose_prompt(prompt), keyframes)
                name = f"{label}_{args.tag}_{embeds_tag}"
                if args.tables:
                    H3MemoryProbe._install_tables(model, adaln_modules, adaln_index, clip, clock, args.check_tables)
                    log.close(f"tables_{name}")
                if not args.skip_clips:
                    H3MemoryProbe._generate(model, clip, clock, log, name, output_dir, args, label, adaln_modules)
                errors = H3MemoryProbe._teacher_forced(model, clock, output_dir, label, embeds_tag, args)
                if errors:
                    report[f"error_{name}"] = errors
                    print(f"error {name}: {json.dumps(errors)}", flush=True)
                model.prompt_embed_cache.clear()
        report.update(log.phases)
        report["lifetime_peak_footprint_gib"] = round(log.sampler.lifetime_peak() / GIB, 2)
        print(f"lifetime peak footprint: {report['lifetime_peak_footprint_gib']} GiB", flush=True)
        H3MemoryProbe._write_report(output_dir, f"clips_{args.tag}", report)

    @staticmethod
    def bench(args) -> None:
        # One transformer call per candidate on identical inputs, alternated so machine drift cancels.
        disable_tf32()
        output_dir = Path(args.output_dir)
        label, prompt, keyframes = H3MemoryProbe._presentations(args)[0]
        candidates = {}
        for name, overrides in (
            ("bf16", {"source": "MiniMaxAI/MiniMax-H3", "variant": "bf16", "stream_blocks": False}),
            ("q8", {"source": args.package, "variant": "stored", "stream_blocks": False}),
            ("bf16_streamed", {"source": "MiniMaxAI/MiniMax-H3", "variant": "bf16", "stream_blocks": True}),
            ("q8_streamed", {"source": args.package, "variant": "stored", "stream_blocks": True}),
        ):
            if name in args.candidates:
                candidates[name] = H3MemoryProbe._load_denoiser(argparse.Namespace(**{**vars(args), **overrides}))
        print(f"loaded {list(candidates)}: MLX active {mx.get_active_memory() / GIB:.1f} GiB", flush=True)
        report = {}
        for width, height in ((640, 352), (960, 544), (1344, 768))[: args.canvases]:
            clip = {"seed": args.seed, "prompt": prompt, "width": width, "height": height, "num_frames": args.frames}
            inputs = None
            clocks = {}
            for name, (model, adaln_modules, adaln_index) in candidates.items():
                H3MemoryProbe._inject(model, output_dir, args.embeds[0], label, compose_prompt(prompt), keyframes)
                clocks[name] = StepClock()
                H3MemoryProbe._install_tables(model, adaln_modules, adaln_index, clip, clocks[name], False)
                inputs = inputs or H3MemoryProbe._first_call_inputs(model, clip)
            rows = int(inputs["position_ids"].shape[-2]) if inputs["position_ids"].ndim > 1 else 0
            timings = {name: [] for name in candidates}
            outputs = {}
            for _ in range(args.rounds):
                for name, (model, _, _) in candidates.items():
                    started = time.perf_counter()
                    video, audio = model.transformer(**inputs)
                    mx.eval(video, audio)
                    timings[name].append(round(time.perf_counter() - started, 2))
                    outputs[name] = video.astype(mx.float32)
                    del video, audio
            entry = {"rows": rows, "seconds": timings}
            for name in candidates:
                if name.endswith("_streamed") and name[: -len("_streamed")] in outputs:
                    resident = outputs[name[: -len("_streamed")]]
                    entry[f"{name}_equals_resident"] = bool(mx.array_equal(outputs[name], resident).item())
            report[f"{width}x{height}"] = entry
            print(f"bench {width}x{height}: {json.dumps(entry)}", flush=True)
        H3MemoryProbe._write_report(output_dir, f"bench_{args.tag}", report)

    @staticmethod
    def _first_call_inputs(model, clip) -> dict:
        transformer = model.transformer
        captured = {}

        def capture(**kwargs):
            captured.update({key: mx.array(kwargs[key]) for key in TRANSFORMER_INPUT_KEYS})
            raise ScheduleComplete

        model.transformer = H3MemoryProbe._callable(capture, transformer)
        try:
            model.generate_video(**clip)
        except ScheduleComplete:
            pass
        finally:
            model.transformer = transformer
        mx.eval(*captured.values())
        return captured

    @staticmethod
    def _generate(model, clip, clock, log, name, output_dir, args, label, adaln_modules) -> None:
        transformer = model.transformer
        capture_steps = set(args.capture_steps) if args.capture else set()
        captures = {}

        def recording(**kwargs):
            video, audio = transformer(**kwargs)
            if clock.step in capture_steps:
                arrays = {key: mx.array(kwargs[key]) for key in TRANSFORMER_INPUT_KEYS}
                arrays["reference_video"] = video.astype(mx.float32)
                arrays["reference_audio"] = audio.astype(mx.float32)
                mx.eval(*arrays.values())
                captures[clock.step] = arrays
            return video, audio

        def on_progress(event) -> None:
            if event.phase == "denoise":
                clock.step = event.step
            elif event.phase == "decode":
                log.close(f"denoise_{name}")
                if args.release_before_decode:
                    model.transformer = SimpleNamespace(
                        patch_size=transformer.patch_size, in_channels=transformer.in_channels
                    )
                    H3MemoryProbe._release(transformer, adaln_modules)
                    log.close(f"release_{name}")
            elif event.phase == "generated":
                log.close(f"decode_{name}")

        clock.step = 0
        if args.capture:
            model.transformer = H3MemoryProbe._callable(recording, transformer)
        try:
            video = model.generate_video(**clip, progress_callback=on_progress)
        finally:
            model.transformer = transformer
        if args.release_before_decode:
            raise SystemExit(H3MemoryProbe._finish(log, output_dir, args, name, video))
        video.save(str(output_dir / f"clip_{name}_seed{args.seed}.mp4"), export_json_metadata=True)
        for step, arrays in captures.items():
            mx.save_safetensors(str(output_dir / f"reference_{label}_step{step}.safetensors"), arrays)
        log.close(f"save_{name}")

    @staticmethod
    def _release(transformer, adaln_modules: list) -> None:
        # Drops every weight the transformer holds; the caller keeps the object for its attributes only.
        for module in adaln_modules:
            module.clear()
        adaln_modules.clear()
        for block in transformer.transformer_blocks:
            getattr(block, "block", block).clear()
        transformer.clear()
        gc.collect()
        mx.clear_cache()

    @staticmethod
    def _finish(log, output_dir, args, name, video) -> int:
        video.save(str(output_dir / f"clip_{name}_seed{args.seed}.mp4"), export_json_metadata=True)
        report = dict(log.phases)
        report["lifetime_peak_footprint_gib"] = round(log.sampler.lifetime_peak() / GIB, 2)
        H3MemoryProbe._write_report(output_dir, f"clips_{args.tag}", report)
        print(f"lifetime peak footprint: {report['lifetime_peak_footprint_gib']} GiB", flush=True)
        return 0

    @staticmethod
    def _callable(function, transformer):
        class Recording:
            patch_size = transformer.patch_size
            in_channels = transformer.in_channels
            audio_in_channels = transformer.audio_in_channels

            def __call__(self, **kwargs):
                return function(**kwargs)

        return Recording()

    @staticmethod
    def _teacher_forced(model, clock, output_dir, label, embeds_tag, args) -> dict:
        # The candidate's velocity on the reference trajectory's own inputs, with the candidate's conditioning.
        errors = {}
        if args.capture:
            return errors
        embeds = mx.load(str(output_dir / f"embeds_{embeds_tag}_{label}.safetensors"))["embeds"]
        for step in args.capture_steps:
            path = output_dir / f"reference_{label}_step{step}.safetensors"
            if not path.exists():
                continue
            captured = mx.load(str(path))
            inputs = {key: captured[key] for key in TRANSFORMER_INPUT_KEYS}
            inputs["encoder_hidden_states"] = embeds
            clock.step = step
            video, audio = model.transformer(**inputs)
            video, audio = video.astype(mx.float32), audio.astype(mx.float32)
            mx.eval(video, audio)
            errors[f"step{step}"] = {
                "video": H3QuantProbe._error(video, captured["reference_video"]),
                "audio": H3QuantProbe._error(audio, captured["reference_audio"]),
            }
            del captured, inputs, video, audio
            gc.collect()
            mx.clear_cache()
        return errors

    @staticmethod
    def _install_tables(model, adaln_modules, adaln_index, clip, clock, check: bool) -> None:
        blocks = [getattr(block, "block", block) for block in model.transformer.transformer_blocks]
        for block, module in zip(blocks, adaln_modules):
            block.adaln_proj = module
        timesteps = AdaLNTables.schedule(model, clip)
        tembs = AdaLNTables.timestep_embeddings(model.transformer, timesteps)
        tables = AdaLNTables.compute(adaln_modules, adaln_index, tembs)
        for block_index, block in enumerate(blocks):
            block.adaln_proj = TableAdaLN(tables[block_index], tembs if block_index == 0 and check else None, clock)

    @staticmethod
    def _load_denoiser(args):
        definition = MiniMaxH3WeightDefinition()
        root = H3QuantProbe._source_root(args.source)
        components = {component.name: component for component in definition.components()}
        model = MiniMaxH3.__new__(MiniMaxH3)
        nn.Module.__init__(model)
        model.model_config = ModelConfig.from_name(args.base_model)
        model.root_path = root
        model.weight_definition = definition
        model.callbacks = CallbackRegistry()
        model.quantize_arg = None
        model.prompt_embed_cache = {}
        model.tokenizers = {}
        # The probe drives its own staging, so the engine's is switched off on this model.
        model.conditioner = None
        model.modulation_tables = None
        model.block_stream = None
        model.resident_modulation = True
        model.deferred_weight_bytes = 0
        model.decoder_signatures = {}
        model.bits = None if args.variant in ("bf16", "stored") else int(args.variant.partition(":")[0][1:])
        vae_config = MiniMaxH3Initializer._read_json(root / "vae" / "config.json")
        model.vae = H3VideoVAE(**video_vae_kwargs(vae_config)) if vae_config else H3VideoVAE()
        audio_config = MiniMaxH3Initializer._read_json(root / "audio_vae" / "config.json")
        model.audio_vae = H3AudioVAE(**audio_vae_kwargs(audio_config)) if audio_config else H3AudioVAE()
        # With `--lazy-vae` the loader's evaluations are skipped: each VAE weight is read when first used.
        evaluate = mx.eval
        if args.lazy_vae:
            mx.eval = lambda *arrays, **kwargs: None
        try:
            StreamingWeightLoader.load_into(model.vae, root, components["vae"], None)
            StreamingWeightLoader.load_into(
                model.audio_vae,
                root,
                components["audio_vae"],
                None,
                post_transform=MiniMaxH3WeightMapping.convert_audio_vae_state,
            )
        finally:
            mx.eval = evaluate
        model.transformer = MiniMaxH3Transformer()
        component = components["transformer"]

        def skip(key: str) -> bool:
            if args.tables and ADALN_FRAGMENT in key:
                return True
            return args.stream_blocks and key.startswith(BlockStreamer.BLOCK_PREFIX)

        if args.variant == "stored":
            linear_adaln = args.tables and args.tables_width == 16
            H3MemoryProbe._load_stored_transformer(model.transformer, root, component, skip, linear_adaln)
        else:
            H3MemoryProbe._stream_transformer(model.transformer, root, component, args.variant, skip, args.reindex)
        model.transformer_geometry = (model.transformer.patch_size, model.transformer.in_channels)
        model.lora_paths, model.lora_scales = MiniMaxH3Initializer._resolve_adapters(model.model_config, None, None)
        MiniMaxH3Initializer._apply_lora(model)
        adaln_modules = [block.adaln_proj for block in model.transformer.transformer_blocks]
        adaln_root = H3QuantProbe._source_root(args.adaln_source) / "transformer"
        adaln_index = AdaLNTables.weight_index(adaln_root) if args.tables else {}
        if args.stream_blocks:
            streamer = BlockStreamer(root / "transformer", component, args.tables, args.variant != "stored")
            streamer.install(model.transformer)
        return model, adaln_modules, adaln_index

    @staticmethod
    def _load_stored_transformer(model, root: Path, component, skip, linear_adaln: bool) -> None:
        # A prepared package at its stored widths; skipped tensors keep their never-evaluated initial values.
        shards = StreamingWeightLoader._prepared_shards(root / component.hf_subdir)
        shapes = StreamingWeightLoader._stored_shapes(shards)
        if linear_adaln:
            shapes = {key: shape for key, shape in shapes.items() if ADALN_FRAGMENT not in key}
        StreamingWeightLoader._quantize_as_stored(model, component, shapes)
        for shard in shards:
            weights = {key: value for key, value in mx.load(str(shard)).items() if not skip(key)}
            model.update(tree_unflatten(list(weights.items())), strict=False)
            mx.eval(*weights.values())
            del weights
            gc.collect()
            mx.clear_cache()

    @staticmethod
    def _stream_transformer(model, root: Path, component, variant: str, skip, reindex: bool = False) -> None:
        # `StreamingWeightLoader.load_into` for a source snapshot, leaving the skipped tensors unread.
        if variant != "bf16":
            predicate = H3QuantProbe._predicate(variant, H3QuantProbe.TRANSFORMER_FAMILIES)
            nn.quantize(model, class_predicate=WeightApplier.quantization_predicate_for_bits(predicate, 8), bits=8)
        quantized = {
            path: module
            for path, module in model.named_modules()
            if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
        }
        component_path = root / component.hf_subdir
        for file in WeightLoader._resolve_weight_files(component_path, component.weight_files, "*.safetensors"):
            raw = {key: value for key, value in mx.load(str(file)).items() if not skip(key)}
            raw = WeightLoader._convert_precision(
                raw, component.precision, precision_override=component.precision_override
            )
            flat = []
            for key, tensor in raw.items():
                parent, _, leaf = key.rpartition(".")
                module = quantized.get(parent)
                if module is not None and leaf == "weight":
                    weight, scales, biases = mx.quantize(tensor, group_size=module.group_size, bits=module.bits)
                    if reindex and module.bits == 8:
                        weight = H3MemoryProbe._reindexed(tensor, weight, scales, biases, module.group_size)
                    flat.extend(((key, weight), (f"{parent}.scales", scales), (f"{parent}.biases", biases)))
                else:
                    flat.append((key, tensor))
            model.update(tree_unflatten(flat), strict=False)
            mx.eval(*[tensor for _, tensor in flat])
            del raw, flat
            gc.collect()
            mx.clear_cache()

    @staticmethod
    def _reindexed(tensor, packed, scales, biases, group_size: int):
        # `mx.quantize` picks indices on the float32 grid and then stores bf16 scales and biases; this picks
        # them again on the grid that is actually stored.
        rows, inputs = tensor.shape
        groups = tensor.astype(mx.float32).reshape(rows, inputs // group_size, group_size)
        step = scales.astype(mx.float32)[..., None]
        offset = biases.astype(mx.float32)[..., None]
        indices = mx.clip(mx.round((groups - offset) / mx.where(step == 0, 1.0, step)), 0, 255)
        return indices.astype(mx.uint8).reshape(-1).view(mx.uint32).reshape(packed.shape)

    @staticmethod
    def _resident_gib(transformer, tables: bool) -> float:
        from mlx.utils import tree_flatten

        total = sum(
            value.nbytes
            for key, value in tree_flatten(transformer.parameters())
            if not (tables and ".adaln_proj." in f".{key}")
        )
        return round(total / GIB, 2)

    @staticmethod
    def _presentations(args) -> list[tuple[str, str, list]]:
        presentations = []
        for entry in args.prompts:
            label, prompt_file = entry.split(":")[:2]
            keyframes = []
            if label == args.image_label and args.image_path:
                keyframe = PIL.Image.open(args.image_path).convert("RGB")
                keyframes = [MiniMaxH3._fit_keyframe(keyframe, args.width, args.height)]
            presentations.append((label, Path(prompt_file).read_text(), keyframes))
        return presentations

    @staticmethod
    def _inject(model, output_dir: Path, embeds_tag: str, label: str, prompt: str, keyframes: list) -> None:
        stored = mx.load(str(output_dir / f"embeds_{embeds_tag}_{label}.safetensors"))
        digest = hashlib.sha1(prompt.encode("utf-8"))
        for frame in keyframes:
            digest.update(str(frame.size).encode())
            digest.update(frame.tobytes())
        model.prompt_embed_cache[digest.hexdigest()] = (stored["embeds"], np.array(stored["tags"], dtype=np.int32))

    @staticmethod
    def _write_report(output_dir: Path, key: str, value: dict) -> None:
        path = output_dir / "memory_report.json"
        report = json.loads(path.read_text()) if path.exists() else {}
        report[key] = value
        path.write_text(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax-H3 staged loading and AdaLN table measurements.")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_presentation_arguments(command) -> None:
        command.add_argument("--source", default="MiniMaxAI/MiniMax-H3")
        command.add_argument("--variant", required=True)
        command.add_argument("--tag", required=True)
        command.add_argument("--prompts", nargs="+", required=True, help="label:prompt_file")
        command.add_argument("--image-path")
        command.add_argument("--image-label", default="image")
        command.add_argument("--width", type=int, default=640)
        command.add_argument("--height", type=int, default=352)
        command.add_argument("--output-dir", required=True)

    encode = commands.add_parser("encode")
    add_presentation_arguments(encode)
    encode.set_defaults(run=H3MemoryProbe.encode)

    encode_streamed = commands.add_parser("encode-streamed")
    add_presentation_arguments(encode_streamed)
    encode_streamed.set_defaults(run=H3MemoryProbe.encode_streamed)

    clips = commands.add_parser("clips")
    add_presentation_arguments(clips)
    clips.add_argument("--base-model", default="minimax-h3-turbo-544p")
    clips.add_argument("--embeds", nargs="+", required=True, help="tags of `encode` runs to condition on")
    clips.add_argument("--tables", action="store_true", help="AdaLN projections as precomputed tables")
    clips.add_argument("--check-tables", action="store_true")
    clips.add_argument("--tables-width", type=int, default=16, help="width of the projections behind the tables")
    clips.add_argument("--adaln-source", default="MiniMaxAI/MiniMax-H3")
    clips.add_argument("--stream-blocks", action="store_true", help="read each block from disk for its call")
    clips.add_argument("--lazy-vae", action="store_true", help="read VAE weights when first used")
    clips.add_argument("--reindex", action="store_true", help="8-bit indices chosen on the stored bf16 grid")
    clips.add_argument("--release-before-decode", action="store_true", help="one clip; drops the transformer")
    clips.add_argument("--capture", action="store_true", help="store this run's transformer inputs as reference")
    clips.add_argument("--capture-steps", nargs="+", type=int, default=[0, 3, 6])
    clips.add_argument("--skip-clips", action="store_true", help="teacher-forced errors only")
    clips.add_argument("--frames", type=int, default=124)
    clips.add_argument("--seed", type=int, default=42)
    clips.set_defaults(run=H3MemoryProbe.clips)

    bench = commands.add_parser("bench")
    add_presentation_arguments(bench)
    bench.add_argument("--base-model", default="minimax-h3-turbo-544p")
    bench.add_argument("--package", default="models/minimax-h3-8bit")
    bench.add_argument("--embeds", nargs="+", required=True)
    bench.add_argument("--candidates", nargs="+", default=["bf16", "q8"])
    bench.add_argument("--rounds", type=int, default=5)
    bench.add_argument("--canvases", type=int, default=2)
    bench.add_argument("--frames", type=int, default=124)
    bench.add_argument("--seed", type=int, default=42)
    bench.set_defaults(
        run=H3MemoryProbe.bench,
        tables=True,
        tables_width=16,
        adaln_source="MiniMaxAI/MiniMax-H3",
        lazy_vae=True,
        reindex=False,
    )

    args = parser.parse_args()
    if getattr(args, "stream_blocks", False) and args.variant not in ("bf16", "stored"):
        parser.error("--stream-blocks reads blocks as stored: use --variant bf16 or --variant stored")
    args.run(args)


if __name__ == "__main__":
    main()
