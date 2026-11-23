import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import PIL.Image
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.lora_resolution import LoraResolution
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.streaming_weight_loader import StreamingWeightLoader
from mflux.models.minimax_h3.minimax_h3_initializer import MiniMaxH3Initializer
from mflux.models.minimax_h3.model.h3_precision import disable_tf32
from mflux.models.minimax_h3.model.h3_text_encoder.h3_conditioner import H3Conditioner
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_model import Qwen3VLModel
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_text_model import Qwen3VLTextModel
from mflux.models.minimax_h3.model.h3_transformer.h3_transformer import MiniMaxH3Transformer
from mflux.models.minimax_h3.variants.minimax_h3 import MiniMaxH3, compose_prompt
from mflux.models.minimax_h3.weights.h3_lora_mapping import MiniMaxH3LoRAMapping
from mflux.models.minimax_h3.weights.h3_weight_definition import MiniMaxH3WeightDefinition
from mflux.models.minimax_h3.weights.h3_weight_mapping import TEXT_ENCODER_NUM_LAYERS

TRANSFORMER_INPUT_KEYS = (
    "hidden_states",
    "audio_hidden_states",
    "encoder_hidden_states",
    "timestep",
    "timestep_indices",
    "token_tags",
    "position_ids",
    "video_indices",
    "audio_indices",
    "text_indices",
)


class TransformerRecorder:
    # Stands in for `model.transformer` during one generation and stores the inputs of selected steps.
    def __init__(self, transformer, steps: set[int], output_dir: Path, label: str):
        self.transformer = transformer
        self.steps = steps
        self.output_dir = output_dir
        self.label = label
        self.calls = 0

    def __getattr__(self, name: str):
        # The model reads the transformer's geometry, blocks and timestep embedding through this stand-in.
        return getattr(self.transformer, name)

    def __call__(self, **kwargs):
        video, audio = self.transformer(**kwargs)
        if self.calls in self.steps:
            arrays = {key: mx.array(kwargs[key]) for key in TRANSFORMER_INPUT_KEYS}
            arrays["q8_video"] = video.astype(mx.float32)
            arrays["q8_audio"] = audio.astype(mx.float32)
            mx.eval(*arrays.values())
            mx.save_safetensors(str(self.output_dir / f"capture_{self.label}_step{self.calls}.safetensors"), arrays)
        self.calls += 1
        return video, audio


class ResidentConditioner:
    # A conditioner held in memory, with the surface of the model's streamed one.
    def __init__(self, model: Qwen3VLModel):
        self.model = model
        self.spatial_merge_size = model.visual.spatial_merge_size

    def encode(self, token_ids, image_patches=None, image_grids=None) -> mx.array:
        return self.model.encode(token_ids, image_patches, image_grids)


class H3QuantProbe:
    TRANSFORMER_FAMILIES = {
        "attn_qk": (".attn.to_q", ".attn.to_k"),
        "attn_v": (".attn.to_v",),
        "attn_out": (".attn.to_out.0",),
        "ff_in": (".ff.net.0.proj",),
        "ff_out": (".ff.net.2",),
        "context": ("context_embedder",),
        "adaln": (".adaln_proj.linear",),
        "norm_out": ("norm_out.linear",),
    }
    TEXT_FAMILIES = {
        "embed": ("language_model.embed_tokens",),
        "attn_qk": (".self_attn.q_proj", ".self_attn.k_proj"),
        "attn_vo": (".self_attn.v_proj", ".self_attn.o_proj"),
        "mlp_in": (".mlp.gate_proj", ".mlp.up_proj"),
        "mlp_out": (".mlp.down_proj",),
        "vision": ("visual",),
    }
    # Left as released by the shipped policy, so a bare `q8` or `q4` leaves them alone too.
    UNQUANTIZED_BY_DEFAULT = ("norm_out", "vision")

    @staticmethod
    def capture(args) -> None:
        model = MiniMaxH3(
            model_config=ModelConfig.from_name(args.package, base_model=args.base_model),
            model_path=args.package,
        )
        H3QuantProbe._run_clips(model, args, record=True)

    @staticmethod
    def generate(args) -> None:
        model = MiniMaxH3(
            model_config=ModelConfig.from_name(args.package, base_model=args.base_model),
            model_path=args.package,
        )
        H3QuantProbe._run_clips(model, args)

    @staticmethod
    def e2e(args) -> None:
        # Clips from a candidate policy, quantized at load time from the source snapshot.
        transformer_predicate = H3QuantProbe._predicate(args.transformer_variant, H3QuantProbe.TRANSFORMER_FAMILIES)
        text_predicate = H3QuantProbe._predicate(args.text_variant, H3QuantProbe.TEXT_FAMILIES)

        def predicate(path: str, module, bits: int | None = None):
            if path.startswith(("language_model", "visual")):
                return text_predicate(path, module)
            return transformer_predicate(path, module)

        MiniMaxH3WeightDefinition.quantization_predicate = staticmethod(predicate)
        model = MiniMaxH3(model_config=ModelConfig.from_name(args.base_model), quantize=8)
        if model.modulation_tables is not None:
            # The AdaLN tables follow the candidate's width for the projections, not the shipped policy's.
            model.modulation_tables.bits = H3QuantProbe.variant_bits(
                args.transformer_variant, H3QuantProbe.TRANSFORMER_FAMILIES, "transformer_blocks.0.adaln_proj.linear"
            )
        H3QuantProbe._run_clips(model, args)

    @staticmethod
    def _run_clips(model: MiniMaxH3, args, record: bool = False) -> None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        transformer = model.transformer
        for label, prompt_file, steps in H3QuantProbe._capture_plan(args):
            if record:
                model.transformer = TransformerRecorder(transformer, steps, output_dir, label)
            started = time.perf_counter()
            mx.reset_peak_memory()
            video = model.generate_video(
                seed=args.seed,
                prompt=Path(prompt_file).read_text(),
                width=args.width,
                height=args.height,
                num_frames=args.frames,
                image_path=args.image_path if label == args.image_label else None,
            )
            model.transformer = transformer
            video.save(str(output_dir / f"e2e_{label}_{args.tag}_seed{args.seed}.mp4"), export_json_metadata=True)
            print(
                f"generated {label} [{args.tag}]: {time.perf_counter() - started:.0f}s, "
                f"MLX peak {mx.get_peak_memory() / 1024**3:.1f} GiB",
                flush=True,
            )

    @staticmethod
    def transformer(args) -> None:
        disable_tf32()
        output_dir = Path(args.output_dir)
        captures = sorted(output_dir.glob("capture_*.safetensors"))
        if not captures:
            raise FileNotFoundError(f"No capture_*.safetensors in {output_dir}; run the capture command first.")
        report_path = output_dir / "transformer_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        for variant in args.variants:
            started = time.perf_counter()
            model, weight_bytes = H3QuantProbe._load_transformer(args.source, variant, args.turbo_lora)
            load_seconds = time.perf_counter() - started
            rows = {}
            for capture in captures:
                inputs = mx.load(str(capture))
                video, audio = model(**{key: inputs[key] for key in TRANSFORMER_INPUT_KEYS})
                video, audio = video.astype(mx.float32), audio.astype(mx.float32)
                mx.eval(video, audio)
                reference_path = output_dir / f"reference_{capture.stem}.safetensors"
                if variant == "bf16":
                    mx.save_safetensors(str(reference_path), {"video": video, "audio": audio})
                reference = mx.load(str(reference_path))
                rows[capture.stem] = {
                    "video": H3QuantProbe._error(video, reference["video"]),
                    "audio": H3QuantProbe._error(audio, reference["audio"]),
                    "q8_package_video": H3QuantProbe._error(inputs["q8_video"], reference["video"]),
                    "q8_package_audio": H3QuantProbe._error(inputs["q8_audio"], reference["audio"]),
                }
            report[variant] = {
                "load_seconds": round(load_seconds, 1),
                "weights_gib": round(weight_bytes / 1024**3, 2),
                "rows": rows,
            }
            report_path.write_text(json.dumps(report, indent=2))
            print(variant, json.dumps(report[variant]), flush=True)
            if variant == "bf16" and args.adaln:
                adaln = H3QuantProbe._adaln_table_errors(model, captures)
                (output_dir / "adaln_report.json").write_text(json.dumps(adaln, indent=2))
                print("adaln", json.dumps(adaln), flush=True)
            del model
            gc.collect()
            mx.clear_cache()

    @staticmethod
    def conditioning(args) -> None:
        # Velocity error caused by the conditioner alone: the BF16 transformer replays each captured step
        # with the embeddings every conditioner variant produced for that capture's prompt.
        disable_tf32()
        output_dir = Path(args.output_dir)
        model, _ = H3QuantProbe._load_transformer(args.source, "bf16", args.turbo_lora)
        report = {}
        for capture in sorted(output_dir.glob("capture_*.safetensors")):
            inputs = mx.load(str(capture))
            prompt_label = capture.stem.split("_")[1]
            reference_embeds = mx.load(str(output_dir / f"reference_text_{prompt_label}.safetensors"))["embeds"]
            if reference_embeds.shape != inputs["encoder_hidden_states"].shape:
                # The capture was tokenized differently from the conditioner sweep; its layout does not fit.
                print(f"skipping {capture.stem}: {inputs['encoder_hidden_states'].shape} vs {reference_embeds.shape}")
                continue
            outputs = {}
            for variant in ["bf16", *args.variants]:
                name = f"reference_text_{prompt_label}" if variant == "bf16" else f"text_{variant}_{prompt_label}"
                embeds = mx.load(str(output_dir / f"{name}.safetensors"))["embeds"].astype(mx.bfloat16)
                kwargs = {key: inputs[key] for key in TRANSFORMER_INPUT_KEYS}
                kwargs["encoder_hidden_states"] = embeds
                video, audio = model(**kwargs)
                outputs[variant] = (video.astype(mx.float32), audio.astype(mx.float32))
                mx.eval(*outputs[variant])
            for variant in args.variants:
                report.setdefault(variant, {})[capture.stem] = {
                    "video": H3QuantProbe._error(outputs[variant][0], outputs["bf16"][0]),
                    "audio": H3QuantProbe._error(outputs[variant][1], outputs["bf16"][1]),
                }
        (output_dir / "conditioning_report.json").write_text(json.dumps(report, indent=2))
        for variant, rows in report.items():
            print(variant, json.dumps(rows), flush=True)

    @staticmethod
    def ballast(args) -> None:
        # Locks `--gib` of RAM until killed, so a run beside it sees a machine with that much less memory:
        # 64 GiB locked on a 128 GiB Mac leaves what a 64 GiB Mac has. Locked pages cannot be swapped or
        # compressed, which an ordinary allocation could be.
        import ctypes
        import mmap

        size = int(args.gib * 1024**3)
        region = mmap.mmap(-1, size)
        address = ctypes.addressof(ctypes.c_char.from_buffer(region))
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        if libc.mlock(address, size) != 0:
            raise OSError(ctypes.get_errno(), f"mlock of {args.gib} GiB failed")
        print(f"ballast ready: {args.gib} GiB locked", flush=True)
        while True:
            time.sleep(3600)

    @staticmethod
    def report(args) -> None:
        # The tables of docs/quantization.md, from the JSON reports, so no number is typed by hand.
        output_dir = Path(args.output_dir)
        transformer = json.loads((output_dir / "transformer_report.json").read_text())
        captures = sorted(next(iter(transformer.values()))["rows"])
        header = " | ".join(capture.removeprefix("capture_").replace("_step", ", step ") for capture in captures)
        print(f"| Transformer widths | Size | {header} |")
        print("| --- | ---: | " + " | ".join("---" for _ in captures) + " |")
        for variant, entry in transformer.items():
            cells = " | ".join(
                f"{entry['rows'][c]['video']['relative_rms'] * 100:.1f} / {entry['rows'][c]['audio']['relative_rms'] * 100:.1f}"
                for c in captures
            )
            print(f"| `{variant}` | {entry['weights_gib']:.1f} GiB | {cells} |")
        for name, value in (("text_encoder_report.json", "token_relative_median"), ("conditioning_report.json", None)):
            path = output_dir / name
            if not path.exists():
                continue
            table = json.loads(path.read_text())
            labels = sorted(next(iter(table.values())).get("rows", next(iter(table.values()))))
            print(f"\n| {name} | " + " | ".join(labels) + " |")
            print("| --- | " + " | ".join("---" for _ in labels) + " |")
            for variant, entry in table.items():
                rows = entry.get("rows", entry)
                if value:
                    cells = [
                        f"{rows[label][value] * 100:.1f} (all tokens {rows[label]['relative_rms'] * 100:.1f})"
                        for label in labels
                    ]
                else:
                    cells = [
                        f"{rows[label]['video']['relative_rms'] * 100:.1f} / {rows[label]['audio']['relative_rms'] * 100:.1f}"
                        for label in labels
                    ]
                size = f" ({entry['weights_gib']:.1f} GiB)" if "weights_gib" in entry else ""
                print(f"| `{variant}`{size} | " + " | ".join(cells) + " |")
        adaln_path = output_dir / "adaln_report.json"
        if adaln_path.exists():
            groups: dict[str, list[float]] = {}
            for key, tables in json.loads(adaln_path.read_text()).items():
                groups.setdefault(key.rsplit("/", 1)[1], []).extend(tables.values())
            print("\n| AdaLN projection | Modulation table error, min to max |")
            print("| --- | --- |")
            for name, values in groups.items():
                print(f"| `{name}` | {min(values) * 100:.3f}% to {max(values) * 100:.3f}% |")

    @staticmethod
    def compare(args) -> None:
        # Median frame PSNR of each tagged clip against the reference tag, same seed and prompt.
        import av

        output_dir = Path(args.output_dir)

        def frames(label: str, tag: str) -> np.ndarray:
            with av.open(str(output_dir / f"e2e_{label}_{tag}_seed{args.seed}.mp4")) as container:
                return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])

        report = {}
        for label in args.labels:
            reference = frames(label, args.reference_tag).astype(np.float32)
            for tag in args.tags:
                mse = ((frames(label, tag).astype(np.float32) - reference) ** 2).mean(axis=(1, 2, 3))
                psnr = 10 * np.log10(255.0**2 / np.maximum(mse, 1e-9))
                report.setdefault(tag, {})[label] = round(float(np.median(psnr)), 1)
        (output_dir / "fidelity_report.json").write_text(json.dumps(report, indent=2))
        print("| Clip | " + " | ".join(args.labels) + f" |   (median frame PSNR against `{args.reference_tag}`, dB)")
        for tag, row in report.items():
            print(f"| `{tag}` | " + " | ".join(f"{row[label]:.1f}" for label in args.labels) + " |")

    @staticmethod
    def text_encoder(args) -> None:
        disable_tf32()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        root = H3QuantProbe._source_root(args.source)
        tokenizers = TokenizerLoader.load_all(
            definitions=MiniMaxH3WeightDefinition.get_tokenizers(), model_path=str(root)
        )
        presentations = []
        for entry in args.prompts:
            label, prompt_file = entry.split(":")[:2]
            keyframes = []
            if label == args.image_label:
                keyframe = PIL.Image.open(args.image_path).convert("RGB")
                keyframes = [MiniMaxH3._fit_keyframe(keyframe, args.width, args.height)]
            presentations.append((label, Path(prompt_file).read_text(), keyframes))
        report_path = output_dir / "text_encoder_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        for variant in args.variants:
            started = time.perf_counter()
            encoder = H3QuantProbe._load_text_encoder(root, variant)
            load_seconds = time.perf_counter() - started
            holder = SimpleNamespace(
                prompt_embed_cache={}, conditioner=ResidentConditioner(encoder), tokenizers=tokenizers
            )
            rows = {}
            for label, prompt, keyframes in presentations:
                embeds, _ = MiniMaxH3._encode_presentation(holder, compose_prompt(prompt), keyframes)
                embeds = embeds.astype(mx.float32)
                mx.eval(embeds)
                reference_path = output_dir / f"reference_text_{label}.safetensors"
                if variant == "bf16":
                    mx.save_safetensors(str(reference_path), {"embeds": embeds})
                else:
                    mx.save_safetensors(str(output_dir / f"text_{variant}_{label}.safetensors"), {"embeds": embeds})
                rows[label] = H3QuantProbe._token_error(embeds, mx.load(str(reference_path))["embeds"])
            report[variant] = {
                "load_seconds": round(load_seconds, 1),
                "weights_gib": round(H3QuantProbe._parameter_bytes(encoder) / 1024**3, 2),
                "rows": rows,
            }
            report_path.write_text(json.dumps(report, indent=2))
            print(variant, json.dumps(report[variant]), flush=True)
            del encoder, holder
            gc.collect()
            mx.clear_cache()

    @staticmethod
    def variant_bits(variant: str, families: dict[str, tuple[str, ...]], path: str) -> int | None:
        # `q8` / `q4`: every linear the shipped policy quantizes, at that width. `q4:attn_v=8,adaln=6`
        # overrides families by name; `=0` leaves a family unquantized, `vision=8` quantizes the vision tower.
        base, _, listed = variant.partition(":")
        overrides = {
            name: int(bits) for name, _, bits in (entry.partition("=") for entry in listed.split(",") if entry)
        }
        dotted = f"{path}."
        default_bits = int(base[1:])
        # `edgeN=bits`: the first and last N of the 50 transformer blocks at another width.
        edge = next((name for name in overrides if name.startswith("edge")), None)
        if edge and path.startswith("transformer_blocks."):
            index, count = int(path.split(".")[1]), int(edge[4:])
            if index < count or index >= 50 - count:
                default_bits = overrides[edge]
        for name, fragments in families.items():
            if any(f"{fragment}." in dotted or dotted.startswith(f"{fragment}.") for fragment in fragments):
                if name in overrides:
                    return overrides[name] or None
                return None if name in H3QuantProbe.UNQUANTIZED_BY_DEFAULT else default_bits
        return default_bits

    @staticmethod
    def _capture_plan(args) -> list[tuple[str, str, set[int]]]:
        plan = []
        for entry in args.prompts:
            label, prompt_file, *steps = entry.split(":")
            plan.append((label, prompt_file, {int(step) for step in steps[0].split(",")} if steps else set()))
        return plan

    @staticmethod
    def _predicate(variant: str, families: dict[str, tuple[str, ...]]):
        def predicate(path: str, module):
            weight = getattr(module, "weight", None)
            if not hasattr(module, "to_quantized") or weight is None or weight.ndim != 2 or weight.shape[-1] % 64:
                return False
            if MiniMaxH3WeightDefinition.is_transformer_fp32_path(path):
                return False
            bits = H3QuantProbe.variant_bits(variant, families, path)
            return {"bits": bits} if bits else False

        return predicate

    @staticmethod
    def _source_root(source: str) -> Path:
        definition = MiniMaxH3WeightDefinition()
        return Path(PathResolution.resolve(path=source, patterns=definition.get_download_patterns()))

    @staticmethod
    def _load_transformer(source: str, variant: str, turbo_lora: str | None) -> tuple[MiniMaxH3Transformer, int]:
        root = H3QuantProbe._source_root(source)
        component = next(c for c in MiniMaxH3WeightDefinition().components() if c.name == "transformer")
        model = MiniMaxH3Transformer()
        StreamingWeightLoader.load_into(
            model=model,
            component_root=root,
            component=component,
            quantize_arg=None if variant == "bf16" else 8,
            quantization_predicate=H3QuantProbe._predicate(variant, H3QuantProbe.TRANSFORMER_FAMILIES),
        )
        weight_bytes = H3QuantProbe._parameter_bytes(model)
        if turbo_lora:
            (path,) = LoraResolution.resolve_paths([turbo_lora])
            LoRALoader.load_and_apply_lora_detailed(
                lora_mapping=MiniMaxH3LoRAMapping.get_mapping(),
                transformer=model,
                lora_paths=[path],
                lora_scales=[MiniMaxH3Initializer.adapter_scale(path)],
                state_dict_transform=MiniMaxH3LoRAMapping.normalize_state_dict,
            )
        return model, weight_bytes

    @staticmethod
    def _load_text_encoder(root: Path, variant: str) -> Qwen3VLModel:
        component = next(c for c in MiniMaxH3WeightDefinition().components() if c.name == "text_encoder")
        component.weight_files = H3Conditioner.needed_shards(root / component.hf_subdir)
        model = Qwen3VLModel(language_model=Qwen3VLTextModel(num_hidden_layers=TEXT_ENCODER_NUM_LAYERS))
        StreamingWeightLoader.load_into(
            model=model,
            component_root=root,
            component=component,
            quantize_arg=None if variant == "bf16" else 8,
            quantization_predicate=H3QuantProbe._predicate(variant, H3QuantProbe.TEXT_FAMILIES),
        )
        return model

    @staticmethod
    def _error(value: mx.array, reference: mx.array) -> dict[str, float]:
        value = np.asarray(value.astype(mx.float32), dtype=np.float64).reshape(-1)
        reference = np.asarray(reference.astype(mx.float32), dtype=np.float64).reshape(-1)
        if not (np.isfinite(value).all() and np.isfinite(reference).all()):
            return {"relative_rms": float("nan"), "cosine": float("nan")}
        # Explicit sums: Accelerate's float64 matmul raises spurious floating-point warnings on Apple Silicon.
        reference_norm = float(np.sqrt(np.sum(reference * reference)))
        value_norm = float(np.sqrt(np.sum(value * value)))
        relative_rms = float(np.sqrt(np.sum((value - reference) ** 2))) / max(reference_norm, 1e-30)
        cosine = float(np.sum(value * reference)) / max(value_norm * reference_norm, 1e-30)
        return {"relative_rms": round(relative_rms, 6), "cosine": round(cosine, 6)}

    @staticmethod
    def _token_error(value: mx.array, reference: mx.array) -> dict[str, float]:
        result = H3QuantProbe._error(value, reference)
        value = np.asarray(value.astype(mx.float32), dtype=np.float64)[0]
        reference = np.asarray(reference.astype(mx.float32), dtype=np.float64)[0]
        norms = np.sqrt(np.sum(reference * reference, axis=-1))
        relative = np.sqrt(np.sum((value - reference) ** 2, axis=-1)) / np.maximum(norms, 1e-30)
        result["token_relative_median"] = round(float(np.median(relative)), 6)
        result["token_relative_p90"] = round(float(np.percentile(relative, 90)), 6)
        result["tokens"] = int(reference.shape[0])
        return result

    @staticmethod
    def _parameter_bytes(module: nn.Module) -> int:
        from mlx.utils import tree_flatten

        return sum(value.nbytes for _, value in tree_flatten(module.parameters()))

    @staticmethod
    def _adaln_table_errors(model: MiniMaxH3Transformer, captures: list[Path]) -> dict:
        # Error of the six modulation tables themselves when only the AdaLN projection is quantized, per
        # (bits, group size), on the real timestep embeddings of the captured steps.
        report = {}
        names = ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
        for capture in captures:
            inputs = mx.load(str(capture))
            temb = model.time_embedder(model.time_proj(inputs["timestep"]).astype(mx.float32))
            activation = nn.silu(temb.astype(mx.float32))
            for block_index in (0, 24, 49):
                linear = model.transformer_blocks[block_index].adaln_proj.linear
                weight = linear.weight.astype(mx.float32)
                bias = linear.bias.astype(mx.float32)
                reference = (activation @ weight.T + bias).reshape(-1, 6, weight.shape[0] // 18)
                # Through 0.36 the module cast `silu(temb)` to `weight.dtype`, the packed uint32 once quantized.
                cast = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=8)
                truncated = cast(activation.astype(cast.weight.dtype)).reshape(-1, 6, weight.shape[0] // 18)
                report[f"{capture.stem}/block{block_index}/q8g64_input_cast_to_uint32"] = {
                    name: H3QuantProbe._error(truncated[:, index], reference[:, index])["relative_rms"]
                    for index, name in enumerate(names)
                }
                for bits, group_size in ((8, 64), (8, 32), (6, 64), (4, 64)):
                    packed, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
                    restored = mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)
                    table = (activation @ restored.T + bias).reshape(-1, 6, weight.shape[0] // 18)
                    key = f"{capture.stem}/block{block_index}/q{bits}g{group_size}"
                    report[key] = {
                        name: H3QuantProbe._error(table[:, index], reference[:, index])["relative_rms"]
                        for index, name in enumerate(names)
                    }
                    mx.eval(table)
                    del packed, scales, biases, restored, table
                mx.clear_cache()
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure MiniMax-H3 quantization error against the BF16 weights.")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_generation_arguments(command) -> None:
        command.add_argument("--package")
        command.add_argument("--base-model", default="minimax-h3-turbo-544p")
        command.add_argument("--prompts", nargs="+", required=True, help="label:prompt_file[:step,step]")
        command.add_argument("--width", type=int, default=640)
        command.add_argument("--height", type=int, default=352)
        command.add_argument("--frames", type=int, default=124)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--tag", required=True)
        command.add_argument("--output-dir", required=True)

    capture = commands.add_parser("capture")
    add_generation_arguments(capture)
    capture.add_argument("--image-path")
    capture.add_argument("--image-label", default="image")
    capture.set_defaults(run=H3QuantProbe.capture)

    generate = commands.add_parser("generate")
    add_generation_arguments(generate)
    generate.add_argument("--image-path")
    generate.add_argument("--image-label", default="image")
    generate.set_defaults(run=H3QuantProbe.generate)

    e2e = commands.add_parser("e2e")
    add_generation_arguments(e2e)
    e2e.add_argument("--transformer-variant", required=True)
    e2e.add_argument("--text-variant", required=True)
    e2e.add_argument("--image-path")
    e2e.add_argument("--image-label", default="image")
    e2e.set_defaults(run=H3QuantProbe.e2e)

    transformer = commands.add_parser("transformer")
    transformer.add_argument("--source", default="MiniMaxAI/MiniMax-H3")
    transformer.add_argument("--variants", nargs="+", required=True)
    transformer.add_argument(
        "--turbo-lora", default="hf:lightx2v/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
    )
    transformer.add_argument("--adaln", action="store_true")
    transformer.add_argument("--output-dir", required=True)
    transformer.set_defaults(run=H3QuantProbe.transformer)

    conditioning = commands.add_parser("conditioning")
    conditioning.add_argument("--source", default="MiniMaxAI/MiniMax-H3")
    conditioning.add_argument("--variants", nargs="+", required=True)
    conditioning.add_argument(
        "--turbo-lora", default="hf:lightx2v/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
    )
    conditioning.add_argument("--output-dir", required=True)
    conditioning.set_defaults(run=H3QuantProbe.conditioning)

    ballast = commands.add_parser("ballast")
    ballast.add_argument("--gib", type=float, required=True)
    ballast.set_defaults(run=H3QuantProbe.ballast)

    report = commands.add_parser("report")
    report.add_argument("--output-dir", required=True)
    report.set_defaults(run=H3QuantProbe.report)

    compare = commands.add_parser("compare")
    compare.add_argument("--reference-tag", required=True)
    compare.add_argument("--tags", nargs="+", required=True)
    compare.add_argument("--labels", nargs="+", required=True)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--output-dir", required=True)
    compare.set_defaults(run=H3QuantProbe.compare)

    text_encoder = commands.add_parser("text-encoder")
    text_encoder.add_argument("--source", default="MiniMaxAI/MiniMax-H3")
    text_encoder.add_argument("--variants", nargs="+", required=True)
    text_encoder.add_argument("--prompts", nargs="+", required=True, help="label:prompt_file")
    text_encoder.add_argument("--image-path")
    text_encoder.add_argument("--image-label", default="image")
    text_encoder.add_argument("--width", type=int, default=640)
    text_encoder.add_argument("--height", type=int, default=352)
    text_encoder.add_argument("--output-dir", required=True)
    text_encoder.set_defaults(run=H3QuantProbe.text_encoder)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
