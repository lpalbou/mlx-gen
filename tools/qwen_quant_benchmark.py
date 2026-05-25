import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx

from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    if args.task == "image":
        model_config = ModelConfig.from_name(args.model_name) if args.model_name else ModelConfig.qwen_image()
        model = QwenImage(model_path=str(args.model_path), model_config=model_config)
    else:
        model_config = ModelConfig.from_name(args.model_name) if args.model_name else ModelConfig.qwen_image_edit()
        model = QwenImageEdit(model_path=str(args.model_path), model_config=model_config)
    load_seconds = time.perf_counter() - started

    _count_quantized_linears(model, args.model_label)

    for repeat in range(args.repeats):
        seed = args.seed + repeat
        output_path = args.output_dir / f"{args.model_label}_{args.task}_r{repeat + 1}.png"
        gc.collect()
        if args.clear_cache_between_repeats:
            mx.clear_cache()

        started = time.perf_counter()
        if args.task == "image":
            generated = model.generate_image(
                seed=seed,
                prompt=args.prompt,
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                guidance=args.guidance,
                negative_prompt=args.negative_prompt,
            )
        else:
            generated = model.generate_image(
                seed=seed,
                prompt=args.prompt,
                image_paths=[str(path) for path in args.image_paths],
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                guidance=args.guidance,
                negative_prompt=args.negative_prompt,
            )
        total_seconds = time.perf_counter() - started
        mx.eval(mx.array(0))
        generated.save(output_path, overwrite=True)

        print(
            json.dumps(
                {
                    "model_label": args.model_label,
                    "model_path": str(args.model_path),
                    "task": args.task,
                    "repeat": repeat + 1,
                    "seed": seed,
                    "steps": args.steps,
                    "width": generated.image.size[0],
                    "height": generated.image.size[1],
                    "load_seconds": load_seconds,
                    "total_seconds": total_seconds,
                    "denoise_seconds": generated.generation_time,
                    "output_path": str(output_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--task", choices=["image", "edit"], required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image-paths", type=Path, nargs="*", default=[])
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("validation_outputs/qwen_quant_bench"))
    parser.add_argument("--clear-cache-between-repeats", action="store_true")
    return parser.parse_args()


def _count_quantized_linears(model, label: str) -> None:
    # Log representative modules so the benchmark proves MLX quantized matmul modules loaded.
    for path in [
        "transformer.transformer_blocks.0.attn.to_q",
        "transformer.transformer_blocks.0.img_mod_linear",
        "transformer.transformer_blocks.0.txt_mod_linear",
        "text_encoder.encoder.layers.0.self_attn.q_proj",
    ]:
        current = _resolve_path(model, path)
        print(
            json.dumps(
                {
                    "model_label": label,
                    "module_path": path,
                    "module_type": type(current).__name__,
                    "bits": getattr(current, "bits", None),
                    "group_size": getattr(current, "group_size", None),
                    "mode": getattr(current, "mode", None),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _resolve_path(model, path: str):
    current = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


if __name__ == "__main__":
    main()
