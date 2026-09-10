"""LoRA targets for the MiniMax-H3 transformer, in both key layouts adapters are published in.

Two producers exist. The lightx2v Turbo adapters are PEFT files over the diffusers module names
(`transformer_blocks.N.attn.to_q`, `ff.net.0.proj`, ...). Community adapters (ai-toolkit, the
reference `generate.py`, ComfyUI, kohya/musubi-tuner) are trained against the original checkpoint's
module names: `blocks.N.attn.qkv_proj` (fused QKV), `attn.out_proj`, `mlp.fc1` / `mlp.fc2`,
`token_refiner.blocks.N`, and the standalone `final_layer.*` / patch / condition / timestep projections.

The original layout differs from ours in two places that are not just renames, and both follow the
diffusers checkpoint converter (`convert_minimax_h3_to_diffusers.py`):

- `attn.qkv_proj` is `[q_all; k_all; v_all]`: `lora_A` is shared by `to_q` / `to_k` / `to_v`, and
  `lora_B` is split into row thirds.
- `mlp.fc1` packs `[gate; value]` and the reference computes `silu(gate) * value`; our SwiGLU packs
  `[value; gate]`, so the two row halves of `lora_B` swap places (`lora_A` is untouched).

DiffSynth-Studio adapters run the raw checkpoint's per-head interleaved fused QKV verbatim, which
is indistinguishable by shape from the reordered layout; they are recognised by PEFT's `.default.`
infix over the original names and rejected until a real file validates the de-interleave.
"""

import re

import mlx.core as mx

from mflux.models.common.lora.mapping.lora_loader import LoRAApplicationError
from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class MiniMaxH3LoRAMapping(LoRAMapping):
    """PEFT / kohya LoRAs over the diffusers or the original MiniMax-H3 transformer module names."""

    TARGET_MODULES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
    _BLOCK_PREFIXES = ("transformer_blocks.{block}", "token_refiner.refiner_blocks.{block}")
    _STATE_DICT_PREFIXES = ("", "transformer.", "diffusion_model.")

    # Original-checkpoint layout: `(source module, target module)` per block and standalone.
    ORIGINAL_STATE_DICT_PREFIXES = ("", "diffusion_model.")
    ORIGINAL_BLOCK_PREFIXES = (
        ("blocks.{block}", "transformer_blocks.{block}"),
        ("token_refiner.blocks.{block}", "token_refiner.refiner_blocks.{block}"),
    )
    ORIGINAL_BLOCK_MODULES = (
        ("attn.out_proj", "attn.to_out.0"),
        ("mlp.fc2", "ff.net.2"),
        ("adaln_proj.linear", "adaln_proj.linear"),
    )
    ORIGINAL_STANDALONE_MODULES = (
        ("video_patch_proj", "proj_in"),
        ("audio_patch_proj", "audio_proj_in"),
        ("condition_proj", "context_embedder"),
        ("time_embedder.proj_in", "time_embedder.linear_1"),
        ("time_embedder.proj_out", "time_embedder.linear_2"),
        ("final_layer.adaln_proj.linear", "norm_out.linear"),
        ("final_layer.video_out", "proj_out"),
        ("final_layer.audio_out", "audio_proj_out"),
    )
    DIFFUSERS_STANDALONE_MODULES = tuple(target for _, target in ORIGINAL_STANDALONE_MODULES)

    _UP_SUFFIXES = ("lora_B.weight", "lora_up.weight")
    _DOWN_SUFFIXES = ("lora_A.weight", "lora_down.weight")

    # musubi-tuner (kohya sd-scripts) flattens every module path under `lora_unet_` with `.` -> `_`.
    # H3's own names contain underscores, so the dot path is recovered by matching whole names.
    _FLATTENED_MODULES = (
        (r"blocks_(\d+)_attn_(qkv|out)_proj", r"blocks.\1.attn.\2_proj"),
        (r"blocks_(\d+)_mlp_fc([12])", r"blocks.\1.mlp.fc\2"),
        (r"blocks_(\d+)_adaln_proj_linear", r"blocks.\1.adaln_proj.linear"),
        (r"token_refiner_blocks_(\d+)_attn_(qkv|out)_proj", r"token_refiner.blocks.\1.attn.\2_proj"),
        (r"token_refiner_blocks_(\d+)_mlp_fc([12])", r"token_refiner.blocks.\1.mlp.fc\2"),
        (r"(video|audio)_patch_proj", r"\1_patch_proj"),
        (r"condition_proj", "condition_proj"),
        (r"time_embedder_proj_(in|out)", r"time_embedder.proj_\1"),
        (r"final_layer_adaln_proj_linear", "final_layer.adaln_proj.linear"),
        (r"final_layer_(video|audio)_out", r"final_layer.\1_out"),
    )
    _FLATTENED_PREFIX = "lora_unet_"

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets: list[LoRATarget] = []
        # diffusers layout (lightx2v Turbo and anything trained on `MiniMaxH3Transformer3DModel`).
        targets.extend(
            MiniMaxH3LoRAMapping._diffusers_target(f"{block_prefix}.{module}")
            for block_prefix in MiniMaxH3LoRAMapping._BLOCK_PREFIXES
            for module in MiniMaxH3LoRAMapping.TARGET_MODULES
        )
        targets.append(MiniMaxH3LoRAMapping._diffusers_target("transformer_blocks.{block}.adaln_proj.linear"))
        targets.extend(
            MiniMaxH3LoRAMapping._diffusers_target(module)
            for module in MiniMaxH3LoRAMapping.DIFFUSERS_STANDALONE_MODULES
        )
        # Original-checkpoint layout.
        for source_prefix, target_prefix in MiniMaxH3LoRAMapping.ORIGINAL_BLOCK_PREFIXES:
            for index, projection in enumerate(("to_q", "to_k", "to_v")):
                targets.append(
                    MiniMaxH3LoRAMapping._original_target(
                        f"{source_prefix}.attn.qkv_proj",
                        f"{target_prefix}.attn.{projection}",
                        up_transform=MiniMaxH3LoRAMapping.qkv_rows_third(index),
                    )
                )
            targets.append(
                MiniMaxH3LoRAMapping._original_target(
                    f"{source_prefix}.mlp.fc1",
                    f"{target_prefix}.ff.net.0.proj",
                    up_transform=MiniMaxH3LoRAMapping.swap_row_halves,
                )
            )
            for source_module, target_module in MiniMaxH3LoRAMapping.ORIGINAL_BLOCK_MODULES:
                if source_module.startswith("adaln_proj") and source_prefix.startswith("token_refiner"):
                    continue  # refiner blocks carry no AdaLN projection
                targets.append(
                    MiniMaxH3LoRAMapping._original_target(
                        f"{source_prefix}.{source_module}", f"{target_prefix}.{target_module}"
                    )
                )
        for source_module, target_module in MiniMaxH3LoRAMapping.ORIGINAL_STANDALONE_MODULES:
            targets.append(MiniMaxH3LoRAMapping._original_target(source_module, target_module))
        return targets

    @staticmethod
    def _diffusers_target(path: str) -> LoRATarget:
        prefixes = MiniMaxH3LoRAMapping._STATE_DICT_PREFIXES
        return LoRATarget(
            model_path=path,
            possible_up_patterns=[f"{p}{path}.{s}" for p in prefixes for s in MiniMaxH3LoRAMapping._UP_SUFFIXES],
            possible_down_patterns=[f"{p}{path}.{s}" for p in prefixes for s in MiniMaxH3LoRAMapping._DOWN_SUFFIXES],
            possible_alpha_patterns=[f"{p}{path}.alpha" for p in prefixes],
        )

    @staticmethod
    def _original_target(source: str, target: str, up_transform=None) -> LoRATarget:
        prefixes = MiniMaxH3LoRAMapping.ORIGINAL_STATE_DICT_PREFIXES
        return LoRATarget(
            model_path=target,
            possible_up_patterns=[f"{p}{source}.{s}" for p in prefixes for s in MiniMaxH3LoRAMapping._UP_SUFFIXES],
            possible_down_patterns=[f"{p}{source}.{s}" for p in prefixes for s in MiniMaxH3LoRAMapping._DOWN_SUFFIXES],
            possible_alpha_patterns=[f"{p}{source}.alpha" for p in prefixes],
            up_transform=up_transform,
        )

    @staticmethod
    def qkv_rows_third(index: int):
        """`lora_B` rows of a fused `[q_all; k_all; v_all]` projection for one of `to_q` / `to_k` / `to_v`."""

        def transform(tensor: mx.array) -> mx.array:
            rows = tensor.shape[0]
            if rows % 3:
                raise LoRAApplicationError(
                    f"A fused MiniMax-H3 QKV LoRA up-projection needs a row count divisible by 3, got {rows}."
                )
            third = rows // 3
            return tensor[index * third : (index + 1) * third]

        return transform

    @staticmethod
    def swap_row_halves(tensor: mx.array) -> mx.array:
        """`[gate; value]` rows of the original `mlp.fc1` to our SwiGLU's `[value; gate]`."""
        rows = tensor.shape[0]
        if rows % 2:
            raise LoRAApplicationError(
                f"A fused MiniMax-H3 SwiGLU LoRA up-projection needs an even row count, got {rows}."
            )
        half = rows // 2
        return mx.concatenate([tensor[half:], tensor[:half]], axis=0)

    @staticmethod
    def normalize_state_dict(weights: dict[str, mx.array], transformer=None) -> dict[str, mx.array]:
        """Recover dotted original-layout keys from musubi-tuner's flattened `lora_unet_` names."""
        prefix = MiniMaxH3LoRAMapping._FLATTENED_PREFIX
        if not any(key.startswith(prefix) for key in weights):
            return weights
        normalized: dict[str, mx.array] = {}
        for key, value in weights.items():
            if not key.startswith(prefix):
                normalized[key] = value
                continue
            module, _, suffix = key[len(prefix) :].partition(".")
            for pattern, replacement in MiniMaxH3LoRAMapping._FLATTENED_MODULES:
                if re.fullmatch(pattern, module):
                    normalized[f"{re.sub(pattern, replacement, module)}.{suffix}"] = value
                    break
            else:
                raise LoRAApplicationError(
                    f"`{key}` does not name a MiniMax-H3 module in musubi-tuner's flattened `lora_unet_` layout."
                )
        return normalized

    @staticmethod
    def reject_unsupported_layout(keys) -> None:
        """Fail before loading on a layout whose fused QKV rows we would apply in the wrong order."""
        if any(".attn.qkv_proj.lora_" in key and ".default.weight" in key for key in keys):
            raise LoRAApplicationError(
                "This MiniMax-H3 adapter keeps the raw checkpoint's per-head interleaved fused QKV "
                "(DiffSynth-Studio layout: `.lora_A.default.` over `attn.qkv_proj`). That order is not "
                "supported yet; re-export the adapter with ai-toolkit, musubi-tuner or diffusers module names."
            )
