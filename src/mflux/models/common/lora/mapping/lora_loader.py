import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget
from mflux.models.common.resolution.lora_resolution import LoraResolution


class LoRAApplicationError(ValueError):
    pass


@dataclass
class PatternMatch:
    source_pattern: str
    target_path: str
    matrix_name: str  # "lora_A", "lora_B", or "alpha"
    transpose: bool
    transform: Callable[[mx.array], mx.array] | None = None


@dataclass(frozen=True)
class LoRAFileReport:
    requested_path: str
    resolved_path: str
    scale: float
    role: str | None
    total_key_count: int
    matched_key_count: int
    unmatched_key_count: int
    applied_target_count: int

    def to_dict(self) -> dict:
        return {
            "requested_path": self.requested_path,
            "resolved_path": self.resolved_path,
            "scale": round(self.scale, 4),
            "role": self.role,
            "total_key_count": self.total_key_count,
            "matched_key_count": self.matched_key_count,
            "unmatched_key_count": self.unmatched_key_count,
            "applied_target_count": self.applied_target_count,
        }


@dataclass(frozen=True)
class LoRAApplicationResult:
    resolved_paths: list[str]
    resolved_scales: list[float]
    reports: tuple[LoRAFileReport, ...]

    def extra_metadata(self) -> dict:
        if not self.reports:
            return {}
        return {
            "lora_application_reports": [report.to_dict() for report in self.reports],
            "lora_applied_file_count": len(self.reports),
            "lora_applied_target_count": sum(report.applied_target_count for report in self.reports),
        }


class LoRALoader:
    _debug_enabled = False

    @staticmethod
    def set_debug_enabled(debug_enabled: bool) -> None:
        LoRALoader._debug_enabled = debug_enabled

    @staticmethod
    def load_and_apply_lora(
        lora_mapping: list[LoRATarget],
        transformer: nn.Module,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        role: str | None = None,
        state_dict_transform: Callable[[dict[str, mx.array], nn.Module], dict[str, mx.array]] | None = None,
    ) -> tuple[list[str], list[float]]:
        result = LoRALoader.load_and_apply_lora_detailed(
            lora_mapping=lora_mapping,
            transformer=transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            role=role,
            state_dict_transform=state_dict_transform,
        )
        return result.resolved_paths, result.resolved_scales

    @staticmethod
    def load_and_apply_lora_detailed(
        lora_mapping: list[LoRATarget],
        transformer: nn.Module,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        role: str | None = None,
        state_dict_transform: Callable[[dict[str, mx.array], nn.Module], dict[str, mx.array]] | None = None,
    ) -> LoRAApplicationResult:
        resolved_paths = LoraResolution.resolve_paths(lora_paths)
        if not resolved_paths:
            if lora_scales:
                raise LoRAApplicationError("--lora-scales requires --lora-paths.")
            return LoRAApplicationResult(resolved_paths=[], resolved_scales=[], reports=())

        resolved_scales = LoraResolution.resolve_scales(lora_scales, len(resolved_paths))
        if len(resolved_scales) != len(resolved_paths):
            raise ValueError(
                f"Number of LoRA scales ({len(resolved_scales)}) must match number of LoRA files ({len(resolved_paths)})"
            )

        print(f"📦 Loading {len(resolved_paths)} LoRA file(s)...")

        reports: list[LoRAFileReport] = []
        requested_paths = lora_paths or resolved_paths
        for requested_path, resolved_path, scale in zip(requested_paths, resolved_paths, resolved_scales):
            reports.append(
                LoRALoader._apply_single_lora(
                    transformer,
                    requested_path=requested_path,
                    resolved_path=resolved_path,
                    scale=scale,
                    lora_mapping=lora_mapping,
                    role=role,
                    state_dict_transform=state_dict_transform,
                )
            )

        print("✅ All LoRA weights applied successfully")

        return LoRAApplicationResult(
            resolved_paths=resolved_paths,
            resolved_scales=resolved_scales,
            reports=tuple(reports),
        )

    @staticmethod
    def extra_metadata_for_model(model) -> dict | None:
        result = getattr(model, "lora_application_result", None)
        if result is None:
            return None
        metadata = result.extra_metadata()
        return metadata or None

    @staticmethod
    def _apply_single_lora(
        transformer: nn.Module,
        requested_path: str,
        resolved_path: str,
        scale: float,
        lora_mapping: list[LoRATarget],
        *,
        role: str | None,
        state_dict_transform: Callable[[dict[str, mx.array], nn.Module], dict[str, mx.array]] | None,
    ) -> LoRAFileReport:
        if not Path(resolved_path).exists():
            raise LoRAApplicationError(f"LoRA file not found: {resolved_path}")

        print(f"🔧 Applying LoRA: {Path(resolved_path).name} (scale={scale})")

        try:
            weights = dict(mx.load(resolved_path, return_metadata=True)[0].items())
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            raise LoRAApplicationError(f"Failed to load LoRA file {resolved_path}: {e}") from e
        if state_dict_transform is not None:
            weights = state_dict_transform(weights, transformer)

        pattern_mappings = LoRALoader._build_pattern_mappings(lora_mapping)

        applied_count, matched_keys = LoRALoader._apply_lora_with_mapping(transformer, weights, scale, pattern_mappings, role=role)

        total_keys = len(weights)
        unmatched_keys = set(weights.keys()) - matched_keys

        if not matched_keys:
            raise LoRAApplicationError(f"LoRA file {resolved_path} did not match any known adapter keys.")
        if applied_count <= 0:
            raise LoRAApplicationError(
                f"LoRA file {resolved_path} matched {len(matched_keys)} keys but did not apply to any model layer."
            )

        print(f"   ✅ Applied to {applied_count} layers ({len(matched_keys)}/{total_keys} keys matched)")

        if unmatched_keys:
            print(f"   ⚠️  {len(unmatched_keys)} unmatched keys in LoRA file:")
            for key in sorted(unmatched_keys)[:5]:
                print(f"      - {key}")
            if len(unmatched_keys) > 5:
                print(f"      ... and {len(unmatched_keys) - 5} more")

        return LoRAFileReport(
            requested_path=requested_path,
            resolved_path=resolved_path,
            scale=scale,
            role=role,
            total_key_count=total_keys,
            matched_key_count=len(matched_keys),
            unmatched_key_count=len(unmatched_keys),
            applied_target_count=applied_count,
        )

    @staticmethod
    def _build_pattern_mappings(targets: list[LoRATarget]) -> list[PatternMatch]:
        mappings = []

        for target in targets:
            # Add up weight patterns (lora_B)
            mappings.extend(
                PatternMatch(
                    source_pattern=pattern,
                    target_path=target.model_path,
                    matrix_name="lora_B",
                    transpose=True,
                    transform=target.up_transform,
                )
                for pattern in target.possible_up_patterns
            )

            # Add down weight patterns (lora_A)
            mappings.extend(
                PatternMatch(
                    source_pattern=pattern,
                    target_path=target.model_path,
                    matrix_name="lora_A",
                    transpose=True,
                    transform=target.down_transform,
                )
                for pattern in target.possible_down_patterns
            )

            # Add alpha patterns (no transpose, no transform)
            mappings.extend(
                PatternMatch(
                    source_pattern=pattern,
                    target_path=target.model_path,
                    matrix_name="alpha",
                    transpose=False,
                    transform=None,
                )
                for pattern in target.possible_alpha_patterns
            )

        return mappings

    @staticmethod
    def _apply_lora_with_mapping(
        transformer: nn.Module,
        weights: dict,
        scale: float,
        pattern_mappings: list[PatternMatch],
        *,
        role: str | None,
    ) -> tuple[int, set]:
        applied_count = 0
        lora_data_by_target: dict[str, dict] = {}
        matched_keys: set[str] = set()

        # For each weight key, find ALL matching patterns (not just first)
        # This allows multiple targets to use the same source (e.g., QKV split)
        for weight_key, weight_value in weights.items():
            for mapping in pattern_mappings:
                match_result = LoRALoader._match_pattern(weight_key, mapping.source_pattern)
                if match_result is None:
                    continue

                matched_keys.add(weight_key)
                block_idx = match_result

                # Resolve target path with block index if needed
                target_path = mapping.target_path
                if block_idx is not None and "{block}" in target_path:
                    target_path = target_path.format(block=block_idx)

                # Apply transform if specified
                transformed_value = weight_value
                if mapping.transform is not None:
                    transformed_value = mapping.transform(weight_value)

                # Apply transpose if needed
                if mapping.transpose:
                    transformed_value = transformed_value.T

                # Store for this target
                if target_path not in lora_data_by_target:
                    lora_data_by_target[target_path] = {}

                lora_data_by_target[target_path][mapping.matrix_name] = transformed_value

        # Apply LoRA to each target
        for target_path, lora_data in lora_data_by_target.items():
            if LoRALoader._apply_lora_matrices_to_target(transformer, target_path, lora_data, scale, role=role):
                applied_count += 1

        return applied_count, matched_keys

    @staticmethod
    def _match_pattern(weight_key: str, pattern: str) -> int | None:
        if "{block}" in pattern:
            # Find all numbers in the weight key
            numbers_in_key = re.findall(r"\d+", weight_key)
            for num_str in numbers_in_key:
                test_block_idx = int(num_str)
                concrete_pattern = pattern.replace("{block}", str(test_block_idx))
                if weight_key == concrete_pattern:
                    return test_block_idx
            return None
        else:
            if weight_key == pattern:
                return 0  # Return 0 to indicate match (no block)
            return None

    @staticmethod
    def _apply_lora_matrices_to_target(
        transformer: nn.Module, target_path: str, lora_data: dict, scale: float, *, role: str | None
    ) -> bool:
        # Navigate to the target layer
        current_module = transformer
        path_parts = target_path.split(".")

        try:
            for part in path_parts:
                if part.isdigit():
                    current_module = current_module[int(part)]
                elif isinstance(current_module, dict) and part in current_module:
                    current_module = current_module[part]
                else:
                    current_module = getattr(current_module, part)
        except (AttributeError, IndexError, KeyError):
            raise LoRAApplicationError(f"Could not find LoRA target path: {target_path}")

        if "lora_A" not in lora_data or "lora_B" not in lora_data:
            raise LoRAApplicationError(f"Missing required LoRA matrices for target path: {target_path}")

        # Values are already transformed and transposed
        lora_A = lora_data["lora_A"]
        lora_B = lora_data["lora_B"]

        alpha_scale = 1.0
        if "alpha" in lora_data:
            alpha_value = lora_data["alpha"]
            rank = lora_A.shape[1]
            alpha_scale = float(alpha_value) / rank

        effective_scale = scale

        is_linear = hasattr(current_module, "weight")
        is_lora_linear = isinstance(current_module, LoRALinear)
        is_fused_linear = isinstance(current_module, FusedLoRALinear)

        if is_linear or is_lora_linear or is_fused_linear:
            LoRALoader._validate_lora_matrix_shapes(current_module, lora_A, lora_B, target_path)
            if is_lora_linear:
                if LoRALoader._debug_enabled:
                    print(f"   🔀 Fusing with existing LoRA at {target_path}")
                lora_layer = LoRALinear.from_linear(current_module.linear, r=lora_A.shape[1], scale=effective_scale)
                lora_layer._mflux_lora_role = role
                lora_layer.lora_A = lora_A
                lora_layer.lora_B = lora_B
                if "alpha" in lora_data:
                    lora_layer.lora_B = lora_layer.lora_B * alpha_scale
                fused_layer = FusedLoRALinear(base_linear=current_module.linear, loras=[current_module, lora_layer])
                replacement_layer = fused_layer
            elif is_fused_linear:
                if LoRALoader._debug_enabled:
                    print(f"   🔀 Adding to existing fusion at {target_path}")
                lora_layer = LoRALinear.from_linear(
                    current_module.base_linear, r=lora_A.shape[1], scale=effective_scale
                )
                lora_layer._mflux_lora_role = role
                lora_layer.lora_A = lora_A
                lora_layer.lora_B = lora_B
                if "alpha" in lora_data:
                    lora_layer.lora_B = lora_layer.lora_B * alpha_scale
                fused_layer = FusedLoRALinear(
                    base_linear=current_module.base_linear, loras=current_module.loras + [lora_layer]
                )
                replacement_layer = fused_layer
            else:
                # First LoRA on this layer
                lora_layer = LoRALinear.from_linear(current_module, r=lora_A.shape[1], scale=effective_scale)
                lora_layer._mflux_lora_role = role
                lora_layer.lora_A = lora_A
                lora_layer.lora_B = lora_B
                if "alpha" in lora_data:
                    lora_layer.lora_B = lora_layer.lora_B * alpha_scale
                replacement_layer = lora_layer

            # Replace the layer in the parent module
            parent_module = transformer
            for part in path_parts[:-1]:
                if part.isdigit():
                    parent_module = parent_module[int(part)]
                elif isinstance(parent_module, dict) and part in parent_module:
                    parent_module = parent_module[part]
                else:
                    parent_module = getattr(parent_module, part)

            final_attr = path_parts[-1]
            if final_attr.isdigit():
                parent_module[int(final_attr)] = replacement_layer
            elif isinstance(parent_module, dict) and final_attr in parent_module:
                parent_module[final_attr] = replacement_layer
            else:
                setattr(parent_module, final_attr, replacement_layer)

            return True

        raise LoRAApplicationError(f"LoRA target path {target_path} is not a linear layer.")

    @staticmethod
    def _validate_lora_matrix_shapes(current_module, lora_A: mx.array, lora_B: mx.array, target_path: str) -> None:
        base_linear = current_module
        if isinstance(current_module, LoRALinear):
            base_linear = current_module.linear
        elif isinstance(current_module, FusedLoRALinear):
            base_linear = current_module.base_linear

        if not hasattr(base_linear, "weight"):
            raise LoRAApplicationError(f"LoRA target path {target_path} is not a linear layer.")

        output_dims, input_dims = base_linear.weight.shape
        if isinstance(base_linear, nn.QuantizedLinear):
            input_dims *= 32 // base_linear.bits

        rank = lora_A.shape[1] if len(lora_A.shape) == 2 else None
        expected_a0 = input_dims
        expected_b1 = output_dims
        if (
            len(lora_A.shape) != 2
            or len(lora_B.shape) != 2
            or lora_A.shape[0] != expected_a0
            or lora_B.shape[1] != expected_b1
            or rank != lora_B.shape[0]
        ):
            raise LoRAApplicationError(
                f"LoRA matrices for {target_path} are incompatible with the selected model: "
                f"expected lora_A ({expected_a0}, rank) and lora_B (rank, {expected_b1}), "
                f"got lora_A {tuple(lora_A.shape)} and lora_B {tuple(lora_B.shape)}."
            )
