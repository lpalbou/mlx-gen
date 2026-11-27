# Planned: Safe Torch checkpoint loading

## Metadata
- Created: 2026-06-27
- Status: Completed
- Completed: 2026-06-27

## ADR status
- Governing ADRs: ADR 0002
- ADR impact: Needs new ADR only if MLX-Gen keeps a long-term trusted-pickle exception policy.

## Context
The audit found a direct unsafe PyTorch checkpoint loading path for model artifacts. This matters
because users can point MLX-Gen at local folders or remote model repositories.

## Current code reality
- `src/mflux/models/common/weights/loading/weight_loader.py` calls
  `torch.load(file_path, map_location="cpu", weights_only=False)` in `_load_torch_checkpoint`.
- The unsafe path is reachable through `torch_checkpoint` and `torch_tensor` loading modes.
- SeedVR2 official layouts and Depth Pro use `.pth` / `.pt` checkpoint inputs.
- Safetensors and MLX-native loading paths do not use this pickle path.

## Problem
`weights_only=False` allows arbitrary Python pickle execution when the checkpoint file is
untrusted. The current loader does not distinguish trusted bundled source layouts from arbitrary
user-provided `.pt` or `.pth` files.

## What we want to do
Make Torch checkpoint loading fail closed by default while preserving supported official model
loads where they are representable with safe PyTorch loading.

## Why
Model loading is a high-risk boundary. A model file should not be able to execute code simply
because a user asked MLX-Gen to prepare or load it.

## Requirements
- Use `torch.load(..., weights_only=True)` for Torch checkpoint and tensor loading where possible.
- Reject checkpoint object types that are not plain tensors or tensor state dicts.
- Keep SeedVR2 official `.pth` and `pos_emb.pt` loading covered by focused tests.
- Document in code errors that unsupported legacy pickle checkpoints must be converted or handled
  by an explicit trusted workflow, not silently loaded.

## Suggested implementation
Change `_load_torch_checkpoint` to call `weights_only=True`, then keep `_torch_object_to_mx_weights`
as the narrowing gate for supported tensor shapes. Add tests that assert the safe flag is used and
that current saved tensor/state-dict fixtures still load.

## Scope
- `WeightLoader` Torch checkpoint and tensor paths.
- Focused checkpoint loader tests.

## Non-goals
- Do not add broad arbitrary pickle allowlists in this item.
- Do not replace all existing official `.pt/.pth` source layouts with safetensors packages.

## Dependencies and related tasks
- Related: `0056_wan_vlm_bf16_performance_hardening.md` for BF16 precision handling.

## Expected outcomes
- The default model loader no longer uses `weights_only=False`.
- Current supported Torch tensor/state-dict fixtures still load.
- Unsupported pickle-only checkpoints fail with a clear error.

## Validation
- `uv run pytest tests/weights/test_seedvr2_official_checkpoint_loading.py -q`
- `uv run ruff check src/mflux/models/common/weights/loading/weight_loader.py tests/weights/test_seedvr2_official_checkpoint_loading.py`

## Progress update: 2026-06-27
- Switched `_load_torch_checkpoint` to `torch.load(..., weights_only=True)`.
- Added a focused test that asserts the safe load flag is used.
- Re-ran SeedVR2 official checkpoint fixture tests to confirm current tensor/state-dict layouts
  still load.
- Rejected empty or mixed non-tensor checkpoint dictionaries instead of silently filtering them.
- Preserved Torch BF16 tensors as MLX BF16 as part of the related 0056 precision policy work.

## Progress checklist
- [x] Switch Torch checkpoint loading to `weights_only=True`.
- [x] Preserve official SeedVR2 checkpoint/tensor fixture loading.
- [x] Add a focused assertion for the safe load flag.
- [x] Reject empty and mixed non-tensor checkpoint payloads.

## Implementation notes
Treat this as a security boundary. If a checkpoint fails after the safe-load change, do not restore
unsafe pickle loading without an explicit trusted-path design.
