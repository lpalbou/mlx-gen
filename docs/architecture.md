# Architecture

MLX-Gen is an independent package forked from mflux. It keeps the MLX-native model runtime from mflux while exposing a cleaner `mlxgen` command surface for new users and applications. The first supported video path is Wan2.2 TI2V text-to-video.

## Package Shape

- PyPI distribution: `mlx-gen`
- Public CLI root: `mlxgen`
- New application import identity: `mlxgen`
- Current runtime internals: primarily `mflux.*`

The project keeps some mflux vocabulary and compatibility entry points while the fork evolves. New docs and integrations should use `mlxgen` commands and treat `mflux.*` internals as inherited implementation detail unless a specific model class currently requires them.

## Command Boundary

The public command surface separates setup from inference:

- `mlxgen download` is an explicit cache population command.
- `mlxgen prepare` is an explicit local model-folder creation command.
- `mlxgen generate` is the inference command and does not start downloads by default.

This boundary is important for embedded workflow systems such as AbstractVision: a generation request should not unexpectedly start a large network transfer in the middle of a larger job.

## Model File Lifecycle

Source model files usually come from Hugging Face. They can be used in two ways:

1. Cache the source files with `mlxgen download` and run by alias or repository id.
2. Create a reusable MLX-Gen folder with `mlxgen prepare --model ... --path ... --quantize ...`.

Prepared folders use the MLX/mflux saved-weight layout. They may contain MLX quantization tensors and generated Hugging Face model cards. They are intended for MLX-Gen and compatible mflux code, not direct Diffusers or Transformers loading.

Video support follows the same setup/runtime boundary. Wan2.2 TI2V text-to-video loads local source files and writes MP4 output; Wan image-to-video is held back until its Diffusers first-frame latent-conditioning semantics are ported.

## Runtime Failure Contract

Runtime model construction and generation are cache-only. Missing required files raise `DownloadRequiredError`, which is also a `FileNotFoundError` for compatibility with existing callers.

The error includes actionable command fields such as `download_command` and, when applicable, `prepare_command`. CLI entry points print the human-readable error without a traceback for common missing-artifact cases.

## Quantization Policy

Quantization is model-specific. Qwen and ERNIE q4 paths use mixed q4/q8 policies because fully q4 checkpoints can lose coherent generative behavior for those model families. Other model families keep their existing quantization predicates unless their model behavior requires a dedicated policy.

See [Quantization](quantization.md) for the current rules.

## Python Integration Boundary

The current Python API still exposes many model classes through inherited mflux modules. MLX-Gen's near-term integration contract is:

- prepare files before constructing models;
- fail early when required artifacts are missing;
- keep model instances as stateful runtime objects;
- expose clearer public orchestration APIs over time without breaking existing compatibility paths unnecessarily.

See [Python Integration](python-integration.md) and [API And CLI](api.md) for current usage.
