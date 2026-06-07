# Architecture

MLX-Gen is an independent package forked from mflux. It keeps the MLX-native model runtime from mflux while exposing a cleaner `mlxgen` command surface for new users and applications. The supported video paths include Wan2.2 TI2V-5B text-to-video, TI2V-5B first-frame image-to-video, Wan2.2 A14B text-to-video, and Wan2.2 A14B image-to-video. SeedVR2 image super-resolution currently uses the dedicated `mflux-upscale-seedvr2` entry point.

## Package Shape

- PyPI distribution: `mlx-gen`
- Public CLI root: `mlxgen`
- Dedicated compatibility CLI for SeedVR2 upscaling: `mflux-upscale-seedvr2`
- New application import identity: `mlxgen`
- Current runtime internals: primarily `mflux.*`

The project keeps some mflux vocabulary and compatibility entry points while the fork evolves. New docs and integrations should use `mlxgen` commands where the workflow is available and treat `mflux.*` internals as inherited implementation detail unless a specific model class or dedicated command currently requires them.

## Command Boundary

The public command surface separates setup from inference:

- `mlxgen download` is an explicit cache population command.
- `mlxgen prepare` is an explicit local model-folder creation command.
- `mlxgen generate` is the inference command and does not start downloads by default.
- `mflux-upscale-seedvr2` is the current dedicated inference command for SeedVR2 image
  super-resolution; it is documented separately until that workflow is unified under `mlxgen`.

This boundary is important for embedded workflow systems such as AbstractVision: a generation request should not unexpectedly start a large network transfer in the middle of a larger job.

## Model File Lifecycle

Source model files usually come from Hugging Face. They can be used in two ways:

1. Cache the source files with `mlxgen download` and run by alias or repository id.
2. Create a reusable local MLX-Gen model package with `mlxgen prepare --model ... --path ... --quantize ...`.

MLX-Gen model packages use the MLX/mflux saved-weight layout. They may contain MLX quantization tensors and generated Hugging Face model cards. They are intended for MLX-Gen and compatible mflux code, not direct Diffusers or Transformers loading.

Video support follows the same setup/runtime boundary. Wan2.2 loads local source files and writes MP4 output. Text-to-video starts from random video latents. TI2V-5B image-to-video VAE-encodes the first frame, masks first-frame timesteps, keeps the condition active during denoising, and reinserts the condition before decode. A14B uses Diffusers-compatible two-transformer boundary routing and, for the separate I2V model, concatenated image-condition latents.

## Runtime Failure Contract

Runtime model construction and generation use files that are already available locally. Missing required files raise `DownloadRequiredError`, which is also a `FileNotFoundError` for compatibility with existing callers.

The error includes actionable command fields such as `download_command` and, when applicable, `prepare_command`. CLI entry points print the human-readable error without a traceback for common missing-artifact cases.

## Quantization Policy

Quantization is model-specific. Qwen and ERNIE q4 paths use mixed q4/q8 policies because fully q4 checkpoints can lose coherent generative behavior for those model families. Bonsai Image uses Prism's pre-packed ternary 2-bit transformer path instead of MLX-Gen's q4/q8 `prepare` flow; it follows the same quality principle of keeping sensitive paths at higher precision, but ships as a pre-packed artifact. Other model families keep their existing quantization predicates unless their model behavior requires a dedicated policy.

See [Quantization](quantization.md) for the current rules.

## Python Integration Boundary

The current Python API still exposes many model classes through inherited mflux modules. MLX-Gen's near-term integration contract is:

- prepare files before constructing models;
- fail early when required artifacts are missing;
- keep model instances as stateful runtime objects;
- publish lightweight progress events through `mflux.callbacks.ProgressEvent` without exposing
  latents or model tensors;
- expose clearer public orchestration APIs over time without breaking existing compatibility paths unnecessarily.

See [Python Integration](python-integration.md) and [API And CLI](api.md) for current usage.
