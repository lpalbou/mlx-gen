# MLX-Gen Documentation

MLX-Gen is an MLX-native generative image and video runtime forked from mflux. It can be used directly from the command line or embedded in Python applications that need local Apple Silicon generation.

## Guides

- [Getting started](getting-started.md): install MLX-Gen, inspect the CLI, prepare model files, run a first generation or edit, and use Wan video commands with duration, FPS, and resolution guidance.
- [Architecture](architecture.md): package shape, command boundaries, model-file lifecycle, runtime failure contract, and Python integration boundary.
- [API and CLI](api.md): public `mlxgen` command surface, generation router behavior, image-to-image modes and canvas policy, negative prompts, Qwen edit variants, Wan video size rules, model-management commands, and Python integration boundary.
- [Spaceship snow workflow](examples/spaceship-snow.md): reproducible model-backed T2I, I2I edit, multi-reference I2I, T2V A14B, and I2V A14B commands with included assets.
- [Image edit capabilities](edit-capabilities.md): current visual edit-validation contact sheets, exact model/package status, and command logs for Qwen Image Edit, Qwen EditPlus, FLUX.2 Klein, and latent I2I rows. FIBO Edit is documented as unsupported through unified `mlxgen generate` in the current release.
- [Model management](model-management.md): explicit download and prepare workflows, cache-only runtime behavior, matching local prepared-folder handle resolution, and Depth Pro downloads.
- [Quantization](quantization.md): current low-bit compatibility by model family, including the complete published package matrix, benchmark panels, Bonsai ternary 2-bit support, Qwen and ERNIE mixed q4/q8 policies, and Wan TI2V/A14B package status.
- [Hugging Face publishing](huggingface-publishing.md): generated model cards, source license/access wording, default `AbstractFramework/<repo-name>` usage, upload flow, and optional collection membership.
- [Python integration](python-integration.md): current in-process API, AbstractVision integration notes, shared progress callbacks, and error handling.
- [Release](release.md): GitHub Release and PyPI trusted publishing workflow.
- [FAQ](faq.md): common questions about `prepare`, downloads, package naming, image-to-image mode selection and output sizing, Qwen edit variants, negative prompts, outpaint/reframe status, Wan resolutions, compatibility, and Wan image-to-video prompting.
- [Troubleshooting](troubleshooting.md): common missing-artifact, cache, local-path, image-to-image sizing, ERNIE Prompt Enhancer, small-resolution ERNIE, unsupported ERNIE edit inputs, and Wan video quality-setting and prompting limits.

The top-level [README](../README.md) remains the starting point for installation, model families, and project relationship details.

## Project Documents

- [Acknowledgements](../ACKNOWLEDGEMENTS.md): upstream mflux credit, post-fork maintainership, and model/community acknowledgements.
- [Changelog](../CHANGELOG.md): release history and migration notes.
- [Contributing](../CONTRIBUTING.md): local development, checks, and pull-request expectations.
- [Architecture Decision Records](adr/README.md): durable validation and architecture policies.
- [Security](../SECURITY.md): vulnerability reporting and model/token safety guidance.
- [Code of Conduct](../CODE_OF_CONDUCT.md): participation expectations.
