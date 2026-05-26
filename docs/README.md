# MLX-Gen Documentation

MLX-Gen is an MLX-native generative image and video runtime forked from mflux. It can be used directly from the command line or embedded in Python applications that need local Apple Silicon generation.

## Guides

- [Getting started](getting-started.md): install MLX-Gen, inspect the CLI, prepare model files, and run a first generation or edit.
- [Architecture](architecture.md): package shape, command boundaries, model-file lifecycle, runtime failure contract, and Python integration boundary.
- [API and CLI](api.md): public `mlxgen` command surface, generation router behavior, image/video examples, model-management commands, and Python integration boundary.
- [Model management](model-management.md): explicit download and prepare workflows, cache-only runtime behavior, and Depth Pro downloads.
- [Quantization](quantization.md): current q4/q8 compatibility by model family, including validation panels and the Qwen and ERNIE mixed q4/q8 policies.
- [Hugging Face publishing](huggingface-publishing.md): generated model cards, source license/access wording, default `AbstractFramework/<repo-name>` usage, upload flow, and optional collection membership.
- [Python integration](python-integration.md): current in-process API, AbstractVision integration notes, progress limitations, and error handling.
- [Release](release.md): GitHub Release and PyPI trusted publishing workflow.
- [FAQ](faq.md): common questions about `prepare`, downloads, package naming, and compatibility.
- [Troubleshooting](troubleshooting.md): common missing-artifact, cache, local-path, ERNIE Prompt Enhancer, small-resolution ERNIE, unsupported ERNIE edit inputs, and Wan image-input limits.

The top-level [README](../README.md) remains the starting point for installation, model families, and project relationship details.

## Project Documents

- [Acknowledgements](../ACKNOWLEDGEMENTS.md): upstream mflux credit, post-fork maintainership, and model/community acknowledgements.
- [Changelog](../CHANGELOG.md): release history and migration notes.
- [Contributing](../CONTRIBUTING.md): local development, checks, and pull-request expectations.
- [Security](../SECURITY.md): vulnerability reporting and model/token safety guidance.
- [Code of Conduct](../CODE_OF_CONDUCT.md): participation expectations.
