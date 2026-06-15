# MLX-Gen Documentation

MLX-Gen is an MLX-native generative image and video runtime forked from mflux. It can be used directly from the command line or embedded in Python applications that need local Apple Silicon generation.

## Guides

- [Getting started](getting-started.md): install MLX-Gen, inspect the CLI, prepare model files, run a first generation or edit, upscale with SeedVR2, and use Wan video commands with duration, FPS, and resolution guidance.
- [Architecture](architecture.md): package shape, command boundaries, model-file lifecycle, runtime failure contract, and Python integration boundary.
- [API and CLI](api.md): public `mlxgen` command surface, generation router behavior, image-to-image modes and canvas policy, experimental generative reframe and backend-specific outpaint, SeedVR2 upscale sizing, negative prompts, Qwen edit variants, Wan video size rules, model-management commands, and Python integration boundary.
- [Image edit modes](image-edit-modes.md): plain-language guide to latent img2img, edit-reference, multi-reference, generative reframe, and outpaint, including what each mode is good at and what to expect from the output.
- [Wan video](wan-video.md): practical Wan2.2 T2V/I2V sizing, 5-second M5 Max comparison clips, recommended A14B lower-resolution settings, and included MP4/frame-strip assets.
- [Spaceship snow workflow](examples/spaceship-snow.md): reproducible model-backed T2I, I2I edit, multi-reference I2I, T2V A14B, and I2V A14B commands with included assets.
- [Image upscaling](upscaling.md): SeedVR2 command usage, published 3B/7B q8/q4 package handles, shortest-edge and scale-factor sizing, quality controls, and real 5x comparisons from a `133x113` source.
- [Image edit capabilities](edit-capabilities.md): current image-edit plus experimental generative reframe and outpaint contact sheets, exact model/package status, the new FLUX.2 Klein base source-model starship proof, and command logs for Qwen Image Edit, Qwen Image Edit 2509/2511, FLUX.2 Klein, and latent I2I rows. FIBO Edit is documented as unsupported through unified `mlxgen generate` in the current release.
- [Reframe and outpaint](reframe-outpaint.md): experimental `--reframe-padding` and `--outpaint-padding` workflows, supported models, the historical mixed June 8 profile, the current FLUX.2 Klein base source-model starship proof, and the validation profile ids for canvas expansion workflows.
- [LoRA](lora.md): experimental route capability fields, explicit adapter download, strict scale matching, model-card base-model compatibility, source/no-LoRA/with-LoRA validation, exact q8 proof rows for original Qwen Image Edit, Qwen 2509/2511, Qwen 2512, Z-Image, FLUX.2 Klein, ERNIE Image Turbo, and all current Wan q8 video routes, including direct `lightx2v/Wan2.2-Lightning` download, recommended `4`-step `lightx2v/Qwen-Image-2512-Lightning` and `lightx2v/Qwen-Image-Edit-2511-Lightning` examples, stable `repo:subdir/file.safetensors` and absolute local-file usage for Wan A14B, same-seed no-LoRA-versus-Lightning A/B sheets, the documented q8-vs-BF16 `720p` keyframe comparison, readable `41`-frame T2V/I2V progress matrices, the `240p`-versus-`480p` T2V sweep, and the remaining base-Qwen experimental limits plus the current Bonsai fail-closed boundary.
- [Model management](model-management.md): explicit download and prepare workflows, generation from local model files, local MLX-Gen package resolution, and Depth Pro downloads.
- [Quantization](quantization.md): current low-bit compatibility by model family, including the complete published package matrix, benchmark panels, Bonsai ternary 2-bit support, published SeedVR2 3B/7B q8/q4 packages, Qwen and ERNIE mixed q4/q8 policies, and Wan TI2V/A14B package status.
- [Hugging Face publishing](huggingface-publishing.md): generated model cards, source license/access wording, default `AbstractFramework/<repo-name>` usage, upload flow, and optional collection membership.
- [Python integration](python-integration.md): current in-process API, AbstractVision and AbstractCore integration notes, shared progress callbacks, and error handling.
- [Release](release.md): GitHub Release and PyPI trusted publishing workflow.
- [FAQ](faq.md): common questions about `prepare`, downloads, package naming, image-to-image mode selection and output sizing, SeedVR2 upscale sizing, Qwen edit variants, negative prompts, experimental generative reframe, experimental canvas outpaint, Wan resolutions, compatibility, and Wan image-to-video prompting.
- [Troubleshooting](troubleshooting.md): common missing-artifact, cache, local-path, image-to-image sizing, LoRA compatibility, ERNIE Prompt Enhancer, small-resolution ERNIE, unsupported ERNIE edit inputs, and Wan video quality-setting and prompting limits.

The top-level [README](../README.md) remains the starting point for installation, model families, and project relationship details.

## Project Documents

- [Acknowledgements](../ACKNOWLEDGEMENTS.md): upstream mflux credit, post-fork maintainership, and model/community acknowledgements.
- [Changelog](../CHANGELOG.md): release history and migration notes.
- [Contributing](../CONTRIBUTING.md): local development, checks, and pull-request expectations.
- [Architecture Decision Records](adr/README.md): durable validation and architecture policies.
- [Security](../SECURITY.md): vulnerability reporting and model/token safety guidance.
- [Code of Conduct](../CODE_OF_CONDUCT.md): participation expectations.
