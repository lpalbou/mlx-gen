# MLX-Gen Documentation

MLX-Gen is an MLX-native generative image and video runtime forked from mflux. It can be used directly from the command line or embedded in Python applications that need local Apple Silicon generation.

## Guides

- [Getting started](getting-started.md): install MLX-Gen, inspect the CLI, prepare model files, run a first generation or edit, restore or upscale with SeedVR2, and use Wan video commands with duration, FPS, and resolution guidance.
- [Architecture](architecture.md): system overview and model-file lifecycle diagrams, package shape, command boundaries, runtime failure contract, and Python integration boundary.
- [API and CLI](api.md): public `mlxgen` command surface, legacy-command migration boundary, generation router behavior, image-to-image modes and canvas policy, Qwen structured control with `--controlnet-image-path`, masked edit through `--mask-path` on Qwen edit and base rows, Z-Image, and FLUX.2 Klein (see [Masked editing](masked-editing.md)), generative reframe and backend-specific outpaint, SeedVR2 image/video restoration sizing and clip controls, negative prompts including the FLUX.2 exception, Qwen edit variants, Wan video size rules, model-management commands, and Python integration boundary.
- [Image edit modes](image-edit-modes.md): plain-language guide to latent img2img, edit-reference, masked edit/inpaint, base-Qwen control-inpaint, Qwen structured control, multi-reference, generative reframe, and outpaint, including what each mode is good at and what to expect from the output.
- [Masked editing](masked-editing.md): the canonical masked-edit page — request contract, per-model route matrix with proof grades, per-family behavior (Qwen edit and base routes with the tunable `--mask-strength` warm start, Z-Image Turbo, FLUX.2 Klein, video masks), and route selection advice.
- [Qwen route matrix](qwen-route-matrix.md): current MLX-Gen route truth for Qwen Image and Qwen Image Edit, mapping public `mlxgen` capability ids to the upstream Diffusers Qwen pipelines and the exact proof surfaces that already exist.
- [Qwen localized editing](qwen-localized-editing.md): plain-language explanation of Qwen masked edit, Qwen structured control, and shipped base-Qwen control-inpaint, including definitions of ControlNet and “sidecar”, the exact proof rows, and when each route is the right tool.
- [Wan video](wan-video.md): practical Wan2.2 T2V/I2V sizing, plain and masked prompt-guided A14B video-to-video, the natively ported Wan2.1-VACE-1.3B route (`wan-vace`: reference-image object injection and learned mask conditioning with `--vace-masked-region`), the measured motion-fidelity ladder (strength vs gesture preservation, with a motion-preserving restyle recipe), the fps-resampling and audio copy-through contract with a playable proof, broader A14B target size families, full example commands, and included MP4/frame-strip assets.
- [Spaceship snow workflow](examples/spaceship-snow.md): reproducible model-backed T2I, I2I edit, multi-reference I2I, T2V A14B, and I2V A14B commands with included assets.
- [Image upscaling](upscaling.md): SeedVR2 command usage for image and video restoration, published 3B/7B q8/q4 package handles, shortest-edge and scale-factor sizing, the conservative safe-video profile, the validated June 21 five-second Eiffel `1x` and `2x` 3B/7B proof bundles plus timings/memory data, and real 5x image comparisons from a `133x113` source.
- [Image edit capabilities](edit-capabilities.md): image-edit plus generative reframe and outpaint contact sheets, exact model/package status, the Qwen Image Edit 2511 q8 masked-edit proof, the exact base Qwen q8 structured-control and control-inpaint proofs, the exact Z-Image Turbo q8 native-inpaint proof, the FLUX.2 Klein base source-model starship proof, and command logs for Qwen Image Edit, Qwen Image Edit 2509/2511, FLUX.2 Klein, Qwen control routes, Z-Image native inpaint, and latent I2I rows. FIBO Edit is unsupported through unified `mlxgen generate`.
- [Reframe and outpaint](reframe-outpaint.md): `--reframe-padding` and `--outpaint-padding` workflows, supported models, the historical mixed June 8 profile, the current FLUX.2 Klein base source-model starship proof, and the validation profile ids for canvas expansion workflows.
- [LoRA](lora.md): route-specific capability fields, explicit adapter download, strict scale matching, model-card base-model compatibility, source/no-LoRA/with-LoRA validation, exact public proof rows for the current Qwen, Z-Image, FLUX.2, ERNIE, and Wan routes, and the current guidance that MLX-Gen q8 packages are the validated Lightning target rather than arbitrary external FP8 checkpoints.
- [Model management](model-management.md): explicit download and prepare workflows, generation from local model files, local MLX-Gen package resolution, and Depth Pro downloads.
- [Model recommendations](recommendations.md): conservative starting picks for `18 GB`, `24 GB`, `32 GB`, `64 GB`, and `128+ GB` Macs, using published MLX-Gen memory measurements rather than package size alone.
- [Quantization](quantization.md): current low-bit compatibility by model family, including the complete published package matrix, benchmark panels, Bonsai ternary 2-bit support, published SeedVR2 3B/7B q8/q4 packages, Qwen and ERNIE mixed q4/q8 policies, Wan TI2V/A14B package status, and the distinction between MLX-Gen q8 packages and third-party FP8 checkpoint guidance.
- [Hugging Face publishing](huggingface-publishing.md): generated model cards, source license/access wording, default `AbstractFramework/<repo-name>` usage, upload flow, and optional collection membership.
- [Python integration](python-integration.md): route-resolved runtime planning/loading, serial multi-output reuse for unified `mlxgen generate` families, overwrite/collision behavior, SeedVR2's direct-model boundary, and shared progress callbacks.
- [Release](release.md): GitHub Release and PyPI trusted publishing workflow.
- [FAQ](faq.md): common questions about `prepare`, downloads, package naming, image-to-image mode selection and output sizing, SeedVR2 image/video restoration sizing and default audio-preservation behavior, Qwen edit variants, negative prompts, reframe/outpaint, Wan resolutions, compatibility, and Wan image-to-video prompting.
- [Troubleshooting](troubleshooting.md): common missing-artifact, cache, local-path, image-to-image sizing, legacy `mflux-generate-*` migration, LoRA compatibility, ERNIE Prompt Enhancer, small-resolution ERNIE, unsupported ERNIE edit inputs, and Wan video quality-setting and prompting limits.

The top-level [README](../README.md) remains the starting point for installation, model families, and project relationship details.

## Project Documents

- [Acknowledgements](../ACKNOWLEDGEMENTS.md): upstream mflux credit, post-fork maintainership, and model/community acknowledgements.
- [Changelog](../CHANGELOG.md): release history and migration notes.
- [Contributing](../CONTRIBUTING.md): local development, checks, and pull-request expectations.
- [Architecture Decision Records](adr/README.md): durable validation and architecture policies.
- [Security](../SECURITY.md): vulnerability reporting and model/token safety guidance.
- [Code of Conduct](../CODE_OF_CONDUCT.md): participation expectations.
