# Acknowledgements

MLX-Gen is an independent project forked from [mflux](https://github.com/filipstrand/mflux).

Most credit for the current codebase goes to Filip Strand and the original mflux contributors. MLX-Gen keeps that attribution visible while the project evolves under the `mlx-gen` package name and the `mlxgen` command surface.

Post-fork MLX-Gen changes are maintained by Laurent-Philippe Albou / AbstractVision.

MLX-Gen depends on [MLX](https://github.com/ml-explore/mlx), Hugging Face Hub,
Diffusers/Transformers model formats, and the broader open-source Python and Apple Silicon
ecosystem.

The project routes or prepares model families from these upstream communities and model owners:

- Black Forest Labs for FLUX.1 and FLUX.2 Klein model families, including fill, depth, Redux,
  Kontext, ControlNet, and Klein routes where supported.
- Qwen / Alibaba Tongyi for Qwen Image, Qwen Image Edit, Qwen Image Edit 2509, and Qwen Image Edit
  2511.
- Wan-AI / Alibaba Tongyi for Wan2.2 TI2V-5B and Wan2.2 A14B T2V/I2V video models.
- Tongyi-MAI for Z-Image and Z-Image Turbo.
- Baidu for ERNIE Image Turbo.
- Bria for FIBO and related FIBO image-generation checkpoints. FIBO Edit is not exposed as a
  supported unified edit route in the current release.
- Prism ML for Bonsai Image ternary/binary MLX checkpoints.
- ByteDance/SeedVR2 for the official SeedVR2 image super-resolution model family.
- InstantX, JasperAI, CatVTON, and Apple Depth Pro for optional control, upscaling, fill, virtual
  try-on, and depth-related model components used by inherited mflux routes.

Optimized model variants published under the AbstractFramework Hugging Face organization remain
derivative MLX-Gen/mflux saved-weight layouts of their respective source repositories. Model
weights, licenses, access restrictions, and attribution requirements remain governed by the original
model owners and source repositories.
