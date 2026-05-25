# Security

## Supported Versions

Security fixes are applied to the current MLX-Gen release line. Users should upgrade to the latest published `mlx-gen` package when a security or dependency fix is released.

## Reporting A Vulnerability

Report security issues privately through the GitHub repository's security advisory flow when available:

https://github.com/lpalbou/mlx-gen/security/advisories

If the advisory flow is unavailable, contact the repository owner before opening a public issue. Please include:

- affected version or commit;
- operating system and Python version;
- steps to reproduce;
- impact and any known workaround.

Do not include secrets, private model tokens, or proprietary model artifacts in reports.

## Model And Token Safety

MLX-Gen uses Hugging Face repositories and local model paths supplied by users. Treat model files, LoRAs, and tokens as untrusted inputs unless you control their source. Avoid sharing private Hugging Face tokens in command logs, screenshots, issue reports, or generated metadata.
