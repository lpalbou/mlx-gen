# Python Integration

MLX-Gen can be embedded directly in Python. The current runtime still exposes most model classes through the original `mflux` package layout, with `mlxgen` available as the package identity for new applications.

## Cache-Only Runtime

Python callers should download or prepare models before constructing model objects. Runtime constructors and generation calls do not download missing artifacts. See [Model Management](model-management.md) for the CLI setup commands.

```python
from mflux.models.common.download_policy import DownloadRequiredError
from mlxgen.models.z_image import ZImageTurbo

try:
    model = ZImageTurbo(quantize=8)
except DownloadRequiredError as exc:
    print(exc.download_command)
    raise
```

For user-facing applications, show the exception message or the `download_command`/`prepare_command` fields and stop the workflow.

## AbstractVision

MLX-Gen is intended to be the Apple Silicon / MLX dependency for AbstractVision while AbstractVision remains a cross-platform orchestration package.

That split means:

- AbstractVision owns provider-neutral request objects, artifact storage, capability checks, and AbstractCore plugin integration.
- MLX-Gen owns MLX model loading, model-family behavior, local quantized formats, and runtime compatibility fixes.
- MLX-Gen should fail early when required local artifacts are missing so AbstractVision can surface a clear remediation message instead of starting a network transfer.

The current integration path is still model-specific Python classes. A future higher-level facade may expose explicit prepared/loaded/warmed model states, but current docs only describe the APIs that exist now.

## Progress And Monitoring

Existing model callbacks remain available through the mflux runtime internals. A stable, backend-facing progress event API is not yet part of MLX-Gen. Applications that need progress should treat progress reporting as best-effort unless they are integrating a specific model class and callback path.

## Threading

MLX model instances should be treated as stateful runtime objects. Applications that multiplex user requests should serialize access to a loaded model instance unless they have tested a narrower concurrency model for that specific backend.
