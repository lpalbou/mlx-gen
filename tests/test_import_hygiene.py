import json
import subprocess
import sys

import pytest

# Dependency-creep gate (0088): `import mflux` must stay free of the heavy
# libraries below. Measured 2026-07-22 on M5 Max: the import dropped from
# ~1.34-1.62 s (cold-ish) / ~240 ms (warm) to ~60 ms warm once huggingface_hub
# (httpx+rich), PIL, and numpy left the module-scope import chain.
# numpy and mlx.core remain acceptable costs for modules that genuinely need
# them, but nothing on the plain `import mflux` chain does today.
FORBIDDEN_TOP_LEVEL_MODULES = (
    "torch",
    "transformers",
    "tokenizers",
    "matplotlib",
    "httpx",
    "huggingface_hub",
    "cv2",
    "av",
    "rich",
    # Achieved by 0088 (output_paths no longer routes through ImageUtil and
    # dimension_resolver defers PIL.Image): keep it locked in.
    "PIL",
)


@pytest.mark.fast
def test_import_mflux_stays_free_of_heavy_libraries():
    # Same interpreter as the test venv; a fresh subprocess gives a clean
    # sys.modules snapshot without pytest's own imports polluting it.
    result = subprocess.run(
        [sys.executable, "-c", "import mflux, sys, json; print(json.dumps(sorted(sys.modules)))"],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded_modules = json.loads(result.stdout)

    offenders = sorted(
        module
        for module in loaded_modules
        for forbidden in FORBIDDEN_TOP_LEVEL_MODULES
        if module == forbidden or module.startswith(f"{forbidden}.")
    )
    assert not offenders, (
        f"`import mflux` pulled forbidden heavy modules: {offenders}. "
        "Keep huggingface_hub/PIL/torch-class imports function-local (see backlog 0088)."
    )
