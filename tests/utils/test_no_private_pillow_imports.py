import re
from pathlib import Path

import pytest

_PRIVATE_PILLOW_IMPORT = re.compile(
    r"^\s*(from\s+PIL\._\w+\s+import|import\s+PIL\._\w+|from\s+PIL\s+import\s+_\w+)", re.MULTILINE
)


@pytest.mark.fast
def test_the_package_imports_no_private_pillow_module():
    """`PIL._typing` does not import in Pillow 13.0.0, which broke saving every image and video."""
    source_root = Path(__file__).resolve().parents[2] / "src"
    offenders = [
        str(path.relative_to(source_root))
        for path in source_root.rglob("*.py")
        if _PRIVATE_PILLOW_IMPORT.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


@pytest.mark.fast
def test_load_image_accepts_paths_and_images(tmp_path):
    import PIL.Image

    from mflux.utils.image_util import ImageUtil

    image = PIL.Image.new("RGB", (4, 4), (10, 20, 30))
    image.save(tmp_path / "pixel.png")
    assert ImageUtil.load_image(tmp_path / "pixel.png").size == (4, 4)
    assert ImageUtil.load_image(str(tmp_path / "pixel.png")).size == (4, 4)
    assert ImageUtil.load_image(image).size == (4, 4)
