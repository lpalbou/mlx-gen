import importlib
import sys

_mflux = importlib.import_module("mflux")

for _name in getattr(_mflux, "__all__", ()):
    globals()[_name] = getattr(_mflux, _name)

__all__ = list(getattr(_mflux, "__all__", ()))

for _subpackage in ("callbacks", "cli", "models", "release", "utils"):
    _module = importlib.import_module(f"mflux.{_subpackage}")
    globals()[_subpackage] = _module
    sys.modules[f"{__name__}.{_subpackage}"] = _module
    if _subpackage not in __all__:
        __all__.append(_subpackage)


def __getattr__(name: str):
    return getattr(_mflux, name)
