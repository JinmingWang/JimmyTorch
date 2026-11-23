"""Mesh utilities, imported when their public functions are requested."""
from importlib import import_module as _import_module

from .._mesh_exports import MESH_MODULES as _modules

__all__ = list(_modules)


def __getattr__(name):
    if name not in _modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(f".{_modules[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
