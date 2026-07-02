"""ALAMODE 2.0dev ALM Python interface (nanobind backend)."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from .alm import ALM

__all__ = ["ALM", "Fcsxml"]

try:
    __version__ = _dist_version("alm")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "2.0.dev0"


def __getattr__(name):
    # Lazy re-export: importing Fcsxml pulls in spglib, which is optional.
    if name == "Fcsxml":
        from .fcsxml import Fcsxml
        return Fcsxml
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
