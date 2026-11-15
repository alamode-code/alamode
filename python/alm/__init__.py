"""ALAMODE 2.0 ALM Python interface (nanobind backend)."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from .alm import ALM, optimizer_control_data_types

__all__ = ["ALM", "Fcsxml", "optimizer_control_data_types"]

try:
    __version__ = _dist_version("alamode-python")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "2.0.2"


def __getattr__(name):
    # Lazy re-export: importing Fcsxml pulls in spglib, which is optional.
    if name == "Fcsxml":
        from .fcsxml import Fcsxml

        return Fcsxml
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
