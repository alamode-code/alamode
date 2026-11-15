"""ALAMODE anphon Python interface (API stability: docs/source/anphon_python.rst).

Importing this module initializes MPI unless something (e.g. mpi4py) already did.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from ._api import Phonon, Result, format_input, read_h5, run

__all__ = ["run", "Result", "Phonon", "format_input", "read_h5"]

try:
    __version__ = _dist_version("alamode")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "2.0.1"
