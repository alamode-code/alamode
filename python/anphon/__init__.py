"""ALAMODE anphon Python interface (API stability: docs/source/anphon_python.rst).

Importing this module initializes MPI unless something (e.g. mpi4py) already did.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    from . import _anphon  # noqa: F401
except ImportError as exc:
    import ctypes
    import sys
    from pathlib import Path

    # The wheels do not ship libmpi. If it is not on the library path, load the one that the
    # "openmpi" wheel (pip install "alamode[openmpi]") or conda installs in <prefix>/lib.
    _libmpi = Path(
        sys.prefix,
        "lib",
        "libmpi.40.dylib" if sys.platform == "darwin" else "libmpi.so.40",
    )
    if _libmpi.name not in str(exc) or not _libmpi.exists():
        raise
    ctypes.CDLL(str(_libmpi), mode=ctypes.RTLD_GLOBAL)

from ._api import Phonon, Result, format_input, read_h5, run

__all__ = ["run", "Result", "Phonon", "format_input", "read_h5"]

try:
    __version__ = _dist_version("alamode")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "2.0.0"
