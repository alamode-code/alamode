import os
import sys

import pytest

TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)


@pytest.fixture(scope="session")
def ase_mod():
    return pytest.importorskip("ase")


@pytest.fixture(scope="session")
def spglib_mod():
    return pytest.importorskip("spglib")


@pytest.fixture(scope="session")
def alm_mod():
    return pytest.importorskip("alm")
