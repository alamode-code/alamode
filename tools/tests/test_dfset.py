import numpy as np
import pytest

from strainkit import dfset
from strainkit.dftio import DftResult, check_geometry, check_same_species


def _result(atoms, forces, shift=None):
    xf = atoms.get_scaled_positions()
    if shift is not None:
        xf = xf + shift
    return DftResult("x", "ase", 1, None, forces, None, atoms.cell[:], xf, atoms.numbers)


def test_refold_and_offset(ase_mod):
    from ase.build import bulk
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    d = np.zeros((4, 3))
    d[0] = [0.002, 0, 0]
    # displacement crossing the periodic boundary: -0.001 in fractional -> +0.999 after wrapping
    xf = a.get_scaled_positions() + d
    xf[0] = xf[0] - 1.0
    r = DftResult("x", "ase", 1, None, np.ones((4, 3)), None, a.cell[:], xf, a.numbers)
    off = _result(a, 0.25 * np.ones((4, 3)))
    u, f = dfset.build_dfset([r], a, off)
    assert np.allclose(u[0][0], [0.002 * 3.6, 0, 0]) and np.allclose(u[0][1:], 0)
    assert np.allclose(f, 0.75)


def test_species_and_geometry_checks(ase_mod):
    from ase.build import bulk
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    b = a.copy()
    b.numbers[0] = 30
    with pytest.raises(ValueError):
        check_same_species(_result(b, np.zeros((4, 3))), a)
    shift = np.zeros((4, 3))
    shift[2] = [0, 0, 1e-3]
    with pytest.raises(ValueError):
        check_geometry(_result(a, np.zeros((4, 3)), shift), a)
    with pytest.warns(UserWarning):
        check_geometry(_result(a, np.zeros((4, 3)), shift), a, strict=False)
    check_geometry(_result(a, np.zeros((4, 3))), a)
