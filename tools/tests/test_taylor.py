import os

import numpy as np
import pytest

FCS = os.path.join(
    os.path.dirname(__file__),
    "../../example/BaTiO3/anharm_IFCs/4_optimize/reference/cBTO222.h5",
)


def _potential(supercell_matrix):
    from ase import Atoms
    from ase.build import make_supercell
    from ase.units import Bohr
    from fcsio.reader import ForceConstantParser
    from taylor import TaylorExpansionPotential

    parser = ForceConstantParser(FCS)
    lavec, xfrac, numbers = parser.primitive_cell  # Bohr, fractional, atomic numbers
    prim = Atoms(numbers=numbers, scaled_positions=xfrac, cell=lavec * Bohr, pbc=True)
    supercell = make_supercell(prim, supercell_matrix)
    potential = TaylorExpansionPotential(supercell, maxorder=4)
    potential.set_forceconstants(parser.read(maxorder=4), parser.primitive_cell)
    return potential


@pytest.mark.parametrize("n", [2, 3])
def test_numba_matches_numpy(ase_mod, spglib_mod, n):
    """Merged-term numba kernel == numpy loop, per order, in the fitting cell and a larger one;
    65 snapshots cross the 64-snapshot block boundary; the backend is toggled on one object."""
    pytest.importorskip("numba")
    pytest.importorskip("h5py")
    potential = _potential(np.eye(3, dtype=int) * n)
    rng = np.random.default_rng(0)
    disp = rng.normal(0.0, 0.1, (65, len(potential.supercell0), 3))
    potential.use_numba = True
    e_fast, f_fast = potential.compute(disp)
    assert potential._terms
    potential.use_numba = False
    e_slow, f_slow = potential.compute(disp)
    for key in e_slow:
        np.testing.assert_allclose(e_fast[key], e_slow[key], rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(f_fast[key], f_slow[key], rtol=1e-10, atol=1e-12)
