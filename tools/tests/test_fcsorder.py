import os

import numpy as np
import pytest

from strainkit import fcsorder


def test_lattice_relation_both_directions():
    a = np.array([[2.0, 0, 0], [-1.0, 1.7320508, 0], [0, 0, 3.3]])
    sup = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]]) @ a
    m, ratio = fcsorder.lattice_relation(sup, a)
    assert ratio == 2.0 and np.array_equal(m, [[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    m, ratio = fcsorder.lattice_relation(a, sup)
    assert ratio == 0.5
    theta = np.deg2rad(7.0)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="not integer combinations"):
        fcsorder.lattice_relation(a @ rot.T, a)
    with pytest.raises(ValueError):
        fcsorder.lattice_relation(1.01 * a, a)


def test_match_primitive_atoms_dft_supercell(ase_mod):
    """The DFT cell may be a supercell of the anphon cell (one image per atom)."""
    from ase.build import bulk
    hcp = bulk("Cu", "hcp", a=2.55, c=4.2)
    prim = fcsorder.AnphonPrimitive(hcp.cell[:], hcp.get_scaled_positions(), hcp.numbers,
                                    hcp.get_chemical_symbols(), None, "test")
    m = fcsorder.match_primitive_atoms(hcp * (2, 1, 1), prim)
    assert m.tolist() == [0, 1]
    m = fcsorder.match_primitive_atoms(hcp[[1, 0]], prim)
    assert m.tolist() == [1, 0]


def test_read_anphon_cell(tmp_path):
    p = tmp_path / "anphon.in"
    p.write_text("&general\n PREFIX = x\n/\n&cell\n  2.0\n  1.0 0.0 0.0\n  0.0 1.0 0.0 ! comment\n  0.0 0.0 1.5\n/\n")
    lav = fcsorder.read_anphon_cell(str(p))
    assert np.allclose(lav, np.diag([2.0, 2.0, 3.0]) * 0.52917721092)
