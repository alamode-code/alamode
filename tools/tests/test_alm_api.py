"""Checks of the in-repo alm Python package that strainkit relies on:
several ALM objects in one process, and the row-vector lattice convention
(compared with the alm CLI on a non-orthogonal cell)."""

import os
import subprocess

import numpy as np
import pytest

from strainkit import almfit, fcsorder
from strainkit.units import BOHR_IN_ANGSTROM

TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALM_BIN = os.path.join(os.path.dirname(TOOLS_DIR), "_build", "alm", "alm")


def _hcp():
    from ase.build import bulk

    return bulk("Cu", "hcp", a=2.55, c=4.2) * (2, 2, 1)


def test_many_instances_in_one_process(ase_mod, alm_mod):
    """Regression: the 3rd-4th ALM instance used to segfault (uninitialized
    symmetry tolerance handed to spglib)."""
    cell = _hcp()
    for _ in range(6):
        pats = almfit.suggest_harmonic_patterns(cell)
        assert len(pats) >= 1


def test_lattice_rows_round_trip(ase_mod, alm_mod, tmp_path):
    """The lattice rows given to ALM come back as the a1..a3 vectors of the XML."""
    from ase.calculators.emt import EMT

    cell = _hcp()
    pats = almfit.suggest_harmonic_patterns(cell)
    u, f = [], []
    for s in almfit.displaced_structures(cell, pats, 0.01):
        s.calc = EMT()
        f.append(s.get_forces())
        u.append(s.get_positions() - cell.get_positions())
    out = str(tmp_path / "hcp.xml")
    almfit.fit_harmonic(cell, np.array(u), np.array(f), out, "xml")
    fcs = fcsorder.read_fcs_structure(out)
    assert np.allclose(fcs.lavec, cell.cell[:], atol=1e-8)
    assert np.allclose(
        fcs.xf - cell.get_scaled_positions(),
        np.round(fcs.xf - cell.get_scaled_positions()),
        atol=1e-8,
    )


@pytest.mark.skipif(
    not os.path.exists(ALM_BIN), reason="alm binary (_build/alm/alm) not found"
)
def test_python_matches_cli_on_skewed_cell(ase_mod, alm_mod, tmp_path):
    """Harmonic force constants from the Python API and from the alm CLI agree
    for a non-orthogonal (hcp) cell and the same displacement-force data."""
    from ase.calculators.emt import EMT
    from strainkit.dfset import write_dfset

    cell = _hcp()
    pats = almfit.suggest_harmonic_patterns(cell)
    u, f = [], []
    for s in almfit.displaced_structures(cell, pats, 0.01):
        s.calc = EMT()
        f.append(s.get_forces())
        u.append(s.get_positions() - cell.get_positions())
    u, f = np.array(u), np.array(f)
    py_h5 = str(tmp_path / "py.h5")
    almfit.fit_harmonic(cell, u, f, py_h5, "h5")
    # CLI run in the same directory
    work = tmp_path / "cli"
    work.mkdir()
    write_dfset(str(work / "DFSET"), u, f)
    lat = cell.cell[:] / BOHR_IN_ANGSTROM
    lines = [
        "&general",
        "  PREFIX = cli",
        "  MODE = optimize",
        "  NAT = %d" % len(cell),
        "  NKD = 1",
        "  KD = Cu",
        "/",
        "&interaction",
        "  NORDER = 1",
        "/",
        "&optimize",
        "  DFSET = DFSET",
        "/",
        "&cell",
        "  1.0",
    ]
    lines += ["  " + " ".join(f"{x:.12f}" for x in lat[i]) for i in range(3)]
    lines += ["/", "&cutoff", "  *-* None", "/", "&position"]
    for x in cell.get_scaled_positions():
        lines.append("  1 " + " ".join(f"{v:.12f}" for v in x))
    lines.append("/")
    (work / "cli.in").write_text("\n".join(lines) + "\n")
    with open(work / "cli.log", "w") as log:
        proc = subprocess.run(
            [ALM_BIN, "cli.in"], cwd=work, stdout=log, stderr=subprocess.STDOUT
        )
    assert proc.returncode == 0, (work / "cli.log").read_text()[-2000:]
    cli_h5 = str(work / "cli.h5")  # the 2.0dev CLI writes HDF5 by default
    assert os.path.exists(cli_h5)
    a, b = fcsorder.read_fcs_structure(py_h5), fcsorder.read_fcs_structure(cli_h5)
    assert np.allclose(a.lavec, b.lavec, atol=1e-8)
    assert np.allclose(a.lavec, cell.cell[:], atol=1e-8)
    assert np.array_equal(a.map_p2s, b.map_p2s)
    import h5py

    def fc2(path):
        with h5py.File(path, "r") as h:
            g = h["/ForceConstants/Order2"]
            n = len(g["force_constant_values"])
            key = np.concatenate(
                [
                    g["atom_indices_supercell"][:].reshape(n, -1),
                    g["coord_indices"][:].reshape(n, -1),
                    np.round(g["shift_vectors"][:].reshape(n, -1) * 1000).astype(int),
                ],
                axis=1,
            )
            vals = g["force_constant_values"][:]
        order = np.lexsort(key.T[::-1])
        return key[order], vals[order]

    ka, va = fc2(py_h5)
    kb, vb = fc2(cli_h5)
    assert ka.shape == kb.shape and np.array_equal(ka, kb)
    # the CLI reads the displacements/forces from a text DFSET (7 decimals in bohr)
    assert np.allclose(va, vb, rtol=1e-4, atol=1e-6)
