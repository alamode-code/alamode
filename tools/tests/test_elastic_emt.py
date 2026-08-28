import os
import sys

import numpy as np
import pytest

from strainkit import workflow_elastic as we
from strainkit.units import EV_PER_ANG3_TO_GPA
from strainkit.writers import read_C1_array_in, read_elastic_constants_in

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emt_helpers import relaxed_cu_fcc, run_emt  # noqa: E402

GPA = EV_PER_ANG3_TO_GPA


def _fd_constants(atoms, h=1e-4):
    from ase.calculators.emt import EMT

    def stress(u):
        a = atoms.copy()
        a.set_cell(atoms.cell[:] @ (np.eye(3) + u).T, scale_atoms=True)
        a.calc = EMT()
        return a.get_stress(voigt=False)

    u = np.zeros((3, 3))
    u[0, 0] = h
    d = (stress(u) - stress(-u)) / (2 * h) * GPA
    c11, c12 = d[0, 0], d[1, 1]
    u = np.zeros((3, 3))
    u[1, 2] = u[2, 1] = h / 2
    c44 = ((stress(u) - stress(-u)) / (2 * h) * GPA)[1, 2]
    return c11, c12, c44


@pytest.fixture(scope="module")
def cu_workdir(tmp_path_factory, ase_mod, spglib_mod):
    import ase.io

    root = tmp_path_factory.mktemp("elastic_cu")
    tmpl = root / "tmpl"
    tmpl.mkdir()
    atoms = relaxed_cu_fcc()
    ase.io.write(tmpl / "input.extxyz", atoms)
    work = root / "work"
    we.generate(
        "ase",
        str(tmpl),
        str(work),
        smag=0.005,
        nmag=2,
        dirset="minimal",
        log=lambda *a: None,
    )
    assert run_emt(str(work)) == 85
    return atoms, str(work)


def test_stress_fit_matches_finite_differences(cu_workdir):
    atoms, work = cu_workdir
    fit, summary = we.fit(work, "stress", log=lambda *a: None)
    c2 = np.array(summary["C2_voigt_GPa"])
    c11, c12, c44 = _fd_constants(atoms)
    assert (
        abs(c2[0, 0] - c11) < 1.0
        and abs(c2[0, 1] - c12) < 1.0
        and abs(c2[3, 3] - c44) < 1.0
    )
    # cubic symmetry after symmetrization, and small symmetrization change
    assert (
        np.allclose(c2[0, 0], c2[1, 1])
        and np.allclose(c2[3, 3], c2[5, 5])
        and abs(c2[0, 3]) < 1e-9
    )
    assert summary["symmetrization_max_change_eV_per_A3"]["C2"] * GPA < 0.1
    assert fit.rank == 83
    # files
    c2f, c3f = read_elastic_constants_in(
        os.path.join(work, "results", "elastic_constants.in")
    )
    assert c2f.shape == (9, 9) and c3f.shape == (9, 9, 9)
    s0 = read_C1_array_in(os.path.join(work, "results", "C1_array.in"))
    assert (
        np.abs(s0).max() * 13.605693122994 / summary["volume_dft_A3"] * GPA < 0.05
    )  # relaxed: sigma0 ~ 0
    assert os.path.exists(os.path.join(work, "results", "elastic_fit.json"))


def test_both_fit_and_show(cu_workdir):
    atoms, work = cu_workdir
    fit, summary = we.fit(work, "both", log=lambda *a: None)
    assert fit.rank == 84
    c2 = np.array(summary["C2_voigt_GPa"])
    c11, c12, c44 = _fd_constants(atoms)
    assert abs(c2[0, 0] - c11) < 1.0 and abs(c2[3, 3] - c44) < 1.0
    text = we.show(
        os.path.join(work, "results", "elastic_constants.in"),
        volume_A3=summary["volume_dft_A3"],
        c1_path=os.path.join(work, "results", "C1_array.in"),
    )
    assert "Second-order elastic constants" in text and f"{c2[0, 0]:.3f}" in text


def test_energy_fit_needs_full_set(cu_workdir):
    atoms, work = cu_workdir
    with pytest.raises(RuntimeError, match="rank-deficient"):
        we.fit(work, "energy", log=lambda *a: None)


def test_relaxed_geometry_is_rejected(cu_workdir, tmp_path):
    import ase.io

    atoms, work = cu_workdir
    out = os.path.join(work, "strain_003", "output.extxyz")
    backup = ase.io.read(out)
    moved = backup.copy()
    moved.set_cell(moved.cell[:] * 1.001, scale_atoms=True)
    moved.calc = backup.calc
    ase.io.write(out, moved)
    try:
        with pytest.raises(ValueError, match="geometry does not match"):
            we.fit(work, "stress", log=lambda *a: None)
    finally:
        ase.io.write(out, backup)
