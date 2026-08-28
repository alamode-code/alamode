import numpy as np
import pytest

from strainkit import strain


def test_mode_tensors():
    assert np.allclose(strain.mode_tensor("xx"), np.diag([1.0, 0, 0]))
    t = strain.mode_tensor("yz")
    assert t[1, 2] == 0.5 and t[2, 1] == 0.5 and np.abs(t).sum() == 1.0
    with pytest.raises(ValueError):
        strain.mode_tensor("xz")


def test_voigt_round_trip():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(strain.voigt_from_sym(strain.sym_from_voigt(v)), v)
    u = strain.u_from_voigt(v)
    assert u[1, 2] == 2.0 and u[2, 0] == 2.5 and u[0, 1] == 3.0


def test_green_lagrange():
    u = strain.u_from_voigt([0.01, -0.02, 0.005, 0.004, 0.0, 0.002])
    F = strain.deformation_gradient(u)
    assert np.allclose(strain.green_lagrange(F), u + 0.5 * u @ u)
    rng = np.random.default_rng(1)
    Fr = np.eye(3) + 0.01 * rng.normal(size=(3, 3))
    assert np.allclose(strain.green_lagrange(Fr), 0.5 * (Fr.T @ Fr - np.eye(3)))


def test_deform_keeps_fractional(ase_mod):
    from ase.build import bulk

    a = bulk("Cu", "hcp", a=2.5, c=4.1)
    u = strain.u_from_voigt([0.01, 0, 0, 0.004, 0, 0])
    F = strain.deformation_gradient(u)
    b = strain.deform(a, F)
    assert np.allclose(b.cell[:], a.cell[:] @ F.T)
    assert np.allclose(b.get_scaled_positions(), a.get_scaled_positions())
    assert list(b.numbers) == list(a.numbers)


def test_strain_set_and_weights():
    pts = strain.ifc_strain_set(strain.default_modes(), central=True)
    assert len(pts) == 12
    assert np.allclose(strain.check_weight_sums(pts), 1.0)
    pts = strain.ifc_strain_set(strain.default_modes(), smag=0.002)
    assert all(p.smag == 0.002 for p in pts)
    with pytest.raises(ValueError):
        strain.ifc_strain_set(strain.default_modes(), smag=0.0)
    partial = strain.ifc_strain_set(strain.modes_from_names(["xx", "yy"]))
    with pytest.raises(ValueError):
        strain.check_weight_sums(partial, require_all=True)
    strain.check_weight_sums(partial, require_all=False)


def test_modes_json(tmp_path):
    import json

    p = tmp_path / "strain_modes.json"
    p.write_text(
        json.dumps(
            [
                {"id": 1, "mode": "xx", "weight": 1.0, "strain_mag": 0.003},
                {"id": 2, "mode": "xy"},
            ]
        )
    )
    modes = strain.load_modes_json(p)
    assert modes[0]["strain_mag"] == 0.003 and modes[1]["weight"] == 1.0
    p.write_text(json.dumps([{"mode": "xz"}]))
    with pytest.raises(ValueError):
        strain.load_modes_json(p)
