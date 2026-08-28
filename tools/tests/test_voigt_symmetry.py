import numpy as np

from strainkit import elasticfit as ef
from strainkit import strain, symmetry


def _random_tensors(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=6), rng.normal(size=21), rng.normal(size=56)


def test_voigt_full_round_trip():
    s0, c2, c3 = _random_tensors()
    c4 = ef.voigt_to_full2(c2)
    assert np.allclose(ef.full2_to_voigt(c4), c2)
    assert np.allclose(c4, np.transpose(c4, (1, 0, 2, 3)))
    assert np.allclose(c4, np.transpose(c4, (2, 3, 0, 1)))
    c6 = ef.voigt_to_full3(c3)
    assert np.allclose(ef.full3_to_voigt(c6), c3)
    assert np.allclose(c6, np.transpose(c6, (2, 3, 0, 1, 4, 5)))
    assert np.allclose(c6, np.transpose(c6, (0, 1, 3, 2, 4, 5)))
    assert np.allclose(ef.from_9x9(ef.full2_to_9x9(c4)), c4)
    assert np.allclose(ef.from_9x9x9(ef.full3_to_9x9x9(c6)), c6)


def test_full_contraction_matches_design_rows():
    """Energy from anphon-style full 9-index sums == Voigt design row."""
    s0, c2, c3 = _random_tensors(3)
    u = strain.u_from_voigt([0.01, 0.0, -0.005, 0.004, 0.002, 0.003])
    eta = strain.green_lagrange(strain.deformation_gradient(u))
    c4, c6 = ef.voigt_to_full2(c2), ef.voigt_to_full3(c3)
    e_full = (
        np.einsum("ij,ij", strain.sym_from_voigt(s0), eta)
        + 0.5 * np.einsum("ijkl,ij,kl", c4, eta, eta)
        + np.einsum("ijklmn,ij,kl,mn", c6, eta, eta, eta) / 6.0
    )
    e_row = ef.design_row_energy(strain.voigt_from_sym(eta)) @ np.concatenate(
        [s0, c2, c3]
    )
    assert abs(e_full - e_row) < 1e-14
    # pure shear and C44-only checks
    c2s = np.zeros(21)
    c2s[ef.POS2[(3, 3)]] = 1.0  # C_yz,yz
    c4s = ef.voigt_to_full2(c2s)
    eta_s = strain.sym_from_voigt([0, 0, 0, 0.01, 0, 0])
    e_full = 0.5 * np.einsum("ijkl,ij,kl", c4s, eta_s, eta_s)
    assert abs(e_full - 0.5 * 4 * 0.01**2) < 1e-15  # 4 = w_yz^2
    e_row = ef.design_row_energy(strain.voigt_from_sym(eta_s)) @ np.concatenate(
        [np.zeros(6), c2s, np.zeros(56)]
    )
    assert abs(e_full - e_row) < 1e-15


def test_stress_is_energy_derivative():
    s0, c2, c3 = _random_tensors(5)
    x = np.concatenate([s0, c2, c3])
    eta6 = np.array([0.01, -0.004, 0.002, 0.003, -0.001, 0.005])
    S = ef.design_rows_stress(eta6) @ x
    h = 1e-6
    for a in range(6):
        ep, em = eta6.copy(), eta6.copy()
        ep[a] += h
        em[a] -= h
        dE = (ef.design_row_energy(ep) @ x - ef.design_row_energy(em) @ x) / (2 * h)
        assert abs(dE / strain.VOIGT_WEIGHT[a] - S[a]) < 1e-8


def test_intrinsic_symmetrization_idempotent():
    rng = np.random.default_rng(2)
    c4 = rng.normal(size=(3, 3, 3, 3))
    s = symmetry.enforce_intrinsic_symmetry2(c4)
    assert np.allclose(symmetry.enforce_intrinsic_symmetry2(s), s)
    c6 = rng.normal(size=(3,) * 6)
    s6 = symmetry.enforce_intrinsic_symmetry3(c6)
    assert np.allclose(symmetry.enforce_intrinsic_symmetry3(s6), s6)
    assert np.allclose(s6, np.transpose(s6, (4, 5, 0, 1, 2, 3)))


def test_cubic_symmetrization(ase_mod, spglib_mod):
    from ase.build import bulk

    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    rots = symmetry.cartesian_rotations(a)
    assert rots.shape == (48, 3, 3)
    rng = np.random.default_rng(7)
    c4 = symmetry.enforce_intrinsic_symmetry2(rng.normal(size=(3, 3, 3, 3)))
    cs = symmetry.symmetrize_rank4(c4, rots)
    m = ef.voigt66(cs)
    assert (
        np.allclose(m[0, 0], m[1, 1])
        and np.allclose(m[0, 1], m[0, 2])
        and np.allclose(m[3, 3], m[5, 5])
    )
    assert abs(m[0, 3]) < 1e-12 and abs(m[3, 4]) < 1e-12
    assert np.allclose(symmetry.symmetrize_rank4(cs, rots), cs)
    # triclinic cell: rotations orthogonal, identity only
    tri = ase_mod.Atoms(
        "CuAg",
        cell=[[3, 0.2, 0.1], [0.3, 3.5, 0.2], [0.1, 0.4, 4.0]],
        scaled_positions=[[0, 0, 0], [0.3, 0.6, 0.2]],
        pbc=True,
    )
    r = symmetry.cartesian_rotations(tri)
    assert len(r) == 1 and np.allclose(r[0], np.eye(3))


def test_hcp_symmetrization_nontrivial_rotations(ase_mod, spglib_mod):
    """Non-orthogonal cell: 24 point-group operations incl. 60-degree rotations."""
    from ase.build import bulk

    a = bulk("Cu", "hcp", a=2.55, c=4.2)
    rots = symmetry.cartesian_rotations(a)
    assert rots.shape == (24, 3, 3)
    assert any(
        np.abs(r - np.diag([1, 1, 1])).max() > 1e-8 and abs(np.trace(r) - 2.0) < 1e-8
        for r in rots
    )
    rng = np.random.default_rng(11)
    c4 = symmetry.enforce_intrinsic_symmetry2(rng.normal(size=(3, 3, 3, 3)))
    cs = symmetry.symmetrize_rank4(c4, rots)
    m = ef.voigt66(cs)
    # hexagonal: C11 = C22, C13 = C23, C44 = C55, C66 = (C11 - C12)/2, no couplings to shears
    assert (
        np.allclose(m[0, 0], m[1, 1])
        and np.allclose(m[0, 2], m[1, 2])
        and np.allclose(m[3, 3], m[4, 4])
    )
    assert np.allclose(m[5, 5], 0.5 * (m[0, 0] - m[0, 1]))
    assert abs(m[0, 3]) < 1e-12 and abs(m[0, 5]) < 1e-12
    assert np.allclose(symmetry.symmetrize_rank4(cs, rots), cs)
