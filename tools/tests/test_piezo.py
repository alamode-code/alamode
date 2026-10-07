"""piezo.py: the Berry (e0, B), DFPT (Lambda) and Berry-Lambda routes and the
sibling-preserving /Piezoelectric writes."""

import json
import os
import shutil

import numpy as np
import pytest

from strainkit import strainfile as sf
from strainkit import workflow_piezo as wp
from strainkit.writers import ReferenceCell

h5py = pytest.importorskip("h5py")
pytest.importorskip("ase")
pytest.importorskip("spglib")
QUIET = lambda *a: None  # noqa: E731
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def _refdir(
    tmp_path,
    poscar_lines,
    incar="ENCUT = 600\nLEPSILON = .TRUE. ; ISYM = 2\nKPAR = 4\n",
):
    d = tmp_path / "refcalc"
    d.mkdir()
    (d / "POSCAR").write_text("\n".join(poscar_lines) + "\n")
    (d / "INCAR").write_text(incar)
    (d / "KPOINTS").write_text("Auto\n0\nGamma\n4 4 4\n")
    (d / "POTCAR").write_text("placeholder\n")
    (d / "OUTCAR").write_text("reference output, must not be copied\n")
    return str(d)


def _lattice(path):
    import ase.io

    return np.asarray(ase.io.read(path, format="vasp").cell[:])


def _berry_outdir(tmp_path):
    """generate the berry grid from the fixture's reference POSCAR and fill in the OUTCARs."""
    fx = _load("zno_berry_grid.json")
    ref = _refdir(tmp_path, fx["poscar_ref"])
    out = str(tmp_path / "berry")
    man = wp.generate("berry", ref, out, h=fx["h"], log=QUIET)
    assert len(man["entries"]) == 73
    for e in man["entries"]:
        pt = fx["points"][e["dir"]]
        want = np.array([[float(x) for x in r.split()] for r in pt["lattice"]])
        assert (
            np.abs(_lattice(os.path.join(out, e["dir"], "POSCAR")) - want).max() < 1e-10
        )
        with open(os.path.join(out, e["dir"], "OUTCAR"), "w") as f:
            f.write(
                "\n".join(
                    fx["outcar_header"]
                    + pt["outcar_geometry"]
                    + [pt["p_ion"], pt["p_elc"]]
                )
                + "\n"
            )
    return out, man


def _zno_cell():
    fx = _load("zno_berry_grid.json")
    lines = fx["poscar_ref"]
    lat = np.array([[float(x) for x in r.split()] for r in lines[2:5]])
    xf = np.array([[float(x) for x in r.split()[:3]] for r in lines[8:12]])
    return ReferenceCell(lat, ["Zn", "Zn", "O", "O"], xf)


def _container(tmp_path, cell=None, name="zno.h5"):
    p = str(tmp_path / name)
    with sf.update(p, cell or _zno_cell()) as f:
        sf.write_elastic(f, stress_gpa=np.eye(3))
    return p


# ------------------------------------------------------------- generate
def test_generate_incar_and_cells(tmp_path):
    fx = _load("zno_berry_grid.json")
    ref = _refdir(tmp_path, fx["poscar_ref"])
    out = str(tmp_path / "g")
    wp.generate("berry", ref, out, log=QUIET)
    incar = open(os.path.join(out, "J4K5_pm", "INCAR")).read()
    assert (
        "LEPSILON" not in incar and "LCALCPOL = .TRUE." in incar and "ISYM = 0" in incar
    )
    assert "ISYM = 2" not in incar and "ENCUT = 600" in incar
    assert not os.path.exists(os.path.join(out, "ref", "OUTCAR"))
    assert os.path.exists(os.path.join(out, "run_all.sh"))
    man = json.load(open(os.path.join(out, wp.MANIFEST)))
    e = next(x for x in man["entries"] if x["dir"] == "J4K5_pm")
    assert np.allclose(e["u"], [[0, 0, -0.01], [0, 0, 0.01], [-0.01, 0.01, 0]])
    with pytest.raises(FileExistsError):
        wp.generate("berry", ref, out, log=QUIET)
    d = str(tmp_path / "d")
    man = wp.generate("dfpt", ref, d, h=0.005, log=QUIET)
    assert len(man["entries"]) == 12
    incar = open(os.path.join(d, "J1_p", "INCAR")).read()
    assert "LEPSILON = .TRUE." in incar and "LCALCPOL" not in incar
    lam = str(tmp_path / "l")
    man = wp.generate(
        "berry-lambda", ref, lam, atom=1, direction="z", strain="xz", log=QUIET
    )
    assert [e["dir"] for e in man["entries"]] == ["DpHp", "DpHm", "DmHp", "DmHm"]
    with pytest.raises(ValueError, match="--atom"):
        wp.generate(
            "berry-lambda",
            ref,
            str(tmp_path / "x"),
            atom=9,
            direction="z",
            strain="xz",
            log=QUIET,
        )


def test_edit_incar():
    t = wp.edit_incar(
        "A = 1 ; LEPSILON = .TRUE. # c\nisym=2\n# only a comment\n",
        {"ISYM": "0"},
        ("LEPSILON",),
    )
    assert t.splitlines()[:2] == ["A = 1 # c", "# only a comment"]
    assert t.splitlines()[-1] == "ISYM = 0"


# ---------------------------------------------------------------- berry
def test_berry_reproduces_zno_B_and_e0(tmp_path):
    out, _ = _berry_outdir(tmp_path)
    res = wp.collect(out, log=QUIET)
    ev, bv, diag = res["e0_voigt"], res["B_voigt"], res["diagnostics"]
    # note, "ZnO: computed coefficients" (12^3 k-points): Voigt 1..6 = xx yy zz yz xz xy
    assert ev[2, 0] == pytest.approx(0.346, abs=2e-3)
    assert ev[2, 2] == pytest.approx(-0.704, abs=2e-3)
    assert ev[0, 4] == pytest.approx(0.369, abs=2e-3)
    table = {
        (3, 1, 1): 0.83,
        (3, 1, 2): -0.25,
        (3, 1, 3): 0.22,
        (3, 3, 3): -2.01,
        (3, 4, 4): -0.32,
        (1, 1, 5): 0.01,
        (1, 2, 5): -0.23,
        (1, 3, 5): 0.86,
    }
    for (i, J, K), v in table.items():
        assert bv[i - 1, J - 1, K - 1] == pytest.approx(v, abs=0.01), (i, J, K)
    # 6mm relations of the symmetrized tensor: B_366 = (B_311 - B_312)/2, B_322 = B_311
    assert bv[2, 5, 5] == pytest.approx(0.5 * (bv[2, 0, 0] - bv[2, 0, 1]), abs=1e-10)
    assert bv[2, 1, 1] == pytest.approx(bv[2, 0, 0], abs=1e-10)
    # the raw differences are those of the note's analysis (berry_B.npz): B_311 0.845, B_322 0.812
    raw = res["B_raw_voigt"]
    assert raw[2, 0, 0] == pytest.approx(0.845, abs=1e-3) and raw[
        2, 1, 1
    ] == pytest.approx(0.812, abs=1e-3)
    assert res["e0_raw_voigt"][2, 1] == pytest.approx(0.347, abs=1e-3)
    assert diag["pair_exchange_violation"] == 0.0
    assert diag["branch_step_max"] < 0.02 and diag["branch_loop_max"] < 0.05
    assert diag["point_group_order"] == 12 and diag["B_symmetry_noise"] < 0.05
    B = res["B"]
    assert (
        B[2, 0, 0, 0, 0] == bv[2, 0, 0]
        and B[0, 2, 0, 0, 0] == B[0, 0, 2, 0, 0] == bv[0, 4, 0]
    )
    raw = wp.collect(out, symmetrize=False, log=QUIET)
    assert np.array_equal(raw["B_voigt"], res["B_raw_voigt"])


def test_berry_writes_into_container_keeping_siblings(tmp_path):
    out, _ = _berry_outdir(tmp_path)
    p = _container(tmp_path)
    lam = np.zeros((4, 3, 3, 3, 3))
    with sf.update(p, _zno_cell()) as f:
        sf.write_piezo_datasets(f, {sf.BORN_DERIV: lam})
    wp.collect(out, strain_file=p, log=QUIET)
    with h5py.File(p, "r") as f:
        assert set(f["Piezoelectric"]) == {
            "clamped_ion",
            "second_order",
            "born_charge_strain_derivative",
        }
        b, attrs = sf.read_piezo_dataset(f, sf.SECOND_ORDER)
        assert attrs["convention"] == "proper, clamped-ion, per linear strain u"
        assert "Berry" in attrs["method"] and attrs["source"].endswith("berry")
        e, _ = sf.read_piezo(f)
        assert "Elastic" in f
    with pytest.raises(ValueError, match="--overwrite"):
        wp.collect(out, strain_file=p, log=QUIET)
    # --no-clamped-ion keeps a DFPT e0
    e_other = e * 1.1
    with sf.update(p, _zno_cell()) as f:
        sf.write_piezo(f, e_other)
    wp.collect(out, strain_file=p, overwrite=True, skip_clamped_ion=True, log=QUIET)
    with h5py.File(p, "r") as f:
        assert np.array_equal(sf.read_piezo(f)[0], e_other)
        assert np.array_equal(sf.read_piezo_dataset(f, sf.SECOND_ORDER)[0], b)


def test_berry_branch_jump_is_unwrapped_and_bad_loop_refused(tmp_path):
    out, man = _berry_outdir(tmp_path)
    ref = wp.collect(out, log=QUIET)["B_voigt"]
    # one point shifted by a full quantum (lattice vector c): unwrapped
    A = _lattice(os.path.join(out, "J3_p", "POSCAR"))
    path = os.path.join(out, "J3_p", "OUTCAR")
    lines = open(path).read().splitlines()
    pe = np.array(lines[-1].split("(")[1].split(")")[0].split(), float) + A[2]
    lines[-1] = (
        "Total electronic dipole moment: p[elc]=( %.10f %.10f %.10f ) |e| Angst"
        % tuple(pe)
    )
    open(path, "w").write("\n".join(lines) + "\n")
    assert np.abs(wp.collect(out, log=QUIET)["B_voigt"] - ref).max() < 1e-3
    # a jump of 0.4 quanta: refused
    pe = pe - 0.6 * A[2]
    lines[-1] = (
        "Total electronic dipole moment: p[elc]=( %.10f %.10f %.10f ) |e| Angst"
        % tuple(pe)
    )
    open(path, "w").write("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="quantum"):
        wp.collect(out, log=QUIET)


def test_berry_malformed_outputs(tmp_path):
    out, _ = _berry_outdir(tmp_path)
    path = os.path.join(out, "J2_m", "OUTCAR")
    fx = _load("zno_berry_grid.json")
    head = fx["outcar_header"] + fx["points"]["J2_m"]["outcar_geometry"]
    open(path, "w").write(" vasp.6.5.1\n")
    with pytest.raises(ValueError, match="no structure lines"):
        wp.collect(out, log=QUIET)
    open(path, "w").write("\n".join(head) + "\n")
    with pytest.raises(ValueError, match="LCALCPOL"):
        wp.collect(out, log=QUIET)
    open(path, "w").write(
        "\n".join(head)
        + "\nTotal electronic dipole moment: p[elc]=( 0.0 nan 1.0 ) |e| Angst\np[ion]=( 0 0 0 )\n"
    )
    with pytest.raises(ValueError, match="malformed"):
        wp.collect(out, log=QUIET)
    shutil.copy(
        os.path.join(out, "J2_p", "POSCAR"), os.path.join(out, "J2_m", "POSCAR")
    )
    with pytest.raises(ValueError, match="F - I"):
        wp.collect(out, log=QUIET)
    with pytest.raises(ValueError, match="generated for"):
        wp.collect(out, route="dfpt", log=QUIET)


# ----------------------------------------------------------------- dfpt
def _dfpt_outcar(fx, name, born=None):
    c = fx["cells"][name]
    return (
        "\n".join(fx["outcar_header"] + c["outcar_geometry"] + (born or c["born"]))
        + "\n"
    )


def _dfpt_outdir(tmp_path):
    fx = _load("zno_dfpt_born.json")
    ref = _refdir(tmp_path, fx["cells"]["ref"]["poscar"])
    out = str(tmp_path / "dfpt")
    man = wp.generate("dfpt", ref, out, h=fx["h"], log=QUIET)
    for e in man["entries"]:
        c = fx["cells"][e["dir"]]
        want = np.array([[float(x) for x in r.split()] for r in c["poscar"][2:5]])
        assert (
            np.abs(_lattice(os.path.join(out, e["dir"], "POSCAR")) - want).max() < 1e-10
        )
        with open(os.path.join(out, e["dir"], "OUTCAR"), "w") as f:
            f.write(_dfpt_outcar(fx, e["dir"]))
    born_ref = str(tmp_path / "OUTCAR_ref")
    with open(born_ref, "w") as f:
        f.write(_dfpt_outcar(fx, "ref"))
    return out, born_ref, fx


def _born(lines, natom=4):
    z = np.zeros((natom, 3, 3))
    for k in range(natom):
        for r in range(3):
            z[k, r] = [float(x) for x in lines[3 + 4 * k + r].split()[1:]]
    return z


def test_dfpt_lambda_formula_and_asr(tmp_path):
    out, born_ref, fx = _dfpt_outdir(tmp_path)
    res = wp.collect(out, born_ref=born_ref, symmetrize=False, log=QUIET)
    lam, diag = res["Lambda"], res["diagnostics"]
    h = fx["h"]
    z = {k: _born(v["born"]) for k, v in fx["cells"].items()}
    # rows of the VASP block = polarization i, columns = displacement b
    # diagonal strain zz: Lambda_k,zz,zz = dZ_zz/du_zz - Z0_zz
    raw = (z["J3_p"] - z["J3_m"]) / (2 * h)
    expect = raw[:, 2, 2] - z["ref"][:, 2, 2]
    asr = np.mean(expect)
    assert np.allclose(lam[:, 2, 2, 2, 2], expect - asr, atol=1e-10)
    # shear xz (u_xz = u_zx = h): dsym = dZ/(4h); reduction -1/2 (d_ix Z_zb + d_iz Z_xb)
    raw = (z["J5_p"] - z["J5_m"]) / (4 * h)
    for i, b in ((0, 0), (2, 2), (0, 2), (2, 0), (1, 1)):
        red = 0.5 * ((i == 0) * z["ref"][:, 2, b] + (i == 2) * z["ref"][:, 0, b])
        e = raw[:, i, b] - red
        assert np.allclose(lam[:, i, b, 0, 2], e - e.mean(), atol=1e-10)
        assert np.array_equal(lam[:, i, b, 0, 2], lam[:, i, b, 2, 0])
    assert np.abs(lam.sum(axis=0)).max() < 1e-12
    assert 0 < diag["asr_residual_removed"] < 1e-2
    # the note: d Zt*_zz(Zn)/d s_K = (0.37, 0.37, -1.10) for K = xx, yy, zz (from the +-0.01 cells)
    assert lam[0, 2, 2, 0, 0] == pytest.approx(0.37, abs=0.02)
    assert lam[0, 2, 2, 2, 2] == pytest.approx(-1.10, abs=0.02)
    # without --born-ref: Z0 = mean of the strained cells (O(h^2))
    lam2 = wp.collect(out, symmetrize=False, log=QUIET)["Lambda"]
    assert np.abs(lam2 - lam).max() < 1e-3


def _ops(cell):
    import ase

    from strainkit.symmetry import space_group_operations

    return space_group_operations(
        ase.Atoms(cell.elements, cell=cell.lavec, scaled_positions=cell.xf, pbc=True)
    )


def _rot4(q, t):
    return np.einsum("ia,jb,mc,nd,kabcd->kijmn", q, q, q, q, t)


def test_symmetrize_atomic_lambda():
    from strainkit.symmetry import symmetrize_atomic

    cell = _zno_cell()
    rots, perms, images = _ops(cell)
    assert len(rots) == 12 and list(images) == [0, 1, 2, 3]  # P6_3mc
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(4, 3, 3, 3, 3))
    raw = 0.5 * (raw + raw.transpose(0, 1, 2, 4, 3))
    raw -= raw.mean(axis=0)
    sym = symmetrize_atomic(raw, rots, perms)
    # idempotent, invariant under every operation, mn symmetry and ASR kept
    assert np.allclose(symmetrize_atomic(sym, rots, perms), sym, atol=1e-13)
    for r, p in zip(rots, perms):
        moved = np.empty_like(sym)
        moved[p] = _rot4(r, sym)
        assert np.allclose(moved, sym, atol=1e-13)
    assert np.allclose(sym, sym.transpose(0, 1, 2, 4, 3))
    assert np.abs(sym.sum(axis=0)).max() < 1e-13
    # 6mm: Lambda_xx,zz = Lambda_yy,zz, Lambda_xy,zz = 0
    assert np.allclose(sym[:, 0, 0, 2, 2], sym[:, 1, 1, 2, 2])
    assert np.allclose(sym[:, 0, 1, 2, 2], 0.0, atol=1e-13)
    # the non-symmetric part is removed exactly
    noise = 0.1 * rng.normal(size=raw.shape)
    noise -= symmetrize_atomic(noise, rots, perms)
    assert np.allclose(symmetrize_atomic(sym + noise, rots, perms), sym, atol=1e-13)
    # equivariance: a rotated frame and permuted atoms give the rotated, permuted result
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    perm = [3, 1, 0, 2]
    rcell = ReferenceCell(
        cell.lavec @ q.T, [cell.elements[i] for i in perm], cell.xf[perm]
    )
    got = symmetrize_atomic(_rot4(q, raw)[perm], *_ops(rcell))
    assert np.allclose(got, _rot4(q, sym)[perm], atol=1e-12)


def test_symmetrize_anisotropic_supercell_reference():
    """A 2x1x1 cubic BaTiO3 cell (tetragonal lattice) gets the full Pm-3m group:
    the result equals the primitive-cell symmetrization of the image average."""
    from strainkit.symmetry import cartesian_rotations, symmetrize_atomic

    a = 4.0
    xf = np.array(
        [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    )
    el = ["Ba", "Ti", "O", "O", "O"]
    prim = ReferenceCell(np.eye(3) * a, el, xf)
    sup = ReferenceCell(
        np.diag([2 * a, a, a]),
        el * 2,
        np.vstack([xf * [0.5, 1, 1], xf * [0.5, 1, 1] + [0.5, 0, 0]]),
    )
    import ase

    rots = cartesian_rotations(
        ase.Atoms(sup.elements, cell=sup.lavec, scaled_positions=sup.xf, pbc=True)
    )
    assert len(rots) == 48
    ops_p, ops_s = _ops(prim), _ops(sup)
    assert len(ops_p[0]) == 48 and len(ops_s[0]) == 48
    assert list(ops_s[2]) == [0, 1, 2, 3, 4] * 2
    rng = np.random.default_rng(5)
    raw = rng.normal(size=(10, 3, 3, 3, 3))
    want = symmetrize_atomic(0.5 * (raw[:5] + raw[5:]), *ops_p)
    got = symmetrize_atomic(raw, *ops_s)
    assert np.allclose(got, np.concatenate([want, want]), atol=1e-12)
    # Pm-3m: Ti Lambda_xx,xx == Lambda_yy,yy == Lambda_zz,zz
    assert np.allclose(
        got[1, [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], got[1, 0, 0, 0, 0]
    )


def test_dfpt_zero_lambda(tmp_path, monkeypatch):
    """An exactly zero Lambda (e.g. a one-atom cell after the ASR) is no error."""
    out, born_ref, _ = _dfpt_outdir(tmp_path)
    monkeypatch.setattr(
        wp, "born_lambda", lambda zp, zm, h, z0: np.zeros((len(z0), 3, 3, 3, 3))
    )
    for sym in (True, False):
        res = wp.collect(out, born_ref=born_ref, symmetrize=sym, log=QUIET)
        assert not res["Lambda"].any()
        assert res["diagnostics"]["Lambda_symmetry_noise_relative"] == 0.0


def test_dfpt_lambda_is_space_group_symmetric(tmp_path):
    from strainkit.symmetry import symmetrize_atomic

    out, born_ref, _ = _dfpt_outdir(tmp_path)
    logs = []
    res = wp.collect(out, born_ref=born_ref, log=logs.append)
    raw = wp.collect(out, born_ref=born_ref, symmetrize=False, log=QUIET)["Lambda"]
    lam, diag = res["Lambda"], res["diagnostics"]
    ops = _ops(_zno_cell())
    assert np.allclose(symmetrize_atomic(lam, *ops), lam, atol=1e-12)
    assert np.allclose(lam, symmetrize_atomic(raw, *ops), atol=1e-14)
    assert diag["space_group_order"] == 12 and diag["symmetrized"]
    assert diag["Lambda_symmetry_noise"] == pytest.approx(np.abs(lam - raw).max())
    assert 0 < diag["Lambda_symmetry_noise_relative"] < 0.05
    assert any("space-group (12 operations) noise of Lambda" in x for x in logs)
    assert np.abs(lam.sum(axis=0)).max() < 1e-12  # ASR kept
    assert np.allclose(lam, lam.transpose(0, 1, 2, 4, 3))


def test_dfpt_maps_onto_permuted_and_nested_reference(tmp_path):
    out, born_ref, _ = _dfpt_outdir(tmp_path)
    lam = wp.collect(out, born_ref=born_ref, log=QUIET)["Lambda"]
    cell = _zno_cell()
    perm = [2, 0, 3, 1]
    permuted = ReferenceCell(
        cell.lavec, [cell.elements[i] for i in perm], cell.xf[perm]
    )
    p = _container(tmp_path, permuted, "perm.h5")
    wp.collect(out, strain_file=p, born_ref=born_ref, log=QUIET)
    with h5py.File(p, "r") as f:
        got, attrs = sf.read_piezo_dataset(f, sf.BORN_DERIV)
        assert set(f["Piezoelectric"]) == {
            "born_charge_strain_derivative"
        }  # Lambda-only group
        assert (
            attrs["unit"] == "e"
            and attrs["convention"] == "reduced (F^-1 Z*), per linear strain u"
        )
        assert attrs["asr_residual_removed"] > 0
    assert np.allclose(got, lam[perm])
    # the c-doubled reference (8 atoms): copies
    lav = cell.lavec.copy()
    lav[2] *= 2
    xf = np.vstack([cell.xf * [1, 1, 0.5], cell.xf * [1, 1, 0.5] + [0, 0, 0.5]])
    big = ReferenceCell(lav, list(cell.elements) * 2, xf)
    p = _container(tmp_path, big, "big.h5")
    wp.collect(out, strain_file=p, born_ref=born_ref, log=QUIET)
    with h5py.File(p, "r") as f:
        got = sf.read_piezo_dataset(f, sf.BORN_DERIV)[0]
    assert np.allclose(got, np.concatenate([lam, lam]))
    from strainkit.symmetry import symmetrize_atomic

    assert np.allclose(symmetrize_atomic(got, *_ops(big)), got, atol=1e-12)
    # the reverse: calculation cell = supercell of the reference -> average of the images
    images = wp.map_to_reference(cell, big)
    assert images == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_dfpt_rejects_stale_or_swapped_outcar(tmp_path):
    out, born_ref, fx = _dfpt_outdir(tmp_path)
    # the reference OUTCAR copied into every strained directory: rejected
    for e in json.load(open(os.path.join(out, wp.MANIFEST)))["entries"]:
        shutil.copy(born_ref, os.path.join(out, e["dir"], "OUTCAR"))
    with pytest.raises(ValueError, match="stale or swapped"):
        wp.collect(out, born_ref=born_ref, log=QUIET)
    # two OUTCARs swapped
    (tmp_path / "b").mkdir()
    out, born_ref, fx = _dfpt_outdir(tmp_path / "b")
    a, b = (os.path.join(out, d, "OUTCAR") for d in ("J5_p", "J5_m"))
    shutil.move(a, a + ".x")
    shutil.move(b, a)
    shutil.move(a + ".x", b)
    with pytest.raises(ValueError, match="stale or swapped"):
        wp.collect(out, log=QUIET)


def test_dfpt_born_ref_permuted_is_mapped_and_wrong_crystal_rejected(tmp_path):
    out, born_ref, fx = _dfpt_outdir(tmp_path)
    lam = wp.collect(out, born_ref=born_ref, log=QUIET)["Lambda"]
    c = fx["cells"]["ref"]
    perm = [2, 3, 0, 1]  # O first
    head = [
        ln for ln in fx["outcar_header"] if "TITEL" not in ln and "ions per" not in ln
    ]
    titel = [ln for ln in fx["outcar_header"] if "TITEL" in ln][::-1]
    geom = c["outcar_geometry"][:5] + [c["outcar_geometry"][5 + i] for i in perm]
    born = c["born"][:2]
    for n, i in enumerate(perm):
        born += [" ion %4d" % (n + 1)] + c["born"][3 + 4 * i : 6 + 4 * i]
    p = str(tmp_path / "OUTCAR_perm")
    with open(p, "w") as f:
        f.write(
            "\n".join(
                head + titel + ["   ions per type =               2   2"] + geom + born
            )
            + "\n"
        )
    assert np.allclose(wp.collect(out, born_ref=p, log=QUIET)["Lambda"], lam)
    wrong = str(tmp_path / "OUTCAR_strained")
    with open(wrong, "w") as f:
        f.write(_dfpt_outcar(fx, "J3_p"))
    with pytest.raises(ValueError, match="not the unstrained reference"):
        wp.collect(out, born_ref=wrong, log=QUIET)


def test_dfpt_refuses_second_order(tmp_path):
    out, _, _ = _dfpt_outdir(tmp_path)
    with pytest.raises(ValueError, match="LEPSILON"):
        wp.collect(out, second_order=True, log=QUIET)


def test_dfpt_malformed_born_block(tmp_path):
    out, _, fx = _dfpt_outdir(tmp_path)
    path = os.path.join(out, "J1_p", "OUTCAR")
    truncated = fx["cells"]["J1_p"]["born"][:-3]
    open(path, "w").write(_dfpt_outcar(fx, "J1_p", truncated))
    with pytest.raises(ValueError, match="BORN EFFECTIVE CHARGES"):
        wp.collect(out, log=QUIET)


# --------------------------------------------------------- berry-lambda
def _geometry_lines(A, xf):
    return (
        ["      direct lattice vectors                 reciprocal lattice vectors"]
        + ["  %.9f %.9f %.9f" % tuple(r) for r in A]
        + [" position of ions in fractional coordinates (direct lattice)"]
        + ["  %.8f %.8f %.8f" % tuple(r) for r in xf]
    )


def test_berry_lambda_displaces_a_constrained_atom(tmp_path):
    fx = _load("zno_berry_grid.json")
    lines = list(fx["poscar_ref"])
    k = lines.index("Direct")
    lines[k:k] = ["Selective dynamics"]
    lines[k + 2 :] = [ln + " F F F" for ln in lines[k + 2 :]]
    ref = _refdir(tmp_path, lines)
    out = str(tmp_path / "bl")
    man = wp.generate(
        "berry-lambda", ref, out, atom=3, direction="z", strain="zz", log=QUIET
    )
    import ase.io

    for e in man["entries"]:
        atoms = ase.io.read(os.path.join(out, e["dir"], "POSCAR"), format="vasp")
        A = np.asarray(man["lattice"]) @ (np.eye(3) + np.asarray(e["u"])).T
        moved = atoms.get_positions() - _zno_cell().xf @ A
        want = np.zeros((4, 3))
        want[2] = e["displacement"]
        assert np.abs(moved - want).max() < 1e-8 and abs(want[2, 2]) == 0.02


def test_berry_lambda_mixed_difference_against_stored(tmp_path):
    """Synthetic dipoles from psi = Omega e0:u + sum_k (Z*_k + Lambda_k:u) u0_k,
    with u0 the Cartesian displacement; p = F psi."""
    cell = _zno_cell()
    fx = _load("zno_berry_grid.json")
    ref = _refdir(tmp_path, fx["poscar_ref"])
    out = str(tmp_path / "bl")
    man = wp.generate(
        "berry-lambda",
        ref,
        out,
        atom=3,
        direction="x",
        strain="xz",
        h=0.01,
        delta=0.02,
        log=QUIET,
    )
    rng = np.random.default_rng(1)
    lam = rng.normal(size=(4, 3, 3, 3, 3))
    lam = 0.5 * (lam + lam.transpose(0, 1, 2, 4, 3))
    lam -= lam.mean(axis=0)
    z = np.array(
        [
            np.diag([2.1, 2.1, 2.2]),
            np.diag([2.1, 2.1, 2.2]),
            -np.diag([2.1, 2.1, 2.2]),
            -np.diag([2.1, 2.1, 2.2]),
        ]
    )
    for e in man["entries"]:
        u = np.asarray(e["u"])
        u0 = np.zeros((4, 3))
        u0[2] = e["displacement"]
        psi = 0.3 * u.sum(axis=0) + np.einsum(
            "kib,kb->i", z + np.einsum("kibmn,mn->kib", lam, u), u0
        )
        p = (np.eye(3) + u) @ psi
        A = np.asarray(man["lattice"]) @ (np.eye(3) + u).T
        xf = cell.xf.copy()
        xf[2] += np.asarray(e["displacement"]) @ np.linalg.inv(A)
        with open(os.path.join(out, e["dir"], "OUTCAR"), "w") as f:
            f.write("\n".join(fx["outcar_header"] + _geometry_lines(A, xf)) + "\n")
            f.write("Ionic dipole moment: p[ion]=( 0.0 0.0 0.0 ) |e| Angst\n")
            f.write(
                "Total electronic dipole moment: p[elc]=( %.10f %.10f %.10f ) |e| Angst\n"
                % tuple(p)
            )
    p = _container(
        tmp_path, ReferenceCell(cell.lavec, cell.elements[::-1], cell.xf[::-1])
    )
    with sf.update(p, cell) as f:
        sf.write_piezo_datasets(f, {sf.BORN_DERIV: lam[::-1]})
    res = wp.collect(out, strain_file=p, log=QUIET)
    assert np.allclose(res["Lambda_column"], lam[2, :, 0, 0, 2], atol=1e-6)
    assert np.allclose(res["Lambda_stored"], lam[2, :, 0, 0, 2])


# --------------------------------------------------- strainfile datasets
def test_write_piezo_keeps_siblings_and_validates(tmp_path):
    p = _container(tmp_path)
    e = np.zeros((3, 3, 3))
    e[2, 0, 0] = e[2, 1, 1] = 0.35
    b = np.zeros((3, 3, 3, 3, 3))
    b[2, 2, 2, 2, 2] = -2.0
    lam = np.zeros((4, 3, 3, 3, 3))
    lam[0, 2, 2, 2, 2] = -1.1
    with sf.update(p, _zno_cell()) as f:
        sf.write_piezo_datasets(f, {sf.SECOND_ORDER: b, sf.BORN_DERIV: lam})
    with sf.update(p, _zno_cell()) as f:
        sf.write_piezo(f, e)  # the phase-1b writer no longer drops the group
    info = sf.summary(p)
    assert info["piezo_second_order"][0][2, 2, 2, 2, 2] == -2.0
    assert info["piezo_born_deriv"][0][0, 2, 2, 2, 2] == -1.1
    assert info["piezo"][0][2, 0, 0] == 0.35
    lines = sf.supported_settings(info)
    assert any("B (EFIELD)" in ln and "present" in ln for ln in lines)
    sf.show(p, log=QUIET)
    bad = b.copy()
    bad[2, 0, 0, 2, 2] = 1.0  # breaks the pair exchange
    for data, match in (
        ({sf.SECOND_ORDER: bad}, "pair exchange"),
        ({sf.SECOND_ORDER: b[:2]}, "shape"),
        ({sf.BORN_DERIV: lam[:3]}, "atoms"),
        ({sf.BORN_DERIV: lam * np.nan}, "non-finite"),
        ({sf.BORN_DERIV: lam + np.arange(3)[:, None]}, "symmetric in mn"),
        ({"other": b}, "unknown"),
    ):
        with pytest.raises(ValueError, match=match):
            with sf.update(p, _zno_cell()) as f:
                sf.write_piezo_datasets(f, data)
    with h5py.File(p, "r") as f:  # untouched by the failed writes
        assert set(f["Piezoelectric"]) == {
            "clamped_ion",
            "second_order",
            "born_charge_strain_derivative",
        }
    q = str(tmp_path / "q.h5")
    shutil.copyfile(p, q)
    with h5py.File(q, "r+") as f:
        f["Piezoelectric/born_charge_strain_derivative"].attrs["convention"] = (
            "Cartesian"
        )
    with h5py.File(q, "r") as f, pytest.raises(ValueError, match="convention"):
        sf.read_piezo_dataset(f, sf.BORN_DERIV)


def test_symmetry_tolerance_matches_anphon_reader(tmp_path):
    """1e-8 + 1e-6 max|.| as in strain_coupling_io.cpp require_symmetric."""
    p = _container(tmp_path)
    b = np.zeros((3, 3, 3, 3, 3))
    b[2, 2, 2, 2, 2] = 1.0
    lam = np.zeros((4, 3, 3, 3, 3))
    lam[0, 2, 2, 2, 2] = 1.0
    for eps, ok in ((1.0e-7, True), (5.0e-6, False)):
        bb, ll = b.copy(), lam.copy()
        bb[2, 0, 1, 2, 2] = bb[2, 1, 0, 2, 2] = bb[2, 2, 2, 0, 1] = 0.3
        bb[2, 2, 2, 1, 0] = 0.3 + eps  # lm asymmetry
        ll[1, 0, 0, 0, 1] = 0.3
        ll[1, 0, 0, 1, 0] = 0.3 + eps  # mn asymmetry
        for data in ({sf.SECOND_ORDER: bb}, {sf.BORN_DERIV: ll}):
            if ok:
                with sf.update(p, _zno_cell()) as f:
                    sf.write_piezo_datasets(f, data)
            else:
                with pytest.raises(ValueError, match="tolerance 1.01e-06"):
                    with sf.update(p, _zno_cell()) as f:
                        sf.write_piezo_datasets(f, data)


def test_lambda_only_group(tmp_path):
    p = _container(tmp_path)
    with sf.update(p, _zno_cell()) as f:
        sf.write_piezo_datasets(f, {sf.BORN_DERIV: np.zeros((4, 3, 3, 3, 3))})
    info = sf.summary(p)
    assert info["piezo"] is None and info["piezo_born_deriv"] is not None
    with (
        h5py.File(p, "r") as f,
        pytest.raises(ValueError, match="no /Piezoelectric/clamped_ion"),
    ):
        sf.read_piezo(f)


def test_cli_wiring(tmp_path):
    import importlib

    cli = importlib.import_module("piezo")
    out, _ = _berry_outdir(tmp_path)
    p = _container(tmp_path)
    cli.main(["collect", out, "--strain-file", p])
    with pytest.raises(SystemExit):
        cli.main(["collect", out, "--strain-file", p])  # exists, no --overwrite
    with pytest.raises(SystemExit):
        cli.main(
            [
                "generate",
                "--route",
                "berry",
                str(tmp_path / "nope"),
                str(tmp_path / "o"),
            ]
        )
