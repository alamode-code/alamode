import json
import os
import shutil
import sys

import numpy as np
import pytest

from strainkit import strainfile as sf
from strainkit.units import (
    BOHR_IN_ANGSTROM,
    gpa_to_ry_per_cell,
    legacy_ry_per_cell_to_gpa,
)
from strainkit.writers import (
    ReferenceCell,
    write_C1_array_in,
    write_elastic_constants_in,
    write_strain_force_in,
    write_strain_harmonic_in,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emt_helpers import fit_reference_fc2  # noqa: E402

h5py = pytest.importorskip("h5py")
QUIET = lambda *a: None  # noqa: E731


def _cell(a=3.0, c=5.0):
    lav = np.array([[a, 0.0, 0.0], [-a / 2, a * np.sqrt(3) / 2, 0.0], [0.0, 0.0, c]])
    xf = np.array(
        [
            [1 / 3, 2 / 3, 0.0],
            [2 / 3, 1 / 3, 0.5],
            [1 / 3, 2 / 3, 0.38],
            [2 / 3, 1 / 3, 0.88],
        ]
    )
    return ReferenceCell(lav, ["Zn", "Zn", "O", "O"], xf)


def _doubled(cell):
    """The c-doubled supercell (a nested cell of the same crystal)."""
    lav = cell.lavec.copy()
    lav[2] *= 2
    xf = np.vstack([cell.xf * [1, 1, 0.5], cell.xf * [1, 1, 0.5] + [0, 0, 0.5]])
    return ReferenceCell(lav, list(cell.elements) * 2, xf)


def _blocks(natom, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (m, 0.005, 1.0, rng.normal(size=(natom, 3)))
        for m in ("xx", "yy", "zz", "yz", "zx", "xy")
    ]


def test_cell_group_round_trip(tmp_path):
    cell = _cell()
    p = str(tmp_path / "c.h5")
    with h5py.File(p, "w") as f:
        sf.write_cell_group(f.create_group("X"), cell.lavec, cell.xf, cell.elements)
        g = f["X"]
        assert g["lattice_vector"].attrs["unit"] == "bohr"
        assert (
            g["atomic_kinds"].dtype == np.int32
            and g["number_of_atoms"].dtype == np.uint64
        )
        assert g["number_of_atoms"][()] == 4 and [
            sf._decode(e) for e in g["elements"][()]
        ] == ["Zn", "O"]
        assert (
            g["mapping_table"].shape == (4, 1)
            and g["number_of_primitive_translations"][()] == 1
        )
        # rows of the file are the lattice vectors (in bohr): no transpose on the Python side
        assert np.allclose(g["lattice_vector"][()] * BOHR_IN_ANGSTROM, cell.lavec)
        back = sf.read_cell_group(g)
    assert (
        np.allclose(back.lavec, cell.lavec)
        and back.elements == cell.elements
        and np.allclose(back.xf, cell.xf)
    )


def test_container_round_trip(tmp_path):
    cell = _cell()
    p = str(tmp_path / "s.h5")
    stress = np.diag([0.1, 0.1, -0.2])
    c2 = np.random.default_rng(1).normal(size=(9, 9))
    c3 = np.random.default_rng(2).normal(size=(9, 9, 9))
    blocks = _blocks(4)
    with sf.update(p, cell, sf.provenance_record("Elastic"), source="test") as f:
        sf.write_elastic(
            f,
            stress,
            c2,
            c3,
            {"source": "test", "rank": 83, "rms_energy_GPa": float("nan")},
        )
    with sf.update(p, cell, sf.provenance_record("StrainForce")) as f:
        sf.write_strain_force(f, blocks, cell, {"central": False})
    with h5py.File(p, "r") as f:
        assert sf._decode(f.attrs["schema"]) == sf.SCHEMA
        assert (
            f.attrs["format_version"] == 1
            and f.attrs["format_version"].dtype == np.int32
        )
        assert set(f) == {"ReferenceCell", "Elastic", "StrainForce"}
        s, a2, a3, attrs = sf.read_elastic(f)
        assert np.allclose(s, stress) and np.allclose(a2, c2) and np.allclose(a3, c3)
        assert (
            attrs["rank"] == 83 and "rms_energy_GPa" not in attrs
        )  # NaN attributes are skipped
        assert f["Elastic/stress"].attrs["unit"] == "GPa" and f[
            "Elastic/toec"
        ].shape == (9, 9, 9)
        b, c = sf.read_strain_force(f)
        assert [x[0] for x in b] == [x[0] for x in blocks] and np.allclose(
            b[3][3], blocks[3][3]
        )
        assert f["StrainForce/forces"].attrs["unit"] == "eV/angstrom" and c.natom == 4
        u = f["StrainForce/displacement_gradient"][()]
        assert np.allclose(
            u[3], [[0, 0, 0], [0, 0, 0.0025], [0, 0.0025, 0]]
        ) and np.allclose(u[0], np.diag([0.005, 0, 0]))
        prov = json.loads(sf._decode(f.attrs["provenance"]))
        assert [r["group"] for r in prov] == ["Elastic", "StrainForce"]
    lines = sf.supported_settings(sf.summary(p))
    assert any("bit 1 (elastic constants)" in ln and ": yes" in ln for ln in lines)
    assert any("bit 2 (strain-force coupling)" in ln and ": yes" in ln for ln in lines)
    assert any(
        "bit 4 (strain-harmonic-IFC coupling)" in ln and ": no" in ln for ln in lines
    )
    assert any(
        "supported STRAIN_COUPLING values" in ln and ": 0 1 2 3" in ln for ln in lines
    )


def test_update_is_transactional_and_checks_the_crystal(tmp_path):
    cell = _cell()
    p = str(tmp_path / "s.h5")
    with sf.update(p, cell) as f:
        sf.write_elastic(f, np.zeros((3, 3)))
    size = os.path.getsize(p)
    # a failing writer leaves the file untouched and no .part behind
    with pytest.raises(ValueError, match="nothing to write"):
        with sf.update(p, cell) as f:
            sf.write_elastic(f)
    assert not os.path.exists(p + ".part") and os.path.getsize(p) == size
    # a different crystal is refused
    other = _cell(a=3.1)
    with pytest.raises(ValueError, match="different crystal"):
        with sf.update(p, other) as f:
            sf.write_strain_force(f, _blocks(4), other)
    # a nested (c-doubled) cell describes the same crystal and is accepted
    big = _doubled(cell)
    with sf.update(p, big) as f:
        sf.write_strain_force(f, _blocks(8), big)
    with h5py.File(p, "r") as f:
        assert (
            sf.read_cell_group(f["ReferenceCell"]).natom == 4
            and sf.read_strain_force(f)[1].natom == 8
        )
    # --force rebuilds the file with the new reference cell and drops the old groups
    with sf.update(p, other, force=True) as f:
        sf.write_elastic(f, np.zeros((3, 3)))
    with h5py.File(p, "r") as f:
        assert "StrainForce" not in f and np.isclose(
            sf.read_cell_group(f["ReferenceCell"]).lavec[0, 0], 3.1
        )


def test_mode_table_validation(tmp_path):
    cell = _cell()
    p = str(tmp_path / "s.h5")
    with pytest.raises(ValueError, match="invalid strain mode"):
        with sf.update(p, cell) as f:
            sf.write_strain_force(f, [("xz", 0.005, 1.0, np.zeros((4, 3)))], cell)
    with pytest.raises(ValueError, match="shape"):
        with sf.update(p, cell) as f:
            sf.write_strain_force(f, [("xx", 0.005, 1.0, np.zeros((3, 3)))], cell)
    with pytest.raises(ValueError, match="soec and toec"):
        with sf.update(p, cell) as f:
            sf.write_elastic(f, None, np.zeros((9, 9)), None)


def test_legacy_conversion_is_anphon_inverse():
    v = 319.7193027516895  # ZnO tutorial cell, bohr^3
    x = np.array([1.0, -2.5, 100.0])
    assert np.allclose(
        legacy_ry_per_cell_to_gpa(x * gpa_to_ry_per_cell(v), v), x, rtol=1e-14
    )
    # anphon: 1e9 * V[m^3] / Ryd with Ryd = 4.35974394e-18 / 2 J
    assert np.isclose(
        gpa_to_ry_per_cell(v),
        1.0e9 * v * (0.52917721092e-10) ** 3 / (4.35974394e-18 / 2),
        rtol=1e-15,
    )


@pytest.fixture(scope="module")
def cu_fc2(tmp_path_factory, ase_mod, spglib_mod, alm_mod):
    from ase.build import bulk

    root = tmp_path_factory.mktemp("strainfile_fc2")
    sc = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    xml = fit_reference_fc2(sc, str(root / "fc2.xml"), "xml")
    h5 = fit_reference_fc2(sc, str(root / "fc2.h5"), "h5")
    return sc, xml, h5, str(root)


def _fc2_rows(g):
    ai = g["atom_indices"][()]
    ais = g["atom_indices_supercell"][()]
    ci = g["coord_indices"][()]
    sh = g["shift_vectors"][()]
    v = g["force_constant_values"][()]
    rows = {}
    for k in range(len(v)):
        key = (tuple(ai[k]), tuple(ais[k]), tuple(ci[k]), tuple(np.round(sh[k], 6)))
        assert key not in rows
        rows[key] = (sh[k], v[k])
    return rows


def test_xml_to_fc2_group_matches_alm_h5(cu_fc2, tmp_path):
    sc, xml, h5, root = cu_fc2
    p = str(tmp_path / "conv.h5")
    with h5py.File(p, "w") as f:
        sf.xml_to_fc2_group(xml, f.create_group("entry"))
    with h5py.File(p, "r") as f, h5py.File(h5, "r") as ref:
        a = _fc2_rows(f["entry/ForceConstants/Order2"])
        b = _fc2_rows(ref["ForceConstants/Order2"])
        assert len(a) == len(b) and set(a) == set(
            b
        )  # no unmatched rows in either direction
        for k in a:
            assert np.allclose(a[k][0], b[k][0], atol=1e-8)
            assert np.isclose(a[k][1], b[k][1], rtol=1e-10, atol=1e-12)
        for name in (
            "lattice_vector",
            "fractional_coordinate",
            "atomic_kinds",
            "mapping_table",
        ):
            assert np.allclose(
                f["entry/SuperCell"][name][()], ref["SuperCell"][name][()]
            )
        o2 = f["entry/ForceConstants/Order2"]
        assert (
            o2["shift_vectors"].attrs["basis"] == "Cartesian"
            and o2["shift_vectors"].attrs["unit"] == "bohr"
        )
        assert o2["force_constant_values"].attrs["unit"] == "Ry/bohr^2"
        assert (
            o2["atom_indices"].dtype == np.int32
            and o2["shift_vectors"].compression == "gzip"
        )


def test_pack_from_legacy_text(cu_fc2, tmp_path):
    sc, xml, h5, root = cu_fc2
    d = tmp_path / "strain_IFC"
    d.mkdir()
    cell = sf.cell_from_atoms(
        sc
    )  # /PrimitiveCell of the h5 reference is the supercell itself
    vol_bohr3 = sc.get_volume() / BOHR_IN_ANGSTROM**3
    rng = np.random.default_rng(3)
    c2, c3, s0 = (
        rng.normal(size=(9, 9)),
        rng.normal(size=(9, 9, 9)),
        np.diag([0.5, 0.5, 0.5]),
    )
    fac = gpa_to_ry_per_cell(vol_bohr3)
    write_elastic_constants_in(
        str(d / "elastic_constants.in"), c2 * fac, c3 * fac
    )  # legacy Ry layout
    write_C1_array_in(str(d / "C1_array.in"), s0, unit="GPa")
    blocks = _blocks(32)
    write_strain_force_in(str(d / "strain_force.in"), blocks, reference_cell=cell)
    shutil.copy(xml, d / "strain_001.xml")
    shutil.copy(h5, d / "strain_002.h5")
    write_strain_harmonic_in(
        str(d / "strain_harmonic.in"),
        [("xx", 0.005, 0.5, "strain_001.xml"), ("xx", -0.005, 0.5, "strain_002.h5")],
    )
    out = str(tmp_path / "packed.h5")
    sf.pack(out, str(d), None, h5, log=QUIET)
    with h5py.File(out, "r") as f:
        s, a2, a3, attrs = sf.read_elastic(f)
        assert (
            np.allclose(s, s0)
            and np.allclose(a2, c2, rtol=1e-12)
            and np.allclose(a3, c3, rtol=1e-12)
        )
        assert attrs["stress_source"] == "C1_array.in" and np.isclose(
            attrs["legacy_volume_bohr3"], vol_bohr3
        )
        b, c = sf.read_strain_force(f)
        assert c.natom == 32 and np.allclose(
            b[0][3], blocks[0][3], atol=1e-14
        )  # text precision .15f
        rows, entries = sf.read_strain_harmonic(f)
        assert [r[0] for r in rows] == ["xx", "xx"] and len(entries) == 2
        assert "PrimitiveCell" in f[entries[1]] and "PrimitiveCell" not in f[entries[0]]
        assert f[entries[0]].attrs["source_file"] == "strain_001.xml"
    # the entries are the undeformed supercell labeled as strained: check must complain
    problems = sf.check(out, None, h5, log=QUIET)
    assert any("entry_001" in p and "lattice" in p for p in problems)
    with pytest.raises(ValueError, match="exists"):
        sf.pack(out, str(d), None, h5, log=QUIET)
    # stress-only pack
    d2 = tmp_path / "only_c1"
    d2.mkdir()
    write_C1_array_in(str(d2 / "C1_array.in"), s0, unit="GPa")
    out2 = str(tmp_path / "stress_only.h5")
    sf.pack(out2, str(d2), None, h5, log=QUIET)
    with h5py.File(out2, "r") as f:
        s, a2, a3, _ = sf.read_elastic(f)
        assert np.allclose(s, s0) and a2 is None
    assert any(
        "no /Elastic/soec" in ln for ln in sf.supported_settings(sf.summary(out2))
    )


# ---------------------------------------------------------------- piezo
OUTCAR_ZNO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "OUTCAR_ZnO_piezo"
)


def _zno_container(tmp_path, cell=None):
    """A container whose /ReferenceCell is the crystal of the OUTCAR fixture, with /Elastic."""
    if cell is None:
        cell = sf.parse_outcar_piezo(OUTCAR_ZNO)[2]
    p = str(tmp_path / "zno.h5")
    with sf.update(p, cell) as f:
        sf.write_elastic(f, np.diag([0.1, 0.2, 0.3]))
    return p


def test_parse_outcar_piezo():
    e, version, cell = sf.parse_outcar_piezo(OUTCAR_ZNO)
    assert version == "vasp.6.5.1"
    # the last C/m^2 block; shear column ZX is e_xxz = e_xzx (no factor 2)
    assert e[0, 0, 2] == e[0, 2, 0] == 0.41468 and e[1, 1, 2] == e[1, 2, 1] == 0.41468
    assert e[2, 0, 0] == e[2, 1, 1] == 0.38574 and e[2, 2, 2] == -0.78317
    assert np.allclose(
        sf.piezo_to_voigt(e)[2], [0.38574, 0.38574, -0.78317, 0.0, 0.0, 0.0]
    )
    assert cell.elements == ["Zn", "Zn", "O", "O"]
    assert np.isclose(abs(np.linalg.det(cell.lavec)), 47.3775, atol=1e-3)
    assert np.isclose(cell.xf[2, 2], 0.37957704)


def test_parse_outcar_piezo_refuses_ionic_and_extensive(tmp_path):
    text = open(OUTCAR_ZNO).read()
    head = "PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z"
    cm2 = head + "        (C/m^2)"
    # only the e*Angst block left
    p = tmp_path / "OUTCAR_eA"
    p.write_text(text.replace(cm2, "SOMETHING ELSE"))
    with pytest.raises(ValueError, match="extensive"):
        sf.parse_outcar_piezo(str(p))
    # only IONIC CONTR blocks
    p = tmp_path / "OUTCAR_ion"
    p.write_text(
        text.replace(head, "PIEZOELECTRIC TENSOR IONIC CONTR  for field in x, y, z")
    )
    with pytest.raises(ValueError, match="IONIC CONTR"):
        sf.parse_outcar_piezo(str(p))
    # a later, truncated C/m^2 block is skipped: the last complete one is used
    p = tmp_path / "OUTCAR_trunc"
    p.write_text(text + "\n " + cm2 + "\n          XX          YY\n")
    assert sf.parse_outcar_piezo(str(p))[0][0, 0, 2] == 0.41468
    # a later block cut inside its z row: the previous complete block is used
    block = text[text.index(cm2) :].split("\n")[:6]
    block[5] = block[5].split("-0.78317")[0]
    p = tmp_path / "OUTCAR_trunc_row"
    p.write_text(text + "\n " + "\n".join(block).replace("0.41468", "9.99999") + "\n")
    e = sf.parse_outcar_piezo(str(p))[0]
    assert e[0, 0, 2] == 0.41468 and e[2, 2, 2] == -0.78317


def test_piezo_from_outcar_replaces_only_its_group(tmp_path):
    p = _zno_container(tmp_path)
    with h5py.File(p, "r") as f:
        stress_before = f["Elastic/stress"][()]
    sf.add_piezo(p, outcar=OUTCAR_ZNO, log=QUIET)
    with h5py.File(p, "r") as f:
        assert set(f) == {"ReferenceCell", "Elastic", "Piezoelectric"}
        assert np.array_equal(f["Elastic/stress"][()], stress_before)
        e, attrs = sf.read_piezo(f)
        assert attrs["unit"] == "C/m^2" and attrs["convention"] == "proper, clamped-ion"
        assert attrs["code_version"] == "vasp.6.5.1" and "LEPSILON" in attrs["method"]
        assert attrs["source"].endswith("OUTCAR_ZnO_piezo")
        assert e[0, 2, 0] == 0.41468
        prov = json.loads(sf._decode(f.attrs["provenance"]))
        assert prov[-1]["group"] == "Piezoelectric"
    with pytest.raises(ValueError, match="--overwrite"):
        sf.add_piezo(p, outcar=OUTCAR_ZNO, log=QUIET)
    sf.add_piezo(p, outcar=OUTCAR_ZNO, overwrite=True, log=QUIET)
    info = sf.summary(p)
    assert info["piezo"] is not None and info["elastic"][0] is not None
    assert any(
        "piezoelectric" in ln and "present" in ln for ln in sf.supported_settings(info)
    )
    assert not sf.check(p, log=QUIET)
    # the CLI wiring
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import importlib

    cli = importlib.import_module("strainfile")
    cli.main(["piezo", p, "--outcar", OUTCAR_ZNO, "--overwrite"])
    with pytest.raises(SystemExit):
        cli.main(["piezo", p, "--outcar", OUTCAR_ZNO])


def test_piezo_refuses_another_crystal(tmp_path):
    cell = sf.parse_outcar_piezo(OUTCAR_ZNO)[2]
    other = ReferenceCell(cell.lavec * 1.01, cell.elements, cell.xf)
    p = _zno_container(tmp_path, other)
    with pytest.raises(ValueError, match="not the crystal"):
        sf.add_piezo(p, outcar=OUTCAR_ZNO, log=QUIET)
    with pytest.raises(ValueError, match="does not exist"):
        sf.add_piezo(str(tmp_path / "none.h5"), outcar=OUTCAR_ZNO, log=QUIET)


def test_piezo_from_voigt_table(tmp_path, ase_mod):
    cell = sf.parse_outcar_piezo(OUTCAR_ZNO)[2]
    p = _zno_container(tmp_path)
    table = np.arange(1.0, 19.0).reshape(3, 6) / 10.0
    t = tmp_path / "e.txt"
    np.savetxt(t, table)
    with pytest.raises(ValueError, match="--structure"):
        sf.add_piezo(p, voigt=str(t), log=QUIET)
    poscar = tmp_path / "POSCAR"
    lines = ["ZnO", "1.0"] + ["  %.10f %.10f %.10f" % tuple(r) for r in cell.lavec]
    lines += ["Zn O", "2 2", "Direct"] + [
        "  %.10f %.10f %.10f" % tuple(r) for r in cell.xf
    ]
    poscar.write_text("\n".join(lines) + "\n")
    e = sf.add_piezo(p, voigt=str(t), structure=str(poscar), log=QUIET)
    # standard Voigt order xx yy zz yz xz xy, shear = tensor component
    assert e[1, 1, 2] == e[1, 2, 1] == table[1, 3]
    assert e[2, 0, 2] == e[2, 2, 0] == table[2, 4]
    assert e[0, 0, 1] == e[0, 1, 0] == table[0, 5]
    assert np.allclose(sf.piezo_to_voigt(e), table)
    bad = tmp_path / "bad.txt"
    np.savetxt(bad, table[:, :5])
    with pytest.raises(ValueError, match="3 x 6"):
        sf.add_piezo(
            p, voigt=str(bad), structure=str(poscar), overwrite=True, log=QUIET
        )


def test_read_piezo_rejects_malformed(tmp_path):
    p = _zno_container(tmp_path)
    sf.add_piezo(p, outcar=OUTCAR_ZNO, log=QUIET)
    e = sf.parse_outcar_piezo(OUTCAR_ZNO)[0]
    asym = e.copy()
    asym[0, 0, 2] += 1.0e-3
    for mutate, match in (
        (lambda d: d.attrs.__setitem__("unit", "e/bohr^2"), "unit"),
        (lambda d: d.attrs.__setitem__("convention", "improper"), "convention"),
        (lambda d: d.__setitem__(Ellipsis, asym), "symmetric"),
    ):
        q = str(tmp_path / "m.h5")
        shutil.copyfile(p, q)
        with h5py.File(q, "r+") as f:
            mutate(f["Piezoelectric/clamped_ion"])
        with h5py.File(q, "r") as f, pytest.raises(ValueError, match=match):
            sf.read_piezo(f)
    with pytest.raises(ValueError, match="symmetric"):
        with sf.update(p, sf.parse_outcar_piezo(OUTCAR_ZNO)[2]) as f:
            sf.write_piezo(f, asym)
