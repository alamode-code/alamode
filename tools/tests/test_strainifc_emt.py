import os
import sys

import numpy as np
import pytest

from strainkit import fcsorder
from strainkit import workflow_ifc as wi
from strainkit.writers import read_strain_force_in, read_strain_harmonic_in

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emt_helpers import fit_reference_fc2, run_emt  # noqa: E402

QUIET = lambda *a: None  # noqa: E731


@pytest.fixture(scope="module")
def cu_supercell(tmp_path_factory, ase_mod, spglib_mod, alm_mod):
    import ase.io
    from ase.build import bulk

    root = tmp_path_factory.mktemp("ifc_harm")
    sc = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    tmpl = root / "tmpl"
    tmpl.mkdir()
    ase.io.write(tmpl / "input.extxyz", sc)
    ref_xml = fit_reference_fc2(sc, str(root / "ref_fc2.xml"), "xml")
    ref_h5 = fit_reference_fc2(sc, str(root / "ref_fc2.h5"), "h5")
    return sc, str(tmpl), ref_xml, ref_h5, str(root)


def test_harmonic_coupling(cu_supercell):
    sc, tmpl, ref_xml, ref_h5, root = cu_supercell
    work = os.path.join(root, "harm")
    m = wi.generate(
        "harmonic",
        "ase",
        tmpl,
        work,
        smag=0.005,
        central=True,
        mode_names=["xx", "yz"],
        with_reference=True,
        log=QUIET,
    )
    assert len(m["entries"]) == 5 and all(
        e["nodisp_dir"] == "nodisp" for e in m["entries"]
    )
    assert m["entries"][0]["reference"] and m["entries"][0]["dir"] == "strain_000"
    assert run_emt(work) == sum(e["n_disp"] + 1 for e in m["entries"])
    with pytest.raises(ValueError, match="--fcs"):
        wi.collect(work, log=QUIET)
    logs = []
    fname = wi.collect(
        work, fcs=ref_xml, fcs_format="xml", write_dfset_files=True, log=logs.append
    )
    rows = read_strain_harmonic_in(fname)
    assert [r[0] for r in rows] == ["xx", "xx", "yz", "yz"]  # strain_000 not listed
    assert os.path.exists(os.path.join(work, "results", "strain_000.xml"))
    assert any("FC2 vs" in ln and "max|dPhi2|" in ln for ln in logs)
    st = fcsorder.fc2_difference(
        os.path.join(work, "results", "strain_000.xml"), ref_xml
    )
    assert st["n_only_a"] == 0 and st["max_abs"] < 1e-6  # same data and fit settings
    assert rows[0][1] == 0.005 and rows[1][1] == -0.005 and rows[0][2] == 0.5
    ref = fcsorder.read_fcs_structure(ref_xml)
    for r in rows:
        g = fcsorder.verify_generated_fcs(os.path.join(work, "results", r[3]), ref)
        assert g.nat == 32
    assert os.path.exists(os.path.join(work, "results", "DFSET_strain_001"))
    # h5 output, verified against the h5 reference as well
    fname = wi.collect(
        work, fcs=ref_h5, fcs_format="h5", results_dir="results_h5", log=QUIET
    )
    assert all(r[3].endswith(".h5") for r in read_strain_harmonic_in(fname))
    # the strain-coupling container: h5 entries embedded, checked against the reference
    import h5py
    from strainkit import strainfile as sf

    cont = os.path.join(root, "cu.strain.h5")
    with pytest.raises(ValueError, match="fcs-format h5"):
        wi.collect(
            work,
            fcs=ref_h5,
            fcs_format="xml",
            results_dir="results_x",
            strain_file=cont,
            log=QUIET,
        )
    wi.collect(
        work,
        fcs=ref_h5,
        fcs_format="h5",
        results_dir="results_h5c",
        strain_file=cont,
        log=QUIET,
    )
    with h5py.File(cont, "r") as f:
        rows, entries = sf.read_strain_harmonic(f)
        assert [r[0] for r in rows] == ["xx", "xx", "yz", "yz"] and len(entries) == 4
        assert sf.read_cell_group(f["ReferenceCell"]).natom == 32
        v_emb = f[entries[0]]["ForceConstants/Order2/force_constant_values"][()]
        with h5py.File(os.path.join(work, "results_h5c", "strain_001.h5"), "r") as s:
            assert np.array_equal(
                v_emb, s["ForceConstants/Order2/force_constant_values"][()]
            )
        assert (
            f[entries[0]].attrs["mode"] == "xx"
            and f["StrainHarmonic"].attrs["central"] == 1
        )
    assert sf.check(cont, None, ref_h5, log=QUIET) == []
    lines = sf.supported_settings(sf.summary(cont))
    assert any(
        "bit 4 (strain-harmonic-IFC coupling)" in ln and ": yes (subset" in ln
        for ln in lines
    )  # xx and yz only
    # a DFT output with displaced atoms is rejected
    import ase.io

    out = os.path.join(work, "strain_001", "nodisp", "output.extxyz")
    a = ase.io.read(out)
    b = a.copy()
    b.positions[0] += [0.01, 0, 0]
    b.calc = a.calc
    ase.io.write(out, b)
    with pytest.raises(ValueError, match="geometry does not match"):
        wi.collect(work, fcs=ref_xml, results_dir="results_bad", log=QUIET)
    ase.io.write(out, a)


@pytest.fixture(scope="module")
def hcp_setup(tmp_path_factory, ase_mod, spglib_mod, alm_mod):
    import ase.io
    from ase.build import bulk

    root = tmp_path_factory.mktemp("ifc_force")
    hcp = bulk("Cu", "hcp", a=2.55, c=4.2)
    tmpl = root / "tmpl"
    tmpl.mkdir()
    ase.io.write(tmpl / "input.extxyz", hcp)
    ase.io.write(root / "cell_prim.extxyz", hcp)
    ase.io.write(root / "cell_211.extxyz", hcp * (2, 1, 1))
    ref = fit_reference_fc2(hcp * (2, 2, 2), str(root / "ref_hcp.xml"), "xml")
    perm = hcp[[1, 0]]  # permuted template
    ptmpl = root / "tmpl_perm"
    ptmpl.mkdir()
    ase.io.write(ptmpl / "input.extxyz", perm)
    return hcp, str(tmpl), str(ptmpl), ref, str(root)


def test_force_coupling(hcp_setup):
    hcp, tmpl, ptmpl, ref, root = hcp_setup
    work = os.path.join(root, "force")
    m = wi.generate("force", "ase", tmpl, work, smag=0.005, log=QUIET)
    assert len(m["entries"]) == 6 and os.path.isdir(
        os.path.join(work, "strain_000", "primitive")
    )
    assert run_emt(work) == 7
    with pytest.raises(
        ValueError, match="&cell"
    ):  # XML reference needs the anphon cell
        wi.collect(work, fcs=ref, log=QUIET)
    fname = wi.collect(
        work, fcs=ref, anphon_cell=os.path.join(root, "cell_prim.extxyz"), log=QUIET
    )
    blocks, ref_cell = read_strain_force_in(fname, 2)
    assert [b[0] for b in blocks] == ["xx", "yy", "zz", "yz", "zx", "xy"]
    # the &reference_cell header names the anphon primitive cell (here the DFT cell)
    assert ref_cell is not None and ref_cell.natom == 2
    assert set(ref_cell.elements) == set(hcp.get_chemical_symbols())
    assert np.isclose(abs(np.linalg.det(ref_cell.lavec)), hcp.get_volume())
    fxx = blocks[0][3]
    assert np.abs(fxx).max() > 1e-3 and np.allclose(
        fxx[0], -fxx[1]
    )  # internal relaxation of hcp
    assert np.abs(blocks[2][3]).max() < 1e-6  # zz strain: no internal forces
    # tiled conventional (2x1x1) anphon cell
    fname = wi.collect(
        work,
        fcs=ref,
        anphon_cell=os.path.join(root, "cell_211.extxyz"),
        results_dir="results_211",
        log=QUIET,
    )
    b4, ref4 = read_strain_force_in(fname)  # natmin from the header
    assert ref4.natom == 4 and np.isclose(
        abs(np.linalg.det(ref4.lavec)), 2 * hcp.get_volume()
    )
    assert np.allclose(b4[0][3][:2], fxx) and np.allclose(b4[0][3][2:], fxx)
    # without --fcs: template order, and no header (the anphon cell is unknown)
    fname = wi.collect(work, results_dir="results_nofcs", log=QUIET)
    b_nofcs, ref_nofcs = read_strain_force_in(fname, 2)
    assert ref_nofcs is None and np.allclose(b_nofcs[0][3], fxx)
    # the strain-coupling container: the primitive cell first, then the tiled
    # 2x1x1 cell updates the same file (nested cells are the same crystal)
    import h5py
    from strainkit import strainfile as sf

    cont = os.path.join(root, "hcp.strain.h5")
    cell_prim = os.path.join(root, "cell_prim.extxyz")
    wi.collect(
        work,
        fcs=ref,
        anphon_cell=cell_prim,
        results_dir="results_c",
        strain_file=cont,
        log=QUIET,
    )
    with h5py.File(cont, "r") as f:
        b, c = sf.read_strain_force(f)
        assert (
            c.natom == 2
            and np.allclose(b[0][3], fxx)
            and f["StrainForce"].attrs["dft_code"] == "ase"
        )
    wi.collect(
        work,
        fcs=ref,
        anphon_cell=os.path.join(root, "cell_211.extxyz"),
        results_dir="results_211c",
        strain_file=cont,
        log=QUIET,
    )
    with h5py.File(cont, "r") as f:
        b4, c4 = sf.read_strain_force(f)
        assert c4.natom == 4 and sf.read_cell_group(f["ReferenceCell"]).natom == 2
        assert np.allclose(b4[0][3][:2], fxx)
    assert sf.check(cont, cell_prim, ref, log=QUIET) == []


def test_force_coupling_permuted_template(hcp_setup):
    hcp, tmpl, ptmpl, ref, root = hcp_setup
    work = os.path.join(root, "force_perm")
    wi.generate(
        "force",
        "ase",
        ptmpl,
        work,
        smag=0.005,
        mode_names=["xx", "yy", "zz", "yz", "zx", "xy"],
        log=QUIET,
    )
    run_emt(work)
    cell = os.path.join(root, "cell_prim.extxyz")
    with pytest.raises(ValueError, match="permutation"):
        wi.collect(work, fcs=ref, anphon_cell=cell, log=QUIET)
    fname = wi.collect(work, fcs=ref, anphon_cell=cell, reorder=True, log=QUIET)
    ref_rows, _ = read_strain_force_in(
        os.path.join(root, "force", "results", "strain_force.in"), 2
    )
    rows, _ = read_strain_force_in(fname, 2)
    assert np.allclose(
        rows[0][3], ref_rows[0][3]
    )  # rows are in anphon order after --reorder
    # check() reports the ordering without failing
    wi.check(work, fcs=ref, anphon_cell=cell, log=QUIET)


def test_force_coupling_requires_all_modes(hcp_setup):
    hcp, tmpl, ptmpl, ref, root = hcp_setup
    with pytest.raises(ValueError, match="all six strain modes"):
        wi.generate(
            "force",
            "ase",
            tmpl,
            os.path.join(root, "force_partial"),
            smag=0.005,
            mode_names=["xx", "yy"],
            log=QUIET,
        )


def test_verify_generated_fcs_detects_corruption(cu_supercell, tmp_path):
    """A generated FC file with a different translation table is rejected."""
    import shutil

    sc, tmpl, ref_xml, ref_h5, root = cu_supercell
    ref = fcsorder.read_fcs_structure(ref_xml)
    bad = str(tmp_path / "bad.xml")
    text = open(ref_xml).read()
    # swap the supercell indices of two translation entries -> map_p2s/map_s2p change
    import re

    maps = re.findall(r'<map tran="(\d+)" atom="(\d+)">(\d+)</map>', text)
    (t1, a1, v1), (t2, a2, v2) = maps[0], maps[1]
    text = text.replace(
        f'<map tran="{t1}" atom="{a1}">{v1}</map>',
        f'<map tran="{t1}" atom="{a1}">{v2}</map>',
        1,
    )
    text = text.replace(
        f'<map tran="{t2}" atom="{a2}">{v2}</map>',
        f'<map tran="{t2}" atom="{a2}">{v1}</map>',
        1,
    )
    open(bad, "w").write(text)
    with pytest.raises(ValueError, match="map_p2s|map_s2p"):
        fcsorder.verify_generated_fcs(bad, ref)
    # h5: the primitive cell is compared as well
    import h5py

    bad5 = str(tmp_path / "bad.h5")
    shutil.copy(ref_h5, bad5)
    with h5py.File(bad5, "r+") as h:
        xf = h["/PrimitiveCell/fractional_coordinate"]
        xf[0, 0] = xf[0, 0] + 0.01
    with pytest.raises(ValueError, match="PrimitiveCell"):
        fcsorder.verify_generated_fcs(bad5, fcsorder.read_fcs_structure(ref_h5))


def test_elastic_fit_writes_the_same_strain_force(hcp_setup):
    """elastic.py fit derives /StrainForce from its k = +-1 singles; it must equal
    strainifc.py --coupling force --central at the same smag."""
    h5py = pytest.importorskip("h5py")
    from strainkit import strainfile as sf
    from strainkit import workflow_elastic as we

    hcp, tmpl, ptmpl, ref, root = hcp_setup
    cell_prim = os.path.join(root, "cell_prim.extxyz")
    wf = os.path.join(root, "force_central")
    wi.generate("force", "ase", tmpl, wf, smag=0.005, central=True, log=QUIET)
    assert run_emt(wf) == 13
    f_ifc = wi.collect(wf, fcs=ref, anphon_cell=cell_prim, log=QUIET)
    b_ifc, _ = read_strain_force_in(f_ifc, 2)

    wel = os.path.join(root, "elastic")
    we.generate("ase", tmpl, wel, smag=0.005, nmag=1, log=QUIET)
    assert run_emt(wel) == 43
    cont = os.path.join(root, "hcp_elastic.strain.h5")
    we.fit(wel, "stress", fcs=ref, anphon_cell=cell_prim, strain_file=cont, log=QUIET)
    b_el, cell_el = read_strain_force_in(
        os.path.join(wel, "results", "strain_force.in"), 2
    )
    assert cell_el is not None and cell_el.natom == 2
    key = lambda b: (b[0], round(b[1], 8))  # noqa: E731
    d_ifc = {key(b): b for b in b_ifc}
    assert len(b_el) == 12 and set(d_ifc) == {key(b) for b in b_el}
    for b in b_el:
        r = d_ifc[key(b)]
        assert b[2] == r[2] == 0.5 and np.allclose(b[3], r[3], atol=1e-10)
    assert np.abs(b_el[0][3]).max() > 1e-3  # xx: internal relaxation of hcp
    with h5py.File(cont, "r") as f:
        blocks, cell = sf.read_strain_force(f)
        assert len(blocks) == 12 and cell.natom == 2
        assert f["StrainForce"].attrs["source"] == "elastic.py fit"
        assert bool(f["StrainForce"].attrs["central"])
        assert "Elastic" in f
    assert sf.check(cont, anphon_cell=cell_prim, fcs=ref, log=QUIET) == []
    assert any(
        "bit 2 (strain-force coupling)" in ln and ": yes" in ln
        for ln in sf.supported_settings(sf.summary(cont))
    )
