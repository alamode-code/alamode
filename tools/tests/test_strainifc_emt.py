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
    assert any("FC2 vs" in l and "max|dPhi2|" in l for l in logs)
    st = fcsorder.fc2_difference(os.path.join(work, "results", "strain_000.xml"), ref_xml)
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
    blocks = read_strain_force_in(fname, 2)
    assert [b[0] for b in blocks] == ["xx", "yy", "zz", "yz", "zx", "xy"]
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
    b4 = read_strain_force_in(fname, 4)
    assert np.allclose(b4[0][3][:2], fxx) and np.allclose(b4[0][3][2:], fxx)
    # without --fcs: template order
    fname = wi.collect(work, results_dir="results_nofcs", log=QUIET)
    assert np.allclose(read_strain_force_in(fname, 2)[0][3], fxx)


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
    ref_rows = read_strain_force_in(
        os.path.join(root, "force", "results", "strain_force.in"), 2
    )
    rows = read_strain_force_in(fname, 2)
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
