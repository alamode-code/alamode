"""VASP template handling: species blocks, POTCAR symlink/copy, write/read round trip."""
import os

import numpy as np
import pytest

from strainkit import structure


def _write_template(tmp_path, atoms):
    import ase.io
    d = tmp_path / "tmpl"
    d.mkdir()
    ase.io.write(d / "POSCAR", atoms, format="vasp", direct=True, sort=False, vasp5=True)
    (d / "INCAR").write_text("ENCUT = 400\n")
    (d / "KPOINTS").write_text("Automatic mesh\n0\nGamma\n2 2 2\n0 0 0\n")
    (d / "POTCAR").write_text("fake POTCAR\n")
    return d


def test_interleaved_species_rejected(ase_mod, tmp_path):
    from ase.build import bulk
    sc = bulk("NaCl", "rocksalt", a=5.6) * (2, 1, 1)  # Na Cl Na Cl
    assert list(sc.numbers) != sorted(sc.numbers)
    d = _write_template(tmp_path, sc)
    with pytest.raises(ValueError, match="not contiguous"):
        structure.read_template("VASP", str(d))


def test_grouped_template_round_trip(ase_mod, tmp_path):
    from ase.build import bulk
    sc = bulk("NaCl", "rocksalt", a=5.6) * (2, 1, 1)
    sc = sc[np.argsort(sc.numbers, kind="stable")]
    d = _write_template(tmp_path, sc)
    t = structure.read_template("VASP", str(d))
    assert [os.path.basename(f) for f in t.extra_files] == ["INCAR", "KPOINTS", "POTCAR"]
    out = tmp_path / "out"
    p = structure.write_structure(t, t.atoms, str(out))
    assert p.endswith("POSCAR") and os.path.islink(out / "POTCAR") and (out / "INCAR").exists()
    back = ase_mod.io.read(p, format="vasp")
    assert list(back.numbers) == list(sc.numbers)
    assert np.allclose(back.cell[:], sc.cell[:])
    p2 = structure.write_structure(t, t.atoms, str(tmp_path / "out2"), link_potcar=False)
    assert not os.path.islink(tmp_path / "out2" / "POTCAR")


def _fake_potcar(*symbols):
    return "".join(f"  PAW_PBE {sym} 01Jan2000\n   VRHFIN ={sym}: s2p4\n   End of Dataset\n" for sym in symbols)


def test_potcar_order_checked(ase_mod, tmp_path):
    from ase.build import bulk
    sc = bulk("NaCl", "rocksalt", a=5.6) * (2, 1, 1)
    sc = sc[np.argsort(sc.numbers, kind="stable")]  # Na block, then Cl block
    d = _write_template(tmp_path, sc)
    (d / "POTCAR").write_text(_fake_potcar("Cl", "Na"))
    with pytest.raises(ValueError, match="POTCAR species order"):
        structure.read_template("VASP", str(d))
    (d / "POTCAR").write_text(_fake_potcar("Na", "Cl"))
    t = structure.read_template("VASP", str(d))
    assert list(t.atoms.numbers) == list(sc.numbers)
