"""QE pw.in rewriting: CELL_PARAMETERS/ATOMIC_POSITIONS replacement, unit
qualifiers and lattice-parameter cleanup in the namelists."""
import io

import numpy as np
import pytest

from strainkit import structure

TEMPLATE = """&control
 calculation = 'scf', prefix = 'x'
/
&system
 ibrav = 0, A = 3.2
 nat = 2, ntyp = 1
 celldm(1) = 6.0
 ecutwfc = 40
/
&electrons
/
ATOMIC_SPECIES
Cu 63.546 Cu.upf
CELL_PARAMETERS alat
  1.0 0.0 0.0
  -0.5 0.8660254 0.0
  0.0 0.0 1.633
ATOMIC_POSITIONS crystal
Cu 0.0 0.0 0.0 0 0 1
Cu1 0.3333333 0.6666667 0.5
K_POINTS automatic
4 4 4 0 0 0
"""


def _atoms():
    from ase.build import bulk
    return bulk("Cu", "hcp", a=2.55, c=4.2)


def test_replace_qe_blocks(ase_mod):
    import ase.io
    a = _atoms()
    with pytest.warns(UserWarning):
        text = structure.replace_qe_blocks(TEMPLATE, a)
    assert "celldm" not in text and "A = 3.2" not in text and "ibrav = 0" in text
    assert "CELL_PARAMETERS angstrom" in text and "ATOMIC_POSITIONS crystal" in text
    assert "K_POINTS automatic\n4 4 4 0 0 0" in text and "ecutwfc = 40" in text
    # species labels and if_pos flags preserved
    lines = text.splitlines()
    k = [i for i, l in enumerate(lines) if l.startswith("ATOMIC_POSITIONS")][0]
    pos = [l for l in lines[k + 1:] if l.startswith("Cu")]
    assert pos[0].split()[0] == "Cu" and pos[0].split()[4:7] == ["0", "0", "1"]
    assert pos[1].split()[0] == "Cu1"
    back = ase_mod.io.read(io.StringIO(text), format="espresso-in")
    assert np.allclose(back.cell[:], a.cell[:], atol=1e-8)
    d = back.get_scaled_positions() - a.get_scaled_positions()
    assert np.abs(d - np.round(d)).max() < 1e-8


def test_replace_qe_blocks_errors(ase_mod):
    a = _atoms()
    with pytest.raises(ValueError, match="ATOMIC_POSITIONS has"):
        structure.replace_qe_blocks(TEMPLATE, a * (2, 1, 1))
    bad = TEMPLATE.replace("Cu1 0.3333333", "O 0.3333333")
    with pytest.raises(ValueError, match="does not match"):
        structure.replace_qe_blocks(bad, a)
    with pytest.raises(ValueError, match="required"):
        structure.replace_qe_blocks(TEMPLATE.replace("CELL_PARAMETERS alat", "CELL_PARAM_X"), a)


def test_read_template_and_write(ase_mod, tmp_path):
    (tmp_path / "pw.in").write_text(TEMPLATE)
    t = structure.read_template("QE", str(tmp_path))  # no pseudo_dir -> no warning
    assert t.structure_name == "pw.in" and len(t.atoms) == 2
    out = tmp_path / "out"
    with pytest.warns(UserWarning):
        p = structure.write_structure(t, t.atoms, str(out))
    assert p.endswith("pw.in")
    with pytest.raises(ValueError, match="ibrav"):
        (tmp_path / "bad").mkdir()
        (tmp_path / "bad" / "pw.in").write_text(TEMPLATE.replace("ibrav = 0", "ibrav = 4"))
        structure.read_template("QE", str(tmp_path / "bad"))
