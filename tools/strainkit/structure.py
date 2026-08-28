# Copyright (c) 2023 Ryota Masuki (strainIFCcoupling,
#                    https://github.com/r-masuki/strainIFCcoupling)
# Copyright (c) 2026 Terumasa Tadano
# MIT license.  See LICENCE.txt of the ALAMODE package.
"""Structure templates and per-code structure writers (ase based).

The atom order of the template is preserved in every file written here and
verified by re-reading the written file.
"""

import os
import re
import shutil
import warnings
from dataclasses import dataclass, field

import numpy as np

CODES = ("VASP", "QE", "ase")
STRUCTURE_FILENAME = {"VASP": "POSCAR", "QE": "pw.in", "ase": "input.extxyz"}
OUTPUT_FILENAME = {"VASP": "vasprun.xml", "QE": "pw.out", "ase": "output.extxyz"}
_ALIASES = {"vasp": "VASP", "qe": "QE", "espresso": "QE", "quantum-espresso": "QE",
            "quantumespresso": "QE", "ase": "ase", "extxyz": "ase"}


def normalize_code(code):
    key = str(code).strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"unsupported code {code!r}; choose one of {', '.join(CODES)}")
    return _ALIASES[key]


@dataclass
class Template:
    code: str
    atoms: object  # ase.Atoms
    template_dir: str
    structure_file: str
    extra_files: list = field(default_factory=list)
    raw_text: str = None

    @property
    def structure_name(self):
        return os.path.basename(self.structure_file)


def _ase_read(path, fmt):
    import ase.io
    return ase.io.read(path, format=fmt)


def read_template(code, template_dir, structure_file=None):
    """Read the reference structure (and remember the auxiliary input files)."""
    code = normalize_code(code)
    template_dir = os.path.abspath(template_dir)
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(f"template directory {template_dir} does not exist")
    sfile = os.path.join(template_dir, structure_file or STRUCTURE_FILENAME[code])
    if not os.path.isfile(sfile):
        raise FileNotFoundError(f"structure file {sfile} not found")
    raw = None
    extra = []
    if code == "VASP":
        atoms = _ase_read(sfile, "vasp")
        for name in sorted(os.listdir(template_dir)):
            p = os.path.join(template_dir, name)
            if os.path.isfile(p) and os.path.abspath(p) != os.path.abspath(sfile) \
                    and not name.startswith("."):
                extra.append(p)
    elif code == "QE":
        with open(sfile) as f:
            raw = f.read()
        m = re.search(r"ibrav\s*=\s*([-+]?\d+)", raw, flags=re.IGNORECASE)
        if m is None or int(m.group(1)) != 0:
            raise ValueError(f"{sfile}: ibrav must be 0 (CELL_PARAMETERS card required)")
        if not re.search(r"^\s*CELL_PARAMETERS", raw, flags=re.IGNORECASE | re.MULTILINE):
            raise ValueError(f"{sfile}: CELL_PARAMETERS card not found")
        atoms = _ase_read(sfile, "espresso-in")
        pd = re.search(r"pseudo_dir\s*=\s*['\"]([^'\"]+)['\"]", raw, flags=re.IGNORECASE)
        if pd and not os.path.isabs(pd.group(1)):
            warnings.warn(
                f"pseudo_dir = {pd.group(1)!r} is a relative path; it is copied verbatim "
                "into every generated pw.in and must be valid from those directories")
    else:
        atoms = _ase_read(sfile, None)
    if len(atoms) == 0:
        raise ValueError(f"{sfile}: no atoms found")
    return Template(code, atoms, template_dir, sfile, extra, raw)


# ----------------------------------------------------------------- QE cards
_CARD_RE = re.compile(
    r"^\s*(ATOMIC_SPECIES|ATOMIC_POSITIONS|K_POINTS|CELL_PARAMETERS|CONSTRAINTS|"
    r"OCCUPATIONS|ATOMIC_VELOCITIES|ATOMIC_FORCES|ADDITIONAL_K_POINTS|SOLVENTS|HUBBARD)\b",
    re.IGNORECASE)


def _strip_comment(line):
    for c in ("!", "#"):
        k = line.find(c)
        if k >= 0:
            line = line[:k]
    return line.strip()


def replace_qe_blocks(text, atoms):
    """Return the pw.in text with the CELL_PARAMETERS and ATOMIC_POSITIONS cards
    replaced by the cell / fractional coordinates of ``atoms`` (Angstrom, crystal).

    Everything else is kept verbatim except that ``celldm(1)`` / ``A`` entries
    are removed from the namelists (QE rejects a lattice parameter given twice
    once CELL_PARAMETERS is in angstrom).  Species labels and if_pos flags of the
    original ATOMIC_POSITIONS lines are preserved.
    """
    lines = text.splitlines()
    starts = [i for i, l in enumerate(lines) if _CARD_RE.match(l)]
    if not starts:
        raise ValueError("no cards found in the QE input")
    first_card = starts[0]
    bounds = {}
    for n, i in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        bounds[_CARD_RE.match(lines[i]).group(1).upper()] = (i, end)
    if "CELL_PARAMETERS" not in bounds or "ATOMIC_POSITIONS" not in bounds:
        raise ValueError("CELL_PARAMETERS and ATOMIC_POSITIONS cards are required")

    nat = len(atoms)
    cell = np.asarray(atoms.cell[:], dtype=float)
    xf = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)

    # namelist part: drop celldm(1) / A entries
    head = []
    for l in lines[:first_card]:
        s = _strip_comment(l)
        if re.match(r"^(celldm\(1\)|a)\s*=", s, flags=re.IGNORECASE):
            warnings.warn(f"removed {s!r} from the QE namelist (CELL_PARAMETERS angstrom is written)")
            continue
        head.append(l)

    def card_body(name):
        i, end = bounds[name]
        return [l for l in lines[i + 1:end] if _strip_comment(l)]

    pos_lines = card_body("ATOMIC_POSITIONS")
    if len(pos_lines) != nat:
        raise ValueError(f"ATOMIC_POSITIONS has {len(pos_lines)} entries but the structure has {nat} atoms")
    new_pos = ["ATOMIC_POSITIONS crystal"]
    symbols = atoms.get_chemical_symbols()
    for k, l in enumerate(pos_lines):
        tok = _strip_comment(l).split()
        label = tok[0]
        base = re.sub(r"[^A-Za-z].*$", "", label)
        if base.lower() != symbols[k].lower():
            raise ValueError(f"ATOMIC_POSITIONS line {k + 1}: species {label!r} does not match {symbols[k]!r}")
        flags = tok[4:7] if len(tok) >= 7 else []
        new_pos.append("{:6s} {:20.14f} {:20.14f} {:20.14f}{}".format(
            label, xf[k, 0], xf[k, 1], xf[k, 2], (" " + " ".join(flags)) if flags else ""))
    new_cell = ["CELL_PARAMETERS angstrom"]
    for i in range(3):
        new_cell.append("  {:20.14f} {:20.14f} {:20.14f}".format(*cell[i]))

    out = list(head)
    for n, i in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        name = _CARD_RE.match(lines[i]).group(1).upper()
        if name == "ATOMIC_POSITIONS":
            out.extend(new_pos)
            out.append("")
        elif name == "CELL_PARAMETERS":
            out.extend(new_cell)
            out.append("")
        else:
            out.extend(lines[i:end])
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ writers
def write_structure(template, atoms, dest_dir, copy_extra=True, link_potcar=True):
    """Write ``atoms`` in the format of the template into ``dest_dir``.

    Returns the path of the structure file.  The written file is re-read and
    its species sequence compared with ``atoms`` (atom order guarantee).
    """
    import ase.io

    os.makedirs(dest_dir, exist_ok=True)
    code = template.code
    path = os.path.join(dest_dir, template.structure_name)
    if code == "VASP":
        ase.io.write(path, atoms, format="vasp", direct=True, sort=False, vasp5=True)
        back = ase.io.read(path, format="vasp")
        if copy_extra:
            for src in template.extra_files:
                name = os.path.basename(src)
                dst = os.path.join(dest_dir, name)
                if os.path.lexists(dst):
                    os.remove(dst)
                if name.upper().startswith("POTCAR") and link_potcar:
                    os.symlink(os.path.relpath(src, dest_dir), dst)
                else:
                    shutil.copy(src, dst)
    elif code == "QE":
        text = replace_qe_blocks(template.raw_text, atoms)
        with open(path, "w") as f:
            f.write(text)
        back = ase.io.read(path, format="espresso-in")
    else:
        ase.io.write(path, atoms, format="extxyz")
        back = ase.io.read(path)
    if list(back.numbers) != list(atoms.numbers):
        raise RuntimeError(f"{path}: atom order changed on writing (this is a bug)")
    if np.abs(back.cell[:] - atoms.cell[:]).max() > 1.0e-6:
        raise RuntimeError(f"{path}: cell changed on writing (this is a bug)")
    d = back.get_scaled_positions(wrap=False) - atoms.get_scaled_positions(wrap=False)
    d -= np.round(d)
    if np.abs(d).max() > 1.0e-7:
        raise RuntimeError(f"{path}: positions changed on writing (this is a bug)")
    return path
