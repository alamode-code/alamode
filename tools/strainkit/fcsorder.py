"""Atom ordering and cell relations between the DFT cells, ALM force-constant
files and anphon's primitive cell.

anphon's "primitive cell" order is
  (i)  with a ``&cell`` field: the FCS supercell atoms folded into the &cell
       lattice, keeping the first occurrence in supercell order
       (System::update_primitive_lattice, anphon/system.cpp);
  (ii) h5 file without &cell: the stored /PrimitiveCell;
  (iii) xml file without &cell: not allowed by anphon.
Nothing here reorders atoms; the functions only compute and verify mappings.
"""

import os
import re
from dataclasses import dataclass

import numpy as np

from .units import BOHR_IN_ANGSTROM

TOL_COORD = 1.0e-4  # anphon: System::tolerance_for_coordinates (fractional), system.cpp


@dataclass
class FcsStructure:
    path: str
    fmt: str  # "xml" | "h5"
    lavec: np.ndarray  # (3,3) Angstrom, rows = lattice vectors
    xf: np.ndarray  # (nat, 3)
    numbers: np.ndarray  # (nat,)
    elements: list  # element symbol per atom
    map_p2s: np.ndarray  # (ntran, natmin) supercell indices (0-based)
    map_s2p: np.ndarray  # (nat,) primitive atom index of each supercell atom
    prim_lavec: np.ndarray = None  # h5 only (Angstrom, rows)
    prim_xf: np.ndarray = None
    prim_numbers: np.ndarray = None
    prim_elements: list = None

    @property
    def nat(self):
        return len(self.numbers)

    @property
    def natmin(self):
        return self.map_p2s.shape[1]

    @property
    def ntran(self):
        return self.map_p2s.shape[0]


def _numbers_from_symbols(symbols):
    from ase.data import atomic_numbers

    return np.array([atomic_numbers[s] for s in symbols], dtype=int)


def read_fcs_structure(path):
    """Structure and translation mapping stored in an ALM .xml / .h5 file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xml",):
        return _read_xml(path)
    if ext in (".h5", ".hdf5"):
        return _read_h5(path)
    raise ValueError(f"{path}: unknown force-constant file extension (use .xml or .h5)")


def _read_xml(path):
    from lxml import etree

    try:
        root = etree.parse(path).getroot()
    except etree.XMLSyntaxError:
        root = etree.parse(path, parser=etree.XMLParser(recover=True)).getroot()
    nat = int(root.find("Structure/NumberOfAtoms").text)
    ntran = int(root.find("Symmetry/NumberOfTranslations").text)
    natmin = nat // ntran
    lavec = (
        np.array(
            [
                [
                    float(t)
                    for t in root.find(f"Structure/LatticeVector/a{i}").text.split()
                ]
                for i in (1, 2, 3)
            ]
        )
        * BOHR_IN_ANGSTROM
    )
    xf = np.zeros((nat, 3))
    elements = [None] * nat
    for elem in root.findall("Structure/Position/pos"):
        i = int(elem.get("index")) - 1
        xf[i] = [float(t) for t in elem.text.split()]
        elements[i] = elem.get("element")
    map_p2s = np.zeros((ntran, natmin), dtype=int)
    for m in root.findall("Symmetry/Translations/map"):
        map_p2s[int(m.get("tran")) - 1, int(m.get("atom")) - 1] = int(m.text) - 1
    map_s2p = np.zeros(nat, dtype=int)
    for itran in range(ntran):
        for iat in range(natmin):
            map_s2p[map_p2s[itran, iat]] = iat
    return FcsStructure(
        path,
        "xml",
        lavec,
        xf,
        _numbers_from_symbols(elements),
        elements,
        map_p2s,
        map_s2p,
    )


def _read_h5(path):
    import h5py

    def cell(f, group):
        lv = f[f"/{group}/lattice_vector"][:].astype(float)
        unit = f[f"/{group}/lattice_vector"].attrs.get("unit", "bohr")
        if isinstance(unit, bytes):
            unit = unit.decode()
        if str(unit).lower().startswith("bohr"):
            lv = lv * BOHR_IN_ANGSTROM
        xf = f[f"/{group}/fractional_coordinate"][:].astype(float)
        kinds = f[f"/{group}/atomic_kinds"][:].astype(int)
        elems = [
            e.decode() if isinstance(e, bytes) else str(e)
            for e in f[f"/{group}/elements"][:]
        ]
        elements = [elems[k] for k in kinds]
        return lv, xf, elements

    with h5py.File(path, "r") as f:
        lavec, xf, elements = cell(f, "SuperCell")
        plavec, pxf, pelements = cell(f, "PrimitiveCell")
        map_p2s = f["/SuperCell/mapping_table"][:].astype(int).T
    nat = len(elements)
    map_s2p = np.zeros(nat, dtype=int)
    for itran in range(map_p2s.shape[0]):
        for iat in range(map_p2s.shape[1]):
            map_s2p[map_p2s[itran, iat]] = iat
    return FcsStructure(
        path,
        "h5",
        lavec,
        xf,
        _numbers_from_symbols(elements),
        elements,
        map_p2s,
        map_s2p,
        plavec,
        pxf,
        _numbers_from_symbols(pelements),
        pelements,
    )


# --------------------------------------------------------------- cell files
def read_anphon_cell(path):
    """Lattice vectors (Angstrom, rows) of an anphon input (&cell field) or of any
    ase-readable structure file."""
    with open(path, errors="replace") as f:
        text = f.read()
    m = re.search(r"&cell\s*\n(.*?)\n\s*/", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        vals = []
        for line in m.group(1).splitlines():
            line = line.split("#")[0].split("!")[0].strip()
            if line:
                vals.append([float(t) for t in line.split()])
        if len(vals) != 4 or len(vals[0]) != 1 or any(len(v) != 3 for v in vals[1:]):
            raise ValueError(
                f"{path}: could not parse the &cell field (scale + 3 vectors expected)"
            )
        return np.array(vals[1:]) * vals[0][0] * BOHR_IN_ANGSTROM
    import ase.io

    return np.asarray(ase.io.read(path).cell[:], dtype=float)


def lattice_relation(a_target, a_dft, tol=1.0e-5):
    """Integer relation between two commensurate lattices (rows = vectors).

    Returns ``(M, ratio)`` with either ``a_target = M @ a_dft`` (target cell is
    a supercell of the DFT cell, ``ratio = |det M| >= 1``) or
    ``a_dft = M @ a_target`` (DFT cell is a supercell of the target cell,
    ``ratio = 1/|det M| < 1``); ``ratio`` is always V_target / V_dft.
    Raises ValueError if the lattices are not related by an integer matrix in
    either direction (rotated with respect to each other, or incommensurate).
    """
    at = np.asarray(a_target, dtype=float)
    ad = np.asarray(a_dft, dtype=float)
    for first, second, forward in ((at, ad, True), (ad, at, False)):
        m = first @ np.linalg.inv(second)
        mi = np.round(m).astype(int)
        if (
            np.abs(m - mi).max() <= tol * max(1.0, np.abs(m).max())
            and abs(round(np.linalg.det(mi))) >= 1
        ):
            det = abs(int(round(np.linalg.det(mi))))
            return mi, (float(det) if forward else 1.0 / det)
    m = at @ np.linalg.inv(ad)
    raise ValueError(
        "the target (anphon) cell and the DFT cell are not integer combinations of each other:\n"
        f"M = a_target inv(a_dft) =\n{np.array2string(m, precision=6)}\n"
        "Both cells must describe the same lattice in the same Cartesian frame (same orientation "
        "and lattice constants); rotated settings are not supported."
    )


@dataclass
class AnphonPrimitive:
    lavec: np.ndarray  # Angstrom, rows
    xf: np.ndarray  # (natmin, 3)
    numbers: np.ndarray
    elements: list
    super_indices: (
        np.ndarray
    )  # first-occurrence supercell atom per primitive atom (or None)
    source: str


def fold_supercell(fcs, lavec_cell, tol=TOL_COORD):
    """anphon's System::update_primitive_lattice: fold the FCS supercell into
    ``lavec_cell`` keeping first occurrences in supercell order."""
    xf_p_all = fcs.xf @ fcs.lavec @ np.linalg.inv(lavec_cell)
    ndiv = int(round(abs(np.linalg.det(fcs.lavec) / np.linalg.det(lavec_cell))))
    if ndiv < 1 or fcs.nat % ndiv != 0:
        raise ValueError("the given cell is incommensurate with the FCS supercell")
    natmin = fcs.nat // ndiv
    xf_unique, kinds, idx = [], [], []
    for i in range(fcs.nat):
        x = np.fmod(xf_p_all[i], 1.0)
        x = np.where(x < -1.0e-6, x + 1.0, x)
        dup = False
        for k, xu in enumerate(xf_unique):
            d = np.fmod(x - xu, 1.0)
            d = np.where(d < -0.5, d + 1.0, d)
            d = np.where(d >= 0.5, d - 1.0, d)
            if np.linalg.norm(d) < tol:
                dup = True
                if kinds[k] != fcs.numbers[i]:
                    raise ValueError(
                        "different elements occupy the same site after folding"
                    )
                break
        if not dup:
            xf_unique.append(x)
            kinds.append(int(fcs.numbers[i]))
            idx.append(i)
    if len(xf_unique) != natmin:
        raise ValueError(
            f"folding into the given cell yielded {len(xf_unique)} atoms, expected {natmin}"
        )
    idx = np.array(idx, dtype=int)
    return AnphonPrimitive(
        np.asarray(lavec_cell, dtype=float),
        np.array(xf_unique),
        np.array(kinds),
        [fcs.elements[i] for i in idx],
        idx,
        "FCS supercell folded into the given cell (anphon &cell rule)",
    )


def anphon_primitive_cell(fcs, anphon_cell=None):
    """The primitive cell anphon uses (order included).  See module docstring."""
    if anphon_cell is not None:
        return fold_supercell(fcs, np.asarray(anphon_cell, dtype=float))
    if fcs.fmt == "h5":
        return AnphonPrimitive(
            fcs.prim_lavec,
            fcs.prim_xf,
            fcs.prim_numbers,
            fcs.prim_elements,
            None,
            f"/PrimitiveCell of {os.path.basename(fcs.path)}",
        )
    raise ValueError(
        f"{fcs.path} is an XML force-constant file: anphon requires the &cell field for it, "
        "so the anphon cell must be given (--anphon-cell FILE with the anphon input or a structure file)"
    )


def match_primitive_atoms(atoms_dft, prim, tol=1.0e-3):
    """Map every anphon primitive atom to a DFT-cell atom that is a translation
    image of it.  Returns an integer array of DFT atom indices.

    The two cells must be commensurate (integer relation in either direction).
    When the DFT cell is a supercell of the anphon cell, several DFT atoms are
    translation images of the same anphon atom and carry identical forces under
    homogeneous strain; the first one (lowest index) is used.
    """
    a_dft = np.asarray(atoms_dft.cell[:], dtype=float)
    _, ratio = lattice_relation(prim.lavec, a_dft)  # raises if rotated/incommensurate
    z_dft = np.asarray(atoms_dft.numbers)
    # Two atoms are the same crystallographic site if their difference is a
    # translation of the crystal, i.e. of the SMALLER of the two (commensurate)
    # cells: compare fractional coordinates of that cell modulo 1.
    small = a_dft if ratio >= 1.0 else np.asarray(prim.lavec, dtype=float)
    inv_small = np.linalg.inv(small)
    y = (prim.xf @ prim.lavec) @ inv_small
    x_dft = np.asarray(atoms_dft.get_positions(), dtype=float) @ inv_small
    n_images = max(1, int(round(1.0 / ratio)))  # DFT atoms per anphon atom
    mapping = np.full(len(prim.xf), -1, dtype=int)
    for i in range(len(prim.xf)):
        cand = []
        for j in range(len(x_dft)):
            if z_dft[j] != prim.numbers[i]:
                continue
            d = y[i] - x_dft[j]
            d -= np.round(d)
            if np.linalg.norm(d) < tol:
                cand.append(j)
        if len(cand) != n_images:
            raise ValueError(
                f"anphon primitive atom {i} ({prim.elements[i]}, fractional {np.array2string(prim.xf[i], precision=6)}) "
                f"matches {len(cand)} atoms of the DFT cell (expected {n_images}); the DFT cell and the "
                "anphon cell must describe the same crystal in the same Cartesian frame"
            )
        mapping[i] = cand[0]
    return mapping


def check_supercell_equivalence(atoms, fcs, tol_cell=1.0e-4, tol_frac=1.0e-5):
    """Index-wise comparison of an (undeformed) ase structure with the FCS supercell."""
    problems = []
    a = np.asarray(atoms.cell[:], dtype=float)
    if np.abs(a - fcs.lavec).max() > tol_cell:
        problems.append(
            f"lattice vectors differ by up to {np.abs(a - fcs.lavec).max():.3e} A"
        )
    if len(atoms) != fcs.nat:
        problems.append(f"{len(atoms)} atoms vs {fcs.nat} in the FCS file")
    else:
        if not np.array_equal(np.asarray(atoms.numbers), fcs.numbers):
            bad = np.nonzero(np.asarray(atoms.numbers) != fcs.numbers)[0]
            problems.append(
                f"species differ at atom index {bad[:10].tolist()} (0-based)"
            )
        d = np.asarray(atoms.get_scaled_positions(wrap=False)) - fcs.xf
        d -= np.round(d)
        if np.abs(d).max() > tol_frac:
            bad = np.nonzero(np.abs(d).max(axis=1) > tol_frac)[0]
            problems.append(
                f"fractional positions differ (max {np.abs(d).max():.3e}) at atom index {bad[:10].tolist()}"
            )
    if problems:
        raise ValueError(
            f"the template structure is not the supercell of {fcs.path} (same atom order required): "
            + "; ".join(problems)
        )


def verify_generated_fcs(path, fcs_ref, F=None, tol_cell=1.0e-4, tol_frac=1.0e-5):
    """Verify that an FC file written for a (deformed) supercell has exactly the
    same atom order and translation tables as the reference FC file."""
    g = read_fcs_structure(path)
    problems = []
    if g.nat != fcs_ref.nat:
        problems.append(f"{g.nat} atoms vs {fcs_ref.nat}")
    else:
        if not np.array_equal(g.numbers, fcs_ref.numbers):
            problems.append("species order differs")
        d = g.xf - fcs_ref.xf
        d -= np.round(d)
        if np.abs(d).max() > tol_frac:
            problems.append(
                f"fractional positions differ by up to {np.abs(d).max():.3e}"
            )
        if g.map_p2s.shape != fcs_ref.map_p2s.shape or not np.array_equal(
            g.map_p2s, fcs_ref.map_p2s
        ):
            problems.append("translation mapping table (map_p2s) differs")
        if not np.array_equal(g.map_s2p, fcs_ref.map_s2p):
            problems.append("supercell-to-primitive mapping (map_s2p) differs")
    if F is not None:
        expected = fcs_ref.lavec @ np.asarray(F, dtype=float).T
        if np.abs(g.lavec - expected).max() > tol_cell:
            problems.append(
                f"lattice differs from F applied to the reference by {np.abs(g.lavec - expected).max():.3e} A"
            )
    if g.fmt == "h5" and fcs_ref.fmt == "h5":
        if g.prim_xf.shape != fcs_ref.prim_xf.shape or not np.array_equal(
            g.prim_numbers, fcs_ref.prim_numbers
        ):
            problems.append("/PrimitiveCell (atoms/species) differs")
        else:
            dp = g.prim_xf - fcs_ref.prim_xf
            dp -= np.round(dp)
            if np.abs(dp).max() > tol_frac:
                problems.append(
                    f"/PrimitiveCell fractional coordinates differ by up to {np.abs(dp).max():.3e}"
                )
            expected_p = (
                fcs_ref.prim_lavec
                if F is None
                else fcs_ref.prim_lavec @ np.asarray(F, dtype=float).T
            )
            if np.abs(g.prim_lavec - expected_p).max() > tol_cell:
                problems.append(
                    "/PrimitiveCell lattice differs from the (deformed) reference"
                )
    if problems:
        raise ValueError(
            f"{path}: generated force-constant file is not index-compatible with "
            f"{fcs_ref.path}: " + "; ".join(problems)
        )
    return g


def describe_ordering(prim, mapping=None, atoms_dft=None):
    """Text table of anphon's primitive order and the DFT-cell mapping."""
    lines = [f"anphon primitive cell ({prim.source}): {len(prim.xf)} atoms"]
    for i in range(len(prim.xf)):
        s = f"  {i + 1:4d} {prim.elements[i]:>3s}  " + " ".join(
            f"{x:12.8f}" for x in prim.xf[i]
        )
        if prim.super_indices is not None:
            s += f"   <- FCS supercell atom {prim.super_indices[i] + 1}"
        if mapping is not None:
            s += f"   = DFT-cell atom {mapping[i] + 1}"
        lines.append(s)
    if mapping is not None:
        n = len(mapping)
        if atoms_dft is not None and n == len(atoms_dft):
            ident = np.array_equal(mapping, np.arange(n))
            lines.append(
                "  mapping is the identity"
                if ident
                else "  WARNING: mapping is a permutation of the DFT cell (not the identity)"
            )
        elif atoms_dft is not None and n > len(atoms_dft):
            lines.append(
                f"  anphon cell contains {n // len(atoms_dft)} DFT cells (rows will be tiled)"
            )
        elif atoms_dft is not None:
            lines.append(
                f"  the DFT cell contains {len(atoms_dft) // n} anphon cells (one image per anphon atom is used)"
            )
    return "\n".join(lines)


# ------------------------------------------------------------ FC2 comparison
def read_fc2_table(path):
    """Harmonic force constants of an ALM file as {key: value (Ry/bohr^2)}.

    XML keys are the (pair1, pair2) attribute strings as stored; h5 keys are
    (atom_indices_supercell, coord_indices, shift) tuples.  Keys are only
    comparable between files of the same format and the same cell.
    """
    ext = os.path.splitext(path)[1].lower()
    table = {}
    if ext == ".xml":
        from lxml import etree

        try:
            root = etree.parse(path).getroot()
        except etree.XMLSyntaxError:
            root = etree.parse(path, parser=etree.XMLParser(recover=True)).getroot()
        for fc in root.findall("ForceConstants/HARMONIC/FC2"):
            table[(fc.get("pair1"), fc.get("pair2"))] = float(fc.text)
        return table, "xml"
    if ext in (".h5", ".hdf5"):
        import h5py

        with h5py.File(path, "r") as f:
            g = f["/ForceConstants/Order2"]
            n = len(g["force_constant_values"])
            at = g["atom_indices_supercell"][:].reshape(n, -1)
            cd = g["coord_indices"][:].reshape(n, -1)
            sh = np.round(g["shift_vectors"][:].reshape(n, -1) * 1.0e4).astype(int)
            val = g["force_constant_values"][:]
        for k in range(n):
            table[(tuple(at[k]), tuple(cd[k]), tuple(sh[k]))] = float(val[k])
        return table, "h5"
    raise ValueError(f"{path}: unknown force-constant file extension")


def fc2_difference(path_a, path_b):
    """Statistics of Phi2(a) - Phi2(b) over the common entries of two FC files."""
    ta, fa = read_fc2_table(path_a)
    tb, fb = read_fc2_table(path_b)
    if fa != fb:
        raise ValueError(f"cannot compare {fa} with {fb} force-constant files")
    common = [k for k in ta if k in tb]
    if not common:
        raise ValueError("no common force-constant entries (different supercell or indexing)")
    d = np.array([ta[k] - tb[k] for k in common])
    ref = np.array([tb[k] for k in common])
    return {
        "n_common": len(common),
        "n_only_a": len(ta) - len(common),
        "n_only_b": len(tb) - len(common),
        "max_abs": float(np.abs(d).max()),
        "rms": float(np.sqrt(np.mean(d**2))),
        "rms_ref": float(np.sqrt(np.mean(ref**2))),
    }
