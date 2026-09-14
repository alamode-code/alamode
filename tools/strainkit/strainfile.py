"""The strain-coupling container: one HDF5 file (schema ``alamode:strain_coupling``)
holding everything anphon needs for the cell relaxation (``STRAINFILE``).

Layout (every group except /ReferenceCell is optional)::

    /                    attrs schema, format_version, created_date, strainkit_version, provenance (JSON)
    /ReferenceCell/      cell group (alm force-constant layout): the reference structure
    /Elastic/            stress (3,3), soec (9,9), toec (9,9,9)  [GPa]
    /StrainForce/        modes, smag, weight, displacement_gradient, forces [n, natom, 3] (eV/A), Cell/
    /StrainHarmonic/     modes, smag, weight, displacement_gradient, entry_NNN/{PrimitiveCell, SuperCell,
                         ForceConstants/Order2}  (verbatim alm force-constant layout)

Cell groups store the lattice vectors as rows in bohr (the alm convention),
fractional coordinates, 0-based ``atomic_kinds`` into ``elements``.  The
readers of anphon (anphon/strain_coupling_io.cpp) are the counterpart of the
writers here; both are kept byte-compatible with the files alm writes.
"""

import contextlib
import datetime
import json
import os
import shutil
import sys

import numpy as np

from . import __version__ as STRAINKIT_VERSION
from .strain import MODE_NAMES, mode_tensor
from .units import BOHR_IN_ANGSTROM, EV_PER_ANG3_TO_GPA, legacy_ry_per_cell_to_gpa
from .writers import ReferenceCell

SCHEMA = "alamode:strain_coupling"
FORMAT_VERSION = 1
REFERENCE = "ReferenceCell"
ELASTIC = "Elastic"
FORCE = "StrainForce"
HARMONIC = "StrainHarmonic"
UNIT_FORCE = "eV/angstrom"


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the strain-coupling container needs the h5py package"
        ) from exc
    return h5py


def _decode(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def _plain(v):
    """numpy scalars/arrays -> plain Python for h5 attributes and JSON."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _set_attrs(g, attrs):
    import math

    for k, v in (attrs or {}).items():
        if v is None:
            continue
        v = _plain(v)
        if isinstance(v, float) and math.isnan(v):
            continue
        if isinstance(v, bool):
            v = np.int32(v)
        elif isinstance(v, (dict, list, tuple)):
            v = json.dumps(v)
        g.attrs[k] = v


# ------------------------------------------------------------------ cells
def write_cell_group(g, lavec_rows_ang, xf, symbols, source=None, mapping_table=None):
    """Write a cell group in the alm layout.

    ``lavec_rows_ang``: (3, 3) lattice vectors as rows in Angstrom (stored as
    rows in bohr, the transpose of anphon's column convention, which anphon
    transposes back).  ``symbols``: element symbol per atom.  ``mapping_table``
    (natmin, ntran): the translation table of a supercell; a descriptive cell
    (the default) is its own primitive cell.
    """
    h5py = _h5py()
    lavec = np.asarray(lavec_rows_ang, dtype=float).reshape(3, 3)
    xf = np.asarray(xf, dtype=float).reshape(-1, 3)
    symbols = [str(s) for s in symbols]
    natom = len(symbols)
    if xf.shape[0] != natom or natom == 0:
        raise ValueError(
            "cell group: one symbol per atom of fractional_coordinate is required"
        )
    if not (np.all(np.isfinite(lavec)) and np.all(np.isfinite(xf))):
        raise ValueError("cell group: non-finite lattice or coordinates")
    if abs(np.linalg.det(lavec)) < 1.0e-8:
        raise ValueError("cell group: singular lattice")
    elements, kinds = [], []
    for s in symbols:
        if s not in elements:
            elements.append(s)
        kinds.append(elements.index(s))
    d = g.create_dataset("lattice_vector", data=lavec / BOHR_IN_ANGSTROM)
    d.attrs["unit"] = "bohr"
    g.create_dataset("number_of_atoms", data=np.uint64(natom))
    g.create_dataset("number_of_elements", data=np.uint64(len(elements)))
    g.create_dataset("fractional_coordinate", data=xf)
    g.create_dataset("atomic_kinds", data=np.asarray(kinds, dtype=np.int32))
    g.create_dataset(
        "elements", data=np.array(elements, dtype=object), dtype=h5py.string_dtype()
    )
    g.create_dataset("spin_polarized", data=np.int32(0))
    if mapping_table is None:
        mapping_table = np.arange(natom, dtype=np.int32).reshape(natom, 1)
    mapping_table = np.asarray(mapping_table, dtype=np.int32)
    g.create_dataset(
        "number_of_primitive_translations", data=np.uint64(mapping_table.shape[1])
    )
    g.create_dataset("mapping_table", data=mapping_table)
    if source:
        g.attrs["source"] = str(source)


def read_cell_group(g):
    """The cell group as a :class:`strainkit.writers.ReferenceCell` (lattice rows in Angstrom)."""
    lv = np.asarray(g["lattice_vector"][()], dtype=float).reshape(3, 3)
    unit = _decode(g["lattice_vector"].attrs.get("unit", "bohr")).lower()
    if unit.startswith("bohr"):
        lv = lv * BOHR_IN_ANGSTROM
    elif not unit.startswith("angstrom"):
        raise ValueError(f"{g.name}/lattice_vector: unsupported unit {unit!r}")
    xf = np.asarray(g["fractional_coordinate"][()], dtype=float).reshape(-1, 3)
    kinds = np.asarray(g["atomic_kinds"][()], dtype=int)
    elements = [_decode(e) for e in g["elements"][()]]
    if len(kinds) != len(xf) or kinds.min() < 0 or kinds.max() >= len(elements):
        raise ValueError(f"{g.name}: inconsistent atomic_kinds / elements")
    return ReferenceCell(lv, [elements[k] for k in kinds], xf)


def cell_from_atoms(atoms):
    """ReferenceCell of an ase Atoms object."""
    return ReferenceCell(
        np.asarray(atoms.cell[:], dtype=float),
        [str(s) for s in atoms.get_chemical_symbols()],
        np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float),
    )


def cell_from_primitive(prim):
    """ReferenceCell of a :class:`strainkit.fcsorder.AnphonPrimitive`."""
    return ReferenceCell(prim.lavec, prim.elements, prim.xf)


def same_crystal(a, b):
    """V_a / V_b when the two ReferenceCells describe the same crystal (nested cells), else ValueError."""
    from .fcsorder import same_crystal as _same

    return _same(a.lavec, a.xf, a.elements, b.lavec, b.xf, b.elements)


# --------------------------------------------------------------- root/update
def _now():
    return datetime.datetime.now().strftime("%Y-%b-%d %H:%M:%S")


def stamp_root(f, provenance=None):
    f.attrs["schema"] = SCHEMA
    f.attrs["format_version"] = np.int32(FORMAT_VERSION)
    f.attrs["created_date"] = _now()
    f.attrs["strainkit_version"] = STRAINKIT_VERSION
    f.attrs["provenance"] = json.dumps(provenance or [])


def check_schema(f):
    if "schema" not in f.attrs or _decode(f.attrs["schema"]) != SCHEMA:
        raise ValueError(
            f"{f.filename}: not a strain-coupling container (schema {SCHEMA} expected)"
        )
    version = int(f.attrs.get("format_version", 0))
    if version > FORMAT_VERSION:
        raise ValueError(
            f"{f.filename}: format_version {version} is newer than this strainkit ({FORMAT_VERSION})"
        )
    if REFERENCE not in f:
        raise ValueError(f"{f.filename}: /{REFERENCE} is missing")
    return version


def provenance_record(group, **extra):
    rec = {
        "group": group,
        "date": _now(),
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "strainkit_version": STRAINKIT_VERSION,
    }
    rec.update({k: _plain(v) for k, v in extra.items() if v is not None})
    return rec


def _append_provenance(f, record):
    try:
        prov = json.loads(_decode(f.attrs.get("provenance", "[]")))
    except json.JSONDecodeError:
        prov = []
    if not isinstance(prov, list):
        prov = [prov]
    prov.append(record)
    f.attrs["provenance"] = json.dumps(prov)


@contextlib.contextmanager
def update(path, reference_cell, record=None, force=False, source=None):
    """Create or update a container transactionally.

    The work is done on ``path.part``, which replaces ``path`` only when the
    block completes; an existing file is copied first and must carry the same
    crystal in /ReferenceCell (nested cells accepted), otherwise ValueError is
    raised.  ``force=True`` recreates the whole file from scratch with the new
    reference cell (all groups of the old file are dropped, and listed).
    Yields the open h5py.File.
    """
    h5py = _h5py()
    part = path + ".part"
    dropped = []
    if os.path.exists(path) and not force:
        shutil.copyfile(path, part)
        f = h5py.File(part, "r+")
        try:
            check_schema(f)
            stored = read_cell_group(f[REFERENCE])
            try:
                same_crystal(stored, reference_cell)
            except ValueError as exc:
                raise ValueError(
                    f"{path}: /ReferenceCell describes a different crystal than the data being added "
                    f"({exc}). Use --force to start a new container (its existing groups are discarded)."
                ) from None
        except Exception:
            f.close()
            os.remove(part)
            raise
    else:
        if os.path.exists(path) and force:
            with h5py.File(path, "r") as old:
                dropped = [k for k in (ELASTIC, FORCE, HARMONIC) if k in old]
        f = h5py.File(part, "w")
        stamp_root(f)
        write_cell_group(
            f.create_group(REFERENCE),
            reference_cell.lavec,
            reference_cell.xf,
            reference_cell.elements,
            source=source,
        )
    try:
        yield f
        if record is not None:
            _append_provenance(f, record)
        f.close()
        os.replace(part, path)
    except Exception:
        f.close()
        if os.path.exists(part):
            os.remove(part)
        raise
    if dropped:
        print(
            f"  NOTE: {path} was recreated; the groups {', '.join('/' + d for d in dropped)} of the old file were dropped"
        )


# ------------------------------------------------------------------ modes
def _validate_modes(modes, smag, weight, where):
    modes = [str(m) for m in modes]
    smag = np.asarray(smag, dtype=float)
    weight = np.asarray(weight, dtype=float)
    n = len(modes)
    if n == 0 or smag.shape != (n,) or weight.shape != (n,):
        raise ValueError(
            f"{where}: modes, smag and weight must be non-empty and of equal length"
        )
    for m, s, w in zip(modes, smag, weight):
        if m not in MODE_NAMES:
            raise ValueError(f"{where}: invalid strain mode {m!r}")
        if not np.isfinite(s) or s == 0.0 or not np.isfinite(w):
            raise ValueError(f"{where}: invalid smag/weight for mode {m}")
    return modes, smag, weight


def _write_mode_table(g, modes, smag, weight):
    h5py = _h5py()
    g.create_dataset(
        "modes", data=np.array(modes, dtype=object), dtype=h5py.string_dtype()
    )
    g.create_dataset("smag", data=np.asarray(smag, dtype=float))
    g.create_dataset("weight", data=np.asarray(weight, dtype=float))
    g.create_dataset(
        "displacement_gradient",
        data=np.array(
            [mode_tensor(m) * s for m, s in zip(modes, smag)], dtype=float
        ).reshape(len(modes), 3, 3),
    )


def _read_mode_table(g):
    modes = [_decode(m) for m in g["modes"][()]]
    smag = np.asarray(g["smag"][()], dtype=float)
    weight = np.asarray(g["weight"][()], dtype=float)
    return _validate_modes(modes, smag, weight, g.name)


def weight_sum_matrix(modes, weights):
    """anphon's 3x3 weight-sum matrix of a mode list."""
    from .strain import mode_pair

    w = np.zeros((3, 3))
    for m, wt in zip(modes, weights):
        i, j = mode_pair(m)
        w[i, j] += wt
        if i != j:
            w[j, i] += wt
    return w


# ---------------------------------------------------------------- elastic
def write_elastic(f, stress_gpa=None, soec_gpa=None, toec_gpa=None, attrs=None):
    """Replace /Elastic.  stress (3,3), soec (9,9), toec (9,9,9) in GPa; soec and
    toec come together; every argument may be None."""
    if (soec_gpa is None) != (toec_gpa is None):
        raise ValueError("/Elastic: soec and toec must be written together")
    if stress_gpa is None and soec_gpa is None:
        raise ValueError("/Elastic: nothing to write")
    if ELASTIC in f:
        del f[ELASTIC]
    g = f.create_group(ELASTIC)
    if stress_gpa is not None:
        s = np.asarray(stress_gpa, dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(s)):
            raise ValueError("/Elastic/stress: non-finite values")
        d = g.create_dataset("stress", data=s)
        d.attrs["unit"] = "GPa"
    if soec_gpa is not None:
        c2 = np.asarray(soec_gpa, dtype=float).reshape(9, 9)
        c3 = np.asarray(toec_gpa, dtype=float).reshape(9, 9, 9)
        if not (np.all(np.isfinite(c2)) and np.all(np.isfinite(c3))):
            raise ValueError("/Elastic: non-finite elastic constants")
        d = g.create_dataset("soec", data=c2)
        d.attrs["unit"] = "GPa"
        d = g.create_dataset("toec", data=c3)
        d.attrs["unit"] = "GPa"
    _set_attrs(g, attrs)
    return g


def read_elastic(f):
    """(stress (3,3) or None, soec (9,9) or None, toec (9,9,9) or None, attrs) in GPa."""
    if ELASTIC not in f:
        raise ValueError(f"{f.filename}: no /Elastic group")
    g = f[ELASTIC]
    stress = soec = toec = None
    if "stress" in g:
        _require_unit(g["stress"], "GPa")
        stress = np.asarray(g["stress"][()], dtype=float).reshape(3, 3)
    if "soec" in g or "toec" in g:
        if "soec" not in g or "toec" not in g:
            raise ValueError(
                f"{f.filename}: /Elastic must contain both soec and toec or neither"
            )
        _require_unit(g["soec"], "GPa")
        _require_unit(g["toec"], "GPa")
        soec = np.asarray(g["soec"][()], dtype=float).reshape(9, 9)
        toec = np.asarray(g["toec"][()], dtype=float).reshape(9, 9, 9)
    return stress, soec, toec, {k: _plain(v) for k, v in g.attrs.items()}


def _require_unit(dset, expected):
    unit = _decode(dset.attrs.get("unit", ""))
    if unit.lower() != expected.lower():
        raise ValueError(f"{dset.name}: unit {unit!r}, expected {expected!r}")


# ----------------------------------------------------------- strain force
def write_strain_force(f, blocks, cell, attrs=None):
    """Replace /StrainForce.  blocks: (mode, smag, weight, forces (natom, 3) eV/A);
    cell: the ReferenceCell whose atoms the rows follow."""
    modes, smag, weight = _validate_modes(
        [b[0] for b in blocks],
        [b[1] for b in blocks],
        [b[2] for b in blocks],
        "/StrainForce",
    )
    natom = cell.natom
    forces = np.zeros((len(blocks), natom, 3))
    for k, b in enumerate(blocks):
        fk = np.asarray(b[3], dtype=float)
        if fk.shape != (natom, 3) or not np.all(np.isfinite(fk)):
            raise ValueError(
                f"/StrainForce: block {k + 1} must hold finite forces of shape ({natom}, 3)"
            )
        forces[k] = fk
    if FORCE in f:
        del f[FORCE]
    g = f.create_group(FORCE)
    _write_mode_table(g, modes, smag, weight)
    d = g.create_dataset("forces", data=forces)
    d.attrs["unit"] = UNIT_FORCE
    write_cell_group(g.create_group("Cell"), cell.lavec, cell.xf, cell.elements)
    _set_attrs(g, attrs)
    return g


def read_strain_force(f):
    """(blocks, cell) as written by :func:`write_strain_force`."""
    if FORCE not in f:
        raise ValueError(f"{f.filename}: no /StrainForce group")
    g = f[FORCE]
    modes, smag, weight = _read_mode_table(g)
    cell = read_cell_group(g["Cell"])
    _require_unit(g["forces"], UNIT_FORCE)
    forces = np.asarray(g["forces"][()], dtype=float)
    if forces.shape != (len(modes), cell.natom, 3):
        raise ValueError(
            f"{f.filename}: /StrainForce/forces has shape {forces.shape}, expected "
            f"({len(modes)}, {cell.natom}, 3)"
        )
    blocks = [
        (m, float(s), float(w), forces[k])
        for k, (m, s, w) in enumerate(zip(modes, smag, weight))
    ]
    return blocks, cell


# -------------------------------------------------------- strain harmonic
def _image_shifts():
    """The 27 lattice images in alm's order (index 0 = no shift)."""
    import itertools

    shifts = [(0, 0, 0)]
    for s in itertools.product((-1, 0, 1), repeat=3):
        if s != (0, 0, 0):
            shifts.append(s)
    return np.array(shifts, dtype=float)


def xml_to_fc2_group(xml_path, eg):
    """Write the SuperCell and ForceConstants/Order2 groups of an alm xml file into
    the entry group ``eg`` in the alm h5 layout (one row per xml element).

    /PrimitiveCell is not written (the xml does not store the primitive
    lattice); anphon does not need it for the strained entries.
    """
    from lxml import etree

    from .fcsorder import _read_xml

    st = _read_xml(xml_path)
    try:
        root = etree.parse(xml_path).getroot()
    except etree.XMLSyntaxError:
        root = etree.parse(xml_path, parser=etree.XMLParser(recover=True)).getroot()
    fcs = root.findall("ForceConstants/HARMONIC/FC2")
    if not fcs:
        raise ValueError(
            f"{xml_path}: no harmonic force constants (ForceConstants/HARMONIC/FC2)"
        )
    lat_bohr = st.lavec / BOHR_IN_ANGSTROM  # rows
    xc = st.xf @ lat_bohr
    images = _image_shifts() @ lat_bohr
    n = len(fcs)
    atom_indices = np.zeros((n, 2), dtype=np.int32)
    atom_indices_super = np.zeros((n, 2), dtype=np.int32)
    coord_indices = np.zeros((n, 2), dtype=np.int32)
    shift = np.zeros((n, 3))
    values = np.zeros(n)
    first_image = st.map_p2s[0]
    for i, fc in enumerate(fcs):
        p1, xyz1 = [int(t) - 1 for t in fc.get("pair1").split()]
        a2, xyz2, icell = [int(t) - 1 for t in fc.get("pair2").split()]
        if not (0 <= p1 < st.natmin and 0 <= a2 < st.nat and 0 <= icell < 27):
            raise ValueError(f"{xml_path}: FC2 entry {i + 1} has out-of-range indices")
        a1 = first_image[p1]
        atom_indices[i] = (p1, st.map_s2p[a2])
        atom_indices_super[i] = (a1, a2)
        coord_indices[i] = (xyz1, xyz2)
        shift[i] = xc[a2] + images[icell] - xc[a1]
        values[i] = float(fc.text)
    write_cell_group(
        eg.create_group("SuperCell"),
        st.lavec,
        st.xf,
        st.elements,
        mapping_table=st.map_p2s.T,
    )
    g = eg.create_group("ForceConstants/Order2")
    opts = dict(compression="gzip", compression_opts=4)
    g.create_dataset("atom_indices", data=atom_indices, **opts)
    g.create_dataset("atom_indices_supercell", data=atom_indices_super, **opts)
    g.create_dataset("coord_indices", data=coord_indices, **opts)
    d = g.create_dataset("shift_vectors", data=shift, **opts)
    d.attrs["unit"] = "bohr"
    d.attrs["basis"] = "Cartesian"
    d = g.create_dataset("force_constant_values", data=values, **opts)
    d.attrs["unit"] = "Ry/bohr^2"


def _copy_fc_h5(src_path, eg):
    h5py = _h5py()
    with h5py.File(src_path, "r") as src:
        for name in ("SuperCell", "ForceConstants"):
            if name not in src:
                raise ValueError(
                    f"{src_path}: no /{name} group (an alm force-constant file is expected)"
                )
        if "ForceConstants/Order2" not in src:
            raise ValueError(
                f"{src_path}: no harmonic force constants (/ForceConstants/Order2)"
            )
        for name in ("PrimitiveCell", "SuperCell", "ForceConstants"):
            if name in src:
                src.copy(src[name], eg, name=name)


def write_strain_harmonic(f, rows, fc_paths, attrs=None):
    """Replace /StrainHarmonic.  rows: (mode, smag, weight); fc_paths: the
    force-constant file (.h5 copied group-wise, .xml converted) of every row."""
    if len(rows) != len(fc_paths):
        raise ValueError("/StrainHarmonic: one force-constant file per row is required")
    modes, smag, weight = _validate_modes(
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        "/StrainHarmonic",
    )
    if HARMONIC in f:
        del f[HARMONIC]
    g = f.create_group(HARMONIC)
    _write_mode_table(g, modes, smag, weight)
    for k, (m, s, w, path) in enumerate(zip(modes, smag, weight, fc_paths), start=1):
        eg = g.create_group(f"entry_{k:03d}")
        eg.attrs["mode"] = m
        eg.attrs["smag"] = float(s)
        eg.attrs["weight"] = float(w)
        eg.attrs["source_file"] = os.path.basename(str(path))
        ext = os.path.splitext(str(path))[1].lower()
        if ext in (".h5", ".hdf5"):
            _copy_fc_h5(path, eg)
        elif ext == ".xml":
            xml_to_fc2_group(path, eg)
        else:
            raise ValueError(
                f"{path}: unknown force-constant file extension (.xml or .h5)"
            )
    _set_attrs(g, attrs)
    return g


def read_strain_harmonic(f):
    """(rows (mode, smag, weight), entry group names) of /StrainHarmonic."""
    if HARMONIC not in f:
        raise ValueError(f"{f.filename}: no /StrainHarmonic group")
    g = f[HARMONIC]
    modes, smag, weight = _read_mode_table(g)
    entries = []
    for k in range(1, len(modes) + 1):
        name = f"entry_{k:03d}"
        if name not in g:
            raise ValueError(f"{f.filename}: /StrainHarmonic/{name} is missing")
        entries.append(g[name].name)
    return [(m, float(s), float(w)) for m, s, w in zip(modes, smag, weight)], entries


# ---------------------------------------------------------------- summary
def summary(path):
    """Inventory of a container as a dict (used by show/check)."""
    h5py = _h5py()
    out = {"path": path}
    with h5py.File(path, "r") as f:
        out["format_version"] = check_schema(f)
        out["attrs"] = {
            k: _decode(v) if isinstance(v, (bytes, str)) else _plain(v)
            for k, v in f.attrs.items()
        }
        out["reference_cell"] = read_cell_group(f[REFERENCE])
        out["reference_source"] = _decode(f[REFERENCE].attrs.get("source", ""))
        out["elastic"] = read_elastic(f) if ELASTIC in f else None
        out["strain_force"] = read_strain_force(f) if FORCE in f else None
        out["strain_force_attrs"] = (
            {k: _plain(v) for k, v in f[FORCE].attrs.items()} if FORCE in f else {}
        )
        if HARMONIC in f:
            rows, entries = read_strain_harmonic(f)
            cells = [read_cell_group(f[e]["SuperCell"]) for e in entries]
            nrows = [
                int(f[e]["ForceConstants/Order2/force_constant_values"].shape[0])
                for e in entries
            ]
            out["strain_harmonic"] = (rows, entries, cells, nrows)
            out["strain_harmonic_attrs"] = {
                k: _plain(v) for k, v in f[HARMONIC].attrs.items()
            }
        else:
            out["strain_harmonic"] = None
    return out


def supported_settings(info):
    """Which anphon settings the container supports, as text lines: one per
    bit of STRAIN_COUPLING and the list of values the file can serve."""

    lines = []
    bits = 0
    el = info["elastic"]
    ok = bool(el and el[1] is not None)
    bits |= 1 if ok else 0
    lines.append(
        "STRAIN_COUPLING bit 1 (elastic constants)            : "
        + ("yes (C2, C3 present)" if ok else "no (no /Elastic/soec,toec)")
    )
    lines.append(
        "reference stress                                     : "
        + ("present" if el and el[0] is not None else "absent (sigma0 = 0)")
    )
    sf = info["strain_force"]
    if sf:
        w = weight_sum_matrix([b[0] for b in sf[0]], [b[2] for b in sf[0]])
        ok = np.allclose(w, 1.0, atol=1.0e-6)
        bits |= 2 if ok else 0
        lines.append(
            "STRAIN_COUPLING bit 2 (strain-force coupling)        : "
            + ("yes" if ok else f"no (weight sums are not 1 for every component:\n{w})")
        )
    else:
        lines.append(
            "STRAIN_COUPLING bit 2 (strain-force coupling)        : no (no /StrainForce)"
        )
    sh = info["strain_harmonic"]
    label = "STRAIN_COUPLING bit 4 (strain-harmonic-IFC coupling) : "
    if sh:
        w = weight_sum_matrix([r[0] for r in sh[0]], [r[2] for r in sh[0]])
        full = np.allclose(w, 1.0, atol=1.0e-6)
        partial = np.all(
            np.isclose(w, 1.0, atol=1.0e-6) | np.isclose(w, 0.0, atol=1.0e-6)
        )
        if full:
            bits |= 4
            lines.append(label + "yes (all components covered)")
        elif partial:
            bits |= 4
            covered = [
                f"{'xyz'[i]}{'xyz'[j]}"
                for i in range(3)
                for j in range(3)
                if abs(w[i, j] - 1.0) < 1.0e-6
            ]
            lines.append(
                label
                + "yes (subset; anphon completes the rest by symmetry; covered components: "
                + ", ".join(covered)
                + ")"
            )
        else:
            lines.append(label + f"no (weight sums must be 1 or 0 per component):\n{w}")
    else:
        lines.append(label + "no (no /StrainHarmonic)")
    values = [v for v in range(8) if v & ~bits == 0]
    lines.append(
        "supported STRAIN_COUPLING values                     : "
        + " ".join(str(v) for v in values)
        + (" (default 7)" if 7 in values else " (the default 7 needs all three groups)")
    )
    return lines


# -------------------------------------------------------------- legacy text
def _read_elastic_lenient(path, log=print):
    """elastic_constants.in as anphon reads it: each section with its own unit
    token (legacy = V*C in Ry), mixed units and trailing tokens only warned."""
    from .writers import _read_section, _tokens

    tok = _tokens(path)
    c2, u2, k = _read_section(tok, 0, "SOEC", 81, path)
    c3, u3, k = _read_section(tok, k, "TOEC", 729, path)
    if k != len(tok):
        log(
            f"  WARNING: {path}: {len(tok) - k} trailing tokens ignored (as anphon does)"
        )
    if u2 != u3:
        log(
            f"  WARNING: {path}: SOEC ({u2}) and TOEC ({u3}) use different units; each converted separately"
        )
    return c2.reshape(9, 9), u2, c3.reshape(9, 9, 9), u3


def pack(
    out,
    strain_ifc_dir=None,
    c1=None,
    fcs=None,
    anphon_cell=None,
    legacy_cell=None,
    force=False,
    log=print,
):
    """Build a container from the legacy text files.

    ``fcs`` (the FC2FILE/FCSFILE of the anphon run) and, for an xml file,
    ``anphon_cell`` (the anphon input with &cell) define the reference cell in
    anphon's atom order.  Legacy Ry files are converted with the volume of
    ``legacy_cell`` (the &cell of the run they were made for; default: the
    reference cell, with a warning).
    """
    from .fcsorder import anphon_primitive_cell, read_anphon_cell, read_fcs_structure
    from .writers import read_C1_array_in, read_strain_force_in, read_strain_harmonic_in

    if fcs is None:
        raise ValueError(
            "--fcs (the force-constant file of the anphon run) is required to define the reference cell"
        )
    if os.path.exists(out) and not force:
        raise ValueError(f"{out} exists; use --force to overwrite it")
    fcs_struct = read_fcs_structure(fcs)
    prim = anphon_primitive_cell(
        fcs_struct, read_anphon_cell(anphon_cell) if anphon_cell else None
    )
    reference = cell_from_primitive(prim)
    log(f"  reference cell: {reference.natom} atoms ({prim.source})")

    if legacy_cell:
        lav = read_anphon_cell(legacy_cell)
        legacy_volume = abs(np.linalg.det(lav)) / BOHR_IN_ANGSTROM**3
        legacy_note = f"volume of {legacy_cell}"
    else:
        legacy_volume = abs(np.linalg.det(reference.lavec)) / BOHR_IN_ANGSTROM**3
        legacy_note = (
            "volume of the reference cell (assumed; give --legacy-cell to be explicit)"
        )

    sdir = strain_ifc_dir
    f_elastic = os.path.join(sdir, "elastic_constants.in") if sdir else None
    f_force = os.path.join(sdir, "strain_force.in") if sdir else None
    f_harm = os.path.join(sdir, "strain_harmonic.in") if sdir else None
    if c1 is None and sdir and os.path.exists(os.path.join(sdir, "C1_array.in")):
        c1 = os.path.join(sdir, "C1_array.in")
    written, sources = [], {}

    def exists(p):
        return p is not None and os.path.exists(p)

    with update(out, reference, None, force=True, source=prim.source) as f:
        # ---- elastic constants and reference stress
        stress = soec = toec = None
        eattrs = {"source": "strainfile.py pack (legacy text files)"}
        legacy_used = False
        if exists(f_elastic):
            c2, u2, c3, u3 = _read_elastic_lenient(f_elastic, log)
            if u2 == "Ry":
                c2 = legacy_ry_per_cell_to_gpa(c2, legacy_volume)
                legacy_used = True
            if u3 == "Ry":
                c3 = legacy_ry_per_cell_to_gpa(c3, legacy_volume)
                legacy_used = True
            soec, toec = c2, c3
            sources["elastic_constants.in"] = os.path.abspath(f_elastic)
        if exists(c1):
            s, u = read_C1_array_in(c1)
            if u == "Ry":
                s = legacy_ry_per_cell_to_gpa(s, legacy_volume)
                legacy_used = True
            stress = s
            sources["C1_array.in"] = os.path.abspath(c1)
        if legacy_used:
            eattrs["legacy_ry_conversion"] = legacy_note
            eattrs["legacy_volume_bohr3"] = legacy_volume
            log(
                f"  legacy Ry values converted to GPa with the {legacy_note} ({legacy_volume:.4f} bohr^3)"
            )
        if stress is not None or soec is not None:
            eattrs["stress_source"] = "C1_array.in" if stress is not None else "absent"
            write_elastic(f, stress, soec, toec, eattrs)
            written.append(
                "/Elastic"
                + (
                    " (stress only)"
                    if soec is None
                    else ""
                    if stress is not None
                    else " (no stress)"
                )
            )
        # ---- strain-force coupling
        if exists(f_force):
            blocks, ref = read_strain_force_in(
                f_force, natmin=None if _has_header(f_force) else reference.natom
            )
            cell = ref if ref is not None else reference
            write_strain_force(
                f, blocks, cell, {"source": "strainfile.py pack (strain_force.in)"}
            )
            sources["strain_force.in"] = os.path.abspath(f_force)
            written.append(
                f"/StrainForce ({len(blocks)} blocks x {cell.natom} atoms; cell from the "
                + ("&reference_cell header" if ref is not None else "reference cell")
                + ")"
            )
        # ---- strain-harmonic coupling
        if exists(f_harm):
            rows = read_strain_harmonic_in(f_harm)
            paths = [os.path.join(sdir, r[3]) for r in rows]
            for p in paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"{p} (listed in {f_harm}) not found")
            write_strain_harmonic(
                f,
                [r[:3] for r in rows],
                paths,
                {"source": "strainfile.py pack (strain_harmonic.in)"},
            )
            sources["strain_harmonic.in"] = os.path.abspath(f_harm)
            written.append(
                f"/StrainHarmonic ({len(rows)} strained supercells embedded)"
            )
        if not written:
            raise ValueError(
                "nothing to pack: no elastic_constants.in, C1_array.in, strain_force.in or strain_harmonic.in found"
            )
        _append_provenance(
            f,
            provenance_record(
                "pack",
                sources=sources,
                fcs=os.path.abspath(fcs),
                anphon_cell=anphon_cell and os.path.abspath(anphon_cell),
                legacy_volume_bohr3=legacy_volume if legacy_used else None,
            ),
        )
    log(f"  written: {out}")
    for w in written:
        log(f"    {w}")
    return out


def _has_header(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                return line.strip().lower().startswith("&reference_cell")
    return False


# ------------------------------------------------------------------- show
def show(path, min_c3=0.5, log=print):
    from . import elasticfit as ef

    info = summary(path)
    a = info["attrs"]
    log(
        f"{path}: schema {a.get('schema')} v{info['format_version']}, written by strainkit "
        f"{a.get('strainkit_version', '?')} on {a.get('created_date', '?')}"
    )
    ref = info["reference_cell"]
    log(
        f"  /ReferenceCell: {ref.natom} atoms, V = {abs(np.linalg.det(ref.lavec)):.4f} A^3 ({info['reference_source']})"
    )
    for v in ref.lavec:
        log("    " + " ".join(f"{x:14.8f}" for x in v))
    for s, x in zip(ref.elements, ref.xf):
        log(f"    {s:4s}" + " ".join(f"{t:14.8f}" for t in x))
    el = info["elastic"]
    if el:
        stress, c2, c3, eattrs = el
        log(
            "  /Elastic: "
            + ", ".join(f"{k}={v}" for k, v in eattrs.items() if k not in ("eta_list",))
        )
        if stress is not None:
            log("    reference stress (GPa):")
            for row in stress:
                log("      " + "".join(f"{x:11.4f}" for x in row))
        else:
            log("    reference stress: absent (sigma0 = 0)")
        if c2 is not None:
            # the elasticfit report helpers take eV/A^3 and print GPa
            c2_ev = ef.from_9x9(c2) / EV_PER_ANG3_TO_GPA
            c3_ev = ef.from_9x9x9(c3) / EV_PER_ANG3_TO_GPA
            log(
                "    "
                + ef.voigt_table_gpa(
                    c2_ev, "second-order elastic constants (GPa, Voigt):"
                ).replace("\n", "\n    ")
            )
            log(f"    third-order elastic constants (GPa, |C| >= {min_c3}):")
            log(
                "    "
                + ef.format_c3_gpa(ef.full3_to_voigt(c3_ev), min_c3).replace(
                    "\n", "\n    "
                )
            )
    else:
        log("  /Elastic: absent")
    sf = info["strain_force"]
    if sf:
        blocks, cell = sf
        log(
            f"  /StrainForce: {len(blocks)} blocks x {cell.natom} atoms (eV/A); "
            + ", ".join(f"{k}={v}" for k, v in info["strain_force_attrs"].items())
        )
        for m, s, w, fr in blocks:
            log(
                f"    {m:3s} smag {s:+.6f} weight {w:.4f}  max|F| = {np.abs(fr).max():.3e}"
            )
    else:
        log("  /StrainForce: absent")
    sh = info["strain_harmonic"]
    if sh:
        rows, entries, cells, nrows = sh
        log(
            f"  /StrainHarmonic: {len(rows)} strained supercells; "
            + ", ".join(f"{k}={v}" for k, v in info["strain_harmonic_attrs"].items())
        )
        for (m, s, w), e, c, n in zip(rows, entries, cells, nrows):
            log(
                f"    {e:30s} {m:3s} smag {s:+.6f} weight {w:.4f}  {c.natom} atoms, {n} FC2 rows"
            )
    else:
        log("  /StrainHarmonic: absent")
    log("  supports:")
    for line in supported_settings(info):
        log("    " + line)
    return info


# ------------------------------------------------------------------ check
def check(path, anphon_cell=None, fcs=None, log=print):
    """Mirror anphon's checks of a container against a planned run.  Returns the
    list of problems (empty = OK)."""
    from .fcsorder import anphon_primitive_cell, read_anphon_cell, read_fcs_structure

    h5py = _h5py()
    problems = []
    info = summary(path)
    ref = info["reference_cell"]
    log(f"{path}: /ReferenceCell with {ref.natom} atoms")
    prim = fcs_struct = None
    if fcs:
        fcs_struct = read_fcs_structure(fcs)
        try:
            prim = anphon_primitive_cell(
                fcs_struct, read_anphon_cell(anphon_cell) if anphon_cell else None
            )
        except ValueError as exc:
            problems.append(str(exc))
    if prim is not None:
        pcell = cell_from_primitive(prim)
        try:
            ratio = same_crystal(ref, pcell)
            log(
                f"  /ReferenceCell vs anphon primitive cell: same crystal, V(anphon)/V(reference) = {1.0 / ratio:.4f}"
            )
        except ValueError as exc:
            problems.append(
                f"/ReferenceCell does not describe the crystal of the anphon run: {exc}"
            )
    sf = info["strain_force"]
    if sf:
        blocks, cell = sf
        try:
            same_crystal(cell, ref)
        except ValueError as exc:
            problems.append(
                f"/StrainForce/Cell is not the crystal of /ReferenceCell: {exc}"
            )
        if prim is not None:
            try:
                same_crystal(cell, cell_from_primitive(prim))
                log(
                    f"  /StrainForce: {len(blocks)} blocks, rows mapped onto the anphon cell OK"
                )
            except ValueError as exc:
                problems.append(
                    f"/StrainForce/Cell cannot be mapped onto the anphon primitive cell: {exc}"
                )
        w = weight_sum_matrix([b[0] for b in blocks], [b[2] for b in blocks])
        if not np.allclose(w, 1.0, atol=1.0e-6):
            problems.append(
                f"/StrainForce: the weight sums are not 1 for every component:\n{w}"
            )
    sh = info["strain_harmonic"]
    if sh:
        rows, entries, cells, nrows = sh
        w = weight_sum_matrix([r[0] for r in rows], [r[2] for r in rows])
        if not np.all(
            np.isclose(w, 1.0, atol=1.0e-6) | np.isclose(w, 0.0, atol=1.0e-6)
        ):
            problems.append(
                f"/StrainHarmonic: the weight sums must be 1 or 0 per component:\n{w}"
            )
        if fcs_struct is not None:
            for (m, s, wt), e, c in zip(rows, entries, cells):
                F = np.eye(3) + mode_tensor(m) * s
                expected = fcs_struct.lavec @ F.T
                bad = []
                if np.abs(c.lavec - expected).max() > 1.0e-4:
                    bad.append(
                        f"lattice differs from the strained reference supercell by {np.abs(c.lavec - expected).max():.2e} A"
                    )
                if c.natom != fcs_struct.nat:
                    bad.append(f"{c.natom} atoms vs {fcs_struct.nat} in {fcs}")
                else:
                    if [x.lower() for x in c.elements] != [
                        x.lower() for x in fcs_struct.elements
                    ]:
                        bad.append("species order differs from the reference supercell")
                    d = c.xf - fcs_struct.xf
                    d -= np.round(d)
                    if np.abs(d).max() > 1.0e-5:
                        bad.append(
                            f"fractional coordinates differ by up to {np.abs(d).max():.2e}"
                        )
                if bad:
                    problems.append(f"{e}: " + "; ".join(bad))
            if not any(p.startswith("/StrainHarmonic/entry") for p in problems):
                log(
                    f"  /StrainHarmonic: {len(rows)} entries match the supercell of {fcs} (index-wise)"
                )
        with h5py.File(path, "r") as f:
            for e, c in zip(entries, cells):
                g = f[e]["ForceConstants/Order2"]
                n = g["force_constant_values"].shape[0]
                shapes = {
                    k: g[k].shape
                    for k in (
                        "atom_indices",
                        "atom_indices_supercell",
                        "coord_indices",
                        "shift_vectors",
                    )
                }
                if (
                    shapes["atom_indices"] != (n, 2)
                    or shapes["atom_indices_supercell"] != (n, 2)
                    or shapes["coord_indices"] != (n, 2)
                    or shapes["shift_vectors"] != (n, 3)
                ):
                    problems.append(
                        f"{e}: inconsistent force-constant dataset shapes {shapes}"
                    )
                    continue
                ci = g["coord_indices"][()]
                ais = g["atom_indices_supercell"][()]
                if ci.min() < 0 or ci.max() > 2:
                    problems.append(f"{e}: coord_indices outside 0..2")
                if ais.min() < 0 or ais.max() >= c.natom:
                    problems.append(
                        f"{e}: atom_indices_supercell outside the SuperCell"
                    )
                if (
                    _decode(g["shift_vectors"].attrs.get("basis", "Cartesian"))
                    != "Cartesian"
                ):
                    problems.append(f"{e}: shift_vectors are not Cartesian")
    el = info["elastic"]
    if el and el[0] is not None:
        from .writers import asymmetry_rank2

        if asymmetry_rank2(el[0]) > 1.0e-6 * max(1.0, np.abs(el[0]).max()):
            problems.append("/Elastic/stress is not symmetric")
    for line in supported_settings(info):
        log("  " + line)
    if problems:
        log("PROBLEMS:")
        for p in problems:
            log("  - " + p)
    else:
        log("  no problems found")
    return problems
