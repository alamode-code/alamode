# Copyright (c) 2026 Terumasa Tadano
# MIT license.  See LICENCE.txt of the ALAMODE package.
"""Clamped-ion nonlinear electromechanical coefficients from VASP (generate / collect).

Routes (all clamped-ion: fractional coordinates fixed, lattice rows a (I + u),
u symmetric; a shear step sets u_mn = u_nm = +-h, i.e. the engineering strain 2h):

* ``berry``: the Berry-phase dipole psi(u) = F^-1 (p_elc + p_ion) on the
  73-point grid (reference, +-h per Voigt direction, (+-h, +-h) for the 15
  pairs) gives e0 = (1/Omega0) dpsi/du and B = (1/Omega0) d^2 psi/du^2.
* ``dfpt``: the DFPT Born charges of the 12 cells +-h give
  Lambda_k,ib,mn = dsym_{u_mn} Z*_k,ib - 1/2 (delta_im Z*_k,nb + delta_in Z*_k,mb),
  the strain derivative of the reduced charges F^-1 Z*.
* ``berry-lambda``: an independent check of one Lambda column: one atom
  displaced by +-delta (Cartesian, added after straining) at strain +-h, and the
  mixed difference of psi.

Everything is per linear strain u in the reference frame (note
anphon/efield_note, Sec. "Nonlinear electromechanical terms").
"""

import os
import shutil
from collections import deque

import numpy as np

from . import strainfile as sf
from .jobscript import write_run_all
from .manifest import load_manifest, save_manifest
from .strain import deform
from .structure import read_template, write_structure
from .units import EV_IN_J
from .writers import ReferenceCell

MANIFEST = "piezo_manifest.json"
ROUTES = ("berry", "dfpt", "berry-lambda")
VOIGT = sf.VOIGT_PAIRS  # xx yy zz yz xz xy
VOIGT_NAMES = ("xx", "yy", "zz", "yz", "xz", "xy")
CFAC = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])  # d(engineering s_J) / d(step)
E_ANG2_TO_CM2 = EV_IN_J * 1.0e20  # |e| / Angst^2 -> C/m^2
VASP_INPUTS = ("INCAR", "KPOINTS", "POTCAR")
# INCAR tags managed per route: (set, removed)
INCAR_TAGS = {
    "berry": (
        {"LCALCPOL": ".TRUE.", "ISYM": "0", "IBRION": "-1", "NSW": "0"},
        ("LEPSILON", "LCALCEPS", "LPEAD", "EFIELD_PEAD"),
    ),
    "dfpt": (
        {"LEPSILON": ".TRUE.", "IBRION": "-1", "NSW": "0"},
        ("LCALCPOL", "LCALCEPS", "EFIELD_PEAD"),
    ),
}
INCAR_TAGS["berry-lambda"] = INCAR_TAGS["berry"]
B_PITFALL = (
    "B (second_order) is not derived from the DFPT route: differentiating the LEPSILON piezoelectric "
    "tensors of strained cells gave a B that violates the (jk)<->(lm) symmetry by up to 1.3 C/m^2 for "
    "ZnO (anphon/efield_note, 'Caveat: strained-cell LEPSILON is not usable for B'). Use "
    "'piezo.py generate --route berry' (73 small SCF runs) for B."
)


def unit_strain(J):
    """Symmetric unit step of Voigt direction J: 1 on (m, n) and (n, m)."""
    e = np.zeros((3, 3))
    m, n = VOIGT[J]
    e[m, n] = e[n, m] = 1.0
    return e


def voigt_index(name):
    names = {n: J for J, n in enumerate(VOIGT_NAMES)}
    names["zx"] = 4
    if name not in names:
        raise ValueError(
            f"unknown strain component {name!r}; use one of {', '.join(VOIGT_NAMES)}"
        )
    return names[name]


def edit_incar(text, set_tags, remove=()):
    """INCAR text with the tags of ``set_tags`` and ``remove`` dropped (also inside
    ';'-separated statements), and ``set_tags`` appended."""
    drop = {k.upper() for k in set_tags} | {k.upper() for k in remove}
    out = []
    for line in text.splitlines():
        body, comment = line, ""
        for c in "#!":
            if c in body:
                body, rest = body.split(c, 1)
                comment = c + rest + comment
        kept = [
            s for s in body.split(";") if s.split("=", 1)[0].strip().upper() not in drop
        ]
        kept = [s for s in kept if s.strip()]
        if kept or (comment and not body.strip()):
            out.append(
                ";".join(kept).rstrip()
                + (" " + comment if comment and kept else comment)
            )
    out.append("# set by piezo.py generate")
    out += [f"{k} = {v}" for k, v in set_tags.items()]
    return "\n".join(out) + "\n"


def _grid(route, h, delta=None, direction=None, strain=None):
    """[(dir name, integer step vector, u (3,3), displacement (3,) or None)]."""
    pts = []
    if route == "berry":
        pts.append(("ref", (0,) * 6))
        for J in range(6):
            for s, t in ((1, "p"), (-1, "m")):
                n = [0] * 6
                n[J] = s
                pts.append((f"J{J + 1}_{t}", tuple(n)))
        for J in range(6):
            for K in range(J + 1, 6):
                for sj, tj in ((1, "p"), (-1, "m")):
                    for sk, tk in ((1, "p"), (-1, "m")):
                        n = [0] * 6
                        n[J], n[K] = sj, sk
                        pts.append((f"J{J + 1}K{K + 1}_{tj}{tk}", tuple(n)))
        return [
            (d, n, h * sum(c * unit_strain(J) for J, c in enumerate(n)), None)
            for d, n in pts
        ]
    if route == "dfpt":
        out = []
        for J in range(6):
            for s, t in ((1, "p"), (-1, "m")):
                n = [0] * 6
                n[J] = s
                out.append((f"J{J + 1}_{t}", tuple(n), s * h * unit_strain(J), None))
        return out
    # berry-lambda: (displacement sign, strain sign)
    out = []
    for sd, td in ((1, "p"), (-1, "m")):
        for sh, th in ((1, "p"), (-1, "m")):
            disp = np.zeros(3)
            disp[direction] = sd * delta
            out.append((f"D{td}H{th}", (sd, sh), sh * h * unit_strain(strain), disp))
    return out


def generate(
    route,
    refdir,
    outdir,
    h=0.01,
    atom=None,
    direction=None,
    strain=None,
    delta=0.02,
    force=False,
    log=print,
):
    """Write the clamped-ion strained VASP inputs of ``route`` and the manifest.

    ``atom`` (1-based, POSCAR order), ``direction`` (x|y|z), ``strain``
    (xx..xy) and ``delta`` (Angst) are used only by ``berry-lambda``.
    """
    if route not in ROUTES:
        raise ValueError(f"unknown route {route!r}; use one of {', '.join(ROUTES)}")
    if not (0.0 < h < 0.1):
        raise ValueError(f"--h {h}: expected a small positive strain step")
    refdir = os.path.abspath(refdir)
    template = read_template("VASP", refdir)
    missing = [n for n in VASP_INPUTS if not os.path.isfile(os.path.join(refdir, n))]
    if "INCAR" in missing:
        raise ValueError(
            f"{refdir}: no INCAR (the reference calculation's INCAR is edited per route)"
        )
    if missing:
        log(
            f"  WARNING: {refdir}: missing {', '.join(missing)}; add them to every directory"
        )
    # only the inputs are copied, never the outputs of the reference run
    template.extra_files = [
        os.path.join(refdir, n) for n in ("KPOINTS", "POTCAR") if n not in missing
    ]
    natom = len(template.atoms)
    k = b = J = None
    if route == "berry-lambda":
        if atom is None or direction is None or strain is None:
            raise ValueError(
                "--route berry-lambda needs --atom K --dir x|y|z --strain mn"
            )
        k = int(atom) - 1
        if not 0 <= k < natom:
            raise ValueError(f"--atom {atom}: the cell has atoms 1..{natom}")
        if direction not in ("x", "y", "z"):
            raise ValueError(f"--dir {direction!r}: use x, y or z")
        b = "xyz".index(direction)
        J = voigt_index(strain)
        if not (0.0 < delta < 0.1):
            raise ValueError(
                f"--delta {delta}: expected a small positive displacement (Angst)"
            )
    points = _grid(route, h, delta, b, J)
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    mpath = os.path.join(outdir, MANIFEST)
    existing = [d for d, *_ in points if os.path.exists(os.path.join(outdir, d))]
    if (existing or os.path.exists(mpath)) and not force:
        raise FileExistsError(
            f"{outdir} already holds a piezo manifest or {len(existing)} of the directories; use --force"
        )
    for d in existing:
        shutil.rmtree(os.path.join(outdir, d))
    with open(os.path.join(refdir, "INCAR")) as f:
        incar = edit_incar(f.read(), *INCAR_TAGS[route])
    entries = []
    for d, n, u, disp in points:
        atoms = deform(template.atoms, np.eye(3) + u)
        if disp is not None:
            # selective-dynamics constraints (FixAtoms, ...) would silently drop the
            # imposed displacement: remove them for this copy and verify the result
            del atoms.constraints
            ideal = atoms.get_positions()
            pos = ideal.copy()
            pos[k] += disp
            atoms.set_positions(pos)
            moved = atoms.get_positions() - ideal
            want = np.zeros_like(moved)
            want[k] = disp
            if np.abs(moved - want).max() > 1.0e-10:
                raise RuntimeError(
                    f"{d}: the displacement of atom {k + 1} was not applied as requested"
                )
        path = os.path.join(outdir, d)
        write_structure(template, atoms, path)
        with open(os.path.join(path, "INCAR"), "w") as f:
            f.write(incar)
        entries.append({"dir": d, "step": list(n), "u": u, "displacement": disp})
    write_run_all(os.path.join(outdir, "run_all.sh"), [e["dir"] for e in entries])
    manifest = {
        "tool": "piezo",
        "route": route,
        "refdir": refdir,
        "h": float(h),
        "lattice": np.asarray(template.atoms.cell[:], dtype=float),
        "symbols": [str(s) for s in template.atoms.get_chemical_symbols()],
        "xf": template.atoms.get_scaled_positions(wrap=False),
        "entries": entries,
    }
    if route == "berry-lambda":
        manifest.update(
            {
                "atom": k + 1,
                "dir_index": b,
                "strain": VOIGT_NAMES[J],
                "delta": float(delta),
            }
        )
    save_manifest(manifest, mpath)
    log(f"Generated {len(entries)} clamped-ion VASP calculations ({route}) in {outdir}")
    log("  INCAR: " + ", ".join(f"{t} = {v}" for t, v in INCAR_TAGS[route][0].items()))
    if route == "berry-lambda":
        log(
            f"  atom {k + 1} ({manifest['symbols'][k]}) displaced by +-{delta} Angst along {direction}, "
            f"strain {VOIGT_NAMES[J]} +-{h}"
        )
    log(f"  Run VASP in every directory (run_all.sh), then: piezo.py collect {outdir}")
    return manifest


# ---------------------------------------------------------------- parsing
def parse_outcar_dipole(path):
    """(p_elc, p_ion) in |e| Angst from the LAST 'p[elc]=(' and 'p[ion]=(' lines (LCALCPOL)."""
    pe = pi = None
    with open(path) as f:
        for line in f:
            for key in ("p[elc]=(", "p[ion]=("):
                if key in line:
                    try:
                        v = np.array(
                            [
                                float(x)
                                for x in line.split(key, 1)[1].split(")")[0].split()
                            ]
                        )
                    except ValueError:
                        raise ValueError(
                            f"{path}: malformed line {line.strip()!r}"
                        ) from None
                    if v.shape != (3,) or not np.all(np.isfinite(v)):
                        raise ValueError(f"{path}: malformed line {line.strip()!r}")
                    if key == "p[elc]=(":
                        pe = v
                    else:
                        pi = v
    if pe is None or pi is None:
        raise ValueError(
            f"{path}: no 'p[elc]=(' / 'p[ion]=(' dipole lines (run VASP with LCALCPOL = .TRUE.)"
        )
    return pe, pi


def parse_outcar_born(path, natom):
    """Z* (natom, 3, 3) [atom, i, b] of the LAST complete 'BORN EFFECTIVE CHARGES
    (including local field effects)' block.

    VASP writes row IDIR = BORN_CHARGES(IDIR, :, N) with IDIR the electric-field
    direction and the columns the force (displacement) direction
    (linear_response.F: FORCE(:,N) += BORN_CHARGES(IDIR,:,N) * DFIELD(IDIR)),
    so the row index is the polarization i and the column the displacement b.
    """
    with open(path) as f:
        lines = f.readlines()
    blocks = []
    for i, line in enumerate(lines):
        if "BORN EFFECTIVE CHARGES" in line and "including local field effects" in line:
            try:
                z = np.zeros((natom, 3, 3))
                p = i + 2
                for k in range(natom):
                    if lines[p].split() != ["ion", str(k + 1)]:
                        raise ValueError
                    for r in range(3):
                        t = lines[p + 1 + r].split()
                        if len(t) != 4 or t[0] != str(r + 1):
                            raise ValueError
                        z[k, r] = [float(x) for x in t[1:]]
                    p += 4
                if not np.all(np.isfinite(z)):
                    raise ValueError
                blocks.append(z)
            except (IndexError, ValueError):
                continue  # truncated or for another atom count
    if not blocks:
        raise ValueError(
            f"{path}: no complete 'BORN EFFECTIVE CHARGES (including local field effects)' block for "
            f"{natom} atoms (run VASP with LEPSILON = .TRUE.)"
        )
    return blocks[-1]


# ---------------------------------------------------------------- helpers
def _ref_atoms(manifest):
    import ase

    return ase.Atoms(
        symbols=manifest["symbols"],
        cell=np.asarray(manifest["lattice"]),
        scaled_positions=np.asarray(manifest["xf"]),
        pbc=True,
    )


def _read_cell(manifest, outdir, entry):
    """F (3,3) of the entry's POSCAR relative to the manifest lattice; checks the
    species, the (clamped-ion) positions and the step."""
    import ase.io

    path = os.path.join(outdir, entry["dir"], "POSCAR")
    if not os.path.isfile(path):
        raise ValueError(f"{path}: missing")
    atoms = ase.io.read(path, format="vasp")
    if [str(s) for s in atoms.get_chemical_symbols()] != manifest["symbols"]:
        raise ValueError(f"{path}: species or atom order differ from the manifest")
    A0 = np.asarray(manifest["lattice"])
    A = np.asarray(atoms.cell[:])
    F = np.linalg.solve(A0, A).T  # rows a' = a F^T
    u = np.asarray(entry["u"])
    if np.abs(F - np.eye(3) - u).max() > 1.0e-8:
        raise ValueError(
            f"{path}: the lattice is not the generated strain (F - I != u)"
        )
    expected = np.asarray(manifest["xf"]) @ A  # fixed fractional coordinates
    if entry.get("displacement") is not None:
        expected[manifest["atom"] - 1] += np.asarray(entry["displacement"])
    d = (atoms.get_positions() - expected) @ np.linalg.inv(A)
    if np.abs(d - np.round(d)).max() > 1.0e-6:
        raise ValueError(
            f"{path}: the positions are not the generated clamped-ion positions"
        )
    return F


def _check_outcar(manifest, outdir, entry):
    """The structure recorded in the entry's OUTCAR (last lattice, fractional
    coordinates, species in POTCAR order) must be the generated one: a stale or
    swapped OUTCAR is rejected.  Returns the OUTCAR lines."""
    path = os.path.join(outdir, entry["dir"], "OUTCAR")
    if not os.path.isfile(path):
        raise ValueError(f"{path}: missing")
    with open(path) as f:
        lines = f.readlines()
    cell = sf.parse_outcar_cell(lines, path)
    if cell is None:
        raise ValueError(
            f"{path}: no structure lines (TITEL, 'ions per type', 'direct lattice vectors', "
            "'position of ions in fractional coordinates'); cannot verify that it belongs to this cell"
        )
    if [str(s) for s in cell.elements] != manifest["symbols"]:
        raise ValueError(
            f"{path}: species/atom order {cell.elements} differ from the manifest {manifest['symbols']}"
        )
    u = np.asarray(entry["u"])
    A = np.asarray(manifest["lattice"]) @ (np.eye(3) + u).T
    if np.abs(cell.lavec - A).max() > 2.0e-6:
        raise ValueError(
            f"{path}: the lattice of this run is not the generated strained lattice "
            "(stale or swapped OUTCAR?)"
        )
    xf = np.asarray(manifest["xf"], dtype=float).copy()
    if entry.get("displacement") is not None:
        xf[manifest["atom"] - 1] += np.asarray(entry["displacement"]) @ np.linalg.inv(A)
    d = np.asarray(cell.xf) - xf
    if np.abs(d - np.round(d)).max() > 1.0e-6:
        raise ValueError(
            f"{path}: the atomic positions of this run are not the generated ones (stale or swapped OUTCAR?)"
        )
    return lines


def _psi(manifest, outdir, clamped=True, log=print):
    """{step tuple: psi (3,)} with psi = F^-1 (p_elc + p_ion) in |e| Angst, reference frame.

    For clamped-ion cells (``clamped``) F^-1 p_ion is exactly constant (fixed
    fractional coordinates); its grid mean is used, so the 5-decimal rounding
    of the printed p_ion does not enter the second differences.  The spread
    (modulo whole lattice quanta) is checked.
    """
    A0 = np.asarray(manifest["lattice"])
    elc, ion = {}, {}
    for e in manifest["entries"]:
        F = _read_cell(manifest, outdir, e)
        _check_outcar(manifest, outdir, e)
        pe, pi = parse_outcar_dipole(os.path.join(outdir, e["dir"], "OUTCAR"))
        key = tuple(e["step"])
        elc[key] = np.linalg.solve(F, pe)
        ion[key] = np.linalg.solve(F, pi)
    if not clamped:
        return {k: elc[k] + ion[k] for k in elc}
    f = np.array([v @ np.linalg.inv(A0) for v in ion.values()])
    f -= np.round(f - f[0])
    spread = float(np.abs(f - f.mean(axis=0)).max())
    if spread > 1.0e-3:
        raise ValueError(
            f"the reduced ionic dipole F^-1 p_ion varies by {spread:.2e} lattice units over the grid; "
            "it must be constant for clamped-ion cells (were the ions moved?)"
        )
    log(
        f"  reduced ionic dipole constant over the grid to {spread:.1e} lattice units (its mean is used)"
    )
    ion0 = f.mean(axis=0) @ A0
    return {k: v + ion0 for k, v in elc.items()}


def unwrap(psi, lattice, log=print):
    """Remove the Berry-phase branch jumps on the connected grid.

    The dipole is expressed in reference-lattice units (psi = f A0, one quantum
    = one lattice vector, the conservative choice for any spin degeneracy).
    Grid points whose steps differ in one coordinate are neighbours; a BFS tree
    from the first point fixes the branch, and every edge (tree and
    loop-closing) must then change f by less than a quarter quantum.
    Returns (unwrapped psi dict, max tree step, max loop-closing step).
    """
    A0 = np.asarray(lattice)
    f = {k: v @ np.linalg.inv(A0) for k, v in psi.items()}
    keys = list(f)
    nbr = {
        a: [b for b in keys if b != a and sum(x != y for x, y in zip(a, b)) == 1]
        for a in keys
    }
    root = (0,) * len(keys[0]) if (0,) * len(keys[0]) in f else keys[0]
    seen, tree, queue = {root}, set(), deque([root])
    while queue:
        a = queue.popleft()
        for b in nbr[a]:
            if b not in seen:
                f[b] = f[b] - np.round(f[b] - f[a])
                seen.add(b)
                tree.add(frozenset((a, b)))
                queue.append(b)
    if len(seen) != len(keys):
        raise ValueError("the strain grid is not connected (missing points?)")
    step_tree = step_loop = 0.0
    for a in keys:
        for b in nbr[a]:
            s = float(np.abs(f[b] - f[a]).max())
            if frozenset((a, b)) in tree:
                step_tree = max(step_tree, s)
            else:
                step_loop = max(step_loop, s)
    log(
        f"  branch: largest change between neighbouring points {step_tree:.4f} (tree), "
        f"{step_loop:.4f} (loop-closing edges), in lattice units (quantum = 1)"
    )
    if max(step_tree, step_loop) > 0.25:
        raise ValueError(
            f"the Berry-phase dipole changes by {max(step_tree, step_loop):.3f} lattice units between "
            "neighbouring grid points (> 1/4 quantum): the branch is ambiguous; use a smaller --h"
        )
    return {k: v @ A0 for k, v in f.items()}, step_tree, step_loop


def map_to_reference(ref, calc, tol_bohr=1.0e-3):
    """For each /ReferenceCell atom, the list of calculation-cell atoms that are its
    translation images (one for the same or a larger reference cell, several when
    the calculation cell is a supercell of the reference cell)."""
    sf.same_crystal(ref, calc)  # validates nesting, frame and composition
    vr = abs(np.linalg.det(ref.lavec))
    vc = abs(np.linalg.det(calc.lavec))
    small = ref.lavec if vr <= vc else calc.lavec
    inv = np.linalg.inv(small)
    yr = (np.asarray(ref.xf) @ ref.lavec) @ inv
    yc = (np.asarray(calc.xf) @ calc.lavec) @ inv
    n_img = max(1, int(round(vc / vr)))
    tol = tol_bohr * sf.BOHR_IN_ANGSTROM
    out = []
    for i, ei in enumerate(ref.elements):
        hits = []
        for j, ej in enumerate(calc.elements):
            d = yc[j] - yr[i]
            d -= np.round(d)
            if str(ei).lower() == str(ej).lower() and np.linalg.norm(d @ small) < tol:
                hits.append(j)
        if len(hits) != n_img:
            raise ValueError(
                f"reference atom {i + 1} ({ei}) has {len(hits)} images in the calculation cell, expected {n_img}"
            )
        out.append(hits)
    return out


def _voigt_table(t, fmt="{:9.3f}"):
    return ["      " + "".join(fmt.format(x) for x in row) for row in t]


def _rotations(manifest):
    from .symmetry import cartesian_rotations

    return cartesian_rotations(_ref_atoms(manifest))


# --------------------------------------------------------------- collect
def collect_berry(manifest, outdir, symmetrize=True, log=print):
    """(e0 (3,3,3), B (3,3,3,3,3), their Voigt tables, the raw Voigt tables,
    diagnostics) in C/m^2 from the 73-point grid; ``symmetrize`` averages e0 and B
    over the point group of the reference cell."""
    h = manifest["h"]
    A0 = np.asarray(manifest["lattice"])
    om0 = abs(np.linalg.det(A0))
    psi, step_tree, step_loop = unwrap(_psi(manifest, outdir, log=log), A0, log)

    def P(*steps):
        """psi at the grid point with the (Voigt J, sign) steps."""
        n = [0] * 6
        for J, sgn in steps:
            n[J] = sgn
        return psi[tuple(n)]

    ev = np.zeros((3, 6))
    bv = np.zeros((3, 6, 6))
    for J in range(6):
        ev[:, J] = (P((J, 1)) - P((J, -1))) / (2 * h * CFAC[J])
        bv[:, J, J] = (P((J, 1)) - 2 * P() + P((J, -1))) / (h * CFAC[J]) ** 2
        for K in range(J + 1, 6):
            mixed = sum(
                sj * sk * P((J, sj), (K, sk)) for sj in (1, -1) for sk in (1, -1)
            )
            bv[:, J, K] = bv[:, K, J] = mixed / (4 * h * h * CFAC[J] * CFAC[K])
    ev *= E_ANG2_TO_CM2 / om0
    bv *= E_ANG2_TO_CM2 / om0
    e0 = sf.piezo_from_columns(ev, VOIGT)
    B = np.zeros((3, 3, 3, 3, 3))
    for J, (j, k) in enumerate(VOIGT):
        for K, (p, q) in enumerate(VOIGT):
            for jj, kk in ((j, k), (k, j)):
                for ll, mm in ((p, q), (q, p)):
                    B[:, jj, kk, ll, mm] = bv[:, J, K]
    pair = float(np.abs(B - B.transpose(0, 3, 4, 1, 2)).max())
    rots = _rotations(manifest)
    e_sym = sum(np.einsum("ia,jb,kc,abc->ijk", r, r, r, e0) for r in rots) / len(rots)
    b_sym = sum(
        np.einsum("ia,jb,kc,ld,me,abcde->ijklm", r, r, r, r, r, B) for r in rots
    ) / len(rots)
    diag = {
        "omega0_A3": om0,
        "branch_step_max": step_tree,
        "branch_loop_max": step_loop,
        "pair_exchange_violation": pair,
        "point_group_order": len(rots),
        "e0_symmetry_noise": float(np.abs(e0 - e_sym).max()),
        "B_symmetry_noise": float(np.abs(B - b_sym).max()),
        "symmetrized": bool(symmetrize),
    }
    log(f"  Omega0 = {om0:.4f} A^3, h = {h}, {len(psi)} grid points")
    log(
        f"  B pair exchange (jk)<->(lm): max violation {pair:.1e} (0 by construction); "
        f"point-group ({len(rots)} operations) noise: e0 {diag['e0_symmetry_noise']:.3f}, "
        f"B {diag['B_symmetry_noise']:.3f} C/m^2 (the numerical precision)"
    )
    if symmetrize:
        e0, B = e_sym, b_sym
    ev_out = sf.piezo_to_voigt(e0)
    bv_out = np.array(
        [[[B[i, j, k, p, q] for p, q in VOIGT] for j, k in VOIGT] for i in range(3)]
    )
    what = "point-group symmetrized" if symmetrize else "raw"
    log(f"  e0 ({what}; C/m^2, rows x y z, Voigt xx yy zz yz xz xy):")
    for line in _voigt_table(ev_out, "{:9.4f}"):
        log(line)
    for i in range(3):
        log(f"  B_{'xyz'[i]},JK ({what}; C/m^2, Voigt JK):")
        for line in _voigt_table(bv_out[i]):
            log(line)
    return e0, B, ev_out, bv_out, ev, bv, diag


def born_lambda(zp, zm, h, z0):
    """Lambda (natom,3,3,3,3) [k, i, b, m, n] from Born charges Z[J][k, i, b] of the
    +-h cells (lists over the six Voigt directions) and the reference Z0 (rev.2 P1)."""
    nat = z0.shape[0]
    lam = np.zeros((nat, 3, 3, 3, 3))
    for J, (m, n) in enumerate(VOIGT):
        d = (zp[J] - zm[J]) / (2 * h * CFAC[J])  # dsym Z / du_mn
        lam[..., m, n] = lam[..., n, m] = d
    eye = np.eye(3)
    lam -= 0.5 * (
        np.einsum("im,knb->kibmn", eye, z0) + np.einsum("in,kmb->kibmn", eye, z0)
    )
    return lam


def _born_ref_charges(manifest, path):
    """Z* of the unstrained reference OUTCAR in the manifest's atom order: its
    structure must be the reference crystal in the same frame (permuted atoms or
    a nested cell are mapped; translation images are averaged)."""
    with open(path) as f:
        cell = sf.parse_outcar_cell(f.readlines(), path)
    if cell is None:
        raise ValueError(
            f"--born-ref {path}: no structure lines in the OUTCAR; its atom order cannot be verified"
        )
    calc = ReferenceCell(
        np.asarray(manifest["lattice"]), manifest["symbols"], np.asarray(manifest["xf"])
    )
    try:
        images = map_to_reference(calc, cell)
    except ValueError as exc:
        raise ValueError(
            f"--born-ref {path}: not the unstrained reference crystal of the manifest ({exc})"
        ) from None
    z = parse_outcar_born(path, len(cell.elements))
    return np.array([z[im].mean(axis=0) for im in images])


def collect_dfpt(manifest, outdir, born_ref=None, symmetrize=True, log=print):
    """(Lambda (natom,3,3,3,3) ASR-corrected and, with ``symmetrize``, averaged over
    the space group of the reference cell, diagnostics) from the 12 strained cells."""
    h = manifest["h"]
    nat = len(manifest["symbols"])
    z = {}
    for e in manifest["entries"]:
        _read_cell(manifest, outdir, e)
        _check_outcar(manifest, outdir, e)
        z[tuple(e["step"])] = parse_outcar_born(
            os.path.join(outdir, e["dir"], "OUTCAR"), nat
        )
    unit = [tuple(int(J == K) for K in range(6)) for J in range(6)]
    zp = [z[n] for n in unit]
    zm = [z[tuple(-x for x in n)] for n in unit]
    zmean = np.mean(list(z.values()), axis=0)
    if born_ref is not None:
        z0 = _born_ref_charges(manifest, born_ref)
        log(
            f"  reference Born charges from {born_ref}; max |Z0 - mean of the strained cells| = "
            f"{np.abs(z0 - zmean).max():.5f} e"
        )
    else:
        z0 = zmean
        log(
            "  reference Born charges: the mean of the 12 strained cells (O(h^2); --born-ref to give them)"
        )
    z_asr = float(np.abs(z0.sum(axis=0) / nat).max())
    lam = born_lambda(zp, zm, h, z0)
    resid = lam.mean(axis=0)
    asr = float(np.abs(resid).max())
    lam = lam - resid
    log(
        f"  Lambda ASR residual (max |mean over atoms|) {asr:.2e} e, removed; Z0 ASR residual {z_asr:.1e} e"
    )
    from .symmetry import space_group_operations, symmetrize_atomic

    ops = space_group_operations(_ref_atoms(manifest))
    lam_sym = symmetrize_atomic(lam, *ops)
    lmax = float(np.abs(lam).max())
    noise = float(np.abs(lam_sym - lam).max())
    rel = noise / lmax if lmax > 0.0 else 0.0
    log(
        f"  space-group ({len(ops[0])} operations) noise of Lambda: max {noise:.4f} e "
        f"({rel:.1%} of max |Lambda| = {lmax:.4f} e)"
        + ("; symmetrized" if symmetrize else "; raw (--no-symmetrize)")
    )
    if symmetrize:
        lam = lam_sym
    log(
        "  (the PIEZOELECTRIC TENSOR blocks of these OUTCARs are ignored: "
        "B comes only from the Berry route)"
    )
    return lam, {
        "asr_residual_removed": asr,
        "z0_asr_residual": z_asr,
        "h": h,
        "space_group_order": len(ops[0]),
        "Lambda_symmetry_noise": noise,
        "Lambda_symmetry_noise_relative": rel,
        "symmetrized": bool(symmetrize),
    }


def collect_berry_lambda(manifest, outdir, log=print):
    """Lambda_K,ib,mn (3,) [i] by the mixed difference of psi over (+-delta, +-h)."""
    h, delta = manifest["h"], manifest["delta"]
    J = voigt_index(manifest["strain"])
    psi, *_ = unwrap(_psi(manifest, outdir, clamped=False), manifest["lattice"], log)
    mixed = sum(sd * sh * psi[(sd, sh)] for sd in (1, -1) for sh in (1, -1))
    col = mixed / (4 * delta * h * CFAC[J])
    return col


def _print_lambda(lam, symbols, log):
    for k, s in enumerate(symbols):
        log(f"  Lambda atom {k + 1} ({s}), d Zt*_ib / d s_J (e; rows ib, Voigt J):")
        for i in range(3):
            for b in range(3):
                row = [lam[k, i, b, m, n] for m, n in VOIGT]
                log(f"    {'xyz'[i]}{'xyz'[b]} " + "".join(f"{x:9.4f}" for x in row))


def collect(
    outdir,
    route=None,
    strain_file=None,
    overwrite=False,
    born_ref=None,
    second_order=False,
    skip_clamped_ion=False,
    symmetrize=True,
    verbose=False,
    log=print,
):
    """Evaluate a generated route and optionally write it into ``strain_file``.

    Returns a dict with the computed tensors (calculation-cell atom order).
    """
    outdir = os.path.abspath(outdir)
    manifest = load_manifest(os.path.join(outdir, MANIFEST), expect_tool="piezo")
    route = route or manifest["route"]
    if route != manifest["route"]:
        raise ValueError(
            f"--route {route}, but {outdir} was generated for {manifest['route']!r}"
        )
    if second_order and route != "berry":
        raise ValueError(B_PITFALL)
    calc = ReferenceCell(
        np.asarray(manifest["lattice"]), manifest["symbols"], np.asarray(manifest["xf"])
    )
    ref = sf.reference_cell_of(strain_file) if strain_file else None
    images = map_to_reference(ref, calc) if ref is not None else None
    log(f"piezo collect ({route}): {outdir}")
    src = {"source": outdir, "h": manifest["h"]}
    if route == "berry":
        e0, B, ev, bv, ev_raw, bv_raw, diag = collect_berry(
            manifest, outdir, symmetrize, log
        )
        result = {
            "e0": e0,
            "B": B,
            "e0_voigt": ev,
            "B_voigt": bv,
            "e0_raw_voigt": ev_raw,
            "B_raw_voigt": bv_raw,
            "diagnostics": diag,
        }
        if ref is not None:
            method = (
                "VASP LCALCPOL Berry-phase dipole on the clamped-ion 73-point strain grid, central differences"
                + (", point-group symmetrized" if symmetrize else "")
            )
            data = {sf.SECOND_ORDER: B}
            attrs = {sf.SECOND_ORDER: {"method": method, **src, **diag}}
            if not skip_clamped_ion:
                data[sf.CLAMPED_ION] = e0
                attrs[sf.CLAMPED_ION] = {"method": method, **src}
            sf.add_piezo_datasets(strain_file, data, attrs, overwrite, log)
        return result
    if route == "dfpt":
        lam, diag = collect_dfpt(manifest, outdir, born_ref, symmetrize, log)
        if verbose or ref is None:
            _print_lambda(lam, manifest["symbols"], log)
        if ref is not None:
            mapped = np.array([lam[im].mean(axis=0) for im in images])
            after = float(np.abs(mapped.mean(axis=0)).max())
            log(
                f"  mapped onto {strain_file}:/ReferenceCell ({ref.natom} atoms); ASR residual after mapping {after:.1e}"
            )
            attrs = {
                "method": "VASP LEPSILON Born charges of the clamped-ion +-h cells, central differences, "
                "reduced by F^-1 (rev.2 P1); ASR: atomic mean subtracted"
                + ("; space-group symmetrized" if symmetrize else ""),
                **src,
                **diag,
                "born_ref": os.path.abspath(born_ref)
                if born_ref
                else "mean of the strained cells",
            }
            sf.add_piezo_datasets(
                strain_file,
                {sf.BORN_DERIV: mapped},
                {sf.BORN_DERIV: attrs},
                overwrite,
                log,
            )
        return {"Lambda": lam, "diagnostics": diag}
    col = collect_berry_lambda(manifest, outdir, log)
    k, b, mn = manifest["atom"], manifest["dir_index"], manifest["strain"]
    log(
        f"  Berry mixed difference: Lambda[atom {k} ({manifest['symbols'][k - 1]}), i, b={'xyz'[b]}, {mn}] "
        f"(i = x y z) = " + " ".join(f"{x:.4f}" for x in col) + " e"
    )
    result = {"Lambda_column": col}
    if ref is not None:
        import h5py

        with h5py.File(strain_file, "r") as f:
            got = sf.read_piezo_dataset(f, sf.BORN_DERIV)
        if got is None:
            log(f"  {strain_file} has no /{sf.PIEZO}/{sf.BORN_DERIV} to compare with")
        else:
            iref = next(i for i, im in enumerate(images) if k - 1 in im)
            m, n = VOIGT[voigt_index(mn)]
            other = got[0][iref, :, b, m, n]
            log(
                f"  stored Lambda (reference atom {iref + 1}):      "
                + " ".join(f"{x:.4f}" for x in other)
                + f" e; max difference {np.abs(col - other).max():.4f} e"
            )
            result["Lambda_stored"] = other
    return result
