"""DFT finite-strain workflow for the elastic constants (generate / fit)."""

import json
import os
import shutil
import warnings

import numpy as np

from . import elasticfit as ef
from .dftio import check_geometry, read_dft_output
from .fcsorder import (
    anphon_primitive_cell,
    describe_ordering,
    lattice_relation,
    match_primitive_atoms,
    read_anphon_cell,
    read_fcs_structure,
)
from .manifest import ELASTIC_MANIFEST, load_manifest, save_manifest
from .strain import MODE_NAMES, deform, deformation_gradient, voigt_from_sym
from .structure import OUTPUT_FILENAME, normalize_code, read_template, write_structure
from .symmetry import (
    cartesian_rotations,
    enforce_intrinsic_symmetry2,
    enforce_intrinsic_symmetry3,
    max_change,
    symmetrize_rank2,
    symmetrize_rank4,
    symmetrize_rank6,
)
from .units import EV_PER_ANG3_TO_GPA
from .writers import (
    anphon_file_locations,
    write_C1_array_in,
    write_elastic_constants_in,
    write_strain_force_in,
)
from .jobscript import render_job_script, write_run_all

RESULTS_DIR = "results"


def _dirname(index):
    return f"strain_{index:03d}"


def generate(
    code,
    template_dir,
    outdir=".",
    structure_file=None,
    smag=0.01,
    nmag=2,
    dirset="minimal",
    job_template=None,
    dft_command=None,
    force=False,
    log=print,
):
    code = normalize_code(code)
    template = read_template(code, template_dir, structure_file)
    points = ef.strain_points(ef.direction_set(dirset), smag, nmag)
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    existing = [
        p for p in points if os.path.exists(os.path.join(outdir, _dirname(p.index)))
    ]
    if existing and not force:
        raise FileExistsError(
            f"{len(existing)} strain_* directories already exist in {outdir}; use --force to overwrite"
        )
    for p in existing:
        shutil.rmtree(os.path.join(outdir, _dirname(p.index)))

    ref_dir = os.path.join(outdir, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    write_structure(template, template.atoms, ref_dir, copy_extra=False)

    entries = []
    for p in points:
        d = os.path.join(outdir, _dirname(p.index))
        write_structure(template, deform(template.atoms, p.F), d)
        entries.append(
            {
                "index": p.index,
                "dir": _dirname(p.index),
                "label": p.label,
                "k": p.k,
                "d6": p.d6,
                "u": p.u,
                "eta": p.eta,
            }
        )
    subdirs = [e["dir"] for e in entries]
    dft_text = open(dft_command).read() if dft_command else None
    if job_template:
        with open(os.path.join(outdir, "job.sh"), "w") as f:
            f.write(
                render_job_script(open(job_template).read(), dft_text or "", subdirs)
            )
    else:
        write_run_all(os.path.join(outdir, "run_all.sh"), subdirs, dft_text)

    manifest = {
        "tool": "elastic",
        "code": code,
        "template_dir": template.template_dir,
        "structure_file": template.structure_name,
        "output_file": OUTPUT_FILENAME[code],
        "smag": float(smag),
        "nmag": int(nmag),
        "dirset": dirset,
        "volume_A3": float(abs(np.linalg.det(template.atoms.cell[:]))),
        "nat": len(template.atoms),
        "numbers": np.asarray(template.atoms.numbers).tolist(),
        "entries": entries,
    }
    save_manifest(manifest, os.path.join(outdir, ELASTIC_MANIFEST))
    log(
        f"Generated {len(points)} calculations ({len(points) - 1} strained + 1 reference) in {outdir}"
    )
    log(
        f"  direction set: {dirset} ({len(ef.direction_set(dirset))} directions), "
        f"magnitudes: +-{smag} x 1..{nmag}"
    )
    log(
        "  Run the DFT code in every strain_*/ directory (fixed cell, fixed ions; clamped-ion)."
    )
    log(f"  Expected output file per directory: {OUTPUT_FILENAME[code]}")
    log(f"  Manifest: {os.path.join(outdir, ELASTIC_MANIFEST)}")
    return manifest


def _reference_atoms(manifest, outdir):
    from .structure import _ase_read

    fmt = {"VASP": "vasp", "QE": "espresso-in", "ase": None}[manifest["code"]]
    return _ase_read(os.path.join(outdir, "reference", manifest["structure_file"]), fmt)


def anphon_cell_relation(ref_atoms, fcs=None, anphon_cell=None, log=print):
    """Report how the anphon primitive cell relates to the DFT cell.

    Returns (V_anphon / V_dft, anphon lattice rows in Angstrom, anphon primitive
    cell or None), or (None, None, None) when neither --fcs nor --anphon-cell is
    given.  Informative for the text files, which are written in GPa and do not
    depend on the cell anphon uses; the primitive cell labels the container.
    """
    if fcs is None and anphon_cell is None:
        return None, None, None
    a_dft = np.asarray(ref_atoms.cell[:], dtype=float)
    prim = None
    if fcs is not None:
        fcs_struct = read_fcs_structure(fcs)
        cell = read_anphon_cell(anphon_cell) if anphon_cell else None
        prim = anphon_primitive_cell(fcs_struct, cell)
        lav = prim.lavec
    else:
        lav = read_anphon_cell(anphon_cell)
    m, ratio = lattice_relation(lav, a_dft)
    note = (
        "(informative only: the files are written in GPa and do not depend on the cell)"
    )
    if ratio >= 1.0:
        log(f"  anphon cell = M x DFT cell with det(M) = {int(round(ratio))} {note}")
    else:
        log(
            f"  DFT cell = M x anphon cell with det(M) = {int(round(1.0 / ratio))} {note}"
        )
    return float(ratio), lav, prim


def fit(
    outdir=".",
    mode="stress",
    fcs=None,
    anphon_cell=None,
    allow_relaxed=False,
    force_write=False,
    symmetrize=True,
    symprec=1.0e-5,
    results_dir=RESULTS_DIR,
    compare=None,
    min_c3=0.5,
    exclude=(),
    strain_file=None,
    force=False,
    log=print,
):
    outdir = os.path.abspath(outdir)
    manifest = load_manifest(
        os.path.join(outdir, ELASTIC_MANIFEST), expect_tool="elastic"
    )
    code = manifest["code"]
    ref_atoms = _reference_atoms(manifest, outdir)
    v_dft = float(abs(np.linalg.det(ref_atoms.cell[:])))
    data = []
    forces = {}  # "reference" and (mode, +-1) -> forces of the strained primitive cell
    n_missing = 0
    geometry_ok = True
    for e in manifest["entries"]:
        if e["dir"] in exclude:
            continue
        out = os.path.join(outdir, e["dir"], manifest["output_file"])
        if not os.path.exists(out):
            if e["index"] == 0:
                raise FileNotFoundError(f"reference calculation output {out} not found")
            n_missing += 1
            continue
        F = deformation_gradient(np.array(e["u"]))
        r = read_dft_output(out, code, image=-1)
        # the observables needed by the fit mode must be present (no silent skipping)
        if mode in ("stress", "both"):
            r.require("stress")
        if mode in ("energy", "both"):
            r.require("energy")
        try:
            check_geometry(r, deform(ref_atoms, F), strict=True)
        except ValueError as exc:
            if not allow_relaxed:
                raise
            geometry_ok = False
            warnings.warn(str(exc), stacklevel=2)
        eta6 = voigt_from_sym(np.array(e["eta"]))
        s6 = None
        if r.stress is not None:
            s6 = voigt_from_sym(ef.second_pk_from_cauchy(r.stress, F))
        label = f"{e['label']} k={e['k']:+d}" if e["index"] else "reference"
        data.append(ef.FitData(label, eta6, r.energy, s6))
        if e["index"] == 0:
            forces["reference"] = r.forces
        elif e["label"] in MODE_NAMES and abs(e["k"]) == 1:
            forces[(e["label"], e["k"])] = r.forces
    if n_missing:
        log(f"  WARNING: {n_missing} calculations have no output yet and were skipped")
    log(f"  {len(data)} configurations read; fitting with mode '{mode}'")
    fit_res = ef.fit_elastic(data, v_dft, mode)
    if fit_res.rank < fit_res.expected_rank:
        hint = " (use --dirset full for energy-only fits)" if mode == "energy" else ""
        raise RuntimeError(
            f"rank-deficient fit: rank {fit_res.rank} < {fit_res.expected_rank}{hint}"
        )
    if fit_res.cond > 1.0e8:
        warnings.warn(
            f"ill-conditioned design matrix (condition number {fit_res.cond:.2e})",
            stacklevel=2,
        )
    if fit_res.block_rms:
        log(
            "  'both' mode: rows weighted by the inverse residual RMS of the single-block fits: "
            + ", ".join(
                f"{k} {v * EV_PER_ANG3_TO_GPA:.3e} GPa"
                for k, v in fit_res.block_rms.items()
            )
        )

    s0 = fit_res.sigma0_full
    c2 = fit_res.c2_full
    c3 = fit_res.c3_full
    changes = None
    if symmetrize:
        rots = cartesian_rotations(ref_atoms, symprec)
        s0s, c2s, c3s = (
            symmetrize_rank2(s0, rots),
            symmetrize_rank4(c2, rots),
            symmetrize_rank6(c3, rots),
        )
        c2s = enforce_intrinsic_symmetry2(c2s)
        c3s = enforce_intrinsic_symmetry3(c3s)
        changes = {
            "sigma0": max_change(s0, s0s),
            "C2": max_change(c2, c2s),
            "C3": max_change(c3, c3s),
        }
        log(f"  point group: {len(rots)} operations")
        s0, c2, c3 = s0s, c2s, c3s

    log("")
    log(ef.format_report(fit_res, s0, c2, c3, changes, min_c3))
    log("")

    ratio, _, prim = anphon_cell_relation(ref_atoms, fcs, anphon_cell, log)

    rdir = os.path.join(outdir, results_dir)
    os.makedirs(rdir, exist_ok=True)
    unsafe = (not geometry_ok) and not force_write
    suffix = ".relaxed_ion.NOT_FOR_ANPHON" if unsafe else ""
    f_ec = os.path.join(rdir, "elastic_constants.in" + suffix)
    f_c1 = os.path.join(rdir, "C1_array.in" + suffix)
    # s0, c2, c3 are densities in eV/A^3; the files store GPa (cell-independent)
    write_elastic_constants_in(
        f_ec,
        ef.full2_to_9x9(c2) * EV_PER_ANG3_TO_GPA,
        ef.full3_to_9x9x9(c3) * EV_PER_ANG3_TO_GPA,
        unit="GPa",
    )
    write_C1_array_in(f_c1, s0 * EV_PER_ANG3_TO_GPA, unit="GPa")
    blocks = _strain_force_blocks(forces, manifest["smag"], ref_atoms, prim, log)
    f_sf = None
    if blocks:
        f_sf = os.path.join(rdir, "strain_force.in" + suffix)
        write_strain_force_in(f_sf, blocks, reference_cell=prim)
    summary = {
        "fit_mode": mode,
        "rank": fit_res.rank,
        "expected_rank": fit_res.expected_rank,
        "condition_number": fit_res.cond,
        "rms_energy_eV_per_A3": fit_res.rms_energy,
        "rms_stress_eV_per_A3": fit_res.rms_stress,
        "volume_dft_A3": v_dft,
        "volume_anphon_over_dft": ratio,
        "file_unit": "GPa",
        "symmetrization_max_change_eV_per_A3": changes,
        "sigma0_GPa": (s0 * EV_PER_ANG3_TO_GPA).tolist(),
        "C2_voigt_GPa": (ef.voigt66(c2) * EV_PER_ANG3_TO_GPA).tolist(),
        "C3_voigt_GPa": {
            f"{a + 1}{b + 1}{c + 1}": float(v * EV_PER_ANG3_TO_GPA)
            for (a, b, c), v in zip(ef.IDX3, ef.full3_to_voigt(c3))
        },
        "eta_list": [d.eta6.tolist() for d in data],
        "geometry_clamped": geometry_ok,
    }
    with open(os.path.join(rdir, "elastic_fit.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(
        f"  written: {f_ec}\n           {f_c1}\n           {os.path.join(rdir, 'elastic_fit.json')}"
        + (
            f"\n           {f_sf} (strain-force coupling, central difference)"
            if f_sf
            else ""
        )
    )
    log(
        "  units: GPa (cell-independent; anphon multiplies by the volume of its own primitive cell)"
    )
    if unsafe:
        log(
            "  WARNING: the DFT outputs are not clamped-ion/fixed-cell; the files are marked "
            "NOT_FOR_ANPHON (use --force-write to override)"
        )
    if strain_file:
        if unsafe:
            log(
                f"  NOTE: relaxed-ion data are not written to {strain_file} (use --force-write to override)"
            )
        else:
            _write_container(
                strain_file,
                force,
                ref_atoms,
                prim,
                s0,
                c2,
                c3,
                fit_res,
                manifest,
                mode,
                symmetrize,
                geometry_ok,
                blocks,
                outdir,
                rdir,
                log,
            )
    else:
        log(anphon_file_locations())
        log(
            f"  e.g.:  cp {f_ec} {f_sf or ''} <STRAIN_IFC_DIR>/ ;  cp {f_c1} <anphon working directory>/"
        )
    if compare:
        log(compare_with_anphon_log(compare, c2))
    return fit_res, summary


def _strain_force_blocks(forces, smag, ref_atoms, prim, log):
    """strain_force.in blocks (mode, +-smag, 1/2, F - F0) from the k = +-1 single-mode runs.

    The single-mode points of every direction set are exactly the strained
    primitive cells of ``strainifc.py --coupling force --central``, so the
    strain-force coupling comes for free.  Returns None (with a note) when a
    needed run is missing or has no forces.
    """
    need = ["reference"] + [(m, k) for m in MODE_NAMES for k in (1, -1)]
    missing = [k for k in need if forces.get(k) is None]
    if missing:
        log(
            "  NOTE: strain-force coupling not written (no forces for "
            + ", ".join(str(k) for k in missing[:4])
            + (", ..." if len(missing) > 4 else "")
            + ")"
        )
        return None
    f0 = np.asarray(forces["reference"])
    fmax = float(np.abs(f0).max())
    log(f"  reference (unstrained) forces: max |F| = {fmax:.3e} eV/A (subtracted)")
    if fmax > 1.0e-3:
        warnings.warn(
            "the reference structure is not well relaxed (max |F| > 1e-3 eV/A); "
            "the strain-force coupling then depends on the residual forces",
            stacklevel=3,
        )
    mapping = None
    if prim is not None:
        mapping = match_primitive_atoms(ref_atoms, prim)
        log(describe_ordering(prim, mapping, ref_atoms))
    blocks = []
    for m in MODE_NAMES:
        for k in (1, -1):
            df = np.asarray(forces[(m, k)]) - f0
            if mapping is not None:
                df = df[mapping]
            blocks.append((m, k * float(smag), 0.5, df))
    return blocks


def _write_container(
    strain_file,
    force,
    ref_atoms,
    prim,
    s0,
    c2,
    c3,
    fit_res,
    manifest,
    mode,
    symmetrize,
    geometry_ok,
    blocks,
    outdir,
    rdir,
    log,
):
    """Add /Elastic (and /StrainForce) to the strain-coupling container."""
    from . import strainfile as sfile

    if prim is not None:
        # The DFT reference structure must be the crystal of the anphon cell
        # before the container is labeled with the anphon cell.
        try:
            sfile.same_crystal(
                sfile.cell_from_atoms(ref_atoms), sfile.cell_from_primitive(prim)
            )
        except ValueError as exc:
            raise ValueError(
                f"the DFT reference structure is not the crystal of the anphon cell: {exc}"
            ) from None
        reference, ref_source = sfile.cell_from_primitive(prim), prim.source
    else:
        reference, ref_source = (
            sfile.cell_from_atoms(ref_atoms),
            "DFT reference structure",
        )
    g = EV_PER_ANG3_TO_GPA
    attrs = {
        "source": "elastic.py fit",
        "dft_code": manifest["code"],
        "fit_mode": mode,
        "smag": manifest.get("smag"),
        "nmag": manifest.get("nmag"),
        "dirset": manifest.get("dirset"),
        "symmetrized": bool(symmetrize),
        "rank": fit_res.rank,
        "expected_rank": fit_res.expected_rank,
        "condition_number": fit_res.cond,
        "rms_stress_GPa": fit_res.rms_stress * g,
        "rms_energy_GPa": fit_res.rms_energy * g,
        "geometry_clamped": bool(geometry_ok),
        "stress_source": "fit",
    }
    rec = sfile.provenance_record(
        "Elastic+StrainForce" if blocks else "Elastic",
        manifest=os.path.join(outdir, ELASTIC_MANIFEST),
        results=rdir,
    )
    with sfile.update(strain_file, reference, rec, force, ref_source) as f:
        sfile.write_elastic(
            f, s0 * g, ef.full2_to_9x9(c2) * g, ef.full3_to_9x9x9(c3) * g, attrs
        )
        if blocks:
            sfile.write_strain_force(
                f,
                blocks,
                reference,
                {
                    "source": "elastic.py fit",
                    "dft_code": manifest["code"],
                    "central": True,
                },
            )
    log(
        f"  written: {strain_file} (/Elastic: reference stress, C2, C3 in GPa"
        + ("; /StrainForce: 12 blocks" if blocks else "")
        + ")"
    )
    log("  give it to anphon as STRAINFILE in the &relax field")


def parse_anphon_elastic_log(path):
    """6x6 Voigt table (GPa) printed by anphon when the elastic constants are
    computed from the IFCs (STRAIN_COUPLING bit 1 clear)."""
    with open(path, errors="replace") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "second-order elastic constants" in line and "GPa" in line:
            rows = []
            for k in range(1, 7):
                rows.append([float(t) for t in lines[i + k].split()[:6]])
            return np.array(rows)
    raise ValueError(
        f"{path}: no elastic-constant table found (STRAIN_COUPLING bit 1 clear)"
    )


def compare_with_anphon_log(path, c2_full):
    ref = parse_anphon_elastic_log(path)
    mine = ef.voigt66(c2_full) * EV_PER_ANG3_TO_GPA
    out = [
        f"Comparison with the IFC-derived elastic constants in {path} (GPa, this fit - anphon):"
    ]
    names = ["xx", "yy", "zz", "yz", "zx", "xy"]
    out.append("        " + "".join(f"{n:>11s}" for n in names))
    for a in range(6):
        out.append(
            f"  {names[a]:>4s}  "
            + "".join(f"{mine[a, b] - ref[a, b]:11.3f}" for b in range(6))
        )
    return "\n".join(out)


def show(path, volume_A3=None, structure=None, c1_path=None, min_c3=0.5):
    """Pretty-print an elastic_constants.in (and C1_array.in) in GPa.

    GPa files need no volume.  Legacy files (V*C in Ry) need the volume of the
    cell they were written for (``volume_A3`` or ``structure``).
    """
    from .units import ry_to_c_ev_per_ang3
    from .writers import read_C1_array_in, read_elastic_constants_in

    if structure is not None:
        import ase.io

        volume_A3 = float(abs(np.linalg.det(ase.io.read(structure).cell[:])))

    def to_ev_per_ang3(values, unit, what):
        if unit == "GPa":
            return values / EV_PER_ANG3_TO_GPA
        if volume_A3 is None:
            raise ValueError(
                f"{what} is in the legacy per-cell unit (V*C in Ry): the cell volume is "
                "needed (--volume or --structure)"
            )
        return ry_to_c_ev_per_ang3(values, volume_A3)

    c2, c3, unit = read_elastic_constants_in(path)
    c2_ev = to_ev_per_ang3(ef.from_9x9(c2), unit, path)
    c3_ev = to_ev_per_ang3(ef.from_9x9x9(c3), unit, path)
    head = f"{path}  (unit in file: {unit}"
    if unit != "GPa":
        head += f", V = {volume_A3:.4f} A^3"
    out = [head + ")"]
    out.append(
        ef.voigt_table_gpa(
            c2_ev, "Second-order elastic constants (GPa, Voigt notation):"
        )
    )
    out.append(f"Third-order elastic constants (GPa, |C| >= {min_c3}):")
    out.append(ef.format_c3_gpa(ef.full3_to_voigt(c3_ev), min_c3))
    if c1_path:
        s, unit_c1 = read_C1_array_in(c1_path)
        s = to_ev_per_ang3(s, unit_c1, c1_path) * EV_PER_ANG3_TO_GPA
        out.append(f"{c1_path}: reference stress (GPa; unit in file: {unit_c1})")
        for i in range(3):
            out.append("   " + "".join(f"{s[i, j]:11.4f}" for j in range(3)))
    return "\n".join(out)
