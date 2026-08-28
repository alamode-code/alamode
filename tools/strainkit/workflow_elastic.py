"""DFT finite-strain workflow for the elastic constants (generate / fit)."""

import json
import os
import re
import shutil
import warnings

import numpy as np

from . import elasticfit as ef
from .dftio import check_geometry, read_dft_output
from .fcsorder import (anphon_primitive_cell, lattice_relation, read_anphon_cell,
                       read_fcs_structure)
from .manifest import ELASTIC_MANIFEST, load_manifest, save_manifest
from .strain import deform, deformation_gradient, sym_from_voigt, voigt_from_sym
from .structure import OUTPUT_FILENAME, normalize_code, read_template, write_structure
from .symmetry import (cartesian_rotations, enforce_intrinsic_symmetry2,
                       enforce_intrinsic_symmetry3, max_change, symmetrize_rank2,
                       symmetrize_rank4, symmetrize_rank6)
from .units import EV_PER_ANG3_TO_GPA, volume_times_c_to_ry
from .writers import (anphon_file_locations, write_C1_array_in,
                      write_elastic_constants_in)
from .jobscript import render_job_script, write_run_all

RESULTS_DIR = "results"


def _dirname(index):
    return f"strain_{index:03d}"


def generate(code, template_dir, outdir=".", structure_file=None, smag=0.01, nmag=2,
             dirset="minimal", job_template=None, dft_command=None, force=False, log=print):
    code = normalize_code(code)
    template = read_template(code, template_dir, structure_file)
    points = ef.strain_points(ef.direction_set(dirset), smag, nmag)
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    existing = [p for p in points if os.path.exists(os.path.join(outdir, _dirname(p.index)))]
    if existing and not force:
        raise FileExistsError(
            f"{len(existing)} strain_* directories already exist in {outdir}; use --force to overwrite")
    for p in existing:
        shutil.rmtree(os.path.join(outdir, _dirname(p.index)))

    ref_dir = os.path.join(outdir, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    write_structure(template, template.atoms, ref_dir, copy_extra=False)

    entries = []
    for p in points:
        d = os.path.join(outdir, _dirname(p.index))
        write_structure(template, deform(template.atoms, p.F), d)
        entries.append({"index": p.index, "dir": _dirname(p.index), "label": p.label,
                        "k": p.k, "d6": p.d6, "u": p.u, "eta": p.eta})
    subdirs = [e["dir"] for e in entries]
    dft_text = open(dft_command).read() if dft_command else None
    if job_template:
        with open(os.path.join(outdir, "job.sh"), "w") as f:
            f.write(render_job_script(open(job_template).read(), dft_text or "", subdirs))
    else:
        write_run_all(os.path.join(outdir, "run_all.sh"), subdirs, dft_text)

    manifest = {
        "tool": "elastic", "code": code, "template_dir": template.template_dir,
        "structure_file": template.structure_name, "output_file": OUTPUT_FILENAME[code],
        "smag": float(smag), "nmag": int(nmag), "dirset": dirset,
        "volume_A3": float(abs(np.linalg.det(template.atoms.cell[:]))),
        "nat": len(template.atoms), "numbers": np.asarray(template.atoms.numbers).tolist(),
        "entries": entries,
    }
    save_manifest(manifest, os.path.join(outdir, ELASTIC_MANIFEST))
    log(f"Generated {len(points)} calculations ({len(points) - 1} strained + 1 reference) in {outdir}")
    log(f"  direction set: {dirset} ({len(ef.direction_set(dirset))} directions), "
        f"magnitudes: +-{smag} x 1..{nmag}")
    log("  Run the DFT code in every strain_*/ directory (fixed cell, fixed ions; clamped-ion).")
    log(f"  Expected output file per directory: {OUTPUT_FILENAME[code]}")
    log(f"  Manifest: {os.path.join(outdir, ELASTIC_MANIFEST)}")
    return manifest


def _reference_atoms(manifest, outdir):
    import ase.io
    from .structure import _ase_read
    fmt = {"VASP": "vasp", "QE": "espresso-in", "ase": None}[manifest["code"]]
    return _ase_read(os.path.join(outdir, "reference", manifest["structure_file"]), fmt)


def target_cell_scale(ref_atoms, fcs=None, anphon_cell=None, log=print):
    """Volume ratio V_anphon / V_dft (and the anphon lattice) for the file units."""
    a_dft = np.asarray(ref_atoms.cell[:], dtype=float)
    if fcs is None and anphon_cell is None:
        return 1.0, None
    if fcs is not None:
        fcs_struct = read_fcs_structure(fcs)
        cell = read_anphon_cell(anphon_cell) if anphon_cell else None
        prim = anphon_primitive_cell(fcs_struct, cell)
        lav = prim.lavec
    else:
        lav = read_anphon_cell(anphon_cell)
    m = lattice_relation(lav, a_dft)
    scale = abs(int(round(np.linalg.det(m))))
    log(f"  anphon cell = M x DFT cell with det(M) = {scale}; V*C values are scaled accordingly")
    return float(scale), lav


def fit(outdir=".", mode="stress", fcs=None, anphon_cell=None, allow_relaxed=False,
        force_write=False, symmetrize=True, symprec=1.0e-5, results_dir=RESULTS_DIR,
        compare=None, min_c3=0.5, exclude=(), log=print):
    outdir = os.path.abspath(outdir)
    manifest = load_manifest(os.path.join(outdir, ELASTIC_MANIFEST), expect_tool="elastic")
    code = manifest["code"]
    ref_atoms = _reference_atoms(manifest, outdir)
    v_dft = float(abs(np.linalg.det(ref_atoms.cell[:])))
    data = []
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
        try:
            check_geometry(r, deform(ref_atoms, F), strict=True)
        except ValueError as exc:
            if not allow_relaxed:
                raise
            geometry_ok = False
            warnings.warn(str(exc))
        eta6 = voigt_from_sym(np.array(e["eta"]))
        s6 = None
        if r.stress is not None:
            s6 = voigt_from_sym(ef.second_pk_from_cauchy(r.stress, F))
        label = f"{e['label']} k={e['k']:+d}" if e["index"] else "reference"
        data.append(ef.FitData(label, eta6, r.energy, s6))
    if n_missing:
        log(f"  WARNING: {n_missing} calculations have no output yet and were skipped")
    log(f"  {len(data)} configurations read; fitting with mode '{mode}'")
    fit_res = ef.fit_elastic(data, v_dft, mode)
    if fit_res.rank < fit_res.expected_rank:
        hint = " (use --dirset full for energy-only fits)" if mode == "energy" else ""
        raise RuntimeError(f"rank-deficient fit: rank {fit_res.rank} < {fit_res.expected_rank}{hint}")
    if fit_res.cond > 1.0e8:
        warnings.warn(f"ill-conditioned design matrix (condition number {fit_res.cond:.2e})")

    s0 = fit_res.sigma0_full
    c2 = fit_res.c2_full
    c3 = fit_res.c3_full
    changes = None
    if symmetrize:
        rots = cartesian_rotations(ref_atoms, symprec)
        s0s, c2s, c3s = symmetrize_rank2(s0, rots), symmetrize_rank4(c2, rots), symmetrize_rank6(c3, rots)
        c2s = enforce_intrinsic_symmetry2(c2s)
        c3s = enforce_intrinsic_symmetry3(c3s)
        changes = {"sigma0": max_change(s0, s0s), "C2": max_change(c2, c2s), "C3": max_change(c3, c3s)}
        log(f"  point group: {len(rots)} operations")
        s0, c2, c3 = s0s, c2s, c3s

    log("")
    log(ef.format_report(fit_res, s0, c2, c3, changes, min_c3))
    log("")

    scale, lav = target_cell_scale(ref_atoms, fcs, anphon_cell, log)
    v_target = v_dft * scale
    if scale == 1.0 and fcs is None and anphon_cell is None:
        log("  NOTE: no anphon cell given (--fcs/--anphon-cell); the files assume the anphon "
            f"primitive cell has the volume of the DFT cell ({v_dft:.4f} A^3)")

    rdir = os.path.join(outdir, results_dir)
    os.makedirs(rdir, exist_ok=True)
    unsafe = (not geometry_ok) and not force_write
    suffix = ".relaxed_ion.NOT_FOR_ANPHON" if unsafe else ""
    f_ec = os.path.join(rdir, "elastic_constants.in" + suffix)
    f_c1 = os.path.join(rdir, "C1_array.in" + suffix)
    write_elastic_constants_in(f_ec, volume_times_c_to_ry(v_target, ef.full2_to_9x9(c2)),
                               volume_times_c_to_ry(v_target, ef.full3_to_9x9x9(c3)))
    write_C1_array_in(f_c1, volume_times_c_to_ry(v_target, s0))
    summary = {
        "fit_mode": mode, "rank": fit_res.rank, "expected_rank": fit_res.expected_rank,
        "condition_number": fit_res.cond, "rms_energy_eV_per_A3": fit_res.rms_energy,
        "rms_stress_eV_per_A3": fit_res.rms_stress, "volume_dft_A3": v_dft,
        "volume_anphon_A3": v_target, "symmetrization_max_change_eV_per_A3": changes,
        "sigma0_GPa": (s0 * EV_PER_ANG3_TO_GPA).tolist(),
        "C2_voigt_GPa": (ef.voigt66(c2) * EV_PER_ANG3_TO_GPA).tolist(),
        "C3_voigt_GPa": {f"{a + 1}{b + 1}{c + 1}": float(v * EV_PER_ANG3_TO_GPA)
                         for (a, b, c), v in zip(ef.IDX3, ef.full3_to_voigt(c3))},
        "eta_list": [d.eta6.tolist() for d in data], "geometry_clamped": geometry_ok,
    }
    with open(os.path.join(rdir, "elastic_fit.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  written: {f_ec}\n           {f_c1}\n           {os.path.join(rdir, 'elastic_fit.json')}")
    if unsafe:
        log("  WARNING: the DFT outputs are not clamped-ion/fixed-cell; the files are marked "
            "NOT_FOR_ANPHON (use --force-write to override)")
    log(anphon_file_locations())
    log(f"  e.g.:  cp {f_ec} <STRAIN_IFC_DIR>/ ;  cp {f_c1} <anphon working directory>/")
    if compare:
        log(compare_with_anphon_log(compare, c2))
    return fit_res, summary


def parse_anphon_elastic_log(path):
    """6x6 Voigt table (GPa) printed by anphon for ELASTIC_CONST = 1."""
    with open(path, errors="replace") as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        if "second-order elastic constants" in l and "GPa" in l:
            rows = []
            for k in range(1, 7):
                rows.append([float(t) for t in lines[i + k].split()[:6]])
            return np.array(rows)
    raise ValueError(f"{path}: no ELASTIC_CONST = 1 table found")


def compare_with_anphon_log(path, c2_full):
    ref = parse_anphon_elastic_log(path)
    mine = ef.voigt66(c2_full) * EV_PER_ANG3_TO_GPA
    out = [f"Comparison with the ELASTIC_CONST = 1 table in {path} (GPa, this fit - anphon):"]
    names = ["xx", "yy", "zz", "yz", "zx", "xy"]
    out.append("        " + "".join(f"{n:>11s}" for n in names))
    for a in range(6):
        out.append(f"  {names[a]:>4s}  " + "".join(f"{mine[a, b] - ref[a, b]:11.3f}" for b in range(6)))
    return "\n".join(out)


def show(path, volume_A3=None, structure=None, c1_path=None, min_c3=0.5):
    """Pretty-print an elastic_constants.in (and C1_array.in) in GPa."""
    from .units import ry_to_c_ev_per_ang3
    from .writers import read_C1_array_in, read_elastic_constants_in
    if structure is not None:
        import ase.io
        volume_A3 = float(abs(np.linalg.det(ase.io.read(structure).cell[:])))
    if volume_A3 is None:
        raise ValueError("the cell volume is needed (--volume or --structure)")
    c2, c3 = read_elastic_constants_in(path)
    out = [f"{path}  (V = {volume_A3:.4f} A^3)"]
    out.append(ef.voigt_table_gpa(ry_to_c_ev_per_ang3(ef.from_9x9(c2), volume_A3),
                                  "Second-order elastic constants (GPa, Voigt notation):"))
    out.append(f"Third-order elastic constants (GPa, |C| >= {min_c3}):")
    out.append(ef.format_c3_gpa(ef.full3_to_voigt(ry_to_c_ev_per_ang3(ef.from_9x9x9(c3), volume_A3)), min_c3))
    if c1_path:
        s = ry_to_c_ev_per_ang3(read_C1_array_in(c1_path), volume_A3) * EV_PER_ANG3_TO_GPA
        out.append(f"{c1_path}: reference stress (GPa)")
        for i in range(3):
            out.append("   " + "".join(f"{s[i, j]:11.4f}" for j in range(3)))
    return "\n".join(out)
