# Copyright (c) 2023 Ryota Masuki (strainIFCcoupling,
#                    https://github.com/r-masuki/strainIFCcoupling)
# Copyright (c) 2026 Terumasa Tadano
# MIT license.  See LICENCE.txt of the ALAMODE package.
"""Strain-IFC coupling workflow (generate / collect / check).

* coupling = "harmonic": harmonic force constants of strained supercells
  -> strain_harmonic.in + one force-constant file per strain (RENORM_3TO2ND = 2/3)
* coupling = "force": forces in strained primitive cells
  -> strain_force.in (RENORM_2TO1ST = 2)
"""

import os
import shutil
import warnings

import numpy as np

from .dfset import build_dfset, write_dfset
from .dftio import check_geometry, read_dft_output
from .fcsorder import (
    anphon_primitive_cell,
    check_supercell_equivalence,
    describe_ordering,
    match_primitive_atoms,
    read_anphon_cell,
    read_fcs_structure,
    verify_generated_fcs,
)
from .jobscript import render_job_script, write_run_all
from .manifest import IFC_MANIFEST, load_manifest, save_manifest
from .strain import (
    StrainPoint,
    check_weight_sums,
    default_modes,
    deform,
    deformation_gradient,
    ifc_strain_set,
    load_modes_json,
    modes_from_names,
)
from .structure import OUTPUT_FILENAME, normalize_code, read_template, write_structure
from .writers import (
    anphon_file_locations,
    write_strain_force_in,
    write_strain_harmonic_in,
)

RESULTS_DIR = "results"
COUPLINGS = ("harmonic", "force")


def _dirname(index):
    return f"strain_{index:03d}"


def _points_from_manifest(manifest):
    return [
        StrainPoint(
            e["label"],
            e["mode"],
            float(e["smag"]),
            float(e["weight"]),
            np.array(e["u"]),
        )
        for e in manifest["entries"]
    ]


def generate(
    coupling,
    code,
    template_dir,
    outdir=".",
    structure_file=None,
    smag=None,
    dmag=0.01,
    central=False,
    no_offset=False,
    modes_json=None,
    mode_names=None,
    nbody=2,
    cutoff=None,
    job_template=None,
    dft_command=None,
    copy_potcar=False,
    force=False,
    log=print,
):
    if coupling not in COUPLINGS:
        raise ValueError("coupling must be 'harmonic' or 'force'")
    code = normalize_code(code)
    template = read_template(code, template_dir, structure_file)
    if modes_json:
        modes = load_modes_json(modes_json)
    elif mode_names:
        modes = modes_from_names(mode_names)
    else:
        modes = default_modes()
    points = ifc_strain_set(modes, smag, central)
    try:
        check_weight_sums(points, require_all=(coupling == "force"))
    except ValueError as exc:
        if coupling == "force":
            raise ValueError(
                "strain_force.in needs all six strain modes (anphon requires every component "
                "to be covered): " + str(exc)
            ) from None
        warnings.warn(
            str(exc)
            + " -- strain_harmonic.in will be usable only with RENORM_3TO2ND = 3"
        )

    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    n_dirs = len(points) + (1 if coupling == "force" else 0)
    first = 0 if coupling == "force" else 1
    existing = [
        os.path.join(outdir, _dirname(k))
        for k in range(first, first + n_dirs)
        if os.path.exists(os.path.join(outdir, _dirname(k)))
    ]
    if existing and not force:
        raise FileExistsError(
            f"{len(existing)} strain_* directories already exist in {outdir}; use --force to overwrite"
        )
    for d in existing:
        shutil.rmtree(d)
    ref_dir = os.path.join(outdir, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    write_structure(template, template.atoms, ref_dir, copy_extra=False)

    dft_text = open(dft_command).read() if dft_command else None
    job_text = open(job_template).read() if job_template else None
    entries = []
    all_calc_dirs = []
    if coupling == "harmonic":
        from .almfit import displaced_structures, suggest_harmonic_patterns

        for k, p in enumerate(points, start=1):
            dname = _dirname(k)
            d = os.path.join(outdir, dname)
            strained = deform(template.atoms, p.F)
            patterns = suggest_harmonic_patterns(strained, nbody, cutoff)
            structs = displaced_structures(strained, patterns, dmag)
            width = max(2, len(str(len(structs))) + 1)
            # undisplaced strained cell (offset calculation)
            write_structure(
                template,
                strained,
                os.path.join(d, "nodisp"),
                link_potcar=not copy_potcar,
            )
            disp_dirs = []
            for m, s in enumerate(structs, start=1):
                sub = f"disp_{m:0{width}d}"
                write_structure(
                    template, s, os.path.join(d, sub), link_potcar=not copy_potcar
                )
                disp_dirs.append(sub)
            subdirs = ([] if no_offset else ["nodisp"]) + disp_dirs
            if job_text:
                with open(os.path.join(d, "job.sh"), "w") as f:
                    f.write(render_job_script(job_text, dft_text or "", subdirs))
            all_calc_dirs += [os.path.join(dname, s) for s in subdirs]
            entries.append(
                {
                    "index": k,
                    "dir": dname,
                    "label": p.label,
                    "mode": p.mode,
                    "smag": p.smag,
                    "weight": p.weight,
                    "u": p.u,
                    "n_disp": len(structs),
                    "disp_dirs": disp_dirs,
                    "nodisp_dir": None if no_offset else "nodisp",
                    "patterns": [
                        [[int(i), np.asarray(v).tolist()] for i, v, _ in pat]
                        for pat in patterns
                    ],
                }
            )
            log(
                f"  {dname}: mode {p.mode:>3s} smag {p.smag:+.5f} weight {p.weight:g}  "
                f"-> {len(structs)} displacement patterns"
                + ("" if no_offset else " + nodisp")
            )
    else:
        # reference (unstrained) primitive cell: strain_000
        write_structure(
            template,
            template.atoms,
            os.path.join(outdir, _dirname(0), "primitive"),
            link_potcar=not copy_potcar,
        )
        all_calc_dirs.append(os.path.join(_dirname(0), "primitive"))
        if job_text:
            with open(os.path.join(outdir, _dirname(0), "job.sh"), "w") as f:
                f.write(render_job_script(job_text, dft_text or "", ["primitive"]))
        for k, p in enumerate(points, start=1):
            dname = _dirname(k)
            d = os.path.join(outdir, dname)
            write_structure(
                template,
                deform(template.atoms, p.F),
                os.path.join(d, "primitive"),
                link_potcar=not copy_potcar,
            )
            if job_text:
                with open(os.path.join(d, "job.sh"), "w") as f:
                    f.write(render_job_script(job_text, dft_text or "", ["primitive"]))
            all_calc_dirs.append(os.path.join(dname, "primitive"))
            entries.append(
                {
                    "index": k,
                    "dir": dname,
                    "label": p.label,
                    "mode": p.mode,
                    "smag": p.smag,
                    "weight": p.weight,
                    "u": p.u,
                    "primitive_dir": "primitive",
                }
            )
            log(f"  {dname}: mode {p.mode:>3s} smag {p.smag:+.5f} weight {p.weight:g}")
    write_run_all(os.path.join(outdir, "run_all.sh"), all_calc_dirs, dft_text)

    manifest = {
        "tool": "strainifc",
        "code": code,
        "coupling": coupling,
        "template_dir": template.template_dir,
        "structure_file": template.structure_name,
        "output_file": OUTPUT_FILENAME[code],
        "dmag": float(dmag),
        "central": bool(central),
        "offset": not no_offset,
        "nbody": int(nbody),
        "cutoff": None if cutoff is None else float(cutoff),
        "nat": len(template.atoms),
        "numbers": np.asarray(template.atoms.numbers).tolist(),
        "entries": entries,
    }
    save_manifest(manifest, os.path.join(outdir, IFC_MANIFEST))
    log(f"Generated {len(all_calc_dirs)} calculation directories in {outdir}")
    log(f"  expected output file per directory: {OUTPUT_FILENAME[code]}")
    log(f"  manifest: {os.path.join(outdir, IFC_MANIFEST)}")
    log(
        "  NOTE: the atom order of the template is kept in every generated structure; anphon's "
        "&cell / FC2FILE must use the same order (run 'strainifc.py check --fcs ...')."
    )
    return manifest


def _reference_atoms(manifest, outdir):
    from .structure import _ase_read

    fmt = {"VASP": "vasp", "QE": "espresso-in", "ase": None}[manifest["code"]]
    return _ase_read(os.path.join(outdir, "reference", manifest["structure_file"]), fmt)


def _read(path, code, last):
    if not os.path.exists(path):
        raise FileNotFoundError(f"DFT output {path} not found")
    r = read_dft_output(path, code, image=-1)
    if r.nimages != 1 and not last:
        raise ValueError(
            f"{path}: contains {r.nimages} ionic steps; a single-point calculation is "
            "expected (use --last to take the last step at your own risk)"
        )
    r.require("forces")
    return r


def collect_harmonic(
    outdir,
    manifest,
    fcs=None,
    unchecked=False,
    fcs_format="xml",
    prefix="strain",
    results_dir=RESULTS_DIR,
    last=False,
    solver="dense",
    write_dfset_files=False,
    log=print,
):
    from .almfit import displaced_structures, fit_harmonic

    code = manifest["code"]
    ref_atoms = _reference_atoms(manifest, outdir)
    fcs_struct = None
    if fcs:
        fcs_struct = read_fcs_structure(fcs)
        check_supercell_equivalence(ref_atoms, fcs_struct)
        log(f"  template supercell is index-compatible with {fcs}")
    elif not unchecked:
        raise ValueError(
            "give the reference force-constant file used by anphon (--fcs REF.xml|.h5) "
            "or --unchecked to skip the index-compatibility checks"
        )
    rdir = os.path.join(outdir, results_dir)
    os.makedirs(rdir, exist_ok=True)
    rows = []
    for e in manifest["entries"]:
        F = deformation_gradient(np.array(e["u"]))
        strained = deform(ref_atoms, F)
        d = os.path.join(outdir, e["dir"])
        offset = None
        if e.get("nodisp_dir"):
            offset = _read(
                os.path.join(d, e["nodisp_dir"], manifest["output_file"]), code, last
            )
            check_geometry(offset, strained)
        patterns = [
            [(i, np.array(v), "Cartesian") for i, v in pat] for pat in e["patterns"]
        ]
        expected = displaced_structures(strained, patterns, manifest["dmag"])
        results = []
        for sub, exp in zip(e["disp_dirs"], expected):
            r = _read(os.path.join(d, sub, manifest["output_file"]), code, last)
            check_geometry(r, exp)
            results.append(r)
        u, f = build_dfset(results, strained, offset, manifest["dmag"])
        if write_dfset_files:
            write_dfset(os.path.join(rdir, f"DFSET_{e['dir']}"), u, f, e["disp_dirs"])
        out_name = f"{prefix}_{e['index']:03d}.{fcs_format}"
        out = os.path.join(rdir, out_name)
        info = fit_harmonic(
            strained,
            u,
            f,
            out,
            fcs_format,
            manifest["nbody"],
            manifest["cutoff"],
            solver,
        )
        if fcs_struct is not None:
            verify_generated_fcs(out, fcs_struct, F)
        rows.append((e["mode"], e["smag"], e["weight"], out_name))
        log(
            f"  {e['dir']}: {info['nsnap']} snapshots, {info['n_free_parameters']} free parameters -> {out_name}"
        )
    fname = os.path.join(rdir, "strain_harmonic.in")
    write_strain_harmonic_in(fname, rows)
    try:
        check_weight_sums(_points_from_manifest(manifest), require_all=True)
        log("  weight sums are 1 for all components (RENORM_3TO2ND = 2 or 3)")
    except ValueError:
        check_weight_sums(_points_from_manifest(manifest), require_all=False)
        log(
            "  NOTE: not all strain components are covered; use RENORM_3TO2ND = 3 (symmetry completion)"
        )
    log(f"  written: {fname} (+ {len(rows)} force-constant files in {rdir})")
    log(anphon_file_locations())
    return fname


def _anphon_mapping(ref_atoms, fcs, anphon_cell, reorder, log):
    """Mapping anphon-primitive-atom -> DFT-cell atom, or None (DFT order used)."""
    if fcs is None:
        if anphon_cell is not None:
            raise ValueError(
                "--anphon-cell requires --fcs (anphon folds the FCS supercell into &cell)"
            )
        log(
            "  NOTE: no --fcs given; strain_force.in rows are written in the order of the template "
            "structure, which must be anphon's primitive-cell atom order"
        )
        return None
    fcs_struct = read_fcs_structure(fcs)
    cell = read_anphon_cell(anphon_cell) if anphon_cell else None
    prim = anphon_primitive_cell(fcs_struct, cell)
    mapping = match_primitive_atoms(ref_atoms, prim)
    log(describe_ordering(prim, mapping, ref_atoms))
    n = len(mapping)
    if n == len(ref_atoms):
        if not np.array_equal(mapping, np.arange(n)):
            msg = (
                "the atom order of the DFT primitive cell differs from anphon's primitive-cell order "
                f"(permutation {(mapping + 1).tolist()}); reorder the template accordingly"
            )
            if not reorder:
                raise ValueError(
                    msg + " or pass --reorder to let the tool permute the force rows"
                )
            log("  WARNING: " + msg + " -- rows are permuted (--reorder)")
    elif n > len(ref_atoms):
        log(
            f"  anphon cell contains {n // len(ref_atoms)} copies of the DFT cell; force rows are tiled "
            "(requires full translational symmetry of the DFT setup, e.g. no magnetic order enlarging the cell)"
        )
    else:
        log(
            f"  the DFT cell contains {len(ref_atoms) // n} copies of the anphon cell; one translation image "
            "per anphon atom is used"
        )
    return mapping


def collect_force(
    outdir,
    manifest,
    fcs=None,
    anphon_cell=None,
    results_dir=RESULTS_DIR,
    last=False,
    reorder=False,
    log=print,
):
    code = manifest["code"]
    ref_atoms = _reference_atoms(manifest, outdir)
    mapping = _anphon_mapping(ref_atoms, fcs, anphon_cell, reorder, log)
    ref_out = os.path.join(outdir, _dirname(0), "primitive", manifest["output_file"])
    f0 = np.zeros((len(ref_atoms), 3))
    if os.path.exists(ref_out):
        r0 = _read(ref_out, code, last)
        check_geometry(r0, ref_atoms)
        f0 = r0.forces
        fmax = float(np.abs(f0).max())
        log(f"  reference (unstrained) forces: max |F| = {fmax:.3e} eV/A (subtracted)")
        if fmax > 1.0e-3:
            warnings.warn(
                "the reference structure is not well relaxed (max |F| > 1e-3 eV/A); "
                "the strain-force coupling then depends on the residual forces"
            )
    else:
        log(
            "  NOTE: no reference calculation output (strain_000) found; forces of the unstrained cell "
            "are assumed to vanish"
        )
    blocks = []
    for e in manifest["entries"]:
        F = deformation_gradient(np.array(e["u"]))
        r = _read(
            os.path.join(outdir, e["dir"], e["primitive_dir"], manifest["output_file"]),
            code,
            last,
        )
        check_geometry(r, deform(ref_atoms, F))
        forces = np.asarray(r.forces) - f0
        if mapping is not None:
            forces = forces[mapping]
        blocks.append((e["mode"], e["smag"], e["weight"], forces))
        log(
            f"  {e['dir']}: mode {e['mode']:>3s} smag {e['smag']:+.5f}  max |F - F0| = {np.abs(forces).max():.3e} eV/A"
        )
    # anphon requires every strain component to be covered with weights summing to 1
    check_weight_sums(_points_from_manifest(manifest), require_all=True)
    rdir = os.path.join(outdir, results_dir)
    os.makedirs(rdir, exist_ok=True)
    fname = os.path.join(rdir, "strain_force.in")
    write_strain_force_in(fname, blocks)
    log(
        f"  written: {fname} ({len(blocks)} blocks x {blocks[0][3].shape[0]} atoms, eV/A)"
    )
    log(anphon_file_locations())
    return fname


def collect(
    outdir=".",
    coupling=None,
    fcs=None,
    anphon_cell=None,
    unchecked=False,
    fcs_format="xml",
    prefix="strain",
    results_dir=RESULTS_DIR,
    last=False,
    reorder=False,
    solver="dense",
    write_dfset_files=False,
    log=print,
):
    outdir = os.path.abspath(outdir)
    manifest = load_manifest(
        os.path.join(outdir, IFC_MANIFEST), expect_tool="strainifc"
    )
    coupling = coupling or manifest["coupling"]
    if coupling != manifest["coupling"]:
        raise ValueError(
            f"the manifest was generated for coupling='{manifest['coupling']}'"
        )
    if coupling == "harmonic":
        return collect_harmonic(
            outdir,
            manifest,
            fcs,
            unchecked,
            fcs_format,
            prefix,
            results_dir,
            last,
            solver,
            write_dfset_files,
            log,
        )
    return collect_force(
        outdir, manifest, fcs, anphon_cell, results_dir, last, reorder, log
    )


def check(outdir=".", fcs=None, anphon_cell=None, log=print):
    """Report the lattice relation, supercell equivalence and atom ordering."""
    if fcs is None:
        raise ValueError("--fcs is required")
    outdir = os.path.abspath(outdir)
    manifest = load_manifest(
        os.path.join(outdir, IFC_MANIFEST), expect_tool="strainifc"
    )
    ref_atoms = _reference_atoms(manifest, outdir)
    fcs_struct = read_fcs_structure(fcs)
    log(
        f"Reference force-constant file: {fcs} ({fcs_struct.fmt}, {fcs_struct.nat} atoms, "
        f"{fcs_struct.natmin} true-primitive atoms x {fcs_struct.ntran} translations)"
    )
    if manifest["coupling"] == "harmonic":
        check_supercell_equivalence(ref_atoms, fcs_struct)
        log("Template supercell: index-compatible with the force-constant file (OK)")
    cell = read_anphon_cell(anphon_cell) if anphon_cell else None
    try:
        prim = anphon_primitive_cell(fcs_struct, cell)
    except ValueError as exc:
        log(f"anphon primitive cell: {exc}")
        return
    if manifest["coupling"] == "force":
        mapping = match_primitive_atoms(ref_atoms, prim)
        log(describe_ordering(prim, mapping, ref_atoms))
    else:
        log(describe_ordering(prim))
