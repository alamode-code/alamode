#!/usr/bin/env python
# Copyright (c) 2023 Ryota Masuki (strainIFCcoupling,
#                    https://github.com/r-masuki/strainIFCcoupling)
# Copyright (c) 2026 Terumasa Tadano
# MIT license.  See LICENCE.txt of the ALAMODE package.
"""strainifc.py -- strain-IFC coupling inputs for the SCPH/QHA cell relaxation of anphon.

  strainifc.py generate --coupling harmonic --code VASP --template DIR [--smag 0.005 --dmag 0.01 --central]
      strained supercells with ALM displacement patterns (strain_NNN/disp_MM, strain_NNN/nodisp)
  strainifc.py generate --coupling force --code QE --template DIR
      strained primitive cells (strain_NNN/primitive) + the unstrained reference (strain_000)
  (run the DFT code in every directory: single-point, fixed cell, fixed ions)
  strainifc.py collect --fcs FC2FILE.xml [--anphon-cell anphon.in]
      harmonic: fits the force constants of every strained supercell (nanobind alm package)
                and writes results/strain_harmonic.in + results/strain_NNN.xml|h5
      force:    writes results/strain_force.in (eV/Angstrom, anphon primitive-cell order)
  strainifc.py check --fcs FC2FILE.xml [--anphon-cell anphon.in]
      reports lattice relation, supercell equivalence and atom ordering

Port of the strainIFCcoupling scripts by Ryota Masuki onto the in-repo alm package.
References: R. Masuki et al., Phys. Rev. B 106, 224104 (2022); Phys. Rev. B 107, 134119 (2023).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strainkit import workflow_ifc as wi  # noqa: E402
from strainkit.structure import CODES  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--debug", action="store_true", help="show Python tracebacks")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create the strained structures")
    g.add_argument("--coupling", choices=wi.COUPLINGS, required=True)
    g.add_argument("--code", required=True, help="DFT code: " + "|".join(CODES))
    g.add_argument("--template", required=True, help="directory with the reference input (supercell for harmonic, primitive cell for force)")
    g.add_argument("--structure", default=None, help="structure file name inside --template")
    g.add_argument("--smag", type=float, default=None, help="strain magnitude (default 0.005 or the value in the modes file)")
    g.add_argument("--dmag", type=float, default=0.01, help="atomic displacement magnitude in Angstrom (harmonic)")
    g.add_argument("--central", action="store_true", help="central finite differences (+smag and -smag, weight 1/2 each)")
    g.add_argument("--no-offset", action="store_true", help="harmonic: skip the undisplaced (nodisp) reference run")
    g.add_argument("--modes", default=None, help="strain_modes.json or comma-separated mode names (xx,yy,zz,yz,zx,xy)")
    g.add_argument("--outdir", default=".", help="working directory (default .)")
    g.add_argument("--nbody", type=int, default=2, help="ALM NBODY for the harmonic model (default 2)")
    g.add_argument("--cutoff", type=float, default=None, help="ALM cutoff radius in Angstrom (default: none)")
    g.add_argument("--job-template", default=None, help="job script template containing RUN_DFT_CALCULATION")
    g.add_argument("--dft-command", default=None, help="file with the shell lines that run the DFT code in a directory")
    g.add_argument("--copy-potcar", action="store_true", help="copy POTCAR instead of symlinking it")
    g.add_argument("--force", action="store_true", help="overwrite existing strain_* directories")

    c = sub.add_parser("collect", help="build the anphon input files from the DFT outputs")
    c.add_argument("--outdir", default=".")
    c.add_argument("--coupling", choices=wi.COUPLINGS, default=None, help="default: from the manifest")
    c.add_argument("--fcs", default=None, help="reference force-constant file used by anphon (FC2FILE / FCSFILE)")
    c.add_argument("--anphon-cell", default=None, help="anphon input with the &cell field (or a structure file); required with an XML --fcs for the force coupling")
    c.add_argument("--unchecked", action="store_true", help="harmonic: skip the checks against --fcs (not recommended)")
    c.add_argument("--fcs-format", choices=("xml", "h5"), default="xml", help="format of the written force-constant files")
    c.add_argument("--prefix", default="strain", help="prefix of the written force-constant files")
    c.add_argument("--results-dir", default=wi.RESULTS_DIR)
    c.add_argument("--last", action="store_true", help="accept outputs with several ionic steps (use the last)")
    c.add_argument("--reorder", action="store_true", help="force: permute the rows into anphon's atom order if the template order differs")
    c.add_argument("--solver", default="dense", help="ALM solver (dense or a sparse solver name)")
    c.add_argument("--write-dfset", action="store_true", help="also write classic DFSET files (Bohr, Ry/Bohr)")

    k = sub.add_parser("check", help="report cell relations and atom ordering against --fcs")
    k.add_argument("--outdir", default=".")
    k.add_argument("--fcs", required=True)
    k.add_argument("--anphon-cell", default=None)

    args = p.parse_args(argv)
    try:
        if args.cmd == "generate":
            modes_json, names = None, None
            if args.modes:
                if os.path.isfile(args.modes):
                    modes_json = args.modes
                else:
                    names = [m.strip() for m in args.modes.split(",") if m.strip()]
            wi.generate(args.coupling, args.code, args.template, args.outdir, args.structure, args.smag,
                        args.dmag, args.central, args.no_offset, modes_json, names, args.nbody, args.cutoff,
                        args.job_template, args.dft_command, args.copy_potcar, args.force)
        elif args.cmd == "collect":
            wi.collect(args.outdir, args.coupling, args.fcs, args.anphon_cell, args.unchecked, args.fcs_format,
                       args.prefix, args.results_dir, args.last, args.reorder, args.solver, args.write_dfset)
        elif args.cmd == "check":
            wi.check(args.outdir, args.fcs, args.anphon_cell)
    except Exception as exc:  # noqa: BLE001
        if args.debug:
            raise
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
