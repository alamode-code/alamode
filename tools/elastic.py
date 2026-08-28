#!/usr/bin/env python
"""elastic.py -- DFT finite-strain workflow for the elastic constants used by
anphon's SCPH/QHA cell relaxation (elastic_constants.in and C1_array.in).

  elastic.py generate --code VASP --template DIR [--smag 0.01 --nmag 2 --dirset minimal]
      writes strain_000 (reference) and strain_NNN/ with strained, clamped-ion
      structures plus a manifest and a job script skeleton.
  (run the DFT code in every directory; fixed cell, fixed ions)
  elastic.py fit [--fit stress|energy|both] [--fcs REF.xml --anphon-cell anphon.in]
      fits sigma0, C2 (SOEC) and C3 (TOEC) as derivatives with respect to the
      Green-Lagrange strain and writes results/elastic_constants.in, C1_array.in.
  elastic.py show elastic_constants.in --structure POSCAR
      prints any elastic_constants.in as GPa tables.

Requires numpy, ase and spglib.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strainkit import workflow_elastic as we  # noqa: E402
from strainkit.structure import CODES  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--debug", action="store_true", help="show Python tracebacks")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create the strained structures")
    g.add_argument("--code", required=True, help="DFT code: " + "|".join(CODES))
    g.add_argument("--template", required=True, help="directory with the reference input (POSCAR+INCAR+..., pw.in)")
    g.add_argument("--structure", default=None, help="structure file name inside --template (default POSCAR / pw.in / input.extxyz)")
    g.add_argument("--smag", type=float, default=0.01, help="strain magnitude step (default 0.01)")
    g.add_argument("--nmag", type=int, default=2, help="magnitudes +-1..+-nmag times smag per direction (default 2)")
    g.add_argument("--dirset", choices=("minimal", "full"), default="minimal",
                   help="minimal: 21 directions (stress fits); full: 56 directions (energy fits)")
    g.add_argument("--outdir", default=".", help="working directory (default .)")
    g.add_argument("--job-template", default=None, help="job script template containing RUN_DFT_CALCULATION")
    g.add_argument("--dft-command", default=None, help="file with the shell lines that run the DFT code in a directory")
    g.add_argument("--force", action="store_true", help="overwrite existing strain_* directories")

    f = sub.add_parser("fit", help="fit the elastic constants from the DFT outputs")
    f.add_argument("--outdir", default=".", help="working directory of the generate step")
    f.add_argument("--fit", choices=("stress", "energy", "both"), default="stress")
    f.add_argument("--fcs", default=None, help="reference force-constant file (.xml/.h5) used by anphon")
    f.add_argument("--anphon-cell", default=None, help="anphon input with the &cell field (or a structure file) defining the anphon primitive cell")
    f.add_argument("--allow-relaxed", action="store_true", help="accept outputs whose geometry changed (diagnostic; not for anphon)")
    f.add_argument("--force-write", action="store_true", help="write anphon-named files even for relaxed-ion data")
    f.add_argument("--no-symmetrize", action="store_true", help="do not symmetrize over the point group")
    f.add_argument("--symprec", type=float, default=1.0e-5)
    f.add_argument("--results-dir", default=we.RESULTS_DIR)
    f.add_argument("--compare", default=None, help="anphon log with an ELASTIC_CONST = 1 table to compare with")
    f.add_argument("--min-c3", type=float, default=0.5, help="print TOEC components above this value (GPa)")
    f.add_argument("--exclude", default="", help="comma-separated strain_NNN directories to exclude")

    s = sub.add_parser("show", help="print an elastic_constants.in in GPa")
    s.add_argument("file")
    s.add_argument("--volume", type=float, default=None, help="cell volume in A^3")
    s.add_argument("--structure", default=None, help="structure file (ase-readable) giving the cell volume")
    s.add_argument("--c1", default=None, help="C1_array.in to print as well")
    s.add_argument("--min-c3", type=float, default=0.5)

    args = p.parse_args(argv)
    try:
        if args.cmd == "generate":
            we.generate(args.code, args.template, args.outdir, args.structure, args.smag, args.nmag,
                        args.dirset, args.job_template, args.dft_command, args.force)
        elif args.cmd == "fit":
            excl = [x for x in args.exclude.split(",") if x]
            we.fit(args.outdir, args.fit, args.fcs, args.anphon_cell, args.allow_relaxed,
                   args.force_write, not args.no_symmetrize, args.symprec, args.results_dir,
                   args.compare, args.min_c3, excl)
        elif args.cmd == "show":
            print(we.show(args.file, args.volume, args.structure, args.c1, args.min_c3))
    except Exception as exc:  # noqa: BLE001
        if args.debug:
            raise
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
