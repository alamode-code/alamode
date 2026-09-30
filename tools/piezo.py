#!/usr/bin/env python
"""piezo.py -- clamped-ion nonlinear electromechanical coefficients for EFIELD
(STRAINFILE /Piezoelectric) from VASP.

  piezo.py generate --route berry|dfpt REFDIR OUTDIR [--h 0.01] [--force]
  piezo.py generate --route berry-lambda REFDIR OUTDIR --atom K --dir x|y|z --strain mn
                    [--h 0.01] [--delta 0.02]
      REFDIR: the VASP reference calculation (POSCAR, INCAR, KPOINTS, POTCAR).
      Writes clamped-ion strained cells (fractional coordinates fixed, lattice
      rows a (I + u), u symmetric; a shear step sets u_mn = u_nm = +-h) with
      edited INCARs, run_all.sh and piezo_manifest.json:
        berry        73 LCALCPOL SCFs (ISYM = 0): reference, +-h per Voigt
                     direction, (+-h, +-h) for the 15 pairs -> e0 and B
        dfpt         12 LEPSILON runs, +-h per Voigt direction -> Lambda
        berry-lambda 4 LCALCPOL SCFs: atom K displaced by +-delta (Angst,
                     Cartesian) at strain mn +-h -> one Lambda column (check)
  (run VASP in every directory)
  piezo.py collect OUTDIR [--route ...] [--strain-file FILE [--overwrite]] [--print]
      berry: e0 -> clamped_ion and B -> second_order, averaged over the point
             group (--no-symmetrize: raw; --no-clamped-ion keeps an existing
             clamped_ion);
      dfpt:  Lambda -> born_charge_strain_derivative ([--born-ref OUTCAR] for
             the reference Born charges; the ASR residual is removed);
      berry-lambda: prints the Lambda column and compares it with the stored
             Lambda of --strain-file.
      Only the named datasets are replaced; atoms are mapped onto the
      /ReferenceCell of FILE (permuted or nested cells).

B must come from the Berry route: the strained-cell LEPSILON piezoelectric
tensors do not give a consistent B (see the docs).  Requires numpy, ase, spglib, h5py.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strainkit import workflow_piezo as wp  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--debug", action="store_true", help="show Python tracebacks")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write the strained VASP inputs")
    g.add_argument("--route", required=True, choices=wp.ROUTES)
    g.add_argument(
        "refdir", help="VASP reference calculation (POSCAR, INCAR, KPOINTS, POTCAR)"
    )
    g.add_argument("outdir", help="directory for the calculations and the manifest")
    g.add_argument("--h", type=float, default=0.01, help="strain step (default 0.01)")
    g.add_argument(
        "--atom",
        type=int,
        default=None,
        help="berry-lambda: displaced atom (1-based, POSCAR order)",
    )
    g.add_argument(
        "--dir",
        default=None,
        choices=("x", "y", "z"),
        help="berry-lambda: displacement direction",
    )
    g.add_argument(
        "--strain",
        default=None,
        help="berry-lambda: strain component xx yy zz yz xz xy",
    )
    g.add_argument(
        "--delta",
        type=float,
        default=0.02,
        help="berry-lambda: displacement in Angst (default 0.02)",
    )
    g.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing directories and manifest",
    )

    c = sub.add_parser("collect", help="evaluate the VASP outputs")
    c.add_argument("outdir")
    c.add_argument(
        "--route", default=None, choices=wp.ROUTES, help="default: the manifest's route"
    )
    c.add_argument(
        "--strain-file",
        default=None,
        help="existing STRAINFILE container to write into",
    )
    c.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing datasets of the same name",
    )
    c.add_argument(
        "--print", dest="verbose", action="store_true", help="print the full tensors"
    )
    c.add_argument(
        "--born-ref",
        default=None,
        help="dfpt: LEPSILON OUTCAR of the unstrained reference",
    )
    c.add_argument(
        "--no-clamped-ion",
        action="store_true",
        help="berry: write only second_order and keep the stored clamped_ion",
    )
    c.add_argument(
        "--no-symmetrize",
        action="store_true",
        help="berry: write the raw e0 and B instead of their point-group average",
    )
    c.add_argument(
        "--second-order",
        action="store_true",
        help="require B (only the berry route provides it; refused for dfpt)",
    )

    args = p.parse_args(argv)
    try:
        if args.cmd == "generate":
            wp.generate(
                args.route,
                args.refdir,
                args.outdir,
                h=args.h,
                atom=args.atom,
                direction=args.dir,
                strain=args.strain,
                delta=args.delta,
                force=args.force,
            )
        else:
            wp.collect(
                args.outdir,
                route=args.route,
                strain_file=args.strain_file,
                overwrite=args.overwrite,
                born_ref=args.born_ref,
                second_order=args.second_order,
                skip_clamped_ion=args.no_clamped_ion,
                symmetrize=not args.no_symmetrize,
                verbose=args.verbose,
            )
    except Exception as exc:  # noqa: BLE001
        if args.debug:
            raise
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
