#!/usr/bin/env python
"""strainfile.py: pack, inspect and check the strain-coupling container of anphon.

    strainfile.py pack  --strain-ifc-dir DIR [--c1 C1_array.in] --fcs FC2FILE [--anphon-cell IN]
                        [--legacy-cell IN] -o OUT.h5 [--force]
    strainfile.py show  FILE.h5 [--min-c3 0.5]
    strainfile.py check FILE.h5 [--anphon-cell IN] [--fcs FC2FILE]
    strainfile.py piezo FILE.h5 (--outcar OUTCAR | --voigt TABLE --structure POSCAR)
                        [--structure POSCAR|vasprun.xml] [--overwrite]

The container (HDF5, schema alamode:strain_coupling) replaces the text files
elastic_constants.in, C1_array.in, strain_force.in and strain_harmonic.in (plus
the strained force-constant files); anphon reads it through the STRAINFILE tag
of the &relax field.  elastic.py fit and strainifc.py collect write into it
directly (--strain-file); pack converts existing text files.  piezo adds the
clamped-ion piezoelectric tensor (/Piezoelectric) that EFIELD uses with a
strained cell, replacing only that group.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strainkit import strainfile as sf  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--debug", action="store_true", help="show tracebacks")
    sub = p.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("pack", help="build a container from legacy text files")
    k.add_argument(
        "--strain-ifc-dir",
        default=None,
        help="STRAIN_IFC_DIR with the text files and strained FC files",
    )
    k.add_argument(
        "--c1",
        default=None,
        help="C1_array.in (default: STRAIN_IFC_DIR/C1_array.in if present)",
    )
    k.add_argument(
        "--fcs",
        required=True,
        help="force-constant file (FC2FILE/FCSFILE) of the anphon run",
    )
    k.add_argument(
        "--anphon-cell",
        default=None,
        help="anphon input with the &cell field (required with an xml --fcs)",
    )
    k.add_argument(
        "--legacy-cell",
        default=None,
        help="anphon input whose &cell volume the legacy Ry files were made for (default: the reference cell)",
    )
    k.add_argument("-o", "--output", required=True, help="container to write")
    k.add_argument(
        "--force", action="store_true", help="overwrite an existing container"
    )

    s = sub.add_parser("show", help="print the contents of a container")
    s.add_argument("file")
    s.add_argument(
        "--min-c3",
        type=float,
        default=0.5,
        help="print TOEC components above this value (GPa)",
    )

    c = sub.add_parser("check", help="check a container against a planned anphon run")
    c.add_argument("file")
    c.add_argument(
        "--anphon-cell", default=None, help="anphon input with the &cell field"
    )
    c.add_argument(
        "--fcs",
        default=None,
        help="force-constant file (FC2FILE/FCSFILE) of the anphon run",
    )

    z = sub.add_parser(
        "piezo",
        help="add the clamped-ion piezoelectric tensor (/Piezoelectric) to a container",
    )
    z.add_argument("file")
    src = z.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--outcar",
        default=None,
        help="VASP LEPSILON OUTCAR: the last 'PIEZOELECTRIC TENSOR (including local field effects) "
        "... (C/m^2)' block (never the IONIC CONTR block)",
    )
    src.add_argument(
        "--voigt",
        default=None,
        help="3 x 6 text table in C/m^2, rows x y z, columns xx yy zz yz xz xy (proper, clamped-ion)",
    )
    z.add_argument(
        "--structure",
        default=None,
        help="POSCAR or vasprun.xml of the calculation for the crystal check (required with --voigt)",
    )
    z.add_argument(
        "--overwrite", action="store_true", help="replace an existing /Piezoelectric"
    )

    args = p.parse_args(argv)
    try:
        if args.cmd == "pack":
            sf.pack(
                args.output,
                args.strain_ifc_dir,
                args.c1,
                args.fcs,
                args.anphon_cell,
                args.legacy_cell,
                args.force,
            )
        elif args.cmd == "show":
            sf.show(args.file, args.min_c3)
        elif args.cmd == "check":
            problems = sf.check(args.file, args.anphon_cell, args.fcs)
            if problems:
                sys.exit(1)
        elif args.cmd == "piezo":
            sf.add_piezo(
                args.file, args.outcar, args.voigt, args.structure, args.overwrite
            )
    except Exception as exc:  # noqa: BLE001
        if args.debug:
            raise
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
