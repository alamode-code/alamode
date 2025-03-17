#!/usr/bin/env python
#
# extract.py
#
# Simple script to extract atomic displacements, atomic forces, and
# energies from output files.
# Currently, VASP, Quantum-ESPRESSO, LAMMPS, OpenMX, and xTAPP are supported.
#
# Copyright (c) 2014-2020 Terumasa Tadano
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#

"""
This python script extracts atomic displacements, atomics forces,
and energies.
"""

from __future__ import print_function

import argparse

try:
    from interface.LAMMPS import LammpsParser
    from interface.OpenMX import OpenmxParser
    from interface.QE import QEParser
    from interface.VASP import VaspParser
    from interface.xTAPP import XtappParser

except ModuleNotFoundError:  # occurs when it is called as a library
    from .interface.LAMMPS import LammpsParser
    from .interface.OpenMX import OpenmxParser
    from .interface.QE import QEParser
    from .interface.VASP import VaspParser
    from .interface.xTAPP import XtappParser


def check_options(args):
    # Check the calculator option

    conditions = [
        args.VASP is None,
        args.QE is None,
        args.xTAPP is None,
        args.LAMMPS is None,
        args.OpenMX is None,
    ]

    if conditions.count(True) == len(conditions):
        raise RuntimeError(
            "Either --VASP, --QE, --xTAPP, --LAMMPS, \
                --OpenMX option must be given."
        )

    elif len(conditions) - conditions.count(True) > 1:
        raise RuntimeError(
            "Error : --VASP, --QE, --xTAPP, --LAMMPS, and \
            --OpenMX cannot be given simultaneously."
        )

    elif args.VASP:
        code = "VASP"
        file_original = args.VASP

    elif args.QE:
        code = "QE"
        file_original = args.QE

    elif args.xTAPP:
        code = "xTAPP"
        file_original = args.xTAPP

    elif args.LAMMPS:
        code = "LAMMPS"
        file_original = args.LAMMPS

    elif args.OpenMX:
        code = "OpenMX"
        file_original = args.OpenMX

    # Check output option
    str_get = args.get.lower()
    if str_get not in ["disp-force", "disp", "force", "energy", "born", "dielec"]:
        raise RuntimeError(
            "Error: Please specify which quantity to extract by the --get option."
        )

    print_disp = False
    print_force = False
    print_energy = False
    print_borninfo = False

    if str_get == "disp-force":
        print_disp = True
        print_force = True
    elif str_get == "disp":
        print_disp = True
    elif str_get == "force":
        print_force = True
    elif str_get == "energy":
        print_energy = True
    elif str_get == "born" or str_get == "dielec":
        print_borninfo = True
        if code != "VASP" and code != "QE":
            raise RuntimeError("Sorry, --get born is available only for VASP and QE.")

    output_flags = [print_disp, print_force, print_energy, print_borninfo]

    # Check unit option
    str_unit = args.unitname.lower()
    if str_unit in ["ev", "electron_volt"]:
        str_unit = "ev"
    elif str_unit in ["ry", "ryd", "rydberg"]:
        str_unit = "rydberg"
    elif str_unit in ["ha", "hartree"]:
        str_unit = "hartree"
    else:
        print("Error: Invalid unit name : %s" % args.unitname)

    return code, file_original, output_flags, str_unit


def run_parse(args, code, file_original, file_results, output_flags, str_unit):
    # Print data
    if code == "VASP":
        handler = VaspParser()
        handler.set_vca_mode(args.vca)

    elif code == "QE":
        handler = QEParser()

    elif code == "xTAPP":
        handler = XtappParser()

    elif code == "OpenMX":
        handler = OpenmxParser()

    elif code == "LAMMPS":
        handler = LammpsParser()

    handler.parse(
        file_original,
        file_results,
        args.offset,
        str_unit,
        output_flags,
        args.emin,
        args.emax,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--VASP",
        metavar="SPOSCAR",
        help="VASP POSCAR file with equilibrium atomic \
                            positions (default: None)",
    )

    parser.add_argument(
        "--QE",
        metavar="supercell.pw.in",
        help="Quantum-ESPRESSO input file with equilibrium\
                    atomic positions (default: None)",
    )

    parser.add_argument(
        "--xTAPP",
        metavar="supercell.cg",
        help="xTAPP CG file with equilibrium atomic \
                            positions (default: None)",
    )

    parser.add_argument(
        "--LAMMPS",
        metavar="supercell.lammps",
        help="LAMMPS structure file with equilibrium atomic \
                            positions (default: None)",
    )

    parser.add_argument(
        "--OpenMX",
        metavar="supercell.dat",
        help="OpenMX dat file with equilibrium atomic \
                            positions (default: None)",
    )

    parser.add_argument(
        "--get",
        default="disp-force",
        help="specify which quantity to extract. \
                            Available options are 'disp-force', 'disp', \
                            'force', 'energy', and 'born'. \
                            (default: disp-force)",
    )

    parser.add_argument(
        "--unit",
        action="store",
        metavar="OUTPUT_UNIT",
        dest="unitname",
        default="Rydberg",
        help="print atomic displacements and forces in units of UNIT. \
                            Available options are 'eV', 'Rydberg' (default), and 'Hartree'.",
    )

    parser.add_argument(
        "--offset",
        help="Specify an output file (either *.xml, *.pw.out, or *.str) of an\
                            equilibrium structure to subtract residual forces, \
                            displacements, or energies.",
    )

    parser.add_argument(
        "--emin",
        default=None,
        type=float,
        help="Lower bound of the energy filter (eV) used for selecting output structures.\
                            Available only in the VASP parser.",
    )

    parser.add_argument(
        "--emax",
        default=None,
        type=float,
        help="Upper bound of the energy filter (eV) used for selecting output structures.\
                            Available only in the VASP parser.",
    )

    parser.add_argument(
        "target_file",
        metavar="file_to_parse",
        type=str,
        nargs="+",
        help="Output file of DFT codes, e.g., vasprun.xml.",
    )

    parser.add_argument(
        "--vca",
        action="store_true",
        dest="vca",
        default=False,
        help="Use VCA mode for VASP parser. (default: False)",
    )

    args = parser.parse_args()
    file_results = args.target_file

    code, file_original, output_flags, str_unit = check_options(args)
    run_parse(args, code, file_original, file_results, output_flags, str_unit)
