import argparse

import numpy as np
from ase.io import read
from ase.units import Ry, Bohr

from fcsio.reader import ForceConstantParser
from taylor import TaylorExpansionPotential


def get_original_structure(fname_in, format):
    """
    Read the original structure from a file.

    Parameters:
    - fname_in (str): The file name of the structure file.
    - format (str): The format of the structure file. If None, the format is auto-detected.

    Returns:
    - structure: The structure object read from the file.
    """
    if format is None:
        structure = read(fname_in)
    else:
        structure = read(fname_in, format=format)

    return structure


def print_energies(energies, unit="rydberg"):
    """
    Print the calculated potential energies.

    Parameters:
    - energies (dict): A dictionary containing arrays of energies. The key 'total' should contain the total energy array.
    """
    if unit == "rydberg":
        factor = 1.0
        print("# potential energies (Ry)")
    elif unit == "hartree":
        factor = 0.5
        print("# potential energies (Ha)")
    elif unit == "eV":
        factor = Ry
        print("# potential energies (eV)")
    else:
        raise RuntimeError("Invalid unit: %s" % unit)
    print("#", end="")

    keys = list(energies.keys())
    key_list = ["total"] + list(keys[:-1])

    for i, key in enumerate(key_list):
        if i == 0:
            print("{:>14s}".format(key), end="")
        else:
            print("{:>15s}".format(key), end="")
    print("")

    energies_array = np.zeros((energies["total"].shape[0], len(key_list)), dtype=float)

    for i, key in enumerate(key_list):
        energies_array[:, i] = energies[key]

    for i in range(energies["total"].shape[0]):
        for j in range(len(key_list)):
            print("{:15.8e}".format(energies_array[i, j] * factor), end="")
        print("")


def print_forces(forces, unit="rydberg"):
    """
    Print the calculated forces.

    Parameters:
    - forces (numpy.ndarray): The array of forces.
    """

    if unit == "rydberg":
        factor = 1.0
        print("# atomic forces (Ry/Bohr)")
    elif unit == "hartree":
        factor = 0.5
        print("# atomic forces (Ha/Bohr)")
    elif unit == "eV":
        factor = Ry / Bohr
        print("# atomic forces (eV/Angstrom)")
    else:
        raise RuntimeError("Invalid unit: %s" % unit)
    print("#", end="")

    forces_array = forces["total"] * factor
    for i in range(forces_array.shape[0]):
        print("# Snapshot %d" % (i + 1))
        for j in range(forces_array.shape[1]):
            print(
                "{:15.8e} {:15.8e} {:15.8e}".format(
                    forces_array[i, j, 0], forces_array[i, j, 1], forces_array[i, j, 2]
                )
            )


def get_argoptions():
    """
    Parse and return command line arguments.

    Returns:
    - args: The parsed command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--fcs", type=str, default=None, help="Force constant file")

    parser.add_argument(
        "--structure",
        metavar="SPOSCAR",
        default=None,
        help="The structure file that contains the information of original supercell",
    )

    parser.add_argument(
        "--format",
        metavar="vasp",
        default=None,
        help="The format of the structure specified by --structure.",
    )

    parser.add_argument(
        "--maxorder",
        type=int,
        default=4,
        help="The maximum order of the Taylor expansion. (2: Harmonic, 3: Cubic, etc.)",
    )

    parser.add_argument(
        "--disp",
        type=str,
        default=None,
        help="File containing the displacements in units of Bohr",
    )

    parser.add_argument(
        "--unit",
        type=str,
        default="rydberg",
        choices=["rydberg", "hartree", "eV"],
        help="The unit of the output potential energies.",
    )

    parser.add_argument(
        "--print-forces",
        action="store_true",
        default=False,
        help="Print the forces in the output.",
    )

    return parser.parse_args()


def main():
    """
    Main function to execute the Taylor expansion potential calculation.
    """
    args = get_argoptions()
    structure = get_original_structure(args.structure, args.format)

    parser = ForceConstantParser(args.fcs)
    fcs_dic = parser.read(maxorder=args.maxorder)
    taylor = TaylorExpansionPotential(structure, maxorder=args.maxorder)
    taylor.set_forceconstants(fcs_dic, parser.primitive_cell)

    if args.disp is None:
        raise RuntimeError("Displacement file is not specified.")

    displacements = np.loadtxt(args.disp)
    nlen = len(displacements) // (structure.get_global_number_of_atoms())
    if nlen * structure.get_global_number_of_atoms() != len(displacements):
        raise RuntimeError("The number of displacements is inconsistent.")

    displacements = displacements.reshape(-1, structure.get_global_number_of_atoms(), 3)
    energies, forces = taylor.compute(displacements)
    if args.print_forces:
        print_forces(forces, args.unit)
    else:
        print_energies(energies, args.unit)


if __name__ == "__main__":
    main()
