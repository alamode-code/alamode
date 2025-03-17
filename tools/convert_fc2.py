import argparse
import sys

import numpy as np
from ase.data import atomic_numbers
from ase.units import Bohr, Rydberg
from lxml import etree


def parse_xml(fname_xml):
    try:
        # Attempt to parse the XML file directly with lxml
        tree = etree.parse(fname_xml)
        return tree
    except etree.XMLSyntaxError:
        # If a parsing error occurs with lxml,
        # attempt to repair by re-reading the file with the 'recover' parser
        repair_parser = etree.XMLParser(recover=True)
        tree = etree.parse(fname_xml, parser=repair_parser)
        return tree


def get_forceconstants_xml(fname_xml):
    """
    Parse the XML file and return the force constants in compact form.
    Also, parse the mapping table and the cell information for later use.
    The returned force constants are in units of eV/A^2.
    """
    xml = parse_xml(fname_xml)
    root = xml.getroot()

    natom_super = int(root.find("Structure/NumberOfAtoms").text)
    ntrans = int(root.find("Symmetry/NumberOfTranslations").text)
    natom_prim = natom_super // ntrans

    # parse structures
    numbers = np.zeros(natom_super, dtype=int)
    positions = np.zeros((natom_super, 3), dtype=float)
    lattice = np.zeros((3, 3), dtype=float)
    for i in range(3):
        lattice[:, i] = [
            float(t)
            for t in root.find(f"Structure/LatticeVector/a{i + 1}").text.split()
        ]
    for elems in root.findall("Structure/Position/pos"):
        iatom = int(elems.get("index")) - 1
        numbers[iatom] = atomic_numbers[elems.get("element")]
        positions[iatom] = [float(t) for t in elems.text.split()]

    cell = (lattice, positions, numbers)

    # parse mapping table
    map_p2s = np.zeros((ntrans, natom_prim), dtype=int)
    for elems in root.findall("Symmetry/Translations/map"):
        itran = int(elems.get("tran")) - 1
        iatom = int(elems.get("atom")) - 1
        map_p2s[itran, iatom] = int(elems.text) - 1

    # parse harmonic force constants
    fc2_compact = np.zeros((natom_prim, natom_super, 3, 3), dtype=float)

    for elems in root.findall("ForceConstants/HARMONIC/FC2"):
        atom1, xyz1 = [int(t) - 1 for t in elems.get("pair1").split()]
        atom2, xyz2, _ = [int(t) - 1 for t in elems.get("pair2").split()]
        fcsval = float(elems.text)

        fc2_compact[atom1, atom2, xyz1, xyz2] += fcsval

    fc2_compact *= Rydberg / (Bohr**2)

    return map_p2s, fc2_compact, cell


def get_fc2_full(cell, fc2_compact, map_p2s):
    """
    Convert the compact form of the force constants to the full form.
    """
    natom_prim, natom_super = fc2_compact.shape[:2]
    fc2_full = np.zeros((natom_super, natom_super, 3, 3), dtype=float)

    ntrans = map_p2s.shape[0]
    map_atoms_translation = np.zeros((natom_super, ntrans), dtype=int)
    positions = cell[1]
    # Construct the mapping table by translating the positions
    for i in range(ntrans):
        translation_vector = positions[map_p2s[i, 0], :] - positions[map_p2s[0, 0], :]
        positions_trans = positions + translation_vector

        for j in range(natom_super):
            iloc = -1
            for k in range(natom_super):
                diff = positions_trans[k] - positions[j]
                diff -= np.rint(diff)
                if np.linalg.norm(diff) < 1e-5:
                    map_atoms_translation[j, i] = k
                    iloc = k
                    break
            if iloc == -1:
                print(
                    "Error: equivalent atom not found for atom {} in translation {}".format(
                        j, i
                    )
                )

    # Copy the compact force constant values to the full force constant tensor
    for i in range(ntrans):
        for j in range(natom_prim):
            jat_trans = map_atoms_translation[map_p2s[0, j], i]
            for k in range(natom_super):
                kat_trans = map_atoms_translation[k, i]
                for m in range(3):
                    for n in range(3):
                        fc2_full[
                            jat_trans,
                            kat_trans,
                            m,
                            n,
                        ] = fc2_compact[j, k, m, n]

    return fc2_full


def print_fc2_phonopy(map_p2s, fc2_compact, outfile=None):
    if outfile is None:
        outfile = sys.stdout

    natom_prim, natom_super = fc2_compact.shape[:2]

    with open(outfile, "w") if outfile != sys.stdout else outfile as f:
        print("{:5d} {:5d}".format(natom_prim, natom_super), file=f)
        for i in range(natom_prim):
            for j in range(natom_super):
                print("{:5d} {:5d}".format(map_p2s[0, i] + 1, j + 1), file=f)
                for k in range(3):
                    for m in range(3):
                        print(
                            "{:20.15f}".format(fc2_compact[i, j, k, m]),
                            end="",
                            file=f,
                        )
                    print("", file=f)


def print_fc2_phonopy_full(fc2_full, outfile=None):
    if outfile is None:
        outfile = sys.stdout

    natom_super = fc2_full.shape[0]

    with open(outfile, "w") if outfile != sys.stdout else outfile as f:
        print("{:5d} {:5d}".format(natom_super, natom_super), file=f)
        for i in range(natom_super):
            for j in range(natom_super):
                print("{:5d} {:5d}".format(i + 1, j + 1), file=f)
                for k in range(3):
                    for m in range(3):
                        print("{:20.15f}".format(fc2_full[i, j, k, m]), end="", file=f)
                    print("", file=f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Input XML file", required=True)
    parser.add_argument("--output", "-o", help="Output file (default: print to stdout)")
    parser.add_argument(
        "--full",
        "-f",
        action="store_true",
        help="Print full force constants instead of the compact form",
    )
    return parser.parse_args()


def main():
    args = get_args()
    map_p2s, fc2_compact, cell = get_forceconstants_xml(args.input)
    if args.full:
        fc2_full = get_fc2_full(cell, fc2_compact, map_p2s)
        print_fc2_phonopy_full(fc2_full, outfile=args.output)
    else:
        print_fc2_phonopy(map_p2s, fc2_compact, outfile=args.output)


if __name__ == "__main__":
    main()
