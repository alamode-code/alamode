import argparse
import collections
import os

import numpy as np

try:
    from pymatgen.core import Structure
except ImportError:
    from pymatgen import Structure

import seekpath
from GenDisplacement import AlamodeDisplace
from interface.QE import QEParser
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.io.vasp import inputs

ALAMODE_root = "~/src/alamode/_build"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mag",
    type=float,
    default=0.02,
    help="Magnitude of displacement in units of \
                        Angstrom (default: 0.02)",
)

parser.add_argument(
    "--prefix",
    type=str,
    default="disp",
    help="Prefix of the files to be created. (default: disp)",
)

parser.add_argument("-d", "--dim", default=None, type=str, help="Transformation matrix")

parser.add_argument(
    "file_primitive",
    metavar="primitive.pw.in",
    default=None,
    help="Original primitive cell input file for pw.x",
)

parser.add_argument(
    "--dfset",
    type=str,
    default=None,
    help="The displacement-force datasets for harmonic phonon calculation.",
)

parser.add_argument(
    "--symprec",
    type=float,
    default=1.0e-3,
    help="Symmetry tolerance for space group determination.",
)

parser.add_argument(
    "--alamode_ver", type=int, default=1, help="ALAMODE major version (default: 1)"
)


def gen_kpoints_file(structure):
    kmesh = inputs.Kpoints.automatic_density_by_vol(structure, 450)
    kmesh_dict = kmesh.as_dict()

    return kmesh_dict["kpoints"], kmesh_dict["usershift"]


def gen_species_dictionary(atomic_number_uniq):
    species_dict = {}
    counter = 1
    for num in atomic_number_uniq:
        species_dict[num] = counter
        counter += 1
    return species_dict


def gen_alm_input(
    filename,
    prefix,
    mode,
    structure,
    norder,
    str_cutoff,
    ndata=0,
    dfset="DFSET",
    symprec=1.0e-3,
    alamode_ver=1,
):
    Bohr = 0.52917721067

    if mode != "suggest" and mode != "optimize":
        raise RuntimeError("Invalid MODE: %s" % mode)

    atomic_numbers_uniq = list(
        collections.OrderedDict.fromkeys(structure.atomic_numbers)
    )

    species_index = gen_species_dictionary(atomic_numbers_uniq)
    # Make input for ALM
    with open(filename, "w") as f:
        f.write("&general\n")
        f.write(" PREFIX = %s\n" % prefix)
        f.write(" MODE = %s\n" % mode)
        f.write(" NAT = %i\n" % structure.num_sites)
        str_spec = ""
        for num in atomic_numbers_uniq:
            str_spec += str(get_el_sp(num)) + " "
        f.write(" NKD = %i; KD = %s\n" % (structure.ntypesp, str_spec))
        f.write(" TOLERANCE = {:e}\n".format(symprec))
        f.write("/\n\n")
        f.write("&interaction\n")
        f.write(" NORDER = %i\n" % norder)
        f.write("/\n\n")
        f.write("&cutoff\n")
        f.write(" %s\n" % str_cutoff)
        f.write("/\n\n")
        f.write("&cell\n")
        f.write("%20.14f\n" % (1.0 / Bohr))
        for i in range(3):
            for j in range(3):
                f.write("%20.13f" % structure.lattice.matrix[i][j])
            f.write("\n")
        f.write("/\n\n")
        f.write("&position\n")
        for i in range(len(structure.frac_coords)):
            f.write("%4i " % species_index[structure.atomic_numbers[i]])
            f.write("%20.14f " % structure.frac_coords[i][0])
            f.write("%20.14f " % structure.frac_coords[i][1])
            f.write("%20.14f\n" % structure.frac_coords[i][2])
        f.write("/\n\n")

        if mode == "optimize":
            f.write("&optimize\n")
            f.write(" DFSET = %s\n" % dfset)
            f.write(" SPARSE = 1\n")
            f.write("/\n\n")


def gen_anphon_input(
    filename,
    prefix,
    mode,
    structure,
    path_info,
    npoints=51,
    symprec=1.0e-3,
    alamode_ver=1,
):
    Bohr = 0.52917721067

    if mode != "phonons":
        raise RuntimeError("Invalid MODE: %s" % mode)

    atomic_numbers_uniq = list(
        collections.OrderedDict.fromkeys(structure.atomic_numbers)
    )

    # Make input for ALM
    with open(filename, "w") as f:
        f.write("&general\n")
        f.write(" PREFIX = %s\n" % prefix)
        f.write(" MODE = %s\n" % mode)
        str_spec = ""
        for num in atomic_numbers_uniq:
            str_spec += str(get_el_sp(num)) + " "
        f.write(" KD = %s\n" % str_spec)
        f.write(" TOLERANCE = {:f}\n".format(symprec))
        if alamode_ver == 1:
            f.write(" FCSFILE = %s.xml\n" % prefix)
        else:
            f.write(" FCSFILE = %s.xml\n" % prefix)
        f.write("/\n\n")
        f.write("&cell\n")
        f.write("%20.14f\n" % (1.0 / Bohr))
        for i in range(3):
            for j in range(3):
                f.write("%20.13f" % structure.lattice.matrix[i][j])
            f.write("\n")
        f.write("/\n\n")
        f.write("&kpoint\n")
        f.write(" 1\n")

        kpath = path_info["path"]
        point_coords = path_info["point_coords"]

        for line in kpath:
            f.write(" %s" % line[0])
            coord_s = point_coords[line[0]]
            coord_e = point_coords[line[1]]
            for coord in coord_s:
                f.write(" %15.8f" % coord)

            f.write(" %s" % line[1])
            for coord in coord_e:
                f.write("%15.8f" % coord)
            f.write(" %d\n" % npoints)
        f.write("/\n\n")


def process_args(args):
    if args.dim is None:
        scaling_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        str_dim = args.dim.strip().split()
        if len(str_dim) == 1:
            nsize = int(str_dim[0])
            scaling_matrix = [[nsize, 0, 0], [0, nsize, 0], [0, 0, nsize]]
        elif len(str_dim) == 3:
            nsize1 = int(str_dim[0])
            nsize2 = int(str_dim[1])
            nsize3 = int(str_dim[2])
            scaling_matrix = [[nsize1, 0, 0], [0, nsize2, 0], [0, 0, nsize3]]
        elif len(str_dim) == 9:
            nsizes = [int(t) for t in str_dim]
            scaling_matrix = [
                [nsizes[0], nsizes[1], nsizes[2]],
                [nsizes[3], nsizes[4], nsizes[5]],
                [nsizes[6], nsizes[7], nsizes[8]],
            ]
        else:
            raise RuntimeError("Invalid format of --dim.")

    return args.mag, args.prefix, scaling_matrix


def update_qeobj(qeparse_in, structure_in):
    kmesh, kshift = gen_kpoints_file(structure_in)

    # Update nat entry
    list_system_new = qeparse_in.list_system
    for i, elems in enumerate(list_system_new):
        if "nat" in elems:
            list_system_new[i] = "    nat = %d\n" % structure_in.num_sites

    # Update Cell parameters
    list_CELL_PARAMETERS_new = []
    list_CELL_PARAMETERS_new.append(qeparse_in.list_cell_parameters[0])
    for i in range(3):
        str_tmp = ""
        for j in range(3):
            str_tmp += str("%20.15f" % structure_in.lattice.matrix[i][j])
        str_tmp += "\n"

        list_CELL_PARAMETERS_new.append(str_tmp)

    # Update Kpoint
    list_kmesh = []
    list_kmesh.extend([str(i) for i in kmesh[0]])
    list_kmesh.extend([str(i) for i in kshift])
    str_kmesh = ""
    for entry in list_kmesh:
        str_kmesh += "%3s" % entry

    str_kmesh += "\n"
    list_K_POINTS_new = []
    list_K_POINTS_new.append(qeparse_in.list_k_points[0])
    list_K_POINTS_new.append(str_kmesh)

    qeparse_in.lattice_vector = structure_in.lattice.matrix.transpose()
    qeparse_in.x_fractional = structure_in.frac_coords
    qeparse_in.kd_in_str = [str(t) for t in structure_in.species]
    qeparse_in.nat = structure_in.num_sites
    qeparse_in.list_system = list_system_new
    qeparse_in.list_cell_parameters = list_CELL_PARAMETERS_new
    qeparse_in.list_k_points = list_K_POINTS_new

    return qeparse_in


def run_displacement(
    file_primitive, prefix, scaling_matrix, disp_magnitude_angstrom, symprec
):
    qeobj = QEParser()
    qeobj.load_initial_structure(file_primitive)

    structure = Structure(
        qeobj.lattice_vector.transpose(), qeobj.kd_in_str, qeobj.x_fractional
    )

    Structure.make_supercell(structure, scaling_matrix)

    print("Supercell generated. # Atoms: %i" % structure.num_sites)
    print("")

    prefix0 = "supercell"
    disp = np.zeros((structure.num_sites, 3))

    # update structural information of qeobj
    qeobj_mod = update_qeobj(qeobj, structure)

    # create the supercell structure
    qeobj_mod.generate_structures(prefix0, ["original"], [disp])
    # rename the file
    command = "mv %s1.pw.in %s0.scf.in" % (prefix0, prefix0)
    os.system(command)

    # Generate displacement files
    gen_alm_input(
        "ALM0.in", prefix0, "suggest", structure, 1, "*-* None", symprec=symprec
    )
    command = "%s/alm/alm ALM0.in > ALM0.log" % ALAMODE_root
    os.system(command)

    dispobj = AlamodeDisplace("fd", qeobj_mod, verbosity=0)
    header_list, disp_list, _ = dispobj.generate(
        file_pattern=["%s.pattern_HARMONIC" % prefix0],
        magnitude=disp_magnitude_angstrom,
    )

    qeobj_mod.generate_structures(prefix, header_list, disp_list)


def run_optimize(file_primitive, file_dfset, scaling_matrix, symprec, alamode_ver):
    qeobj = QEParser()
    qeobj.load_initial_structure(file_primitive)

    structure = Structure(
        qeobj.lattice_vector.transpose(), qeobj.kd_in_str, qeobj.x_fractional
    )

    Structure.make_supercell(structure, scaling_matrix)

    print("Supercell generated. # Atoms: %i" % structure.num_sites)
    print("")

    prefix0 = "supercell"
    # Generate displacement files
    gen_alm_input(
        "ALM1.in",
        prefix0,
        "optimize",
        structure,
        1,
        "*-* None",
        dfset=file_dfset,
        symprec=symprec,
    )
    command = "%s/alm/alm ALM1.in > ALM1.log" % ALAMODE_root
    os.system(command)


def gen_bzpath(structure):
    cell = (structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers)

    path_info = seekpath.get_path(cell, True, "hpkot", 1.0e-3, 1.0e-3, 1.0)

    return path_info


def gen_phband(file_primitive, symprec=1.0e-3, alamode_ver=1):
    qeobj = QEParser()
    qeobj.load_initial_structure(file_primitive)

    structure = Structure(
        qeobj.lattice_vector.transpose(), qeobj.kd_in_str, qeobj.x_fractional
    )

    path_info = gen_bzpath(structure)

    prefix0 = "supercell"
    gen_anphon_input(
        "phband.in",
        prefix0,
        "phonons",
        structure,
        path_info,
        alamode_ver=alamode_ver,
        symprec=symprec,
    )

    command = "%s/anphon/anphon phband.in > phband.log" % ALAMODE_root
    os.system(command)


if __name__ == "__main__":
    args = parser.parse_args()
    disp_magnitude_angstrom, prefix, scaling_matrix = process_args(args)

    if args.dfset is None:
        run_displacement(
            args.file_primitive,
            prefix,
            scaling_matrix,
            disp_magnitude_angstrom,
            args.symprec,
        )
    else:
        run_optimize(
            args.file_primitive,
            args.dfset,
            scaling_matrix,
            args.symprec,
            args.alamode_ver,
        )

        gen_phband(
            args.file_primitive, symprec=args.symprec, alamode_ver=args.alamode_ver
        )
