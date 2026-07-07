/*
system.cpp

Copyright (c) 2014, 2015, 2016 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory 
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "system.h"
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include "constants.h"
#include "error.h"
#include "fcs_phonon.h"
#include "hdf5_parser.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "relaxation.h"
#include "symmetry_core.h"
#include "xml_parser.h"

#include <highfive/H5Easy.hpp>

using namespace PHON_NS;

System::System(PHON *phon) : Pointers(phon)
{
    set_default_variables();
}

System::~System()
{
    deallocate_variables();
}

void System::set_default_variables()
{
    load_primitive_from_file = 0;
    symbol_kd.clear();
    mass_kd.clear();
    tolerance_for_coordinates = 1.0e-4;
}

void System::deallocate_variables()
{}

void System::setup()
{
    MPI_Bcast(&load_primitive_from_file, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(lavec_p_input.data(), 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Tmin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Tmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dT, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    load_system_info_from_file();
    update_primitive_lattice();

    if (mympi->my_rank == 0) {
        if (mass_kd.empty()) {
            const auto nkd_tmp = symbol_kd.size();
            mass_kd.resize(nkd_tmp);
            set_mass_elem_from_database(nkd_tmp, symbol_kd, mass_kd);
        } else {
            if (mass_kd.size() != symbol_kd.size()) {
                exit("System::setup", "The number of elements in mass_kd and symbol_kd.");
            }
        }
    } else {
        mass_kd.resize(symbol_kd.size());
    }
    MPI_Bcast(&mass_kd[0], symbol_kd.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    mass_super.resize(supercell[0].number_of_atoms);
    mass_prim.resize(primcell.number_of_atoms);
    invsqrt_mass_p.resize(primcell.number_of_atoms);

    for (auto i = 0; i < supercell[0].number_of_atoms; ++i) {
        mass_super[i] = mass_kd[supercell[0].kind[i]] * amu_ry;
    }
    for (auto i = 0; i < primcell.number_of_atoms; ++i) {
        mass_prim[i] = mass_kd[primcell.kind[i]] * amu_ry;
    }
    for (auto i = 0; i < primcell.number_of_atoms; ++i) {
        invsqrt_mass_p[i] = 1.0 / std::sqrt(mass_prim[i]);
    }

    generate_mapping_tables();

    initialize_distorted_primitive_cell();

    // Set atomic types (kind + magmom)
    set_atomtype_group(primcell, spin_prim, atomtype_group_prim);
    set_atomtype_group(primcell_distort, spin_prim, atomtype_group_prim_distort);

    if (mympi->my_rank == 0) {
        print_structure_information_stdout();
    }
}


void System::print_structure_information_stdout() const
{
    using namespace std;

    cout << " -----------------------------------------------------------------\n\n";
    cout << " ===================\n";
    cout << "  Crystal Structure \n";
    cout << " ===================\n\n";
    cout << " Lattice Vectors:\n\n";
    cout.setf(ios::scientific);

    cout << " * Primitive cell ";
    if (load_primitive_from_file) {
        cout << " (from input file)\n\n";
    } else {
        if (!fcs_phonon->file_fcs.empty()) {
            cout << " (from " << fcs_phonon->file_fcs << ")\n\n";
        } else {
            cout << " (from " << fcs_phonon->file_fc2 << ")\n\n";
        }
    }

    for (auto i = 0; i < 3; ++i) {
        cout << setw(16) << primcell.lattice_vector(0, i);
        cout << setw(15) << primcell.lattice_vector(1, i);
        cout << setw(15) << primcell.lattice_vector(2, i);
        cout << (" : a" + std::to_string(i + 1)) << '\n';
    }
    cout << '\n';

    for (auto i = 0; i < 3; ++i) {
        cout << setw(16) << primcell.reciprocal_lattice_vector(i, 0);
        cout << setw(15) << primcell.reciprocal_lattice_vector(i, 1);
        cout << setw(15) << primcell.reciprocal_lattice_vector(i, 2);
        cout << (" : b" + std::to_string(i + 1)) << '\n';
    }
    cout << "\n\n";

    cout << "  Volume of the primitive cell : " << primcell.volume << " (a.u.)^3\n\n";
    cout << "  Number of atoms in the primitive cell: " << primcell.number_of_atoms << "\n\n";

    cout << "  Atomic positions in the primitive cell (fractional):\n";
    for (auto i = 0; i < primcell.number_of_atoms; ++i) {
        cout << setw(4) << i + 1 << ":";
        for (auto j = 0; j < 3; ++j) {
            cout << setw(15) << primcell.x_fractional(i, j);
        }
        cout << setw(4) << symbol_kd[primcell.kind[i]] << '\n';
    }
    cout << '\n';

    if (spin_prim.lspin) {
        cout << "  MagneticMoments entry found in the XML file. \n";
        cout << "  Magnetic moment in Cartesian coordinates: \n";
        for (auto i = 0; i < primcell.number_of_atoms; ++i) {
            cout << setw(4) << i + 1 << ":";
            for (auto j = 0; j < 3; ++j) {
                cout << setw(15) << spin_prim.magmom[i][j];
            }
            cout << '\n';
        }
        cout << '\n';
        if (spin_prim.noncollinear == 0) {
            cout << "  Collinear calculation: magnetic moments are considered as scalar variables.\n";
        } else if (spin_prim.noncollinear == 1) {
            cout << "  Noncollinear calculation: magnetic moments are considered as vector variables.\n";
            if (spin_prim.time_reversal_symm) {
                cout << "  Time-reversal symmetry will be considered for generating magnetic space group\n";
            } else {
                cout << "  Time-reversal symmetry will NOT be considered for generating magnetic space group\n";
            }
        }
        cout << '\n';
    }

    cout << "  Mass of atomic species (u):\n";
    for (auto i = 0; i < primcell.number_of_elems; ++i) {
        cout << setw(4) << symbol_kd[i] << ":";
        cout << fixed << setw(12) << mass_kd[i] << '\n';
    }
    cout << "\n\n";

    Eigen::Matrix3d transformation_matrix;

    transformation_matrix = primcell.lattice_vector.inverse() * supercell[0].lattice_vector;

    cout << " * Supercell for harmonic \n\n";
    for (auto i = 0; i < 3; ++i) {
        cout << setw(16) << supercell[0].lattice_vector(0, i);
        cout << setw(15) << supercell[0].lattice_vector(1, i);
        cout << setw(15) << supercell[0].lattice_vector(2, i);
        cout << (" : a" + std::to_string(i + 1)) << '\n';
    }
    cout << '\n';
    cout << "  Transformation matrix:\n";
    for (auto i = 0; i < 3; ++i) {
        cout << setw(16) << transformation_matrix(0, i);
        cout << setw(16) << transformation_matrix(1, i);
        cout << setw(16) << transformation_matrix(2, i);
        cout << '\n';
    }
    const auto det_mat = transformation_matrix.determinant();
    if (std::abs(static_cast<double>(nint(det_mat)) - det_mat) > eps4) {
        exit("print_structure_information_stdout",
             "The transformation matrix should be composed of integers. Something is wrong.");
    }
    cout << '\n' << "  Number of atoms in the supercell     : " << supercell[0].number_of_atoms << "\n\n\n";
}

void System::load_system_info_from_file()
{
    // Parse structure information either from the XML file or h5 file
    int filetype[4]; // filetype[0] for FCSFILE, filetype[X-1] for FCXFILE (X=2, 3, 4)
    // -1: FC?FILE not given, 0: FC?FILE in XML format, 1: FC?FLIE in HDF5 format
    std::vector<std::string> filename_list{fcs_phonon->file_fcs,
                                           fcs_phonon->file_fc2,
                                           fcs_phonon->file_fc3,
                                           fcs_phonon->file_fc4};

    supercell.resize(3);
    map_super_alm.resize(3);
    map_prim_alm.resize(3);

    if (mympi->my_rank == 0) {
        for (auto i = 0; i < filename_list.size(); ++i) {
            const auto filename = filename_list[i];
            if (!filename.empty()) {
                const auto file_extension = filename.substr(filename.find_last_of('.') + 1);
                if (file_extension == "xml" || file_extension == "XML") {
                    filetype[i] = 0;
                } else if (file_extension == "h5" || file_extension == "hdf5") {
                    filetype[i] = 1;
                }
            } else {
                filetype[i] = -1;
            }
        }
    } else {
        filename_list.resize(4);
    }
    MPI_Bcast(&filetype[0], 4, MPI_INT, 0, MPI_COMM_WORLD);
    for (auto i = 0; i < 4; ++i) {
        mympi->MPI_Bcast_string(filename_list[i], 0, MPI_COMM_WORLD);
    }

    Spin spin_super_fc2, spin_prim_fc2;
    std::vector<std::string> elements_base, elements_fc2, elements_tmp;
    Cell primcell_base, primcell_fc2;
    Cell supercell_base;
    MappingTable map_scell_base, map_pcell_base;
    Spin spin_super_base, spin_prim_base;


    for (auto i = 0; i < 4; ++i) {

        Cell scell, pcell;
        Spin spin_s, spin_p;
        MappingTable map_s, map_p;

        if (filetype[i] == -1) continue;

        if (filetype[i] == 0) {
            get_structure_and_mapping_table_xml(filename_list[i],
                                                scell,
                                                pcell,
                                                spin_s,
                                                spin_p,
                                                map_s,
                                                map_p,
                                                elements_tmp);
        } else if (filetype[i] == 1) {
            get_structure_and_mapping_table_h5(filename_list[i],
                                               scell,
                                               pcell,
                                               spin_s,
                                               spin_p,
                                               map_s,
                                               map_p,
                                               elements_tmp);
        } else {
            exit("load_system_info_from_file", "This cannot happen.");
        }

        if (i == 0) {
            supercell_base = scell;
            primcell_base = pcell;
            map_scell_base = map_s;
            map_pcell_base = map_p;
            spin_super_base = spin_s;
            spin_prim_base = spin_p;
            elements_base = elements_tmp;

        } else if (i == 1) {
            supercell[0] = scell;
            map_super_alm[0] = map_s;
            map_prim_alm[0] = map_p;
            primcell_fc2 = pcell;
            spin_super_fc2 = spin_s;
            spin_prim_fc2 = spin_p;
            elements_fc2 = elements_tmp;

        } else if (i == 2) {
            supercell[1] = scell;
            map_super_alm[1] = map_s;
            map_prim_alm[1] = map_p;

        } else if (i == 3) {
            supercell[2] = scell;
            map_super_alm[2] = map_s;
            map_prim_alm[2] = map_p;
        }
    }

    // If FCSFILE is not defined in input, set FC2FILE information as base.
    if (filetype[0] == -1) {
        supercell_base = supercell[0];
        primcell_base = primcell_fc2;
        map_scell_base = map_super_alm[0];
        map_pcell_base = map_prim_alm[0];
        spin_super_base = spin_super_fc2;
        spin_prim_base = spin_prim_fc2;
        elements_base = elements_fc2;
    }

    if (filetype[1] == -1) {
        supercell[0] = supercell_base;
        map_super_alm[0] = map_scell_base;
        map_prim_alm[0] = map_pcell_base;
    }

    // Copy data if FC3FILE is not given.
    if (filetype[2] == -1) {
        supercell[1] = supercell_base;
        map_super_alm[1] = map_scell_base;
        map_prim_alm[1] = map_pcell_base;
    }

    // Copy data if FC4FILE is not given.
    if (filetype[3] == -1) {
        supercell[2] = supercell_base;
        map_super_alm[2] = map_scell_base;
        map_prim_alm[2] = map_pcell_base;
    }

    primcell = primcell_base;
    spin_super = spin_super_base;
    spin_prim = spin_prim_base;

    if (symbol_kd.empty()) {
        std::copy(elements_base.begin(), elements_base.end(), std::back_inserter(symbol_kd));
    } else {
        if (symbol_kd.size() != elements_base.size()) {
            exit("load_system_info_from_file",
                 "The number of elements in the input file (KD) and that from the FCSFILE are different.");
        }
    }
}

void System::get_structure_and_mapping_table_xml(const std::string &filename, Cell &scell_out, Cell &pcell_out,
                                                 Spin &spin_super_out, Spin &spin_prim_out, MappingTable &map_super_out,
                                                 MappingTable &map_prim_out, std::vector<std::string> &elements) const
{
    unsigned int nat_tmp, nkd_tmp, ntran_tmp, natmin_tmp;
    double lavec_s_tmp[3][3];
    NDArray<double, 2> xr_s_tmp;
    NDArray<int, 1> kd_tmp;
    NDArray<unsigned int, 2> map_p2s_tmp;
    NDArray<Maps, 1> map_s2p_tmp;
    NDArray<double, 2> magmom_tmp;
    int lspin_tmp, noncollinear_tmp, time_reversal_symmetry_tmp;

    elements.clear();

    if (mympi->my_rank == 0) {

        using namespace boost::property_tree;
        ptree pt;

        std::map<std::string, int> dict_atomic_kind;

        try {
            read_xml(filename, pt);
        } catch (std::exception &e) {
            std::string str_error = "Cannot open file FCSFILE ( " + filename + " )";
            exit("get_structure_and_mapping_table_xml", str_error.c_str());
        }

        // Parse nat_base and ntran_super
        nat_tmp = boost::lexical_cast<unsigned int>(get_value_from_xml(pt, "Data.Structure.NumberOfAtoms"));
        nkd_tmp = boost::lexical_cast<unsigned int>(get_value_from_xml(pt, "Data.Structure.NumberOfElements"));
        ntran_tmp = boost::lexical_cast<unsigned int>(get_value_from_xml(pt, "Data.Symmetry.NumberOfTranslations"));

        natmin_tmp = nat_tmp / ntran_tmp;

        // Parse lattice vectors
        std::stringstream ss;

        for (auto i = 0; i < 3; ++i) {
            ss.str("");
            ss.clear();
            ss << get_value_from_xml(pt, "Data.Structure.LatticeVector.a" + std::to_string(i + 1));
            ss >> lavec_s_tmp[0][i] >> lavec_s_tmp[1][i] >> lavec_s_tmp[2][i];
        }

        // Parse atomic elements and coordinates

        xr_s_tmp.resize(nat_tmp, 3);
        kd_tmp.resize(nat_tmp);

        BOOST_FOREACH (const ptree::value_type &child_, pt.get_child("Data.Structure.AtomicElements")) {
            const auto &child = child_.second;
            const auto icount_kd = child.get<unsigned int>("<xmlattr>.number");
            dict_atomic_kind[boost::lexical_cast<std::string>(child_.second.data())] = icount_kd - 1;
            elements.emplace_back(child_.second.data());
        }

        unsigned int index;

        BOOST_FOREACH (const ptree::value_type &child_, pt.get_child("Data.Structure.Position")) {
            const auto &child = child_.second;
            const auto str_index = child.get<std::string>("<xmlattr>.index");
            const auto str_element = child.get<std::string>("<xmlattr>.element");

            ss.str("");
            ss.clear();
            ss << child.data();

            index = boost::lexical_cast<unsigned int>(str_index) - 1;

            if (index >= nat_tmp) exit("load_system_info_xml", "index is out of range");

            kd_tmp[index] = dict_atomic_kind[str_element];
            ss >> xr_s_tmp[index][0] >> xr_s_tmp[index][1] >> xr_s_tmp[index][2];
        }

        dict_atomic_kind.clear();

        // Parse mapping information

        map_p2s_tmp.resize(natmin_tmp, ntran_tmp);
        map_s2p_tmp.resize(nat_tmp);

        BOOST_FOREACH (const ptree::value_type &child_, pt.get_child("Data.Symmetry.Translations")) {
            const auto &child = child_.second;
            const auto str_tran = child.get<std::string>("<xmlattr>.tran");
            const auto str_atom = child.get<std::string>("<xmlattr>.atom");

            const auto tran = boost::lexical_cast<unsigned int>(str_tran) - 1;
            const auto atom_p = boost::lexical_cast<unsigned int>(str_atom) - 1;
            const auto atom_s = boost::lexical_cast<unsigned int>(child.data()) - 1;

            if (tran >= ntran_tmp || atom_p >= natmin_tmp || atom_s >= nat_tmp) {
                exit("load_system_info_xml", "index is out of range");
            }

            map_p2s_tmp[atom_p][tran] = atom_s;
            map_s2p_tmp[atom_s].atom_num = atom_p;
            map_s2p_tmp[atom_s].tran_num = tran;
        }

        // Parse magnetic moments

        NDArray<double, 2> magmom_tmp2;
        magmom_tmp2.resize(nat_tmp, 3);
        magmom_tmp.resize(natmin_tmp, 3);

        lspin_tmp = true;
        try {
            BOOST_FOREACH (const ptree::value_type &child_, pt.get_child("Data.MagneticMoments")) {
                if (child_.first == "mag") {
                    const auto &child = child_.second;
                    const auto str_index = child.get<std::string>("<xmlattr>.index");

                    ss.str("");
                    ss.clear();
                    ss << child.data();

                    index = boost::lexical_cast<unsigned int>(str_index) - 1;

                    if (index >= nat_tmp) exit("load_system_info_xml", "index is out of range");

                    ss >> magmom_tmp2[index][0] >> magmom_tmp2[index][1] >> magmom_tmp2[index][2];
                }
            }

        } catch (...) {
            lspin_tmp = false;
        }

        if (lspin_tmp) {
            for (auto i = 0; i < natmin_tmp; ++i) {
                for (int j = 0; j < 3; ++j) {
                    magmom_tmp[i][j] = magmom_tmp2[map_p2s_tmp[i][0]][j];
                }
            }

            try {
                noncollinear_tmp =
                    boost::lexical_cast<int>(get_value_from_xml(pt, "Data.MagneticMoments.Noncollinear"));
            } catch (...) {
                noncollinear_tmp = 0;
            }

            try {

                time_reversal_symmetry_tmp =
                    boost::lexical_cast<int>(get_value_from_xml(pt, "Data.MagneticMoments.TimeReversalSymmetry"));
            } catch (...) {
                time_reversal_symmetry_tmp = 1;
            }
        } else {
            for (auto i = 0; i < natmin_tmp; ++i) {
                for (int j = 0; j < 3; ++j) {
                    magmom_tmp[i][j] = 0.0;
                }
            }
            noncollinear_tmp = 0;
            time_reversal_symmetry_tmp = 1;
        }
        magmom_tmp2.clear();
    }

    MPI_Bcast(&lavec_s_tmp[0][0], 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nkd_tmp, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nat_tmp, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&natmin_tmp, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ntran_tmp, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lspin_tmp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&noncollinear_tmp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&time_reversal_symmetry_tmp, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mympi->my_rank > 0) {
        xr_s_tmp.resize(nat_tmp, 3);
        kd_tmp.resize(nat_tmp);
        map_p2s_tmp.resize(natmin_tmp, ntran_tmp);
        map_s2p_tmp.resize(nat_tmp);

        if (lspin_tmp) {
            magmom_tmp.resize(natmin_tmp, 3);
        }
        elements.resize(nkd_tmp);
    }

    MPI_Bcast(&xr_s_tmp[0][0], 3 * nat_tmp, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kd_tmp[0], nat_tmp, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    for (auto i = 0; i < nkd_tmp; ++i) {
        mympi->MPI_Bcast_string(elements[i], 0, MPI_COMM_WORLD);
    }

    MPI_Bcast(&map_p2s_tmp[0][0], natmin_tmp * ntran_tmp, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&map_s2p_tmp[0], nat_tmp * sizeof map_s2p_tmp[0], MPI_BYTE, 0, MPI_COMM_WORLD);
    if (lspin_tmp) MPI_Bcast(&magmom_tmp[0][0], 3 * natmin_tmp, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    Eigen::Matrix3d lattice_vector, reciprocal_lattice_vector;
    Eigen::MatrixXd xf(nat_tmp, 3);
    std::vector<int> kind(nat_tmp);

    // Structure information
    scell_out.number_of_atoms = nat_tmp;
    scell_out.number_of_elems = nkd_tmp;
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            lattice_vector(i, j) = lavec_s_tmp[i][j];
        }
    }
    reciprocal_lattice_vector = tpi * lattice_vector.inverse();
    scell_out.lattice_vector = lattice_vector;
    scell_out.reciprocal_lattice_vector = reciprocal_lattice_vector;
    for (auto i = 0; i < nat_tmp; ++i) {
        for (auto j = 0; j < 3; ++j) {
            xf(i, j) = xr_s_tmp[i][j];
        }
        kind[i] = kd_tmp[i];
    }
    scell_out.x_fractional = xf;
    scell_out.x_cartesian = xf * lattice_vector.transpose();
    scell_out.kind = kind;
    scell_out.volume = volume(lattice_vector, Direct);

    // Magnetism information
    spin_super_out.lspin = lspin_tmp;
    spin_super_out.time_reversal_symm = time_reversal_symmetry_tmp;
    spin_super_out.noncollinear = noncollinear_tmp;

    if (lspin_tmp) {
        spin_super_out.magmom.resize(nat_tmp, std::vector<double>(3));
        for (auto i = 0; i < nat_tmp; ++i) {
            for (auto j = 0; j < 3; ++j) {
                spin_super_out.magmom[i][j] = magmom_tmp[i][j];
            }
        }
    }

    // Mapping table
    map_super_out.to_true_primitive.resize(nat_tmp);
    map_super_out.from_true_primitive.resize(natmin_tmp, std::vector<unsigned int>(ntran_tmp));

    for (auto i = 0; i < nat_tmp; ++i) {
        map_super_out.to_true_primitive[i] = map_s2p_tmp[i];
    }

    for (auto i = 0; i < natmin_tmp; ++i) {
        for (auto j = 0; j < ntran_tmp; ++j) {
            map_super_out.from_true_primitive[i][j] = map_p2s_tmp[i][j];
        }
    }

    scell_out.has_entry = 1;

    xr_s_tmp.clear();
    kd_tmp.clear();
    magmom_tmp.clear();
    map_p2s_tmp.clear();
}

void System::get_structure_and_mapping_table_h5(const std::string &filename, Cell &scell_out, Cell &pcell_out,
                                                Spin &spin_super_out, Spin &spin_prim_out, MappingTable &map_super_out,
                                                MappingTable &map_prim_out, std::vector<std::string> &elements) const
{
    using namespace H5Easy;

    int natmin_tmp, ntran_tmp;

    if (mympi->my_rank == 0) {
        File file(filename, File::ReadOnly);

        const std::string celltype_s = "SuperCell";
        const std::string celltype_p = "PrimitiveCell";
        std::string unit_lavec_s, unit_lavec_p;

        get_structures_from_h5(file,
                               celltype_s,
                               scell_out.lattice_vector,
                               scell_out.x_fractional,
                               scell_out.kind,
                               elements,
                               &unit_lavec_s);

        get_structures_from_h5(file,
                               celltype_p,
                               pcell_out.lattice_vector,
                               pcell_out.x_fractional,
                               pcell_out.kind,
                               elements,
                               &unit_lavec_p);

        // The helper already converted the lattice; report only when the
        // stored unit differs from the internal one (bohr).
        if ((!unit_lavec_s.empty() && unit_lavec_s != "bohr") || (!unit_lavec_p.empty() && unit_lavec_p != "bohr")) {
            std::cout << "  " << filename << ": lattice_vector stored in "
                      << (unit_lavec_s.empty() ? unit_lavec_p : unit_lavec_s) << " -> converted to bohr\n\n";
        }

        get_magnetism_from_h5(file,
                              celltype_s,
                              spin_super_out.lspin,
                              spin_super_out.magmom,
                              spin_super_out.noncollinear,
                              spin_super_out.time_reversal_symm);

        get_magnetism_from_h5(file,
                              celltype_p,
                              spin_prim_out.lspin,
                              spin_prim_out.magmom,
                              spin_prim_out.noncollinear,
                              spin_prim_out.time_reversal_symm);

        std::vector<std::vector<int>> mapping_table;
        get_mapping_table_from_h5(file, celltype_s, mapping_table);

        natmin_tmp = mapping_table.size();
        ntran_tmp = mapping_table[0].size();

        map_super_out.from_true_primitive.resize(natmin_tmp, std::vector<unsigned int>(ntran_tmp));
        map_super_out.to_true_primitive.resize(natmin_tmp * ntran_tmp);

        size_t atom_s;

        for (auto i = 0; i < natmin_tmp; ++i) {
            for (auto j = 0; j < ntran_tmp; ++j) {
                atom_s = mapping_table[i][j];
                map_super_out.from_true_primitive[i][j] = atom_s;
                map_super_out.to_true_primitive[atom_s].atom_num = i;
                map_super_out.to_true_primitive[atom_s].tran_num = j;
            }
        }

        mapping_table.clear();
        get_mapping_table_from_h5(file, celltype_p, mapping_table);

        natmin_tmp = mapping_table.size();
        ntran_tmp = mapping_table[0].size();

        map_prim_out.from_true_primitive.resize(natmin_tmp, std::vector<unsigned int>(ntran_tmp));
        map_prim_out.to_true_primitive.resize(natmin_tmp * ntran_tmp);

        for (auto i = 0; i < natmin_tmp; ++i) {
            for (auto j = 0; j < ntran_tmp; ++j) {
                atom_s = mapping_table[i][j];
                map_prim_out.from_true_primitive[i][j] = atom_s;
                map_prim_out.to_true_primitive[atom_s].atom_num = i;
                map_prim_out.to_true_primitive[atom_s].tran_num = j;
            }
        }
    }

    // Broadcast data
    mympi->MPI_Bcast_CellClass(scell_out, 0, MPI_COMM_WORLD);
    mympi->MPI_Bcast_CellClass(pcell_out, 0, MPI_COMM_WORLD);
    mympi->MPI_Bcast_SpinClass(spin_super_out, 0, MPI_COMM_WORLD);
    mympi->MPI_Bcast_SpinClass(spin_prim_out, 0, MPI_COMM_WORLD);
    mympi->MPI_Bcast_MappingTable(map_super_out, 0, MPI_COMM_WORLD);
    mympi->MPI_Bcast_MappingTable(map_prim_out, 0, MPI_COMM_WORLD);

    int nkd_tmp;
    if (mympi->my_rank == 0) {
        nkd_tmp = elements.size();
    }
    MPI_Bcast(&nkd_tmp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mympi->my_rank != 0) {
        elements.resize(nkd_tmp);
    }
    for (auto i = 0; i < nkd_tmp; ++i) {
        mympi->MPI_Bcast_string(elements[i], 0, MPI_COMM_WORLD);
    }

    // Fill in missing data
    scell_out.number_of_atoms = scell_out.x_fractional.rows();
    scell_out.number_of_elems = elements.size();
    pcell_out.number_of_atoms = pcell_out.x_fractional.rows();
    pcell_out.number_of_elems = elements.size();
    scell_out.reciprocal_lattice_vector = tpi * scell_out.lattice_vector.inverse();
    pcell_out.reciprocal_lattice_vector = tpi * pcell_out.lattice_vector.inverse();
    scell_out.x_cartesian = scell_out.x_fractional * scell_out.lattice_vector.transpose();
    pcell_out.x_cartesian = pcell_out.x_fractional * pcell_out.lattice_vector.transpose();
    scell_out.volume = volume(scell_out.lattice_vector, Direct);
    pcell_out.volume = volume(pcell_out.lattice_vector, Direct);

    scell_out.has_entry = 1;
    pcell_out.has_entry = 1;
}


void System::update_primitive_lattice()
{
    // Update primcell_base and spin_prim_base.

    if (!primcell.has_entry && (load_primitive_from_file == 0)) {
        exit("System::update_primitive_lattice()",
             "Primitive lattice vectors must be given in the &cell field, \n"
             "because the corresponding data does not exist in FCSFILE.");
    }

    MPI_Bcast(lavec_p_input.data(), 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (load_primitive_from_file == 1) {
        // Update the information of primcell using the information of supercell[0]
        // and the given lattice vector.

        Eigen::Matrix3d transmat_to_prim = supercell[0].lattice_vector.inverse() * lavec_p_input;

        //std::cout << "transmat_to_prin:" << transmat_to_prim << '\n';

        primcell.lattice_vector = lavec_p_input;
        recips(primcell.lattice_vector, primcell.reciprocal_lattice_vector);
        primcell.volume = volume(primcell.lattice_vector, Direct);

        const auto ndiv = nint(1.0 / transmat_to_prim.determinant());
        if (supercell[0].number_of_atoms % ndiv != 0) {
            exit("update_primitive_lattice",
                 "The input primitive cell lattice vector is incommensurate \n "
                 " with the supercell information parsed from FCSFILE.");
        }

        primcell.number_of_atoms = supercell[0].number_of_atoms / ndiv;
        primcell.number_of_elems = supercell[0].number_of_elems;

        spin_prim.lspin = spin_super.lspin;
        spin_prim.noncollinear = spin_super.noncollinear;
        spin_prim.time_reversal_symm = spin_super.time_reversal_symm;
        spin_prim.magmom.clear();
        spin_prim.magmom.shrink_to_fit();

        // Convert the basis of coordinates from inputcell to primitive fractional
        // (a_in, b_in, c_in) * xf_in = xc_in
        // (a_p, b_p, c_p) * xf_p = xc_in
        // xf_p = (a_p, b_p, c_p)^{-1} * (a_in, b_in, c_in) * xf_in
        //      = [Mat(inp->p)]^{-1} * xf_in
        Eigen::Matrix3d conversion_mat = transmat_to_prim.inverse().transpose();
        Eigen::MatrixXd xf_prim_all = supercell[0].x_fractional * conversion_mat;

        std::vector<std::vector<double>> xf_unique, magmom_unique;
        std::vector<double> xf_tmp_vec(3), magmom_tmp_vec(3);
        std::vector<int> kind_unique;
        Eigen::VectorXd xf_tmp(3), xf_tmp2(3), xf_diff(3);

        for (auto i = 0; i < supercell[0].number_of_atoms; ++i) {
            xf_tmp = xf_prim_all.row(i);
            xf_tmp = xf_tmp.unaryExpr([](const double x) { return std::fmod(x, 1.0); });
            for (auto j = 0; j < 3; ++j) {
                if (xf_tmp[j] < -eps6) xf_tmp[j] += 1.0;
            }

            bool is_duplicate = false;
            // Just linear search for simplicity. Should be OK for relatively small number of inputs.
            for (auto k = 0; k < xf_unique.size(); ++k) {
                for (auto kk = 0; kk < 3; ++kk) {
                    xf_tmp2[kk] = xf_unique[k][kk];
                }
                xf_diff = (xf_tmp - xf_tmp2).unaryExpr([](const double x) { return std::fmod(x, 1.0); });
                for (auto j = 0; j < 3; ++j) {
                    if (xf_diff[j] < -0.5) xf_diff[j] += 1.0;
                    if (xf_diff[j] >= 0.5) xf_diff[j] -= 1.0;
                }
                //std::cout << "xf_diff.norm=" << xf_diff.norm() << '\n';
                if (xf_diff.norm() < tolerance_for_coordinates) {
                    is_duplicate = true;

                    if (kind_unique[k] != supercell[0].kind[i]) {
                        exit("update_primitive_lattice",
                             "Different atoms with different element types occupy the same atomic site.\n"
                             "This is strange. Please check the PRIMCELL and input structure carefully.");
                    }

                    if (spin_super.lspin) {
                        double norm_magmom = 0.0;
                        for (auto kk = 0; kk < 3; ++kk) {
                            norm_magmom += pow2(spin_super.magmom[i][kk] - magmom_unique[k][kk]);
                        }
                        if (std::sqrt(norm_magmom) > eps6) {
                            exit("update_primitive_lattice",
                                 "Different atoms with different MAGMOM entries occupy the same atomic site.\n"
                                 "This is strange. Please check the PRIMCELL, MAGMOM, and input structure carefully.");
                        }
                    }
                }
            }

            if (!is_duplicate) {
                for (auto j = 0; j < 3; ++j) xf_tmp_vec[j] = xf_tmp[j];
                xf_unique.emplace_back(xf_tmp_vec);
                kind_unique.emplace_back(supercell[0].kind[i]);
                if (spin_super.lspin) {
                    for (auto j = 0; j < 3; ++j) magmom_tmp_vec[j] = spin_super.magmom[i][j];
                    magmom_unique.emplace_back(magmom_tmp_vec);
                }
            }
        }


        if (xf_unique.size() != primcell.number_of_atoms) {
            std::cout << "primcell.number_of_atoms = " << primcell.number_of_atoms << '\n';
            std::cout << "xf_unique.size() = " << xf_unique.size() << '\n';
            exit("update_primitive_lattice",
                 "Mapping to the primitive cell failed. "
                 "Please check the lattice constant in the &cell field.");
        }

        primcell.x_fractional.resize(primcell.number_of_atoms, 3);
        primcell.kind.resize(primcell.number_of_atoms);

        for (auto i = 0; i < primcell.number_of_atoms; ++i) {
            for (auto j = 0; j < 3; ++j) {
                primcell.x_fractional(i, j) = xf_unique[i][j];
            }
            primcell.kind[i] = kind_unique[i];
        }

        primcell.x_cartesian = primcell.x_fractional * primcell.lattice_vector.transpose();

        if (spin_prim.lspin) {
            std::copy(magmom_unique.begin(), magmom_unique.end(), std::back_inserter(spin_prim.magmom));
        }
    }
}


void System::generate_mapping_tables()
{
    map_s2p_new.resize(3);
    map_p2s_new.resize(3);
    for (auto i = 0; i < 3; ++i) {
        generate_mapping_primitive_super(primcell, supercell[i], map_p2s_new[i], map_s2p_new[i]);
    }
}

void System::initialize_distorted_primitive_cell()
{
    Eigen::Matrix3d lavec_p_strain, rlavec_p_strain, mat_strain;
    double u_tensor_tmp[3][3];

    if (mympi->my_rank == 0) {
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                u_tensor_tmp[i][j] = relaxation->init_u_tensor[i][j];
            }
        }
    }

    MPI_Bcast(&u_tensor_tmp[0][0], 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (auto i = 0; i < 3; i++) {
        for (auto j = 0; j < 3; j++) {
            mat_strain(i, j) = u_tensor_tmp[i][j];
        }
        mat_strain(i, i) += 1.0;
    }
    lavec_p_strain = mat_strain * primcell.lattice_vector;
    rlavec_p_strain = lavec_p_strain.inverse() * tpi;

    primcell_distort.lattice_vector = lavec_p_strain;
    primcell_distort.reciprocal_lattice_vector = rlavec_p_strain;
    primcell_distort.volume = volume(lavec_p_strain, Direct);
    primcell_distort.number_of_atoms = primcell.number_of_atoms;
    primcell_distort.number_of_elems = primcell.number_of_elems;
    primcell_distort.x_fractional = primcell.x_fractional;

    Eigen::MatrixXd xdisp(primcell_distort.number_of_atoms, 3);
    xdisp.setZero();

    if (mympi->my_rank == 0) {
        if (relaxation->relax_str != 0 && relaxation->init_u0.size() != primcell.number_of_atoms * 3)
            exit("initialize_distorted_primitive_cell",
                 "The number of atoms in the primitive cell"
                 " \n is not consistent with the &displace field in the input file.");

        if (!relaxation->init_u0.empty()) {
            for (auto i = 0; i < primcell_distort.number_of_atoms; ++i) {
                // set displacement in Cartesian coordinates
                for (auto j = 0; j < 3; ++j) {
                    xdisp(i, j) = relaxation->init_u0[i * 3 + j];
                }
            }
        }
    }
    mympi->mpiBcastEigen(xdisp, 0, MPI_COMM_WORLD);

    // Move the basis to the fractional coordinate
    xdisp = inv_tpi * xdisp * primcell_distort.reciprocal_lattice_vector.transpose();
    primcell_distort.x_fractional += xdisp;
    primcell_distort.x_cartesian = primcell_distort.x_fractional * lavec_p_strain.transpose();
    primcell_distort.kind = primcell.kind;
}


void System::generate_mapping_primitive_super(const Cell &pcell, const Cell &scell,
                                              std::vector<std::vector<unsigned int>> &map_p2s_out,
                                              std::vector<Maps> &map_s2p_out) const
{

    Eigen::Vector3d x1, x2, x3, xdiff, tran_d;
    Eigen::Vector3i tran;

    const Eigen::MatrixXd x_super_in_primitive_frac =
        scell.x_fractional * scell.lattice_vector.transpose() * pcell.lattice_vector.inverse().transpose();
    const Eigen::Matrix3d transform_basis_primitive_to_super = scell.lattice_vector.inverse() * pcell.lattice_vector;

    const auto ntran_tmp = scell.number_of_atoms / pcell.number_of_atoms;

    if (scell.number_of_atoms != pcell.number_of_atoms * ntran_tmp) {
        exit("generate_mapping_primitive_super",
             "The number of atoms in the supercell is not divisible "
             "by the number of atoms in the primitive cell. Something is wrong.");
    }

    map_p2s_out.resize(pcell.number_of_atoms, std::vector<unsigned int>(ntran_tmp));

    std::vector<int> flag_found(scell.number_of_atoms, 0);
    std::map<int, int> map_index;
    std::vector<std::map<int, int>> map_index_lists;
    std::vector<int> atom_num_prim;
    std::vector<std::vector<int>> trans_vecs;

    atom_num_prim.clear();
    trans_vecs.clear();
    map_index_lists.clear();

    for (auto iat = 0; iat < scell.number_of_atoms; ++iat) {

        if (flag_found[iat]) continue;

        x1 = x_super_in_primitive_frac.row(iat);

        auto iloc = -1;
        for (auto jat = 0; jat < pcell.number_of_atoms; ++jat) {
            x2 = pcell.x_fractional.row(jat);
            xdiff = (x1 - x2).unaryExpr([](const double x) { return x - static_cast<double>(nint(x)); });

            if (xdiff.norm() < tolerance_for_coordinates && (scell.kind[iat] == pcell.kind[jat])) {
                tran_d = (x1 - x2).unaryExpr([](const double x) { return static_cast<double>(nint(x)); });
                // First move back to the fractional coordinate of the supercell and
                // make sure that the shift vectors are in the 0<=x<1 region in that basis.
                tran_d = transform_basis_primitive_to_super * tran_d;
                tran_d = tran_d.unaryExpr([](const double x) { return x - static_cast<double>(nint(x)); });
                // Then, transform it back to the components in the primitive cell basis.
                // All components should be integer.
                tran_d = transform_basis_primitive_to_super.inverse() * tran_d;
                tran = tran_d.unaryExpr([](const double x) { return nint(x); });
                iloc = jat;
                break;
            }
        }

        if (iloc == -1) {
            exit("generate_mapping_primitive_super", "An equivalent atom not found.");
        } else {
            atom_num_prim.emplace_back(iloc);
            std::vector<int> vtmp(&tran[0], tran.data() + tran.cols() * tran.rows());
            trans_vecs.emplace_back(vtmp);
        }

        tran_d = tran.unaryExpr([](const int x) { return static_cast<double>(x); });

        map_index.clear();

        // Now, we search all atoms in the supercell that can be mapped by the found trans value.
        for (auto j = 0; j < pcell.number_of_atoms; ++j) {
            x2 = pcell.x_fractional.row(j);
            x2 += tran_d;
            x2 = transform_basis_primitive_to_super * x2;

            for (auto k = 0; k < scell.number_of_atoms; ++k) {
                if (flag_found[k]) continue;

                x3 = scell.x_fractional.row(k);
                xdiff = (x3 - x2).unaryExpr([](const double x) { return x - static_cast<double>(nint(x)); });

                if (xdiff.norm() < tolerance_for_coordinates) {
                    flag_found[k] = 1;
                    map_index.insert({j, k});
                }
            }
        }
        map_index_lists.emplace_back(map_index);
    }

    std::set<std::vector<int>> unique_shifts_set;
    std::vector<std::vector<int>> unique_shifts_vec;
    std::vector<int> unique_itran_indices;

    for (auto itran = 0; itran < trans_vecs.size(); ++itran) {
        if (unique_shifts_set.find(trans_vecs[itran]) == unique_shifts_set.end()) {
            unique_shifts_set.insert(trans_vecs[itran]);
            unique_shifts_vec.emplace_back(trans_vecs[itran]);
            unique_itran_indices.emplace_back(itran);
        }
    }

    map_s2p_out.resize(scell.number_of_atoms);

    for (auto iat = 0; iat < pcell.number_of_atoms; ++iat) {
        for (auto itran = 0; itran < unique_itran_indices.size(); ++itran) {
            auto jat = map_index_lists[unique_itran_indices[itran]][iat];
            map_p2s_out[iat][itran] = jat;
            map_s2p_out[jat].atom_num = iat;
            map_s2p_out[jat].tran_num = itran;
        }
    }
}

void System::recips(const Eigen::Matrix3d &mat_in, Eigen::Matrix3d &rmat_out)
{
    const auto det = mat_in.determinant();

    if (std::abs(det) < eps12) {
        exit("recips", "Lattice Vector is singular");
    }
    rmat_out = tpi * mat_in.inverse();
}

//void System::check_consistency_primitive_lattice() const
//{
//    // Check if the ordering of atoms in the primitive cells derived
//    // from FCSFILE and FC2FILE are same or not. If not, the ordering for the
//    // FCSFILE (anharmonic terms) will be changed so that it becomes equivalent
//    // to that of FC2FILE.
//    // This operation is necessary for obtaining correct computational results.
//
//    int i, j, k;
//    Eigen::Vector3d xdiff;
//    Eigen::MatrixXd x_harm(natmin, 3), x_anharm(natmin, 3);
//
//    std::vector<int> map_anh2harm;
//    map_anh2harm.resize(natmin);
//
//    for (i = 0; i < natmin; ++i) {
//        for (j = 0; j < 3; ++j) {
////            x_harm(i, j) = xr_s(map_p2s[i][0], j);
////            x_anharm(i, j) = xr_s_anharm(map_p2s_anharm[i][0], j);
//        }
//    }
//
////    x_harm = x_harm * lavec_s.transpose() * lavec_p_input.inverse().transpose();
//    //x_anharm = x_anharm * lavec_s_anharm.transpose() * lavec_p_input.inverse().transpose();
//
//    for (i = 0; i < natmin; ++i) {
//
//        int iloc = -1;
//
//        for (j = 0; j < natmin; ++j) {
//
//            xdiff = (x_anharm.row(i) - x_harm.row(j)).unaryExpr(
//                    [](const double x) { return x - static_cast<double>(nint(x)); });
//
//            //const auto norm = xdiff[0] * xdiff[0] + xdiff[1] * xdiff[1] + xdiff[2] * xdiff[2];
//            const auto norm = xdiff.squaredNorm();
//            if (norm < eps4 && kd[map_p2s[j][0]] == kd_anharm[map_p2s_anharm[i][0]]) {
//                iloc = j;
//                break;
//            }
//        }
//
//        if (iloc == -1) {
//            exit("check_consistency_primitive",
//                 "Could not find equivalent atom. Probably, the crystal structure is different.");
//        }
//
//        map_anh2harm[i] = iloc;
//    }
//
//    // Rebuild the mapping information for anharmonic terms.
//
//    unsigned int **map_p2s_tmp;
//
//    map_p2s_tmp.resize(natmin, ntran_anharm);
//
//    for (i = 0; i < natmin; ++i) {
//        for (j = 0; j < ntran_anharm; ++j) {
//            map_p2s_anharm_orig[i][j] = map_p2s_anharm[i][j];
//        }
//    }
//
//    for (i = 0; i < ntran_anharm; ++i) {
//        for (j = 0; j < natmin; ++j) {
//            map_p2s_tmp[j][i] = map_p2s_anharm[j][i];
//        }
//    }
//
//    for (i = 0; i < ntran_anharm; ++i) {
//        for (j = 0; j < natmin; ++j) {
//            map_p2s_anharm[map_anh2harm[j]][i] = map_p2s_tmp[j][i];
//        }
//    }
//
//    for (i = 0; i < ntran_anharm; ++i) {
//        for (j = 0; j < natmin; ++j) {
//            k = map_p2s_anharm[j][i];
//            map_s2p_anharm[k].atom_num = j;
//            map_s2p_anharm[k].tran_num = i;
//        }
//    }
//
//    map_p2s_tmp.clear();
//    map_anh2harm.clear();
//}

int System::get_atomic_number_by_name(const std::string &kdname_in) const
{
    auto kdname_copy = kdname_in;
    kdname_copy[0] = toupper(kdname_copy[0]);

    int ret = -1;

    for (auto i = 0; i < element_names.size(); ++i) {
        if (kdname_copy == element_names[i]) {
            ret = i;
            break;
        }
    }
    return ret;
}

void System::set_mass_elem_from_database(const unsigned int nkd, const std::vector<std::string> &symbol_in,
                                         std::vector<double> &mass_kd_out)
{
    for (int i = 0; i < nkd; ++i) {
        const auto atom_number = get_atomic_number_by_name(symbol_in[i]);
        if (atom_number >= element_names.size() || atom_number == -1) {
            exit(
                "set_mass_elem_from_database",
                "Atomic mass for the given element doesn't exist in the database.\nTherefore, please input MASS manually.");
        }
        const auto mass_tmp = atomic_masses[atom_number];
        if (mass_tmp < 0.0) {
            exit("set_mass_elem_from_database",
                 "One of the elements in the KD-tag is unstable. \nTherefore, please input MASS manually.");
        }
        mass_kd_out[i] = mass_tmp;
    }
}

void System::set_atomtype_group(const Cell &cell_in, const Spin &spin_in,
                                std::vector<std::vector<unsigned int>> &atomtype_group_out)
{
    // In the case of collinear calculation, spin moments are considered as scalar
    // variables. Therefore, the same elements with different magnetic moments are
    // considered as different types. In noncollinear calculations,
    // magnetic moments are not considered in this stage. They will be treated
    // separately in symmetry.cpp where spin moments will be rotated and flipped
    // using time-reversal symmetry.

    unsigned int i;
    AtomType type_tmp{};
    std::set<AtomType> set_type;
    set_type.clear();

    for (i = 0; i < cell_in.number_of_atoms; ++i) {
        type_tmp.element = cell_in.kind[i];

        if (spin_in.lspin == 1 && spin_in.noncollinear == 0) {
            type_tmp.magmom = spin_in.magmom[i][2];
        } else {
            type_tmp.magmom = 0.0;
        }
        set_type.insert(type_tmp);
    }

    const auto natomtypes = set_type.size();
    atomtype_group_out.resize(natomtypes);

    for (i = 0; i < cell_in.number_of_atoms; ++i) {
        int count = 0;
        for (auto it: set_type) {
            if (spin_in.noncollinear || spin_in.lspin == 0) {
                if (cell_in.kind[i] == it.element) {
                    atomtype_group_out[count].push_back(i);
                }
            } else {
                if ((cell_in.kind[i] == it.element) && (std::abs(spin_in.magmom[i][2] - it.magmom) < eps6)) {
                    atomtype_group_out[count].push_back(i);
                }
            }
            ++count;
        }
    }
    set_type.clear();
}


double System::volume(const Eigen::Matrix3d &mat_in, const LatticeType latttype_in)
{
    Eigen::Matrix3d mat;
    Eigen::Vector3d v1, v2, v3;

    if (latttype_in == Direct) {
        mat = mat_in.transpose();
    } else if (latttype_in == Reciprocal) {
        mat = mat_in;
    } else {
        exit("volume", "Invalid LatticeType is given");
    }

    v1 = mat.row(0);
    v2 = mat.row(1);
    v3 = mat.row(2);

    const auto vol = std::abs(v1.dot(v2.cross(v3)));
    return vol;
}

auto System::get_primcell(const bool distorted) const -> const Cell &
{
    // return the primitive cell.
    // Note that the "primitive cell" is the one that is used in the phonon calculation.
    // It may not be a true primitive cell if a conventional cell is spefified in the &cell field.
    if (distorted) {
        return primcell_distort;
    }
    return primcell;
}

auto System::get_supercell(const int index) const -> const Cell &
{
    return supercell[index];
}

auto System::get_mapping_super_alm(const int index) const -> const MappingTable &
{
    return map_super_alm[index];
}

const MappingTable &System::get_mapping_prim_alm(const int index) const
{
    return map_prim_alm[index];
}

const std::vector<std::vector<unsigned int>> &System::get_atomtype_group(const bool distort) const
{
    if (distort) return atomtype_group_prim_distort;
    return atomtype_group_prim;
}


const std::vector<double> &System::get_mass_prim() const
{
    return mass_prim;
}

auto System::get_mass_super() const -> const std::vector<double> &
{
    return mass_super;
}

auto System::get_invsqrt_mass() const -> const std::vector<double> &
{
    return invsqrt_mass_p;
}

auto System::get_map_s2p(const int index) const -> const std::vector<Maps> &
{
    return map_s2p_new[index];
}

auto System::get_map_p2s(const int index) const -> const std::vector<std::vector<unsigned int>> &
{
    return map_p2s_new[index];
}

auto System::get_spin_prim() const -> const Spin &
{
    return spin_prim;
}

auto System::get_spin_super() const -> const Spin &
{
    return spin_super;
}

void System::get_minimum_distances(const unsigned int nsize[3], NDArray<MinimumDistList, 3> &mindist_list_out)
{
    // Compute mindist_list necessary to calculate dynamical matrix
    // from the real-space force constants

    int i, j;
    int ix, iy, iz;
    const auto natmin_tmp = primcell.number_of_atoms;
    unsigned int iat;

    NDArray<int, 2> shift_cell;
    NDArray<int, 2> shift_cell_super;
    NDArray<double, 4> x_all;

    const int nkx = static_cast<int>(nsize[0]); // This should be int (must not be unsigned int).
    const int nky = static_cast<int>(nsize[1]); // same as above
    const int nkz = static_cast<int>(nsize[2]); // same as above

    const auto ncell = nsize[0] * nsize[1] * nsize[2];
    const auto ncell_s = 27;

    shift_cell.resize(ncell, 3);
    shift_cell_super.resize(ncell_s, 3);
    x_all.resize(ncell_s, ncell, natmin_tmp, 3);

    // Fresh elements on every call, matching the historical dealloc+alloc.
    mindist_list_out.clear();
    mindist_list_out.resize(natmin_tmp, natmin_tmp, ncell);

    unsigned int icell = 0;
    for (ix = 0; ix < nkx; ++ix) {
        for (iy = 0; iy < nky; ++iy) {
            for (iz = 0; iz < nkz; ++iz) {

                shift_cell[icell][0] = ix;
                shift_cell[icell][1] = iy;
                shift_cell[icell][2] = iz;

                ++icell;
            }
        }
    }

    for (i = 0; i < 3; ++i) shift_cell_super[0][i] = 0;
    icell = 1;
    for (ix = -1; ix <= 1; ++ix) {
        for (iy = -1; iy <= 1; ++iy) {
            for (iz = -1; iz <= 1; ++iz) {
                if (ix == 0 && iy == 0 && iz == 0) continue;

                shift_cell_super[icell][0] = ix;
                shift_cell_super[icell][1] = iy;
                shift_cell_super[icell][2] = iz;

                ++icell;
            }
        }
    }

    const auto xf_p = primcell.x_fractional;

    for (i = 0; i < ncell_s; ++i) {
        for (j = 0; j < ncell; ++j) {
            for (iat = 0; iat < natmin_tmp; ++iat) {
                x_all[i][j][iat][0] = xf_p(iat, 0) + static_cast<double>(shift_cell[j][0]) +
                                      static_cast<double>(nkx * shift_cell_super[i][0]);
                x_all[i][j][iat][1] = xf_p(iat, 1) + static_cast<double>(shift_cell[j][1]) +
                                      static_cast<double>(nky * shift_cell_super[i][1]);
                x_all[i][j][iat][2] = xf_p(iat, 2) + static_cast<double>(shift_cell[j][2]) +
                                      static_cast<double>(nkz * shift_cell_super[i][2]);

                rotvec(x_all[i][j][iat], x_all[i][j][iat], primcell.lattice_vector);
            }
        }
    }

    double dist;
    std::vector<DistList> dist_tmp;
    ShiftCell shift_tmp{};
    std::vector<int> vec_tmp;

    for (iat = 0; iat < natmin_tmp; ++iat) {
        for (unsigned int jat = 0; jat < natmin_tmp; ++jat) {
            for (icell = 0; icell < ncell; ++icell) {

                dist_tmp.clear();
                for (i = 0; i < ncell_s; ++i) {
                    dist = distance(x_all[0][0][iat], x_all[i][icell][jat]);
                    dist_tmp.emplace_back(i, dist);
                }
                std::sort(dist_tmp.begin(), dist_tmp.end());

                const auto dist_min = dist_tmp[0].dist;
                mindist_list_out[iat][jat][icell].dist = dist_min;

                for (i = 0; i < ncell_s; ++i) {
                    dist = dist_tmp[i].dist;

                    if (std::abs(dist_min - dist) < eps8) {

                        shift_tmp.sx = shift_cell[icell][0] + nkx * shift_cell_super[dist_tmp[i].cell_s][0];
                        shift_tmp.sy = shift_cell[icell][1] + nky * shift_cell_super[dist_tmp[i].cell_s][1];
                        shift_tmp.sz = shift_cell[icell][2] + nkz * shift_cell_super[dist_tmp[i].cell_s][2];

                        mindist_list_out[iat][jat][icell].shift.push_back(shift_tmp);
                    }
                }
            }
        }
    }

    shift_cell.clear();
    shift_cell_super.clear();
    x_all.clear();
}
