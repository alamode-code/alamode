/*
 parse_fcsxml.cpp

 Copyright (c) 2021 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "parse_fcsxml.h"
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/version.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include "../include/mathfunctions.h"
#include "memory.h"
#include "xml_parser.h"

using namespace std;

int main(int argc, char *argv[])
{
    std::string file_xml;
    int maxorder;

    file_xml = argv[1];
    maxorder = boost::lexical_cast<int>(argv[2]);

    if (maxorder <= 1) {
        std::cout << "Maxorder should be larger than 1. " << std::endl;
        exit(EXIT_FAILURE);
    }

    maxorder -= 1;

    std::vector<FcsArrayWithCell> *fc;
    StructureProperty structure;

    allocate(fc, maxorder);
    load_fcs_xml(file_xml, maxorder, structure, fc);

    std::string::size_type const p(file_xml.find_last_of('.'));
    std::string prefix = file_xml.substr(0, p);

    int **map_p2s;
    double ***x_image;

    allocate(map_p2s, structure.natmin, structure.ntran);
    allocate(x_image, 27, structure.nat, 3);

    auto icell = 0;
    for (auto i = 0; i < structure.nat; ++i) {
        x_image[0][i][0] = structure.atoms[i].x;
        x_image[0][i][1] = structure.atoms[i].y;
        x_image[0][i][2] = structure.atoms[i].z;
    }
    // Convert to Cartesian coordinate
    frac2cart(x_image[0], structure.nat, structure.lattice_vector);

    for (auto ia = -1; ia <= 1; ++ia) {
        for (auto ja = -1; ja <= 1; ++ja) {
            for (auto ka = -1; ka <= 1; ++ka) {

                if (ia == 0 && ja == 0 && ka == 0) continue;

                ++icell;
                for (auto i = 0; i < structure.nat; ++i) {
                    x_image[icell][i][0] = structure.atoms[i].x + static_cast<double>(ia);
                    x_image[icell][i][1] = structure.atoms[i].y + static_cast<double>(ja);
                    x_image[icell][i][2] = structure.atoms[i].z + static_cast<double>(ka);
                }
                // Convert to Cartesian coordinate
                frac2cart(x_image[icell], structure.nat, structure.lattice_vector);
            }
        }
    }

    for (auto i = 0; i < structure.nat; ++i) {
        map_p2s[structure.atoms[i].atom][structure.atoms[i].tran] = i;
    }

    for (auto order = 0; order < maxorder; ++order) {
        std::string fname_fc = prefix + ".fc" + std::to_string(order + 2);
        write_fcs_to_file(fname_fc, order, structure, x_image, map_p2s, fc[order]);
    }

    deallocate(fc);
    deallocate(x_image);
    deallocate(map_p2s);
}


void load_fcs_xml(const std::string file_in, const int maxorder, StructureProperty &StructProp,
                  std::vector<FcsArrayWithCell> *force_constant_with_cell)
{
    using namespace boost::property_tree;
    ptree pt;
    map<string, int> dict_atomic_kind;
    std::stringstream ss;

    try {
        read_xml(file_in, pt);
    } catch (exception &e) {
        cout << "Cannot open file " + file_in << endl;
        exit(EXIT_FAILURE);
    }

    StructProp.nat = boost::lexical_cast<unsigned int>(get_value_from_xml(pt, "Data.Structure.NumberOfAtoms"));
    StructProp.nspecies = boost::lexical_cast<unsigned int>(get_value_from_xml(pt, "Data.Structure.NumberOfElements"));

    StructProp.ntran = boost::lexical_cast<unsigned int>(get_value_from_xml(pt, "Data.Symmetry.NumberOfTranslations"));


    for (auto i = 0; i < 3; ++i) {
        ss.str("");
        ss.clear();
        ss << get_value_from_xml(pt, "Data.Structure.LatticeVector.a" + boost::lexical_cast<string>(i + 1));
        ss >> StructProp.lattice_vector[0][i] >> StructProp.lattice_vector[1][i] >> StructProp.lattice_vector[2][i];
    }

    ss.str("");
    ss.clear();
    ss << get_value_from_xml(pt, "Data.Structure.Periodicity");
    ss >> StructProp.is_periodic[0] >> StructProp.is_periodic[1] >> StructProp.is_periodic[2];

    // Parse atomic elements and coordinates

    StructProp.kd_symbol.resize(StructProp.nspecies);
    StructProp.atoms.resize(StructProp.nat);

    int i = 0;

    BOOST_FOREACH (const ptree::value_type &child_, pt.get_child("Data.Structure.AtomicElements")) {
        const ptree &child = child_.second;
        const unsigned int icount_kd = child.get<unsigned int>("<xmlattr>.number");
        dict_atomic_kind[boost::lexical_cast<string>(child_.second.data())] = icount_kd - 1;
        StructProp.kd_symbol[i++] = boost::lexical_cast<string>(child_.second.data());
    }

    unsigned int index;

    BOOST_FOREACH (const ptree::value_type &child_, pt.get_child("Data.Structure.Position")) {
        const ptree &child = child_.second;
        const string str_index = child.get<string>("<xmlattr>.index");
        const string str_element = child.get<string>("<xmlattr>.element");

        ss.str("");
        ss.clear();
        ss << child.data();

        index = boost::lexical_cast<unsigned int>(str_index) - 1;

        if (index >= StructProp.nat) {
            cout << "index is out of range" << endl;
            exit(EXIT_FAILURE);
        }

        StructProp.atoms[index].kind = dict_atomic_kind[str_element];
        ss >> StructProp.atoms[index].x >> StructProp.atoms[index].y >> StructProp.atoms[index].z;
    }

    dict_atomic_kind.clear();

    // Parse mapping information

    StructProp.natmin = StructProp.nat / StructProp.ntran;

    unsigned int tran, atom_p, atom_s;

    BOOST_FOREACH (const ptree::value_type &child_, pt.get_child("Data.Symmetry.Translations")) {
        const ptree &child = child_.second;
        const string str_tran = child.get<string>("<xmlattr>.tran");
        const string str_atom = child.get<string>("<xmlattr>.atom");

        tran = boost::lexical_cast<unsigned int>(str_tran) - 1;
        atom_p = boost::lexical_cast<unsigned int>(str_atom) - 1;
        atom_s = boost::lexical_cast<unsigned int>(child.data()) - 1;

        if (tran >= StructProp.ntran || atom_p >= StructProp.natmin || atom_s >= StructProp.nat) {
            cout << "index is out of range" << endl;
            exit(EXIT_FAILURE);
        }

        StructProp.atoms[atom_s].atom = atom_p;
        StructProp.atoms[atom_s].tran = tran;
    }

    // Parse force constants

    std::vector<AtomCellSuper> ivec_with_cell;
    std::string str_tag;
    double fcs_val;
    unsigned int atmn, xyz, cell_s;
    std::string str_pairs;
    std::string str_attr;
    AtomCellSuper ivec_tmp;

    for (auto order = 0; order < maxorder; ++order) {

        if (order == 0) {
            str_tag = "Data.ForceConstants.HARMONIC";
        } else {
            str_tag = "Data.ForceConstants.ANHARM" + std::to_string(order + 2);
        }

        boost::optional<ptree &> child_ = pt.get_child_optional(str_tag);

        if (!child_) {
            std::string str_tmp = str_tag + " flag not found in the XML file";
            exit(EXIT_FAILURE);
        }

        BOOST_FOREACH (const ptree::value_type &child_, pt.get_child(str_tag)) {
            const ptree &child = child_.second;

            fcs_val = boost::lexical_cast<double>(child.data());

            ivec_with_cell.clear();

            for (i = 0; i < order + 2; ++i) {
                str_attr = "<xmlattr>.pair" + std::to_string(i + 1);
                str_pairs = child.get<std::string>(str_attr);

                ss.str("");
                ss.clear();
                ss << str_pairs;

                if (i == 0) {
                    ss >> atmn >> xyz;
                    ivec_tmp.index = 3 * (atmn - 1) + xyz - 1;
                    ivec_tmp.cell_s = 0;
                    ivec_tmp.tran = 0; // dummy
                    ivec_with_cell.push_back(ivec_tmp);

                } else {
                    ss >> atmn >> xyz >> cell_s;
                    ivec_tmp.index = 3 * (atmn - 1) + xyz - 1;
                    ivec_tmp.cell_s = cell_s - 1;
                    ivec_tmp.tran = 0; // dummy
                    ivec_with_cell.push_back(ivec_tmp);
                }
            }

            force_constant_with_cell[order].emplace_back(fcs_val, ivec_with_cell);
        }
    }
}

void write_fcs_to_file(const std::string fname_fc, const int order, const StructureProperty &structure,
                       double ***x_image, int **map_p2s, const std::vector<FcsArrayWithCell> &fc_in)
{

    std::ofstream ofs;
    ofs.open(fname_fc, std::ios::out);

    const auto nelems = order + 2;

    ofs << '#';
    for (auto i = 0; i < nelems; ++i) {
        ofs << std::setw(6) << "atom" + std::to_string(i + 1);
        ofs << std::setw(5) << "xyz" + std::to_string(i + 1);
    }
    ofs << std::setw(20) << "IFC" + std::to_string(nelems) + " (Ry/bohr^" + std::to_string(nelems) + ")";
    ofs << std::setw(20) << "distances (bohr)";
    ofs << '\n';

    for (const auto &it: fc_in) {
        ofs << std::setw(7) << map_p2s[it.pairs[0].index / 3][0] + 1;
        ofs << std::setw(5) << it.pairs[0].index % 3 + 1;

        for (auto i = 1; i < nelems; ++i) {
            ofs << std::setw(6) << it.pairs[i].index / 3 + 1;
            ofs << std::setw(5) << it.pairs[i].index % 3 + 1;
        }

        ofs << std::setw(20) << it.fcs_val;

        double distance;
        double xdiff[3];

        for (auto i = 1; i < nelems; ++i) {
            for (auto j = 0; j < 3; ++j) {
                xdiff[j] = x_image[it.pairs[i].cell_s][it.pairs[i].index / 3][j] -
                           x_image[0][map_p2s[it.pairs[0].index / 3][0]][j];
            }
            distance = std::sqrt(xdiff[0] * xdiff[0] + xdiff[1] * xdiff[1] + xdiff[2] * xdiff[2]);
            ofs << std::setw(10) << distance;
        }
        ofs << '\n';
    }

    ofs.close();
}

void frac2cart(double **xf, const int nat, const double lattice_vector[3][3])
{
    // x_cartesian = A x_fractional

    double *x_tmp;
    allocate(x_tmp, 3);

    for (size_t i = 0; i < nat; ++i) {

        rotvec(x_tmp, xf[i], lattice_vector);

        for (auto j = 0; j < 3; ++j) {
            xf[i][j] = x_tmp[j];
        }
    }
    deallocate(x_tmp);
}
