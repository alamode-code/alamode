/*
 fc_virtual.cpp

 Copyright (c) 2018 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "fc_virtual.h"
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/version.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include "memory.h"
#include "xml_parser.h"

using namespace std;

int main(int argc, char *argv[])
{
    std::string file_xml1, file_xml2;
    double alpha = 1.0;
    int maxorder;

    if (argc == 1) {
        cout << " FC_Virtual -- virtual crystal approximation of force constants" << endl;
        cout << " First  FCSXML file (A) :  ";
        cin >> file_xml1;
        cout << " Second FCSXML file (B) :  ";
        cin >> file_xml2;
        cout << " Mixing alpha [alpha * A + (1 - alpha) * B] : ";
        cin >> alpha;
        cout << " Maxorder (2: harmonic, 3: cubic, 4: quartic, ...) : ";
        cin >> maxorder;

    } else if (argc == 5) {

        file_xml1 = argv[1];
        file_xml2 = argv[2];
        alpha = boost::lexical_cast<double>(argv[3]);
        maxorder = boost::lexical_cast<int>(argv[4]);

    } else {
        std::cout << "Usage: " << std::endl;
        std::cout << "(command line) > fc_virtual FCSXML1 FCSXML2 mixalpha maxorder " << std::endl;
        std::cout << "(interactive) > fc_virtual " << std::endl;
        std::cout << std::endl;
    }

    if (alpha < 0.0 || alpha > 1.0) {
        std::cout << "alpha must be 0 <= alpha <= 1" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (maxorder <= 1) {
        std::cout << "Maxorder should be larger than 1. " << std::endl;
        exit(EXIT_FAILURE);
    }

    maxorder = maxorder - 1;

    std::vector<FcsArrayWithCell> *fc_orig1, *fc_orig2, *fc_new;
    StructureProperty structure1, structure2, structure_new;

    allocate(fc_orig1, maxorder);
    allocate(fc_orig2, maxorder);
    allocate(fc_new, maxorder);

    load_fcs_xml(file_xml1, maxorder, structure1, fc_orig1);
    load_fcs_xml(file_xml2, maxorder, structure2, fc_orig2);

    mix_structure(structure1, structure2, structure_new, alpha);
    mix_forceconstant(fc_orig1, fc_orig2, fc_new, alpha, maxorder);

    deallocate(fc_orig1);
    deallocate(fc_orig2);

    write_new_xml("VCA.xml", file_xml1, file_xml2, maxorder, alpha, structure_new, fc_new);

    std::cout << std::endl;
    std::cout << " A new FCSXML is generated as VCA.xml." << std::endl;

    deallocate(fc_new);
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

void mix_structure(const StructureProperty &Structure1, const StructureProperty &Structure2,
                   StructureProperty &Structure_out, const double alpha)
{
    // First, check the consistency of the two structures.

    if (Structure1.nat != Structure2.nat || Structure1.nspecies != Structure2.nspecies ||
        Structure1.ntran != Structure2.ntran || Structure1.atoms.size() != Structure2.atoms.size())
    {
        std::cout << " Atomic structures of two XML files are different." << std::endl;
        exit(EXIT_FAILURE);
    }

    for (auto i = 0; i < Structure1.nspecies; ++i) {
        if (Structure1.kd_symbol[i] != Structure2.kd_symbol[i]) {
            std::cout << " Warning: the atomic symbols don't match." << std::endl;
        }
    }
    for (auto i = 0; i < Structure1.nat; ++i) {
        if (Structure1.atoms[i].kind != Structure2.atoms[i].kind) {
            std::cout << " Warning: the kind index doesn't match." << std::endl;
        }
    }
    for (auto i = 0; i < 3; ++i) {
        if (Structure1.is_periodic[i] != Structure2.is_periodic[i]) {
            std::cout << " Warning: the periodicity doesn't match." << std::endl;
        }
    }

    for (auto i = 0; i < Structure1.nat; ++i) {
        if (Structure1.atoms[i].atom != Structure2.atoms[i].atom ||
            Structure1.atoms[i].tran != Structure2.atoms[i].tran)
        {
            std::cout << " The mapping information doesn't match." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Copy common variables

    Structure_out.nat = Structure1.nat;
    Structure_out.nspecies = Structure1.nspecies;
    Structure_out.ntran = Structure1.ntran;
    Structure_out.natmin = Structure1.natmin;
    for (auto i = 0; i < 3; ++i)
        Structure_out.is_periodic[i] = Structure1.is_periodic[i];

    for (const auto &it: Structure1.kd_symbol) {
        Structure_out.kd_symbol.push_back(it);
    }

    // Mix lattice constants

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            Structure_out.lattice_vector[i][j] =
                alpha * Structure1.lattice_vector[i][j] + (1.0 - alpha) * Structure2.lattice_vector[i][j];
        }
    }

    // Mix fractional coordinate

    AtomProperty atoms;

    for (auto i = 0; i < Structure1.nat; ++i) {

        atoms.kind = Structure1.atoms[i].kind;
        atoms.atom = Structure1.atoms[i].atom;
        atoms.tran = Structure1.atoms[i].tran;

        atoms.x = alpha * Structure1.atoms[i].x + (1.0 - alpha) * Structure2.atoms[i].x;
        atoms.y = alpha * Structure1.atoms[i].y + (1.0 - alpha) * Structure2.atoms[i].y;
        atoms.z = alpha * Structure1.atoms[i].z + (1.0 - alpha) * Structure2.atoms[i].z;

        Structure_out.atoms.push_back(atoms);
    }
}


void mix_forceconstant(std::vector<FcsArrayWithCell> *fc_orig1, std::vector<FcsArrayWithCell> *fc_orig2,
                       std::vector<FcsArrayWithCell> *fc_new, const double alpha, const int maxorder)
{
    // Mix force constants with the given mixing fraction "alpha"

    for (auto order = 0; order < maxorder; ++order) {

        std::vector<FcsArrayWithCell> fc_tmp;

        for (const auto &it: fc_orig1[order]) {
            fc_tmp.emplace_back(alpha * it.fcs_val, it.pairs);
        }

        for (const auto &it: fc_orig2[order]) {
            fc_tmp.emplace_back((1.0 - alpha) * it.fcs_val, it.pairs);
        }

        std::sort(fc_tmp.begin(), fc_tmp.end());

        FcsArrayWithCell fc_now = fc_tmp[0];
        double fcs_val = fc_now.fcs_val;

        for (auto i = 1; i < fc_tmp.size(); ++i) {
            if (fc_tmp[i] == fc_now) {
                fcs_val += fc_tmp[i].fcs_val;
            } else {
                fc_new[order].emplace_back(fcs_val, fc_now.pairs);
                fc_now = fc_tmp[i];
                fcs_val = fc_now.fcs_val;
            }
        }

        fc_new[order].emplace_back(fcs_val, fc_now.pairs);
    }
}


void write_new_xml(const std::string file_xml, const std::string file_xml1, const std::string file_xml2,
                   const int maxorder, const double alpha, const StructureProperty &structure,
                   std::vector<FcsArrayWithCell> *fcs)
{
    int i, j, k;
    using boost::property_tree::ptree;
    ptree pt;

    pt.put("Data.VCA.Original1", file_xml1);
    pt.put("Data.VCA.Original2", file_xml2);
    pt.put("Data.VCA.Mixalpha", alpha);
    pt.put("Data.Structure.NumberOfAtoms", structure.nat);
    pt.put("Data.Structure.NumberOfElements", structure.nspecies);

    for (i = 0; i < structure.nspecies; ++i) {
        ptree &child = pt.add("Data.Structure.AtomicElements.element", structure.kd_symbol[i]);
        child.put("<xmlattr>.number", i + 1);
    }

    std::string str_pos[3];
    for (i = 0; i < 3; ++i) {
        str_pos[i].clear();
        for (j = 0; j < 3; ++j) {
            str_pos[i] += " " + double2string(structure.lattice_vector[j][i]);
        }
    }
    pt.put("Data.Structure.LatticeVector", "");
    pt.put("Data.Structure.LatticeVector.a1", str_pos[0]);
    pt.put("Data.Structure.LatticeVector.a2", str_pos[1]);
    pt.put("Data.Structure.LatticeVector.a3", str_pos[2]);

    std::stringstream ss;
    ss << structure.is_periodic[0] << " " << structure.is_periodic[1] << " " << structure.is_periodic[2];
    pt.put("Data.Structure.Periodicity", ss.str());

    pt.put("Data.Structure.Position", "");
    std::string str_tmp;

    for (i = 0; i < structure.nat; ++i) {
        str_tmp.clear();
        str_tmp = " " + double2string(structure.atoms[i].x) + " " + double2string(structure.atoms[i].y) + " " +
                  double2string(structure.atoms[i].z);
        ptree &child = pt.add("Data.Structure.Position.pos", str_tmp);
        child.put("<xmlattr>.index", i + 1);
        child.put("<xmlattr>.element", structure.kd_symbol[structure.atoms[i].kind]);
    }

    int **map_p2s;
    allocate(map_p2s, structure.natmin, structure.ntran);
    for (i = 0; i < structure.nat; ++i) {
        map_p2s[structure.atoms[i].atom][structure.atoms[i].tran] = i;
    }

    pt.put("Data.Symmetry.NumberOfTranslations", structure.ntran);
    for (i = 0; i < structure.ntran; ++i) {
        for (j = 0; j < structure.natmin; ++j) {
            ptree &child = pt.add("Data.Symmetry.Translations.map", map_p2s[j][i] + 1);
            child.put("<xmlattr>.tran", i + 1);
            child.put("<xmlattr>.atom", j + 1);
        }
    }
    deallocate(map_p2s);

    std::string elementname;

    for (auto order = 0; order < maxorder; ++order) {

        if (order == 0) {
            elementname = "Data.ForceConstants.HARMONIC.FC2";
        } else {
            elementname = "Data.ForceConstants.ANHARM" + std::to_string(order + 2) + ".FC" + std::to_string(order + 2);
        }

        for (const auto &it: fcs[order]) {

            ptree &child = pt.add(elementname, double2string(it.fcs_val));
            child.put("<xmlattr>.pair1",
                      std::to_string(it.pairs[0].index / 3 + 1) + " " + std::to_string(it.pairs[0].index % 3 + 1));

            for (k = 1; k < order + 2; ++k) {
                child.put("<xmlattr>.pair" + std::to_string(k + 1),
                          std::to_string(it.pairs[k].index / 3 + 1) + " " + std::to_string(it.pairs[k].index % 3 + 1) +
                              " " + std::to_string(it.pairs[k].cell_s + 1));
            }
        }
    }

    using namespace boost::property_tree::xml_parser;
    const int indent = 2;

#if BOOST_VERSION >= 105600
    write_xml(file_xml,
              pt,
              std::locale(),
              xml_writer_make_settings<ptree::key_type>(' ', indent, widen<std::string>("utf-8")));
#else
    write_xml(file_xml, pt, std::locale(), xml_writer_make_settings(' ', indent, widen<char>("utf-8")));
#endif
}


std::string double2string(const double d, const int nprec)
{
    std::string rt;
    std::stringstream ss;

    ss << std::scientific << std::setprecision(nprec) << d;
    ss >> rt;
    return rt;
}
