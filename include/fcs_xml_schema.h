/*
 fcs_xml_schema.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

// Writers of the ALAMODE force-constant XML sections (Data.Structure,
// Data.Symmetry, and the Cartesian force-constant blocks), shared between
// alm and anphon so both codes emit byte-compatible layouts. Only plain
// Eigen/std types appear in the interfaces; adapting from each code's own
// structure classes is the caller's job, as are the code-specific sections
// (Data.ALM_version / Data.Optimize / *Unique blocks on the alm side,
// Data.ANPHON_version / Data.Description on the anphon side).

#pragma once

#include <Eigen/Core>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/version.hpp>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace fcsxml
{

inline auto double2string(const double d, const int nprec = 15) -> std::string
{
    std::string rt;
    std::stringstream ss;

    ss << std::scientific << std::setprecision(nprec) << d;
    ss >> rt;
    return rt;
}

// Data.Structure.* section. lattice_vector(i, j) is the i-th Cartesian
// component of the j-th lattice vector in bohr; x_fractional rows are atoms;
// atomic_kinds are 0-based indices into element_names. The alm-only
// Periodicity element is emitted between LatticeVector and Position when
// periodicity is non-empty.
inline auto add_structure_group_xml(boost::property_tree::ptree &pt, const Eigen::Matrix3d &lattice_vector,
                                    const Eigen::MatrixXd &x_fractional, const std::vector<int> &atomic_kinds,
                                    const std::vector<std::string> &element_names,
                                    const std::string &periodicity = "") -> void
{
    const auto nat = static_cast<size_t>(x_fractional.rows());

    pt.put("Data.Structure.NumberOfAtoms", nat);
    pt.put("Data.Structure.NumberOfElements", element_names.size());

    for (size_t i = 0; i < element_names.size(); ++i) {
        auto &child = pt.add("Data.Structure.AtomicElements.element", element_names[i]);
        child.put("<xmlattr>.number", i + 1);
    }

    std::string str_pos[3];
    for (auto i = 0; i < 3; ++i) {
        str_pos[i].clear();
        for (auto j = 0; j < 3; ++j) {
            str_pos[i] += " " + double2string(lattice_vector(j, i));
        }
    }
    pt.put("Data.Structure.LatticeVector", "");
    pt.put("Data.Structure.LatticeVector.a1", str_pos[0]);
    pt.put("Data.Structure.LatticeVector.a2", str_pos[1]);
    pt.put("Data.Structure.LatticeVector.a3", str_pos[2]);

    if (!periodicity.empty()) {
        pt.put("Data.Structure.Periodicity", periodicity);
    }

    pt.put("Data.Structure.Position", "");
    std::string str_tmp;

    for (size_t i = 0; i < nat; ++i) {
        str_tmp.clear();
        for (auto j = 0; j < 3; ++j) str_tmp += " " + double2string(x_fractional(i, j));
        auto &child = pt.add("Data.Structure.Position.pos", str_tmp);
        child.put("<xmlattr>.index", i + 1);
        child.put("<xmlattr>.element", element_names[atomic_kinds[i]]);
    }
}

// Data.Symmetry.* section. mapping_table[atom_prim][tran] is the 0-based
// supercell atom obtained by translating primitive atom atom_prim.
template <class MappingTable>
inline auto add_symmetry_group_xml(boost::property_tree::ptree &pt, const MappingTable &mapping_table) -> void
{
    const auto natmin = mapping_table.size();
    const auto ntran = mapping_table[0].size();

    pt.put("Data.Symmetry.NumberOfTranslations", ntran);
    for (size_t i = 0; i < ntran; ++i) {
        for (size_t j = 0; j < natmin; ++j) {
            auto &child = pt.add("Data.Symmetry.Translations.map", mapping_table[j][i] + 1);
            child.put("<xmlattr>.tran", i + 1);
            child.put("<xmlattr>.atom", j + 1);
        }
    }
}

// One Cartesian force-constant entry of order n. Written as
// pair1 = "(atom1_prim+1) (coords[0]+1)" and, for k = 2..n,
// pairk = "(atoms_super[k-2]+1) (coords[k-1]+1) (cells[k-2]+1)".
struct FcCartesianRowXml
{
    double value;
    int atom1_prim;               // 0-based primitive-cell atom of the first leg
    std::vector<int> atoms_super; // n-1 entries, 0-based supercell atoms
    std::vector<int> coords;      // n entries, 0-based Cartesian components
    std::vector<int> cells;       // n-1 entries, 0-based multiplicity cell indices
};

// Data.ForceConstants.<blockname>.FC<n> entries (blockname = "HARMONIC",
// "ANHARM3", ...). The caller is responsible for the initial
// pt.put("Data.ForceConstants", "") and for the order of block additions.
inline auto add_fc_cartesian_group_xml(boost::property_tree::ptree &pt, const std::string &blockname, const int order_n,
                                       const std::vector<FcCartesianRowXml> &rows) -> void
{
    const auto elementname = "Data.ForceConstants." + blockname + ".FC" + std::to_string(order_n);

    for (const auto &row: rows) {
        auto &child = pt.add(elementname, double2string(row.value));

        child.put("<xmlattr>.pair1", std::to_string(row.atom1_prim + 1) + " " + std::to_string(row.coords[0] + 1));

        for (auto k = 1; k < order_n; ++k) {
            child.put("<xmlattr>.pair" + std::to_string(k + 1),
                      std::to_string(row.atoms_super[k - 1] + 1) + " " + std::to_string(row.coords[k] + 1) + " " +
                          std::to_string(row.cells[k - 1] + 1));
        }
    }
}

inline auto write_fcs_xml_file(const std::string &filename, const boost::property_tree::ptree &pt,
                               const int indent = 2) -> void
{
    using namespace boost::property_tree::xml_parser;
    using boost::property_tree::ptree;

#if BOOST_VERSION >= 105600
    write_xml(filename,
              pt,
              std::locale(),
              xml_writer_make_settings<ptree::key_type>(' ', indent, widen<std::string>("utf-8")));
#else
    write_xml(filename, pt, std::locale(), xml_writer_make_settings(' ', indent, widen<char>("utf-8")));
#endif
}

} // namespace fcsxml
