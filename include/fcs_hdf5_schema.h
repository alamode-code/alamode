/*
 fcs_hdf5_schema.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

// Writers of the ALAMODE force-constant HDF5 schema (the /PrimitiveCell,
// /SuperCell, and /ForceConstants/Order{n} groups), shared between alm and
// anphon so both codes emit byte-compatible group layouts. Only plain
// Eigen/std types appear in the interfaces; adapting from each code's own
// structure classes is the caller's job. The corresponding readers live in
// hdf5_parser.h.

#pragma once

#include <string>
#include <vector>

#ifndef H5_USE_EIGEN
#define H5_USE_EIGEN 1
#endif
#include <highfive/H5Easy.hpp>

#include "constants.h"
#include "units.h"

// Write one structure group ("PrimitiveCell" or "SuperCell").
// lattice_vector holds the basis vectors a1,a2,a3 as columns, in bohr;
// atomic_kinds are 0-based; magnetic datasets are written only when
// spin_polarized is nonzero. The optional masses dataset (amu) is an
// anphon extension absent from alm-written files; readers must tolerate
// its absence.
inline auto write_cell_group_h5(H5Easy::File &file, const std::string &celltype,
                                const Eigen::Matrix3d &lattice_vector,
                                const Eigen::MatrixXd &x_fractional,
                                const std::vector<int> &atomic_kinds,
                                const std::vector<std::string> &element_names,
                                const int spin_polarized,
                                const std::vector<std::vector<double>> &magnetic_moments,
                                const int noncollinear,
                                const int time_reversal_symmetry,
                                const size_t number_of_primitive_translations,
                                const std::vector<std::vector<int>> &mapping_table,
                                const units::FcUnitSystem unit_system,
                                const std::vector<double> &masses_amu = {}) -> void
{
    using namespace H5Easy;
    const auto in_ev_ang = unit_system == units::FcUnitSystem::ev_angstrom;
    const std::string unitname(in_ev_ang ? "angstrom" : "bohr");
    const auto factor_length = in_ev_ang ? Bohr_in_Angstrom : 1.0;

    dump(file, "/" + celltype + "/lattice_vector",
         Eigen::Matrix3d(lattice_vector.transpose() * factor_length));
    dumpAttribute(file, "/" + celltype + "/lattice_vector", "unit", unitname);
    dump(file, "/" + celltype + "/number_of_atoms", static_cast<size_t>(x_fractional.rows()));
    dump(file, "/" + celltype + "/number_of_elements", element_names.size());
    dump(file, "/" + celltype + "/fractional_coordinate", x_fractional);
    dump(file, "/" + celltype + "/atomic_kinds", atomic_kinds);
    dump(file, "/" + celltype + "/elements", element_names);
    dump(file, "/" + celltype + "/spin_polarized", spin_polarized);
    if (spin_polarized) {
        dump(file, "/" + celltype + "/magnetic_moments", magnetic_moments);
        dumpAttribute(file, "/" + celltype + "/magnetic_moments", "noncollinear", noncollinear);
        dumpAttribute(file, "/" + celltype + "/magnetic_moments", "time_reversal_symmetry",
                      time_reversal_symmetry);
    }
    dump(file, "/" + celltype + "/number_of_primitive_translations", number_of_primitive_translations);
    dump(file, "/" + celltype + "/mapping_table", mapping_table);
    if (!masses_amu.empty()) {
        dump(file, "/" + celltype + "/masses", masses_amu);
        dumpAttribute(file, "/" + celltype + "/masses", "unit", std::string("amu"));
    }
}

// Write the index/shift/value datasets of /ForceConstants/Order{order+2}.
// Inputs are in the internal units (bohr, Ry/bohr^{order+2}) and are
// converted on the fly when unit_system requests eV/angstrom; every value
// dataset carries a "unit" attribute either way.
inline auto write_fc_order_group_h5(H5Easy::File &file, const int order,
                                    const Eigen::MatrixXi &atom_indices,
                                    const Eigen::MatrixXi &atom_indices_super,
                                    const Eigen::MatrixXi &coord_indices,
                                    Eigen::MatrixXd shift_vectors,
                                    Eigen::ArrayXd fcs_values,
                                    const units::FcUnitSystem unit_system,
                                    const int compression_level) -> void
{
    using namespace H5Easy;
    const auto in_ev_ang = unit_system == units::FcUnitSystem::ev_angstrom;
    if (in_ev_ang) {
        shift_vectors *= Bohr_in_Angstrom;
        fcs_values *= units::fc_ry_bohr_to_ev_ang(order + 2);
    }

    const std::string str_ordername = "Order" + std::to_string(order + 2);
    dump(file, "/ForceConstants/" + str_ordername + "/atom_indices", atom_indices, Compression(compression_level));
    dump(file,
         "/ForceConstants/" + str_ordername + "/atom_indices_supercell",
         atom_indices_super,
         Compression(compression_level));
    dump(file, "/ForceConstants/" + str_ordername + "/coord_indices", coord_indices, Compression(compression_level));
    dump(file, "/ForceConstants/" + str_ordername + "/shift_vectors", shift_vectors, Compression(compression_level));
    std::string unitname = in_ev_ang ? "angstrom" : "bohr";
    const std::string basisname = "Cartesian";
    dumpAttribute(file, "/ForceConstants/" + str_ordername + "/shift_vectors", "unit", unitname);
    dumpAttribute(file, "/ForceConstants/" + str_ordername + "/shift_vectors", "basis", basisname);
    dump(file,
         "/ForceConstants/" + str_ordername + "/force_constant_values",
         fcs_values,
         Compression(compression_level));
    unitname = (in_ev_ang ? "eV/angstrom^" : "Ry/bohr^") + std::to_string(order + 2);
    dumpAttribute(file, "/ForceConstants/" + str_ordername + "/force_constant_values", "unit", unitname);
}
