/*
 hdf5_parser.h

 Copyright (c) 2023 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/


#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

#ifndef H5_USE_EIGEN
#define H5_USE_EIGEN 1
#endif
#include <highfive/H5Easy.hpp>

#include "constants.h"
#include "units.h"

// Read the "unit" attribute of a dataset; returns an empty string when the
// attribute is absent (files written before the attribute existed).
inline auto get_unit_attribute_h5(const H5Easy::File &file, const std::string &path) -> std::string
{
    if (!file.getDataSet(path).hasAttribute("unit")) return "";
    return H5Easy::loadAttribute<std::string>(file, path, "unit");
}

// Multiply-by factor that converts a length-type dataset ("bohr"/"angstrom")
// into bohr. An absent attribute is interpreted as bohr (legacy files).
inline auto h5_length_factor_to_bohr(const H5Easy::File &file, const std::string &path,
                                     std::string *unit_detected = nullptr) -> double
{
    const auto unit = get_unit_attribute_h5(file, path);
    if (unit_detected) *unit_detected = unit;
    if (unit.empty() || unit == "bohr") return 1.0;
    if (unit == "angstrom") return 1.0 / Bohr_in_Angstrom;
    std::cout << "Error: unsupported unit \"" << unit << "\" of dataset " << path << " in file " << file.getName()
              << ". Supported units are \"bohr\" and \"angstrom\".\n";
    exit(1);
}

// Multiply-by factor that converts force constant values of the given order
// ("Ry/bohr^m" / "eV/angstrom^m" with m = order + 2) into Ry/bohr^m.
// An absent attribute is interpreted as Ry/bohr^m (legacy files).
inline auto h5_fc_factor_to_ry_bohr(const H5Easy::File &file, const std::string &path, const int order,
                                    std::string *unit_detected = nullptr) -> double
{
    const auto unit = get_unit_attribute_h5(file, path);
    if (unit_detected) *unit_detected = unit;
    const auto str_m = std::to_string(order + 2);
    if (unit.empty() || unit == "Ry/bohr^" + str_m) return 1.0;
    if (unit == "eV/angstrom^" + str_m) return 1.0 / units::fc_ry_bohr_to_ev_ang(order + 2);
    std::cout << "Error: unsupported unit \"" << unit << "\" of dataset " << path << " in file " << file.getName()
              << ". Supported units are \"Ry/bohr^" << str_m << "\" and \"eV/angstrom^" << str_m << "\".\n";
    exit(1);
}

// Resolve a user-facing cell-type string ("prim"/"super"/...) to the HDF5 group
// name ("PrimitiveCell"/"SuperCell"). Shared by the get_*_from_h5 helpers below.
inline auto resolve_h5_cell_name(const std::string &celltype) -> std::string
{
    if (celltype == "prim" || celltype == "primitive" || celltype == "PrimitiveCell") {
        return "PrimitiveCell";
    }
    if (celltype == "super" || celltype == "supercell" || celltype == "SuperCell") {
        return "SuperCell";
    }
    std::cout << "resolve_h5_cell_name: Invalid celltype " << celltype << '\n';
    exit(1);
}

inline auto get_structures_from_h5(const H5Easy::File &file, const std::string &celltype, Eigen::Matrix3d &lavec,
                                   Eigen::MatrixXd &x_fractional, std::vector<int> &kind_index,
                                   std::vector<std::string> &element_names,
                                   std::string *length_unit_detected = nullptr) -> void
{
    using namespace H5Easy;
    const std::string search_cell = resolve_h5_cell_name(celltype);

    lavec = load<Eigen::Matrix3d>(file, "/" + search_cell + "/lattice_vector");
    lavec.transposeInPlace();
    // Convert the lattice into the internal unit (bohr) if the file declares
    // another unit; files without the attribute are assumed to be in bohr.
    lavec *= h5_length_factor_to_bohr(file, "/" + search_cell + "/lattice_vector", length_unit_detected);
    x_fractional = load<Eigen::MatrixXd>(file, "/" + search_cell + "/fractional_coordinate");
    kind_index = load<std::vector<int>>(file, "/" + search_cell + "/atomic_kinds");
    element_names = load<std::vector<std::string>>(file, "/" + search_cell + "/elements");
}

inline auto get_mapping_table_from_h5(const H5Easy::File &file, const std::string &celltype,
                                      std::vector<std::vector<int>> &mapping_table) -> void
{
    const std::string search_cell = resolve_h5_cell_name(celltype);
    mapping_table = H5Easy::load<std::vector<std::vector<int>>>(file, "/" + search_cell + "/mapping_table");
}

inline auto get_magnetism_from_h5(const H5Easy::File &file, const std::string &celltype, int &lspin,
                                  std::vector<std::vector<double>> &magmom, int &noncollinear,
                                  int &time_reversal_symmetry) -> void
{
    using namespace H5Easy;
    const std::string search_cell = resolve_h5_cell_name(celltype);

    lspin = load<int>(file, "/" + search_cell + "/spin_polarized");
    if (lspin > 0) {
        magmom = load<std::vector<std::vector<double>>>(file, "/" + search_cell + "/magnetic_moments");
        noncollinear = loadAttribute<int>(file, "/" + search_cell + "/magnetic_moments", "noncollinear");
        time_reversal_symmetry =
            loadAttribute<int>(file, "/" + search_cell + "/magnetic_moments", "time_reversal_symmetry");
    } else {
        noncollinear = 0;
        time_reversal_symmetry = 1;
    }
}

inline auto get_force_constants_from_h5(const H5Easy::File &file, const int order, Eigen::MatrixXi &atom_indices,
                                        Eigen::MatrixXi &atom_indices_super, Eigen::MatrixXi &coord_indices,
                                        Eigen::MatrixXd &shift_vectors, Eigen::ArrayXd &fcs_values,
                                        std::string *shift_unit_detected = nullptr,
                                        std::string *fc_unit_detected = nullptr) -> void
{
    using namespace H5Easy;
    const std::string str_ordername = "Order" + std::to_string(order + 2);
    atom_indices = load<Eigen::MatrixXi>(file, "/ForceConstants/" + str_ordername + "/atom_indices");
    atom_indices_super = load<Eigen::MatrixXi>(file, "/ForceConstants/" + str_ordername + "/atom_indices_supercell");
    coord_indices = load<Eigen::MatrixXi>(file, "/ForceConstants/" + str_ordername + "/coord_indices");
    shift_vectors = load<Eigen::MatrixXd>(file, "/ForceConstants/" + str_ordername + "/shift_vectors");
    fcs_values = load<Eigen::ArrayXd>(file, "/ForceConstants/" + str_ordername + "/force_constant_values");

    // Convert into the internal units (bohr, Ry/bohr^m) if the file declares
    // other units; datasets without the attribute are assumed to already be in
    // the internal units (files written before the attributes existed).
    std::string unit_shift, unit_fc;
    const auto factor_shift =
        h5_length_factor_to_bohr(file, "/ForceConstants/" + str_ordername + "/shift_vectors", &unit_shift);
    const auto factor_fc =
        h5_fc_factor_to_ry_bohr(file, "/ForceConstants/" + str_ordername + "/force_constant_values", order, &unit_fc);
    // Reject partially annotated files when a non-default unit is declared:
    // one dataset in a non-default unit while its sibling carries no unit
    // attribute is ambiguous, most likely a hand-edited or third-party file.
    if ((unit_shift.empty() && factor_fc != 1.0) || (unit_fc.empty() && factor_shift != 1.0)) {
        std::cout << "Error: /ForceConstants/" << str_ordername << " in file " << file.getName()
                  << " declares a non-default unit on one dataset while its sibling"
                  << " (shift_vectors / force_constant_values) has no unit attribute.\n";
        exit(1);
    }
    if (factor_shift != 1.0) shift_vectors *= factor_shift;
    if (factor_fc != 1.0) fcs_values *= factor_fc;
    if (shift_unit_detected) *shift_unit_detected = unit_shift;
    if (fc_unit_detected) *fc_unit_detected = unit_fc;
}
