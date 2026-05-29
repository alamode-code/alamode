/*
 hdf5_parser.h

 Copyright (c) 2023 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/


#pragma once

#include <iostream>
#include <string>

#ifndef H5_USE_EIGEN
#define H5_USE_EIGEN 1
#endif
#include <highfive/H5Easy.hpp>

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
                                   std::vector<std::string> &element_names) -> void
{
    using namespace H5Easy;
    const std::string search_cell = resolve_h5_cell_name(celltype);

    lavec = load<Eigen::Matrix3d>(file, "/" + search_cell + "/lattice_vector");
    lavec.transposeInPlace();
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
                                        Eigen::MatrixXd &shift_vectors, Eigen::ArrayXd &fcs_values) -> void
{
    using namespace H5Easy;
    const std::string str_ordername = "Order" + std::to_string(order + 2);
    atom_indices = load<Eigen::MatrixXi>(file, "/ForceConstants/" + str_ordername + "/atom_indices");
    atom_indices_super = load<Eigen::MatrixXi>(file, "/ForceConstants/" + str_ordername + "/atom_indices_supercell");
    coord_indices = load<Eigen::MatrixXi>(file, "/ForceConstants/" + str_ordername + "/coord_indices");
    shift_vectors = load<Eigen::MatrixXd>(file, "/ForceConstants/" + str_ordername + "/shift_vectors");
    fcs_values = load<Eigen::ArrayXd>(file, "/ForceConstants/" + str_ordername + "/force_constant_values");
}
