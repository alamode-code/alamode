/*
 symmetry.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>
#include "system.h"
#include "timer.h"

namespace ALM_NS
{
class SymmetryOperation
{
public:
    Eigen::Matrix3i rotation;      // in lattice basis
    Eigen::Vector3d tran;          // in lattice basis
    Eigen::Matrix3d rotation_cart; // in Cartesian basis
    bool compatible_with_lattice;
    bool compatible_with_cartesian;
    bool is_translation;

    SymmetryOperation();

    SymmetryOperation(const Eigen::Matrix3i &rot_in, const Eigen::Vector3d &tran_in, const Eigen::Matrix3d &rot_cart_in,
                      const bool compatibility_lat, const bool compatibility_cart, const bool is_trans_in)
    {
        rotation = rot_in;
        rotation_cart = rot_cart_in;
        tran = tran_in;
        compatible_with_lattice = compatibility_lat;
        compatible_with_cartesian = compatibility_cart;
        is_translation = is_trans_in;
    }

    // Operator definition to sort
    bool operator<(const SymmetryOperation &a) const
    {
        std::vector<double> v1, v2;
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                v1.push_back(static_cast<double>(rotation(i, j)));
                v2.push_back(static_cast<double>(a.rotation(i, j)));
            }
        }
        for (auto i = 0; i < 3; ++i) {
            if (tran[i] < 0.0) {
                v1.push_back(1.0 + tran[i]);
            } else {
                v1.push_back(tran[i]);
            }
            if (a.tran[i] < 0.0) {
                v2.push_back(1.0 + a.tran[i]);
            } else {
                v2.push_back(a.tran[i]);
            }
        }
        return std::lexicographical_compare(v1.begin(), v1.end(), v2.begin(), v2.end());
    }
};

class RotationMatrix
{
public:
    Eigen::Matrix3i mat;

    RotationMatrix();

    RotationMatrix(const int rot[3][3])
    {
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                mat(i, j) = rot[i][j];
            }
        }
    }

    RotationMatrix(const Eigen::Matrix3i &rot_in)
    {
        mat = rot_in;
    }
};

class Maps
{
public:
    int atom_num;
    int tran_num;
};

class PrimitiveGroup
{
public:
    std::map<int, int> map_index_p2s;
    Eigen::Vector3i shift_vector;

    PrimitiveGroup();

    PrimitiveGroup(const std::map<int, int> &map_index_p2s_in, const Eigen::Vector3i &shift_vector_in)
    {
        map_index_p2s = map_index_p2s_in;
        shift_vector = shift_vector_in;
    }
};

class Symmetry
{
public:
    Symmetry();

    ~Symmetry();

    void init(const std::unique_ptr<System> &system, const int verbosity, std::unique_ptr<Timer> &timer);

    [[nodiscard]] double get_tolerance() const;

    void set_tolerance(const double);

    [[nodiscard]] int get_print_symmetry() const;

    void set_print_symmetry(const int);

    [[nodiscard]] const std::vector<Maps> &get_map_super_to_trueprim() const;

    [[nodiscard]] const std::vector<Maps> &get_map_prim_to_trueprim() const;

    [[nodiscard]] const std::vector<std::vector<int>> &get_map_trueprim_to_super() const;

    [[nodiscard]] const std::vector<std::vector<int>> &get_map_trueprim_to_prim() const;

    [[nodiscard]] const std::vector<SymmetryOperation> &get_symmetry_data(const std::string &cell = "super") const;

    [[nodiscard]] const std::vector<std::vector<int>> &get_map_sym() const;

    [[nodiscard]] const std::vector<int> &get_symnum_tran(const std::string &cell = "super") const;

    [[nodiscard]] size_t get_nsym(const std::string &cell = "super") const;

    [[nodiscard]] size_t get_ntran(const std::string &cell = "super") const;

    [[nodiscard]] size_t get_nat_trueprim() const;

private:
    size_t nsym_super, ntran_super;
    size_t nsym_prim, ntran_prim;
    size_t nat_trueprim;
    // nat_trueprim is the number of atoms included in a true primitive cell.
    // This value can be different from primcell.number_of_atoms because
    // the latter value is calculated from the PRIMCELL value given by users,
    // which does not necessary reduces the inputcell to a true primitive cell.
    // When nat_trueprim != primcell.number_of_atoms, ntran_prim will be larger
    // than 1.

    std::vector<std::vector<int>> map_fullsymmetry_super; // [nat_base, nsym_super]
    std::vector<std::vector<int>> map_fullsymmetry_prim;  // [nat_base, nsym_super]

    std::vector<std::vector<int>> map_trueprim_to_super; // [nat_trueprim, ntran_super]
    std::vector<Maps> map_super_to_trueprim;             // [nat_super]
    std::vector<std::vector<int>> map_trueprim_to_prim;  // [nat_trueprim, ntran_prim]
    std::vector<Maps> map_prim_to_trueprim;              // [nat_prim]
    std::vector<SymmetryOperation> symmetry_data_super;  // [nsym_super]
    std::vector<SymmetryOperation> symmetry_data_prim;   // [nsym_prim]
    std::vector<int> symnum_tran_super;                  // [ntran_super]
    std::vector<int> symnum_tran_prim;                   // [ntran_prim]
    std::vector<PrimitiveGroup> atomgroup_super;

    double tolerance;
    int printsymmetry;

    void set_default_variables();

    void deallocate_variables();

    void setup_symmetry_operation(const Cell &pcell, const Spin &spin_prim,
                                  const std::vector<std::vector<unsigned int>> &atomtype_prim, const Cell &scell,
                                  const Spin &spin_super, const std::vector<std::vector<unsigned int>> &atomtype_super,
                                  const int is_periodic[3], const int verbosity);

    void gen_mapping_information(const Cell &scell, const std::vector<std::vector<unsigned int>> &atomtype_group_super,
                                 const std::vector<SymmetryOperation> &symm_super, const Cell &pcell,
                                 const std::vector<std::vector<unsigned int>> &atomtype_group_prim,
                                 const std::vector<SymmetryOperation> &symm_prim);

    void findsym_alm(const Cell &, const std::vector<std::vector<unsigned int>> &, const Spin &,
                     std::vector<SymmetryOperation> &symm_out) const;

    int findsym_spglib(const Cell &, const std::vector<std::vector<unsigned int>> &, const Spin &, std::string &,
                       std::vector<SymmetryOperation> &symm_out) const;

    [[nodiscard]] bool is_translation(const int[3][3]) const;

    [[nodiscard]] bool is_translation(const Eigen::Matrix3i &rot) const;

    [[nodiscard]] bool is_proper(const Eigen::Matrix3d &rot) const;

    void symop_in_cart(Eigen::Matrix3d &rot_cart, const Eigen::Matrix3i &rot_lattice, const Eigen::Matrix3d &lavec,
                       const Eigen::Matrix3d &rlavec) const;

    void print_symminfo_stdout() const;

    template <typename T>
    bool is_compatible(const T[3][3], double tolerance_zero = 1.0e-5) const;

    template <typename T>
    bool is_compatible(const Eigen::MatrixBase<T> &mat, double tolerance_zero = 1.0e-5) const;

    void find_lattice_symmetry(const Eigen::Matrix3d &aa, std::vector<RotationMatrix> &) const;

    void find_crystal_symmetry(const Cell &, const std::vector<std::vector<unsigned int>> &, const Spin &,
                               const std::vector<RotationMatrix> &, std::vector<SymmetryOperation> &symm_out) const;


    void assign_shift_vectors_supercell(const ALM_NS::Cell &cell_prim, const std::vector<SymmetryOperation> &symm_prim,
                                        const ALM_NS::Cell &cell_super,
                                        std::vector<PrimitiveGroup> &atomgroup_out) const;


    void print_symmetry_infomation(const int verbosity) const;
};
} // namespace ALM_NS
