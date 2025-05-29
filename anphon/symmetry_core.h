/*
 symmetry_core.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "pointers.h"
#include "system.h"
#include <string>
#include <vector>
#include <set>
#include <Eigen/Core>

namespace PHON_NS
{
class SymmetryOperation
{
public:
    Eigen::Matrix3i rotation;      // in lattice basis
    Eigen::Vector3d tran;          // in lattice basis
    Eigen::Matrix3d rotation_cart; // in Cartesian basis
    bool is_translation;

    SymmetryOperation();

    SymmetryOperation(const Eigen::Matrix3i &rot_in,
                      const Eigen::Vector3d &tran_in,
                      const Eigen::Matrix3d &rot_cart_in,
                      const bool is_trans_in)
    {
        rotation = rot_in;
        rotation_cart = rot_cart_in;
        tran = tran_in;
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
        return std::lexicographical_compare(v1.begin(),
                                            v1.end(),
                                            v2.begin(),
                                            v2.end());
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

class SymmetryOperationWithMapping
{
public:
    std::vector<double> rot;            // Rotation matrix in Cartesian basis
    std::vector<double> rot_real;       // Rotation matrix in fractional basis
    std::vector<double> rot_reciprocal; // Rotation matrix in reciprocal (fractional) basis
    std::vector<unsigned int> mapping;
    double shift[3]; // Translation vector in fractional basis

    SymmetryOperationWithMapping();

    SymmetryOperationWithMapping(const double S[3][3],
                                 const double T[3][3],
                                 const double R[3][3],
                                 unsigned int *mapping_info,
                                 const unsigned int n,
                                 const double shift_in[3])
    {
        unsigned int i, j;

        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                rot.push_back(S[i][j]);
                rot_real.push_back(T[i][j]);
                rot_reciprocal.push_back(R[i][j]);
            }
        }
        for (i = 0; i < n; ++i) {
            mapping.push_back(mapping_info[i]);
        }
        for (i = 0; i < 3; ++i) {
            shift[i] = shift_in[i];
        }
    }

    SymmetryOperationWithMapping(const Eigen::Matrix3d &S,
                                 const Eigen::Matrix3d &T,
                                 const Eigen::Matrix3d &R,
                                 unsigned int *mapping_info,
                                 const unsigned int n,
                                 const Eigen::Vector3d &shift_in)
    {
        unsigned int i, j;

        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                rot.push_back(S(i, j));
                rot_real.push_back(T(i, j));
                rot_reciprocal.push_back(R(i, j));
            }
        }
        for (i = 0; i < n; ++i) {
            mapping.push_back(mapping_info[i]);
        }
        for (i = 0; i < 3; ++i) {
            shift[i] = shift_in[i];
        }
    }
};


class Symmetry: protected Pointers
{
public:
    Symmetry(class PHON *);

    ~Symmetry();

    unsigned int nsym, nsym_ref;
    bool time_reversal_sym;
    bool printsymmetry;
    double tolerance;
    std::vector<SymmetryOperation> SymmList;
    std::vector<SymmetryOperationWithMapping> SymmListWithMap;

    std::vector<SymmetryOperation> SymmList_ref;
    std::vector<SymmetryOperationWithMapping> SymmListWithMap_ref;

    void setup_symmetry();

private:
    std::string file_sym;

    void set_default_variables();

    void setup_symmetry_operation(const Cell &cell_in,
                                  const Spin &spin_in,
                                  const std::vector<std::vector<unsigned int>> &atomtype_in,
                                  std::vector<SymmetryOperation> &symlist,
                                  const int verbosity = 1);

    void gensym_withmap(const Eigen::Matrix3d &aa,
                        const Eigen::MatrixXd &x,
                        const std::vector<int> &kd,
                        const std::vector<SymmetryOperation> &symmlist_in,
                        std::vector<SymmetryOperationWithMapping> &symmlist_withmap_out) const;

    bool is_proper(const Eigen::Matrix3d &rot) const;

    bool is_translation(const Eigen::Matrix3i &rot) const;

    void find_lattice_symmetry(const Eigen::Matrix3d &aa,
                               std::vector<RotationMatrix> &) const;

    void find_crystal_symmetry(const Cell &cell,
                               const std::vector<std::vector<unsigned int>> &atomtype_group,
                               const Spin &spin,
                               const std::vector<RotationMatrix> &LatticeSymmList,
                               std::vector<SymmetryOperation> &symm_out) const;

    void findsym_alm(const Cell &,
                     const std::vector<std::vector<unsigned int>> &,
                     const Spin &,
                     std::vector<SymmetryOperation> &symm_out) const;

    int findsym_spglib(const Cell &,
                       const std::vector<std::vector<unsigned int>> &,
                       const Spin &,
                       std::string &,
                       std::vector<SymmetryOperation> &symm_out) const;

    void symop_in_cart(Eigen::Matrix3d &rot_cart,
                       const Eigen::Matrix3i &rot_lattice,
                       const Eigen::Matrix3d &lavec,
                       const Eigen::Matrix3d &rlavec) const;


    void broadcast_symmlist(std::vector<SymmetryOperation> &) const;
};
}
