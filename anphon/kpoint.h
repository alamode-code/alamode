/*
kpoint.h

Copyright (c) 2014-2021 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Core>
#include <map>
#include <string>
#include <vector>
#include "memory.h"
#include "pointers.h"
#include "symmetry_core.h"

namespace PHON_NS
{
class KpointList
{
public:
    std::vector<double> kval;
    unsigned int knum;

    KpointList() {};

    KpointList(const KpointList &obj) : kval(obj.kval), knum(obj.knum) {};

    KpointList(const unsigned int knum_in, const std::vector<double> &vec) : kval(vec), knum(knum_in) {};
};

class KpointInp
{
public:
    std::vector<std::string> kpelem;

    KpointInp() {};

    KpointInp(const std::vector<std::string> &obj) : kpelem(obj) {};
};

class KpointPlaneGeometry
{
public:
    double xk_origin[3];
    double xk_edges[2][3];
    int npoints[2];

    KpointPlaneGeometry() {};

    KpointPlaneGeometry(const double *xk_origin_in, const double *xk_edge1_in, const double *xk_edge2_in,
                        const int *num)
    {
        for (int i = 0; i < 3; ++i) {
            xk_origin[i] = xk_origin_in[i];
            xk_edges[0][i] = xk_edge1_in[i];
            xk_edges[1][i] = xk_edge2_in[i];
        }
        for (int i = 0; i < 2; ++i) {
            npoints[i] = num[i];
        }
    }
};

class KpointPlane
{
public:
    double k[3];
    int n[2];

    KpointPlane() {};

    KpointPlane(const double *xk_in, const int *n_in)
    {
        for (int i = 0; i < 3; ++i)
            k[i] = xk_in[i];
        for (int i = 0; i < 2; ++i)
            n[i] = n_in[i];
    }
};

class KpointPlaneTriangle
{
public:
    int index;
    int knum[3];

    KpointPlaneTriangle() {};

    KpointPlaneTriangle(int index_in, const int *nk_in)
    {
        index = index_in;

        for (int i = 0; i < 3; ++i) {
            knum[i] = nk_in[i];
        }
    }
};

class KsList
{
public:
    std::vector<int> ks;
    int symnum;

    KsList();

    KsList(const KsList &a) : ks(a.ks), symnum(a.symnum) {};

    KsList(const int n, int *ks_in, const int sym)
    {
        for (int i = 0; i < n; ++i) {
            ks.push_back(ks_in[i]);
        }
        symnum = sym;
    }

    bool operator<(const KsList &obj) const
    {
        return std::lexicographical_compare(ks.begin(), ks.end(), obj.ks.begin(), obj.ks.end());
    }
};

class KsListGroup
{
public:
    std::vector<KsList> group;

    KsListGroup();

    KsListGroup(const std::vector<KsList> &a) : group(a) {};
};

class KpointGeneral
{
public:
    KpointGeneral()
    {
        nk = 0;
        xk = nullptr;
        kvec_na = nullptr;
    };

    KpointGeneral(const unsigned int nk_in, const double *const *xk_in, const double *const *kvec_na_in)
    {
        nk = nk_in;
        if (xk) deallocate(xk);
        if (kvec_na) deallocate(kvec_na);
        allocate(xk, nk, 3);
        allocate(kvec_na, nk, 3);

        for (auto i = 0; i < nk; ++i) {
            for (auto j = 0; j < 3; ++j) {
                xk[i][j] = xk_in[i][j];
                kvec_na[i][j] = kvec_na_in[i][j];
            }
        }
    };

    ~KpointGeneral()
    {
        if (xk) {
            deallocate(xk);
            xk = nullptr;
        }
        if (kvec_na) {
            deallocate(kvec_na);
            kvec_na = nullptr;
        }
    }

    unsigned int nk;
    double **xk = nullptr;
    double **kvec_na = nullptr;
};

struct KpointSymmetry
{
public:
    int symmetry_op;
    unsigned int knum_irred_orig;
    unsigned int knum_orig;
};

class KpointMeshUniform
{
public:
    KpointMeshUniform() = default;

    KpointMeshUniform(const unsigned int nk_in[3])
    {
        for (auto i = 0; i < 3; ++i) {
            nk_i[i] = nk_in[i];
        }
        nk = nk_i[0] * nk_i[1] * nk_i[2];
        if (xk) deallocate(xk);
        if (kvec_na) deallocate(kvec_na);
        allocate(xk, nk, 3);
        allocate(kvec_na, nk, 3);
    };

    ~KpointMeshUniform()
    {
        if (xk) {
            deallocate(xk);
            xk = nullptr;
        }
        if (kvec_na) {
            deallocate(kvec_na);
            kvec_na = nullptr;
        }
    };

    unsigned int nk_i[3]{};
    unsigned int nk{}, nk_irred{};

    double **xk = nullptr;
    double **kvec_na = nullptr;
    std::vector<double> weight_k;
    std::vector<unsigned int> kmap_to_irreducible;
    std::vector<std::vector<KpointList>> kpoint_irred_all;
    std::vector<std::vector<int>> small_group_of_k;
    std::vector<unsigned int> kindex_minus_xk;
    std::vector<std::vector<int>> symop_minus_at_k;
    std::vector<KpointSymmetry> kpoint_map_symmetry;

    bool niggli_reduced = false;

    void setup(const std::vector<SymmetryOperation> &symmlist, const Eigen::Matrix3d &rlavec_p,
               const bool time_reversal_symmetry = true, const bool niggli_reduce_in = false);

    int get_knum(const double xk[3]) const;

    int knum_sym(const unsigned int ik, const Eigen::Matrix3i &rot) const;

    void get_unique_triplet_k(const int ik, const std::vector<SymmetryOperation> &symmlist,
                              const bool use_triplet_symmetry, const bool use_permutation_symmetry,
                              std::vector<KsListGroup> &triplet, const int sign = -1) const;

    void get_unique_quartet_k(const int ik, const std::vector<SymmetryOperation> &symmlist,
                              const bool use_quartet_symmetry, const bool use_permutation_symmetry,
                              std::vector<KsListGroup> &quartet, const int sign = -1) const;

    void setup_kpoint_symmetry(const std::vector<SymmetryOperationWithMapping> &symmlist);

private:
    void gen_kmesh(const std::vector<SymmetryOperation> &symmlist, const Eigen::Matrix3d &rlavec_p, const bool usesym,
                   const bool time_reversal_symmetry);

    void gen_kmesh_niggli(const std::vector<SymmetryOperation> &symmlist, const Eigen::Matrix3d &rlavec_p,
                          const bool usesym, const bool time_reversal_symmetry);

    void reduce_kpoints(const unsigned int nsym, const std::vector<SymmetryOperation> &symmlist,
                        const bool time_reversal_symmetry, const double *const *xkr);

    void gen_nkminus();

    void set_small_groups_k_irred(const bool usesym, const std::vector<SymmetryOperation> &symmlist);

    std::vector<int> get_small_group_of_k(const unsigned int ik, const bool usesym,
                                          const std::vector<SymmetryOperation> &symmlist) const;
};

class KpointBandStructure
{
public:
    KpointBandStructure()
    {
        nk = 0;
        xk = nullptr;
        kvec_na = nullptr;
        kaxis = nullptr;
    };

    KpointBandStructure(const unsigned int nk_in, const double *const *xk_in, const double *const *kvec_na_in,
                        const double *kaxis_in)
    {
        nk = nk_in;
        if (xk) deallocate(xk);
        if (kvec_na) deallocate(kvec_na);
        if (kaxis) deallocate(kaxis);

        allocate(xk, nk, 3);
        allocate(kvec_na, nk, 3);
        allocate(kaxis, nk);

        for (auto i = 0; i < nk; ++i) {
            for (auto j = 0; j < 3; ++j) {
                xk[i][j] = xk_in[i][j];
                kvec_na[i][j] = kvec_na_in[i][j];
            }
            kaxis[i] = kaxis_in[i];
        }
    };

    ~KpointBandStructure()
    {
        if (xk) {
            deallocate(xk);
            xk = nullptr;
        }
        if (kvec_na) {
            deallocate(kvec_na);
            kvec_na = nullptr;
        }
        if (kaxis) {
            deallocate(kaxis);
            kaxis = nullptr;
        }
    }

    unsigned int nk;
    double **xk = nullptr;
    double **kvec_na = nullptr;
    double *kaxis = nullptr;
};

class Kpoint: protected Pointers
{
public:
    Kpoint(class PHON *);

    ~Kpoint();

    void kpoint_setups(std::string);

    int kpoint_mode;

    std::vector<KpointInp> kpInp;

    std::vector<KpointPlane> *kp_planes;
    std::vector<KpointPlaneGeometry> kp_plane_geometry;
    std::vector<KpointPlaneTriangle> *kp_planes_tri;

    KpointBandStructure *kpoint_bs;
    KpointGeneral *kpoint_general;

    int get_knum(const double[3], const unsigned int[3]) const;

    void get_symmetrization_matrix_at_k(const double *xk_in, std::vector<int> &sym_list, double S_avg[3][3]) const;

    void get_commensurate_kpoints(const Eigen::Matrix3d &lavec_super, const Eigen::Matrix3d &lavec_prim,
                                  std::vector<std::vector<double>> &klist) const;

    int get_kmap_coarse_to_dense(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                 std::vector<int> &kmap) const;

private:
    void set_default_variables();

    void deallocate_variables();

    void setup_kpoint_given(const std::vector<KpointInp> &kpinfo, const Eigen::Matrix3d &rlavec_p);

    void setup_kpoint_band(const std::vector<KpointInp> &kpinfo, const Eigen::Matrix3d &rlavec_p);

    void setup_kpoint_plane(const std::vector<KpointInp> &, unsigned int &, std::vector<KpointPlane> *&);

    void gen_kpoints_plane(const std::vector<KpointInp> &, std::vector<KpointPlane> *,
                           std::vector<KpointPlaneTriangle> *);

    bool in_first_BZ(const double *) const;

    void mpi_broadcast_kplane_vector(unsigned int, std::vector<KpointPlane> *&) const;
};
} // namespace PHON_NS
