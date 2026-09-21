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
#include <memory>
#include <string>
#include <vector>
#include "memory.h"
#include "ndarray.h"
#include "phonon.h"
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

    KpointList &operator=(const KpointList &obj) = default;

    KpointList(const unsigned int knum_in, const std::vector<double> &vec) : kval(vec), knum(knum_in) {};
};

class KpointInp
{
public:
    std::vector<std::string> kpelem;

    KpointInp() {};

    KpointInp(const std::vector<std::string> &obj) : kpelem(obj) {};
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
    };

    KpointGeneral(const unsigned int nk_in, const double *const *xk_in, const double *const *kvec_na_in)
    {
        nk = nk_in;
        xk.resize(nk, 3);
        kvec_na.resize(nk, 3);

        for (auto i = 0; i < nk; ++i) {
            for (auto j = 0; j < 3; ++j) {
                xk[i][j] = xk_in[i][j];
                kvec_na[i][j] = kvec_na_in[i][j];
            }
        }
    };

    KpointGeneral(const KpointGeneral &) = delete;
    KpointGeneral &operator=(const KpointGeneral &) = delete;

    unsigned int nk;
    NDArray<double, 2> xk;
    NDArray<double, 2> kvec_na;
};

struct KpointSymmetry
{
public:
    // k = (-1)^time_reversal S(symmetry_op) k_orig, modulo reciprocal lattice vectors.
    int symmetry_op = -1;
    bool time_reversal = false;
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
        xk.resize(nk, 3);
        kvec_na.resize(nk, 3);
    };

    KpointMeshUniform(const KpointMeshUniform &) = delete;
    KpointMeshUniform &operator=(const KpointMeshUniform &) = delete;

    unsigned int nk_i[3]{};
    unsigned int nk{}, nk_irred{};

    NDArray<double, 2> xk;
    NDArray<double, 2> kvec_na;
    std::vector<double> weight_k;
    std::vector<unsigned int> kmap_to_irreducible;
    std::vector<std::vector<KpointList>> kpoint_irred_all;
    std::vector<std::vector<int>> small_group_of_k;
    std::vector<unsigned int> kindex_minus_xk;
    std::vector<std::vector<int>> symop_minus_at_k;
    std::vector<KpointSymmetry> kpoint_map_symmetry;

    bool niggli_reduced = false;

    void setup(const std::vector<SymmetryOperation> &symmlist, const Eigen::Matrix3d &rlavec_p,
               const bool time_reversal_symmetry = false, const bool niggli_reduce_in = false);

    int get_knum(const double xk[3]) const;

    int knum_sym(const unsigned int ik, const Eigen::Matrix3i &rot) const;

    void get_unique_triplet_k(const int ik, const std::vector<SymmetryOperation> &symmlist,
                              const bool use_triplet_symmetry, const bool use_permutation_symmetry,
                              std::vector<KsListGroup> &triplet, const int sign = -1) const;

    // Partners of knum, without crystal-symmetry reduction: sign*k1 + k2 + k3 = G.
    // sign = -1 for squared vertices, +1 for Phi. Permutations share a group when enabled.
    // Matches get_unique_triplet_k at star representatives with crystal symmetry disabled.
    void get_triplets_at_k(const int knum, const bool use_permutation_symmetry, std::vector<KsListGroup> &triplet,
                           const int sign = -1) const;

    void get_quartets_at_k(const int knum, const bool use_permutation_symmetry, std::vector<KsListGroup> &quartet,
                           const int sign = -1) const;

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
    };

    KpointBandStructure(const unsigned int nk_in, const double *const *xk_in, const double *const *kvec_na_in,
                        const double *kaxis_in)
    {
        nk = nk_in;
        xk.resize(nk, 3);
        kvec_na.resize(nk, 3);
        kaxis.resize(nk);

        for (auto i = 0; i < nk; ++i) {
            for (auto j = 0; j < 3; ++j) {
                xk[i][j] = xk_in[i][j];
                kvec_na[i][j] = kvec_na_in[i][j];
            }
            kaxis[i] = kaxis_in[i];
        }
    };

    KpointBandStructure(const KpointBandStructure &) = delete;
    KpointBandStructure &operator=(const KpointBandStructure &) = delete;

    unsigned int nk;
    NDArray<double, 2> xk;
    NDArray<double, 2> kvec_na;
    NDArray<double, 1> kaxis;
};

class Kpoint
{
public:
    Kpoint(const RunInfo &run, const System *system, const Symmetry *symmetry);

    ~Kpoint();

    void kpoint_setups(std::string);

    int kpoint_mode;

    // KPMODE = 2: mesh dimensions from &kpoint, valid on all ranks after kpoint_setups().
    unsigned int nk_mesh[3]{0, 0, 0};

    void print_uniform_mesh_info(const KpointMeshUniform &kmesh) const;

    std::vector<KpointInp> kpInp;

    // Self-energy targets: KPMODE 0 list or 1 path. kpInp/kpoint_mode hold KMESH.
    int target_mode = -1;
    std::vector<KpointInp> kpInp_targets;

    std::unique_ptr<KpointBandStructure> kpoint_bs;
    std::unique_ptr<KpointGeneral> kpoint_general;

    void get_symmetrization_matrix_at_k(const double *xk_in, std::vector<int> &sym_list, double S_avg[3][3]) const;

    void setup_kpoint_band(const std::vector<KpointInp> &kpinfo, const Eigen::Matrix3d &rlavec_p);

    static int get_kmap_coarse_to_dense(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                        std::vector<int> &kmap);

private:
    void set_default_variables();

    void deallocate_variables();

    void setup_kpoint_given(const std::vector<KpointInp> &kpinfo, const Eigen::Matrix3d &rlavec_p);

private:
    // Collaborators (non-owning; owned by PHON, which outlives this object).
    const RunInfo &run;
    const System *system;
    const Symmetry *symmetry;
};
} // namespace PHON_NS
