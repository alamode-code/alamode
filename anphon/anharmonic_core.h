/*
anharmonic_core.h

Copyright (c) 2014 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <vector>
#include "fcs_phonon.h"
#include "kpoint.h"
#include "pointers.h"

namespace PHON_NS
{

struct QuartS
{
    // for an (k,s)
    // we store for a single k quartic pair and {s1,s2,s3}, with the smearing result,
    // only for smearing method

    int s1, s2, s3;

    double delta1;
    double delta2;

    QuartS();

    QuartS(int in1, int in2, int in3) : s1(in1), s2(in2), s3(in3)
    {}

    QuartS(int in1, int in2, int in3, double d1, double d2) : s1(in1), s2(in2), s3(in3), delta1(d1), delta2(d2)
    {}
};

class RelativeVector
{
public:
    double vecs[3][3];

    RelativeVector();

    // Constructor for cubic term
    RelativeVector(const double vec1[3], const double vec2[3])
    {
        for (int i = 0; i < 3; ++i) {
            vecs[0][i] = vec1[i];
            vecs[1][i] = vec2[i];
            vecs[2][i] = 0.0;
        }
    }

    // Constructor for quartic term
    RelativeVector(const double vec1[3], const double vec2[3], const double vec3[3])
    {
        for (int i = 0; i < 3; ++i) {
            vecs[0][i] = vec1[i];
            vecs[1][i] = vec2[i];
            vecs[2][i] = vec3[i];
        }
    }
};

class PhaseFactorStorage
{
public:
    PhaseFactorStorage() {};

    PhaseFactorStorage(const unsigned int nk_grid_in[3])
    {
        for (auto i = 0; i < 3; ++i) {
            nk_grid[i] = static_cast<int>(nk_grid_in[i]);
        }

        if (exp_phase) deallocate(exp_phase);
        if (exp_phase3) deallocate(exp_phase3);
    };

    ~PhaseFactorStorage()
    {
        if (exp_phase) deallocate(exp_phase);
        if (exp_phase3) deallocate(exp_phase3);
    };

    void create(const bool use_tuned_ver, const bool switch_to_type2 = false);

    unsigned int get_tune_type() const;

    std::complex<double> get_exp_type1(const double phase_in) const;

    std::complex<double> get_exp_type2(const double phase3_in[3]) const;

private:
    int nk_represent, nk_grid[3]; // This type must NOT be changed to unsigned int
    // because these variables are used as a divisor of modulo.
    // If the type is unsigned int, the phase factor returned by get_exp_type[1,2] becomes incorrect.
    unsigned int tune_type;
    double dnk_represent;
    double dnk[3];
    std::complex<double> *exp_phase = nullptr;
    std::complex<double> ***exp_phase3 = nullptr;
};

class AnharmonicCore: protected Pointers
{
public:
    AnharmonicCore(class PHON *);

    ~AnharmonicCore();

    void setup();

    void calc_damping_smearing(const unsigned int ntemp, const double *temp_in, const double omega_in,
                               const unsigned int ik_in, const unsigned int is_in, const KpointMeshUniform *kmesh_in,
                               const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                               double *ret);

    void calc_damping_tetrahedron(const unsigned int ntemp, const double *temp_in, const double omega_in,
                                  const unsigned int ik_in, const unsigned int is_in, const KpointMeshUniform *kmesh_in,
                                  const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                                  double *ret);

    void calc_damping4_smearing(const unsigned int ntemp, const double *temp_in, const double omega_in,
                                const unsigned int ik_in, const unsigned int is_in, const KpointMeshUniform *kmesh_in,
                                const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                                double *ret);

    void calc_damping4_smearing(const unsigned int ntemp, const double *temp_in, const double omega_in,
                                const unsigned int ik_in, const unsigned int is_in, const KpointMeshUniform *kmesh_in,
                                const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                                const PhaseFactorStorage *phase_storage_in, double *ret);

    void calc_damping4_smearing_batch(const unsigned int ntemp, const double *temp_in, const double omega_in,
                                      const unsigned int ik_in, const unsigned int is_in,
                                      const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                      const std::complex<double> *const *const *evec_in,
                                      const PhaseFactorStorage *phase_storage_in, double *ret);

    // a wrapper to return v3
    //std::complex<double> get_v3(const unsigned int [3],
    //                        double **,
    //                        std::complex<double> ***);

    int quartic_mode;
    bool use_tuned_ver;
    bool use_triplet_symmetry;
    bool use_quartet_symmetry;

    std::complex<double> V3(const unsigned int[3]);

    std::complex<double> V4(const unsigned int[4]);

    std::complex<double> Phi3(const unsigned int[3]);

    std::complex<double> Phi4(const unsigned int[4]);

    std::complex<double> V3(const unsigned int ks[3], const double *const *xk_in, const double *const *eval_in,
                            const std::complex<double> *const *const *evec_in);

    std::complex<double> V3(const unsigned int ks[3], const double *const *xk_in, const double *const *eval_in,
                            const std::complex<double> *const *const *evec_in,
                            const PhaseFactorStorage *phase_storage_in);

    // Thread-safe serial variants: the per-triplet reciprocal-FC3 cache
    // lives in caller-provided storage (phi3_work of size ngroup_v3 and
    // kindex_work[2] initialized to -1) and no OpenMP region is entered.
    // Parallelism belongs to the caller's triplet loop; entering a parallel
    // region per V3 call is far too fine-grained (measured to give negative
    // scaling in the SERTA and IBTE setups).
    std::complex<double> V3(const unsigned int ks[3], const double *const *xk_in, const double *const *eval_in,
                            const std::complex<double> *const *const *evec_in,
                            std::complex<double> *phi3_work, int *kindex_work);

    std::complex<double> V3(const unsigned int ks[3], const double *const *xk_in, const double *const *eval_in,
                            const std::complex<double> *const *const *evec_in,
                            const PhaseFactorStorage *phase_storage_in, std::complex<double> *phi3_work,
                            int *kindex_work);

    std::complex<double> V4(const unsigned int ks[4], const double *const *xk_in, const double *const *eval_in,
                            const std::complex<double> *const *const *evec_in,
                            const PhaseFactorStorage *phase_storage_in);

    std::complex<double> Phi3(const unsigned int ks[3], const double *const *xk_in, const double *const *eval_in,
                              const std::complex<double> *const *const *evec_in,
                              const PhaseFactorStorage *phase_storage_in);

    std::complex<double> Phi4(const unsigned int ks[4], const double *const *xk_in, const double *const *eval_in,
                              const std::complex<double> *const *const *evec_in,
                              const PhaseFactorStorage *phase_storage_in);

    std::complex<double> V3_mode(int, const double *, const double *, int, int, double **,
                                 std::complex<double> ***) const;

    static void prepare_relative_vector(const std::vector<FcsArrayWithCell> &fcs_in, const int number_of_groups,
                                        std::vector<double> *fcs_group, std::vector<RelativeVector> *&vec_out);

    static void prepare_group_of_force_constants(const std::vector<FcsArrayWithCell> &fcs_in, int &number_of_groups,
                                                 std::vector<double> *&fcs_group_out);

    void calc_self3omega_tetrahedron(const double Temp, const KpointMeshUniform *kmesh_in, const double *const *eval,
                                     const std::complex<double> *const *const *evec, const unsigned int ik_in,
                                     const unsigned int snum, const unsigned int nomega, const double *omega,
                                     double *ret);

    void calc_phi3_reciprocal(const double *xk1, const double *xk2, const int ngroup_v3_in,
                              std::vector<double, std::allocator<double>> *fcs_group_v3_in,
                              const std::vector<RelativeVector> *relvec_v3_in,
                              const PhaseFactorStorage *phase_storage_in, std::complex<double> *ret,
                              const bool use_openmp = true);

    void calc_phi4_reciprocal(const double *xk1, const double *xk2, const double *xk3,
                              const PhaseFactorStorage *phase_storage_in, std::complex<double> *ret);

    int get_ngroup_fcs(const unsigned int order) const;

    std::vector<double> *get_fcs_group(const unsigned int order) const;

    double *get_invmass_factor(const unsigned int order) const;

    int **get_evec_index(const unsigned int order) const;

    std::vector<RelativeVector> *get_relvec(const unsigned int order) const;

    void calc_analytic_k_from_FcsArrayWithCell(const double *, const std::vector<FcsArrayWithCell> &,
                                               std::complex<double> **) const;

private:
    void set_default_variables();

    void deallocate_variables();

    double *invmass_v3;
    double *invmass_v4;
    int **evec_index_v3;
    int **evec_index_v4;
    int ngroup_v3;
    int ngroup_v4;
    std::vector<double> *fcs_group_v3;
    std::vector<double> *fcs_group_v4;
    std::complex<double> *phi3_reciprocal, *phi4_reciprocal;
    std::vector<RelativeVector> *relvec_v3, *relvec_v4;

    PhaseFactorStorage *phase_storage_dos;

    bool sym_permutation;

    int kindex_phi3_stored[2] = {-1, -1};
    int kindex_phi4_stored[3] = {-1, -1, -1};

    void setup_cubic();

    void setup_quartic();

    std::vector<std::vector<QuartS>> reduce_pair(const int k_in, const int s0, const double omega, const int ismear,
                                                 const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                                 std::vector<KsListGroup> &quartet);

    void reduce_pair_simple(const int ik_in, const int snum, const double omega, const int ismear,
                            const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                            std::vector<KsListGroup> &quartet);
};
} // namespace PHON_NS
