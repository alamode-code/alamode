/*
anharmonic_core.h

Copyright (c) 2014 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <memory>
#include <vector>
#include "fcs_phonon.h"
#include "kpoint.h"
#include "ndarray.h"
#include "pointers.h"

namespace PHON_NS
{

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

class PhaseFactorCache
{
public:
    PhaseFactorCache() {};

    PhaseFactorCache(const unsigned int nk_grid_in[3])
    {
        for (auto i = 0; i < 3; ++i) {
            nk_grid[i] = static_cast<int>(nk_grid_in[i]);
        }
    };

    PhaseFactorCache(const PhaseFactorCache &) = delete;
    PhaseFactorCache &operator=(const PhaseFactorCache &) = delete;

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
    NDArray<std::complex<double>, 1> exp_phase;
    NDArray<std::complex<double>, 3> exp_phase3;
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

    // Four-phonon linewidth with smearing (four_phonon.cpp): the quartic
    // matrix elements of all band triples of a quartet are formed by
    // successive eigenvector contractions at O(n^4) per quartet instead of
    // O(ns^3 * N_terms).
    void calc_damping4_smearing(const unsigned int ntemp, const double *temp_in, const double omega_in,
                                const unsigned int ik_in, const unsigned int is_in, const KpointMeshUniform *kmesh_in,
                                const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                                double *ret);

    // a wrapper to return v3
    //std::complex<double> get_v3(const unsigned int [3],
    //                        double **,
    //                        std::complex<double> ***);

    int quartic_mode;
    bool use_tuned_ver;
    bool use_triplet_symmetry;
    bool use_quartet_symmetry;

    // ModeAnalysis legitimately overrides the parser defaults: mode-resolved
    // analyses on a uniform mesh need the full triplet/quartet lists, so the
    // symmetry reduction must be switched off through these entry points.
    void disable_triplet_symmetry()
    {
        use_triplet_symmetry = false;
    }

    void disable_quartet_symmetry()
    {
        use_quartet_symmetry = false;
    }

    std::complex<double> V3(const unsigned int[3]);

    std::complex<double> V4(const unsigned int[4]);

    std::complex<double> Phi3(const unsigned int[3]);

    std::complex<double> Phi4(const unsigned int[4]);

    std::complex<double> V3(const unsigned int ks[3], const double *const *xk_in, const double *const *eval_in,
                            const std::complex<double> *const *const *evec_in);

    std::complex<double> V3(const unsigned int ks[3], const double *const *xk_in, const double *const *eval_in,
                            const std::complex<double> *const *const *evec_in,
                            const PhaseFactorCache *phase_storage_in);

    // Thread-safe serial variants: the per-triplet reciprocal-FC3 cache
    // lives in caller-provided storage (phi3_work of size ngroup_v3 and
    // kindex_work[2] initialized to -1) and no OpenMP region is entered.
    // Parallelism belongs to the caller's triplet loop; entering a parallel
    // region per V3 call is far too fine-grained (measured to give negative
    // scaling in the SERTA and IBTE setups).
    std::complex<double> V3(const unsigned int ks[3], const double *const *xk_in, const double *const *eval_in,
                            const std::complex<double> *const *const *evec_in, std::complex<double> *phi3_work,
                            int *kindex_work);

    std::complex<double> V3(const unsigned int ks[3], const double *const *xk_in, const double *const *eval_in,
                            const std::complex<double> *const *const *evec_in, const PhaseFactorCache *phase_storage_in,
                            std::complex<double> *phi3_work, int *kindex_work);

    std::complex<double> V4(const unsigned int ks[4], const double *const *xk_in, const double *const *eval_in,
                            const std::complex<double> *const *const *evec_in,
                            const PhaseFactorCache *phase_storage_in);

    // Thread-safe serial variant of V4 (same design as the V3 overload
    // above): phi4_work has size ngroup_v4 and kindex_work[3] starts at -1.
    std::complex<double> V4(const unsigned int ks[4], const double *const *xk_in, const double *const *eval_in,
                            const std::complex<double> *const *const *evec_in, const PhaseFactorCache *phase_storage_in,
                            std::complex<double> *phi4_work, int *kindex_work);

    std::complex<double> Phi3(const unsigned int ks[3], const double *const *xk_in, const double *const *eval_in,
                              const std::complex<double> *const *const *evec_in,
                              const PhaseFactorCache *phase_storage_in);

    std::complex<double> Phi4(const unsigned int ks[4], const double *const *xk_in, const double *const *eval_in,
                              const std::complex<double> *const *const *evec_in,
                              const PhaseFactorCache *phase_storage_in);

    static void prepare_relative_vector(const std::vector<FcsArrayWithCell> &fcs_in, const int number_of_groups,
                                        std::vector<double> *fcs_group, std::vector<RelativeVector> *vec_out);

    static void prepare_group_of_force_constants(const std::vector<FcsArrayWithCell> &fcs_in, int &number_of_groups,
                                                 NDArray<std::vector<double>, 1> &fcs_group_out);

    void calc_self3omega_tetrahedron(const double Temp, const KpointMeshUniform *kmesh_in, const double *const *eval,
                                     const std::complex<double> *const *const *evec, const unsigned int ik_in,
                                     const unsigned int snum, const unsigned int nomega, const double *omega,
                                     double *ret);

    void calc_phi3_reciprocal(const double *xk1, const double *xk2, const int ngroup_v3_in,
                              const std::vector<double> *fcs_group_v3_in,
                              const std::vector<RelativeVector> *relvec_v3_in, const PhaseFactorCache *phase_storage_in,
                              std::complex<double> *ret, const bool use_openmp = true);

    // ---- Factorized cubic matrix elements (three_phonon.cpp) ----
    // Per-thread scratch of the V3 kernels; a fresh instance is set up by the
    // first call for a given mesh. psi_K (first-leg phase folded in) is
    // cached inside for v3sq_triples, whose contractions run in chunks of
    // s0_chunk first-leg bands to bound the buffers.
    struct V3Workspace
    {
        const KpointMeshUniform *kmesh = nullptr;
        int nkgrid[3] = {0, 0, 0};
        int s0_chunk = 0;
        std::vector<std::complex<double>> exp_table[3];
        std::vector<std::complex<double>> exp_dr, A, B, D, e1, e2;
        bool want_full = false;
        int psiK_index = -1;
        std::vector<std::complex<double>> psiK, phi_full, T1, T2, T3, e0;
        std::vector<size_t> touched;
    };

    void prepare_fc3_compressed();

    // psi_mode for the first leg (K, s0) on kmesh_in; call outside parallel
    // regions before v3sq_pairs.
    void prepare_v3_mode(const KpointMeshUniform *kmesh_in, const int kfirst, const int sfirst,
                         const std::complex<double> *const *const *evec_in);

    // |V3(K s0; k1 s1; k2 s2)|^2 for all (s1, s2) of the triplet, K + k1 + k2 = G,
    // with (K, s0) from the last prepare_v3_mode. out[s1 * ns + s2]. Thread-safe.
    void v3sq_pairs(V3Workspace &ws, const KpointMeshUniform *kmesh_in, const int k1, const int k2,
                    const double *const *eval_in, const std::complex<double> *const *const *evec_in, double *out) const;

    // |V3(K s0; k1 s1; k2 s2)|^2 for all (s0, s1, s2), K + k1 + k2 = G.
    // out[(s0 * ns + s1) * ns + s2]. Thread-safe; psi_K cached in ws.
    void v3sq_triples(V3Workspace &ws, const KpointMeshUniform *kmesh_in, const int kfirst, const int k1, const int k2,
                      const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                      double *out) const;

    void calc_phi4_reciprocal(const double *xk1, const double *xk2, const double *xk3,
                              const PhaseFactorCache *phase_storage_in, std::complex<double> *ret,
                              const bool use_openmp = true);

    int get_ngroup_fcs(const unsigned int order) const;

    const std::vector<double> *get_fcs_group(const unsigned int order) const;

    const double *get_invmass_factor(const unsigned int order) const;

    const int *const *get_evec_index(const unsigned int order) const;

    const std::vector<RelativeVector> *get_relvec(const unsigned int order) const;

private:
    void set_default_variables();

    void deallocate_variables();

    NDArray<double, 1> invmass_v3;
    NDArray<double, 1> invmass_v4;
    NDArray<int, 2> evec_index_v3;
    NDArray<int, 2> evec_index_v4;
    int ngroup_v3;
    int ngroup_v4;
    NDArray<std::vector<double>, 1> fcs_group_v3;
    NDArray<std::vector<double>, 1> fcs_group_v4;
    NDArray<std::complex<double>, 1> phi3_reciprocal, phi4_reciprocal;
    NDArray<std::vector<RelativeVector>, 1> relvec_v3, relvec_v4;

    std::unique_ptr<PhaseFactorCache> phase_storage_dos;

    // Quartic force constants regrouped for the factorized V4 evaluation
    // (four_phonon.cpp): terms sorted by (b c d, R2 - R3, R1, R3, a). With
    // k3 = k - k1 - k2 the phase of a quartet is
    //   k1.R1 + k2.R2 + k3.R3 = k1.R1 + (k - k1).R3 + k2.(R2 - R3),
    // so the first leg (a) is folded in once per mode, the R1 and R3 phases
    // once per k1 run, and the per-quartet Fourier sum runs over the distinct
    // (b c d, R2 - R3) terms only.
    struct FC4Compressed
    {
        int n = 0;                       // 3 * natmin
        int nr1 = 0, nr3 = 0, ndiff = 0; // unique R1 / R3 / (R2 - R3) vectors
        std::vector<int> rvec1;          // nr1 x 3 integer components of R1
        std::vector<int> rvec3;          // nr3 x 3 integer components of R3
        std::vector<int> dvec;           // ndiff x 3 integer components of R2 - R3
        std::vector<long long> row_bcd;  // flattened (b, c, d) of each non-empty row
        std::vector<int> row_ptr;        // row -> range in q_diff (size nrow + 1)
        std::vector<int> q_diff;         // (R2 - R3) index of each (b c d, R2 - R3) term
        std::vector<int> q_ptr;          // term -> range in pair_r1 / pair_r3 (size nq + 1)
        std::vector<int> pair_r1;        // R1 index of each (b c d, R1 R2 R3) pair
        std::vector<int> pair_r3;        // R3 index of each pair
        std::vector<int> pair_ptr;       // pair -> range in entry_a / entry_val (size npair + 1)
        std::vector<int> entry_a;        // first-leg index of each force-constant term
        std::vector<double> entry_val;   // force constant of each term
    };

    std::unique_ptr<FC4Compressed> fc4_compressed;

    void prepare_fc4_compressed();

    // Cubic force constants regrouped for the factorized V3 evaluation
    // (three_phonon.cpp): terms sorted by (b c, Rb - Rc, a, Rc) so that the
    // phase of the first leg can be folded in once per K, the first-leg
    // eigenvector once per mode, and a triplet costs one sparse sum over the
    // (b c, Rb - Rc) terms plus two dense products.
    struct FC3Compressed
    {
        int n = 0;                     // 3 * natmin
        int ndr = 0, nrc = 0;          // unique (Rb - Rc) / Rc vectors
        std::vector<int> dr_vec;       // ndr x 3
        std::vector<int> rc_vec;       // nrc x 3
        std::vector<long long> row_bc; // flattened (b, c) of each non-empty row
        std::vector<int> row_ptr;      // row -> range in grp_dr (size nrow + 1)
        std::vector<int> grp_dr;       // (Rb - Rc) index of each (b c, dR) group
        std::vector<int> grp_ptr;      // group -> range in sub_a (size ngrp + 1)
        std::vector<int> sub_a;        // first-leg index of each (a, b c, dR) subgroup
        std::vector<int> sub_ptr;      // subgroup -> range in entry_rc / entry_val (size nsub + 1)
        std::vector<int> entry_rc;     // Rc index of each force-constant term
        std::vector<double> entry_val; // force constant of each term
    };

    std::unique_ptr<FC3Compressed> fc3_compressed;

    // First-leg eigenvector folded into psi_K (prepare_v3_mode).
    std::vector<std::complex<double>> psi_mode;
    int psi_mode_kfirst = -1, psi_mode_sfirst = -1;

    // Unique quartets (k1, k2, k3) and multiplicities of the k point handled
    // last, reused across its bands.
    std::vector<int> quartet_cache_k;
    std::vector<double> quartet_cache_multi;
    const KpointMeshUniform *quartet_cache_kmesh = nullptr;
    int quartet_cache_ik = -1;
    int quartet_cache_flags = -1;
    bool fourph_memory_reported = false;

    bool sym_permutation;

    int kindex_phi3_stored[2] = {-1, -1};
    int kindex_phi4_stored[3] = {-1, -1, -1};

    void setup_cubic();

    void setup_quartic();

    V3Workspace v3_ws_mode;

    void v3_setup_workspace(V3Workspace &ws, const KpointMeshUniform *kmesh_in) const;

    void v3_fold_first_k(V3Workspace &ws, const int kfirst, std::vector<std::complex<double>> &psi_out) const;
};
} // namespace PHON_NS
