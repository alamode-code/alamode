/*
four_phonon.cpp

Four-phonon scattering rates (imaginary part of the four-phonon self-energy)
evaluated with a factorized quartic matrix element.

Copyright (c) 2026 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "integration.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "mpi_common.h"
#include "symmetry_core.h"
#include "system.h"
#include "thermodynamics.h"
#include "write_phonons.h"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace PHON_NS;

// ----------------------------------------------------------------------------
// Background
//
// The four-phonon linewidth of mode (k, s) needs |V4(-k s, k1 s1, k2 s2, k3 s3)|^2
// for every momentum-conserving quartet (k1, k2, k3) and every band triple
// (s1, s2, s3). With n = 3 * natmin Cartesian indices per cell,
//
//   V4 = sum_{a b c d} e0[a] e1[b] e2[c] e3[d] Phi(k1,k2,k3)[a,b,c,d],
//   Phi(k1,k2,k3)[a,b,c,d] = sum_{R1 R2 R3} phi(a b c d; R1 R2 R3) exp(i(k1 R1 + k2 R2 + k3 R3)),
//
// where e_m are the mass-scaled eigenvectors.
// The summation over sum_{R1 R2 R3} is performed by scanning non-zero
// quartic IFCs phi(a b c d; R1 R2 R3), which costs O(N_FC) per quartet,
// yielding Phi(k1,k2,k3)[a,b,c,d] (dimension is N_grp per (k1, k2, k3)).
//
// Evaluating the four-fold sum separately for each band triple costs O(N_FC + ns^3 * N_grp) per quartet.
// Instead the legs are contracted one at a time:
//
//   A[b,c,d]      = sum_a e0[a] Phi[a,b,c,d]        (sparse; e0 fixed per mode)
//   B[s1,c,d]     = sum_b e1[s1,b] A[b,c,d]         (dense, ns x n x n^2)
//   C[s1,s2,d]    = sum_c e2[s2,c] B[s1,c,d]        (dense, ns x (ns x n x n))
//   D[s1,s2,s3]   = sum_d e3[s3,d] C[s1,s2,d]       (only where energy conservation allows)
//
// which costs O(n^4) per quartet independent of the number of IFCs,
// plus the Fourier sum for A. That sum is split in two stages. With
// k3 = k - k1 - k2 the phase of a quartet is
//   k1.R1 + k2.R2 + k3.R3 = k1.R1 + (k - k1).R3 + k2.(R2 - R3),
// so the e0 contraction is folded into the real-space force constants once
// per mode, the R1 and R3 phases once per k1 (quartets sharing k1 are
// processed together), and only the distinct (b c d, R2 - R3) terms remain
// per quartet.
//
// Band triples that fail the smearing window are located with a bisection
// over the sorted frequencies of k3 and dropped before any matrix element is
// formed; the Bose factors are tabulated once per call.
// ----------------------------------------------------------------------------

namespace
{

struct FC4Term
{
    long long bcd;
    int diff; // unique (R2 - R3) index
    int r1;   // unique R1 index
    int r3;   // unique R3 index
    int a;
    double val;
};

// Timing accumulators for the optional profile (VERBOSITY >= 2).
struct Profile
{
    double t_filter = 0.0;
    double t_fourier = 0.0;
    double t_contract = 0.0;
    double t_accumulate = 0.0;
    long long n_quartet = 0;
    long long n_quartet_skipped = 0;
    long long n_triplet = 0;
    long long n_triplet_kept = 0;

    void add(const Profile &o)
    {
        t_filter += o.t_filter;
        t_fourier += o.t_fourier;
        t_contract += o.t_contract;
        t_accumulate += o.t_accumulate;
        n_quartet += o.n_quartet;
        n_quartet_skipped += o.n_quartet_skipped;
        n_triplet += o.n_triplet;
        n_triplet_kept += o.n_triplet_kept;
    }
};

inline double wall_seconds()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline int positive_modulo(const int a, const int b)
{
    const int m = a % b;
    return m < 0 ? m + b : m;
}

// exp(i k.R) for integer R and k = q / N on the uniform grid.
inline std::complex<double> phase_factor(const int *q, const int *R, const int *ngrid,
                                         const std::vector<std::complex<double>> *table)
{
    std::complex<double> e(1.0, 0.0);
    for (auto icrd = 0; icrd < 3; ++icrd) {
        e *= table[icrd][positive_modulo(q[icrd] * R[icrd], ngrid[icrd])];
    }
    return e;
}

// Stage 2 of the Fourier sum for a block of NQ quartets sharing k1:
//   A_q[b c d] = sum_{dR} chi_k1(b c d, dR) * E_q(dR),  dR = R2 - R3,  q = 0..NQ-1.
// Each force-constant term is loaded once per block and the phase tables of
// the block stay in cache. Complex products are written out in real
// arithmetic to keep the loop free of the NaN-checking slow paths of
// std::complex operator*.
template <int NQ>
void fourier_stage2_block(const int nrow, const int *row_ptr, const long long *row_bcd, const int *q_diff,
                          const std::complex<double> *chi_k1, const std::complex<double> *const *exp_diff,
                          std::complex<double> *const *A_out)
{
    for (auto irow = 0; irow < nrow; ++irow) {
        double sre[NQ], sim[NQ];
        for (auto q = 0; q < NQ; ++q) {
            sre[q] = 0.0;
            sim[q] = 0.0;
        }
        for (auto iq = row_ptr[irow]; iq < row_ptr[irow + 1]; ++iq) {
            const double pr = chi_k1[iq].real();
            const double pi_ = chi_k1[iq].imag();
            const int r = q_diff[iq];
            for (auto q = 0; q < NQ; ++q) {
                const double er = exp_diff[q][r].real();
                const double ei = exp_diff[q][r].imag();
                sre[q] += pr * er - pi_ * ei;
                sim[q] += pr * ei + pi_ * er;
            }
        }
        const auto bcd = row_bcd[irow];
        for (auto q = 0; q < NQ; ++q) A_out[q][bcd] = std::complex<double>(sre[q], sim[q]);
    }
}

} // namespace

void AnharmonicCore::prepare_fc4_compressed()
{
    // Build the (b c d, R2 - R3, R1, R3, a)-grouped representation of the
    // quartic force constants used by calc_damping4_smearing. Independent of
    // the mode and of the k mesh, so it is built once.

    auto cfc = std::make_unique<FC4Compressed>();
    const int n = dynamical->neval;
    cfc->n = n;

    std::map<std::array<int, 3>, int> r1map, r3map, dmap;
    std::vector<FC4Term> terms;

    size_t nterms_total = 0;
    for (auto i = 0; i < ngroup_v4; ++i) nterms_total += fcs_group_v4[i].size();
    terms.reserve(nterms_total);

    std::array<int, 3> key1{}, key3{}, keyd{};

    for (auto i = 0; i < ngroup_v4; ++i) {
        const int a = evec_index_v4[i][0];
        const long long b = evec_index_v4[i][1];
        const long long c = evec_index_v4[i][2];
        const long long d = evec_index_v4[i][3];
        const long long bcd = (b * n + c) * n + d;
        const auto nsize_group = fcs_group_v4[i].size();

        for (size_t j = 0; j < nsize_group; ++j) {
            const auto &vecs = relvec_v4[i][j].vecs;
            for (auto icrd = 0; icrd < 3; ++icrd) {
                const int r2 = nint(vecs[1][icrd] * inv_tpi);
                key1[icrd] = nint(vecs[0][icrd] * inv_tpi);
                key3[icrd] = nint(vecs[2][icrd] * inv_tpi);
                keyd[icrd] = r2 - key3[icrd];
            }
            const auto it1 = r1map.emplace(key1, static_cast<int>(r1map.size())).first;
            const auto it3 = r3map.emplace(key3, static_cast<int>(r3map.size())).first;
            const auto itd = dmap.emplace(keyd, static_cast<int>(dmap.size())).first;
            terms.push_back({bcd, itd->second, it1->second, it3->second, a, fcs_group_v4[i][j]});
        }
    }

    cfc->nr1 = static_cast<int>(r1map.size());
    cfc->nr3 = static_cast<int>(r3map.size());
    cfc->ndiff = static_cast<int>(dmap.size());
    cfc->rvec1.resize(3 * cfc->nr1);
    cfc->rvec3.resize(3 * cfc->nr3);
    cfc->dvec.resize(3 * cfc->ndiff);
    for (const auto &it: r1map) {
        for (auto m = 0; m < 3; ++m) cfc->rvec1[3 * it.second + m] = it.first[m];
    }
    for (const auto &it: r3map) {
        for (auto m = 0; m < 3; ++m) cfc->rvec3[3 * it.second + m] = it.first[m];
    }
    for (const auto &it: dmap) {
        for (auto m = 0; m < 3; ++m) cfc->dvec[3 * it.second + m] = it.first[m];
    }

    std::sort(terms.begin(), terms.end(), [](const FC4Term &x, const FC4Term &y) {
        if (x.bcd != y.bcd) return x.bcd < y.bcd;
        if (x.diff != y.diff) return x.diff < y.diff;
        if (x.r1 != y.r1) return x.r1 < y.r1;
        if (x.r3 != y.r3) return x.r3 < y.r3;
        return x.a < y.a;
    });

    cfc->entry_a.resize(terms.size());
    cfc->entry_val.resize(terms.size());
    cfc->pair_r1.clear();
    cfc->pair_r3.clear();
    cfc->pair_ptr.clear();
    cfc->q_diff.clear();
    cfc->q_ptr.clear();
    cfc->row_bcd.clear();
    cfc->row_ptr.clear();

    long long bcd_prev = -1;
    int diff_prev = -1, r1_prev = -1, r3_prev = -1;
    for (size_t it = 0; it < terms.size(); ++it) {
        const auto &t = terms[it];
        if (t.bcd != bcd_prev) {
            cfc->row_bcd.push_back(t.bcd);
            cfc->row_ptr.push_back(static_cast<int>(cfc->q_diff.size()));
            bcd_prev = t.bcd;
            diff_prev = -1;
            r1_prev = -1;
            r3_prev = -1;
        }
        if (t.diff != diff_prev) {
            cfc->q_diff.push_back(t.diff);
            cfc->q_ptr.push_back(static_cast<int>(cfc->pair_r1.size()));
            diff_prev = t.diff;
            r1_prev = -1;
            r3_prev = -1;
        }
        if (t.r1 != r1_prev || t.r3 != r3_prev) {
            cfc->pair_r1.push_back(t.r1);
            cfc->pair_r3.push_back(t.r3);
            cfc->pair_ptr.push_back(static_cast<int>(it));
            r1_prev = t.r1;
            r3_prev = t.r3;
        }
        cfc->entry_a[it] = t.a;
        cfc->entry_val[it] = t.val;
    }
    cfc->pair_ptr.push_back(static_cast<int>(terms.size()));
    cfc->q_ptr.push_back(static_cast<int>(cfc->pair_r1.size()));
    cfc->row_ptr.push_back(static_cast<int>(cfc->q_diff.size()));

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << "\n";
        std::cout << " Four-phonon matrix elements: factorized evaluation\n";
        std::cout << "  Cartesian indices per cell (n)        : " << n << '\n';
        std::cout << "  Quartic force constant terms           : " << terms.size() << '\n';
        std::cout << "  Unique R1 / R3 / (R2 - R3) vectors     : " << cfc->nr1 << " / " << cfc->nr3 << " / "
                  << cfc->ndiff << '\n';
        std::cout << "  (b c d, R1 R2 R3) pairs (per k1)       : " << cfc->pair_r1.size() << '\n';
        std::cout << "  (b c d, R2 - R3) terms (per quartet)   : " << cfc->q_diff.size() << '\n';
        std::cout << "  Non-empty (b c d) rows                 : " << cfc->row_bcd.size() << " of "
                  << static_cast<long long>(n) * n * n << '\n';
        std::cout << std::flush;
    }

    fc4_compressed = std::move(cfc);
}

void AnharmonicCore::calc_damping4_smearing(const unsigned int ntemp, const double *temp_in, const double omega_in,
                                            const unsigned int ik_in, const unsigned int is_in,
                                            const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                            const std::complex<double> *const *const *evec_in, double *ret)
{
    using Cplx = std::complex<double>;
    using MatrixRow = Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MapRow = Eigen::Map<MatrixRow>;
    using ConstMapRow = Eigen::Map<const MatrixRow>;

    const int nk = kmesh_in->nk;
    const int ns = dynamical->neval;
    const int n = ns;
    const size_t ns2 = static_cast<size_t>(ns) * ns;
    const size_t ns3 = ns2 * ns;
    const size_t n2 = static_cast<size_t>(n) * n;
    const size_t n3 = n2 * n;

    for (unsigned int i = 0; i < ntemp; ++i) ret[i] = 0.0;

    if (ngroup_v4 == 0) return;
    if (!fc4_compressed) prepare_fc4_compressed();
    const auto &cfc = *fc4_compressed;
    if (cfc.n != n) {
        exit("calc_damping4_smearing", "Inconsistent number of Cartesian indices in the compressed FC4.");
    }

    const int knum = kmesh_in->kpoint_irred_all[ik_in][0].knum;
    const int knum_minus = kmesh_in->kindex_minus_xk[knum];
    const double omega0 = eval_in[knum_minus][is_in];

    // A mode with an imaginary frequency has a vanishing matrix element.
    if (omega0 < eps8) return;

    // VERBOSITY >= 2 prints a per-mode timing breakdown of the kernel.
    const bool profile = writes->getVerbosity() > 1;
    Profile prof;
    double t_start = 0.0, t_mode_prep = 0.0;
    if (profile) t_start = wall_seconds();

    // Quartet list, cached across the bands of the same k point in the compact
    // form (k1, k2, k3, multiplicity); the full symmetry groups are not kept.
    if (quartet_cache_kmesh != kmesh_in || quartet_cache_ik != static_cast<int>(ik_in) ||
        quartet_cache_flags != (use_quartet_symmetry ? 2 : 0) + (sym_permutation ? 1 : 0))
    {
        std::vector<KsListGroup> quartet_full;
        kmesh_in->get_unique_quartet_k(ik_in, symmetry->SymmList, use_quartet_symmetry, sym_permutation, quartet_full);
        quartet_cache_k.resize(3 * quartet_full.size());
        quartet_cache_multi.resize(quartet_full.size());
        for (size_t iq = 0; iq < quartet_full.size(); ++iq) {
            for (auto m = 0; m < 3; ++m) quartet_cache_k[3 * iq + m] = quartet_full[iq].group[0].ks[m];
            quartet_cache_multi[iq] = static_cast<double>(quartet_full[iq].group.size());
        }
        quartet_cache_kmesh = kmesh_in;
        quartet_cache_ik = static_cast<int>(ik_in);
        quartet_cache_flags = (use_quartet_symmetry ? 2 : 0) + (sym_permutation ? 1 : 0);
    }
    const int *quartet_k = quartet_cache_k.data();
    const double *quartet_multi = quartet_cache_multi.data();
    const int nquartet = static_cast<int>(quartet_cache_multi.size());

    // Runs of consecutive quartets sharing k1 (the generator emits them in
    // k1 order); the k1-dependent phases are applied once per run.
    std::vector<int> run_start;
    for (auto iq = 0; iq < nquartet; ++iq) {
        if (iq == 0 || quartet_k[3 * iq] != quartet_k[3 * (iq - 1)]) run_start.push_back(iq);
    }
    run_start.push_back(nquartet);
    const int nrun = static_cast<int>(run_start.size()) - 1;

    // Mass-scaled eigenvector of the fixed leg (-k, s) and the e0-folded
    // real-space force constants phitilde(b c d, R1 R2 R3).
    const auto &invsqrt_mass = system->get_invsqrt_mass();
    std::vector<Cplx> e0(n);
    for (auto a = 0; a < n; ++a) e0[a] = evec_in[knum_minus][is_in][a] * invsqrt_mass[a / 3];

    const int npair = static_cast<int>(cfc.pair_r1.size());
    const int nq = static_cast<int>(cfc.q_diff.size());
    const int nrow = static_cast<int>(cfc.row_bcd.size());
    std::vector<Cplx> phitilde(npair);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int ip = 0; ip < npair; ++ip) {
        Cplx sum(0.0, 0.0);
        for (auto ie = cfc.pair_ptr[ip]; ie < cfc.pair_ptr[ip + 1]; ++ie) {
            sum += e0[cfc.entry_a[ie]] * cfc.entry_val[ie];
        }
        phitilde[ip] = sum;
    }

    // Integer k-point coordinates and the exp(2 pi i j / N) tables.
    int nkgrid[3];
    for (auto i = 0; i < 3; ++i) nkgrid[i] = static_cast<int>(kmesh_in->nk_i[i]);
    std::vector<int> qint(3 * static_cast<size_t>(nk));
    for (auto ik = 0; ik < nk; ++ik) {
        for (auto i = 0; i < 3; ++i) {
            qint[3 * ik + i] = nint(kmesh_in->xk[ik][i] * static_cast<double>(nkgrid[i]));
        }
    }
    std::vector<Cplx> exp_table[3];
    for (auto i = 0; i < 3; ++i) {
        exp_table[i].resize(nkgrid[i]);
        for (auto j = 0; j < nkgrid[i]; ++j) {
            exp_table[i][j] = std::exp(im * (tpi * static_cast<double>(j) / static_cast<double>(nkgrid[i])));
        }
    }

    // Occupation numbers occ[itemp][k * ns + s].
    const bool classical = thermodynamics->classical;
    const size_t nks = static_cast<size_t>(nk) * ns;
    std::vector<double> occ(static_cast<size_t>(ntemp) * nks);
    for (unsigned int it = 0; it < ntemp; ++it) {
        const double T = temp_in[it];
        for (auto ik = 0; ik < nk; ++ik) {
            for (auto is = 0; is < ns; ++is) {
                const auto w = eval_in[ik][is];
                double f;
                if (w < eps8) {
                    f = 0.0;
                } else if (classical) {
                    f = Thermodynamics::fC(w, T);
                } else {
                    f = Thermodynamics::fB(w, T);
                }
                occ[it * nks + ik * ns + is] = f;
            }
        }
    }

    // Smearing settings. For the adaptive scheme the widths are quadratic
    // forms of the projections of the group velocities on the mesh spacings.
    const int ismear = integration->ismear_4ph;
    const double epsilon = integration->epsilon_4ph;
    const bool adaptive = (ismear == 2);
    std::vector<double> proj;
    double adaptive_factor2 = 0.0;
    double sigma_min2 = 0.0;
    if (adaptive) {
        proj.resize(3 * nks);
        for (auto ik = 0; ik < nk; ++ik) {
            for (auto is = 0; is < ns; ++is) {
                integration->adaptive_sigma4->get_projected_velocity(ik, is, &proj[3 * (ik * ns + is)]);
            }
        }
        const auto f = integration->adaptive_sigma4->get_adaptive_factor();
        adaptive_factor2 = f * f / 12.0;
        sigma_min2 = AdaptiveSmearingSigma::sigma_min * AdaptiveSmearingSigma::sigma_min;
    }

    // The seven frequency sums omega +- w1 +- w2 +- w3, the delta term each
    // one feeds, its sign there, and the width index it uses.
    constexpr double signs_omega[7][3] =
        {{-1, -1, -1}, {-1, -1, 1}, {1, 1, -1}, {1, -1, -1}, {-1, 1, 1}, {-1, 1, -1}, {1, -1, 1}};
    constexpr int delta_of_sum[7] = {0, 1, 1, 2, 2, 3, 3};
    constexpr double sign_of_sum[7] = {1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0};

    if (profile) t_mode_prep = wall_seconds() - t_start;

    // Quartets of a k1 run are processed in blocks so that each (b c d, R2 - R3)
    // term is streamed once per block; the block's A arrays are kept below
    // about 64 MB per thread. The contractions run in chunks of s1 so that
    // the B, C and D buffers stay below the same budget; A itself must hold
    // all n^3 entries.
    constexpr size_t budget_bytes = static_cast<size_t>(64) << 20;
    constexpr int nq_block_max = 4;
    const int nq_block =
        static_cast<int>(std::max<size_t>(1, std::min<size_t>(nq_block_max, budget_bytes / (n3 * sizeof(Cplx)))));
    const int s1_chunk =
        static_cast<int>(std::max<size_t>(1, std::min<size_t>(ns, budget_bytes / (3 * n2 * sizeof(Cplx)))));

    if (!fourph_memory_reported && mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        const double mb = 1.0 / (1024.0 * 1024.0);
        const double a_mb = static_cast<double>(nq_block) * n3 * sizeof(Cplx) * mb;
        const double bcd_mb = 3.0 * s1_chunk * n2 * sizeof(Cplx) * mb;
        const double fc_mb = (static_cast<double>(cfc.q_diff.size()) + cfc.pair_r1.size()) * sizeof(Cplx) * mb;
        const double kept_mb = static_cast<double>(nq_block) * ns3 * (sizeof(int) + 4 * sizeof(double)) * mb;
        std::cout << "  Four-phonon work arrays per thread: " << std::fixed << std::setprecision(1)
                  << a_mb + bcd_mb + fc_mb << " MB (A blocks " << a_mb << " MB for " << nq_block
                  << " quartets, contraction buffers " << bcd_mb << " MB for " << s1_chunk
                  << " bands at a time, folded force constants " << fc_mb << " MB), plus up to " << kept_mb
                  << " MB of kept-triple lists; " << nthreads << " threads.\n"
                  << std::defaultfloat << std::flush;
        fourph_memory_reported = true;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        Profile prof_loc;
        std::vector<double> ret_loc(ntemp, 0.0);

        // Per-thread work arrays.
        std::vector<Cplx> exp_r1(cfc.nr1), exp_r3(cfc.nr3);
        std::vector<std::vector<Cplx>> exp_diff(nq_block, std::vector<Cplx>(cfc.ndiff));
        std::vector<Cplx> chi_k1(nq);
        // Entries of A outside the non-empty rows stay zero for the whole run.
        std::vector<std::vector<Cplx>> A(nq_block, std::vector<Cplx>(n3, Cplx(0.0, 0.0)));
        std::vector<Cplx> B(static_cast<size_t>(s1_chunk) * n2), C(static_cast<size_t>(s1_chunk) * ns * n), D;
        std::vector<Cplx> e1(ns * n), e2(ns * n), e3(ns * n);
        std::vector<std::vector<int>> tri_index(nq_block);    // (s1 * ns + s2) * ns + s3
        std::vector<std::vector<double>> tri_delta(nq_block); // 4 per kept triple
        std::vector<double> tri_v4;
        std::vector<double> pm13, pp13, pm23, pp23; // adaptive widths: velocity parts
        std::vector<double> pmax13, pmax23;         // per-band upper bounds of the widths
        if (adaptive) {
            pm13.resize(ns2);
            pp13.resize(ns2);
            pm23.resize(ns2);
            pp23.resize(ns2);
            pmax13.resize(ns);
            pmax23.resize(ns);
        }
        std::vector<int> order3(ns), stamp(ns, -1), candidates;
        std::vector<double> w3_sorted(ns);
        int pair_stamp = 0; // generation counter of stamp[], reset per quartet
        std::vector<const Cplx *> exp_diff_act(nq_block);
        std::vector<Cplx *> A_act(nq_block);
        std::vector<int> q_act(nq_block);

        // ---- Energy-conservation filter of one quartet: collects the band
        //      triples with a non-vanishing delta function.
        auto filter_quartet = [&](const int k1,
                                  const int k2,
                                  const int k3,
                                  std::vector<int> &tri_index_q,
                                  std::vector<double> &tri_delta_q) {
            tri_index_q.clear();
            tri_delta_q.clear();

            if (adaptive) {
                for (auto s1 = 0; s1 < ns; ++s1) {
                    const double *p1 = &proj[3 * (k1 * ns + s1)];
                    double vmax = 0.0;
                    for (auto s3 = 0; s3 < ns; ++s3) {
                        const double *p3 = &proj[3 * (k3 * ns + s3)];
                        double pm = 0.0, pp = 0.0;
                        for (auto u = 0; u < 3; ++u) {
                            pm += (p1[u] - p3[u]) * (p1[u] - p3[u]);
                            pp += (p1[u] + p3[u]) * (p1[u] + p3[u]);
                        }
                        pm13[s1 * ns + s3] = pm;
                        pp13[s1 * ns + s3] = pp;
                        vmax = std::max(vmax, std::max(pm, pp));
                    }
                    pmax13[s1] = vmax;
                }
                for (auto s2 = 0; s2 < ns; ++s2) {
                    const double *p2 = &proj[3 * (k2 * ns + s2)];
                    double vmax = 0.0;
                    for (auto s3 = 0; s3 < ns; ++s3) {
                        const double *p3 = &proj[3 * (k3 * ns + s3)];
                        double pm = 0.0, pp = 0.0;
                        for (auto u = 0; u < 3; ++u) {
                            pm += (p2[u] - p3[u]) * (p2[u] - p3[u]);
                            pp += (p2[u] + p3[u]) * (p2[u] + p3[u]);
                        }
                        pm23[s2 * ns + s3] = pm;
                        pp23[s2 * ns + s3] = pp;
                        vmax = std::max(vmax, std::max(pm, pp));
                    }
                    pmax23[s2] = vmax;
                }
            }

            // Frequencies of k3 in ascending order for the window search
            // (not needed for the Lorentzian, which keeps every triple).
            if (ismear != 0) {
                std::iota(order3.begin(), order3.end(), 0);
                std::sort(order3.begin(), order3.end(), [&](const int x, const int y) {
                    return eval_in[k3][x] < eval_in[k3][y];
                });
                for (auto j = 0; j < ns; ++j) w3_sorted[j] = eval_in[k3][order3[j]];
                std::fill(stamp.begin(), stamp.end(), -1);
                pair_stamp = 0;
            }

            for (auto s1 = 0; s1 < ns; ++s1) {
                const double w1 = eval_in[k1][s1];
                if (w1 < eps8) continue;
                for (auto s2 = 0; s2 < ns; ++s2) {
                    const double w2 = eval_in[k2][s2];
                    if (w2 < eps8) continue;

                    // Candidate s3: all bands (Lorentzian) or those inside the
                    // widest possible window around the seven sums.
                    candidates.clear();
                    if (ismear == 0) {
                        for (auto s3 = 0; s3 < ns; ++s3) candidates.push_back(s3);
                    } else {
                        double sigma2_max;
                        if (adaptive) {
                            sigma2_max = std::max(sigma_min2, adaptive_factor2 * (pmax13[s1] + pmax23[s2]));
                        } else {
                            sigma2_max = epsilon * epsilon;
                        }
                        const double width = 2.0 * std::sqrt(sigma2_max);
                        ++pair_stamp;
                        // Each sum is c + sign3 * w3 with c = omega +- w1 +- w2; it
                        // vanishes at w3 = -c / sign3, so the window maps onto
                        // w3 in (center - width, center + width).
                        for (auto ii = 0; ii < 7; ++ii) {
                            const double c = omega_in + signs_omega[ii][0] * w1 + signs_omega[ii][1] * w2;
                            const double center = -c / signs_omega[ii][2];
                            if (center + width <= 0.0) continue;
                            const auto lo = std::lower_bound(w3_sorted.begin(), w3_sorted.end(), center - width) -
                                            w3_sorted.begin();
                            const auto hi = std::upper_bound(w3_sorted.begin(), w3_sorted.end(), center + width) -
                                            w3_sorted.begin();
                            for (auto j = lo; j < hi; ++j) {
                                const int s3 = order3[j];
                                if (stamp[s3] != pair_stamp) {
                                    stamp[s3] = pair_stamp;
                                    candidates.push_back(s3);
                                }
                            }
                        }
                    }

                    for (const int s3: candidates) {
                        const double w3 = eval_in[k3][s3];
                        if (w3 < eps8) continue;

                        double sum_omega[7];
                        for (auto ii = 0; ii < 7; ++ii) {
                            sum_omega[ii] =
                                omega_in + signs_omega[ii][0] * w1 + signs_omega[ii][1] * w2 + signs_omega[ii][2] * w3;
                        }

                        double delta[4] = {0.0, 0.0, 0.0, 0.0};
                        bool keep = false;

                        if (ismear == 0) {
                            delta[0] = delta_lorentz(sum_omega[0], epsilon);
                            delta[1] = delta_lorentz(sum_omega[1], epsilon) - delta_lorentz(sum_omega[2], epsilon);
                            delta[2] = delta_lorentz(sum_omega[3], epsilon) - delta_lorentz(sum_omega[4], epsilon);
                            delta[3] = delta_lorentz(sum_omega[5], epsilon) - delta_lorentz(sum_omega[6], epsilon);
                            keep = true;
                        } else {
                            double sigma2[4], sig[4];
                            if (adaptive) {
                                const double parts0 = pm13[s1 * ns + s3];
                                const double parts1 = pm23[s2 * ns + s3];
                                const double parts2 = pp13[s1 * ns + s3];
                                const double parts3 = pp23[s2 * ns + s3];
                                sigma2[0] = std::max(sigma_min2, adaptive_factor2 * (parts0 + parts1));
                                sigma2[1] = std::max(sigma_min2, adaptive_factor2 * (parts2 + parts3));
                                sigma2[2] = std::max(sigma_min2, adaptive_factor2 * (parts2 + parts1));
                                sigma2[3] = std::max(sigma_min2, adaptive_factor2 * (parts0 + parts3));
                                for (auto ii = 0; ii < 4; ++ii) sig[ii] = -1.0; // computed lazily
                            } else {
                                for (auto ii = 0; ii < 4; ++ii) {
                                    sigma2[ii] = epsilon * epsilon;
                                    sig[ii] = epsilon;
                                }
                            }
                            for (auto ii = 0; ii < 7; ++ii) {
                                const int js = delta_of_sum[ii];
                                if (sum_omega[ii] * sum_omega[ii] < 4.0 * sigma2[js]) {
                                    if (sig[js] < 0.0) sig[js] = std::sqrt(sigma2[js]);
                                    delta[js] += sign_of_sum[ii] * delta_gauss(sum_omega[ii], sig[js]);
                                    keep = true;
                                }
                            }
                            if (keep) {
                                // Same acceptance rule as the reference implementation.
                                keep = delta[0] > 0.0 || std::abs(delta[1]) > 0.0 || std::abs(delta[2]) > 0.0 ||
                                       std::abs(delta[3]) > 0.0;
                            }
                        }

                        if (keep) {
                            tri_index_q.push_back((s1 * ns + s2) * ns + s3);
                            tri_delta_q.push_back(delta[0]);
                            tri_delta_q.push_back(delta[1]);
                            tri_delta_q.push_back(delta[2]);
                            tri_delta_q.push_back(delta[3]);
                        }
                    }
                }
            }
            return tri_index_q.size();
        };

        // ---- Eigenvector contractions and temperature sums of one quartet
        //      from its A array. The contractions run over chunks of s1; the
        //      kept triples are ordered by s1, so each chunk consumes a
        //      contiguous range of them.
        auto contract_quartet = [&](const int k1,
                                    const int k2,
                                    const int k3,
                                    const double multi,
                                    const Cplx *A_q,
                                    const std::vector<int> &tri_index_q,
                                    const std::vector<double> &tri_delta_q) {
            const size_t nkept = tri_index_q.size();
            double t0 = 0.0;
            if (profile) t0 = wall_seconds();

            for (auto s = 0; s < ns; ++s) {
                for (auto a = 0; a < n; ++a) {
                    const double m = invsqrt_mass[a / 3];
                    e1[s * n + a] = evec_in[k1][s][a] * m;
                    e2[s * n + a] = evec_in[k2][s][a] * m;
                    e3[s * n + a] = evec_in[k3][s][a] * m;
                }
            }

            const double inv_omega0 = 1.0 / omega0;
            tri_v4.resize(nkept);
            const bool dense_last = nkept * 4 > ns3;
            if (dense_last && D.size() < static_cast<size_t>(s1_chunk) * ns2)
                D.resize(static_cast<size_t>(s1_chunk) * ns2);

            ConstMapRow Am(A_q, n, static_cast<Eigen::Index>(n2));
            ConstMapRow E2m(e2.data(), ns, n);
            ConstMapRow E3m(e3.data(), ns, n);

            size_t it = 0; // running index into the kept triples
            for (auto s1a = 0; s1a < ns; s1a += s1_chunk) {
                const int nsc = std::min(s1_chunk, ns - s1a);

                // B (nsc x n^2) = E1[s1a:s1a+nsc] (nsc x n) * A (n x n^2)
                ConstMapRow E1m(e1.data() + static_cast<size_t>(s1a) * n, nsc, n);
                MapRow Bm(B.data(), nsc, static_cast<Eigen::Index>(n2));
                Bm.noalias() = E1m * Am;

                // C[s1] (ns x n) = E2 (ns x n) * B[s1] (n x n)
                for (auto j = 0; j < nsc; ++j) {
                    ConstMapRow Bs(B.data() + j * n2, n, n);
                    MapRow Cs(C.data() + static_cast<size_t>(j) * ns * n, ns, n);
                    Cs.noalias() = E2m * Bs;
                }

                const size_t it_end_bound = static_cast<size_t>(s1a + nsc) * ns2; // first ib of the next chunk
                if (dense_last) {
                    // D (nsc*ns x ns) = C (nsc*ns x n) * E3^T (n x ns)
                    MapRow Dm(D.data(), static_cast<Eigen::Index>(nsc) * ns, ns);
                    ConstMapRow Cm(C.data(), static_cast<Eigen::Index>(nsc) * ns, n);
                    Dm.noalias() = Cm * E3m.transpose();
                    for (; it < nkept && static_cast<size_t>(tri_index_q[it]) < it_end_bound; ++it) {
                        const int ib = tri_index_q[it];
                        const int s3 = ib % ns;
                        const int s2 = (ib / ns) % ns;
                        const int s1 = ib / ns2;
                        tri_v4[it] = std::norm(D[ib - static_cast<size_t>(s1a) * ns2]) * multi * inv_omega0 /
                                     (eval_in[k1][s1] * eval_in[k2][s2] * eval_in[k3][s3]);
                    }
                } else {
                    for (; it < nkept && static_cast<size_t>(tri_index_q[it]) < it_end_bound; ++it) {
                        const int ib = tri_index_q[it];
                        const int s3 = ib % ns;
                        const int s2 = (ib / ns) % ns;
                        const int s1 = ib / ns2;
                        const Cplx *crow = C.data() + (static_cast<size_t>(s1 - s1a) * ns + s2) * n;
                        const Cplx *erow = e3.data() + static_cast<size_t>(s3) * n;
                        double sre = 0.0, sim = 0.0;
                        for (auto d = 0; d < n; ++d) {
                            sre += crow[d].real() * erow[d].real() - crow[d].imag() * erow[d].imag();
                            sim += crow[d].real() * erow[d].imag() + crow[d].imag() * erow[d].real();
                        }
                        tri_v4[it] = (sre * sre + sim * sim) * multi * inv_omega0 /
                                     (eval_in[k1][s1] * eval_in[k2][s2] * eval_in[k3][s3]);
                    }
                }
            }
            if (profile) {
                prof_loc.t_contract += wall_seconds() - t0;
                t0 = wall_seconds();
            }

            // Temperature loop over the kept triples.
            for (unsigned int it = 0; it < ntemp; ++it) {
                const double *occ_t = &occ[it * nks];
                const double *occ1 = occ_t + k1 * ns;
                const double *occ2 = occ_t + k2 * ns;
                const double *occ3 = occ_t + k3 * ns;
                double sum = 0.0;
                for (size_t jt = 0; jt < nkept; ++jt) {
                    const int ib = tri_index_q[jt];
                    const int s3 = ib % ns;
                    const int s2 = (ib / ns) % ns;
                    const int s1 = ib / ns2;
                    const double f1 = occ1[s1];
                    const double f2 = occ2[s2];
                    const double f3 = occ3[s3];
                    const double f12 = f1 * f2, f23 = f2 * f3, f13 = f1 * f3;
                    double n1 = f12 + f23 + f13 + f1 + f2 + f3;
                    if (!classical) n1 += 1.0;
                    const double *dl = &tri_delta_q[4 * jt];
                    sum += tri_v4[jt] * (n1 * dl[0] + (f13 + f23 - f12 + f3) * dl[1] + (f12 + f13 - f23 + f1) * dl[2] +
                                         (f23 + f12 - f13 + f2) * dl[3]);
                }
                ret_loc[it] += sum;
            }
            if (profile) prof_loc.t_accumulate += wall_seconds() - t0;
        };

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (int irun = 0; irun < nrun; ++irun) {

            const int k1 = quartet_k[3 * run_start[irun]];

            // ---- Stage 1 of the Fourier sum: apply the k1.R1 and (k - k1).R3
            //      phases for this k1.
            double t0 = 0.0;
            if (profile) t0 = wall_seconds();
            int qkk1[3];
            for (auto icrd = 0; icrd < 3; ++icrd) qkk1[icrd] = qint[3 * knum + icrd] - qint[3 * k1 + icrd];
            for (auto ir = 0; ir < cfc.nr1; ++ir) {
                exp_r1[ir] = phase_factor(&qint[3 * k1], &cfc.rvec1[3 * ir], nkgrid, exp_table);
            }
            for (auto ir = 0; ir < cfc.nr3; ++ir) {
                exp_r3[ir] = phase_factor(qkk1, &cfc.rvec3[3 * ir], nkgrid, exp_table);
            }
            for (auto iq = 0; iq < nq; ++iq) {
                double sre = 0.0, sim = 0.0;
                for (auto ip = cfc.q_ptr[iq]; ip < cfc.q_ptr[iq + 1]; ++ip) {
                    const auto &ph = phitilde[ip];
                    const auto ex = exp_r1[cfc.pair_r1[ip]] * exp_r3[cfc.pair_r3[ip]];
                    sre += ph.real() * ex.real() - ph.imag() * ex.imag();
                    sim += ph.real() * ex.imag() + ph.imag() * ex.real();
                }
                chi_k1[iq] = Cplx(sre, sim);
            }
            if (profile) prof_loc.t_fourier += wall_seconds() - t0;

            for (int iq0 = run_start[irun]; iq0 < run_start[irun + 1]; iq0 += nq_block) {

                const int nq = std::min(nq_block, run_start[irun + 1] - iq0);

                // ---- 1. Filter every quartet of the block; only those with
                //         surviving triples take part in the Fourier stage.
                if (profile) t0 = wall_seconds();
                int nact = 0;
                for (auto q = 0; q < nq; ++q) {
                    const int k2 = quartet_k[3 * (iq0 + q) + 1];
                    const int k3 = quartet_k[3 * (iq0 + q) + 2];
                    const auto nkept = filter_quartet(k1, k2, k3, tri_index[q], tri_delta[q]);
                    if (profile) {
                        prof_loc.n_quartet += 1;
                        prof_loc.n_triplet += static_cast<long long>(ns3);
                        prof_loc.n_triplet_kept += static_cast<long long>(nkept);
                        if (nkept == 0) prof_loc.n_quartet_skipped += 1;
                    }
                    if (nkept == 0) continue;
                    for (auto ir = 0; ir < cfc.ndiff; ++ir) {
                        exp_diff[q][ir] = phase_factor(&qint[3 * k2], &cfc.dvec[3 * ir], nkgrid, exp_table);
                    }
                    q_act[nact] = q;
                    exp_diff_act[nact] = exp_diff[q].data();
                    A_act[nact] = A[q].data();
                    ++nact;
                }
                if (profile) {
                    prof_loc.t_filter += wall_seconds() - t0;
                    t0 = wall_seconds();
                }
                if (nact == 0) continue;

                // ---- 2. Stage 2 of the Fourier sum for the active quartets.
                switch (nact) {
                case 4:
                    fourier_stage2_block<4>(nrow,
                                            cfc.row_ptr.data(),
                                            cfc.row_bcd.data(),
                                            cfc.q_diff.data(),
                                            chi_k1.data(),
                                            exp_diff_act.data(),
                                            A_act.data());
                    break;
                case 3:
                    fourier_stage2_block<3>(nrow,
                                            cfc.row_ptr.data(),
                                            cfc.row_bcd.data(),
                                            cfc.q_diff.data(),
                                            chi_k1.data(),
                                            exp_diff_act.data(),
                                            A_act.data());
                    break;
                case 2:
                    fourier_stage2_block<2>(nrow,
                                            cfc.row_ptr.data(),
                                            cfc.row_bcd.data(),
                                            cfc.q_diff.data(),
                                            chi_k1.data(),
                                            exp_diff_act.data(),
                                            A_act.data());
                    break;
                default:
                    fourier_stage2_block<1>(nrow,
                                            cfc.row_ptr.data(),
                                            cfc.row_bcd.data(),
                                            cfc.q_diff.data(),
                                            chi_k1.data(),
                                            exp_diff_act.data(),
                                            A_act.data());
                    break;
                }
                if (profile) prof_loc.t_fourier += wall_seconds() - t0;

                // ---- 3./4. Contractions and temperature sums per quartet.
                for (auto ia = 0; ia < nact; ++ia) {
                    const int q = q_act[ia];
                    const int iq = iq0 + q;
                    contract_quartet(k1,
                                     quartet_k[3 * iq + 1],
                                     quartet_k[3 * iq + 2],
                                     quartet_multi[iq],
                                     A[q].data(),
                                     tri_index[q],
                                     tri_delta[q]);
                }
            }
        }

#ifdef _OPENMP
#pragma omp critical
#endif
        {
            for (unsigned int it = 0; it < ntemp; ++it) ret[it] += ret_loc[it];
            if (profile) prof.add(prof_loc);
        }
    }

    for (unsigned int i = 0; i < ntemp; ++i) {
        ret[i] *= pi * std::pow(0.5, 5) / (3.0 * static_cast<double>(nk) * static_cast<double>(nk));
    }

    if (profile && mympi->my_rank == 0) {
        const double t_total = wall_seconds() - t_start;
        std::cout << std::fixed << std::setprecision(3) << "\n [4ph profile] ik=" << ik_in << " is=" << is_in
                  << " quartets=" << prof.n_quartet << " (skipped " << prof.n_quartet_skipped << ", k1 runs " << nrun
                  << ")"
                  << " kept triples=" << prof.n_triplet_kept << "/" << prof.n_triplet << " | wall " << t_total
                  << " s; thread-sum: prep " << t_mode_prep << " filter " << prof.t_filter << " fourier "
                  << prof.t_fourier << " contract " << prof.t_contract << " accumulate " << prof.t_accumulate
                  << std::flush;
    }
}
