/*
 quartic_real_space.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.

 The quartic (and cubic) couplings of the SCPH theory as contractions over the
 stored real-space IFCs instead of the reciprocal-space tensor
 V4[(k, k')][a, b][c, d] (anphon/FINITE_Q_HESSIAN_PLAN.md, steps A-D).

 The IFCs are folded onto the cells of the dense k-mesh: every stored entry,
 divided by the square roots of its masses, is summed into the slot of its
 atom indices and of the cell offsets of its other legs relative to the first,
 reduced modulo the mesh. The reciprocal-space couplings are built from the
 same entries with phases e^{2 pi i k.R} of mesh vectors, which do not see
 the reduction, so every contraction below is exact for the loaded IFCs, also
 when the output is wanted at the k-points of a coarser mesh dividing the
 dense one (KMESH_INTERPOLATE vs KMESH_SCPH).

 RealSpaceV4 replaces the V4 tensor in the SCP loop (V4_REAL_SPACE = 1): the
 contraction F(a,b) = sum_cd V4(a,b;c,d) D(c,d) for the D matrices of the
 dense mesh, the q0 sweep of the structural optimization, and the on-site
 diagonal elements. It runs on one process (OpenMP); its memory is the folded
 FC4 list instead of nk_irred x nk_dense x ns^4 complex numbers.
*/

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <tuple>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "error.h"
#include "fcs_phonon.h"

namespace PHON_NS::quartic_rs
{
using cplx = std::complex<double>;

// Cell of a lattice offset (integer lattice coordinates) on a k-mesh of
// n1 x n2 x n3 points, reduced modulo the mesh.
inline bool in_parallel()
{
#ifdef _OPENMP
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}

inline int mesh_cell(const std::array<int, 3> &nki, const Eigen::Vector3d &v)
{
    int r = 0;
    for (int d = 0; d < 3; ++d) {
        auto n = static_cast<int>(std::lround(v[d])) % nki[d];
        if (n < 0) n += nki[d];
        r = r * nki[d] + n;
    }
    return r;
}

// Mass-weighted IFCs of order n folded onto the cells of a k-mesh (steps A-C
// of FINITE_Q_HESSIAN_PLAN.md): every stored entry, divided by the square
// roots of its masses, goes to the slot of its atom indices and of the cell
// offsets of legs 2..n relative to the first leg (reduced modulo the mesh);
// entries that land in one slot add up. This is the element-wise sum from
// which the reciprocal-space vertices are built, since the phases of the mesh
// do not see the reduction. slot = (index[0..n-1], cell[0..n-2]).
template <int N>
struct FoldedIfcs
{
    unsigned int ns = 0;
    unsigned int nk = 0;
    std::array<int, 3> nki{};
    std::vector<std::array<unsigned int, 2 * N - 1>> slot;
    std::vector<double> weight;

    FoldedIfcs(const std::vector<FcsArrayWithCell> &fcs, const std::vector<double> &invsqrt_mass,
               const unsigned int ns_in, const unsigned int *nk_in) : ns(ns_in)
    {
        nki = {static_cast<int>(nk_in[0]), static_cast<int>(nk_in[1]), static_cast<int>(nk_in[2])};
        nk = static_cast<unsigned int>(nki[0] * nki[1] * nki[2]);
        if (N * std::log2(static_cast<double>(ns)) + (N - 1) * std::log2(static_cast<double>(nk)) >= 63.0) {
            exit("FoldedIfcs", "the IFC slots of this cell and k-mesh do not fit a 64-bit key.");
        }
        std::unordered_map<unsigned long long, double> sum;
        sum.reserve(fcs.size());
        for (const auto &e: fcs) {
            unsigned long long key = 0;
            double w = e.fcs_val;
            for (int n = 0; n < N; ++n) {
                key = key * ns + e.pairs[n].index;
                w *= invsqrt_mass[e.pairs[n].index / 3];
            }
            for (int n = 0; n < N - 1; ++n) key = key * nk + mesh_cell(nki, e.relvecs[n]);
            sum[key] += w;
        }
        slot.reserve(sum.size());
        weight.reserve(sum.size());
        for (const auto &[key, w]: sum) {
            slot.push_back(decode(key));
            weight.push_back(w);
        }
    }

    // Several weight channels on one slot set (STRAIN_FC5: the 9 strain derivatives
    // dPhi_N/du_{mu nu} of DerivativeIFC::compute_dV_dumn_all_real_space). A group g
    // has legs g.pairs[0..N-1], cells g.relvecs[0..N-2] and one value per channel,
    // g.values[c]; channel[c][slot] holds the folded sum. weight (the values the
    // contractions use) starts at zero, see RealSpaceV4::set_channel_weights.
    std::vector<std::vector<double>> channel;

    template <class Group>
    FoldedIfcs(const std::vector<Group> &groups, const std::size_t nchannel, const std::vector<double> &invsqrt_mass,
               const unsigned int ns_in, const unsigned int *nk_in) : ns(ns_in)
    {
        nki = {static_cast<int>(nk_in[0]), static_cast<int>(nk_in[1]), static_cast<int>(nk_in[2])};
        nk = static_cast<unsigned int>(nki[0] * nki[1] * nki[2]);
        if (N * std::log2(static_cast<double>(ns)) + (N - 1) * std::log2(static_cast<double>(nk)) >= 63.0) {
            exit("FoldedIfcs", "the IFC slots of this cell and k-mesh do not fit a 64-bit key.");
        }
        std::unordered_map<unsigned long long, std::size_t> index;
        index.reserve(groups.size());
        channel.assign(nchannel, {});
        for (const auto &g: groups) {
            unsigned long long key = 0;
            double w = 1.0;
            for (int n = 0; n < N; ++n) {
                key = key * ns + g.pairs[n].index;
                w *= invsqrt_mass[g.pairs[n].index / 3];
            }
            for (int n = 0; n < N - 1; ++n) key = key * nk + mesh_cell(nki, g.relvecs[n]);
            const auto [it, added] = index.emplace(key, slot.size());
            if (added) {
                slot.push_back(decode(key));
                for (auto &ch: channel) ch.push_back(0.0);
            }
            for (std::size_t c = 0; c < nchannel; ++c) channel[c][it->second] += w * g.values[c];
        }
        weight.assign(slot.size(), 0.0);
    }

    std::array<unsigned int, 2 * N - 1> decode(unsigned long long key) const
    {
        std::array<unsigned int, 2 * N - 1> id{};
        for (int d = 2 * N - 2; d >= 0; --d) {
            const unsigned long long base = d >= N ? nk : ns;
            id[d] = static_cast<unsigned int>(key % base);
            key /= base;
        }
        return id;
    }
};

// The quartic ladder in real space with momentum transfer Q:
//   z_ij(R_j) = (1/4) sum W_ijlm(R_j, R_l, R_m) e^{iQ.R_l} x_lm(R_m - R_l)
// (Cartesian, mass weighted, unconjugated; x and z hold one ns x ns matrix per
// cell of the mesh; qphase[r] = e^{2 pi i Q.R_r}, nullptr for Q = 0).
struct QuarticMesh: FoldedIfcs<4>
{
    std::vector<unsigned int> cell_lm; // cell(R_m - R_l) per slot
    std::vector<size_t> row_begin;     // slots of output row (cell(R_j), i) = [row_begin[r], row_begin[r + 1])

    // The slots are sorted by their output row, so that the contraction below is
    // row-parallel with a fixed summation order (independent of the thread count)
    // and needs no per-thread accumulators.
    QuarticMesh(const std::vector<FcsArrayWithCell> &fcs, const std::vector<double> &invsqrt_mass,
                const unsigned int ns_in, const unsigned int *nk_in) : FoldedIfcs<4>(fcs, invsqrt_mass, ns_in, nk_in)
    {
        finish();
    }

    template <class Group>
    QuarticMesh(const std::vector<Group> &groups, const std::size_t nchannel, const std::vector<double> &invsqrt_mass,
                const unsigned int ns_in, const unsigned int *nk_in) :
        FoldedIfcs<4>(groups, nchannel, invsqrt_mass, ns_in, nk_in)
    {
        finish();
    }

    void finish()
    {
        std::vector<size_t> order(slot.size());
        for (size_t e = 0; e < order.size(); ++e) order[e] = e;
        std::sort(order.begin(), order.end(), [&](const size_t a, const size_t b) {
            const auto &sa = slot[a], &sb = slot[b];
            return std::make_tuple(sa[4], sa[0], sa[1], sa[2], sa[3], sa[5], sa[6]) <
                   std::make_tuple(sb[4], sb[0], sb[1], sb[2], sb[3], sb[5], sb[6]);
        });
        decltype(slot) slot_sorted(slot.size());
        std::vector<double> weight_sorted(slot.size());
        for (size_t e = 0; e < order.size(); ++e) {
            slot_sorted[e] = slot[order[e]];
            weight_sorted[e] = weight[order[e]];
        }
        slot.swap(slot_sorted);
        weight.swap(weight_sorted);
        for (auto &ch: channel) {
            for (size_t e = 0; e < order.size(); ++e) weight_sorted[e] = ch[order[e]];
            ch.swap(weight_sorted);
        }
        cell_lm.reserve(slot.size());
        row_begin.assign(static_cast<size_t>(nk) * ns + 1, slot.size());
        for (size_t e = slot.size(); e-- > 0;) {
            row_begin[static_cast<size_t>(slot[e][4]) * ns + slot[e][0]] = e;
        }
        for (size_t r = row_begin.size() - 1; r-- > 0;) row_begin[r] = std::min(row_begin[r], row_begin[r + 1]);
        for (const auto &id: slot) {
            cell_lm.push_back(static_cast<unsigned int>(mesh_cell(nki, cell_vector(id[6]) - cell_vector(id[5]))));
        }
    }

    // w: the weights to use instead of weight (one channel of a multi-channel mesh)
    void apply(const std::vector<Eigen::MatrixXcd> &x, std::vector<Eigen::MatrixXcd> &z,
               const std::vector<cplx> *qphase = nullptr, const std::vector<double> *w_in = nullptr) const
    {
        const auto &wv = w_in ? *w_in : weight;
        z.assign(nk, Eigen::MatrixXcd::Zero(ns, ns));
        const auto nrow = static_cast<long>(row_begin.size() - 1);
#pragma omp parallel for schedule(dynamic, 16) if (!in_parallel())
        for (long r = 0; r < nrow; ++r) {
            const auto rj = static_cast<unsigned int>(r / ns), i = static_cast<unsigned int>(r % ns);
            for (size_t e = row_begin[r]; e < row_begin[r + 1]; ++e) {
                const auto &id = slot[e];
                // slot: i, j, l, m, cell(R_j), cell(R_l), cell(R_m)
                const cplx w = qphase ? wv[e] * (*qphase)[id[5]] : cplx(wv[e], 0.0);
                z[rj](i, id[1]) += w * x[cell_lm[e]](id[2], id[3]);
            }
        }
        for (auto &m: z) m *= 0.25;
    }

    Eigen::Vector3d cell_vector(const unsigned int r) const
    {
        Eigen::Vector3d v;
        auto rr = static_cast<int>(r);
        for (int d = 2; d >= 0; --d) {
            v[d] = rr % nki[d];
            rr /= nki[d];
        }
        return v;
    }
};

// The cubic source of a displacement pattern u(R) = eps e^{iQ.R} (mass
// weighted): y_ij(R_j) = sum W_ijl(R_j, R_l) eps_l e^{iQ.R_l}, the change of
// the (mass-weighted) harmonic IFCs Phi(i 0, j R_j) per unit amplitude.
struct CubicMesh: FoldedIfcs<3>
{
    using FoldedIfcs<3>::FoldedIfcs;

    void source(const Eigen::VectorXcd &eps, const std::vector<cplx> &qphase, std::vector<Eigen::MatrixXcd> &y) const
    {
        y.assign(nk, Eigen::MatrixXcd::Zero(ns, ns));
        for (size_t e = 0; e < weight.size(); ++e) {
            const auto &id = slot[e]; // i, j, l, cell(R_j), cell(R_l)
            y[id[3]](id[0], id[1]) += weight[e] * eps[id[2]] * qphase[id[4]];
        }
    }
};

// V4 on the dense mesh from the folded FC4 (see the header comment). Harmonic
// eigenvectors evec[k][s][a] (mode s, Cartesian a) of the dense mesh; the
// acoustic modes at Gamma are dropped on every Gamma leg, as the V4 builder does.
class RealSpaceV4
{
public:
    RealSpaceV4(const std::vector<FcsArrayWithCell> &fc4, const std::vector<double> &invsqrt_mass, unsigned int ns,
                const unsigned int *nk_dense_i, const double *const *xk_dense, const cplx *const *const *evec,
                const std::vector<bool> &acoustic_gamma, unsigned int jk_gamma_dense,
                const std::vector<unsigned int> &knum_of_irred, bool offdiag);

    // From a prebuilt (e.g. multi-channel) mesh; with_diag = false skips the on-site
    // diagonal (v4_diag() is then empty).
    RealSpaceV4(QuarticMesh &&mesh, unsigned int ns, const unsigned int *nk_dense_i, const double *const *xk_dense,
                const cplx *const *const *evec, const std::vector<bool> &acoustic_gamma, unsigned int jk_gamma_dense,
                const std::vector<unsigned int> &knum_of_irred, bool offdiag, bool with_diag);

    // Multi-channel mesh: weight = sum_c coef[c] channel[c]; every contraction below
    // then uses these weights.
    void set_channel_weights(const double *coef);

    // F[ik_irred](a,b) += sum_jk sum_cd V4[(ik_irred,jk)][a,b][c,d] D_jk(c,d), dvec[jk*ns2 + c*ns + d],
    // on the lower triangle (offdiag) or the diagonal (D restricted to its diagonal), as
    // V4Service::fmat.
    void fmat(const cplx *dvec, cplx ***fmat_inout) const;

    // As V4Service::fmat_batch: fout[m*nk_irred*ns2 + ik_irred*ns2 + a*ns + b] = full contraction.
    void fmat_batch(const cplx *dmat, std::size_t nrhs, cplx *fout) const;

    // As V4Service::q0_sweep (q0_contraction.h):
    //   v3_renorm[jk][b][c,d] = v3_with_umn[jk][b][c,d] + sum_a V4[(g,jk)][a,b][c,d] q0[a]
    //   q4_q0[ik][a][b]       = sum_cd V4[(ik,jg)][a,b][c,d] q0[c] q0[d]
    // accumulate: add the contributions to v3_renorm and q4_q0 instead (v3_with_umn unused).
    void q0_sweep(const double *q0, const cplx *const *const *v3_with_umn, cplx ***v3_renorm, cplx ***q4_q0,
                  bool accumulate = false) const;

    // V4[(ik_irred, knum)][a,a][a,a] (real part), as V4Service::v4_diag
    const double *const *v4_diag() const
    {
        return diag_ptr_.data();
    }

    std::size_t folded_entries() const
    {
        return quartic_.weight.size();
    }

    // F_k for the D blocks of the dense mesh, at the dense k-points kout (full matrices, row-major)
    void contract(const cplx *d, const std::vector<unsigned int> &kout, cplx *f) const;

    // As contract, once per channel c of the mesh, with the channel weights instead of the
    // current ones: f + c * kout.size() * ns^2 (one D, several vertices).
    void contract_channels(const cplx *d, const std::vector<unsigned int> &kout, cplx *f) const;

private:
    void init(const double *const *xk_dense, const cplx *const *const *evec, bool with_diag);
    void contract_impl(const cplx *d, const std::vector<unsigned int> &kout,
                       const std::vector<const std::vector<double> *> &weights, cplx *f) const;

    unsigned int ns_, ns2_, nk_, jg_;
    QuarticMesh quartic_;
    std::vector<Eigen::MatrixXcd> U_;
    std::vector<std::vector<cplx>> kphase_;
    std::vector<bool> acoustic_;
    std::vector<unsigned int> knum_of_irred_;
    bool offdiag_;
    std::vector<std::vector<double>> diag_;
    std::vector<const double *> diag_ptr_;
};

} // namespace PHON_NS::quartic_rs
