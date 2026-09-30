/*
 quartic_real_space.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.

 RealSpaceV4: see quartic_real_space.h. In terms of the folded FC4 W (mass
 weighted) and the harmonic eigenvectors U_k of the dense mesh,

   V4[(k,k')][a,b][c,d] = (1/4N) sum_slots W conj(U_k(i,a)) U_k(j,b) U_k'(l,c) conj(U_k'(m,d))
                          x e^{2 pi i [k.R_j - k'.(R_m - R_l)]},

 i the leg in the home cell, R_j, R_l, R_m the cells of the other legs; the
 contractions evaluate this without forming it.
*/

#include "quartic_real_space.h"
#include <algorithm>

namespace PHON_NS::quartic_rs
{

RealSpaceV4::RealSpaceV4(const std::vector<FcsArrayWithCell> &fc4, const std::vector<double> &invsqrt_mass,
                         const unsigned int ns, const unsigned int *nk_dense_i, const double *const *xk_dense,
                         const cplx *const *const *evec, const std::vector<bool> &acoustic_gamma,
                         const unsigned int jk_gamma_dense, const std::vector<unsigned int> &knum_of_irred,
                         const bool offdiag) :
    ns_(ns), ns2_(ns * ns), nk_(nk_dense_i[0] * nk_dense_i[1] * nk_dense_i[2]), jg_(jk_gamma_dense),
    quartic_(fc4, invsqrt_mass, ns, nk_dense_i), acoustic_(acoustic_gamma), knum_of_irred_(knum_of_irred),
    offdiag_(offdiag)
{
    init(xk_dense, evec, true);
}

RealSpaceV4::RealSpaceV4(QuarticMesh &&mesh, const unsigned int ns, const unsigned int *nk_dense_i,
                         const double *const *xk_dense, const cplx *const *const *evec,
                         const std::vector<bool> &acoustic_gamma, const unsigned int jk_gamma_dense,
                         const std::vector<unsigned int> &knum_of_irred, const bool offdiag, const bool with_diag) :
    ns_(ns), ns2_(ns * ns), nk_(nk_dense_i[0] * nk_dense_i[1] * nk_dense_i[2]), jg_(jk_gamma_dense),
    quartic_(std::move(mesh)), acoustic_(acoustic_gamma), knum_of_irred_(knum_of_irred), offdiag_(offdiag)
{
    init(xk_dense, evec, with_diag);
}

void RealSpaceV4::set_channel_weights(const double *coef)
{
    auto &w = quartic_.weight;
    std::fill(w.begin(), w.end(), 0.0);
    for (std::size_t c = 0; c < quartic_.channel.size(); ++c) {
        if (coef[c] == 0.0) continue;
        const auto &ch = quartic_.channel[c];
        for (std::size_t e = 0; e < w.size(); ++e) w[e] += coef[c] * ch[e];
    }
}

void RealSpaceV4::init(const double *const *xk_dense, const cplx *const *const *evec, const bool with_diag)
{
    using namespace Eigen;
    U_.assign(nk_, MatrixXcd(ns_, ns_));
    kphase_.assign(nk_, std::vector<cplx>(nk_));
    for (unsigned int ik = 0; ik < nk_; ++ik) {
        for (unsigned int s = 0; s < ns_; ++s) {
            for (unsigned int a = 0; a < ns_; ++a) U_[ik](a, s) = evec[ik][s][a];
        }
        for (unsigned int r = 0; r < nk_; ++r) {
            const auto R = quartic_.cell_vector(r);
            double arg = 0.0;
            for (int d = 0; d < 3; ++d) arg += xk_dense[ik][d] * R[d];
            kphase_[ik][r] = std::exp(cplx(0.0, 2.0 * M_PI * arg));
        }
    }

    if (!with_diag) return;
    // on-site diagonal V4[(k,k)][a,a][a,a], k the irreducible points
    const auto nirr = knum_of_irred_.size();
    diag_.assign(nirr, std::vector<double>(ns_, 0.0));
    const auto n = static_cast<long>(quartic_.weight.size());
    for (size_t ir = 0; ir < nirr; ++ir) {
        const auto k = knum_of_irred_[ir];
#pragma omp parallel for schedule(dynamic, 1)
        for (long a_l = 0; a_l < static_cast<long>(ns_); ++a_l) {
            const auto a = static_cast<Index>(a_l);
            if (k == jg_ && acoustic_[a]) continue;
            const auto u = U_[k].col(a);
            cplx sum(0.0, 0.0);
            for (long e = 0; e < n; ++e) {
                const auto &id = quartic_.slot[e];
                sum += quartic_.weight[e] * std::conj(u(id[0])) * u(id[1]) * u(id[2]) * std::conj(u(id[3])) *
                       kphase_[k][id[4]] * std::conj(kphase_[k][quartic_.cell_lm[e]]);
            }
            diag_[ir][a] = sum.real() / (4.0 * nk_);
        }
    }
    for (auto &d: diag_) diag_ptr_.push_back(d.data());
}

void RealSpaceV4::contract(const cplx *d, const std::vector<unsigned int> &kout, cplx *f) const
{
    contract_impl(d, kout, {nullptr}, f);
}

void RealSpaceV4::contract_channels(const cplx *d, const std::vector<unsigned int> &kout, cplx *f) const
{
    std::vector<const std::vector<double> *> w;
    for (const auto &ch: quartic_.channel) w.push_back(&ch);
    contract_impl(d, kout, w, f);
}

void RealSpaceV4::contract_impl(const cplx *d, const std::vector<unsigned int> &kout,
                                const std::vector<const std::vector<double> *> &weights, cplx *f) const
{
    using namespace Eigen;
    using MatrixXcdRow = Matrix<cplx, Dynamic, Dynamic, RowMajor>;
    const auto drop = [&](MatrixXcd &M, const unsigned int k) {
        if (k != jg_) return;
        for (unsigned int a = 0; a < ns_; ++a) {
            if (!acoustic_[a]) continue;
            M.row(a).setZero();
            M.col(a).setZero();
        }
    };
    const bool nested = in_parallel();
    std::vector<MatrixXcd> X(nk_);
#pragma omp parallel for schedule(dynamic, 1) if (!nested)
    for (long ik = 0; ik < static_cast<long>(nk_); ++ik) {
        MatrixXcd D = Map<const MatrixXcdRow>(d + static_cast<size_t>(ik) * ns2_, ns_, ns_);
        drop(D, static_cast<unsigned int>(ik));
        X[ik] = U_[ik] * D * U_[ik].adjoint();
    }
    std::vector<MatrixXcd> x(nk_), z;
#pragma omp parallel for schedule(static) if (!nested)
    for (long r = 0; r < static_cast<long>(nk_); ++r) {
        x[r] = MatrixXcd::Zero(ns_, ns_);
        for (unsigned int ik = 0; ik < nk_; ++ik) x[r] += std::conj(kphase_[ik][r]) * X[ik];
        x[r] /= static_cast<double>(nk_);
    }
    for (std::size_t c = 0; c < weights.size(); ++c) {
        quartic_.apply(x, z, nullptr, weights[c]);
        cplx *fc = f + c * kout.size() * ns2_;
#pragma omp parallel for schedule(dynamic, 1) if (!nested)
        for (long io = 0; io < static_cast<long>(kout.size()); ++io) {
            const auto k = kout[io];
            MatrixXcd Z = MatrixXcd::Zero(ns_, ns_);
            for (unsigned int r = 0; r < nk_; ++r) Z += kphase_[k][r] * z[r];
            MatrixXcd F = U_[k].adjoint() * Z * U_[k];
            drop(F, k);
            Map<MatrixXcdRow>(fc + static_cast<size_t>(io) * ns2_, ns_, ns_) = F;
        }
    }
}

void RealSpaceV4::fmat(const cplx *dvec, cplx ***fmat_inout) const
{
    const auto nirr = knum_of_irred_.size();
    std::vector<cplx> f(nirr * ns2_);
    if (offdiag_) {
        contract(dvec, knum_of_irred_, f.data());
    } else {
        // the diagonal-only tensor: only the diagonal of D enters
        std::vector<cplx> dd(static_cast<size_t>(nk_) * ns2_, cplx(0.0, 0.0));
        for (unsigned int ik = 0; ik < nk_; ++ik) {
            for (unsigned int c = 0; c < ns_; ++c) {
                const auto i = static_cast<size_t>(ik) * ns2_ + c * (ns_ + 1);
                dd[i] = dvec[i];
            }
        }
        contract(dd.data(), knum_of_irred_, f.data());
    }
    for (size_t ir = 0; ir < nirr; ++ir) {
        for (unsigned int a = 0; a < ns_; ++a) {
            if (!offdiag_) {
                fmat_inout[ir][a][a] += f[ir * ns2_ + a * (ns_ + 1)];
                continue;
            }
            for (unsigned int b = 0; b <= a; ++b) fmat_inout[ir][a][b] += f[ir * ns2_ + a * ns_ + b];
        }
    }
}

void RealSpaceV4::fmat_batch(const cplx *dmat, const std::size_t nrhs, cplx *fout) const
{
    const auto nirr = knum_of_irred_.size();
    for (std::size_t m = 0; m < nrhs; ++m) {
        contract(dmat + m * nk_ * ns2_, knum_of_irred_, fout + m * nirr * ns2_);
    }
}

void RealSpaceV4::q0_sweep(const double *q0, const cplx *const *const *v3_with_umn, cplx ***v3_renorm, cplx ***q4_q0,
                           const bool accumulate) const
{
    using namespace Eigen;
    const auto nirr = knum_of_irred_.size();

    // q4_q0: the contraction of D = q0 q0^T at Gamma
    {
        std::vector<cplx> d(static_cast<size_t>(nk_) * ns2_, cplx(0.0, 0.0)), f(nirr * ns2_);
        for (unsigned int c = 0; c < ns_; ++c) {
            for (unsigned int e = 0; e < ns_; ++e) d[jg_ * ns2_ + c * ns_ + e] = q0[c] * q0[e];
        }
        contract(d.data(), knum_of_irred_, f.data());
        for (size_t ir = 0; ir < nirr; ++ir) {
            for (unsigned int a = 0; a < ns_; ++a) {
                for (unsigned int b = 0; b < ns_; ++b) {
                    if (accumulate) {
                        q4_q0[ir][a][b] += f[ir * ns2_ + a * ns_ + b];
                    } else {
                        q4_q0[ir][a][b] = f[ir * ns2_ + a * ns_ + b];
                    }
                }
            }
        }
    }

    // v3_renorm: the home leg contracted with conj(eps), eps = sum_a U_G(:,a) q0[a] (no
    // acoustic a), folded to (j, l, m, cell(R_m - R_l)); then, per free Gamma mode b,
    //   Y_b(r)_lm = sum W' U_G(j,b),  v3[jk][b](c,d) = (1/4N) [U^T Yhat_b(jk) conj(U)](c,d),
    //   Yhat_b(k) = sum_r e^{-ik.R_r} Y_b(r).
    VectorXcd eps = VectorXcd::Zero(ns_);
    for (unsigned int a = 0; a < ns_; ++a) {
        if (!acoustic_[a]) eps += U_[jg_].col(a) * q0[a];
    }
    std::unordered_map<unsigned long long, cplx> sum;
    for (size_t e = 0; e < quartic_.weight.size(); ++e) {
        const auto &id = quartic_.slot[e];
        const cplx w = quartic_.weight[e] * std::conj(eps(id[0]));
        if (w == cplx(0.0, 0.0)) continue;
        const unsigned long long key =
            ((static_cast<unsigned long long>(id[1]) * ns_ + id[2]) * ns_ + id[3]) * nk_ + quartic_.cell_lm[e];
        sum[key] += w;
    }
    std::vector<std::array<unsigned int, 4>> slot3;
    std::vector<cplx> w3;
    slot3.reserve(sum.size());
    w3.reserve(sum.size());
    for (const auto &[key, w]: sum) {
        auto r = key;
        const auto cell = static_cast<unsigned int>(r % nk_);
        r /= nk_;
        const auto m = static_cast<unsigned int>(r % ns_);
        r /= ns_;
        const auto l = static_cast<unsigned int>(r % ns_);
        r /= ns_;
        slot3.push_back({static_cast<unsigned int>(r), l, m, cell});
        w3.push_back(w);
    }
    const double factor = 1.0 / (4.0 * nk_);
    // one Y (nk ns^2) per concurrent mode b: at most ~2 GB of them
    int nconc = 1;
#ifdef _OPENMP
    nconc = omp_get_max_threads();
#endif
    nconc = std::max(1, std::min(nconc, static_cast<int>(2.0e9 / (16.0 * nk_ * ns2_))));
#pragma omp parallel for schedule(dynamic, 1) num_threads(nconc)
    for (long b_l = 0; b_l < static_cast<long>(ns_); ++b_l) {
        const auto b = static_cast<unsigned int>(b_l);
        std::vector<MatrixXcd> Y(nk_, MatrixXcd::Zero(ns_, ns_));
        if (!acoustic_[b]) {
            for (size_t e = 0; e < w3.size(); ++e) {
                const auto &id = slot3[e];
                Y[id[3]](id[1], id[2]) += w3[e] * U_[jg_](id[0], b);
            }
        }
        for (unsigned int jk = 0; jk < nk_; ++jk) {
            MatrixXcd Yk = MatrixXcd::Zero(ns_, ns_);
            if (!acoustic_[b]) {
                for (unsigned int r = 0; r < nk_; ++r) Yk += std::conj(kphase_[jk][r]) * Y[r];
            }
            MatrixXcd V = factor * (U_[jk].transpose() * Yk * U_[jk].conjugate());
            if (jk == jg_) {
                for (unsigned int c = 0; c < ns_; ++c) {
                    if (!acoustic_[c]) continue;
                    V.row(c).setZero();
                    V.col(c).setZero();
                }
            }
            for (unsigned int c = 0; c < ns_; ++c) {
                for (unsigned int d = 0; d < ns_; ++d) {
                    if (accumulate) {
                        v3_renorm[jk][b][c * ns_ + d] += V(c, d);
                    } else {
                        v3_renorm[jk][b][c * ns_ + d] = v3_with_umn[jk][b][c * ns_ + d] + V(c, d);
                    }
                }
            }
        }
    }
}

} // namespace PHON_NS::quartic_rs
