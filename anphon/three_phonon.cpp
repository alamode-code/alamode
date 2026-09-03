/*
three_phonon.cpp

Factorized evaluation of the cubic matrix elements and the three-phonon
linewidth kernels built on it.

Copyright (c) 2026 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "integration.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "mpi_common.h"
#include "phonon_dos.h"
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
// With compound Cartesian indices a, b, c of the primitive cell and mass-scaled
// eigenvectors e~, the cubic coupling of the legs (K s0), (k1 s1), (k2 s2) with
// K + k1 + k2 = G is
//
//   V3 = sum_{abc} e~_a(K s0) e~_b(k1 s1) e~_c(k2 s2) Phi(K,k1,k2)[a,b,c] / sqrt(w0 w1 w2),
//   Phi[a,b,c] = sum_{Rb Rc} phi(a b c; Rb Rc) exp(i(k1 Rb + k2 Rc))
//              = sum_{Rb Rc} phi(a b c; Rb Rc) exp(i k1 (Rb - Rc)) exp(-i K Rc),
//
// where momentum conservation was used to eliminate k2. The force constants
// are therefore stored grouped by (b c, dR = Rb - Rc, a, Rc): the phase of the
// first leg, exp(-i K Rc), is folded in once per K (psi_K), the first-leg
// eigenvector once per mode (psi_mode), and a triplet costs one sparse sum
// over the (b c, dR) terms plus two dense products
//
//   A[b,c] = sum_dR psi_mode(b c, dR) exp(i k1 dR)        (sparse)
//   B[s1,c] = sum_b e~_b(k1 s1) A[b,c]                     (ns x n x n)
//   V3[s1,s2] * sqrt(...) = sum_c e~_c(k2 s2) B[s1,c]      (ns x n x ns)
//
// instead of ns^2 * N_groups. When the first-leg band is free as well (the
// IBTE collision operator needs all (s0, s1, s2)), the eigenvector fold is
// replaced by a third dense product at O(n^4) per triplet.
// ----------------------------------------------------------------------------

namespace
{

using Cplx = std::complex<double>;
using MatrixRow = Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MapRow = Eigen::Map<MatrixRow>;
using ConstMapRow = Eigen::Map<const MatrixRow>;

struct FC3Term
{
    long long bc;
    int dr;
    int a;
    int rc;
    double val;
};

inline int positive_modulo(const int a, const int b)
{
    const int m = a % b;
    return m < 0 ? m + b : m;
}

// exp(i sign k.R) for integer R and k = q / N on the uniform grid.
inline Cplx phase_factor(const int *q, const int *R, const int sign, const int *ngrid, const std::vector<Cplx> *table)
{
    Cplx e(1.0, 0.0);
    for (auto icrd = 0; icrd < 3; ++icrd) {
        e *= table[icrd][positive_modulo(sign * q[icrd] * R[icrd], ngrid[icrd])];
    }
    return e;
}

inline void integer_kpoint(const KpointMeshUniform *kmesh, const int ik, const int *ngrid, int q[3])
{
    for (auto i = 0; i < 3; ++i) q[i] = nint(kmesh->xk[ik][i] * static_cast<double>(ngrid[i]));
}

} // namespace

void AnharmonicCore::prepare_fc3_compressed()
{
    if (fc3_compressed) return;

    auto cfc = std::make_unique<FC3Compressed>();
    const int n = dynamical->neval;
    cfc->n = n;

    std::map<std::array<int, 3>, int> drmap, rcmap;
    std::vector<FC3Term> terms;

    size_t nterms_total = 0;
    for (auto i = 0; i < ngroup_v3; ++i) nterms_total += fcs_group_v3[i].size();
    terms.reserve(nterms_total);

    std::array<int, 3> keydr{}, keyrc{};
    for (auto i = 0; i < ngroup_v3; ++i) {
        const int a = evec_index_v3[i][0];
        const long long b = evec_index_v3[i][1];
        const long long c = evec_index_v3[i][2];
        const long long bc = b * n + c;
        const auto nsize_group = fcs_group_v3[i].size();
        for (size_t j = 0; j < nsize_group; ++j) {
            const auto &vecs = relvec_v3[i][j].vecs;
            for (auto icrd = 0; icrd < 3; ++icrd) {
                const int rb = nint(vecs[0][icrd] * inv_tpi);
                const int rc = nint(vecs[1][icrd] * inv_tpi);
                keydr[icrd] = rb - rc;
                keyrc[icrd] = rc;
            }
            const auto itdr = drmap.emplace(keydr, static_cast<int>(drmap.size())).first;
            const auto itrc = rcmap.emplace(keyrc, static_cast<int>(rcmap.size())).first;
            terms.push_back({bc, itdr->second, a, itrc->second, fcs_group_v3[i][j]});
        }
    }

    cfc->ndr = static_cast<int>(drmap.size());
    cfc->nrc = static_cast<int>(rcmap.size());
    cfc->dr_vec.resize(3 * cfc->ndr);
    cfc->rc_vec.resize(3 * cfc->nrc);
    for (const auto &it: drmap) {
        for (auto m = 0; m < 3; ++m) cfc->dr_vec[3 * it.second + m] = it.first[m];
    }
    for (const auto &it: rcmap) {
        for (auto m = 0; m < 3; ++m) cfc->rc_vec[3 * it.second + m] = it.first[m];
    }

    std::sort(terms.begin(), terms.end(), [](const FC3Term &x, const FC3Term &y) {
        if (x.bc != y.bc) return x.bc < y.bc;
        if (x.dr != y.dr) return x.dr < y.dr;
        if (x.a != y.a) return x.a < y.a;
        return x.rc < y.rc;
    });

    cfc->entry_rc.resize(terms.size());
    cfc->entry_val.resize(terms.size());
    long long bc_prev = -1;
    int dr_prev = -1, a_prev = -1;
    for (size_t it = 0; it < terms.size(); ++it) {
        const auto &t = terms[it];
        if (t.bc != bc_prev) {
            cfc->row_bc.push_back(t.bc);
            cfc->row_ptr.push_back(static_cast<int>(cfc->grp_dr.size()));
            bc_prev = t.bc;
            dr_prev = -1;
            a_prev = -1;
        }
        if (t.dr != dr_prev) {
            cfc->grp_dr.push_back(t.dr);
            cfc->grp_ptr.push_back(static_cast<int>(cfc->sub_a.size()));
            dr_prev = t.dr;
            a_prev = -1;
        }
        if (t.a != a_prev) {
            cfc->sub_a.push_back(t.a);
            cfc->sub_ptr.push_back(static_cast<int>(it));
            a_prev = t.a;
        }
        cfc->entry_rc[it] = t.rc;
        cfc->entry_val[it] = t.val;
    }
    cfc->sub_ptr.push_back(static_cast<int>(terms.size()));
    cfc->grp_ptr.push_back(static_cast<int>(cfc->sub_a.size()));
    cfc->row_ptr.push_back(static_cast<int>(cfc->grp_dr.size()));

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << "\n";
        std::cout << " Three-phonon matrix elements: factorized evaluation\n";
        std::cout << "  Cartesian indices per cell (n)       : " << n << '\n';
        std::cout << "  Cubic force constant terms            : " << terms.size() << '\n';
        std::cout << "  Unique (Rb - Rc) / Rc lattice vectors : " << cfc->ndr << " / " << cfc->nrc << '\n';
        std::cout << "  (a b c, Rb - Rc) terms (per K)        : " << cfc->sub_a.size() << '\n';
        std::cout << "  (b c, Rb - Rc) terms (per triplet)    : " << cfc->grp_dr.size() << '\n';
        std::cout << "  Non-empty (b c) rows                  : " << cfc->row_bc.size() << " of "
                  << static_cast<long long>(n) * n << '\n';
        std::cout << std::flush;
    }

    fc3_compressed = std::move(cfc);
}

void AnharmonicCore::v3_setup_workspace(V3Workspace &ws, const KpointMeshUniform *kmesh_in) const
{
    const auto &cfc = *fc3_compressed;
    const int n = cfc.n;
    const size_t n2 = static_cast<size_t>(n) * n;
    const size_t n3 = n2 * n;

    if (ws.kmesh != kmesh_in) {
        ws.kmesh = kmesh_in;
        for (auto i = 0; i < 3; ++i) {
            ws.nkgrid[i] = static_cast<int>(kmesh_in->nk_i[i]);
            ws.exp_table[i].resize(ws.nkgrid[i]);
            for (auto j = 0; j < ws.nkgrid[i]; ++j) {
                ws.exp_table[i][j] = std::exp(im * (tpi * static_cast<double>(j) / static_cast<double>(ws.nkgrid[i])));
            }
        }
        ws.psiK_index = -1;
    }
    if (ws.exp_dr.size() != static_cast<size_t>(cfc.ndr)) ws.exp_dr.resize(cfc.ndr);
    if (ws.A.size() != n2) ws.A.assign(n2, Cplx(0.0, 0.0));
    if (ws.B.size() != n2) ws.B.resize(n2);
    if (ws.D.size() != n2) ws.D.resize(n2);
    if (ws.e1.size() != n2) ws.e1.resize(n2);
    if (ws.e2.size() != n2) ws.e2.resize(n2);
    if (ws.want_full) {
        // Phi must hold all n^3 entries; T1, T2 and T3 are chunked over the
        // first-leg band to stay below about 64 MB in total.
        constexpr size_t budget_bytes = static_cast<size_t>(64) << 20;
        ws.s0_chunk =
            static_cast<int>(std::max<size_t>(1, std::min<size_t>(n, budget_bytes / (3 * n2 * sizeof(Cplx)))));
        const size_t chunk_n2 = static_cast<size_t>(ws.s0_chunk) * n2;
        if (ws.phi_full.size() != n3) ws.phi_full.assign(n3, Cplx(0.0, 0.0));
        if (ws.T1.size() != chunk_n2) ws.T1.resize(chunk_n2);
        if (ws.T2.size() != chunk_n2) ws.T2.resize(chunk_n2);
        if (ws.T3.size() != chunk_n2) ws.T3.resize(chunk_n2);
        if (ws.e0.size() != n2) ws.e0.resize(n2);
    }
}

void AnharmonicCore::v3_fold_first_k(V3Workspace &ws, const int kfirst, std::vector<Cplx> &psi_out) const
{
    // psi_K(a b c, dR) = sum_{Rc} phi(a b c; dR + Rc, Rc) exp(-i K Rc)
    const auto &cfc = *fc3_compressed;
    int q[3];
    integer_kpoint(ws.kmesh, kfirst, ws.nkgrid, q);

    std::vector<Cplx> exp_rc(cfc.nrc);
    for (auto ir = 0; ir < cfc.nrc; ++ir) {
        exp_rc[ir] = phase_factor(q, &cfc.rc_vec[3 * ir], -1, ws.nkgrid, ws.exp_table);
    }

    const int nsub = static_cast<int>(cfc.sub_a.size());
    psi_out.resize(nsub);
    for (auto isub = 0; isub < nsub; ++isub) {
        double sre = 0.0, sim = 0.0;
        for (auto ie = cfc.sub_ptr[isub]; ie < cfc.sub_ptr[isub + 1]; ++ie) {
            const auto &e = exp_rc[cfc.entry_rc[ie]];
            sre += cfc.entry_val[ie] * e.real();
            sim += cfc.entry_val[ie] * e.imag();
        }
        psi_out[isub] = Cplx(sre, sim);
    }
}

void AnharmonicCore::prepare_v3_mode(const KpointMeshUniform *kmesh_in, const int kfirst, const int sfirst,
                                     const std::complex<double> *const *const *evec_in)
{
    // psi_mode(b c, dR) = sum_a e~_a(K s0) psi_K(a b c, dR); psi_K is kept
    // across the bands of the same K.
    prepare_fc3_compressed();
    v3_ws_mode.want_full = false;
    v3_setup_workspace(v3_ws_mode, kmesh_in);

    if (v3_ws_mode.psiK_index != kfirst) {
        v3_fold_first_k(v3_ws_mode, kfirst, v3_ws_mode.psiK);
        v3_ws_mode.psiK_index = kfirst;
    }

    const auto &cfc = *fc3_compressed;
    const int n = cfc.n;
    const auto &invsqrt_mass = system->get_invsqrt_mass();
    std::vector<Cplx> e0(n);
    for (auto a = 0; a < n; ++a) e0[a] = evec_in[kfirst][sfirst][a] * invsqrt_mass[a / 3];

    const int ngrp = static_cast<int>(cfc.grp_dr.size());
    psi_mode.resize(ngrp);
    const auto &psiK = v3_ws_mode.psiK;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int ig = 0; ig < ngrp; ++ig) {
        double sre = 0.0, sim = 0.0;
        for (auto isub = cfc.grp_ptr[ig]; isub < cfc.grp_ptr[ig + 1]; ++isub) {
            const auto &e = e0[cfc.sub_a[isub]];
            const auto &p = psiK[isub];
            sre += e.real() * p.real() - e.imag() * p.imag();
            sim += e.real() * p.imag() + e.imag() * p.real();
        }
        psi_mode[ig] = Cplx(sre, sim);
    }
    psi_mode_kfirst = kfirst;
    psi_mode_sfirst = sfirst;
}

void AnharmonicCore::v3sq_pairs(V3Workspace &ws, const KpointMeshUniform *kmesh_in, const int k1, const int k2,
                                const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                                double *out) const
{
    // |V3(K s0; k1 s1; k2 s2)|^2 for all (s1, s2) -> out[s1 * ns + s2], using
    // the psi_mode of prepare_v3_mode. Thread-safe: only ws is written.
    const auto &cfc = *fc3_compressed;
    const int n = cfc.n;
    const int ns = n;
    const size_t ns2 = static_cast<size_t>(ns) * ns;
    v3_setup_workspace(ws, kmesh_in);

    const double omega0 = eval_in[psi_mode_kfirst][psi_mode_sfirst];
    if (omega0 < eps8) {
        std::fill(out, out + ns2, 0.0);
        return;
    }

    int q1[3];
    integer_kpoint(ws.kmesh, k1, ws.nkgrid, q1);
    for (auto idr = 0; idr < cfc.ndr; ++idr) {
        ws.exp_dr[idr] = phase_factor(q1, &cfc.dr_vec[3 * idr], 1, ws.nkgrid, ws.exp_table);
    }

    const int nrow = static_cast<int>(cfc.row_bc.size());
    for (auto irow = 0; irow < nrow; ++irow) {
        double sre = 0.0, sim = 0.0;
        for (auto ig = cfc.row_ptr[irow]; ig < cfc.row_ptr[irow + 1]; ++ig) {
            const auto &p = psi_mode[ig];
            const auto &e = ws.exp_dr[cfc.grp_dr[ig]];
            sre += p.real() * e.real() - p.imag() * e.imag();
            sim += p.real() * e.imag() + p.imag() * e.real();
        }
        ws.A[cfc.row_bc[irow]] = Cplx(sre, sim);
    }

    const auto &invsqrt_mass = system->get_invsqrt_mass();
    for (auto s = 0; s < ns; ++s) {
        for (auto a = 0; a < n; ++a) {
            const double m = invsqrt_mass[a / 3];
            ws.e1[s * n + a] = evec_in[k1][s][a] * m;
            ws.e2[s * n + a] = evec_in[k2][s][a] * m;
        }
    }
    {
        ConstMapRow Am(ws.A.data(), n, n);
        ConstMapRow E1m(ws.e1.data(), ns, n);
        ConstMapRow E2m(ws.e2.data(), ns, n);
        MapRow Bm(ws.B.data(), ns, n);
        MapRow Dm(ws.D.data(), ns, ns);
        Bm.noalias() = E1m * Am;
        Dm.noalias() = Bm * E2m.transpose();
    }

    for (auto s1 = 0; s1 < ns; ++s1) {
        const double w1 = eval_in[k1][s1];
        for (auto s2 = 0; s2 < ns; ++s2) {
            const double w2 = eval_in[k2][s2];
            if (w1 < eps8 || w2 < eps8) {
                out[s1 * ns + s2] = 0.0;
            } else {
                out[s1 * ns + s2] = std::norm(ws.D[s1 * ns + s2]) / (omega0 * w1 * w2);
            }
        }
    }
}

void AnharmonicCore::v3sq_triples(V3Workspace &ws, const KpointMeshUniform *kmesh_in, const int kfirst, const int k1,
                                  const int k2, const double *const *eval_in,
                                  const std::complex<double> *const *const *evec_in, double *out) const
{
    // |V3(K s0; k1 s1; k2 s2)|^2 for all (s0, s1, s2) -> out[(s0 * ns + s1) * ns + s2].
    // psi_K is cached in ws across calls with the same K. Thread-safe.
    const auto &cfc = *fc3_compressed;
    const int n = cfc.n;
    const int ns = n;
    const size_t n2 = static_cast<size_t>(n) * n;
    ws.want_full = true;
    v3_setup_workspace(ws, kmesh_in);

    if (ws.psiK_index != kfirst) {
        v3_fold_first_k(ws, kfirst, ws.psiK);
        ws.psiK_index = kfirst;
    }

    int q1[3];
    integer_kpoint(kmesh_in, k1, ws.nkgrid, q1);
    for (auto idr = 0; idr < cfc.ndr; ++idr) {
        ws.exp_dr[idr] = phase_factor(q1, &cfc.dr_vec[3 * idr], 1, ws.nkgrid, ws.exp_table);
    }

    // Phi[a][b c] (row-major a x n^2). Only the touched entries are reset.
    for (const auto idx: ws.touched) ws.phi_full[idx] = Cplx(0.0, 0.0);
    ws.touched.clear();
    const int nrow = static_cast<int>(cfc.row_bc.size());
    for (auto irow = 0; irow < nrow; ++irow) {
        const auto bc = cfc.row_bc[irow];
        for (auto ig = cfc.row_ptr[irow]; ig < cfc.row_ptr[irow + 1]; ++ig) {
            const auto &e = ws.exp_dr[cfc.grp_dr[ig]];
            for (auto isub = cfc.grp_ptr[ig]; isub < cfc.grp_ptr[ig + 1]; ++isub) {
                const size_t idx = static_cast<size_t>(cfc.sub_a[isub]) * n2 + bc;
                const auto &p = ws.psiK[isub];
                if (ws.phi_full[idx] == Cplx(0.0, 0.0)) ws.touched.push_back(idx);
                ws.phi_full[idx] +=
                    Cplx(p.real() * e.real() - p.imag() * e.imag(), p.real() * e.imag() + p.imag() * e.real());
            }
        }
    }

    const auto &invsqrt_mass = system->get_invsqrt_mass();
    for (auto s = 0; s < ns; ++s) {
        for (auto a = 0; a < n; ++a) {
            const double m = invsqrt_mass[a / 3];
            ws.e0[s * n + a] = evec_in[kfirst][s][a] * m;
            ws.e1[s * n + a] = evec_in[k1][s][a] * m;
            ws.e2[s * n + a] = evec_in[k2][s][a] * m;
        }
    }
    // Contractions in chunks of s0 (first-leg band); Phi (n x n^2) is shared.
    ConstMapRow Pm(ws.phi_full.data(), n, static_cast<Eigen::Index>(n2));
    ConstMapRow E1m(ws.e1.data(), ns, n);
    ConstMapRow E2m(ws.e2.data(), ns, n);
    for (auto s0a = 0; s0a < ns; s0a += ws.s0_chunk) {
        const int nsc = std::min(ws.s0_chunk, ns - s0a);

        // T1 (nsc x n^2) = E0[s0a:s0a+nsc] (nsc x n) * Phi (n x n^2)
        ConstMapRow E0m(ws.e0.data() + static_cast<size_t>(s0a) * n, nsc, n);
        MapRow T1m(ws.T1.data(), nsc, static_cast<Eigen::Index>(n2));
        T1m.noalias() = E0m * Pm;

        // T2[j] (ns x n) = E1 (ns x n) * T1[j] (n x n)
        for (auto j = 0; j < nsc; ++j) {
            ConstMapRow T1s(ws.T1.data() + j * n2, n, n);
            MapRow T2s(ws.T2.data() + static_cast<size_t>(j) * ns * n, ns, n);
            T2s.noalias() = E1m * T1s;
        }

        // T3 (nsc*ns x ns) = T2 (nsc*ns x n) * E2^T (n x ns)
        ConstMapRow T2m(ws.T2.data(), static_cast<Eigen::Index>(nsc) * ns, n);
        MapRow T3m(ws.T3.data(), static_cast<Eigen::Index>(nsc) * ns, ns);
        T3m.noalias() = T2m * E2m.transpose();

        for (auto j = 0; j < nsc; ++j) {
            const int s0 = s0a + j;
            const double w0 = eval_in[kfirst][s0];
            for (auto s1 = 0; s1 < ns; ++s1) {
                const double w1 = eval_in[k1][s1];
                for (auto s2 = 0; s2 < ns; ++s2) {
                    const double w2 = eval_in[k2][s2];
                    const size_t idx = (static_cast<size_t>(s0) * ns + s1) * ns + s2;
                    if (w0 < eps8 || w1 < eps8 || w2 < eps8) {
                        out[idx] = 0.0;
                    } else {
                        out[idx] = std::norm(ws.T3[(static_cast<size_t>(j) * ns + s1) * ns + s2]) / (w0 * w1 * w2);
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Three-phonon linewidth kernels (relaxation time approximation)
// ----------------------------------------------------------------------------

namespace
{

// Occupation numbers occ[itemp][k * ns + s] on a mesh.
void tabulate_occupations(const unsigned int ntemp, const double *temp_in, const int nk, const int ns,
                          const double *const *eval_in, const bool classical, std::vector<double> &occ)
{
    const size_t nks = static_cast<size_t>(nk) * ns;
    occ.resize(static_cast<size_t>(ntemp) * nks);
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
}

} // namespace

void AnharmonicCore::calc_damping_smearing(const unsigned int ntemp, const double *temp_in, const double omega_in,
                                           const unsigned int ik_in, const unsigned int is_in,
                                           const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                           const std::complex<double> *const *const *evec_in, double *ret)
{
    // Imaginary part of the phonon self-energy at omega_in with Lorentzian,
    // Gaussian or adaptive Gaussian smearing (ISMEAR = 0, 1, 2).

    const int nk = kmesh_in->nk;
    const int ns = dynamical->neval;
    const size_t ns2 = static_cast<size_t>(ns) * ns;
    const size_t nks = static_cast<size_t>(nk) * ns;

    for (unsigned int i = 0; i < ntemp; ++i) ret[i] = 0.0;
    if (ngroup_v3 == 0) return;

    std::vector<KsListGroup> triplet;
    kmesh_in->get_unique_triplet_k(ik_in, symmetry->SymmList, false, false, triplet);
    const int npair_uniq = static_cast<int>(triplet.size());

    const int knum = kmesh_in->kpoint_irred_all[ik_in][0].knum;
    const int knum_minus = kmesh_in->kindex_minus_xk[knum];

    prepare_v3_mode(kmesh_in, knum_minus, static_cast<int>(is_in), evec_in);

    const bool classical = thermodynamics->classical;
    std::vector<double> occ;
    tabulate_occupations(ntemp, temp_in, nk, ns, eval_in, classical, occ);

    const int ismear = integration->ismear;
    const double epsilon = integration->epsilon;
    const bool adaptive = ismear == 2;
    std::vector<double> proj;
    double adaptive_factor = 0.0;
    if (adaptive) {
        proj.resize(3 * nks);
        for (auto ik = 0; ik < nk; ++ik) {
            for (auto is = 0; is < ns; ++is) {
                integration->adaptive_sigma->get_projected_velocity(ik, is, &proj[3 * (ik * ns + is)]);
            }
        }
        adaptive_factor = integration->adaptive_sigma->get_adaptive_factor();
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        V3Workspace ws;
        std::vector<double> v3sq(ns2), delta(2 * ns2);
        std::vector<double> ret_loc(ntemp, 0.0);
        ws.kmesh = nullptr;
        v3_setup_workspace(ws, kmesh_in);

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 4)
#endif
        for (int ik = 0; ik < npair_uniq; ++ik) {
            const int k1 = triplet[ik].group[0].ks[0];
            const int k2 = triplet[ik].group[0].ks[1];
            const double multi = static_cast<double>(triplet[ik].group.size());

            v3sq_pairs(ws, kmesh_in, k1, k2, eval_in, evec_in, v3sq.data());

            for (auto is = 0; is < ns; ++is) {
                const double w1 = eval_in[k1][is];
                for (auto js = 0; js < ns; ++js) {
                    const double w2 = eval_in[k2][js];
                    const size_t ib = static_cast<size_t>(is) * ns + js;
                    double d0, d1;
                    if (ismear == 0) {
                        d0 = delta_lorentz(omega_in - w1 - w2, epsilon) - delta_lorentz(omega_in + w1 + w2, epsilon);
                        d1 = delta_lorentz(omega_in - w1 + w2, epsilon) - delta_lorentz(omega_in + w1 - w2, epsilon);
                    } else if (ismear == 1) {
                        d0 = delta_gauss(omega_in - w1 - w2, epsilon) - delta_gauss(omega_in + w1 + w2, epsilon);
                        d1 = delta_gauss(omega_in - w1 + w2, epsilon) - delta_gauss(omega_in + w1 - w2, epsilon);
                    } else {
                        const double *p1 = &proj[3 * (k1 * ns + is)];
                        const double *p2 = &proj[3 * (k2 * ns + js)];
                        double pm = 0.0, pp = 0.0;
                        for (auto u = 0; u < 3; ++u) {
                            pm += (p1[u] - p2[u]) * (p1[u] - p2[u]);
                            pp += (p1[u] + p2[u]) * (p1[u] + p2[u]);
                        }
                        const double sig0 =
                            std::max(AdaptiveSmearingSigma::sigma_min, adaptive_factor * std::sqrt(pm / 12.0));
                        const double sig1 =
                            std::max(AdaptiveSmearingSigma::sigma_min, adaptive_factor * std::sqrt(pp / 12.0));
                        d0 = delta_gauss(omega_in - w1 - w2, sig0) - delta_gauss(omega_in + w1 + w2, sig0);
                        d1 = delta_gauss(omega_in - w1 + w2, sig1) - delta_gauss(omega_in + w1 - w2, sig1);
                    }
                    delta[2 * ib] = d0;
                    delta[2 * ib + 1] = d1;
                }
            }

            for (unsigned int it = 0; it < ntemp; ++it) {
                const double *occ1 = &occ[it * nks + k1 * ns];
                const double *occ2 = &occ[it * nks + k2 * ns];
                double sum = 0.0;
                for (auto is = 0; is < ns; ++is) {
                    const double f1 = occ1[is];
                    for (auto js = 0; js < ns; ++js) {
                        const double f2 = occ2[js];
                        const size_t ib = static_cast<size_t>(is) * ns + js;
                        const double n1 = classical ? f1 + f2 : f1 + f2 + 1.0;
                        const double n2 = f1 - f2;
                        sum += v3sq[ib] * (n1 * delta[2 * ib] - n2 * delta[2 * ib + 1]);
                    }
                }
                ret_loc[it] += multi * sum;
            }
        }

#ifdef _OPENMP
#pragma omp critical
#endif
        {
            for (unsigned int it = 0; it < ntemp; ++it) ret[it] += ret_loc[it];
        }
    }

    for (unsigned int i = 0; i < ntemp; ++i) ret[i] *= pi * std::pow(0.5, 4) / static_cast<double>(nk);
}

void AnharmonicCore::calc_damping_tetrahedron(const unsigned int ntemp, const double *temp_in, const double omega_in,
                                              const unsigned int ik_in, const unsigned int is_in,
                                              const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                              const std::complex<double> *const *const *evec_in, double *ret)
{
    // Imaginary part of the phonon self-energy at omega_in with the
    // tetrahedron method. The crystal symmetry reduces the triplets.

    const int nk = kmesh_in->nk;
    const int ns = dynamical->neval;
    const size_t ns2 = static_cast<size_t>(ns) * ns;
    const size_t nks = static_cast<size_t>(nk) * ns;

    for (unsigned int i = 0; i < ntemp; ++i) ret[i] = 0.0;
    if (ngroup_v3 == 0) return;

    std::vector<KsListGroup> triplet;
    kmesh_in->get_unique_triplet_k(ik_in, symmetry->SymmList, use_triplet_symmetry, sym_permutation, triplet);
    const int npair_uniq = static_cast<int>(triplet.size());

    NDArray<double, 3> delta_arr;
    delta_arr.resize(npair_uniq, ns2, 2);

    const int knum = kmesh_in->kpoint_irred_all[ik_in][0].knum;
    const int knum_minus = kmesh_in->kindex_minus_xk[knum];
    const auto &xk = kmesh_in->xk;

    NDArray<unsigned int, 1> kmap_identity;
    kmap_identity.resize(nk);
    for (auto i = 0; i < nk; ++i) kmap_identity[i] = i;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        NDArray<double, 2> energy_tmp, weight_tetra;
        energy_tmp.resize(3, nk);
        weight_tetra.resize(3, nk);
        double xk_tmp[3];

#ifdef _OPENMP
#pragma omp for
#endif
        for (int ib = 0; ib < static_cast<int>(ns2); ++ib) {
            const int is = ib / ns;
            const int js = ib % ns;

            for (auto k1 = 0; k1 < nk; ++k1) {
                // Two-phonon frequencies for the tetrahedron method
                for (auto i = 0; i < 3; ++i) xk_tmp[i] = xk[knum][i] - xk[k1][i];
                const auto k2 = kmesh_in->get_knum(xk_tmp);
                energy_tmp[0][k1] = eval_in[k1][is] + eval_in[k2][js];
                energy_tmp[1][k1] = eval_in[k1][is] - eval_in[k2][js];
                energy_tmp[2][k1] = -energy_tmp[1][k1];
            }

            for (auto i = 0; i < 3; ++i) {
                integration->calc_weight_tetrahedron(nk,
                                                     kmap_identity,
                                                     energy_tmp[i],
                                                     omega_in,
                                                     dos->tetra_nodes_dos->get_ntetra(),
                                                     dos->tetra_nodes_dos->get_tetras(),
                                                     weight_tetra[i]);
            }

            for (auto ik = 0; ik < npair_uniq; ++ik) {
                const auto jk = triplet[ik].group[0].ks[0];
                delta_arr[ik][ib][0] = weight_tetra[0][jk];
                delta_arr[ik][ib][1] = weight_tetra[1][jk] - weight_tetra[2][jk];
            }
        }
    }

    // Average the weights over degenerate branches so that the result does
    // not depend on the gauge of degenerate eigenvectors.
    const auto tol_degenerate = 1.0e-7 * time_ry / Hz_to_kayser;
    auto get_degenerate_blocks = [&](const unsigned int knum_in) {
        std::vector<std::pair<unsigned int, unsigned int>> blocks;
        auto begin = 0U;
        auto omega_ref = eval_in[knum_in][0];
        for (auto s = 1U; s < static_cast<unsigned int>(ns); ++s) {
            const auto omega_now = eval_in[knum_in][s];
            if (std::abs(omega_now - omega_ref) >= tol_degenerate) {
                blocks.emplace_back(begin, s);
                begin = s;
                omega_ref = omega_now;
            }
        }
        blocks.emplace_back(begin, ns);
        return blocks;
    };

    for (auto ik = 0; ik < npair_uniq; ++ik) {
        const auto k1 = triplet[ik].group[0].ks[0];
        const auto k2 = triplet[ik].group[0].ks[1];
        const auto blocks1 = get_degenerate_blocks(k1);
        const auto blocks2 = get_degenerate_blocks(k2);
        for (const auto &block1: blocks1) {
            for (const auto &block2: blocks2) {
                const auto nblock =
                    static_cast<double>((block1.second - block1.first) * (block2.second - block2.first));
                if (nblock <= 1.0) continue;
                std::array<double, 2> delta_sum{};
                for (auto is = block1.first; is < block1.second; ++is) {
                    for (auto js = block2.first; js < block2.second; ++js) {
                        delta_sum[0] += delta_arr[ik][ns * is + js][0];
                        delta_sum[1] += delta_arr[ik][ns * is + js][1];
                    }
                }
                delta_sum[0] /= nblock;
                delta_sum[1] /= nblock;
                for (auto is = block1.first; is < block1.second; ++is) {
                    for (auto js = block2.first; js < block2.second; ++js) {
                        delta_arr[ik][ns * is + js][0] = delta_sum[0];
                        delta_arr[ik][ns * is + js][1] = delta_sum[1];
                    }
                }
            }
        }
    }

    prepare_v3_mode(kmesh_in, knum_minus, static_cast<int>(is_in), evec_in);

    const bool classical = thermodynamics->classical;
    std::vector<double> occ;
    tabulate_occupations(ntemp, temp_in, nk, ns, eval_in, classical, occ);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        V3Workspace ws;
        std::vector<double> v3sq(ns2);
        std::vector<double> ret_loc(ntemp, 0.0);
        ws.kmesh = nullptr;
        v3_setup_workspace(ws, kmesh_in);

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 4)
#endif
        for (int ik = 0; ik < npair_uniq; ++ik) {
            const int k1 = triplet[ik].group[0].ks[0];
            const int k2 = triplet[ik].group[0].ks[1];
            const double multi = static_cast<double>(triplet[ik].group.size());

            v3sq_pairs(ws, kmesh_in, k1, k2, eval_in, evec_in, v3sq.data());

            for (unsigned int it = 0; it < ntemp; ++it) {
                const double *occ1 = &occ[it * nks + k1 * ns];
                const double *occ2 = &occ[it * nks + k2 * ns];
                double sum = 0.0;
                for (auto is = 0; is < ns; ++is) {
                    const double f1 = occ1[is];
                    for (auto js = 0; js < ns; ++js) {
                        const double f2 = occ2[js];
                        const size_t ib = static_cast<size_t>(is) * ns + js;
                        const double n1 = classical ? f1 + f2 : f1 + f2 + 1.0;
                        const double n2 = f1 - f2;
                        sum += v3sq[ib] * (n1 * delta_arr[ik][ib][0] - n2 * delta_arr[ik][ib][1]);
                    }
                }
                ret_loc[it] += multi * sum;
            }
        }

#ifdef _OPENMP
#pragma omp critical
#endif
        {
            for (unsigned int it = 0; it < ntemp; ++it) ret[it] += ret_loc[it];
        }
    }

    for (unsigned int i = 0; i < ntemp; ++i) ret[i] *= pi * std::pow(0.5, 4);
}

void AnharmonicCore::calc_self3omega_tetrahedron(const double Temp, const KpointMeshUniform *kmesh_in,
                                                 const double *const *eval,
                                                 const std::complex<double> *const *const *evec,
                                                 const unsigned int ik_in, const unsigned int snum,
                                                 const unsigned int nomega, const double *omega, double *ret)
{
    // Frequency-dependent imaginary part of the self-energy of mode
    // (ik_in, snum) at temperature Temp with the tetrahedron method. The
    // matrix elements are distributed over the MPI ranks by k1.

    const int nk = kmesh_in->nk;
    const int ns = dynamical->neval;
    const size_t ns2 = static_cast<size_t>(ns) * ns;

    for (unsigned int iomega = 0; iomega < nomega; ++iomega) ret[iomega] = 0.0;
    if (ngroup_v3 == 0) return;

    std::vector<KsListGroup> triplet;
    kmesh_in->get_unique_triplet_k(ik_in, symmetry->SymmList, false, false, triplet);
    const int npair_uniq = static_cast<int>(triplet.size());
    if (npair_uniq != nk) {
        exit("calc_self3omega_tetrahedron", "Something is wrong.");
    }

    const int knum = kmesh_in->kpoint_irred_all[ik_in][0].knum;
    const int knum_minus = kmesh_in->kindex_minus_xk[knum];

    NDArray<int, 2> kpairs;
    kpairs.resize(nk, 2);
    for (auto ik = 0; ik < nk; ++ik) {
        kpairs[ik][0] = triplet[ik].group[0].ks[0];
        kpairs[ik][1] = triplet[ik].group[0].ks[1];
    }

    int nk_tmp;
    if (nk % mympi->nprocs != 0) {
        nk_tmp = nk / mympi->nprocs + 1;
    } else {
        nk_tmp = nk / mympi->nprocs;
    }
    std::vector<int> vk_l;
    for (auto ik = 0; ik < nk; ++ik) {
        if (ik % mympi->nprocs == mympi->my_rank) vk_l.push_back(ik);
    }
    if (static_cast<int>(vk_l.size()) < nk_tmp) vk_l.push_back(-1);

    prepare_v3_mode(kmesh_in, knum_minus, static_cast<int>(snum), evec);

    NDArray<double, 1> v3_arr_loc;
    NDArray<double, 2> v3_arr; // gathered matrix elements, root only
    v3_arr_loc.resize(ns2);
    if (mympi->my_rank == 0) v3_arr.resize(nk_tmp * mympi->nprocs, ns2);

    V3Workspace ws;
    ws.kmesh = nullptr;
    v3_setup_workspace(ws, kmesh_in);

    for (auto ik = 0; ik < nk_tmp; ++ik) {
        const int ik_now = vk_l[ik];
        if (ik_now == -1) {
            for (size_t ib = 0; ib < ns2; ++ib) v3_arr_loc[ib] = 0.0; // do nothing
        } else {
            v3sq_pairs(ws, kmesh_in, kpairs[ik_now][0], kpairs[ik_now][1], eval, evec, &v3_arr_loc[0]);
        }
        double *recv = nullptr;
        if (mympi->my_rank == 0) recv = v3_arr[ik * mympi->nprocs];
        MPI_Gather(&v3_arr_loc[0], ns2, MPI_DOUBLE, recv, ns2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    v3_arr_loc.clear();

    if (mympi->my_rank == 0) {
        NDArray<unsigned int, 1> kmap_identity;
        kmap_identity.resize(nk);
        for (auto i = 0; i < nk; ++i) kmap_identity[i] = i;

        const bool classical = thermodynamics->classical;
        std::vector<double> occ;
        tabulate_occupations(1, &Temp, nk, ns, eval, classical, occ);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            NDArray<double, 2> energy_tmp, weight_tetra;
            energy_tmp.resize(2, nk);
            weight_tetra.resize(2, nk);
            std::vector<double> ret_loc(nomega, 0.0);

#ifdef _OPENMP
#pragma omp for
#endif
            for (int ib = 0; ib < static_cast<int>(ns2); ++ib) {
                const int is = ib / ns;
                const int js = ib % ns;
                for (auto ik = 0; ik < nk; ++ik) {
                    const auto k1 = kpairs[ik][0];
                    const auto k2 = kpairs[ik][1];
                    energy_tmp[0][ik] = eval[k1][is] + eval[k2][js];
                    energy_tmp[1][ik] = eval[k1][is] - eval[k2][js];
                }
                for (unsigned int iomega = 0; iomega < nomega; ++iomega) {
                    for (auto i = 0; i < 2; ++i) {
                        integration->calc_weight_tetrahedron(nk,
                                                             kmap_identity,
                                                             energy_tmp[i],
                                                             omega[iomega],
                                                             dos->tetra_nodes_dos->get_ntetra(),
                                                             dos->tetra_nodes_dos->get_tetras(),
                                                             weight_tetra[i]);
                    }
                    double sum = 0.0;
                    for (auto ik = 0; ik < nk; ++ik) {
                        const auto k1 = kpairs[ik][0];
                        const auto k2 = kpairs[ik][1];
                        const double f1 = occ[k1 * ns + is];
                        const double f2 = occ[k2 * ns + js];
                        const double n1 = classical ? f1 + f2 : f1 + f2 + 1.0;
                        const double n2 = f1 - f2;
                        sum += v3_arr[ik][ib] * (n1 * weight_tetra[0][ik] - 2.0 * n2 * weight_tetra[1][ik]);
                    }
                    ret_loc[iomega] += sum;
                }
            }
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                for (unsigned int iomega = 0; iomega < nomega; ++iomega) ret[iomega] += ret_loc[iomega];
            }
        }

        for (unsigned int iomega = 0; iomega < nomega; ++iomega) ret[iomega] *= pi * std::pow(0.5, 4);
    }
}
