/*
 scph_hessian.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.

 BUBBLE = 4: curvature of the SCP free energy with respect to the Gamma
 displacements q0 of the structural optimization (static bubble with the
 quartic ladder resummed; Masuki et al., PRB 106, 224104 (2022), App. B).
 Design and conventions: anphon/FREE_ENERGY_HESSIAN_PLAN.md.

 At a converged SCP solution the SCP force is
   g_i = v1_renorm[i] + sum_k sum_ab v3_renorm[k][i][a*ns+b] G_k(a,b),
   G_k = C_k f(Omega_k) C_k^dag            (harmonic-mode basis, dense mesh),
 and its derivative (the force Jacobian) is
   J_ij = A_ij + sum_k sum_ab v3_renorm[k][i][a*ns+b] dG_k^(j)(a,b),
 with A the SCP matrix at Gamma and dG the occupation response. The response
 is self-consistent: a change y of the SCP matrix on the coarse mesh gives
   dG = T(y)  (interpolation to the dense mesh, then Daleckii-Krein),
 and the quartic contraction of dG changes the SCP matrix again, so
   (I - V4 T) y_j = B_j,   B_j(a,b) = 4 N v3_renorm[kG][j][b*ns+a]
 (the source is the q0 derivative of the renormalized harmonic part, as in
 Relaxation::renormalize_v2_from_q0). First release: KMESH_INTERPOLATE =
 1 1 1, the unrestricted response (no symmetrization inside the solve), the
 Gamma translations frozen, restarted GMRES.
*/

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "fcs_phonon.h"
#include "interpolation.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "relaxation.h"
#include "scph.h"
#include "scph_hessian_kernels.h"
#include "thermodynamics.h"
#include "timer.h"
#include "v4_service.h"

using namespace PHON_NS;

namespace
{
using cplx = std::complex<double>;
using MatrixXcdRow = Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Signed frequencies (cm^-1) of the symmetric part of a real matrix in omega^2 units.
Eigen::VectorXd signed_frequencies(const Eigen::MatrixXd &M)
{
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (M + M.transpose()));
    Eigen::VectorXd w(M.rows());
    for (Eigen::Index i = 0; i < M.rows(); ++i) {
        const auto l = es.eigenvalues()(i);
        w(i) = (l >= 0.0 ? 1.0 : -1.0) * std::sqrt(std::abs(l)) * Ry_to_kayser;
    }
    return w;
}

// State variable m of the strain block (optimizer order): the diagonal u_mm
// for m < 3, the shear pair (m+1, m+2) mod 3 for m >= 3.
std::pair<int, int> strain_state_pair(const Eigen::Index m)
{
    if (m < 3) return {static_cast<int>(m), static_cast<int>(m)};
    return {static_cast<int>((m + 1) % 3), static_cast<int>((m + 2) % 3)};
}

// diag(|M_ii|^-1/2): makes displacement (omega^2) and strain (Ry) blocks
// comparable in the asymmetry and finite-difference measures.
Eigen::VectorXd diagonal_scale(const Eigen::MatrixXd &M)
{
    Eigen::VectorXd d(M.rows());
    for (Eigen::Index i = 0; i < M.rows(); ++i) d(i) = 1.0 / std::sqrt(std::max(std::abs(M(i, i)), 1.0e-300));
    return d;
}

// The quartic ladder done in real space on a k-mesh of n1 x n2 x n3 points
// (FINITE_Q_HESSIAN_PLAN.md, steps A and B). Every stored FC4 entry, divided by
// the square roots of its four masses, goes to the slot
//   (i, j, R_j; l, m, R_m - R_l)   (cell offsets reduced modulo the mesh),
// where i is the leg in the home cell and R_j, R_l, R_m the lattice offsets of
// the other legs; entries that land in one slot add up. This is the
// element-wise sum from which V4 is built, since the phases of the mesh do not
// see the reduction.
struct QuarticMesh
{
    unsigned int ns;
    unsigned int nk;
    std::array<int, 3> nki;
    std::vector<std::array<unsigned int, 6>> index; // i, j, cell(R_j), l, m, cell(R_m - R_l)
    std::vector<double> weight;

    int cell(const Eigen::Vector3d &v) const
    {
        int r = 0;
        for (int d = 0; d < 3; ++d) {
            auto n = static_cast<int>(std::lround(v[d])) % nki[d];
            if (n < 0) n += nki[d];
            r = r * nki[d] + n;
        }
        return r;
    }

    QuarticMesh(const std::vector<PHON_NS::FcsArrayWithCell> &fc4, const std::vector<double> &invsqrt_mass,
                const unsigned int ns_in, const unsigned int *nk_in) : ns(ns_in)
    {
        nki = {static_cast<int>(nk_in[0]), static_cast<int>(nk_in[1]), static_cast<int>(nk_in[2])};
        nk = static_cast<unsigned int>(nki[0] * nki[1] * nki[2]);
        std::unordered_map<unsigned long long, double> sum;
        sum.reserve(fc4.size());
        for (const auto &e: fc4) {
            const unsigned long long i = e.pairs[0].index, j = e.pairs[1].index, l = e.pairs[2].index,
                                     m = e.pairs[3].index;
            const unsigned long long rj = cell(e.relvecs[0]), rlm = cell(e.relvecs[2] - e.relvecs[1]);
            const auto key = (((((i * ns + j) * nk + rj) * ns + l) * ns + m) * nk) + rlm;
            sum[key] +=
                e.fcs_val * invsqrt_mass[i / 3] * invsqrt_mass[j / 3] * invsqrt_mass[l / 3] * invsqrt_mass[m / 3];
        }
        index.reserve(sum.size());
        weight.reserve(sum.size());
        for (const auto &[key, w]: sum) {
            auto r = key;
            std::array<unsigned int, 6> id{};
            const unsigned long long base[6] = {ns, ns, nk, ns, ns, nk};
            for (int d = 5; d >= 0; --d) {
                id[d] = static_cast<unsigned int>(r % base[d]);
                r /= base[d];
            }
            index.push_back(id);
            weight.push_back(w);
        }
    }

    // z_ij(R_j) = (1/4) sum W x_lm(R_m - R_l) (Cartesian, mass weighted,
    // unconjugated); x and z hold one ns x ns matrix per cell of the mesh.
    void apply(const std::vector<Eigen::MatrixXcd> &x, std::vector<Eigen::MatrixXcd> &z) const
    {
        z.assign(nk, Eigen::MatrixXcd::Zero(ns, ns));
        for (size_t e = 0; e < weight.size(); ++e) {
            const auto &id = index[e];
            z[id[2]](id[0], id[1]) += weight[e] * x[id[5]](id[3], id[4]);
        }
        for (auto &m: z) m *= 0.25;
    }
};
} // namespace

bool Scph::compute_scp_hessian(StructuralOptWorkspace &ws, const RelaxationStructureState &solved_state_in,
                               const unsigned int iT, const double temp, std::complex<double> ***cmat_convert,
                               double **omega2_scp, Eigen::MatrixXd &J, const bool report)
{
    // a copy: the caller may pass ws.structure_state itself, which the strain
    // differences below overwrite
    const RelaxationStructureState solved_state = solved_state_in;
    using namespace Eigen;
    const auto nk = kmesh_dense->nk;
    const auto ns = static_cast<Index>(dynamical->neval);
    const auto ns2 = ns * ns;
    const auto &optical = ws.harm_optical_modes;
    const auto nopt = static_cast<Index>(optical.size());
    const auto ikg = static_cast<unsigned int>(kmap_coarse_to_dense[0]);
    const auto t_start = timer->elapsed();

    if (report) {
        std::cout << "\n BUBBLE = 4: curvature of the SCP free energy (force Jacobian dg/dq0) at " << temp << " K\n";
    }

    const auto skip = [&](const std::string &reason) {
        if (report) {
            std::cout << "  skipped: " << reason << ".\n";
            write_scp_hessian(temp, reason, MatrixXd(), MatrixXd(), 0, 0.0, 0.0);
        } else {
            std::cout << " BUBBLE_HESS: no free-energy curvature (" << reason << "); default optimizer Hessian.\n";
        }
        return false;
    };

    // ---- eligibility at the final fixed point
    if (nopt == 0 && !uses_full_strain_derivatives(ws.relax_mode)) return skip("there are no optical modes");
    if (last_scp_repaired) return skip("the final SCP iteration repaired an eigenvalue");
    const auto frozen_gamma = classify_acoustic_modes_from_cmat(cmat_convert[ikg]);
    for (unsigned int ik = 0; ik < nk; ++ik) {
        for (Index s = 0; s < ns; ++s) {
            if (ik == ikg && frozen_gamma[s]) continue;
            if (!(omega2_scp[ik][s] > eps8 * eps8)) return skip("a non-acoustic SCP frequency is zero or imaginary");
        }
    }
    // The Gamma translations are frozen: the SCP modes classified as acoustic
    // must span exactly the harmonic translations.
    {
        Index nfrozen = 0, nacoustic = 0;
        double leakage = 0.0;
        for (Index s = 0; s < ns; ++s) {
            if (is_acoustic_gamma_harm[s]) ++nacoustic;
            if (!frozen_gamma[s]) continue;
            ++nfrozen;
            double weight = 0.0;
            for (Index a = 0; a < ns; ++a) {
                if (is_acoustic_gamma_harm[a]) weight += std::norm(cmat_convert[ikg][a][s]);
            }
            leakage = std::max(leakage, 1.0 - weight);
        }
        if (nfrozen != nacoustic || leakage > 1.0e-8) {
            return skip("the Gamma translations mix with optical modes in the SCP solution");
        }
    }

    // ---- per-k data: harmonic eigenvectors U (Cartesian x mode), SCP
    //      eigenvectors C (harmonic basis), divided differences L
    scph_hessian::OccupationFactor occ;
    occ.kT = Thermodynamics::T_to_Ryd * temp;
    occ.classical = thermodynamics->classical;

    std::vector<MatrixXcd> Uk(nk), Ck(nk), Pk(nk), Lk(nk);
    for (unsigned int ik = 0; ik < nk; ++ik) {
        Uk[ik].resize(ns, ns);
        Ck[ik].resize(ns, ns);
        VectorXd lambda(ns);
        for (Index a = 0; a < ns; ++a) {
            for (Index s = 0; s < ns; ++s) {
                Uk[ik](a, s) = evec_harmonic[ik][s][a];
                Ck[ik](a, s) = cmat_convert[ik][a][s];
            }
            lambda(a) = omega2_scp[ik][a];
        }
        Pk[ik] = Uk[ik] * Ck[ik]; // Cartesian -> SCP eigenbasis
        std::vector<bool> frozen(static_cast<size_t>(ns), false);
        if (ik == ikg) frozen = frozen_gamma;
        Lk[ik] = scph_hessian::divided_difference_matrix(occ, lambda, &frozen).cast<cplx>();
    }

    // ---- T: change y of the SCP matrix at every k (harmonic basis, row-major
    //      a*ns+b, block k at y + k*ns2) -> occupation change dG_k (same
    //      layout). The meshes are matched (KMESH_SCPH = KMESH_INTERPOLATE), so
    //      there is no interpolation, and no symmetrization: the response is
    //      unrestricted. Buffers are per call so that columns can run in parallel.
    const auto apply_T = [&](const cplx *y, cplx *dG) {
        for (unsigned int ik = 0; ik < nk; ++ik) {
            const Map<const MatrixXcdRow> Y(y + static_cast<size_t>(ik) * ns2, ns, ns);
            const MatrixXcd in_eig = Ck[ik].adjoint() * Y * Ck[ik];
            Map<MatrixXcdRow>(dG + static_cast<size_t>(ik) * ns2, ns, ns) =
                Ck[ik] * in_eig.cwiseProduct(Lk[ik]) * Ck[ik].adjoint();
        }
    };

    // ---- P: the projection the SCP loop applies to its Gamma matrix (the
    //      group of the starting structure, symmetrize_dynamical_matrix).
    //      For the optimizer (report = false) the response is projected in the
    //      same way, (I - P V4 T) y = P B. At a structure that keeps the
    //      starting symmetry (and for infinitesimal departures from it) this is
    //      the Jacobian of the forces the relaxation sees; the unrestricted
    //      curvature differs from it in symmetry-breaking directions, and a
    //      Newton step with it amplified noise there. At finite
    //      symmetry-broken iterates the explicit part A, built from the
    //      projected SCP matrix, is not exact. The reported curvature is
    //      unrestricted. P is the identity for a P1 structure.
    // ponytail: nsym ns^3 per column; symmetry-adapted blocks if large cells need it
    const bool project =
        !report && kmesh_coarse->small_group_of_k[0].size() + kmesh_coarse->symop_minus_at_k[0].size() > 1;
    // ponytail: the projected optimizer Hessian is built for one k-point; on a
    // mesh it would need the star replication of the SCP loop
    if (project && nk > 1) return skip("the projected optimizer Hessian needs KMESH_SCPH = 1 1 1");
    const auto apply_P = [&](cplx *y) {
        Map<MatrixXcdRow> Y(y, ns, ns);
        MatrixXcd cart = Uk[ikg] * Y * Uk[ikg].adjoint();
        symmetrize_dynamical_matrix(0, kmesh_coarse.get(), static_cast<unsigned int>(ns), mat_transform_sym, cart);
        Y = Uk[ikg].adjoint() * cart * Uk[ikg];
    };

    // ---- the quartic ladder in real space (BUBBLE_LADDER = 2, and the operator
    //      check of BUBBLE_FD_CHECK): F = P U^dagger Z(U P D P U^dagger) U P in
    //      the harmonic basis at Gamma, P dropping the acoustic modes as the V4
    //      builder does; the same map as fmat_batch at one k-point
    std::unique_ptr<QuarticMesh> quartic_rs;
    std::vector<std::vector<cplx>> kphase; // e^{2 pi i k.R} for k on the mesh and R a cell of it
    if (bubble_ladder == 2 || (report && bubble_fd_check)) {
        quartic_rs = std::make_unique<QuarticMesh>(fcs_phonon->force_constant_with_cell[2],
                                                   system->get_invsqrt_mass(),
                                                   static_cast<unsigned int>(ns),
                                                   kmesh_dense->nk_i);
        kphase.assign(nk, std::vector<cplx>(nk));
        for (unsigned int ik = 0; ik < nk; ++ik) {
            for (unsigned int r = 0; r < nk; ++r) {
                int n[3], rr = static_cast<int>(r);
                for (int d = 2; d >= 0; --d) {
                    n[d] = rr % quartic_rs->nki[d];
                    rr /= quartic_rs->nki[d];
                }
                double arg = 0.0;
                for (int d = 0; d < 3; ++d) arg += kmesh_dense->xk[ik][d] * n[d];
                kphase[ik][r] = std::exp(cplx(0.0, 2.0 * pi * arg));
            }
        }
    }
    const auto fmat_real_space = [&](const cplx *d_in, cplx *f_out) {
        std::vector<MatrixXcd> x(nk, MatrixXcd::Zero(ns, ns)), z;
        for (unsigned int ik = 0; ik < nk; ++ik) {
            MatrixXcd D = Map<const MatrixXcdRow>(d_in + static_cast<size_t>(ik) * ns2, ns, ns);
            if (ik == ikg) {
                for (Index a = 0; a < ns; ++a) {
                    if (!is_acoustic_gamma_harm[a]) continue;
                    D.row(a).setZero();
                    D.col(a).setZero();
                }
            }
            const MatrixXcd X = Uk[ik] * D * Uk[ik].adjoint();
            for (unsigned int r = 0; r < nk; ++r) x[r] += std::conj(kphase[ik][r]) * X;
        }
        for (auto &m: x) m /= static_cast<double>(nk);
        quartic_rs->apply(x, z);
        for (unsigned int ik = 0; ik < nk; ++ik) {
            MatrixXcd Z = MatrixXcd::Zero(ns, ns);
            for (unsigned int r = 0; r < nk; ++r) Z += kphase[ik][r] * z[r];
            MatrixXcd F = Uk[ik].adjoint() * Z * Uk[ik];
            if (ik == ikg) {
                for (Index a = 0; a < ns; ++a) {
                    if (!is_acoustic_gamma_harm[a]) continue;
                    F.row(a).setZero();
                    F.col(a).setZero();
                }
            }
            Map<MatrixXcdRow>(f_out + static_cast<size_t>(ik) * ns2, ns, ns) = F;
        }
    };

    // BUBBLE_FD_CHECK: the real-space ladder against the V4 service on fixed
    // inputs (random Hermitian, random general complex, one off-diagonal unit
    // pair, random with acoustic components) at every k; compared on the
    // irreducible k-points the service returns.
    if (report && bubble_fd_check) {
        const Index m = 4;
        const auto nirr = static_cast<size_t>(kmesh_coarse->nk_irred);
        const size_t col = static_cast<size_t>(nk) * ns2;
        std::vector<cplx> din(m * col), fv4(m * nirr * ns2), frs(m * col);
        std::mt19937 rng(20260927);
        std::normal_distribution<double> gauss(0.0, 1.0);
        for (Index c = 0; c < m; ++c) {
            for (unsigned int ik = 0; ik < nk; ++ik) {
                Map<MatrixXcdRow> D(din.data() + c * col + static_cast<size_t>(ik) * ns2, ns, ns);
                MatrixXcd R(ns, ns);
                for (Index a = 0; a < ns; ++a)
                    for (Index b = 0; b < ns; ++b) R(a, b) = cplx(gauss(rng), c == 3 ? 0.0 : gauss(rng));
                if (c == 0) D = 0.5 * (R + R.adjoint());
                if (c == 1 || c == 3) D = R;
                if (c == 2) {
                    D.setZero();
                    if (ik == ikg && nopt > 0) D(optical[0], optical[nopt > 1 ? 1 : 0]) = cplx(1.0, 0.5);
                }
            }
        }
        v4_service->fmat_batch(din.data(), static_cast<size_t>(m), fv4.data());
        double diff = 0.0, scale = 0.0;
        for (Index c = 0; c < m; ++c) {
            fmat_real_space(din.data() + c * col, frs.data() + c * col);
            for (size_t ir = 0; ir < nirr; ++ir) {
                const auto knum = kmesh_coarse->kpoint_irred_all[ir][0].knum;
                const auto kd = static_cast<size_t>(kmap_coarse_to_dense[knum]);
                for (Index i = 0; i < ns2; ++i) {
                    const auto f4 = fv4[(c * nirr + ir) * ns2 + i];
                    diff = std::max(diff, std::abs(frs[c * col + kd * ns2 + i] - f4));
                    scale = std::max(scale, std::abs(f4));
                }
            }
        }
        std::cout << "  BUBBLE_FD_CHECK: real-space quartic ladder vs V4 service: max |F_rs - F_v4| / max |F_v4| = "
                  << std::scientific << std::setprecision(3) << diff / std::max(scale, 1.0e-300) << " ("
                  << quartic_rs->weight.size() << " folded FC4 entries, " << nirr << " irreducible k)"
                  << std::defaultfloat << '\n';
    }

    // ---- K = V4 T on blocks of right-hand sides (one batched V4 sweep)
    std::vector<cplx> dmat, fout;
    const auto apply_K = [&](const MatrixXcd &Yblk, MatrixXcd &out) {
        const auto m = static_cast<Index>(Yblk.cols());
        dmat.resize(static_cast<size_t>(m) * nk * ns2);
        fout.resize(static_cast<size_t>(m) * nk * ns2);
#pragma omp parallel for schedule(dynamic, 1)
        for (Index c = 0; c < m; ++c) apply_T(Yblk.col(c).data(), dmat.data() + static_cast<size_t>(c) * nk * ns2);
        if (bubble_ladder == 2) {
#pragma omp parallel for schedule(dynamic, 1)
            for (Index c = 0; c < m; ++c) fmat_real_space(dmat.data() + c * nk * ns2, fout.data() + c * nk * ns2);
        } else {
            v4_service->fmat_batch(dmat.data(), static_cast<size_t>(m), fout.data());
        }
        out.resize(nk * ns2, m);
        for (Index c = 0; c < m; ++c) out.col(c) = Map<const VectorXcd>(fout.data() + c * nk * ns2, nk * ns2);
        if (project) {
#pragma omp parallel for schedule(dynamic, 1)
            for (Index c = 0; c < m; ++c) apply_P(out.col(c).data());
        }
    };

    // ---- sources B_j and the solve, in chunks of right-hand sides that bound
    //      the memory: per right-hand side the Krylov basis (restart + 1), B, Y,
    //      KY, R and the D and response blocks, each nk ns^2; the real-space
    //      ladder also holds x and z (2 blocks) per OpenMP thread
    const auto four_n = 4.0 * static_cast<double>(nk);
    const int restart = 30, max_restarts = 60;
    const double block_bytes = static_cast<double>(nk) * static_cast<double>(ns2) * sizeof(cplx);
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    const double fixed_bytes = bubble_ladder == 2 ? 2.0 * nthreads * block_bytes : 0.0;
    const auto chunk =
        std::max<Index>(1, static_cast<Index>(std::max(0.0, 8.0e9 - fixed_bytes) / ((restart + 7.0) * block_bytes)));

    MatrixXd A(nopt, nopt);
    const MatrixXcd F_gamma =
        Ck[ikg] * VectorXd(Map<const VectorXd>(omega2_scp[ikg], ns)).cast<cplx>().asDiagonal() * Ck[ikg].adjoint();
    for (Index i = 0; i < nopt; ++i) {
        for (Index j = 0; j < nopt; ++j) A(i, j) = F_gamma(optical[i], optical[j]).real();
    }

    // ---- strain block (cell relaxation, fixed strain): the state variables of
    //      the optimizer, v_m = u_mm (m < 3) and the shear u_ij = u_ji (m >= 3,
    //      (i, j) = (m+1, m+2) mod 3). The vertex of v_m is dPhi_Gamma/dv_m at
    //      fixed occupations; it is also the source of the response.
    const bool with_strain = uses_full_strain_derivatives(ws.relax_mode);
    const Index nv = with_strain ? 6 : 0;
    const Index ntot = nopt + nv;
    std::vector<std::vector<MatrixXcd>> Vstrain(nv); // [m][k]
    for (Index m = 0; m < nv; ++m) {
        const auto [i, j] = strain_state_pair(m);
        for (unsigned int ik = 0; ik < nk; ++ik) {
            Vstrain[m].push_back(strain_vertex(*ws.del_v_strain,
                                               solved_state.u_tensor,
                                               solved_state.q0,
                                               3 * i + j,
                                               static_cast<int>(ik),
                                               static_cast<int>(nk)));
            if (i != j) {
                Vstrain[m][ik] += strain_vertex(*ws.del_v_strain,
                                                solved_state.u_tensor,
                                                solved_state.q0,
                                                3 * j + i,
                                                static_cast<int>(ik),
                                                static_cast<int>(nk));
            }
        }
    }
    const auto factor2 = 1.0 / four_n;

    // explicit part E = dg/dx at fixed occupations: A for the displacements;
    // the strain columns by central differences of the gradient at fixed G
    // (a polynomial in the strain, of degree <= 5 with C3 through the
    // Green-Lagrange strain: second-order truncation error in h),
    // the displacement-strain rows by the symmetry of E.
    MatrixXd E = MatrixXd::Zero(ntot, ntot);
    E.topLeftCorner(nopt, nopt) = A;
    if (with_strain) {
        const double h = 1.0e-4;
        const auto saved_state = ws.structure_state;
        NDArray<cplx, 1> v1(ns), dv0(9);
        const auto gradient_at = [&](const RelaxationStructureState &state) {
            ws.structure_state = state;
            renormalize_ifcs_at_structure(ws);
            compute_anharmonic_v1_array(v1,
                                        ws.v1_renorm,
                                        ws.v3_renorm,
                                        cmat_convert,
                                        omega2_scp,
                                        temp,
                                        kmesh_dense.get());
            compute_anharmonic_del_v0_del_umn(dv0,
                                              ws.del_v0_del_umn_renorm,
                                              *ws.del_v_strain,
                                              state.u_tensor,
                                              state.q0,
                                              cmat_convert,
                                              omega2_scp,
                                              temp,
                                              kmesh_dense.get());
            VectorXd g(ntot);
            for (Index i = 0; i < nopt; ++i) g(i) = v1[optical[i]].real();
            for (Index m = 0; m < nv; ++m) {
                const auto [i, j] = strain_state_pair(m);
                g(nopt + m) = dv0[3 * i + j].real() + (i != j ? dv0[3 * j + i].real() : 0.0);
            }
            return g;
        };
        for (Index m = 0; m < nv; ++m) {
            const auto [i, j] = strain_state_pair(m);
            auto plus = solved_state, minus = solved_state;
            plus.u_tensor[i][j] += h;
            minus.u_tensor[i][j] -= h;
            if (i != j) {
                plus.u_tensor[j][i] += h;
                minus.u_tensor[j][i] -= h;
            }
            E.col(nopt + m) = (gradient_at(plus) - gradient_at(minus)) / (2.0 * h);
        }
        // the IFCs of the solved structure back in ws
        ws.structure_state = solved_state;
        renormalize_ifcs_at_structure(ws);
        ws.structure_state = saved_state;
        E.block(nopt, 0, nv, nopt) = E.block(0, nopt, nopt, nv).transpose();

        // BUBBLE_FD_CHECK (1 or 2): the transpose above assumes that the force and the
        // stress at fixed occupations derive from one function. Check it
        // independently: the stress differentiated over the displacements.
        if (report && bubble_fd_check && nopt > 0) {
            const double hq = 1.0e-4;
            MatrixXd E_vq(nv, nopt);
            std::vector<cplx> st(9), dv(9);
            std::array<std::array<double, 3>, 3> eta{};
            relaxation->calculate_eta_tensor(eta, solved_state.u_tensor);
            const auto stress_at = [&](const std::vector<double> &q) {
                calculate_del_v0_del_umn_renorm(st.data(),
                                                ws.C1_array,
                                                ws.C2_array,
                                                ws.C3_array,
                                                eta,
                                                solved_state.u_tensor,
                                                *ws.del_v_strain,
                                                q,
                                                ws.pvcell,
                                                kmesh_dense.get());
                compute_anharmonic_del_v0_del_umn(dv.data(),
                                                  st.data(),
                                                  *ws.del_v_strain,
                                                  solved_state.u_tensor,
                                                  q,
                                                  cmat_convert,
                                                  omega2_scp,
                                                  temp,
                                                  kmesh_dense.get());
                VectorXd g(nv);
                for (Index m = 0; m < nv; ++m) {
                    const auto [i, j] = strain_state_pair(m);
                    g(m) = dv[3 * i + j].real() + (i != j ? dv[3 * j + i].real() : 0.0);
                }
                return g;
            };
            for (Index j = 0; j < nopt; ++j) {
                auto qp = solved_state.q0, qm = solved_state.q0;
                qp[optical[j]] += hq;
                qm[optical[j]] -= hq;
                E_vq.col(j) = (stress_at(qp) - stress_at(qm)) / (2.0 * hq);
            }
            const auto scale = std::max(E.block(0, nopt, nopt, nv).cwiseAbs().maxCoeff(), 1.0e-300);
            std::cout << "  BUBBLE_FD_CHECK: explicit stress-displacement block vs transpose of force-strain block:"
                      << " max diff / max = " << std::scientific << std::setprecision(3)
                      << (E_vq - E.block(nopt, 0, nv, nopt)).cwiseAbs().maxCoeff() / scale << " (max " << scale << ")"
                      << std::defaultfloat << '\n';
        }
    }

    int total_applications = 0;
    double worst_residual = 0.0;
    const auto t_solve = timer->elapsed();
    J.resize(ntot, ntot);
    std::vector<cplx> dG(static_cast<size_t>(nk) * ns2);
    for (Index j0 = 0; j0 < ntot; j0 += chunk) {
        const auto m = std::min(chunk, ntot - j0);
        // sources at every k: dPhi_k/dq_j = 4N v3_renorm[k][j]^T, dPhi_k/dv_m = M_k
        MatrixXcd B(static_cast<Index>(nk) * ns2, m);
        for (Index c = 0; c < m; ++c) {
            const auto col = j0 + c;
            for (unsigned int ik = 0; ik < nk; ++ik) {
                const auto off = static_cast<Index>(ik) * ns2;
                for (Index a = 0; a < ns; ++a) {
                    for (Index b = 0; b < ns; ++b) {
                        B(off + a * ns + b, c) = col < nopt ? four_n * ws.v3_renorm[ik][optical[col]][b * ns + a]
                                                            : Vstrain[col - nopt][ik](a, b);
                    }
                }
            }
        }
        if (project) {
            for (Index c = 0; c < m; ++c) apply_P(B.col(c).data());
        }
        MatrixXcd Y;
        if (bubble_ladder) {
            std::vector<double> residual;
            int napply = 0;
            const bool converged =
                scph_hessian::gmres_identity_minus(B, Y, apply_K, restart, max_restarts, bubble_tol, residual, napply);
            total_applications += napply;
            for (const auto r: residual) worst_residual = std::max(worst_residual, r);
            if (!converged) return skip("the response equation did not converge to BUBBLE_TOL (or is not finite)");
        } else {
            Y = B; // static bubble only
        }
        for (Index c = 0; c < m; ++c) {
            apply_T(Y.col(c).data(), dG.data());
            for (Index i = 0; i < ntot; ++i) {
                cplx resp(0.0, 0.0);
                if (i < nopt) {
                    for (unsigned int ik = 0; ik < nk; ++ik) {
                        const cplx *v3 = ws.v3_renorm[ik][optical[i]];
                        const cplx *g = dG.data() + static_cast<size_t>(ik) * ns2;
                        for (Index ab = 0; ab < ns2; ++ab) resp += v3[ab] * g[ab];
                    }
                } else {
                    // stress: sum_k sum_ab M_k(a,b) dG_k(b,a) / (4N)
                    for (unsigned int ik = 0; ik < nk; ++ik) {
                        const Map<const MatrixXcdRow> dGk(dG.data() + static_cast<size_t>(ik) * ns2, ns, ns);
                        resp += factor2 * Vstrain[i - nopt][ik].cwiseProduct(dGk.transpose()).sum();
                    }
                }
                J(i, j0 + c) = E(i, j0 + c) + resp.real();
            }
        }
    }
    if (!J.allFinite()) return skip("the Jacobian is not finite");

    const VectorXd dscale = diagonal_scale(J);
    const MatrixXd Jn = dscale.asDiagonal() * J * dscale.asDiagonal();
    const auto asym = (Jn - Jn.transpose()).norm() / std::max(Jn.norm(), 1.0e-300);
    const MatrixXd Jqq = J.topLeftCorner(nopt, nopt);
    if (!report) {
        if (asym > 1.0e-6) return skip("the force Jacobian is not symmetric");
        std::cout << " BUBBLE_HESS: optimizer Hessian from the free-energy curvature"
                  << (with_strain ? " (with strain)" : "") << ", lowest " << std::fixed << std::setprecision(4)
                  << (nopt > 0 ? signed_frequencies(Jqq)(0) : 0.0) << std::defaultfloat << " cm^-1 ("
                  << total_applications << " applications of V4)\n";
        return true;
    }

    std::cout << "  response solve: "
              << (bubble_ladder == 2 ? "GMRES, real-space ladder"
                                     : (bubble_ladder ? "GMRES" : "none (BUBBLE_LADDER = 0)"))
              << ", " << ntot << " right-hand sides";
    if (bubble_ladder) {
        std::cout << ", " << total_applications << (bubble_ladder == 2 ? " ladder applications" : " applications of V4")
                  << ", max relative residual " << std::scientific << std::setprecision(2) << worst_residual
                  << std::defaultfloat;
    }
    std::cout << " (solve " << std::fixed << std::setprecision(1) << timer->elapsed() - t_solve << " s, curvature "
              << timer->elapsed() - t_start << " s)" << std::defaultfloat << '\n';

    std::cout << "  asymmetry |J - J^T| / |J| (diagonally scaled) = " << std::scientific << std::setprecision(2) << asym
              << std::defaultfloat << '\n';
    // A Jacobian that is not symmetric is not the Hessian of a free energy;
    // its eigenvalues are not reported as curvatures.
    if (asym > 1.0e-6) return skip("the force Jacobian is not symmetric, so it is not a free-energy curvature");

    // elastic curvature of Fbar = F + pV per reference cell, Voigt (engineering
    // shear), GPa: clamped ions (the displacement block held) and relaxed ions
    // (its Schur complement); the phonon occupations respond in both.
    MatrixXd C_clamped, C_relaxed;
    if (with_strain) {
        const MatrixXd Js = 0.5 * (J + J.transpose());
        const MatrixXd Hvv = Js.bottomRightCorner(nv, nv);
        const MatrixXd Hvq = Js.bottomLeftCorner(nv, nopt);
        // relaxed ions only where the atoms have a minimum to relax into
        bool ions_relax = true;
        MatrixXd Hrel = Hvv;
        if (nopt > 0) {
            const SelfAdjointEigenSolver<MatrixXd> es_qq(Js.topLeftCorner(nopt, nopt));
            ions_relax = es_qq.eigenvalues()(0) > 1.0e-8 * es_qq.eigenvalues().cwiseAbs().maxCoeff();
            if (ions_relax) Hrel -= Hvq * Js.topLeftCorner(nopt, nopt).llt().solve(Hvq.transpose());
        }
        const auto gpa_to_ry_bohr3 = 1.0e9 / Ryd * std::pow(Bohr_in_Angstrom, 3) * 1.0e-30;
        VectorXd w(nv);
        w << 1.0, 1.0, 1.0, 0.5, 0.5, 0.5;
        const auto unit = 1.0 / (system->get_primcell().volume * gpa_to_ry_bohr3);
        C_clamped = unit * w.asDiagonal() * Hvv * w.asDiagonal();
        if (ions_relax) {
            C_relaxed = unit * w.asDiagonal() * Hrel * w.asDiagonal();
        } else {
            std::cout << "  relaxed-ion elastic curvature not given: the displacement curvature is not positive"
                         " definite.\n";
        }
    }

    write_scp_hessian(temp, "", Jqq, A, total_applications, worst_residual, asym, C_clamped, C_relaxed);

    // J - A in the Cartesian basis of the Gamma dynamical matrix: added to the
    // SCPH correction in PREFIX.scph_fe.h5 (the translations carry none).
    {
        const auto NT = system->get_num_temperature_points();
        if (fe_dymat_correction.size() != NT) fe_dymat_correction.assign(NT, MatrixXcd());
        MatrixXcd U_opt(ns, nopt);
        for (Index i = 0; i < nopt; ++i) U_opt.col(i) = Uk[ikg].col(optical[i]);
        const MatrixXd dJ = 0.5 * (Jqq + Jqq.transpose()) - A;
        fe_dymat_correction[iT] = U_opt * dJ.cast<cplx>() * U_opt.adjoint();
    }
    if (nopt > 0) export_unstable_directions(solved_state, optical, Jqq, temp);
    return true;
}

void Scph::write_fe_state_h5(const NDArray<std::complex<double>, 4> &delta_dymat_scph,
                             NDArray<std::complex<double>, 4> &delta_harmonic_dymat_renormalize, const unsigned int NT)
{
    // The SCPH state file again, with the Gamma correction J - A of the
    // free-energy curvature added to the SCPH one. Temperatures without a
    // curvature are flagged unconverged so that consumers refuse them.
    if (kmesh_coarse->nk > 1) {
        // ponytail: Gamma-only correction; a mesh needs every Q (finite-Q Hessian) before this file is meaningful
        if (run.my_rank == 0) {
            std::cout << "\n PREFIX.scph_fe.h5 is not written on a k-mesh: the curvature correction is known at"
                         " Gamma only.\n";
        }
        return;
    }
    const auto ns = dynamical->neval;
    NDArray<std::complex<double>, 4> delta_fe;
    delta_fe.resize(NT, ns, ns, kmesh_coarse->nk);
    const auto ik_gamma = 0u; // KMESH_INTERPOLATE = 1 1 1 here
    const auto saved_flags = converged_scph_temp;
    std::string missing;
    for (unsigned int iT = 0; iT < NT; ++iT) {
        const bool have = iT < fe_dymat_correction.size() && fe_dymat_correction[iT].size() > 0;
        for (unsigned int is = 0; is < ns; ++is) {
            for (unsigned int js = 0; js < ns; ++js) {
                for (unsigned int ik = 0; ik < kmesh_coarse->nk; ++ik) {
                    delta_fe[iT][is][js][ik] = delta_dymat_scph[iT][is][js][ik];
                }
                if (have) delta_fe[iT][is][js][ik_gamma] += fe_dymat_correction[iT](is, js);
            }
        }
        if (!have) {
            if (converged_scph_temp.size() == NT) converged_scph_temp[iT] = 0;
            missing += " " + std::to_string(system->Tmin + system->dT * iT);
        }
    }
    const auto with_relax = to_relaxation_str_mode(relaxation->relax_str) != RelaxationStrMode::None;
    write_scph_state_h5(run.job_title + ".scph_fe.h5",
                        "SCPH",
                        NT,
                        dynamical->nonanalytic,
                        selfenergy_offdiagonal,
                        relaxation->relax_str,
                        "scph",
                        delta_fe.ptr(),
                        with_relax ? delta_harmonic_dymat_renormalize.ptr() : nullptr,
                        with_relax ? &V0 : nullptr,
                        kmesh_coarse.get(),
                        mindist_list);
    converged_scph_temp = saved_flags;
    if (!missing.empty()) {
        warn("write_fe_state_h5",
             ("No free-energy curvature at T =" + missing +
              " K; those temperatures are flagged unconverged in PREFIX.scph_fe.h5.")
                 .c_str());
    }
}

void Scph::export_unstable_directions(const RelaxationStructureState &solved_state, const std::vector<int> &optical,
                                      const Eigen::MatrixXd &J, const double temp)
{
    // Every negative curvature: the space group the solved structure takes
    // when moved along it (largest atomic step 0.05 bohr; its isotropy
    // subgroup, for a nondegenerate direction), and that structure as
    // &displace (DISPMODE = 1) and &strain blocks to restart the optimization from.
    using namespace Eigen;
    const SelfAdjointEigenSolver<MatrixXd> es(0.5 * (J + J.transpose()));
    const auto natmin = system->get_primcell().number_of_atoms;
    const auto ns = static_cast<size_t>(dynamical->neval);
    const double amplitude = 0.05;

    std::ofstream ofs;
    for (Index k = 0; k < es.eigenvalues().size() && es.eigenvalues()(k) < 0.0; ++k) {
        if (!ofs.is_open()) {
            ofs.open(run.job_title + ".scph_hessian_displace",
                     hessian_displace_started ? std::ios::app : std::ios::out);
            if (!ofs) exit("export_unstable_directions", "cannot open PREFIX.scph_hessian_displace");
            if (!hessian_displace_started) {
                ofs << "# Unstable directions of the SCP free energy (BUBBLE = 4): each block is a &displace\n"
                       "# field (DISPMODE = 1, Cartesian, bohr) and a &strain field of the converged structure\n"
                       "# moved 0.05 bohr (largest atomic step) along one negative-curvature eigenvector,\n"
                       "# for a new RELAX_STR run in the lower symmetry.\n\n";
                hessian_displace_started = true;
            }
        }
        const auto lambda = es.eigenvalues()(k);
        Index degeneracy = 0;
        for (Index l = 0; l < es.eigenvalues().size(); ++l) {
            if (std::abs(es.eigenvalues()(l) - lambda) <= 1.0e-6 * std::abs(lambda)) ++degeneracy;
        }

        // Cartesian step over the same optical modes as J (calculate_u0 would
        // drop harmonic modes below its coarser cutoff).
        std::vector<double> dq(ns, 0.0), du(ns, 0.0);
        const auto &mass = system->get_mass_prim();
        for (size_t i = 0; i < optical.size(); ++i) {
            dq[optical[i]] = es.eigenvectors()(static_cast<Index>(i), k);
            for (size_t j = 0; j < ns; ++j) {
                du[j] += evec_harmonic[0][optical[i]][j].real() * dq[optical[i]] / std::sqrt(mass[j / 3]);
            }
        }
        double umax = 0.0;
        for (size_t iat = 0; iat < natmin; ++iat) {
            umax = std::max(umax, std::hypot(du[3 * iat], du[3 * iat + 1], du[3 * iat + 2]));
        }
        if (umax <= 0.0) continue;
        const double scale = amplitude / umax;

        auto displaced = solved_state;
        for (size_t is = 0; is < ns; ++is) {
            displaced.q0[is] += scale * dq[is];
            displaced.u0[is] += scale * du[is];
        }
        std::string subgroup;
        relaxation->spacegroup_of(displaced, &subgroup);

        const auto freq = -std::sqrt(-lambda) * Ry_to_kayser;
        std::cout << "  unstable direction " << k + 1 << ": " << std::fixed << std::setprecision(4) << freq
                  << std::defaultfloat << " cm^-1"
                  << (degeneracy > 1 ? " (degenerate x" + std::to_string(degeneracy) + ")" : "")
                  << ", displaced structure: " << subgroup << '\n';

        ofs << "# T = " << temp << " K, direction " << k + 1 << ", " << std::fixed << std::setprecision(4) << freq
            << " cm^-1, displaced structure " << subgroup;
        if (degeneracy > 1) {
            ofs << " (one arbitrary member of a " << degeneracy << "-fold set; others may give other subgroups)";
        }
        // The whole initial structure of the new run: the converged
        // displacements and strain plus the step along the direction.
        ofs << "\n&displace\n 1\n" << std::scientific << std::setprecision(10);
        for (size_t iat = 0; iat < natmin; ++iat) {
            for (auto x = 0; x < 3; ++x) ofs << std::setw(20) << displaced.u0[3 * iat + x];
            ofs << '\n';
        }
        ofs << "/\n&strain\n";
        for (const auto &row: solved_state.u_tensor) {
            for (const auto v: row) ofs << std::setw(20) << v;
            ofs << '\n';
        }
        ofs << "/\n\n" << std::defaultfloat;
    }
}

void Scph::write_scp_hessian(const double temp, const std::string &skip_reason, const Eigen::MatrixXd &J,
                             const Eigen::MatrixXd &A, const int n_applications, const double residual,
                             const double asymmetry, const Eigen::MatrixXd &C_clamped, const Eigen::MatrixXd &C_relaxed)
{
    using namespace Eigen;
    const auto nopt = J.rows();
    VectorXd wJ, wA;
    if (nopt > 0) {
        wJ = signed_frequencies(J);
        wA = signed_frequencies(A);
        std::cout << "  lowest curvature frequencies (cm^-1), SCPH -> free energy:\n";
        for (Index i = 0; i < std::min<Index>(nopt, 12); ++i) {
            std::cout << "   " << std::setw(4) << i + 1 << std::fixed << std::setprecision(4) << std::setw(14) << wA(i)
                      << std::setw(14) << wJ(i) << std::defaultfloat << '\n';
        }
    }

    const auto fname = run.job_title + ".scph_hessian";
    std::ofstream ofs(fname, hessian_file_started ? std::ios::app : std::ios::out);
    if (!ofs) exit("write_scp_hessian", "cannot open PREFIX.scph_hessian");
    if (!hessian_file_started) {
        ofs << "# Curvature of the SCP free energy with respect to the Gamma displacements (BUBBLE = 4):\n"
               "# the force Jacobian dg/dq0 at the converged structure. Frequencies sign(l) sqrt(|l|) of\n"
               "# the eigenvalues l of its symmetric part (reported only when it is symmetric), in cm^-1;\n"
               "# negative values are directions in which the structure is not a free-energy minimum.\n"
               "# Columns: index, SCPH (fixed occupations), free-energy curvature\n";
        hessian_file_started = true;
    }
    ofs << "# T = " << temp << " K";
    if (!skip_reason.empty()) {
        ofs << "  skipped: " << skip_reason << "\n\n";
        return;
    }
    ofs << "  asymmetry = " << std::scientific << std::setprecision(3) << asymmetry
        << "  V4 applications = " << n_applications << "  residual = " << residual << '\n';
    ofs << std::fixed << std::setprecision(6);
    for (Index i = 0; i < nopt; ++i) {
        ofs << std::setw(6) << i + 1 << std::setw(16) << wA(i) << std::setw(16) << wJ(i) << '\n';
    }
    if (C_clamped.size() == 36) {
        const auto print_voigt = [&](std::ostream &os, const char *label, const MatrixXd &C, const char *lead) {
            const SelfAdjointEigenSolver<MatrixXd> es(C);
            os << lead << label << " (lowest eigenvalue " << std::fixed << std::setprecision(3) << es.eigenvalues()(0)
               << "):\n";
            for (Index i = 0; i < 6; ++i) {
                os << lead << "  ";
                for (Index j = 0; j < 6; ++j) os << std::setw(11) << C(i, j);
                os << '\n';
            }
        };
        const char *head = "elastic curvature of F + pV (GPa, Voigt, reference cell; phonon occupations respond)";
        std::cout << "  " << head << ":\n";
        print_voigt(std::cout, "clamped ions", C_clamped, "   ");
        if (C_relaxed.size() == 36) print_voigt(std::cout, "relaxed ions", C_relaxed, "   ");
        std::cout << std::defaultfloat;
        ofs << "# " << head << '\n';
        print_voigt(ofs, "clamped ions", C_clamped, "# ");
        if (C_relaxed.size() == 36) print_voigt(ofs, "relaxed ions", C_relaxed, "# ");
        ofs << std::fixed << std::setprecision(6);
    }
    ofs << '\n';
}

void Scph::report_scp_hessian_fd_check(const Eigen::MatrixXd &J, const Eigen::MatrixXd &jacobian_fd,
                                       const Eigen::Index nv) const
{
    if (J.rows() == 0 || jacobian_fd.rows() != J.rows()) return;
    const Eigen::VectorXd d = diagonal_scale(J);
    const Eigen::MatrixXd Jn = d.asDiagonal() * J * d.asDiagonal();
    const Eigen::MatrixXd Dn = d.asDiagonal() * (J - jacobian_fd) * d.asDiagonal();
    const auto norm = std::max(Jn.cwiseAbs().maxCoeff(), 1.0e-300);
    std::cout << "  BUBBLE_FD_CHECK: max |J - J_fd| / max |J| = " << std::scientific << std::setprecision(3)
              << Dn.cwiseAbs().maxCoeff() / norm;
    if (nv > 0 && J.rows() > nv) {
        const auto nq = J.rows() - nv;
        std::cout << "  (blocks qq " << Dn.topLeftCorner(nq, nq).cwiseAbs().maxCoeff() / norm << ", qu "
                  << Dn.topRightCorner(nq, nv).cwiseAbs().maxCoeff() / norm << ", uq "
                  << Dn.bottomLeftCorner(nv, nq).cwiseAbs().maxCoeff() / norm << ", uu "
                  << Dn.bottomRightCorner(nv, nv).cwiseAbs().maxCoeff() / norm << "; max |J_qu| "
                  << Jn.topRightCorner(nq, nv).cwiseAbs().maxCoeff() / norm << ")";
        // each block relative to its own size, so that a small block cannot
        // hide behind the global normalization
        const auto rel = [](const Eigen::MatrixXd &d, const Eigen::MatrixXd &j) {
            return d.cwiseAbs().maxCoeff() / std::max(j.cwiseAbs().maxCoeff(), 1.0e-300);
        };
        std::cout << "\n  BUBBLE_FD_CHECK: per block, relative to the block (blocks qq "
                  << rel(Dn.topLeftCorner(nq, nq), Jn.topLeftCorner(nq, nq)) << ", qu "
                  << rel(Dn.topRightCorner(nq, nv), Jn.topRightCorner(nq, nv)) << ", uq "
                  << rel(Dn.bottomLeftCorner(nv, nq), Jn.bottomLeftCorner(nv, nq)) << ", uu "
                  << rel(Dn.bottomRightCorner(nv, nv), Jn.bottomRightCorner(nv, nv)) << ";)";
    }
    std::cout << std::defaultfloat << '\n';
}
