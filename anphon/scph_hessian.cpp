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
#include <sstream>
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
#include "ifc_derivative.h"
#include "interpolation.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "quartic_real_space.h"
#include "relaxation.h"
#include "scph.h"
#include "scph_hessian_kernels.h"
#include "thermodynamics.h"
#include "timer.h"
#include "v4_service.h"

using namespace PHON_NS;
using quartic_rs::CubicMesh;
using quartic_rs::QuarticMesh;

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
    std::vector<VectorXd> lambda_k(nk);
    for (unsigned int ik = 0; ik < nk; ++ik) {
        Uk[ik].resize(ns, ns);
        Ck[ik].resize(ns, ns);
        auto &lambda = lambda_k[ik];
        lambda.resize(ns);
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
    //      With momentum transfer Q the blocks couple k+Q (rows) to k (columns):
    //      X_k = U_{k+Q} D_k U_k^dagger, x(R) = (1/N) sum_k e^{-ik.R} X_k,
    //      F_k = U_{k+Q}^dagger [sum_R e^{ik.R} z(R)] U_k, z from x with the phase
    //      e^{iQ.R_l} on the third leg (a block operator O(R1, R2) = e^{iQ.R1} x(R2 - R1)).
    const bool finite_q = report && nk > 1;
    std::unique_ptr<QuarticMesh> quartic_rs;
    if (bubble_ladder == 2 || (report && bubble_fd_check) || (finite_q && bubble_ladder)) {
        quartic_rs = std::make_unique<QuarticMesh>(fcs_phonon->force_constant_with_cell[2],
                                                   system->get_invsqrt_mass(),
                                                   static_cast<unsigned int>(ns),
                                                   kmesh_dense->nk_i);
    }
    std::vector<std::vector<cplx>> kphase(nk, std::vector<cplx>(nk)); // e^{2 pi i k.R}, R a cell of the mesh
    for (unsigned int ik = 0; ik < nk; ++ik) {
        for (unsigned int r = 0; r < nk; ++r) {
            int n[3], rr = static_cast<int>(r);
            for (int d = 2; d >= 0; --d) {
                n[d] = rr % static_cast<int>(kmesh_dense->nk_i[d]);
                rr /= static_cast<int>(kmesh_dense->nk_i[d]);
            }
            double arg = 0.0;
            for (int d = 0; d < 3; ++d) arg += kmesh_dense->xk[ik][d] * n[d];
            kphase[ik][r] = std::exp(cplx(0.0, 2.0 * pi * arg));
        }
    }
    std::vector<unsigned int> k_identity(nk);
    for (unsigned int ik = 0; ik < nk; ++ik) k_identity[ik] = ik;
    // the acoustic Gamma rows (left leg k+Q = Gamma) and columns (right leg k = Gamma)
    const auto drop_acoustic = [&](MatrixXcd &M, const unsigned int left, const unsigned int right) {
        for (Index a = 0; a < ns; ++a) {
            if (!is_acoustic_gamma_harm[a]) continue;
            if (left == ikg) M.row(a).setZero();
            if (right == ikg) M.col(a).setZero();
        }
    };
    const auto fmat_q =
        [&](const cplx *d_in, cplx *f_out, const std::vector<unsigned int> &kq, const std::vector<cplx> *qphase) {
            std::vector<MatrixXcd> x(nk, MatrixXcd::Zero(ns, ns)), z;
            for (unsigned int ik = 0; ik < nk; ++ik) {
                MatrixXcd D = Map<const MatrixXcdRow>(d_in + static_cast<size_t>(ik) * ns2, ns, ns);
                drop_acoustic(D, kq[ik], ik);
                const MatrixXcd X = Uk[kq[ik]] * D * Uk[ik].adjoint();
                for (unsigned int r = 0; r < nk; ++r) x[r] += std::conj(kphase[ik][r]) * X;
            }
            for (auto &m: x) m /= static_cast<double>(nk);
            quartic_rs->apply(x, z, qphase);
            for (unsigned int ik = 0; ik < nk; ++ik) {
                MatrixXcd Z = MatrixXcd::Zero(ns, ns);
                for (unsigned int r = 0; r < nk; ++r) Z += kphase[ik][r] * z[r];
                MatrixXcd F = Uk[kq[ik]].adjoint() * Z * Uk[ik];
                drop_acoustic(F, kq[ik], ik);
                Map<MatrixXcdRow>(f_out + static_cast<size_t>(ik) * ns2, ns, ns) = F;
            }
        };
    const auto fmat_real_space = [&](const cplx *d_in, cplx *f_out) { fmat_q(d_in, f_out, k_identity, nullptr); };

    // BUBBLE_FD_CHECK: the real-space ladder against the V4 service on fixed
    // inputs (random Hermitian, random general complex, one off-diagonal unit
    // pair, random with acoustic components) at every k; compared on the
    // irreducible k-points the service returns.
    if (report && bubble_fd_check && v4_real_space != 1) {
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
        if (report && run.verbosity > 1) {
            // full precision (EFIELD checks: the field terms add -Omega_ref E0 . B and -L here)
            std::cout << "  explicit strain-strain block dG_m/ds_n [Ry] (fixed occupations; xx yy zz yz zx xy):\n"
                      << std::scientific << std::setprecision(15);
            for (Index m = 0; m < nv; ++m) {
                std::cout << "   ss";
                for (Index n = 0; n < nv; ++n) std::cout << std::setw(24) << E(nopt + m, nopt + n);
                std::cout << '\n';
            }
            std::cout << "  explicit force-strain block dg_s/ds_n (optical modes s; xx yy zz yz zx xy):\n";
            for (Index i = 0; i < nopt; ++i) {
                std::cout << "   qs" << std::setw(5) << optical[i];
                for (Index n = 0; n < nv; ++n) std::cout << std::setw(24) << E(i, nopt + n);
                std::cout << '\n';
            }
            std::cout << std::defaultfloat;
        }

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

    // H - A in the Cartesian basis of the dynamical matrix, per coarse k-point,
    // added to the SCPH correction in PREFIX.scph_fe.h5: J - A at Gamma (the
    // translations carry none), and on a mesh the finite-Q curvature elsewhere.
    const auto NT = system->get_num_temperature_points();
    if (fe_dymat_correction.size() != NT) fe_dymat_correction.assign(NT, {});
    fe_dymat_correction[iT].clear();
    std::vector<MatrixXcd> correction(kmesh_coarse->nk, MatrixXcd::Zero(ns, ns));
    {
        MatrixXcd U_opt(ns, nopt);
        for (Index i = 0; i < nopt; ++i) U_opt.col(i) = Uk[ikg].col(optical[i]);
        const MatrixXd dJ = 0.5 * (Jqq + Jqq.transpose()) - A;
        correction[0] = U_opt * dJ.cast<cplx>() * U_opt.adjoint();
    }

    // ---- finite Q on the mesh (FINITE_Q_HESSIAN_PLAN.md, step C): the curvature
    //      of the free energy for u(R) = eps e^{iQ.R} + c.c.,
    //        H(Q) = A(Q) + (1/4N) sum_k tr[ B_s(k)^dagger dG_t(k) ],
    //      with A(Q) the SCP matrix at Q, the sources B_s(k) = U_{k+Q}^dagger
    //      Y_s(k) U_k from the cubic IFCs of the solved structure (the change of
    //      the SCP matrix per unit amplitude of harmonic mode s at Q), and the
    //      response dG_t = T_Q (I - K_Q T_Q)^{-1} B_t, T_Q the Daleckii-Krein map
    //      between the spectra at k+Q and k. At Q = 0 this is J_qq.
    bool complete = true;
    if (finite_q) {
        const auto t_q = timer->elapsed();
        std::vector<cplx>().swap(dmat); // the Gamma solve's scratch
        std::vector<cplx>().swap(fout);
        std::vector<cplx>().swap(dG);
        std::vector<FcsArrayWithCell> fc3_solved;
        double u_flat[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) u_flat[3 * i + j] = solved_state.u_tensor[i][j];
        }
        DerivativeIFC::compute_deformed_cubic_ifcs(fcs_phonon->force_constant_with_cell,
                                                   u_flat,
                                                   solved_state.u0,
                                                   system->get_primcell().lattice_vector,
                                                   fc3_solved);
        const CubicMesh cubic(fc3_solved, system->get_invsqrt_mass(), static_cast<unsigned int>(ns), kmesh_dense->nk_i);
        const size_t col = static_cast<size_t>(nk) * ns2;

        const auto k_plus = [&](const unsigned int iq) {
            std::vector<unsigned int> kq(nk);
            for (unsigned int ik = 0; ik < nk; ++ik) {
                double xk[3];
                for (int d = 0; d < 3; ++d) xk[d] = kmesh_dense->xk[ik][d] + kmesh_dense->xk[iq][d];
                const auto n = kmesh_dense->get_knum(xk);
                if (n < 0) exit("compute_scp_hessian", "k + Q is not on the mesh.");
                kq[ik] = static_cast<unsigned int>(n);
            }
            return kq;
        };
        // sources of the harmonic modes s0 .. s0 + m - 1 at Q (columns), blocks (k+Q, k)
        const auto sources_at =
            [&](const unsigned int iq, const std::vector<unsigned int> &kq, const Index s0, const Index m) {
                MatrixXcd B(static_cast<Index>(col), m);
#pragma omp parallel for schedule(dynamic, 1)
                for (Index c = 0; c < m; ++c) {
                    std::vector<MatrixXcd> y;
                    cubic.source(Uk[iq].col(s0 + c), kphase[iq], y);
                    for (unsigned int ik = 0; ik < nk; ++ik) {
                        MatrixXcd Yk = MatrixXcd::Zero(ns, ns);
                        for (unsigned int r = 0; r < nk; ++r) Yk += kphase[ik][r] * y[r];
                        Map<MatrixXcdRow>(B.data() + c * col + static_cast<size_t>(ik) * ns2, ns, ns) =
                            Uk[kq[ik]].adjoint() * Yk * Uk[ik];
                    }
                }
                return B;
            };
        const auto curvature_at = [&](const unsigned int iq, MatrixXcd &H, int &napply_q, double &residual_q) {
            const auto kq = k_plus(iq);
            const auto *qphase = &kphase[iq];
            std::vector<MatrixXcd> LQ(nk, MatrixXcd::Zero(ns, ns));
            for (unsigned int ik = 0; ik < nk; ++ik) {
                for (Index s = 0; s < ns; ++s) {
                    if (kq[ik] == ikg && frozen_gamma[s]) continue;
                    for (Index t = 0; t < ns; ++t) {
                        if (ik == ikg && frozen_gamma[t]) continue;
                        LQ[ik](s, t) = scph_hessian::divided_difference(occ, lambda_k[kq[ik]](s), lambda_k[ik](t));
                    }
                }
            }
            const auto apply_TQ = [&](const cplx *y, cplx *dGq) {
                for (unsigned int ik = 0; ik < nk; ++ik) {
                    const Map<const MatrixXcdRow> Y(y + static_cast<size_t>(ik) * ns2, ns, ns);
                    const MatrixXcd in_eig = Ck[kq[ik]].adjoint() * Y * Ck[ik];
                    Map<MatrixXcdRow>(dGq + static_cast<size_t>(ik) * ns2, ns, ns) =
                        Ck[kq[ik]] * in_eig.cwiseProduct(LQ[ik]) * Ck[ik].adjoint();
                }
            };
            std::vector<cplx> dq, fq;
            const auto apply_KQ = [&](const MatrixXcd &Yb, MatrixXcd &out) {
                const auto m = Yb.cols();
                dq.resize(static_cast<size_t>(m) * col);
                fq.resize(static_cast<size_t>(m) * col);
#pragma omp parallel for schedule(dynamic, 1)
                for (Index c = 0; c < m; ++c) apply_TQ(Yb.col(c).data(), dq.data() + c * col);
#pragma omp parallel for schedule(dynamic, 1)
                for (Index c = 0; c < m; ++c) fmat_q(dq.data() + c * col, fq.data() + c * col, kq, qphase);
                out = Map<const MatrixXcd>(fq.data(), static_cast<Index>(col), m);
            };
            // in chunks of columns t; the rows s pair with sources made again per
            // chunk unless one chunk holds them all (the memory bound of the solve)
            H = Ck[iq] * lambda_k[iq].cast<cplx>().asDiagonal() * Ck[iq].adjoint();
            napply_q = 0;
            residual_q = 0.0;
            for (Index t0 = 0; t0 < ns; t0 += chunk) {
                const auto m = std::min(chunk, ns - t0);
                const MatrixXcd Bc = sources_at(iq, kq, t0, m);
                MatrixXcd Y = Bc;
                if (bubble_ladder) {
                    std::vector<double> residual;
                    int napply = 0;
                    const bool ok = scph_hessian::gmres_identity_minus(Bc,
                                                                       Y,
                                                                       apply_KQ,
                                                                       restart,
                                                                       max_restarts,
                                                                       bubble_tol,
                                                                       residual,
                                                                       napply);
                    napply_q += napply;
                    for (const auto r: residual) residual_q = std::max(residual_q, r);
                    if (!ok) return false;
                }
                MatrixXcd dGc(static_cast<Index>(col), m);
#pragma omp parallel for schedule(dynamic, 1)
                for (Index c = 0; c < m; ++c) apply_TQ(Y.col(c).data(), dGc.col(c).data());
                Y.resize(0, 0);
                for (Index s0 = 0; s0 < ns; s0 += chunk) {
                    const auto ms = std::min(chunk, ns - s0);
                    if (ms == ns && m == ns) {
                        H.middleCols(t0, m) += Bc.adjoint() * dGc / four_n;
                    } else {
                        H.block(s0, t0, ms, m) += sources_at(iq, kq, s0, ms).adjoint() * dGc / four_n;
                    }
                }
            }
            return H.allFinite();
        };

        // BUBBLE_FD_CHECK: the real-space sources and the whole finite-Q path at
        // Q = 0 against the Gamma construction (4N v3_renorm^T and J_qq)
        if (bubble_fd_check && nopt > 0) {
            const MatrixXcd B = sources_at(ikg, k_identity, 0, ns);
            double diff = 0.0, scale = 0.0;
            for (Index j = 0; j < nopt; ++j) {
                for (unsigned int ik = 0; ik < nk; ++ik) {
                    const Map<const MatrixXcdRow> Bk(B.data() + optical[j] * col + static_cast<size_t>(ik) * ns2,
                                                     ns,
                                                     ns);
                    for (Index a = 0; a < ns; ++a) {
                        for (Index b = 0; b < ns; ++b) {
                            if ((ik == ikg) && (is_acoustic_gamma_harm[a] || is_acoustic_gamma_harm[b])) continue;
                            const auto ref = four_n * ws.v3_renorm[ik][optical[j]][b * ns + a];
                            diff = std::max(diff, std::abs(Bk(a, b) - ref));
                            scale = std::max(scale, std::abs(ref));
                        }
                    }
                }
            }
            MatrixXcd H0;
            int n0 = 0;
            double r0 = 0.0;
            double dh = -1.0;
            if (curvature_at(ikg, H0, n0, r0)) {
                MatrixXd H0opt(nopt, nopt);
                for (Index i = 0; i < nopt; ++i)
                    for (Index j = 0; j < nopt; ++j) H0opt(i, j) = H0(optical[i], optical[j]).real();
                dh = (H0opt - Jqq).cwiseAbs().maxCoeff() / std::max(Jqq.cwiseAbs().maxCoeff(), 1.0e-300);
            }
            std::cout << "  BUBBLE_FD_CHECK: finite-Q sources at Q = 0 vs 4N v3_renorm^T: max diff / max = "
                      << std::scientific << std::setprecision(3) << diff / std::max(scale, 1.0e-300)
                      << "\n  BUBBLE_FD_CHECK: finite-Q curvature at Q = 0 vs J_qq: ";
            if (dh >= 0.0) {
                std::cout << "max diff / max = " << dh;
            } else {
                std::cout << "not obtained (the response equation did not converge)";
            }
            std::cout << std::defaultfloat << '\n';
        }

        // the irreducible Q-points of the mesh other than Gamma (rank 0 runs the
        // structural loop; the other ranks serve V4 requests meanwhile)
        std::vector<unsigned int> qlist;
        for (unsigned int ir = 0; ir < kmesh_coarse->nk_irred; ++ir) {
            const auto knum = kmesh_coarse->kpoint_irred_all[ir][0].knum;
            if (knum != 0) qlist.push_back(knum);
        }
        const auto nq = qlist.size();
        std::vector<cplx> Hall(nq * ns2, cplx(0.0, 0.0));
        std::vector<double> stat(3 * nq, 0.0); // success, applications, residual
        for (size_t n = 0; n < nq; ++n) {
            MatrixXcd H;
            int napply = 0;
            double res = 0.0;
            const bool ok = curvature_at(kmap_coarse_to_dense[qlist[n]], H, napply, res);
            if (ok) Map<MatrixXcd>(Hall.data() + n * ns2, ns, ns) = H;
            stat[3 * n] = ok ? 1.0 : 0.0;
            stat[3 * n + 1] = napply;
            stat[3 * n + 2] = res;
        }

        std::vector<std::string> lines;
        int napply_all = 0;
        for (size_t n = 0; n < nq; ++n) {
            const auto iq = kmap_coarse_to_dense[qlist[n]];
            const Map<const MatrixXcd> H(Hall.data() + n * ns2, ns, ns);
            napply_all += static_cast<int>(stat[3 * n + 1]);
            const auto asym_q = (H - H.adjoint()).norm() / std::max(H.norm(), 1.0e-300);
            std::ostringstream head;
            head << "# Q = " << std::fixed << std::setprecision(6) << kmesh_dense->xk[iq][0] << ' '
                 << kmesh_dense->xk[iq][1] << ' ' << kmesh_dense->xk[iq][2];
            if (stat[3 * n] < 0.5 || asym_q > 1.0e-6) {
                complete = false;
                head << "  skipped: "
                     << (stat[3 * n] < 0.5 ? "the response equation did not converge (or is not finite)"
                                           : "H(Q) is not Hermitian")
                     << '\n';
                lines.push_back(head.str());
                continue;
            }
            const MatrixXcd Hh = 0.5 * (H + H.adjoint());
            const MatrixXcd AQ = Ck[iq] * lambda_k[iq].cast<cplx>().asDiagonal() * Ck[iq].adjoint();
            const SelfAdjointEigenSolver<MatrixXcd> es(Hh);
            VectorXd w_scp(ns);
            for (Index s = 0; s < ns; ++s) w_scp(s) = std::sqrt(std::max(lambda_k[iq](s), 0.0)) * Ry_to_kayser;
            std::sort(w_scp.data(), w_scp.data() + ns);
            head << "  asymmetry = " << std::scientific << std::setprecision(3) << asym_q
                 << "  residual = " << stat[3 * n + 2] << '\n'
                 << std::fixed << std::setprecision(6);
            for (Index s = 0; s < ns; ++s) {
                const auto l = es.eigenvalues()(s);
                head << std::setw(6) << s + 1 << std::setw(16) << w_scp(s) << std::setw(16)
                     << (l >= 0.0 ? 1.0 : -1.0) * std::sqrt(std::abs(l)) * Ry_to_kayser << '\n';
            }
            lines.push_back(head.str());
            correction[qlist[n]] = Uk[iq] * (Hh - AQ) * Uk[iq].adjoint();
        }
        std::cout << "  finite Q: " << nq << " irreducible Q-points besides Gamma, " << napply_all
                  << (bubble_ladder ? " ladder applications" : " (no ladder)") << " (" << std::fixed
                  << std::setprecision(1) << timer->elapsed() - t_q << " s)" << std::defaultfloat
                  << (complete ? "" : "; some Q-points skipped, see PREFIX.scph_hessian_q") << '\n';
        {
            const auto fname = run.job_title + ".scph_hessian_q";
            std::ofstream ofs(fname, hessian_q_file_started ? std::ios::app : std::ios::out);
            if (!ofs) exit("compute_scp_hessian", "cannot open PREFIX.scph_hessian_q");
            if (!hessian_q_file_started) {
                ofs << "# Curvature of the SCP free energy at the irreducible Q-points of the mesh other than\n"
                       "# Gamma (BUBBLE = 4; Gamma is in PREFIX.scph_hessian): frequencies sign(l) sqrt(|l|) of\n"
                       "# the eigenvalues l of the Hermitian H(Q), in cm^-1, Q in reciprocal lattice units.\n"
                       "# Columns: index, SCPH (fixed occupations), free-energy curvature\n";
                hessian_q_file_started = true;
            }
            ofs << "# T = " << temp << " K\n";
            for (const auto &l: lines) ofs << l << '\n';
        }
        if (complete)
            replicate_dymat_for_all_kpoints(kmesh_coarse.get(),
                                            static_cast<unsigned int>(ns),
                                            mat_transform_sym,
                                            correction);
    }
    if (complete) fe_dymat_correction[iT] = std::move(correction);
    if (nopt > 0) export_unstable_directions(solved_state, optical, Jqq, temp);
    return true;
}

void Scph::write_fe_state_h5(const NDArray<std::complex<double>, 4> &delta_dymat_scph,
                             NDArray<std::complex<double>, 4> &delta_harmonic_dymat_renormalize, const unsigned int NT)
{
    // The SCPH state file again, with the correction H - A of the free-energy
    // curvature added to the SCPH one at every point of the mesh. Temperatures
    // without a curvature are flagged unconverged so that consumers refuse them.
    // The SCPH correction is stored per cell of the mesh (k -> R as in
    // Dynamical::calc_new_dymat_with_evec); the curvature correction per k-point
    // goes through the same transform.
    const auto ns = dynamical->neval;
    const auto nk = kmesh_coarse->nk;
    NDArray<std::complex<double>, 4> delta_fe;
    delta_fe.resize(NT, ns, ns, nk);
    NDArray<std::complex<double>, 3> corr_k, corr_r;
    corr_k.resize(ns, ns, nk);
    corr_r.resize(ns, ns, nk);
    const auto saved_flags = converged_scph_temp;
    std::string missing;
    for (unsigned int iT = 0; iT < NT; ++iT) {
        const bool have = iT < fe_dymat_correction.size() && fe_dymat_correction[iT].size() == nk;
        if (have) {
            for (unsigned int is = 0; is < ns; ++is)
                for (unsigned int js = 0; js < ns; ++js)
                    for (unsigned int ik = 0; ik < nk; ++ik) corr_k[is][js][ik] = fe_dymat_correction[iT][ik](is, js);
            fourier_dymat_k_to_r(kmesh_coarse->nk_i[0],
                                 kmesh_coarse->nk_i[1],
                                 kmesh_coarse->nk_i[2],
                                 ns,
                                 corr_k,
                                 corr_r);
        }
        for (unsigned int is = 0; is < ns; ++is) {
            for (unsigned int js = 0; js < ns; ++js) {
                for (unsigned int ir = 0; ir < nk; ++ir) {
                    delta_fe[iT][is][js][ir] = delta_dymat_scph[iT][is][js][ir] + (have ? corr_r[is][js][ir] : 0.0);
                }
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
