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
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "interpolation.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "scph.h"
#include "scph_hessian_kernels.h"
#include "thermodynamics.h"
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
} // namespace

bool Scph::compute_scp_hessian(const StructuralOptWorkspace &ws, const unsigned int iT, const double temp,
                               std::complex<double> ***cmat_convert, double **omega2_scp, Eigen::MatrixXd &J)
{
    using namespace Eigen;
    (void)iT;
    const auto nk = kmesh_dense->nk;
    const auto ns = static_cast<Index>(dynamical->neval);
    const auto ns2 = ns * ns;
    const auto &optical = ws.harm_optical_modes;
    const auto nopt = static_cast<Index>(optical.size());
    const auto ikg = static_cast<unsigned int>(kmap_coarse_to_dense[0]);

    std::cout << "\n BUBBLE = 4: curvature of the SCP free energy (force Jacobian dg/dq0) at " << temp << " K\n";

    const auto skip = [&](const std::string &reason) {
        std::cout << "  skipped: " << reason << ".\n";
        write_scp_hessian(temp, reason, MatrixXd(), MatrixXd(), 0, 0.0, 0.0);
        return false;
    };

    // ---- eligibility at the final fixed point
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

    // ---- T: coarse-Gamma change y of the SCP matrix (harmonic basis, row-major
    //      a*ns+b) -> occupation change dG_k on the dense mesh (row-major, block
    //      k at dG + k*ns2). Interpolation as in interpolate_to_dense_mesh, but
    //      without the symmetrization: the response is unrestricted. Buffers
    //      are per call so that columns can run in parallel.
    const auto apply_T = [&](const cplx *y, cplx *dG) {
        NDArray<cplx, 3> dymat_k(ns, ns, 1), dymat_r(ns, ns, 1);
        NDArray<cplx, 2> mat_k(ns, ns);
        const Map<const MatrixXcdRow> Y(y, ns, ns);
        const MatrixXcd cart_gamma = Uk[ikg] * Y * Uk[ikg].adjoint();
        for (Index a = 0; a < ns; ++a) {
            for (Index b = 0; b < ns; ++b) dymat_k[a][b][0] = cart_gamma(a, b);
        }
        fourier_dymat_k_to_r(1, 1, 1, static_cast<unsigned int>(ns), dymat_k, dymat_r);
        MatrixXcd cart(ns, ns);
        for (unsigned int ik = 0; ik < nk; ++ik) {
            r2q(kmesh_dense->xk[ik], 1, 1, 1, static_cast<unsigned int>(ns), mindist_list, dymat_r, mat_k);
            for (Index a = 0; a < ns; ++a) {
                for (Index b = 0; b < ns; ++b) cart(a, b) = mat_k[a][b];
            }
            const MatrixXcd in_eig = Pk[ik].adjoint() * cart * Pk[ik];
            Map<MatrixXcdRow>(dG + static_cast<size_t>(ik) * ns2, ns, ns) =
                Ck[ik] * in_eig.cwiseProduct(Lk[ik]) * Ck[ik].adjoint();
        }
    };

    // ---- K = V4 T on blocks of right-hand sides (one batched V4 sweep)
    std::vector<cplx> dmat, fout;
    const auto apply_K = [&](const MatrixXcd &Yblk, MatrixXcd &out) {
        const auto m = static_cast<Index>(Yblk.cols());
        dmat.resize(static_cast<size_t>(m) * nk * ns2);
        fout.resize(static_cast<size_t>(m) * ns2);
#pragma omp parallel for schedule(dynamic, 1)
        for (Index c = 0; c < m; ++c) apply_T(Yblk.col(c).data(), dmat.data() + static_cast<size_t>(c) * nk * ns2);
        v4_service->fmat_batch(dmat.data(), static_cast<size_t>(m), fout.data());
        out.resize(ns2, m);
        for (Index c = 0; c < m; ++c) out.col(c) = Map<const VectorXcd>(fout.data() + c * ns2, ns2);
    };

    // ---- sources B_j and the solve, in chunks of right-hand sides that bound
    //      the memory: Krylov basis and work blocks (restart + 4) plus the dense
    //      D and response blocks (2 nk), replicated on every rank for the latter
    const auto four_n = 4.0 * static_cast<double>(nk);
    const int restart = 30, max_restarts = 60;
    const double bytes_per_rhs =
        (static_cast<double>(restart + 4) + 2.0 * static_cast<double>(nk)) * static_cast<double>(ns2) * sizeof(cplx);
    const auto chunk = std::max<Index>(1, std::min<Index>(nopt, static_cast<Index>(8.0e9 / bytes_per_rhs)));

    MatrixXd A(nopt, nopt);
    J.resize(nopt, nopt);
    const MatrixXcd F_gamma =
        Ck[ikg] * VectorXd(Map<const VectorXd>(omega2_scp[ikg], ns)).cast<cplx>().asDiagonal() * Ck[ikg].adjoint();
    for (Index i = 0; i < nopt; ++i) {
        for (Index j = 0; j < nopt; ++j) A(i, j) = F_gamma(optical[i], optical[j]).real();
    }

    int total_applications = 0;
    double worst_residual = 0.0;
    std::vector<cplx> dG(static_cast<size_t>(nk) * ns2);
    for (Index j0 = 0; j0 < nopt; j0 += chunk) {
        const auto m = std::min(chunk, nopt - j0);
        MatrixXcd B(ns2, m);
        for (Index c = 0; c < m; ++c) {
            const auto j = optical[j0 + c];
            for (Index a = 0; a < ns; ++a) {
                for (Index b = 0; b < ns; ++b) B(a * ns + b, c) = four_n * ws.v3_renorm[ikg][j][b * ns + a];
            }
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
            for (Index i = 0; i < nopt; ++i) {
                cplx resp(0.0, 0.0);
                for (unsigned int ik = 0; ik < nk; ++ik) {
                    const cplx *v3 = ws.v3_renorm[ik][optical[i]];
                    const cplx *g = dG.data() + static_cast<size_t>(ik) * ns2;
                    for (Index ab = 0; ab < ns2; ++ab) resp += v3[ab] * g[ab];
                }
                J(i, j0 + c) = A(i, j0 + c) + resp.real();
            }
        }
    }
    if (!J.allFinite()) return skip("the Jacobian is not finite");

    std::cout << "  response solve: " << (bubble_ladder ? "GMRES" : "none (BUBBLE_LADDER = 0)") << ", " << nopt
              << " right-hand sides";
    if (bubble_ladder) {
        std::cout << ", " << total_applications << " applications of V4, max relative residual " << std::scientific
                  << std::setprecision(2) << worst_residual << std::defaultfloat;
    }
    std::cout << '\n';

    const auto asym = (J - J.transpose()).norm() / std::max(J.norm(), 1.0e-300);
    std::cout << "  asymmetry |J - J^T| / |J| = " << std::scientific << std::setprecision(2) << asym
              << std::defaultfloat << '\n';
    // A Jacobian that is not symmetric is not the Hessian of a free energy;
    // its eigenvalues are not reported as curvatures.
    if (asym > 1.0e-6) return skip("the force Jacobian is not symmetric, so it is not a free-energy curvature");

    write_scp_hessian(temp, "", J, A, total_applications, worst_residual, asym);
    return true;
}

void Scph::write_scp_hessian(const double temp, const std::string &skip_reason, const Eigen::MatrixXd &J,
                             const Eigen::MatrixXd &A, const int n_applications, const double residual,
                             const double asymmetry)
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
    ofs << '\n';
}

void Scph::report_scp_hessian_fd_check(const Eigen::MatrixXd &J, const Eigen::MatrixXd &jacobian_fd) const
{
    if (J.rows() == 0 || jacobian_fd.rows() != J.rows()) return;
    const auto diff = (J - jacobian_fd).cwiseAbs().maxCoeff() / std::max(J.cwiseAbs().maxCoeff(), 1.0e-300);
    std::cout << "  BUBBLE_FD_CHECK: max |J - J_fd| / max |J| = " << std::scientific << std::setprecision(3) << diff
              << std::defaultfloat << '\n';
}
