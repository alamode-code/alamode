/*
 scph_hessian_kernels.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.

 Kernels of the free-energy Hessian of the SCPH theory (BUBBLE = 4; see
 anphon/FREE_ENERGY_HESSIAN_PLAN.md). The displacement correlation of a mode
 of squared frequency lambda = w^2 is proportional to

   f(lambda) = (2 n_B(w) + 1) / w        (classical: 2 k_B T / w^2),

 the factor Thermodynamics::disp_corr_factor returns. The occupation matrix
 G = C f(Lambda) C^dagger of an SCP matrix with eigenpairs (lambda_s, C) then
 responds to a Hermitian change dPhi of that matrix as (Daleckii-Krein)

   dG = C [ L o (C^dagger dPhi C) ] C^dagger,
   L[s][t] = (f(lambda_s) - f(lambda_t)) / (lambda_s - lambda_t)   (-> f'(lambda_s) on degeneracies),

 where o is the element-wise product. f decreases with lambda, so every
 L[s][t] <= 0: -L is the (positive) static pair propagator of the static
 bubble. Header-only; Eigen only, no MPI, so that the unit test can include it.
*/

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace PHON_NS::scph_hessian
{

// f(lambda) and df/dlambda for lambda = w^2 > 0, in the frequency units of kT.
// kT = 0 is the quantum ground state (n_B = 0).
struct OccupationFactor
{
    double kT = 0.0;
    bool classical = false;

    double occupation(const double w) const
    {
        if (kT <= 0.0) return 0.0;
        return 1.0 / std::expm1(w / kT);
    }

    double f(const double lambda) const
    {
        if (classical) return 2.0 * kT / lambda;
        const auto w = std::sqrt(lambda);
        return (2.0 * occupation(w) + 1.0) / w;
    }

    double dfdlambda(const double lambda) const
    {
        if (classical) return -2.0 * kT / (lambda * lambda);
        const auto w = std::sqrt(lambda);
        const auto n = occupation(w);
        const auto dndw = kT > 0.0 ? -n * (n + 1.0) / kT : 0.0;
        const auto dfdw = (2.0 * dndw * w - (2.0 * n + 1.0)) / (w * w);
        return dfdw / (2.0 * w);
    }
};

// (f(a) - f(b)) / (a - b). Near a degeneracy the quotient loses digits, so the
// derivative at the midpoint is used there; it differs from the quotient by
// (a - b)^2 f'''/24, far below the cancellation error it avoids.
inline double divided_difference(const OccupationFactor &occ, const double lambda_a, const double lambda_b,
                                 const double rel_tol = 1.0e-6)
{
    const auto scale = std::max(std::abs(lambda_a), std::abs(lambda_b));
    if (std::abs(lambda_a - lambda_b) <= rel_tol * scale) return occ.dfdlambda(0.5 * (lambda_a + lambda_b));
    return (occ.f(lambda_a) - occ.f(lambda_b)) / (lambda_a - lambda_b);
}

// L[s][t] for the eigenvalues lambda of one SCP matrix. Rows and columns of
// frozen modes (the excluded Gamma translations: a fixed translational
// subspace) are zero, so their occupation neither changes nor rotates.
inline Eigen::MatrixXd divided_difference_matrix(const OccupationFactor &occ, const Eigen::VectorXd &lambda,
                                                 const std::vector<bool> *frozen = nullptr)
{
    const auto n = lambda.size();
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n, n);
    for (Eigen::Index s = 0; s < n; ++s) {
        if (frozen && (*frozen)[s]) continue;
        for (Eigen::Index t = s; t < n; ++t) {
            if (frozen && (*frozen)[t]) continue;
            L(s, t) = L(t, s) = divided_difference(occ, lambda(s), lambda(t));
        }
    }
    return L;
}

// dG = C [ L o (C^dagger dPhi C) ] C^dagger for the eigenvectors C (columns).
inline Eigen::MatrixXcd apply_dk(const Eigen::MatrixXcd &C, const Eigen::MatrixXd &L, const Eigen::MatrixXcd &dPhi)
{
    const Eigen::MatrixXcd in_eigenbasis = C.adjoint() * dPhi * C;
    return C * in_eigenbasis.cwiseProduct(L.cast<std::complex<double>>()) * C.adjoint();
}

// V max(|lambda|, floor) V^T of the symmetric part of J: a positive-definite
// optimizer Hessian that keeps the curvature magnitudes but turns the ascent
// a Newton step takes along a negative curvature into descent (saddle-free
// Newton). floor > 0 bounds the step along near-flat directions.
inline Eigen::MatrixXd saddle_free(const Eigen::MatrixXd &J, const double floor)
{
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (J + J.transpose()));
    const Eigen::VectorXd lambda = es.eigenvalues().cwiseAbs().cwiseMax(floor);
    return es.eigenvectors() * lambda.asDiagonal() * es.eigenvectors().transpose();
}

// Restarted GMRES for (I - K) y = b with several right-hand sides (columns of
// B), advanced together so that K is applied to one block per step:
// apply_k(const MatrixXcd &in, MatrixXcd &out) sets out = K in. Each column is
// an independent Krylov problem with its own Hessenberg matrix, rotations and
// dimension; it stops extending at convergence or at a happy breakdown, so an
// early column never meets a singular back substitution. Returns true when every
// column reached rel_residual <= tol; false on a non-finite residual or when
// max_restarts cycles did not suffice.
template <class ApplyK>
bool gmres_identity_minus(const Eigen::MatrixXcd &B, Eigen::MatrixXcd &Y, ApplyK &&apply_k, const int restart,
                          const int max_restarts, const double tol, std::vector<double> &rel_residual,
                          int &n_applications)
{
    using namespace Eigen;
    using cplx = std::complex<double>;
    const auto n = B.rows();
    const auto m = B.cols();
    Y = MatrixXcd::Zero(n, m);
    rel_residual.assign(static_cast<size_t>(m), 0.0);
    n_applications = 0;

    VectorXd bnorm(m);
    for (Index c = 0; c < m; ++c) bnorm(c) = B.col(c).norm();

    MatrixXcd KY(n, m);
    for (int cycle = 0;; ++cycle) {
        apply_k(Y, KY);
        ++n_applications;
        const MatrixXcd R = B - (Y - KY);
        VectorXd beta(m);
        std::vector<bool> active(static_cast<size_t>(m), false);
        bool any_active = false;
        for (Index c = 0; c < m; ++c) {
            beta(c) = R.col(c).norm();
            if (!std::isfinite(beta(c))) return false;
            rel_residual[c] = bnorm(c) > 0.0 ? beta(c) / bnorm(c) : 0.0;
            active[c] = rel_residual[c] > tol;
            any_active = any_active || active[c];
        }
        if (!any_active) return true;
        if (cycle == max_restarts) return false;

        std::vector<MatrixXcd> V;
        V.reserve(restart + 1);
        V.emplace_back(MatrixXcd::Zero(n, m));
        for (Index c = 0; c < m; ++c) {
            if (active[c]) V[0].col(c) = R.col(c) / beta(c);
        }
        std::vector<MatrixXcd> H(m, MatrixXcd::Zero(restart + 1, restart));
        std::vector<VectorXcd> g(m, VectorXcd::Zero(restart + 1));
        std::vector<std::vector<cplx>> cs(m), sn(m);
        std::vector<int> dim(static_cast<size_t>(m), 0);
        for (Index c = 0; c < m; ++c) g[c](0) = beta(c);

        for (int j = 0; j < restart && any_active; ++j) {
            MatrixXcd W(n, m);
            apply_k(V[j], W);
            ++n_applications;
            W = V[j] - W; // (I - K) v
            any_active = false;
            for (Index c = 0; c < m; ++c) {
                if (!active[c]) {
                    W.col(c).setZero();
                    continue;
                }
                for (int i = 0; i <= j; ++i) {
                    const cplx h = V[i].col(c).dot(W.col(c)); // conj(V) . W
                    H[c](i, j) = h;
                    W.col(c) -= h * V[i].col(c);
                }
                const double hn = W.col(c).norm();
                for (int i = 0; i < j; ++i) {
                    const cplx a = H[c](i, j), b = H[c](i + 1, j);
                    H[c](i, j) = std::conj(cs[c][i]) * a + std::conj(sn[c][i]) * b;
                    H[c](i + 1, j) = -sn[c][i] * a + cs[c][i] * b;
                }
                const cplx a = H[c](j, j);
                const double r = std::sqrt(std::norm(a) + hn * hn);
                const cplx cj = r > 0.0 ? a / r : cplx(1.0, 0.0);
                const cplx sj = r > 0.0 ? cplx(hn / r, 0.0) : cplx(0.0, 0.0);
                cs[c].push_back(cj);
                sn[c].push_back(sj);
                H[c](j, j) = r;
                const cplx gj = g[c](j);
                g[c](j) = std::conj(cj) * gj;
                g[c](j + 1) = -sj * gj;
                dim[c] = j + 1;
                // converged, or the Krylov space is exhausted (happy breakdown)
                const bool done = std::abs(g[c](j + 1)) <= tol * bnorm(c) || hn <= 1.0e-14 * r;
                if (done || r == 0.0) {
                    active[c] = false;
                    W.col(c).setZero();
                } else {
                    W.col(c) /= hn;
                    any_active = true;
                }
            }
            V.push_back(std::move(W));
        }
        for (Index c = 0; c < m; ++c) {
            const int k = dim[c];
            if (k == 0) continue;
            const VectorXcd z = H[c].topLeftCorner(k, k).template triangularView<Upper>().solve(g[c].head(k));
            if (!z.allFinite()) return false;
            for (int i = 0; i < k; ++i) Y.col(c) += z(i) * V[i].col(c);
        }
    }
}

} // namespace PHON_NS::scph_hessian
