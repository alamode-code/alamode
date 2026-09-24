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

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
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

} // namespace PHON_NS::scph_hessian
