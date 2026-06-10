//
// Created by Terumasa Tadano on 2025/11/26.
//

// Implementation of the generalized Direct Inversion in the Iterative Subspace (DIIS)

#include "diis.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

GDIIS::GDIIS(int max_history, double mixing_beta, int verbosity) :
    max_history_(max_history), mixing_beta_(mixing_beta), verbosity_(verbosity)
{
    if (max_history_ < 2) {
        std::cerr << "Warning: GDIIS max_history must be at least 2. Setting to 2.\n";
        max_history_ = 2;
    }
    if (mixing_beta_ <= 0.0 || mixing_beta_ > 1.0) {
        std::cerr << "Warning: GDIIS mixing_beta must be in (0, 1]. Setting to 0.5.\n";
        mixing_beta_ = 0.5;
    }
}

void GDIIS::push(const Eigen::VectorXd &x_trial, const Eigen::VectorXd &error)
{
    if (x_trial.size() != error.size()) {
        std::cerr << "Error: GDIIS::push - x_trial and error must have the same size.\n";
        return;
    }

    history_x.push_back(x_trial);
    history_error.push_back(error);

    // Remove oldest entries if history exceeds max_history
    while (static_cast<int>(history_x.size()) > max_history_) {
        history_x.pop_front();
        history_error.pop_front();
    }
}

bool GDIIS::extrapolate(Eigen::VectorXd &x_new)
{
    const int n = size();

    if (n < 1) {
        std::cerr << "Error: GDIIS::extrapolate - no history available.\n";
        return false;
    }

    // If only one vector, return it as-is (it's already the trial vector with error embedded)
    if (n == 1) {
        x_new = history_x[0];
        return true;
    }

    // Check if error norms are getting very small - if so, reduce history to avoid rank issues
    double latest_error_norm = history_error.back().norm();
    if (latest_error_norm < 1e-6 && n > 3) {
        // Keep only the most recent 2 vectors
        while (size() > 2) {
            history_x.pop_front();
            history_error.pop_front();
        }
    }

    // Solve DIIS equations to get optimal coefficients.
    // When the solve fails (rank deficiency, ill-conditioning), the oldest
    // pairs are the most likely culprits: drop them one by one and retry
    // instead of keeping a bad subspace around for several more iterations.
    Eigen::VectorXd coeffs;
    while (!solve_diis_equations(coeffs)) {
        if (size() <= 2) {
            // Keep only the most recent pair and let the caller fall back.
            history_x.pop_front();
            history_error.pop_front();
            x_new = history_x.back();
            return false;
        }
        history_x.pop_front();
        history_error.pop_front();
    }

    // Compute extrapolated vector: x_new = sum_i c_i * (x_i + beta * e_i)
    // beta = 0 reduces to standard DIIS, while beta = 1 reduces to the so-called Anderson mixing
    x_new = Eigen::VectorXd::Zero(history_x[0].size());
    for (int i = 0; i < size(); ++i) {
        x_new += coeffs(i) * (history_x[i] + mixing_beta_ * history_error[i]);
    }

    return true;
}

void GDIIS::clear()
{
    history_x.clear();
    history_error.clear();
}

bool GDIIS::solve_diis_equations(Eigen::VectorXd &coeffs)
{
    const int n = size();

    // Build the error overlap matrix B_ij = <e_i | e_j>
    Eigen::MatrixXd B(n + 1, n + 1);
    B.setZero();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            B(i, j) = history_error[i].dot(history_error[j]);
            B(j, i) = B(i, j); // Symmetric
        }
    }

    if (!B.topLeftCorner(n, n).allFinite()) {
        std::cerr << "Warning: GDIIS error-overlap matrix contains non-finite entries.\n";
        return false;
    }

    // Normalize the error-overlap block so that it has the same scale as the
    // constraint row/column appended below. This leaves the coefficients
    // unchanged (only the Lagrange multiplier rescales) but keeps the rank and
    // residual checks of the augmented system meaningful when the residual
    // vectors have very large or very small norms.
    const double bscale = B.topLeftCorner(n, n).diagonal().maxCoeff();
    if (bscale > 0.0) {
        B.topLeftCorner(n, n) /= bscale;
    }

    // Add small regularization to diagonal for numerical stability
    const double reg = 1e-10 * B.topLeftCorner(n, n).trace() / n;
    for (int i = 0; i < n; ++i) {
        B(i, i) += reg;
    }

    // Check condition number - if too large, reduce history
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(B.topLeftCorner(n, n));
    if (eigensolver.info() == Eigen::Success) {
        double cond = eigensolver.eigenvalues().maxCoeff() / (eigensolver.eigenvalues().minCoeff() + 1e-20);
        if (cond > 1e12) {
            std::cerr << "Warning: GDIIS matrix ill-conditioned (cond=" << cond
                      << "), using only most recent vector.\n";
            return false;
        }
    }

    // Add constraint row and column: sum_i c_i = 1
    for (int i = 0; i < n; ++i) {
        B(i, n) = 1.0;
        B(n, i) = 1.0;
    }
    B(n, n) = 0.0;

    // Right-hand side vector
    Eigen::VectorXd rhs(n + 1);
    rhs.setZero();
    rhs(n) = 1.0; // Constraint: sum c_i = 1

    // Solve the linear system B * [c; lambda] = rhs
    // Use ColPivHouseholderQR for more robust solving
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(B);
    if (qr.rank() < n + 1) {
        // Only report if errors are still significant
        double max_error = 0.0;
        for (const auto &e: history_error) {
            max_error = std::max(max_error, e.norm());
        }
        if (max_error > 1e-6) {
            std::cerr << "Warning: GDIIS matrix rank deficient (rank=" << qr.rank() << ", expected=" << (n + 1)
                      << "), max error=" << max_error << "\n";
        }
        return false;
    }

    Eigen::VectorXd solution = qr.solve(rhs);

    // Verify solution quality
    double residual = (B * solution - rhs).norm();
    if (residual > 1e-6) {
        std::cerr << "Warning: GDIIS solve residual large: " << residual << "\n";
        return false;
    }

    // Extract coefficients (ignore the Lagrange multiplier at index n)
    coeffs = solution.head(n);

    // Verify that coefficients sum to approximately 1
    double sum_coeffs = coeffs.sum();
    if (std::abs(sum_coeffs - 1.0) > 1e-3) {
        std::cerr << "Warning: GDIIS coefficients sum to " << sum_coeffs << " instead of 1.0, rejecting.\n";
        return false;
    }

    return true;
}
