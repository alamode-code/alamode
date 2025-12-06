//
// Created by Terumasa Tadano on 2025/11/26.
//

// Implementation of the generalized Direct Inversion in the Iterative Subspace (DIIS)

#include "diis.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

GDIIS::GDIIS(int max_history, double mixing_beta, int verbosity)
    : max_history_(max_history), mixing_beta_(mixing_beta), verbosity_(verbosity)
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

    // Solve DIIS equations to get optimal coefficients
    Eigen::VectorXd coeffs;
    if (!solve_diis_equations(coeffs)) {
        // Fall back to the most recent trial vector
        x_new = history_x.back();

        // If failing repeatedly, clear old history
        if (n >= max_history_) {
            history_x.pop_front();
            history_error.pop_front();
        }
        return false;
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
            B(j, i) = B(i, j);  // Symmetric
        }
    }

    // Add small regularization to diagonal for numerical stability
    const double reg = 1e-10 * B.topLeftCorner(n, n).trace() / n;
    for (int i = 0; i < n; ++i) {
        B(i, i) += reg;
    }

    // Check condition number - if too large, reduce history
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(B.topLeftCorner(n, n));
    if (eigensolver.info() == Eigen::Success) {
        double cond = eigensolver.eigenvalues().maxCoeff() /
                     (eigensolver.eigenvalues().minCoeff() + 1e-20);
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
    rhs(n) = 1.0;  // Constraint: sum c_i = 1

    // Solve the linear system B * [c; lambda] = rhs
    // Use ColPivHouseholderQR for more robust solving
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(B);
    if (qr.rank() < n + 1) {
        // Only report if errors are still significant
        double max_error = 0.0;
        for (const auto& e : history_error) {
            max_error = std::max(max_error, e.norm());
        }
        if (max_error > 1e-6) {
            std::cerr << "Warning: GDIIS matrix rank deficient (rank=" << qr.rank()
                     << ", expected=" << (n+1) << "), max error=" << max_error << "\n";
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
        std::cerr << "Warning: GDIIS coefficients sum to " << sum_coeffs
                  << " instead of 1.0, rejecting.\n";
        return false;
    }

    return true;
}


// ========================================================================
// GDIIS_Eigen Implementation - Eigenvalue/Eigenvector Tracking
// ========================================================================

GDIIS_Eigen::GDIIS_Eigen(int max_history, double mixing_beta, int verbosity)
    : max_history_(max_history), mixing_beta_(mixing_beta), verbosity_(verbosity)
{
    if (max_history_ < 2) {
        std::cerr << "Warning: GDIIS_Eigen max_history must be at least 2. Setting to 2.\n";
        max_history_ = 2;
    }
    if (mixing_beta_ <= 0.0 || mixing_beta_ > 1.0) {
        std::cerr << "Warning: GDIIS_Eigen mixing_beta must be in (0, 1]. Setting to 0.5.\n";
        mixing_beta_ = 0.5;
    }
}

void GDIIS_Eigen::push(const Eigen::MatrixXd &eigenvalues,
                       const std::vector<Eigen::MatrixXcd> &eigenvectors)
{
    if (eigenvalues.rows() == 0 || eigenvectors.empty()) {
        std::cerr << "Error: GDIIS_Eigen::push - eigenvalues or eigenvectors are empty.\n";
        return;
    }
    if (static_cast<int>(eigenvalues.rows()) != static_cast<int>(eigenvectors.size())) {
        std::cerr << "Error: GDIIS_Eigen::push - mismatch between k-point counts.\n";
        return;
    }

    Eigen::MatrixXd aligned_values = eigenvalues;
    auto aligned_vectors = eigenvectors;
    Eigen::VectorXd error_vec;

    if (!history_eigenvalues.empty()) {
        std::vector<Eigen::VectorXi> permutations;
        find_permutation(history_eigenvalues.back(), aligned_values, permutations);
        apply_permutation(aligned_values, aligned_vectors, permutations);
        error_vec = flatten_eigenvalues(aligned_values - history_eigenvalues.back());
    } else {
        error_vec = flatten_eigenvalues(aligned_values);
    }

    std::vector<Eigen::MatrixXcd> matrices(aligned_vectors.size());
    for (size_t k = 0; k < aligned_vectors.size(); ++k) {
        matrices[k] = reconstruct_hermitian_matrix(aligned_values.row(k).transpose(), aligned_vectors[k]);
    }

    history_eigenvalues.push_back(aligned_values);
    history_eigenvectors.push_back(aligned_vectors);
    history_matrices.push_back(std::move(matrices));
    history_error.push_back(error_vec);

    while (static_cast<int>(history_eigenvalues.size()) > max_history_) {
        history_eigenvalues.pop_front();
        history_eigenvectors.pop_front();
        history_matrices.pop_front();
        history_error.pop_front();
    }
}

bool GDIIS_Eigen::extrapolate(Eigen::MatrixXd &eigenvalues_new,
                              std::vector<Eigen::MatrixXcd> &eigenvectors_new)
{
    const int n = size();
    if (n < 1) {
        std::cerr << "Error: GDIIS_Eigen::extrapolate - no history available.\n";
        return false;
    }

    if (n == 1) {
        eigenvalues_new = history_eigenvalues[0];
        eigenvectors_new = history_eigenvectors[0];
        return true;
    }

    const double latest_error_norm = history_error.back().norm();
    if (latest_error_norm < 1e-8 && n > 3) {
        while (size() > 3) {
            history_eigenvalues.pop_front();
            history_eigenvectors.pop_front();
            history_matrices.pop_front();
            history_error.pop_front();
        }
    }

    Eigen::VectorXd coeffs;
    if (!solve_diis_equations(coeffs)) {
        eigenvalues_new = history_eigenvalues.back();
        eigenvectors_new = history_eigenvectors.back();
        if (n >= max_history_) {
            history_eigenvalues.pop_front();
            history_eigenvectors.pop_front();
            history_matrices.pop_front();
            history_error.pop_front();
        }
        return false;
    }

    const int nk = static_cast<int>(history_matrices.front().size());
    const int ns = history_matrices.front().front().rows();

    eigenvalues_new = Eigen::MatrixXd::Zero(history_eigenvalues.front().rows(),
                                            history_eigenvalues.front().cols());
    eigenvectors_new.assign(nk, Eigen::MatrixXcd(ns, ns));

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver;
    for (int k = 0; k < nk; ++k) {
        Eigen::MatrixXcd combined = Eigen::MatrixXcd::Zero(ns, ns);
        for (int i = 0; i < size(); ++i) {
            combined += coeffs(i) * history_matrices[i][k];
        }
        combined = 0.5 * (combined + combined.adjoint());

        solver.compute(combined);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Error: GDIIS_Eigen failed to diagonalize combined matrix at k=" << k << "\n";
            return false;
        }

        eigenvalues_new.row(k) = solver.eigenvalues().transpose();
        eigenvectors_new[k] = solver.eigenvectors();
    }

    return true;
}

void GDIIS_Eigen::clear()
{
    history_eigenvalues.clear();
    history_eigenvectors.clear();
    history_matrices.clear();
    history_error.clear();
}

bool GDIIS_Eigen::solve_diis_equations(Eigen::VectorXd &coeffs)
{
    const int n = size();
    Eigen::MatrixXd B(n + 1, n + 1);
    B.setZero();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            B(i, j) = history_error[i].dot(history_error[j]);
            B(j, i) = B(i, j);
        }
    }

    const double reg = 1e-10 * B.topLeftCorner(n, n).trace() / n;
    for (int i = 0; i < n; ++i) {
        B(i, i) += reg;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(B.topLeftCorner(n, n));
    if (eigensolver.info() == Eigen::Success) {
        const double cond = eigensolver.eigenvalues().maxCoeff() /
                            (eigensolver.eigenvalues().minCoeff() + 1e-20);
        if (cond > 1e12) {
            if (verbosity_ > 0) {
                std::cerr << "Warning: GDIIS_Eigen matrix ill-conditioned (cond=" << cond << ")\n";
            }
            return false;
        }
    }

    for (int i = 0; i < n; ++i) {
        B(i, n) = 1.0;
        B(n, i) = 1.0;
    }
    B(n, n) = 0.0;

    Eigen::VectorXd rhs(n + 1);
    rhs.setZero();
    rhs(n) = 1.0;

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(B);
    if (qr.rank() < n + 1) {
        double max_error = 0.0;
        for (const auto &e : history_error) {
            max_error = std::max(max_error, e.norm());
        }
        if (max_error > 1e-6 && verbosity_ > 0) {
            std::cerr << "Warning: GDIIS_Eigen matrix rank deficient (rank=" << qr.rank() << ")\n";
        }
        return false;
    }

    const Eigen::VectorXd solution = qr.solve(rhs);
    const double residual = (B * solution - rhs).norm();
    if (residual > 1e-6) {
        if (verbosity_ > 0) {
            std::cerr << "Warning: GDIIS_Eigen solve residual large: " << residual << "\n";
        }
        return false;
    }

    coeffs = solution.head(n);
    const double sum_coeffs = coeffs.sum();
    if (std::abs(sum_coeffs - 1.0) > 1e-3) {
        if (verbosity_ > 0) {
            std::cerr << "Warning: GDIIS_Eigen coefficients sum to " << sum_coeffs << "\n";
        }
        return false;
    }

    return true;
}

void GDIIS_Eigen::find_permutation(const Eigen::MatrixXd &eigenvalues_old,
                                   const Eigen::MatrixXd &eigenvalues_new,
                                   std::vector<Eigen::VectorXi> &permutations)
{
    const int nk = static_cast<int>(eigenvalues_old.rows());
    const int ns = static_cast<int>(eigenvalues_old.cols());
    permutations.resize(nk);

    for (int k = 0; k < nk; ++k) {
        permutations[k].resize(ns);
        std::vector<bool> used(ns, false);
        for (int i = 0; i < ns; ++i) {
            double best_diff = std::numeric_limits<double>::max();
            int best_j = i;
            for (int j = 0; j < ns; ++j) {
                if (used[j]) continue;
                const double diff = std::abs(eigenvalues_old(k, i) - eigenvalues_new(k, j));
                if (diff < best_diff) {
                    best_diff = diff;
                    best_j = j;
                }
            }
            permutations[k](i) = best_j;
            used[best_j] = true;
        }
    }
}

void GDIIS_Eigen::apply_permutation(Eigen::MatrixXd &eigenvalues,
                                    std::vector<Eigen::MatrixXcd> &eigenvectors,
                                    const std::vector<Eigen::VectorXi> &permutations)
{
    for (size_t k = 0; k < permutations.size(); ++k) {
        const auto &perm = permutations[k];
        Eigen::VectorXd reordered_vals(perm.size());
        for (int i = 0; i < perm.size(); ++i) {
            reordered_vals(i) = eigenvalues(static_cast<int>(k), perm(i));
        }
        eigenvalues.row(static_cast<int>(k)) = reordered_vals.transpose();

        Eigen::MatrixXcd reordered_vecs = eigenvectors[k];
        for (int i = 0; i < perm.size(); ++i) {
            reordered_vecs.col(i) = eigenvectors[k].col(perm(i));
        }
        eigenvectors[k] = reordered_vecs;
    }
}

Eigen::MatrixXcd GDIIS_Eigen::reconstruct_hermitian_matrix(const Eigen::VectorXd &eigenvalues,
                                                            const Eigen::MatrixXcd &eigenvectors)
{
    Eigen::MatrixXcd diag = eigenvalues.asDiagonal();
    return eigenvectors * diag * eigenvectors.adjoint();
}

Eigen::VectorXd GDIIS_Eigen::flatten_eigenvalues(const Eigen::MatrixXd &eigenvalues) const
{
    Eigen::VectorXd flat(eigenvalues.size());
    int idx = 0;
    for (int k = 0; k < eigenvalues.rows(); ++k) {
        for (int i = 0; i < eigenvalues.cols(); ++i) {
            flat(idx++) = eigenvalues(k, i);
        }
    }
    return flat;
}


// ========================================================================
// GDIIS_Matrix Implementation - Matrix Mixing with Eigenvalue Error
// ========================================================================

GDIIS_Matrix::GDIIS_Matrix(int max_history, double mixing_beta, int verbosity)
    : max_history_(max_history), mixing_beta_(mixing_beta), verbosity_(verbosity)
{
    if (max_history_ < 2) {
        std::cerr << "Warning: GDIIS_Matrix max_history must be at least 2. Setting to 2.\n";
        max_history_ = 2;
    }
    if (mixing_beta_ <= 0.0 || mixing_beta_ > 1.0) {
        std::cerr << "Warning: GDIIS_Matrix mixing_beta must be in (0, 1]. Setting to 0.5.\n";
        mixing_beta_ = 0.5;
    }
}

void GDIIS_Matrix::push(const std::vector<Eigen::MatrixXcd> &matrices,
                        const Eigen::MatrixXd &eigenvalues)
{
    if (matrices.empty()) {
        std::cerr << "Error: GDIIS_Matrix::push - matrices are empty.\n";
        return;
    }
    if (static_cast<int>(matrices.size()) != eigenvalues.rows()) {
        std::cerr << "Error: GDIIS_Matrix::push - mismatch between matrix count and eigenvalue rows.\n";
        return;
    }

    // Compute error from eigenvalues
    Eigen::VectorXd error_vec;
    Eigen::MatrixXd aligned_eigenvalues = eigenvalues;

    if (!history_eigenvalues.empty()) {
        // Find permutation to align eigenvalues with previous iteration
        std::vector<Eigen::VectorXi> permutations;
        find_permutation(history_eigenvalues.back(), aligned_eigenvalues, permutations);

        // Apply permutation to eigenvalues only (not to matrices, as they define the space)
        for (size_t k = 0; k < permutations.size(); ++k) {
            const auto &perm = permutations[k];
            Eigen::VectorXd reordered_vals(perm.size());
            for (int i = 0; i < perm.size(); ++i) {
                reordered_vals(i) = aligned_eigenvalues(static_cast<int>(k), perm(i));
            }
            aligned_eigenvalues.row(static_cast<int>(k)) = reordered_vals.transpose();
        }

        // Compute error as RELATIVE difference in eigenvalues for better scaling
        Eigen::MatrixXd error_matrix(aligned_eigenvalues.rows(), aligned_eigenvalues.cols());
        for (int k = 0; k < aligned_eigenvalues.rows(); ++k) {
            for (int i = 0; i < aligned_eigenvalues.cols(); ++i) {
                const double old_val = history_eigenvalues.back()(k, i);
                const double new_val = aligned_eigenvalues(k, i);
                const double abs_old = std::abs(old_val);
                const double denominator = std::max(abs_old, 1e-6);  // Avoid division by zero
                error_matrix(k, i) = (new_val - old_val) / denominator;
            }
        }
        error_vec = flatten_eigenvalues(error_matrix);
    } else {
        // First iteration: use small normalized eigenvalues as error
        Eigen::MatrixXd normalized_evals(eigenvalues.rows(), eigenvalues.cols());
        for (int k = 0; k < eigenvalues.rows(); ++k) {
            for (int i = 0; i < eigenvalues.cols(); ++i) {
                const double val = eigenvalues(k, i);
                normalized_evals(k, i) = val / (std::abs(val) + 1.0);
            }
        }
        error_vec = flatten_eigenvalues(normalized_evals);
    }

    history_matrices.push_back(matrices);
    history_eigenvalues.push_back(aligned_eigenvalues);
    history_error.push_back(error_vec);

    // Remove oldest entries if history exceeds max_history
    while (static_cast<int>(history_matrices.size()) > max_history_) {
        history_matrices.pop_front();
        history_eigenvalues.pop_front();
        history_error.pop_front();
    }
}

bool GDIIS_Matrix::extrapolate(std::vector<Eigen::MatrixXcd> &matrices_new)
{
    int n = size();
    if (n < 1) {
        std::cerr << "Error: GDIIS_Matrix::extrapolate - no history available.\n";
        return false;
    }

    // If only one matrix set, return it as-is
    if (n == 1) {
        matrices_new = history_matrices[0];
        return true;
    }

    // Check if error norms are decreasing monotonically (good convergence)
    // If so, reduce history to most recent points for better stability
    const double latest_error_norm = history_error.back().norm();
    if (n >= 2) {
        const double prev_error_norm = history_error[n-2].norm();
        // If error is very small or increasing, be conservative with history
        if (latest_error_norm < 1e-6 || latest_error_norm > 2.0 * prev_error_norm) {
            while (size() > 2) {
                history_matrices.pop_front();
                history_eigenvalues.pop_front();
                history_error.pop_front();
            }
            n = size();  // Update n after modification
        }
    }

    // If errors are oscillating wildly, clear history and restart
    if (n >= 3) {
        const double ratio1 = history_error[n-1].norm() / (history_error[n-2].norm() + 1e-12);
        const double ratio2 = history_error[n-2].norm() / (history_error[n-3].norm() + 1e-12);
        if (ratio1 > 100.0 || ratio2 > 100.0) {
            if (verbosity_ > 1) {
                std::cout << "  GDIIS_Matrix: error oscillating, clearing history\n";
            }
            // Keep only the most recent entry
            auto last_matrix = history_matrices.back();
            auto last_eval = history_eigenvalues.back();
            auto last_error = history_error.back();

            history_matrices.clear();
            history_eigenvalues.clear();
            history_error.clear();

            history_matrices.push_back(last_matrix);
            history_eigenvalues.push_back(last_eval);
            history_error.push_back(last_error);

            // Return the most recent matrices
            matrices_new = last_matrix;
            return false;
        }
    }

    // Verify we still have valid history
    if (history_matrices.empty() || history_matrices.front().empty()) {
        std::cerr << "Error: GDIIS_Matrix::extrapolate - history became empty.\n";
        return false;
    }

    // Solve DIIS equations to get optimal coefficients
    Eigen::VectorXd coeffs;
    if (!solve_diis_equations(coeffs)) {
        // Fall back to the most recent matrices
        if (!history_matrices.empty()) {
            matrices_new = history_matrices.back();
        }
        return false;
    }

    // Compute extrapolated matrices: M_new[k] = sum_i c_i * M_i[k]
    const int nk = static_cast<int>(history_matrices.front().size());
    const int ns = history_matrices.front().front().rows();
    const int current_size = size();  // Get current size for the loop

    matrices_new.clear();
    matrices_new.reserve(nk);

    for (int k = 0; k < nk; ++k) {
        Eigen::MatrixXcd mat_combined = Eigen::MatrixXcd::Zero(ns, ns);
        for (int i = 0; i < current_size; ++i) {
            mat_combined += coeffs(i) * history_matrices[i][k];
        }
        matrices_new.push_back(mat_combined);
    }

    return true;
}

void GDIIS_Matrix::clear()
{
    history_matrices.clear();
    history_eigenvalues.clear();
    history_error.clear();
}

bool GDIIS_Matrix::solve_diis_equations(Eigen::VectorXd &coeffs)
{
    const int n = size();

    // Build the error overlap matrix B_ij = <e_i | e_j>
    Eigen::MatrixXd B(n + 1, n + 1);
    B.setZero();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            B(i, j) = history_error[i].dot(history_error[j]);
            B(j, i) = B(i, j);  // Symmetric
        }
    }

    // Check if error matrix is too small (near convergence)
    double max_error_norm = 0.0;
    for (int i = 0; i < n; ++i) {
        max_error_norm = std::max(max_error_norm, history_error[i].norm());
    }

    // If errors are very small, don't use DIIS (would be numerically unstable)
    if (max_error_norm < 1e-8) {
        if (verbosity_ > 1) {
            std::cout << "  GDIIS_Matrix: errors too small (" << max_error_norm
                     << "), skipping DIIS\n";
        }
        return false;
    }

    // Add regularization proportional to the error magnitude for better stability
    const double trace = B.topLeftCorner(n, n).trace();
    const double reg = std::max(1e-12 * trace / n, 1e-14);
    for (int i = 0; i < n; ++i) {
        B(i, i) += reg;
    }

    // Check condition number - be more strict than before
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(B.topLeftCorner(n, n));
    if (eigensolver.info() == Eigen::Success) {
        const double max_eval = eigensolver.eigenvalues().maxCoeff();
        const double min_eval = eigensolver.eigenvalues().minCoeff();
        const double cond = max_eval / (std::abs(min_eval) + 1e-20);

        // Be more conservative with condition number
        if (cond > 1e10 || min_eval < 1e-14) {
            if (verbosity_ > 0) {
                std::cerr << "Warning: GDIIS_Matrix ill-conditioned (cond=" << cond
                         << ", min_eval=" << min_eval << ")\n";
            }
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
    rhs(n) = 1.0;  // Constraint: sum c_i = 1

    // Solve the linear system - use more robust solver
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(B);
    if (qr.rank() < n + 1) {
        if (verbosity_ > 0) {
            std::cerr << "Warning: GDIIS_Matrix rank deficient (rank=" << qr.rank()
                     << ", expected=" << (n+1) << ")\n";
        }
        return false;
    }

    const Eigen::VectorXd solution = qr.solve(rhs);

    // Verify solution quality
    const double residual = (B * solution - rhs).norm();
    if (residual > 1e-5) {
        if (verbosity_ > 0) {
            std::cerr << "Warning: GDIIS_Matrix solve residual large: " << residual << "\n";
        }
        return false;
    }

    // Extract coefficients
    coeffs = solution.head(n);

    // Verify that coefficients sum to approximately 1
    const double sum_coeffs = coeffs.sum();
    if (std::abs(sum_coeffs - 1.0) > 1e-3) {
        if (verbosity_ > 0) {
            std::cerr << "Warning: GDIIS_Matrix coefficients sum to " << sum_coeffs << "\n";
        }
        return false;
    }

    // Check for unreasonable coefficients (too large or too negative)
    double max_coeff = coeffs.cwiseAbs().maxCoeff();
    if (max_coeff > 10.0) {
        if (verbosity_ > 0) {
            std::cerr << "Warning: GDIIS_Matrix coefficient too large: " << max_coeff << "\n";
        }
        return false;
    }

    if (verbosity_ > 1) {
        std::cout << "GDIIS_Matrix coefficients:";
        for (int i = 0; i < n; ++i) {
            std::cout << " " << coeffs(i);
        }
        std::cout << " (sum=" << sum_coeffs << ")\n";
    }

    return true;
}

void GDIIS_Matrix::find_permutation(const Eigen::MatrixXd &eigenvalues_old,
                                    const Eigen::MatrixXd &eigenvalues_new,
                                    std::vector<Eigen::VectorXi> &permutations)
{
    const int nk = static_cast<int>(eigenvalues_old.rows());
    const int ns = static_cast<int>(eigenvalues_old.cols());
    permutations.resize(nk);

    for (int k = 0; k < nk; ++k) {
        permutations[k].resize(ns);
        std::vector<bool> used(ns, false);
        for (int i = 0; i < ns; ++i) {
            double best_diff = std::numeric_limits<double>::max();
            int best_j = i;
            for (int j = 0; j < ns; ++j) {
                if (used[j]) continue;
                const double diff = std::abs(eigenvalues_old(k, i) - eigenvalues_new(k, j));
                if (diff < best_diff) {
                    best_diff = diff;
                    best_j = j;
                }
            }
            permutations[k](i) = best_j;
            used[best_j] = true;
        }
    }
}

Eigen::VectorXd GDIIS_Matrix::flatten_eigenvalues(const Eigen::MatrixXd &eigenvalues) const
{
    Eigen::VectorXd flat(eigenvalues.size());
    int idx = 0;
    for (int k = 0; k < eigenvalues.rows(); ++k) {
        for (int i = 0; i < eigenvalues.cols(); ++i) {
            flat(idx++) = eigenvalues(k, i);
        }
    }
    return flat;
}


// ========================================================================
// GDIIS_PerKpoint Implementation - Independent DIIS for Each K-point
// ========================================================================

GDIIS_PerKpoint::GDIIS_PerKpoint(int nk, int max_history, double mixing_beta, int verbosity)
    : nk_(nk), max_history_(max_history), mixing_beta_(mixing_beta), verbosity_(verbosity)
{
    if (max_history_ < 2) {
        std::cerr << "Warning: GDIIS_PerKpoint max_history must be at least 2. Setting to 2.\n";
        max_history_ = 2;
    }
    if (mixing_beta_ <= 0.0 || mixing_beta_ > 1.0) {
        std::cerr << "Warning: GDIIS_PerKpoint mixing_beta must be in (0, 1]. Setting to 0.5.\n";
        mixing_beta_ = 0.5;
    }

    // Create one GDIIS_Matrix instance for each k-point
    diis_per_k_.reserve(nk_);
    for (int k = 0; k < nk_; ++k) {
        diis_per_k_.emplace_back(max_history_, mixing_beta_, 0);  // verbosity 0 for individual k-points
    }

    success_count_.resize(nk_, 0);
    failure_count_.resize(nk_, 0);
}

void GDIIS_PerKpoint::push(const std::vector<Eigen::MatrixXcd> &matrices,
                            const Eigen::MatrixXd &eigenvalues)
{
    if (static_cast<int>(matrices.size()) != nk_ || eigenvalues.rows() != nk_) {
        std::cerr << "Error: GDIIS_PerKpoint::push - size mismatch.\n";
        return;
    }

    // Push to each k-point's individual DIIS mixer
    for (int k = 0; k < nk_; ++k) {
        std::vector<Eigen::MatrixXcd> single_matrix = {matrices[k]};
        Eigen::MatrixXd single_eigenvalues = eigenvalues.row(k);
        single_eigenvalues.resize(1, eigenvalues.cols());

        diis_per_k_[k].push(single_matrix, single_eigenvalues);
    }
}

bool GDIIS_PerKpoint::extrapolate(std::vector<Eigen::MatrixXcd> &matrices_new)
{
    if (matrices_new.size() != static_cast<size_t>(nk_)) {
        matrices_new.resize(nk_);
    }

    int success_count = 0;
    int total_ready = 0;

    // Extrapolate each k-point independently
    for (int k = 0; k < nk_; ++k) {
        if (diis_per_k_[k].is_ready()) {
            total_ready++;
            std::vector<Eigen::MatrixXcd> temp_result;
            bool success = diis_per_k_[k].extrapolate(temp_result);

            if (success && !temp_result.empty()) {
                matrices_new[k] = temp_result[0];
                success_count_[k]++;
                success_count++;
            } else {
                // Keep the most recent matrix if DIIS fails
                failure_count_[k]++;
                // matrices_new[k] should already have a fallback value from caller
            }
        }
    }

    if (verbosity_ > 1 && total_ready > 0) {
        std::cout << "  GDIIS_PerKpoint: " << success_count << "/" << total_ready
                  << " k-points used DIIS successfully\n";
    }

    return success_count > 0;
}

void GDIIS_PerKpoint::clear()
{
    for (int k = 0; k < nk_; ++k) {
        diis_per_k_[k].clear();
    }
    std::fill(success_count_.begin(), success_count_.end(), 0);
    std::fill(failure_count_.begin(), failure_count_.end(), 0);
}

int GDIIS_PerKpoint::size() const
{
    int min_size = max_history_;
    for (int k = 0; k < nk_; ++k) {
        min_size = std::min(min_size, diis_per_k_[k].size());
    }
    return min_size;
}

bool GDIIS_PerKpoint::is_ready() const
{
    for (int k = 0; k < nk_; ++k) {
        if (diis_per_k_[k].is_ready()) {
            return true;
        }
    }
    return false;
}

void GDIIS_PerKpoint::set_max_history(int max_hist)
{
    max_history_ = max_hist;
    for (int k = 0; k < nk_; ++k) {
        diis_per_k_[k].set_max_history(max_hist);
    }
}

void GDIIS_PerKpoint::set_mixing_beta(double beta)
{
    mixing_beta_ = beta;
    for (int k = 0; k < nk_; ++k) {
        diis_per_k_[k].set_mixing_beta(beta);
    }
}
