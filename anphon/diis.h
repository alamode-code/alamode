//
// Created by Terumasa Tadano on 2025/11/26.
//

// Header file for performing the generalized Direct Inversion in the Iterative Subspace (DIIS)

#pragma once

#include <Eigen/Core>
#include <vector>
#include <deque>

/**
 * @brief Generalized Direct Inversion in the Iterative Subspace (GDIIS) optimizer.
 *
 * DIIS accelerates the convergence of iterative methods by extrapolating
 * from a linear combination of previous trial vectors to minimize the error.
 *
 */
class GDIIS
{
public:
    /**
     * @brief Constructor for GDIIS optimizer.
     *
     * @param[in] max_history Maximum number of vectors to store in history
     * @param[in] mixing_beta Mixing parameter (0 < beta <= 1)
     * @param[in] verbosity    Verbosity level for logging (0: silent, >0: print info)
     */
    GDIIS(int max_history = 10, double mixing_beta = 0.5, int verbosity = 0);

    ~GDIIS() = default;

    /**
     * @brief Updates the DIIS history with a new trial vector and error vector.
     *
     * @param[in] x_trial  Current trial vector
     * @param[in] error    Error/residual vector at current iteration, e.g., error = f(x) - x
     */
    void push(const Eigen::VectorXd &x_trial, const Eigen::VectorXd &error);

    /**
     * @brief Computes the extrapolated vector using DIIS.
     *
     * @param[out] x_new   Extrapolated vector
     * @return             true if successful, false otherwise
     */
    bool extrapolate(Eigen::VectorXd &x_new);

    /**
     * @brief Clears the DIIS history.
     */
    void clear();

    /**
     * @brief Returns the current history size.
     *
     * @return Number of vectors currently stored
     */
    [[nodiscard]] int size() const { return static_cast<int>(history_x.size()); }

    /**
     * @brief Checks if DIIS has enough history for extrapolation.
     *
     * @return true if at least 2 vectors are stored
     */
    [[nodiscard]] bool is_ready() const { return size() >= 2; }

    /**
     * @brief Sets the maximum history size.
     *
     * @param[in] max_hist New maximum history size
     */
    void set_max_history(int max_hist) { max_history_ = max_hist; }

    /**
     * @brief Sets the mixing parameter.
     *
     * @param[in] beta Mixing parameter (0 < beta <= 1)
     */
    void set_mixing_beta(double beta) { mixing_beta_ = beta; }

private:
    int max_history_;                           ///< Maximum number of vectors to keep
    double mixing_beta_;                        ///< Mixing parameter for simple mixing fallback
    int verbosity_;                             ///< Verbosity level for logging
    std::deque<Eigen::VectorXd> history_x;      ///< History of trial vectors
    std::deque<Eigen::VectorXd> history_error;  ///< History of error vectors

    /**
     * @brief Solves the DIIS linear system to find optimal coefficients.
     *
     * @param[out] coeffs  Optimal linear combination coefficients
     * @return             true if successful, false otherwise
     */
    bool solve_diis_equations(Eigen::VectorXd &coeffs);
};

/**
 * @brief GDIIS optimizer for eigenvalue problems of Hermitian matrices.
 *
 * This improved version reconstructs Hermitian matrices from eigenvalue-eigenvector
 * pairs and extrapolates them with DIIS. It handles eigenvalue reordering by
 * tracking permutations across iterations.
 *
 * Algorithm:
 * 1. Track eigenvalue-eigenvector pairs and detect permutations
 * 2. Compute error from permutation-aligned eigenvalues
 * 3. Reconstruct Hermitian matrices: M_i = V_i† diag(λ_i) V_i
 * 4. DIIS extrapolation: M_new = Σ c_i M_i
 * 5. Diagonalize M_new to get consistent eigenvalues/eigenvectors
 *
 * Use case: Self-consistent field calculations, SCPH with dynamical matrices
 */
class GDIIS_Eigen
{
public:
    /**
     * @brief Constructor for eigenvalue GDIIS optimizer.
     *
     * @param[in] max_history Maximum number of iterations to store
     * @param[in] mixing_beta Mixing parameter (0 < beta <= 1)
     * @param[in] verbosity   Verbosity level for logging (0: silent, >0: print info)
     */
    GDIIS_Eigen(int max_history = 10, double mixing_beta = 0.5, int verbosity = 0);

    ~GDIIS_Eigen() = default;

    /**
     * @brief Updates the DIIS history with eigenvalues and eigenvectors.
     *
     * Automatically detects eigenvalue permutations and tracks them.
     *
     * @param[in] eigenvalues  Current eigenvalues (MatrixXd: nk × ns)
     * @param[in] eigenvectors Current eigenvectors (vector of nk MatrixXcd, each ns × ns)
     */
    void push(const Eigen::MatrixXd &eigenvalues,
              const std::vector<Eigen::MatrixXcd> &eigenvectors);

    /**
     * @brief Computes extrapolated eigenvalues and eigenvectors using DIIS.
     *
     * Reconstructs Hermitian matrix from DIIS-extrapolated components,
     * then diagonalizes to get consistent eigenvalues/eigenvectors.
     *
     * @param[out] eigenvalues_new  Extrapolated eigenvalues
     * @param[out] eigenvectors_new Extrapolated eigenvectors
     * @return                      true if successful, false otherwise
     */
    bool extrapolate(Eigen::MatrixXd &eigenvalues_new,
                     std::vector<Eigen::MatrixXcd> &eigenvectors_new);

    /**
     * @brief Clears the DIIS history.
     */
    void clear();

    /**
     * @brief Returns the current history size.
     *
     * @return Number of eigenvalue-eigenvector pairs stored
     */
    [[nodiscard]] int size() const { return static_cast<int>(history_eigenvalues.size()); }

    /**
     * @brief Checks if DIIS has enough history for extrapolation.
     *
     * @return true if at least 2 pairs are stored
     */
    [[nodiscard]] bool is_ready() const { return size() >= 2; }

    /**
     * @brief Sets the maximum history size.
     *
     * @param[in] max_hist New maximum history size
     */
    void set_max_history(int max_hist) { max_history_ = max_hist; }

    /**
     * @brief Sets the mixing parameter.
     *
     * @param[in] beta Mixing parameter (0 < beta <= 1)
     */
    void set_mixing_beta(double beta) { mixing_beta_ = beta; }

private:
    int max_history_;
    double mixing_beta_;
    int verbosity_;
    std::deque<Eigen::MatrixXd> history_eigenvalues;
    std::deque<std::vector<Eigen::MatrixXcd>> history_eigenvectors;
    std::deque<std::vector<Eigen::MatrixXcd>> history_matrices;
    std::deque<Eigen::VectorXd> history_error;

    void find_permutation(const Eigen::MatrixXd &eigenvalues_old,
                          const Eigen::MatrixXd &eigenvalues_new,
                          std::vector<Eigen::VectorXi> &permutations);

    void apply_permutation(Eigen::MatrixXd &eigenvalues,
                           std::vector<Eigen::MatrixXcd> &eigenvectors,
                           const std::vector<Eigen::VectorXi> &permutations);

    Eigen::MatrixXcd reconstruct_hermitian_matrix(const Eigen::VectorXd &eigenvalues,
                                                  const Eigen::MatrixXcd &eigenvectors);

    Eigen::VectorXd flatten_eigenvalues(const Eigen::MatrixXd &eigenvalues) const;

    bool solve_diis_equations(Eigen::VectorXd &coeffs);
};

