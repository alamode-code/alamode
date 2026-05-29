//
// Created by Terumasa Tadano on 2025/11/26.
//

// Header file for performing the generalized Direct Inversion in the Iterative Subspace (DIIS)

#pragma once

#include <Eigen/Core>
#include <deque>
#include <vector>

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
    [[nodiscard]] int size() const
    {
        return static_cast<int>(history_x.size());
    }

    /**
     * @brief Checks if DIIS has enough history for extrapolation.
     *
     * @return true if at least 2 vectors are stored
     */
    [[nodiscard]] bool is_ready() const
    {
        return size() >= 2;
    }

    /**
     * @brief Sets the maximum history size.
     *
     * @param[in] max_hist New maximum history size
     */
    void set_max_history(int max_hist)
    {
        max_history_ = max_hist;
    }

    /**
     * @brief Sets the mixing parameter.
     *
     * @param[in] beta Mixing parameter (0 < beta <= 1)
     */
    void set_mixing_beta(double beta)
    {
        mixing_beta_ = beta;
    }

private:
    int max_history_;                          ///< Maximum number of vectors to keep
    double mixing_beta_;                       ///< Mixing parameter for simple mixing fallback
    int verbosity_;                            ///< Verbosity level for logging
    std::deque<Eigen::VectorXd> history_x;     ///< History of trial vectors
    std::deque<Eigen::VectorXd> history_error; ///< History of error vectors

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
    void push(const Eigen::MatrixXd &eigenvalues, const std::vector<Eigen::MatrixXcd> &eigenvectors);

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
    bool extrapolate(Eigen::MatrixXd &eigenvalues_new, std::vector<Eigen::MatrixXcd> &eigenvectors_new);

    /**
     * @brief Clears the DIIS history.
     */
    void clear();

    /**
     * @brief Returns the current history size.
     *
     * @return Number of eigenvalue-eigenvector pairs stored
     */
    [[nodiscard]] int size() const
    {
        return static_cast<int>(history_eigenvalues.size());
    }

    /**
     * @brief Checks if DIIS has enough history for extrapolation.
     *
     * @return true if at least 2 pairs are stored
     */
    [[nodiscard]] bool is_ready() const
    {
        return size() >= 2;
    }

    /**
     * @brief Sets the maximum history size.
     *
     * @param[in] max_hist New maximum history size
     */
    void set_max_history(int max_hist)
    {
        max_history_ = max_hist;
    }

    /**
     * @brief Sets the mixing parameter.
     *
     * @param[in] beta Mixing parameter (0 < beta <= 1)
     */
    void set_mixing_beta(double beta)
    {
        mixing_beta_ = beta;
    }

private:
    int max_history_;
    double mixing_beta_;
    int verbosity_;
    std::deque<Eigen::MatrixXd> history_eigenvalues;
    std::deque<std::vector<Eigen::MatrixXcd>> history_eigenvectors;
    std::deque<std::vector<Eigen::MatrixXcd>> history_matrices;
    std::deque<Eigen::VectorXd> history_error;

    void find_permutation(const Eigen::MatrixXd &eigenvalues_old, const Eigen::MatrixXd &eigenvalues_new,
                          std::vector<Eigen::VectorXi> &permutations);

    void apply_permutation(Eigen::MatrixXd &eigenvalues, std::vector<Eigen::MatrixXcd> &eigenvectors,
                           const std::vector<Eigen::VectorXi> &permutations);

    Eigen::MatrixXcd reconstruct_hermitian_matrix(const Eigen::VectorXd &eigenvalues,
                                                  const Eigen::MatrixXcd &eigenvectors);

    Eigen::VectorXd flatten_eigenvalues(const Eigen::MatrixXd &eigenvalues) const;

    bool solve_diis_equations(Eigen::VectorXd &coeffs);
};

/**
 * @brief GDIIS optimizer for complex matrix extrapolation with eigenvalue-based error.
 *
 * This class performs DIIS extrapolation on matrices while using eigenvalues
 * to compute the error vector for convergence acceleration. This is useful
 * for cases where convergence is tested on eigenvalues but mixing is performed
 * on matrices (e.g., D-matrices in SCPH).
 *
 * Algorithm:
 * 1. Store history of matrices (nk x ns x ns complex matrices)
 * 2. Compute eigenvalues from matrices at each iteration
 * 3. Use eigenvalue differences as error vectors
 * 4. DIIS extrapolation: M_new = Σ c_i M_i
 * 5. Return extrapolated matrices
 *
 * Use case: SCPH D-matrix mixing with eigenvalue convergence criterion
 */
class GDIIS_Matrix
{
public:
    /**
     * @brief Constructor for matrix GDIIS optimizer.
     *
     * @param[in] max_history Maximum number of iterations to store
     * @param[in] mixing_beta Mixing parameter (0 < beta <= 1)
     * @param[in] verbosity   Verbosity level for logging (0: silent, >0: print info)
     */
    GDIIS_Matrix(int max_history = 10, double mixing_beta = 0.5, int verbosity = 0);

    ~GDIIS_Matrix() = default;

    /**
     * @brief Updates the DIIS history with new matrices and eigenvalues.
     *
     * @param[in] matrices    Current matrices (vector of nk MatrixXcd, each ns × ns)
     * @param[in] eigenvalues Current eigenvalues for error computation (MatrixXd: nk × ns)
     */
    void push(const std::vector<Eigen::MatrixXcd> &matrices, const Eigen::MatrixXd &eigenvalues);

    /**
     * @brief Computes extrapolated matrices using DIIS.
     *
     * @param[out] matrices_new  Extrapolated matrices
     * @return                   true if successful, false otherwise
     */
    bool extrapolate(std::vector<Eigen::MatrixXcd> &matrices_new);

    /**
     * @brief Clears the DIIS history.
     */
    void clear();

    /**
     * @brief Returns the current history size.
     *
     * @return Number of matrix sets stored
     */
    [[nodiscard]] int size() const
    {
        return static_cast<int>(history_matrices.size());
    }

    /**
     * @brief Checks if DIIS has enough history for extrapolation.
     *
     * @return true if at least 2 sets are stored
     */
    [[nodiscard]] bool is_ready() const
    {
        return size() >= 2;
    }

    /**
     * @brief Sets the maximum history size.
     *
     * @param[in] max_hist New maximum history size
     */
    void set_max_history(int max_hist)
    {
        max_history_ = max_hist;
    }

    /**
     * @brief Sets the mixing parameter.
     *
     * @param[in] beta Mixing parameter (0 < beta <= 1)
     */
    void set_mixing_beta(double beta)
    {
        mixing_beta_ = beta;
    }

private:
    int max_history_;
    double mixing_beta_;
    int verbosity_;
    std::deque<std::vector<Eigen::MatrixXcd>> history_matrices;
    std::deque<Eigen::MatrixXd> history_eigenvalues;
    std::deque<Eigen::VectorXd> history_error;

    void find_permutation(const Eigen::MatrixXd &eigenvalues_old, const Eigen::MatrixXd &eigenvalues_new,
                          std::vector<Eigen::VectorXi> &permutations);

    Eigen::VectorXd flatten_eigenvalues(const Eigen::MatrixXd &eigenvalues) const;

    bool solve_diis_equations(Eigen::VectorXd &coeffs);
};

/**
 * @brief GDIIS optimizer with independent mixing for each k-point.
 *
 * This class maintains separate DIIS histories for each k-point, allowing
 * for different convergence rates at different k-points. This can be more
 * efficient than global mixing when some k-points converge faster than others.
 *
 * Algorithm:
 * 1. Store history of matrices for each k-point independently
 * 2. Compute eigenvalue-based error for each k-point
 * 3. DIIS extrapolation: M_new[k] = Σ c_{i,k} * M_{i}[k] (coefficients differ per k)
 * 4. Each k-point has its own DIIS equation to solve
 *
 * Use case: SCPH with varying convergence rates across k-space
 */
class GDIIS_PerKpoint
{
public:
    /**
     * @brief Constructor for per-k-point GDIIS optimizer.
     *
     * @param[in] nk          Number of k-points
     * @param[in] max_history Maximum number of iterations to store per k-point
     * @param[in] mixing_beta Mixing parameter (0 < beta <= 1)
     * @param[in] verbosity   Verbosity level for logging (0: silent, >0: print info)
     */
    GDIIS_PerKpoint(int nk, int max_history = 5, double mixing_beta = 0.5, int verbosity = 0);

    ~GDIIS_PerKpoint() = default;

    /**
     * @brief Updates the DIIS history with new matrices and eigenvalues.
     *
     * @param[in] matrices    Current matrices (vector of nk MatrixXcd, each ns × ns)
     * @param[in] eigenvalues Current eigenvalues for error computation (MatrixXd: nk × ns)
     */
    void push(const std::vector<Eigen::MatrixXcd> &matrices, const Eigen::MatrixXd &eigenvalues);

    /**
     * @brief Computes extrapolated matrices using per-k-point DIIS.
     *
     * Each k-point is extrapolated independently with its own coefficients.
     *
     * @param[out] matrices_new  Extrapolated matrices
     * @return                   true if at least one k-point used DIIS successfully
     */
    bool extrapolate(std::vector<Eigen::MatrixXcd> &matrices_new);

    /**
     * @brief Clears the DIIS history for all k-points.
     */
    void clear();

    /**
     * @brief Returns the minimum history size across all k-points.
     *
     * @return Minimum number of iterations stored
     */
    [[nodiscard]] int size() const;

    /**
     * @brief Checks if DIIS has enough history for extrapolation.
     *
     * @return true if at least one k-point has >= 2 iterations stored
     */
    [[nodiscard]] bool is_ready() const;

    /**
     * @brief Sets the maximum history size for all k-points.
     *
     * @param[in] max_hist New maximum history size
     */
    void set_max_history(int max_hist);

    /**
     * @brief Sets the mixing parameter for all k-points.
     *
     * @param[in] beta Mixing parameter (0 < beta <= 1)
     */
    void set_mixing_beta(double beta);

    /**
     * @brief Get statistics on DIIS success rate per k-point.
     *
     * @return Vector of success counts for each k-point
     */
    [[nodiscard]] std::vector<int> get_success_stats() const
    {
        return success_count_;
    }

private:
    int nk_;             ///< Number of k-points
    int max_history_;    ///< Maximum history per k-point
    double mixing_beta_; ///< Mixing parameter
    int verbosity_;      ///< Verbosity level

    // Independent GDIIS instance for each k-point
    std::vector<GDIIS_Matrix> diis_per_k_;

    // Track success/failure statistics
    std::vector<int> success_count_;
    std::vector<int> failure_count_;
};
