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
