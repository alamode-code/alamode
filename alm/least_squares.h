//
// Created by Terumasa Tadano on 25/06/04.
//

#pragma once

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cstddef> // for size_t
#include "constraint.h"

/**
 * @brief Solves an unconstrained least squares problem.
 *
 * This function minimizes ||A x - b||_2, where:
 * - A is the data matrix (M×N)
 * - b is the observation vector (length M)
 * - x is the solution vector (length N)
 *
 * The method computes the least squares solution using standard numerical techniques.
 *
 * @param N           Number of columns of A (number of unknowns).
 * @param M           Number of rows of A (number of data points).
 * @param amat        Pointer to A in column-major order (size M×N).
 * @param bvec        Pointer to b (size M).
 * @param param_out   Output array for solution x (size N).
 * @param verbosity   If > 0, print diagnostic messages.
 * @return            0 for success, >0 for failure.
 */
auto least_squares_svd(const size_t N, const size_t M, double *amat, const double *bvec, double *param_out,
                       const int verbosity) -> int;

/**
 * @brief Solves a constrained least squares problem using GQR-based decomposition.
 *
 * This function minimizes ||A x - b||_2 subject to C x = d, where:
 * - A is the data matrix (M×N)
 * - b is the observation vector (length M)
 * - C is the constraint matrix (P×N)
 * - d is the constraint vector (length P)
 *
 * The method uses QR decomposition with column pivoting to identify linearly independent
 * rows of C, extracts a reduced constraint matrix C_red and corresponding d_red, and
 * then calls LAPACK's dgglse to compute the constrained least squares solution.
 *
 * @param N           Number of columns of A (number of unknowns).
 * @param M           Number of rows of A (number of data points).
 * @param P           Number of rows of C (number of constraints).
 * @param amat        Pointer to A in column-major order (size M×N).
 * @param bvec        Pointer to b (size M).
 * @param param_out   Output array for solution x (size N).
 * @param cmat        Array of P pointers, each pointing to an array of length N (C in row-major).
 * @param dvec        Pointer to d (size P).
 * @param verbosity   If > 0, print diagnostic messages.
 * @return            INFO from dgglse (0 for success, >0 for failure).
 */
auto least_squares_with_constraints_gqr(const size_t N, const size_t M, const size_t P, double *amat,
                                        const double *bvec, double *param_out, const double *const *cmat,
                                        const double *dvec, const int verbosity) -> int;

/**
 * @brief Solves a constrained least squares problem using SVD-based decomposition.
 *
 * This function minimizes ||A x - b||_2 subject to C x = d, where:
 * - A is the data matrix (M×N)
 * - b is the observation vector (length M)
 * - C is the constraint matrix (P×N)
 * - d is the constraint vector (length P)
 *
 * The method computes the pseudoinverse of C to find a particular solution x₀ = C⁺d,
 * projects the residual b' = b - A x₀ onto the null space of C, and solves the
 * least squares problem in the null space using SVD.
 *
 * @param N           Number of columns of A (number of unknowns).
 * @param M           Number of rows of A (number of data points).
 * @param P           Number of rows of C (number of constraints).
 * @param amat        Pointer to A in column-major order (size M×N). Can be overwritten.
 * @param bvec        Pointer to b (size M). Can be overwritten.
 * @param param_out   Output array for solution x (size N).
 * @param cmat        Array of P pointers, each pointing to an array of length N (C in row-major).
 * @param dvec_orig   Pointer to d (size P). Will be copied locally.
 * @param verbosity   If > 0, print diagnostic messages.
 * @return            0 for success, >0 for failure.
 */
auto least_squares_with_constraints_svd(const size_t N, const size_t M, const size_t P,
                                        double *amat,              // A: (M×N) column-major, can be overwritten
                                        double *bvec,              // b: (M) vector, can be overwritten
                                        double *param_out,         // output x (length N)
                                        const double *const *cmat, // C[i][j] pointer array (not contiguous)
                                        const double *dvec_orig,   // d: (P) vector, will be copied locally
                                        const int verbosity) -> int;


/**
 * @brief Solves a sparse least squares problem using Eigen's sparse solvers.
 *
 * This function minimizes ||A x - b||_2, where:
 * - A is a sparse matrix represented using Eigen's SparseMatrix.
 * - b is the observation vector (Eigen::VectorXd).
 * - x is the solution vector (Eigen::VectorXd).
 *
 * The method uses a specified sparse solver to compute the least squares solution.
 *
 * @param sp_mat                Sparse matrix A (M×N) in Eigen::SparseMatrix format.
 * @param sp_bvec               Observation vector b (length M) in Eigen::VectorXd format.
 * @param x_out                 Output vector for the solution x (length N).
 * @param solver_type           String specifying the solver type (e.g., "CG", "BiCGSTAB").
 * @param tolerance_iteration   Convergence tolerance for the iterative solver.
 * @param maxnum_iteration      Maximum number of iterations for the solver.
 * @return                      0 for success, >0 for failure.
 */
auto least_squares_eigen_sparse_solver(const Eigen::SparseMatrix<double> &sp_mat, const Eigen::VectorXd &sp_bvec,
                                       Eigen::VectorXd &x_out, const std::string &solver_type,
                                       const double tolerance_iteration, const int maxnum_iteration) -> int;


/**
 * @brief  Build reduced constraint matrix C_red (r×N) and vector d_red (length r).
 *
 * This function takes the original constraint matrix C (size P×N) and constraint vector d
 * (length P), determines the numerical row rank r of C via QR with column pivoting on C^T,
 * and extracts the first r independent rows. The outputs C_red and d_red are stored in
 * column-major order.
 *
 * @param[in]   N          Number of variables (number of columns of C)
 * @param[in]   P          Number of constraints (number of rows of C)
 * @param[in]   cmat       Pointer to constraint matrix C: cmat[i][j] is row i, column j (0 ≤ i < P, 0 ≤ j < N)
 * @param[in]   dvec       Original constraint vector d (length P)
 * @param[in]   verbosity  Verbosity level (0: silent, >0: print info)
 * @param[out]  C_red      Output reduced constraint matrix (r×N) in column-major layout
 * @param[out]  d_red      Output reduced constraint vector (length r)
 * @param[out]  r          Computed numerical row rank (0 ≤ r ≤ min(P, N))
 *
 * @return  0 on success, nonzero LAPACK info code on failure
 */
auto get_independent_rows(const size_t N, const size_t P, const double *const *cmat, const double *dvec,
                          const int verbosity, std::vector<double> &C_red, std::vector<double> &d_red, int &r) -> int;


/**
 * @brief Solves a constrained least squares problem using sparse matrices.
 *
 * This function minimizes ||A x - b||_2 subject to C x = d, where:
 * - A is the data matrix (M×N) in sparse format.
 * - b is the observation vector (length M).
 * - C is the constraint matrix (P×N) in sparse format.
 * - d is the constraint vector (length P).
 * - x is the solution vector (length N).
 * - lambda is the vector of Lagrange multipliers (length P).
 *
 * The method uses QR decomposition with column pivoting to handle the constraints
 * and solve the least squares problem efficiently in sparse form.
 *
 * @param[in] A       Sparse matrix A (M×N) representing the data matrix.
 * @param[in] b       Observation vector b (length M).
 * @param[in] C       Sparse matrix C (P×N) representing the constraint matrix.
 * @param[in] d       Constraint vector d (length P).
 * @param[out] x       Output solution vector x (length N).
 * @param[out] lambda  Output vector of Lagrange multipliers (length P).
 * @param[in] verbosity  Verbosity level (0: silent, >0: print info).
 */
auto solveGQRSparse(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b,
                    const Eigen::SparseMatrix<double> &C, const Eigen::VectorXd &d, Eigen::VectorXd &x,
                    Eigen::VectorXd &lambda, const int verbosity = 0, const std::string &solver_type = "",
                    const double tolerance_iteration = 1.0e-8, const int maxnum_iteration = 10000) -> void;

/**
 * Given a dense column-major matrix A (size M×N) stored in
 * `A_data` (length M*N), find which rows are independent.
 *
 * @param M       # rows of A
 * @param N       # cols of A
 * @param A_data  pointer to column-major data (size M*N)
 * @param tol     optional tolerance; if ≤0, compute tol = max(M,N)*|R₀₀|*ε
 * @param rank    [out] computed numerical rank
 * @param pivots  [out] size-rank array of 0-based row indices in pivot order
 * @param verbosity [in] verbosity level (0: silent, >0: print info)
 *
 * @returns      0 on success, LAPACK INFO otherwise.
 */
auto find_independent_rows_dense(int M, int N, double *A_data, double tol, int &rank, std::vector<int> &pivots,
                                 const int verbosity = 0) -> int;


// Sentinel for find_independent_rows_dense(): use the existing LAPACK-style auto tolerance
// max(M, N) * |R(0,0)| * eps. This preserves the current QR rank policy.
constexpr double rank_tolerance_auto = -1.0;


/// Extract the independent rows of a (P×N) sparse matrix C_sparse,
/// using LAPACK QR-with-pivoting on Cᵀ, just like your original code.
///
/// @param C_sparse  Input (P×N), row-major sparse
/// @param dvec      Input length-P
/// @param verbosity >1 prints debug
/// @param tolerance Rank tolerance. Use rank_tolerance_auto to preserve the auto policy.
/// @param C_red     Output (r×N) row-major sparse of independent rows
/// @param d_red     Output length-r of corresponding dvec entries
/// @param r         Output numerical row-rank
/// @returns 0 on success, non-zero LAPACK INFO on failure
auto get_independent_rows_lapack_sparse(const Eigen::SparseMatrix<double> &C_sparse, const Eigen::VectorXd &dvec,
                                        const int verbosity, const double tolerance, Eigen::SparseMatrix<double> &C_red,
                                        Eigen::VectorXd &d_red, int &r) -> int;


auto get_independent_rows_lapack_sparse(const size_t ncols, ConstraintSparseForm &C_sparse, const int verbosity,
                                        const double tolerance, int &r) -> int;
