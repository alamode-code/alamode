//
// Created by Terumasa Tadano on 25/06/04.
//

#pragma once

#include <cstddef> // for size_t
#include "constraint.h"
#include <Eigen/Sparse>

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
int least_squares_svd(const size_t N,
                      const size_t M,
                      double *amat,
                      const double *bvec,
                      double *param_out,
                      const int verbosity);

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
int least_squares_with_constraints_gqr(const size_t N,
                                       const size_t M,
                                       const size_t P,
                                       double *amat,
                                       const double *bvec,
                                       double *param_out,
                                       const double *const *cmat,
                                       const double *dvec,
                                       const int verbosity);

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
int least_squares_with_constraints_svd(const size_t N,
                                       const size_t M,
                                       const size_t P,
                                       double *amat,              // A: (M×N) column-major, can be overwritten
                                       double *bvec,              // b: (M) vector, can be overwritten
                                       double *param_out,         // output x (length N)
                                       const double *const *cmat, // C[i][j] pointer array (not contiguous)
                                       const double *dvec_orig,   // d: (P) vector, will be copied locally
                                       const int verbosity);



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
int least_squares_eigen_sparse_solver(const Eigen::SparseMatrix<double> &sp_mat,
                                      const Eigen::VectorXd &sp_bvec,
                                      Eigen::VectorXd &x_out,
                                      const std::string &solver_type,
                                      const double tolerance_iteration,
                                      const int maxnum_iteration);
