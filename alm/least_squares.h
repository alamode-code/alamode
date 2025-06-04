//
// Created by Terumasa Tadano on 25/06/04.
//

#pragma once
#include <cstddef> // for size_t

/**
 * GQR-based constrained least squares solver:
 *   minimize ||A x - b||_2  subject to  C x = d
 *
 * This function first identifies linearly independent rows of C
 * via QR with column pivoting, extracts a reduced constraint
 * matrix C_red and corresponding d_red, and then calls LAPACK's
 * dgglse to compute the constrained least squares solution.
 *
 * Parameters:
 *   N           : number of columns of A (number of unknowns)
 *   M           : number of rows of A (number of data points)
 *   P           : number of rows of C (number of constraints)
 *   amat        : pointer to A in column-major order (size M×N)
 *   bvec        : pointer to b (size M)
 *   param_out   : output array for solution x (size N)
 *   cmat        : array of P pointers, each pointing to an array of length N (C in row-major)
 *   dvec        : pointer to d (size P)
 *   verbosity   : if > 0, print diagnostic messages
 *
 * Returns:
 *   INFO from dgglse (0 for success, >0 for failure)
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


//
// Created by Terumasa Tadano on 25/06/04.
//

#pragma once
#include <cstddef> // for size_t

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
