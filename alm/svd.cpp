//
// Created by Terumasa Tadano on 25/06/03.
//

#include "svd.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <tuple>
#include "lapack_wrapper.h"

/**
 * @brief Computes the SVD in thin U/V mode and returns the result as a tuple.
 *
 * @param A           Input matrix (m×n).
 * @param use_eigen   If true, uses Eigen::JacobiSVD; if false, uses LAPACK’s dgesvd_.
 * @return            A tuple {U, S, VT}:
 *                   - U  : Thin U matrix (m×r) as Eigen::MatrixXd
 *                   - S  : Singular values vector (length = r) as Eigen::VectorXd
 *                   - VT : Thin Vᵀ matrix (r×n) as Eigen::MatrixXd
 */
auto compute_svd_thin(const Eigen::MatrixXd &A, const bool use_eigen)
    -> std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd>
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int min_mn = std::min(m, n);

    if (use_eigen) {
        // Eigen version: Compute thin U and V
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_eig(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXd U = svd_eig.matrixU();              // (m × min_mn)
        Eigen::VectorXd S = svd_eig.singularValues();       // (min_mn)
        Eigen::MatrixXd VT = svd_eig.matrixV().transpose(); // (min_mn × n)
        return {U, S, VT};
    } else {
        // LAPACK version: use 'S' for thin U and 'S' for thin VT
        // Prepare a column-major buffer for Fortran.
        // Cast to size_t before multiplying to avoid int overflow for large matrices.
        double *A_lap = new double[static_cast<size_t>(m) * n];
        double *S_lap = new double[min_mn];
        double *U_lap = new double[static_cast<size_t>(m) * min_mn];
        double *VT_lap = new double[static_cast<size_t>(min_mn) * n];

        // Copy A into A_lap in column-major order: A_lap[i + j*m] = A(i, j)
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                A_lap[static_cast<size_t>(i) + static_cast<size_t>(j) * m] = A(i, j);
            }
        }

        char jobu = 'S';  // thin U (m × min_mn)
        char jobvt = 'S'; // thin VT (min_mn × n)
        int lda = m;
        int ldu = m;
        int ldvt = min_mn;
        int mm = m;
        int nn = n;
        int max_mn = std::max(m, n);
        int lwork = std::max(3 * min_mn + max_mn, 5 * min_mn);
        double *work = new double[lwork];
        int info = 0;

        dgesvd_(&jobu, &jobvt, &mm, &nn, A_lap, &lda, S_lap, U_lap, &ldu, VT_lap, &ldvt, work, &lwork, &info);

        if (info < 0) {
            std::cerr << "dgesvd_: Argument " << -info << " had an illegal value.\n";
        } else if (info > 0) {
            std::cerr << "dgesvd_: No convergence (info=" << info << ")\n";
        }

        // Map the Fortran column-major buffers into Eigen matrices
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> U_map(U_lap, m, min_mn);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> VT_map(VT_lap, min_mn, n);
        Eigen::Map<Eigen::VectorXd> S_map(S_lap, min_mn);

        Eigen::MatrixXd U = U_map;   // (m × min_mn)
        Eigen::VectorXd S = S_map;   // (min_mn)
        Eigen::MatrixXd VT = VT_map; // (min_mn × n)

        // Cleanup
        delete[] A_lap;
        delete[] S_lap;
        delete[] U_lap;
        delete[] VT_lap;
        delete[] work;

        return {U, S, VT};
    }
}

/**
 * Compute the Moore–Penrose pseudoinverse A^+ of an m×n matrix A using SVD via pure LAPACK.
 * A is given in column-major layout of size (m×n). The output A_pinv must be a pre-allocated
 * array of size (n×m), also column-major, which will hold A^+.
 *
 * @param m          Number of rows of A.
 * @param n          Number of columns of A.
 * @param A          Pointer to input matrix A (length m*n), column-major.
 *                   A will be overwritten.
 * @param rank       Rank of matrix A
 * @param A_pinv     Pointer to output pseudoinverse (length n*m), column-major.
 *                   On exit, contains A^+.
 * @param tol_factor (optional) Factor for tolerance: tol = tol_factor * max(m,n) * σ_max * ε.
 *                   If <= 0, defaults to ε * max(m,n) * σ_max.
 * @return           0 on success, nonzero LAPACK info on failure.
 */
auto lapack_pseudoinverse(int m, int n, double *A, int &rank, double *A_pinv, const double tol_factor) -> int
{
    assert(m > 0 && n > 0);

    // Copy A into local buffer because dgesdd_ overwrites input.
    std::vector<double> A_copy(static_cast<size_t>(m) * static_cast<size_t>(n));
    std::copy(A, A + static_cast<size_t>(m) * n, A_copy.begin());

    int min_mn = std::min(m, n);
    std::vector<double> S(min_mn);
    std::vector<double> U(static_cast<size_t>(m) * static_cast<size_t>(m));
    std::vector<double> VT(static_cast<size_t>(n) * static_cast<size_t>(n));

    // Query workspace for SVD
    char jobz = 'A';
    int lda = m, ldu = m, ldvt = n, info = 0;
    int lwork = -1;
    std::vector<int> iwork(8 * min_mn);
    double wkopt = 0.0;

    dgesdd_(&jobz,
            &m,
            &n,
            A_copy.data(),
            &lda,
            S.data(),
            U.data(),
            &ldu,
            VT.data(),
            &ldvt,
            &wkopt,
            &lwork,
            iwork.data(),
            &info);
    if (info != 0) return info;

    lwork = static_cast<int>(wkopt);
    std::vector<double> work_svd(static_cast<size_t>(lwork));

    // Compute SVD
    dgesdd_(&jobz,
            &m,
            &n,
            A_copy.data(),
            &lda,
            S.data(),
            U.data(),
            &ldu,
            VT.data(),
            &ldvt,
            work_svd.data(),
            &lwork,
            iwork.data(),
            &info);
    if (info != 0) return info;

    // Determine tolerance
    double eps = dlamch_((char *)"E");
    double sigma_max = (min_mn > 0 ? S[0] : 0.0);
    double tol = (tol_factor > 0.0) ? tol_factor * std::max(m, n) * sigma_max * eps : std::max(m, n) * sigma_max * eps;

    // Build D (n×m) with Σ^+ on the diagonal
    std::vector<double> D(static_cast<size_t>(n) * static_cast<size_t>(m), 0.0);
    rank = 0;
    for (int i = 0; i < min_mn; ++i) {
        if (S[i] > tol) {
            D[static_cast<size_t>(i) + static_cast<size_t>(i) * n] = 1.0 / S[i];
            ++rank;
        }
    }

    // Build U^T in UT (m×m)
    std::vector<double> UT(static_cast<size_t>(m) * static_cast<size_t>(m));
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < m; ++c) {
            UT[static_cast<size_t>(c) * m + r] = U[static_cast<size_t>(r) * m + c];
        }
    }

    // Compute M1 = D * UT  → (n×m) * (m×m) = (n×m)
    std::vector<double> M1(static_cast<size_t>(n) * static_cast<size_t>(m), 0.0);
    {
        char transa = 'N', transb = 'N';
        int mm = n, nn = m, kk = m;
        double alpha = 1.0, beta = 0.0;
        int ldda = n, lddb = m, lddc = n;
        dgemm_(&transa, &transb, &mm, &nn, &kk, &alpha, D.data(), &ldda, UT.data(), &lddb, &beta, M1.data(), &lddc);
    }

    // Compute A_pinv = V * M1, where V = VT^T
    std::vector<double> A_pinv_col(static_cast<size_t>(n) * static_cast<size_t>(m), 0.0);
    {
        char transa = 'T', transb = 'N';
        int mm = n, nn = m, kk = n;
        double alpha = 1.0, beta = 0.0;
        int ldv_t = n, ldm1 = n, ldcp = n;
        dgemm_(&transa,
               &transb,
               &mm,
               &nn,
               &kk,
               &alpha,
               VT.data(),
               &ldv_t,
               M1.data(),
               &ldm1,
               &beta,
               A_pinv_col.data(),
               &ldcp);
    }

    // Copy to output
    std::copy(A_pinv_col.begin(), A_pinv_col.end(), A_pinv);
    return 0;
}

/**
 * Compute the Moore–Penrose pseudoinverse of A using Eigen’s SVD.
 *
 * @param A      Input matrix (m×n).
 * @param rank   Rank of matrix A
 * @param tol    Tolerance for small singular values: tol = tol_factor * max(m,n) * σ_max * ε.
 *               If ≤ 0, Eigen’s default threshold is used.
 * @return       Pseudoinverse A^+ (n×m).
 */
auto eigen_pseudoinverse(const Eigen::MatrixXd &A, int &rank, double tol_factor) -> Eigen::MatrixXd
{
    int m = static_cast<int>(A.rows());
    int n = static_cast<int>(A.cols());
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto &S = svd.singularValues();
    const auto &U = svd.matrixU();
    const auto &V = svd.matrixV();

    int min_mn = std::min(m, n);
    double eps = std::numeric_limits<double>::epsilon();
    double sigma_max = (min_mn > 0 ? S(0) : 0.0);
    double tol = (tol_factor > 0.0) ? tol_factor * std::max(m, n) * sigma_max * eps : eps * std::max(m, n) * sigma_max;

    // Build Σ^+ (n×m)
    Eigen::MatrixXd Sigma_plus = Eigen::MatrixXd::Zero(n, m);
    rank = 0;
    for (int i = 0; i < min_mn; ++i) {
        if (S(i) > tol) {
            Sigma_plus(i, i) = 1.0 / S(i);
            ++rank;
        }
    }
    // A^+ = V * Σ^+ * U^T
    return V * Sigma_plus * U.transpose();
}
