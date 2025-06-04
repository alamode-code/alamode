//
// Created by Terumasa Tadano on 25/06/03.
//

#include "svd.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include <fstream>
#include <chrono>
#include <tuple>

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
std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd>
compute_svd_thin(const Eigen::MatrixXd &A, const bool use_eigen)
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int min_mn = std::min(m, n);

    if (use_eigen) {
        // Eigen version: Compute thin U and V
        Eigen::JacobiSVD svd_eig(
            A,
            Eigen::ComputeThinU | Eigen::ComputeThinV
            );
        Eigen::MatrixXd U = svd_eig.matrixU();              // (m × min_mn)
        Eigen::VectorXd S = svd_eig.singularValues();       // (min_mn)
        Eigen::MatrixXd VT = svd_eig.matrixV().transpose(); // (min_mn × n)
        return {U, S, VT};
    } else {
        // LAPACK version: use 'S' for thin U and 'S' for thin VT
        // Prepare a column-major buffer for Fortran
        double *A_lap = new double[m * n];
        double *S_lap = new double[min_mn];
        double *U_lap = new double[m * min_mn];
        double *VT_lap = new double[min_mn * n];

        // Copy A into A_lap in column-major order: A_lap[i + j*m] = A(i, j)
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                A_lap[i + j * m] = A(i, j);
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

        dgesvd_(
            &jobu,
            &jobvt,
            &mm,
            &nn,
            A_lap,
            &lda,
            S_lap,
            U_lap,
            &ldu,
            VT_lap,
            &ldvt,
            work,
            &lwork,
            &info
            );

        if (info < 0) {
            std::cerr << "dgesvd_: Argument " << -info << " had an illegal value.\n";
        } else if (info > 0) {
            std::cerr << "dgesvd_: No convergence (info=" << info << ")\n";
        }

        // Map the Fortran column-major buffers into Eigen matrices
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
            U_map(U_lap, m, min_mn);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
            VT_map(VT_lap, min_mn, n);
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
