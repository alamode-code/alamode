// least_squares.cpp

#include "least_squares.h"
#include <Eigen/Sparse>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <cmath> // for std::pow, std::sqrt
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>
#include "logger.h"
#include "svd.h"
#ifdef USE_MKL_BACKEND
#define EIGEN_USE_MKL
#include <Eigen/PardisoSupport>
using KKT_Solver = Eigen::PardisoLDLT<Eigen::SparseMatrix<double>>;
#elif defined(USE_ACCEL_BACKEND)
#include <Eigen/AccelerateSupport>
#include <Eigen/SparseCholesky>
using KKT_Solver = Eigen::AccelerateLDLT<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Symmetric>;
#else
#include <Eigen/SparseLU>
using KKT_Solver = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Symmetric>;
#endif

#ifndef EIGEN_USE_ACCELERATE
#include "lapack_wrapper.h"
#endif


auto find_independent_rows_dense(int M, int N, double *A_data, double tol, int &rank, std::vector<int> &pivots,
                                 const int verbosity) -> int
{
    // jpvt array holds pivot indices (1-based)
    std::vector<int> jpvt(M, 0);
    int minNP = std::min(N, M);
    std::vector<double> tau(minNP);

    // Query optimal workspace size for dgeqp3_
    int lwork_qr = -1;
    double work_qr_query = 0.0;
    int INFO_qr = 0;
    dgeqp3_(&N, &M, A_data, &N, jpvt.data(), tau.data(), &work_qr_query, &lwork_qr, &INFO_qr);
    if (INFO_qr != 0) {
        return INFO_qr;
    }
    int LWORK_qr = static_cast<int>(work_qr_query);
    std::vector<double> WORK_qr(LWORK_qr);

    LOG_IF(verbosity, 1, "dgeqp3_ workspace size query completed, INFO=", INFO_qr, ".\n");
    // Perform QR with column pivoting to determine numerical row rank of A_data
    dgeqp3_(&N, &M, A_data, &N, jpvt.data(), tau.data(), WORK_qr.data(), &LWORK_qr, &INFO_qr);
    if (INFO_qr != 0) {
        return INFO_qr;
    }
    LOG_IF(verbosity, 1, "dgeqp3_ completed successfully, INFO=", INFO_qr, ".\n");

    // 4) Determine numerical row rank by inspecting diagonal of R in A_data
    // R[k, k] is stored at A_qr[k + k*N_i] in column-major
    double const R00 = std::abs(A_data[0 + 0 * N]);
    if (tol <= 0.0) {
        // If tol is not provided, compute it based on the maximum of M and N
        tol = std::max(M, N) * R00 * std::numeric_limits<double>::epsilon();
    }
    rank = 0;
    for (int k = 0; k < std::min(M, N); ++k) {
        double const diag = std::abs(A_data[k + k * N]);
        if (diag > tol) {
            ++rank;
        } else {
            break;
        }
    }
    //Extract the indices of the first r pivoted rows of A^T (which correspond to independent rows of A)
    pivots.resize(rank);
    for (int i = 0; i < rank; ++i) {
        pivots[i] = jpvt[i] - 1; // convert 1-based to 0-based
    }

    return INFO_qr;
}


auto get_independent_rows(const size_t N, const size_t P, const double *const *cmat, const double *dvec,
                          const int verbosity, std::vector<double> &C_red, std::vector<double> &d_red, int &r) -> int
{
    // Convert N and P to int for LAPACK calls
    int N_i = static_cast<int>(N);
    int P_i = static_cast<int>(P);

    // 1) Copy original C into a contiguous column-major buffer C_mat (size P×N)
    std::vector<double> C_mat(P_i * N_i);
    for (int row = 0; row < P_i; ++row) {
        for (int col = 0; col < N_i; ++col) {
            // C_mat is stored column-major: index = col * P_i + row
            C_mat[col * P_i + row] = cmat[row][col];
        }
    }
    LOG_IF(verbosity, 1, "Copied C into column-major buffer (", P_i, "×", N_i, ").\n");

    // 2) Prepare to perform QR with column pivoting on C^T (size N×P)
    // Build A_qr = C^T in column-major layout
    std::vector<double> A_qr(N_i * P_i);
    for (int i = 0; i < P_i; ++i) {
        for (int j = 0; j < N_i; ++j) {
            // Copy element (row=i, col=j) of C_mat into (row=j, col=i) of A_qr
            A_qr[j + i * N_i] = C_mat[i + j * P_i];
        }
    }

    // 3) Run QR with column pivoting on A_qr to determine numerical row rank
    std::vector<int> independent_rows;
    auto info_qr = find_independent_rows_dense(P_i, N_i, A_qr.data(), -1.0, r, independent_rows);
    if (info_qr != 0) {
        LOG_ERR_IF(verbosity, 0, "find_independent_rows_dense failed, INFO=", info_qr, ".\n");
        return info_qr;
    }
    // 4) Build C_red (size r×N) and d_red (length r), both in column-major layout
    C_red.assign(static_cast<size_t>(r) * static_cast<size_t>(N), 0.0);
    d_red.assign(static_cast<size_t>(r), 0.0);
    for (int ii = 0; ii < r; ++ii) {
        int orig_row = independent_rows[ii]; // original row index in C
        for (int col = 0; col < N_i; ++col) {
            // Copy C_mat[orig_row, col] into C_red[ii, col]
            // C_mat index: col * P_i + orig_row
            // C_red is r×N in column-major: index = col * r + ii
            C_red[col * r + ii] = C_mat[col * P_i + orig_row];
        }
        // Copy corresponding entry from dvec
        d_red[ii] = dvec[orig_row];
    }
    LOG_IF(verbosity, 1, "Constructed C_red (", r, "×", N_i, ") and d_red (length ", r, ").\n");
    return 0;
}

auto get_independent_rows_lapack_sparse(const size_t ncols, ConstraintSparseForm &C_sparse, const int verbosity,
                                        const double tolerance, int &r) -> int
{
    Eigen::SparseMatrix<double> C_sparse_eigen, C_red_eigen;

    // Copy the sparse matrix C_sparse to Eigen format
    const auto nrows = C_sparse.size();
    C_sparse_eigen.resize(nrows, ncols);
    C_red_eigen.resize(nrows, ncols);

    std::vector<Eigen::Triplet<double>> triplets;
    size_t icount = 0;
    for (const auto &row: C_sparse) {
        for (const auto &elem: row) {
            triplets.emplace_back(icount, elem.first, elem.second);
        }
        ++icount;
    }
    C_sparse_eigen.setFromTriplets(triplets.begin(), triplets.end());
    C_sparse_eigen.makeCompressed();
    LOG_IF(verbosity,
           1,
           "Converted C_sparse to Eigen format (",
           nrows,
           "×",
           ncols,
           "), nnz=",
           C_sparse_eigen.nonZeros(),
           ".\n");

    Eigen::VectorXd dvec(nrows), dvec_red(nrows);
    const auto info = get_independent_rows_lapack_sparse(C_sparse_eigen, dvec, verbosity, C_red_eigen, dvec_red, r);

    // update the sparse matrix C_sparse with the reduced rows
    C_sparse.clear();
    Eigen::SparseMatrix<double, Eigen::RowMajor> C_row = C_red_eigen;

    MapConstraintElement const_tmp;
    for (auto i = 0; i < r; ++i) {
        const_tmp.clear();
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(C_row, i); it; ++it) {
            const_tmp[it.col()] = it.value();
        }
        C_sparse.emplace_back(const_tmp);
    }

    return info;
}

auto get_independent_rows_lapack_sparse(const Eigen::SparseMatrix<double> &C_sparse, const Eigen::VectorXd &dvec,
                                        const int verbosity, Eigen::SparseMatrix<double> &C_red, Eigen::VectorXd &d_red,
                                        int &r) -> int
{
    const int P = C_sparse.rows();
    const int N = C_sparse.cols();
    int N_i = N, P_i = P;

    LOG_IF(verbosity, 1, "P = ", P, ", N = ", N, ", nnz = ", C_sparse.nonZeros(), ".\n");

    // 1) Copy sparse→dense column-major buffer C_mat (size P×N)
    std::vector<double> C_mat(P_i * N_i, 0.0);
    for (int col = 0; col < N; ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(C_sparse, col); it; ++it) {
            // it.row()==i, it.col()==j
            const int row = it.row();
            C_mat[col * P_i + row] = it.value();
        }
    }
    LOG_IF(verbosity, 1, "build dense C_mat (", P_i, "×", N_i, ").\n");

    // 2) Build A_qr = Cᵀ in column-major (size N×P)
    std::vector<double> A_qr(N_i * P_i);
    for (int i = 0; i < P_i; ++i)
        for (int j = 0; j < N_i; ++j)
            A_qr[j + i * N_i] = C_mat[i + j * P_i];

    // 3) Run QR with column pivoting on A_qr to determine numerical row rank
    std::vector<int> independent_rows;
    auto info_qr = find_independent_rows_dense(P_i, N_i, A_qr.data(), -1.0, r, independent_rows, verbosity);
    if (info_qr != 0) {
        LOG_ERR_IF(verbosity, 0, "find_independent_rows_dense failed, INFO=", info_qr, ".\n");
        return info_qr;
    }

    Eigen::SparseMatrix<double, Eigen::RowMajor> C_row = C_sparse;

    // 4) Build C_red (r×N) as sparse
    std::vector<Eigen::Triplet<double, size_t>> tri;
    tri.reserve(C_sparse.nonZeros());
    for (int new_i = 0; new_i < r; ++new_i) {
        int orig_row = independent_rows[new_i];
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(C_row, orig_row); it; ++it) {
            tri.emplace_back(new_i, it.col(), it.value());
        }
    }
    C_red.resize(r, N);
    C_red.setFromTriplets(tri.begin(), tri.end());
    C_red.makeCompressed();
    LOG_IF(verbosity, 1, "built C_red (", r, "×", N, "), nnz=", C_red.nonZeros(), "\n");

    // 5) Build d_red
    d_red.resize(r);
    for (int i = 0; i < r; ++i)
        d_red[i] = dvec[independent_rows[i]];

    LOG_IF(verbosity, 1, "built d_red (len=", r, ").\n");

    return 0;
}

auto least_squares_svd(const size_t N, const size_t M, double *amat, const double *bvec, double *param_out,
                       const int verbosity) -> int
{
    // Local variables
    int nrhs = 1;
    int nrank = 0;
    int INFO = 0;
    int M_tmp = static_cast<int>(M);
    int N_tmp = static_cast<int>(N);
    double rcond = -1.0;   // use default machine precision tolerance
    double f_square = 0.0; // sum of squares of entries of bvec
    const int LMIN = static_cast<int>(std::min(N, M));
    int LMAX = static_cast<int>(std::max(N, M));

    // Compute recommended WORK size:
    // LWORK = 2 * (3*LMIN + max(2*LMIN, LMAX))
    int recommended = 3 * LMIN + std::max(2 * LMIN, LMAX);
    int LWORK = 2 * recommended;

    if (verbosity > 0) {
        std::cout << "  Entering fitting routine: SVD without constraints\n";
    }

    // Use std::vector to manage workspace and buffers, avoiding manual allocate/deallocate
    std::vector<double> work_buf(LWORK);
    std::vector<double> singular_vals(LMIN);
    std::vector<double> fsum2(LMAX);

    // Copy bvec into fsum2 and compute its norm-squared
    for (size_t i = 0; i < M; ++i) {
        fsum2[i] = bvec[i];
        f_square += std::pow(bvec[i], 2);
    }
    // Zero out any remaining entries of fsum2 if M < LMAX
    for (size_t i = M; i < static_cast<size_t>(LMAX); ++i) {
        fsum2[i] = 0.0;
    }

    if (verbosity > 0) {
        std::cout << "  SVD has started ... ";
    }

    // Call LAPACK's DGELSS to solve min || A*x - b || using SVD
    dgelss_(&M_tmp,
            &N_tmp,
            &nrhs,
            amat,
            &M_tmp,
            // leading dimension of A is M
            fsum2.data(),
            // on exit, first N entries contain solution x
            &LMAX,
            // leading dimension of b (stored in fsum2) is LMAX
            singular_vals.data(),
            &rcond,
            &nrank,
            work_buf.data(),
            &LWORK,
            &INFO);

    if (verbosity > 0) {
        std::cout << "finished!\n\n";
        std::cout << "  RANK of the matrix = " << nrank << '\n';
    }

    if (nrank < static_cast<int>(N)) {
        LOG_ERR_IF(verbosity, 0, "Matrix is rank-deficient. Force constants could not be determined uniquely.\n");
    }

    if (nrank == static_cast<int>(N) && verbosity > 0) {
        // Compute residual sum of squares: sum of squares of entries fsum2[N..M-1]
        double f_residual = 0.0;
        for (int i = static_cast<int>(N); i < static_cast<int>(M); ++i) {
            f_residual += std::pow(fsum2[i], 2);
        }
        std::cout << '\n' << "  Residual sum of squares for the solution: " << std::sqrt(f_residual) << '\n';
        std::cout << "  Fitting error (%) : " << std::sqrt(f_residual / f_square) * 100.0 << '\n';
    }

    // Copy solution (first N entries of fsum2) into param_out
    for (size_t i = 0; i < N; ++i) {
        param_out[i] = fsum2[i];
    }

    // Vectors automatically deallocate when going out of scope
    return INFO;
}


auto least_squares_with_constraints_gqr(const size_t N, const size_t M, const size_t P, double *amat,
                                        const double *bvec, double *param_out, const double *const *cmat,
                                        const double *dvec, const int verbosity) -> int
{
    // Copy b into a mutable array b_copy, since dgglse_ overwrites it
    std::vector<double> b_copy(M);
    for (size_t i = 0; i < M; ++i) {
        b_copy[i] = bvec[i];
    }
    const double f_square = std::inner_product(b_copy.begin(), b_copy.end(), b_copy.begin(), 0.0);

    std::vector<double> C_red;
    std::vector<double> d_red;
    int r = 0;
    int ierr = get_independent_rows(N, P, cmat, dvec, verbosity, C_red, d_red, r);
    if (ierr != 0) {
        // If reduction fails, return the error code
        return ierr;
    }
    LOG_IF(verbosity , 1, "get_independent_rows returned r = ", r, ".\n");

    // Step 5: Call dgglse to solve minimize ||A x - b||_2 subject to C_red x = d_red
    int M_i = static_cast<int>(M);
    int N_i = static_cast<int>(N);
    int r_i = r;
    int LWORK_lse = r_i + std::min(M_i, N_i) + 10 * std::max(M_i, N_i);
    std::vector<double> WORK_lse(LWORK_lse);
    std::vector<double> X(N_i, 0.0); // Solution vector x (length N)
    int INFO_lse = 0;

    dgglse_(&M_i,
            &N_i,
            &r_i,
            amat,
            // A (M×N), column-major
            &M_i,
            C_red.data(),
            // B = C_red (r×N), column-major
            &r_i,
            b_copy.data(),
            // b (length M)
            d_red.data(),
            // d_red (length r)
            X.data(),
            // solution x
            WORK_lse.data(),
            &LWORK_lse,
            &INFO_lse);

    if (INFO_lse == 0) {
        LOG_IF(verbosity, 1, "dgglse_ completed successfully.\n");
    } else {
        LOG_ERR_IF(verbosity, 0, "dgglse_ returned INFO =", INFO_lse, ".\n");
    }

    // Copy the solution into param_out
    for (int i = 0; i < N_i; ++i) {
        param_out[i] = X[i];
    }

    auto f_residual = 0.0;
    for (int i = N_i - r; i < M_i; ++i) {
        f_residual += b_copy[i] * b_copy[i];
    }

    if (verbosity > 0) {
        std::cout << '\n' << "  Residual sum of squares for the solution: " << sqrt(f_residual) << '\n';
        std::cout << "  Fitting error (%) : " << std::sqrt(f_residual / f_square) * 100.0 << '\n';
    }

    return INFO_lse;
}

auto least_squares_with_constraints_svd(const size_t N, const size_t M, const size_t P,
                                        double *amat,              // A: (M×N) column-major, may be overwritten
                                        double *bvec,              // b: (M) column vector, may be overwritten
                                        double *param_out,         // output x (length N)
                                        const double *const *cmat, // C[i][j] pointer array (row i, col j)
                                        const double *dvec_orig,   // d: (P) column vector
                                        const int verbosity) -> int
{
    // ------------------------------------------------------------
    // 1) Copy C into a contiguous column-major buffer; copy d into a vector
    // ------------------------------------------------------------
    int p = static_cast<int>(P);
    int n = static_cast<int>(N);
    int m = static_cast<int>(M);

    // C_copy: column-major, size P×N
    std::vector<double> C_copy(static_cast<size_t>(P) * static_cast<size_t>(N));
    for (size_t i = 0; i < P; ++i) {
        for (size_t j = 0; j < N; ++j) {
            // Place row i, col j of C into column-major index j*P + i
            C_copy[j * P + i] = cmat[i][j];
        }
    }

    // d_copy: length P
    std::vector<double> d_copy(dvec_orig, dvec_orig + P);

    // ------------------------------------------------------------
    // 2) Compute C^+ using SVD
    // ------------------------------------------------------------
    int rankC = 0; // Will be determined later
    std::vector<double> C_pinv(static_cast<size_t>(n) * p);
    lapack_pseudoinverse(p, n, C_copy.data(), rankC, C_pinv.data());

    // ------------------------------------------------------------
    // 3) Compute a special solution x0 = C^+ * d
    // ------------------------------------------------------------
    std::vector<double> x0(static_cast<size_t>(n), 0.0);

    if (rankC > 0) {
        char trans = 'N';
        double alpha = 1.0, beta = 0.0;
        int inc = 1;
        dgemv_(&trans, &n, &p, &alpha, C_pinv.data(), &n, d_copy.data(), &inc, &beta, x0.data(), &inc);
    }
    // If rankC == 0, x0 remains all zeros

    // ------------------------------------------------------------
    // 4) Compute residual b' = b − A * x0
    // ------------------------------------------------------------
    std::vector<double> b_prime(m);
    for (int i = 0; i < m; ++i)
        b_prime[i] = bvec[i];
    const auto f_square = std::inner_product(b_prime.begin(), b_prime.end(), b_prime.begin(), 0.0);
    {
        char trans = 'N';
        double alpha = -1.0, beta = 1.0;
        int inc = 1;
        dgemv_(&trans, &m, &n, &alpha, amat, &m, x0.data(), &inc, &beta, b_prime.data(), &inc);
    }
    // ------------------------------------------------------------
    // 5) Projector matrix N = I - C^+ * C (nxn)
    // ------------------------------------------------------------
    std::vector<double> Nproj(static_cast<size_t>(n) * n, 0.0);
    {
        char tA = 'N', tB = 'N';
        double alpha = 1.0, beta = 0.0;
        dgemm_(&tA, &tB, &n, &n, &p, &alpha, C_pinv.data(), &n, C_copy.data(), &p, &beta, Nproj.data(), &n);
        for (int i = 0; i < n * n; ++i) {
            Nproj[i] = -Nproj[i];
        }
        for (int i = 0; i < n; ++i)
            Nproj[i + i * n] += 1.0;
    }

    // ------------------------------------------------------------
    // 6) Compute A2 = A * N = A - A * C^+ * C
    // ------------------------------------------------------------
    std::vector<double> A2(static_cast<size_t>(m) * n, 0.0);
    {
        char tA = 'N', tB = 'N';
        double alpha = 1.0, beta = 0.0;
        dgemm_(&tA, &tB, &m, &n, &n, &alpha, amat, &m, Nproj.data(), &n, &beta, A2.data(), &m);
    }
    // ------------------------------------------------------------
    // 7) Solve least squares system A2 * y = b' without constraints.
    //    δ = A2^+ * b' is the solution.
    // ------------------------------------------------------------
    int rankA2 = 0;
    std::vector<double> A2_pinv(static_cast<size_t>(n) * m);
    lapack_pseudoinverse(m, n, A2.data(), rankA2, A2_pinv.data());

    std::vector<double> delta(n, 0.0);
    {
        char trans = 'N';
        double alpha = 1.0, beta = 0.0;
        int inc = 1;
        dgemv_(&trans, &n, &m, &alpha, A2_pinv.data(), &n, b_prime.data(), &inc, &beta, delta.data(), &inc);
    }

    // ------------------------------------------------------------
    // 8) Final solution x = x0 + δ
    // ------------------------------------------------------------
    for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
        param_out[i] = x0[i] + delta[i];
    }

    if (verbosity > 0) {
        std::vector<double> Ax(m, 0.0);

        char trans = 'N';
        double alpha = 1.0;
        double beta = 0.0;
        int incx = 1, incy = 1;
        dgemv_(&trans, &m, &n, &alpha, amat, &m, param_out, &incx, &beta, Ax.data(), &incy);

        double rss = 0.0;
        for (int i = n - p; i < m; ++i) {
            double ri = Ax[i] - bvec[i];
            rss += ri * ri;
        }

        std::cout << '\n' << "  Residual sum of squares for the solution: " << sqrt(rss) << '\n';
        std::cout << "  Fitting error (%) : " << std::sqrt(rss / f_square) * 100.0 << '\n';
    }


    return 0;
}

auto least_squares_eigen_sparse_solver(const Eigen::SparseMatrix<double> &sp_mat, const Eigen::VectorXd &sp_bvec,
                                       Eigen::VectorXd &x_out, const std::string &solver_type,
                                       const double tolerance_iteration, const int maxnum_iteration) -> int
{
    const auto solver_type_lower = boost::algorithm::to_lower_copy(solver_type);

    using SpMat = Eigen::SparseMatrix<double>;
    if (solver_type_lower == "simplicialldlt") {
        SpMat AtA = sp_mat.transpose() * sp_mat;
        Eigen::VectorXd AtB = sp_mat.transpose() * sp_bvec;
        Eigen::SimplicialLDLT<SpMat> ldlt(AtA);
        x_out = ldlt.solve(AtB);

        if (ldlt.info() != Eigen::Success) {
            std::cerr << "  Fitting by " + solver_type + " failed.\n";
            std::cerr << ldlt.info() << '\n';
            return 1;
        }

    } else if (solver_type_lower == "sparseqr") {

        Eigen::SparseQR<SpMat, Eigen::COLAMDOrdering<int>> qr(sp_mat);
        x_out = qr.solve(sp_bvec);

        if (qr.info() != Eigen::Success) {
            std::cerr << "  Fitting by " + solver_type + " failed.\n";
            std::cerr << qr.info() << '\n';
            return 1;
        }

    } else if (solver_type_lower == "conjugategradient") {
        SpMat AtA = sp_mat.transpose() * sp_mat;
        Eigen::VectorXd AtB = sp_mat.transpose() * sp_bvec;

        Eigen::ConjugateGradient<SpMat> cg(AtA);
        cg.setTolerance(tolerance_iteration);
        cg.setMaxIterations(maxnum_iteration);
        x_out.setZero();
        x_out = cg.solve(AtB);

        if (cg.info() != Eigen::Success) {
            std::cerr << "  Fitting by " + solver_type + " failed.\n";
            std::cerr << cg.info() << '\n';
            return 1;
        }

    } else if (solver_type_lower == "leastsquaresconjugategradient") {

#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
        Eigen::LeastSquaresConjugateGradient<SpMat> lscg(sp_mat);
        lscg.setTolerance(tolerance_iteration);
        lscg.setMaxIterations(maxnum_iteration);
        x_out.setZero();
        x_out = lscg.solve(sp_bvec);

        if (lscg.info() != Eigen::Success) {
            std::cerr << "  Fitting by " + solver_type + " failed.\n";
            std::cerr << lscg.info() << '\n';
            return 1;
        }

#else
        std::cerr << "The linked Eigen version is too old\n";
        std::cerr << solver_type + " is available as of 3.3.0\n";
        return 1;
#endif

    } else if (solver_type_lower == "bicgstab") {
        SpMat AtA = sp_mat.transpose() * sp_mat;
        Eigen::VectorXd AtB = sp_mat.transpose() * sp_bvec;

        Eigen::BiCGSTAB<SpMat> bicg(AtA);
        bicg.setTolerance(tolerance_iteration);
        bicg.setMaxIterations(maxnum_iteration);
        x_out.setZero();
        x_out = bicg.solve(AtB);

        if (bicg.info() != Eigen::Success) {
            std::cerr << "  Fitting by " + solver_type + " failed.\n";
            std::cerr << bicg.info() << '\n';
            return 1;
        }
    }
    return 0;
}


void solveGQRSparse(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b,
                    const Eigen::SparseMatrix<double> &C, const Eigen::VectorXd &d, Eigen::VectorXd &x,
                    Eigen::VectorXd &lambda, const int verbosity)
{
#ifdef USE_MKL_BACKEND
    constexpr auto ldlt_solver_name = "PardisoLDLT";
#elif defined(USE_ACCEL_BACKEND)
    constexpr auto ldlt_solver_name = "AccelerateLDLT";
#else
    constexpr auto ldlt_solver_name = "SimplicialLDLT";
#endif

    const int N = A.cols();
    const int P = C.rows();

    // Construct ATA and ATb
    Eigen::SparseMatrix<double> ATA = A.transpose() * A;
    Eigen::VectorXd ATb = A.transpose() * b;

    LOG_IF(verbosity, 1, "ATA non-zeros: ", ATA.nonZeros(), ", C non-zeros: ", C.nonZeros(), "\n");

    // KKT matrix K = [ATA  C^T;  C  0] in sparse format
    Eigen::SparseMatrix<double> K(N + P, N + P);
    std::vector<Eigen::Triplet<double>> triplets;
    // 1) ATA part (column-major)
    for (int col = 0; col < ATA.outerSize(); ++col) {
        const int start = ATA.outerIndexPtr()[col];
        const int end = ATA.outerIndexPtr()[col + 1];
        for (int idx = start; idx < end; ++idx) {
            int row = ATA.innerIndexPtr()[idx];
            double value = ATA.valuePtr()[idx];
            triplets.emplace_back(row, col, value);
        }
    }

    // 2) C^T/C parts
    for (int col = 0; col < C.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(C, col); it; ++it) {
            // C^T part
            triplets.emplace_back(it.col(), N + it.row(), it.value());
            // C part
            triplets.emplace_back(N + it.row(), it.col(), it.value());
        }
    }
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();

    LOG_IF(verbosity, 1, "KKT matrix K constructed with size (", N + P, "×", N + P, "), non-zeros: ", K.nonZeros(), "\n");
    // 3) generate [A^T b; d]
    Eigen::VectorXd rhs(N + P);
    rhs.head(N) = ATb;
    rhs.tail(P) = d;

    Eigen::VectorXd sol;
    bool solved = false;

#ifdef USE_ACCEL_BACKEND
    // 1) AccelerateLDLT
    {
        LOG_IF(verbosity, 1, "Use AccelerateLDLT to solve the KKT problem.\n");
        Eigen::AccelerateLDLT<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Symmetric> ldlt(K);
        if (ldlt.info() == Eigen::Success) {
            sol = ldlt.solve(rhs);
            if (ldlt.info() == Eigen::Success) {
                solved = true;
            }
        }
        if (solved) {
            LOG_IF(verbosity, 1, "KKT system solved with AccelerateLDLT.\n");
        } else {
            LOG_ERR_IF(verbosity, 0, "AccelerateLDLT failed to solve the KKT system.\n");
        }
    }
#elif defined(USE_MKL_BACKEND)
    // 1) PardisoLDLT
    {
        LOG_IF(verbosity, 1, "Use PardisoLDLT to solve the KKT problem.\n");
        Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> pldlt;
        pldlt.analyzePattern(K);
        pldlt.factorize(K);
        if (pldlt.info() == Eigen::Success) {
            sol = pldlt.solve(rhs);
            if (pldlt.info() == Eigen::Success) {
                solved = true;
            }
        }
        if (solved) {
            LOG_IF(verbosity, 1, "KKT system solved with PardisoLDLT.\n");
        } else {
            LOG_ERR_IF(verbosity, 0, "PardisoLDLT failed to solve the KKT system.\n");
        }
    }
#endif

    if (!solved) {
        LOG_IF(verbosity,1, "Use Eigen::SparseLU to solve the KKT problem.\n");
        Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(K);
        if (lu.info() == Eigen::Success) {
            sol = lu.solve(rhs);
            if (lu.info() == Eigen::Success) {
                solved = true;
            }
        }
        if (solved) {
            LOG_IF(verbosity, 1, "KKT system solved with SparseLU.\n");
        } else {
            LOG_IF(verbosity, 0, "SparseLU failed to solve the KKT system.\n");
        }
    }

    // // SimplicialLDLT is not stable for indefinite KKT matrices
    // KKT_Solver ldlt(K);
    // if (ldlt.info() == Eigen::Success) {
    //     sol = ldlt.solve(rhs);
    //     if (ldlt.info() == Eigen::Success) {
    //         used_solver = ldlt_solver_name;
    //         solved = true;
    //         if (verbosity > 1) std::cout << "  [solveGQRSparse] solved by " << used_solver << "\n";
    //     }
    // }

    // 2) SparseQR
    if (!solved) {
        LOG_IF(verbosity, 1, "Use Eigen::SparseQR to solve the KKT problem.\n");
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr(K);
        qr.analyzePattern(K);
        qr.factorize(K);
        if (qr.info() == Eigen::Success) {
            sol = qr.solve(rhs);
            if (qr.info() == Eigen::Success) {
                solved = true;
            }
        }
        if (solved) {
            LOG_IF(verbosity, 1, "KKT system solved with SparseQR.\n");
        } else {
            LOG_IF(verbosity, 0, "SparseQR failed to solve the KKT system.\n");
        }
    }

    // 3) BiCGSTAB
    if (!solved) {
        LOG_IF(verbosity, 1, "Use Eigen::BiCGSTAB to solve the KKT problem.\n");
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> bicg(K);
        bicg.setTolerance(1e-6);
        bicg.setMaxIterations(1000);
        sol = bicg.solve(rhs);
        if (bicg.info() == Eigen::Success) {
            solved = true;
        }
        if (solved) {
            LOG_IF(verbosity, 1, "KKT system solved with BiCGSTAB.\n");
        } else {
            LOG_IF(verbosity, 0, "BiCGSTAB failed to solve the KKT system.\n");
        }
    }

    // fill the output vectors x and lambda
    if (solved) {
        x = sol.head(A.cols());
        lambda = sol.tail(C.rows());
    } else {
        LOG_ERR_IF(verbosity, 0, "All solvers failed to solve the KKT system.\n");
    }
}
