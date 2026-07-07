// least_squares.cpp

#include "least_squares.h"
#include <Eigen/Dense> // dense LDLT for the Schur-complement block of the KKT preconditioner
#include <Eigen/Sparse>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <cmath> // for std::pow, std::sqrt
#include <iostream>
#include <limits>
#include <numeric>
#include <unsupported/Eigen/IterativeSolvers> // Eigen::MINRES for the iterative KKT solver
#include <vector>
#include "error.h"
#include "logger.h"
#include "svd.h"
#ifdef USE_MKL_BACKEND
#include <Eigen/PardisoSupport>
#elif defined(USE_ACCEL_BACKEND)
// The Accelerate KKT solve lives in accelerate_solver.cpp; its <Eigen/AccelerateSupport> include
// pulls <Accelerate/Accelerate.h>, whose Fortran BLAS/LAPACK prototypes would clash with the
// hand-rolled ones in blas_wrapper.h / lapack_wrapper.h that this TU includes below. Keeping the
// Accelerate header out of this TU is exactly why that solve is isolated.
#include "accelerate_solver.h"
#endif

#ifdef USE_SUITESPARSE_BACKEND
// SuiteSparse sparse solvers, orthogonal to the MKL/Accelerate KKT backend above and reachable as
// SPARSESOLVER options:
//   - CHOLMOD supernodal Cholesky on the PSD normal matrix A^T A, via Eigen's CholmodSupport wrapper.
//   - SuiteSparseQR (rank-revealing multifrontal QR) on the rectangular A, via SuiteSparse's own C
//     interface. We deliberately avoid Eigen 3.4's Eigen::SPQR wrapper: with SuiteSparse >= 6 it
//     instantiates SuiteSparseQR<Scalar>(...) passing an Eigen::Index ('long') where the templated
//     index type resolves to int64_t ('long long' on macOS arm64), and the two deduce to conflicting
//     types -- a hard compile error. SuiteSparseQR_C_backslash_default() sidesteps it entirely.
// Both headers live under <prefix>/include/suitesparse, on the include path via the linked targets.
#include <Eigen/CholmodSupport>
#include <SuiteSparseQR_C.h>
#include <cstring>
#endif

// lapack_wrapper.h is included unconditionally; its BLAS prototypes self-guard for EIGEN_USE_BLAS.
#include "lapack_wrapper.h"

#ifdef USE_SUITESPARSE_BACKEND
// Solve min_x || A x - b ||_2 (least squares for rectangular A; exact solve for square full-rank A)
// with SuiteSparseQR's C interface. The backslash convenience routine is 64-bit-index only, so the
// matrix is shipped as a CHOLMOD_LONG cholmod_sparse: Eigen stores 32-bit CSC indices, widened here.
// Returns 0 on success, 1 on failure. See the include block above for why Eigen::SPQR is bypassed.
static auto solve_least_squares_spqr(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b,
                                     Eigen::VectorXd &x_out) -> int
{
    Eigen::SparseMatrix<double> Ac = A; // column-major copy we can compress and index into
    Ac.makeCompressed();

    const SuiteSparse_long m = Ac.rows();
    const SuiteSparse_long n = Ac.cols();
    const SuiteSparse_long nnz = Ac.nonZeros();

    // Widen Eigen's 32-bit CSC index arrays to the 64-bit indices the long CHOLMOD interface expects.
    std::vector<SuiteSparse_long> outer(n + 1), inner(nnz);
    for (SuiteSparse_long j = 0; j <= n; ++j) outer[j] = Ac.outerIndexPtr()[j];
    for (SuiteSparse_long k = 0; k < nnz; ++k) inner[k] = Ac.innerIndexPtr()[k];

    cholmod_common cc;
    cholmod_l_start(&cc);

    cholmod_sparse Acs;
    std::memset(&Acs, 0, sizeof(Acs));
    Acs.nrow = m;
    Acs.ncol = n;
    Acs.nzmax = nnz;
    Acs.p = outer.data();
    Acs.i = inner.data();
    Acs.x = const_cast<double *>(Ac.valuePtr());
    Acs.stype = 0; // unsymmetric: the full rectangular matrix is stored
    Acs.itype = CHOLMOD_LONG;
    Acs.xtype = CHOLMOD_REAL;
    Acs.dtype = CHOLMOD_DOUBLE;
    Acs.sorted = 1;
    Acs.packed = 1;

    cholmod_dense Bcd;
    std::memset(&Bcd, 0, sizeof(Bcd));
    Bcd.nrow = m;
    Bcd.ncol = 1;
    Bcd.nzmax = m;
    Bcd.d = m;
    Bcd.x = const_cast<double *>(b.data());
    Bcd.xtype = CHOLMOD_REAL;
    Bcd.dtype = CHOLMOD_DOUBLE;

    cholmod_dense *X = SuiteSparseQR_C_backslash_default(&Acs, &Bcd, &cc);

    int status = 1;
    if (X && cc.status == CHOLMOD_OK) {
        x_out.resize(n);
        const auto *xd = static_cast<const double *>(X->x);
        for (SuiteSparse_long i = 0; i < n; ++i) x_out(i) = xd[i];
        status = 0;
    } else {
        // Leave a correctly sized (zero) solution on failure so the caller's residual computation
        // b - A x stays dimensionally valid; the nonzero return status flags the failure.
        x_out = Eigen::VectorXd::Zero(n);
    }
    if (X) cholmod_l_free_dense(&X, &cc);
    cholmod_l_finish(&cc);
    return status;
}

// Rank-revealing reduction of a HOMOGENEOUS constraint matrix C (rows define C x = 0) via SuiteSparseQR
// -- a sparse, multithreaded replacement for the densifying LAPACK dgeqp3 path. We factorize the TALL
// matrix C^T (N x P) with rank detection and column pivoting: the rank-revealing pivot deflates the
// dependent columns of C^T (= dependent rows of C) to the end of the permutation E, so E[0..rank-1] are
// the independent rows of C. C_red is then those ORIGINAL (sparse) rows of C -- NOT the R factor, which
// is densely filled for these constraints. R (only rank x P here) and Q are discarded.
// Only the homogeneous case is handled (every invariance subset, and the merged matrix when no
// FC2FIX/FC3FIX value is imposed); inhomogeneous d != 0 and any SuiteSparseQR failure fall back to
// dgeqp3. Returns 0 on success, 1 on failure.
static auto get_independent_rows_spqr(const Eigen::SparseMatrix<double> &C, const int verbosity, const double tolerance,
                                      Eigen::SparseMatrix<double> &C_red, int &r) -> int
{
    const SuiteSparse_long P = C.rows();
    const SuiteSparse_long Ncol = C.cols();

    Eigen::SparseMatrix<double> Ct = C.transpose(); // N x P; its columns are the rows of C
    Ct.makeCompressed();
    const SuiteSparse_long nnz = Ct.nonZeros();

    // Widen Eigen's 32-bit CSC indices to the 64-bit indices the long CHOLMOD interface expects
    // (same pattern as solve_least_squares_spqr). C^T has P columns, so outer has P+1 entries.
    std::vector<SuiteSparse_long> outer(P + 1), inner(nnz);
    for (SuiteSparse_long j = 0; j <= P; ++j) outer[j] = Ct.outerIndexPtr()[j];
    for (SuiteSparse_long k = 0; k < nnz; ++k) inner[k] = Ct.innerIndexPtr()[k];

    cholmod_common cc;
    cholmod_l_start(&cc);

    cholmod_sparse A; // A = C^T (N x P)
    std::memset(&A, 0, sizeof(A));
    A.nrow = Ncol;
    A.ncol = P;
    A.nzmax = nnz;
    A.p = outer.data();
    A.i = inner.data();
    A.x = const_cast<double *>(Ct.valuePtr());
    A.stype = 0;
    A.itype = CHOLMOD_LONG;
    A.xtype = CHOLMOD_REAL;
    A.dtype = CHOLMOD_DOUBLE;
    A.sorted = 1;
    A.packed = 1;

    // Map the caller's auto sentinel (tol < 0, i.e. rank_tolerance_auto = -1) to SuiteSparseQR's default
    // tolerance SPQR_DEFAULT_TOL (= -2). Passing -1 literally would be SPQR_NO_TOL (rank detection off).
    // An explicit positive tolerance is passed through. NOTE: SuiteSparseQR thresholds column 2-norms
    // whereas dgeqp3 thresholds R-diagonals, so the numerical rank may differ by a few rows on a
    // borderline (badly-scaled) constraint set.
    const double spqr_tol = (tolerance < 0.0) ? SPQR_DEFAULT_TOL : tolerance;

    cholmod_sparse *R = nullptr;
    SuiteSparse_long *E = nullptr; // size P column permutation of C^T = row permutation of C
    const SuiteSparse_long rank = SuiteSparseQR_C(SPQR_ORDERING_DEFAULT,
                                                  spqr_tol,
                                                  /*econ=*/0,
                                                  /*getCTX=*/0,
                                                  &A,
                                                  /*Bsparse=*/nullptr,
                                                  /*Bdense=*/nullptr,
                                                  /*Zsparse=*/nullptr,
                                                  /*Zdense=*/nullptr,
                                                  &R,
                                                  &E,
                                                  /*H=*/nullptr,
                                                  /*HPinv=*/nullptr,
                                                  /*HTau=*/nullptr,
                                                  &cc);

    int status = 1;
    if (rank >= 0 && cc.status == CHOLMOD_OK) {
        r = static_cast<int>(rank);
        // E[0..rank-1] are the independent columns of C^T = independent rows of C (E == NULL => identity).
        // Build C_red from those ORIGINAL sparse rows of C.
        Eigen::SparseMatrix<double, Eigen::RowMajor> Crow = C;
        std::vector<Eigen::Triplet<double>> tri;
        tri.reserve(static_cast<size_t>(C.nonZeros()));
        for (int k = 0; k < r; ++k) {
            const SuiteSparse_long orig_row = E ? E[k] : k;
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(Crow, orig_row); it; ++it) {
                tri.emplace_back(k, static_cast<int>(it.col()), it.value());
            }
        }
        C_red.resize(r, static_cast<int>(Ncol));
        C_red.setFromTriplets(tri.begin(), tri.end());
        C_red.makeCompressed();
        status = 0;
        LOG_IF(verbosity,
               1,
               "SuiteSparseQR reduction: rank ",
               r,
               " of ",
               static_cast<int>(P),
               " rows, C_red nnz=",
               C_red.nonZeros(),
               ".\n");
    }
    if (R) cholmod_l_free_sparse(&R, &cc);
    if (E) cholmod_l_free(P, sizeof(SuiteSparse_long), E, &cc);
    cholmod_l_finish(&cc);
    return status;
}
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
        double const diag = std::abs(A_data[static_cast<size_t>(k) + static_cast<size_t>(k) * N]);
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
    std::vector<double> C_mat(static_cast<size_t>(P_i) * static_cast<size_t>(N_i));
    for (int row = 0; row < P_i; ++row) {
        for (int col = 0; col < N_i; ++col) {
            // C_mat is stored column-major: index = col * P_i + row
            C_mat[static_cast<size_t>(col) * P_i + row] = cmat[row][col];
        }
    }
    LOG_IF(verbosity, 1, "Copied C into column-major buffer (", P_i, "×", N_i, ").\n");

    // 2) Prepare to perform QR with column pivoting on C^T (size N×P)
    // Build A_qr = C^T in column-major layout
    std::vector<double> A_qr(static_cast<size_t>(N_i) * static_cast<size_t>(P_i));
    for (int i = 0; i < P_i; ++i) {
        for (int j = 0; j < N_i; ++j) {
            // Copy element (row=i, col=j) of C_mat into (row=j, col=i) of A_qr
            A_qr[static_cast<size_t>(j) + static_cast<size_t>(i) * N_i] =
                C_mat[static_cast<size_t>(i) + static_cast<size_t>(j) * P_i];
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
            C_red[static_cast<size_t>(col) * r + ii] = C_mat[static_cast<size_t>(col) * P_i + orig_row];
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

    Eigen::VectorXd dvec = Eigen::VectorXd::Zero(nrows);
    Eigen::VectorXd dvec_red(nrows);
    const auto info =
        get_independent_rows_lapack_sparse(C_sparse_eigen, dvec, verbosity, tolerance, C_red_eigen, dvec_red, r);

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
                                        const int verbosity, const double tolerance, Eigen::SparseMatrix<double> &C_red,
                                        Eigen::VectorXd &d_red, int &r) -> int
{
    const int P = C_sparse.rows();
    const int N = C_sparse.cols();
    int N_i = N, P_i = P;

    if (P == 0 || N == 0) {
        LOG_IF(verbosity, 1, "C_sparse has zero rows or columns.\n");
        C_red.resize(0, N);
        C_red.makeCompressed();
        d_red.resize(0);
        return 0;
    }

    LOG_IF(verbosity, 1, "P = ", P, ", N = ", N, ", nnz = ", C_sparse.nonZeros(), ".\n");

#ifdef USE_SUITESPARSE_BACKEND
    // SuiteSparseQR rank-revealing reduction for HOMOGENEOUS constraints (every invariance subset, and
    // the merged matrix when no FC2FIX/FC3FIX value is imposed): sparse + multithreaded, avoiding the
    // dense P x N densification and dgeqp3 below. Inhomogeneous d (d != 0, from FC2FIX/FC3FIX) and any
    // SuiteSparseQR failure fall through to the dgeqp3 path.
    if (dvec.isZero(0)) {
        if (get_independent_rows_spqr(C_sparse, verbosity, tolerance, C_red, r) == 0) {
            d_red = Eigen::VectorXd::Zero(r);
            return 0;
        }
        LOG_ERR_IF(verbosity, 0, "SuiteSparseQR reduction failed; falling back to dgeqp3.\n");
    }
#endif

    // 1) Copy sparse→dense column-major buffer C_mat (size P×N)
    std::vector<double> C_mat(static_cast<size_t>(P_i) * static_cast<size_t>(N_i), 0.0);
    for (int col = 0; col < N; ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(C_sparse, col); it; ++it) {
            // it.row()==i, it.col()==j
            const int row = it.row();
            C_mat[static_cast<size_t>(col) * P_i + row] = it.value();
        }
    }
    LOG_IF(verbosity, 1, "build dense C_mat (", P_i, "×", N_i, ").\n");

    // 2) Build A_qr = Cᵀ in column-major (size N×P)
    std::vector<double> A_qr(static_cast<size_t>(N_i) * static_cast<size_t>(P_i));
    for (int i = 0; i < P_i; ++i)
        for (int j = 0; j < N_i; ++j)
            A_qr[static_cast<size_t>(j) + static_cast<size_t>(i) * N_i] =
                C_mat[static_cast<size_t>(i) + static_cast<size_t>(j) * P_i];

    // 3) Run QR with column pivoting on A_qr to determine numerical row rank
    std::vector<int> independent_rows;
    auto info_qr = find_independent_rows_dense(P_i, N_i, A_qr.data(), tolerance, r, independent_rows, verbosity);
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
    for (int i = 0; i < r; ++i) d_red[i] = dvec[independent_rows[i]];

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
        f_square += pow2(bvec[i]);
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
            f_residual += pow2(fsum2[i]);
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
    LOG_IF(verbosity, 1, "get_independent_rows returned r = ", r, ".\n");

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
    for (int i = 0; i < m; ++i) b_prime[i] = bvec[i];
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
        for (int i = 0; i < n; ++i) Nproj[i + i * n] += 1.0;
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

// 64-bit-index sparse matrix for the normal matrix A^T A. The default SparseMatrix uses a 32-bit
// StorageIndex, whose cumulative outer index overflows once nnz(A^T A) exceeds ~2.1e9 -- which happens
// for large full-range / no-cutoff fits (e.g. ICONST >= 10 with `*-* None`). The overflow corrupts the
// allocation size and aborts with std::bad_alloc even when memory is free. 64-bit indices avoid it.
using SpMat64 = Eigen::SparseMatrix<double, Eigen::ColMajor, std::int64_t>;

// Form A^T A with 64-bit indices. If A^T A is genuinely too large to allocate (it is N x N and can be
// near-dense for no-cutoff problems), abort with guidance to use SuiteSparseQR, which factorizes the
// rectangular A directly and never forms A^T A. (A^T A is computed here, not the dense factor.)
[[nodiscard]] static auto build_normal_matrix_int64(const Eigen::SparseMatrix<double> &A) -> SpMat64
{
    try {
        SpMat64 A64 = A; // widen the 32-bit sensing-matrix index to 64-bit (A itself fits 32-bit)
        return A64.transpose() * A64;
    } catch (const std::bad_alloc &) {
        ALM_NS::exit("least_squares_eigen_sparse_solver",
                     "Out of memory forming the normal matrix A^T A. For large full-range (no-cutoff) "
                     "fits A^T A is N x N and too large to store; use SPARSESOLVER = SuiteSparseQR, which "
                     "factorizes the sensing matrix directly without forming A^T A.");
    }
}

auto least_squares_eigen_sparse_solver(const Eigen::SparseMatrix<double> &sp_mat, const Eigen::VectorXd &sp_bvec,
                                       Eigen::VectorXd &x_out, const std::string &solver_type,
                                       const double tolerance_iteration, const int maxnum_iteration) -> int
{
    const auto solver_type_lower = boost::algorithm::to_lower_copy(solver_type);

    using SpMat = Eigen::SparseMatrix<double>;
    if (solver_type_lower == "simplicialldlt") {
        SpMat64 AtA = build_normal_matrix_int64(sp_mat);
        Eigen::VectorXd AtB = sp_mat.transpose() * sp_bvec;
        Eigen::SimplicialLDLT<SpMat64> ldlt(AtA);
        x_out = ldlt.solve(AtB);

        if (ldlt.info() != Eigen::Success) {
            std::cerr << "  Fitting by " + solver_type + " failed with the error code " << ldlt.info() << ".\n";
            return 1;
        }

    } else if (solver_type_lower == "sparseqr") {

        Eigen::SparseQR<SpMat, Eigen::COLAMDOrdering<int>> qr(sp_mat);
        x_out = qr.solve(sp_bvec);

        if (qr.info() != Eigen::Success) {
            std::cerr << "  Fitting by " + solver_type + " failed with the error code " << qr.info() << ".\n";
            return 1;
        }

    } else if (solver_type_lower == "conjugategradient") {
        SpMat64 AtA = build_normal_matrix_int64(sp_mat);
        Eigen::VectorXd AtB = sp_mat.transpose() * sp_bvec;

        Eigen::ConjugateGradient<SpMat64> cg(AtA);
        cg.setTolerance(tolerance_iteration);
        cg.setMaxIterations(maxnum_iteration);
        x_out.setZero();
        x_out = cg.solve(AtB);

        if (cg.info() != Eigen::Success) {
            std::cerr << "  Fitting by " + solver_type + " failed with the error code " << cg.info() << ".\n";
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
            std::cerr << "  Fitting by " + solver_type + " failed with the error code " << lscg.info() << ".\n";
            return 1;
        }

#else
        std::cerr << "The linked Eigen version is too old\n";
        std::cerr << solver_type + " is available as of 3.3.0\n";
        return 1;
#endif

    } else if (solver_type_lower == "bicgstab") {
        SpMat64 AtA = build_normal_matrix_int64(sp_mat);
        Eigen::VectorXd AtB = sp_mat.transpose() * sp_bvec;

        Eigen::BiCGSTAB<SpMat64> bicg(AtA);
        bicg.setTolerance(tolerance_iteration);
        bicg.setMaxIterations(maxnum_iteration);
        x_out.setZero();
        x_out = bicg.solve(AtB);

        if (bicg.info() != Eigen::Success) {
            std::cerr << "  Fitting by " + solver_type + " failed with the error code " << bicg.info() << ".\n";
            return 1;
        }
    } else if (solver_type_lower == "suitesparseqr") {
#ifdef USE_SUITESPARSE_BACKEND
        // SuiteSparseQR factorizes the rectangular A directly (no normal equations), so it is the
        // most robust choice for ill-conditioned / rank-deficient sensing matrices, and it is
        // multithreaded -- typically much faster than Eigen's built-in SparseQR on large problems.
        if (solve_least_squares_spqr(sp_mat, sp_bvec, x_out) != 0) {
            std::cerr << "  Fitting by " + solver_type + " failed.\n";
            return 1;
        }
#else
        std::cerr << "  SPARSESOLVER = " + solver_type + " requires building ALM with -DUSE_SUITESPARSE_BACKEND=yes.\n";
        return 1;
#endif

    } else if (solver_type_lower == "cholmod") {
#ifdef USE_SUITESPARSE_BACKEND
        // CHOLMOD supernodal Cholesky on the normal matrix A^T A (symmetric positive (semi)definite).
        // Fast for well-conditioned problems; for a rank-deficient A^T A the factorization fails and
        // info() reports it, in which case SuiteSparseQR is the recommended fallback. Check the
        // factorization BEFORE solving -- solving on a failed factor is undefined. A^T A uses 64-bit
        // indices (and CHOLMOD's long interface) so large no-cutoff fits do not overflow the 32-bit
        // sparse index; for genuinely huge A^T A, build_normal_matrix_int64 aborts pointing to SuiteSparseQR.
        SpMat64 AtA = build_normal_matrix_int64(sp_mat);
        Eigen::VectorXd AtB = sp_mat.transpose() * sp_bvec;
        Eigen::CholmodSupernodalLLT<SpMat64, Eigen::Lower> chol(AtA);
        if (chol.info() != Eigen::Success) {
            std::cerr << "  Fitting by " + solver_type + " failed (factorization) with the error code " << chol.info()
                      << ".\n";
            return 1;
        }
        x_out = chol.solve(AtB);
        if (chol.info() != Eigen::Success) {
            std::cerr << "  Fitting by " + solver_type + " failed (solve) with the error code " << chol.info() << ".\n";
            return 1;
        }
#else
        std::cerr << "  SPARSESOLVER = " + solver_type + " requires building ALM with -DUSE_SUITESPARSE_BACKEND=yes.\n";
        return 1;
#endif
    } else {
        // No matching solver branch -- e.g. SPARSESOLVER = MINRES, which is only implemented for the
        // numerically constrained KKT path (solveGQRSparse), not this unconstrained / algebraically
        // constrained path. Abort: merely returning would leave x_out unset and the caller would then
        // form sp_mat * x_out with a zero-length vector.
        ALM_NS::exit("least_squares_eigen_sparse_solver",
                     "SPARSESOLVER is not supported for the unconstrained / algebraically constrained "
                     "sparse fit: ",
                     solver_type.c_str());
    }
    return 0;
}


// Block-diagonal SPD preconditioner for the symmetric-indefinite KKT system
//     K = [ H   C^T ;  C   0 ],     H = A^T A  (N x N, rank-deficient by the gauge modes),
// for use with MINRES. P = diag(G, S) with
//     G = diag(H) + sigma          (a positive diagonal -- a Jacobi approximation of H)
//     S = C G^{-1} C^T             (dense, SPD; the preconditioner's exact Schur complement for that G).
// By Murphy-Golub-Wathen this clusters the spectrum of P^{-1}K, so preconditioned MINRES converges in
// far fewer iterations than the (stalling) unpreconditioned run. A *diagonal* G is used deliberately:
// G^{-1} is then trivial, so the setup needs only sparse products to form S -- no N x P solves, which
// for P ~ 3000 dense rotational constraints would dominate the runtime. sigma keeps G strictly
// positive; it affects only the preconditioner's quality (iteration count), never the final solution.
class KKTBlockDiagPreconditioner
{
public:
    KKTBlockDiagPreconditioner() : m_ready(false), m_N(0), m_P(0)
    {}

    // Eigen's IterativeSolverBase calls analyzePattern/factorize/compute on the system matrix K, but
    // this preconditioner is configured from H and C separately (via setup()); make them no-ops that
    // preserve an existing setup. Call setup() BEFORE minres.compute(K).
    template <typename M>
    KKTBlockDiagPreconditioner &analyzePattern(const M &)
    {
        return *this;
    }
    template <typename M>
    KKTBlockDiagPreconditioner &factorize(const M &)
    {
        return *this;
    }
    template <typename M>
    KKTBlockDiagPreconditioner &compute(const M &)
    {
        return *this;
    }

    void setup(const Eigen::SparseMatrix<double> &H, const Eigen::SparseMatrix<double> &C, double sigma, int verbosity)
    {
        using SpMat = Eigen::SparseMatrix<double>;
        m_N = static_cast<int>(H.rows());
        m_P = static_cast<int>(C.rows());

        // G = diag(H) + sigma  (strictly positive); store its inverse for O(1) application.
        m_Dinv.resize(m_N);
        for (int i = 0; i < m_N; ++i) m_Dinv(i) = 1.0 / (H.coeff(i, i) + sigma);

        // S = C G^{-1} C^T = (C * Dinv) * C^T  -- sparse products only, no solves.
        const SpMat Cs = C * m_Dinv.asDiagonal();                // P x N (scaled columns)
        Eigen::MatrixXd S = Eigen::MatrixXd(Cs * C.transpose()); // P x P dense

        // S can be extremely ill-conditioned -- even when C is full row rank -- because diag(A^T A)
        // has near-zero entries (weakly-sampled parameters) that make Dinv, and hence S, span a huge
        // dynamic range. (This is NOT evidence that C is rank-deficient: C is full row rank after the
        // merge + rank-reduction in Constraint::update_constraint_matrix.) MINRES requires a positive-
        // definite preconditioner, so add a relative ridge to S and grow it until the dense factor is
        // SPD. The ridge perturbs only the preconditioner -- never the system matrix K, so it does not
        // bias the solution; it only affects how fast MINRES converges.
        const double s_scale = S.diagonal().cwiseAbs().mean() + sigma;
        double ridge = 1.0e-8 * s_scale;
        m_ready = false;
        for (int attempt = 0; attempt < 12 && !m_ready; ++attempt, ridge *= 10.0) {
            Eigen::MatrixXd Sr = S;
            Sr.diagonal().array() += ridge;
            m_S_ldlt.compute(Sr);
            // LDLT can factor an indefinite matrix and still report Success, so require isPositive():
            // MINRES needs an SPD preconditioner.
            m_ready = (m_S_ldlt.info() == Eigen::Success) && m_S_ldlt.isPositive();
        }
        m_sigma = sigma;

        LOG_IF(verbosity,
               1,
               "KKT preconditioner: G = diag(A^T A) + ",
               sigma,
               ", dense Schur complement S is ",
               m_P,
               "x",
               m_P,
               " (ridge ",
               ridge / 10.0,
               ")",
               m_ready ? " factorized (SPD).\n" : " -- FAILED to make S SPD.\n");
    }

    // z = P^{-1} b = [ G^{-1} b_1 ;  S^{-1} b_2 ] = [ Dinv .* b_1 ;  S^{-1} b_2 ]
    template <typename Rhs>
    Eigen::VectorXd solve(const Rhs &b) const
    {
        Eigen::VectorXd z(m_N + m_P);
        z.head(m_N) = b.head(m_N).cwiseProduct(m_Dinv);
        z.tail(m_P) = m_S_ldlt.solve(b.tail(m_P));
        return z;
    }

    Eigen::ComputationInfo info() const
    {
        return m_ready ? Eigen::Success : Eigen::NumericalIssue;
    }

private:
    bool m_ready;
    int m_N, m_P;
    double m_sigma = 0.0;
    Eigen::VectorXd m_Dinv; // (diag(H) + sigma)^{-1}
    Eigen::LDLT<Eigen::MatrixXd> m_S_ldlt;
};


void solveGQRSparse(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b,
                    const Eigen::SparseMatrix<double> &C, const Eigen::VectorXd &d, Eigen::VectorXd &x,
                    Eigen::VectorXd &lambda, const int verbosity, const std::string &solver_type,
                    const double tolerance_iteration, const int maxnum_iteration)
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

    LOG_IF(verbosity,
           1,
           "KKT matrix K constructed with size (",
           N + P,
           "×",
           N + P,
           "), non-zeros: ",
           K.nonZeros(),
           "\n");
    // 3) generate [A^T b; d]
    Eigen::VectorXd rhs(N + P);
    rhs.head(N) = ATb;
    rhs.tail(P) = d;

    Eigen::VectorXd sol;
    bool solved = false;

    // Iterative KKT solve (selected via SPARSESOLVER). The KKT matrix K is sparse (it never forms the
    // dense factor that a direct method would), so this is the memory-feasible path when the constraint
    // matrix C has dense rows -- e.g. the rotational-invariance constraints of ICONST = 2, 3 -- for
    // which the direct factorizations below exhaust memory. K is symmetric *indefinite*, so MINRES is
    // the appropriate Krylov method, accelerated by the block-diagonal SPD preconditioner
    // diag(diag(A^T A) + sigma, C G^{-1} C^T) (KKTBlockDiagPreconditioner above). An unpreconditioned /
    // identity-preconditioned MINRES stalls completely on this ill-conditioned saddle-point system.
    const auto solver_lower = boost::algorithm::to_lower_copy(solver_type);
    if (solver_lower == "minres") {
        LOG_IF(verbosity, 1, "Use MINRES (iterative) to solve the KKT problem.\n");

        // sigma keeps the preconditioner block G = diag(A^T A) + sigma strictly positive (not the final
        // solution); scale it to the mean diagonal of A^T A so it is dimensionless w.r.t. the problem.
        double mean_diag = 0.0;
        for (int i = 0; i < N; ++i) mean_diag += ATA.coeff(i, i);
        mean_diag = (N > 0) ? mean_diag / N : 1.0;
        const double sigma = 1.0e-6 * (mean_diag > 0.0 ? mean_diag : 1.0);

        Eigen::MINRES<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper, KKTBlockDiagPreconditioner> minres;
        minres.preconditioner().setup(ATA, C, sigma, verbosity); // configure BEFORE compute(K)
        if (minres.preconditioner().info() != Eigen::Success) {
            ALM_NS::exit("solveGQRSparse",
                         "Failed to build the MINRES KKT preconditioner (could not make the Schur "
                         "complement positive definite).");
        }
        minres.compute(K); // sets the matrix; the preconditioner was already configured above

        // MINRES monitors the *preconditioned* residual, which with this block-diagonal preconditioner
        // can be far smaller than the TRUE residual -- so a single solve at the user's CONV_TOL may stop
        // before the constraints (sum rules) are actually satisfied. Re-solve with a progressively
        // tighter stopping tolerance until the TRUE relative KKT residual meets the acceptance bar; the
        // preconditioner is cheap and each solve takes only a few iterations. Acceptance is judged on the
        // true residual (and the explicit constraint residual ||C x - d||), never on minres.info() alone,
        // so an under-converged x that would violate the sum rules is never returned.
        const double rhs_norm = (rhs.norm() > 0.0) ? rhs.norm() : 1.0;
        const double accept_tol = std::max(tolerance_iteration, 1.0e-10);
        double kkt_rel_res = std::numeric_limits<double>::infinity();
        double cons_abs_res = 0.0; // ||C x - d||: the invariance constraints are homogeneous (d = 0),
                                   // so the absolute norm is the meaningful measure of sum-rule violation
        double mtol = accept_tol;
        for (int attempt = 0; attempt < 8 && !solved; ++attempt, mtol *= 1.0e-3) {
            minres.setTolerance(mtol);
            minres.setMaxIterations(maxnum_iteration);
            sol = minres.solve(rhs);
            kkt_rel_res = (K * sol - rhs).norm() / rhs_norm;
            cons_abs_res = (P > 0) ? (C * sol.head(N) - d).norm() : 0.0;
            LOG_IF(verbosity,
                   1,
                   "MINRES attempt ",
                   attempt + 1,
                   ": tol=",
                   mtol,
                   ", ",
                   minres.iterations(),
                   " iters, true relative KKT residual ",
                   kkt_rel_res,
                   ", ||C x - d|| ",
                   cons_abs_res,
                   "\n");
            if (std::isfinite(kkt_rel_res) && kkt_rel_res <= accept_tol) solved = true;
        }

        if (solved) {
            LOG_IF(verbosity,
                   1,
                   "KKT system solved with MINRES (true relative KKT residual ",
                   kkt_rel_res,
                   ", ||C x - d|| ",
                   cons_abs_res,
                   ").\n");
        } else {
            ALM_NS::exit("solveGQRSparse",
                         "MINRES did not reach the requested accuracy on the KKT system; the solution "
                         "would not satisfy the sum rules. Increase MAXITER or use a direct KKT solver "
                         "(an LDLT backend).");
        }
    }

#ifdef USE_ACCEL_BACKEND
    if (!solved)
    // 1) AccelerateLDLT (implemented in accelerate_solver.cpp; see the include note at the top of this
    //    file for why the Accelerate headers are kept out of this TU).
    {
        LOG_IF(verbosity, 1, "Use AccelerateLDLT to solve the KKT problem.\n");
        solved = solve_kkt_accelerate_ldlt(K, rhs, sol);
        if (solved) {
            LOG_IF(verbosity, 1, "KKT system solved with AccelerateLDLT.\n");
        } else {
            LOG_ERR_IF(verbosity, 0, "AccelerateLDLT failed to solve the KKT system.\n");
        }
    }
#elif defined(USE_MKL_BACKEND)
    if (!solved)
    // 1) PardisoLDLT
    {
        LOG_IF(verbosity, 1, "Use PardisoLDLT to solve the KKT problem.\n");
        Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> pldlt;
        pldlt.pardisoParameterArray()[9] = 8;
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

#ifdef USE_SUITESPARSE_BACKEND
    // Preferred direct solver when ALM is built with SuiteSparse: SuiteSparseQR is multithreaded and
    // rank-revealing, so on a well-conditioned KKT it is far faster than Eigen's serial SparseLU /
    // SparseQR. It is tried before SparseLU (after any symmetric-indefinite LDLT backend). (Earlier
    // this was a post-SparseLU fallback to avoid QR fill-in blowing up memory when the constraint
    // matrix was rank-deficient with dense rows; that pathology came from an unreduced constraint
    // matrix and is prevented upstream by the rank-revealing reduction in Constraint.)
    if (!solved) {
        LOG_IF(verbosity, 1, "Use SuiteSparseQR to solve the KKT problem.\n");
        if (solve_least_squares_spqr(K, rhs, sol) == 0) {
            solved = true;
        }
        if (solved) {
            LOG_IF(verbosity, 1, "KKT system solved with SuiteSparseQR.\n");
        } else {
            LOG_ERR_IF(verbosity, 0, "SuiteSparseQR failed to solve the KKT system.\n");
        }
    }
#endif

    if (!solved) {
        LOG_IF(verbosity, 1, "Use Eigen::SparseLU to solve the KKT problem.\n");
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

    // Sparse QR fallback (Eigen's serial SparseQR) only when SuiteSparse is NOT built -- with
    // SuiteSparse, the multithreaded SuiteSparseQR was already tried above (before SparseLU). QR is the
    // most robust direct method on the symmetric-indefinite KKT but has the heaviest fill-in.
#ifndef USE_SUITESPARSE_BACKEND
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
#endif

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
