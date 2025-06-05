// least_squares.cpp

#include "least_squares.h"
#include "lapack_wrapper.h"
#include "svd.h"
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include <cmath>       // for std::pow, std::sqrt
#include <boost/algorithm/string.hpp>

int least_squares_svd(const size_t N,
                      const size_t M,
                      double *amat,
                      const double *bvec,
                      double *param_out,
                      const int verbosity)
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
        std::cout << "  Entering fitting routine: SVD without constraints" << std::endl;
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
        std::cerr << "　　Warning [fit_without_constraints]: "
            << "Matrix is rank-deficient. Force constants could not be determined uniquely :(\n";
    }

    if (nrank == static_cast<int>(N) && verbosity > 0) {
        // Compute residual sum of squares: sum of squares of entries fsum2[N..M-1]
        double f_residual = 0.0;
        for (int i = static_cast<int>(N); i < static_cast<int>(M); ++i) {
            f_residual += std::pow(fsum2[i], 2);
        }
        std::cout << '\n'
            << "  Residual sum of squares for the solution: "
            << std::sqrt(f_residual) << '\n';
        std::cout << "  Fitting error (%) : "
            << std::sqrt(f_residual / f_square) * 100.0 << '\n';
    }

    // Copy solution (first N entries of fsum2) into param_out
    for (size_t i = 0; i < N; ++i) {
        param_out[i] = fsum2[i];
    }

    // Vectors automatically deallocate when going out of scope
    return INFO;
}


int least_squares_with_constraints_gqr(const size_t N,
                                       const size_t M,
                                       const size_t P,
                                       double *amat,
                                       const double *bvec,
                                       double *param_out,
                                       const double *const *cmat,
                                       const double *dvec,
                                       const int verbosity)
{
    // Copy b into a mutable array b_copy, since dgglse_ overwrites it
    std::vector<double> b_copy(M);
    for (size_t i = 0; i < M; ++i) {
        b_copy[i] = bvec[i];
    }
    const double f_square = std::inner_product(b_copy.begin(),
                                               b_copy.end(),
                                               b_copy.begin(),
                                               0.0);

    // Copy C into a contiguous column-major buffer C_mat (P×N)
    std::vector<double> C_mat(P * N);
    for (size_t row = 0; row < P; ++row) {
        for (size_t col = 0; col < N; ++col) {
            C_mat[row + col * P] = cmat[row][col];
        }
    }

    if (verbosity > 1) {
        std::cout << "　　[least_squares_with_constraints_gqr] Starting row-rank determination of C...\n";
    }

    // Step 1: Compute QR with column pivoting on C^T (size N×P)
    int N_i = static_cast<int>(N);
    int P_i = static_cast<int>(P);

    // Build A_qr = C^T in column-major (N×P)
    std::vector<double> A_qr(N * P);
    for (int i = 0; i < P_i; ++i) {
        for (int j = 0; j < N_i; ++j) {
            A_qr[j + i * N_i] = C_mat[i + j * P_i];
        }
    }

    // jpvt for pivot indices, TAU for scalar factors
    std::vector<int> jpvt(P_i, 0);
    int minNP = std::min(N_i, P_i);
    std::vector<double> tau(minNP);

    // Query optimal WORK size
    int lwork_qr = -1;
    double work_qr_query = 0.0;
    int INFO_qr;
    dgeqp3_(&N_i,
            &P_i,
            A_qr.data(),
            &N_i,
            jpvt.data(),
            tau.data(),
            &work_qr_query,
            &lwork_qr,
            &INFO_qr);

    if (INFO_qr != 0) {
        if (verbosity > 1) {
            std::cerr << "　　[least_squares_with_constraints_gqr] dgeqp3_ (workspace query) failed, INFO = "
                << INFO_qr << "\n";
        }
        return INFO_qr;
    }

    int LWORK_qr = static_cast<int>(work_qr_query);
    std::vector<double> WORK_qr(LWORK_qr);

    // Perform QR with column pivoting
    dgeqp3_(&N_i,
            &P_i,
            A_qr.data(),
            &N_i,
            jpvt.data(),
            tau.data(),
            WORK_qr.data(),
            &LWORK_qr,
            &INFO_qr);

    if (INFO_qr != 0) {
        if (verbosity > 1) {
            std::cerr << "　　[least_squares_with_constraints_gqr] dgeqp3_ failed, INFO = "
                << INFO_qr << "\n";
        }
        return INFO_qr;
    }

    // Step 2: Determine numerical rank r of C by inspecting R diagonal in A_qr
    double R00 = std::abs(A_qr[0 + 0 * N_i]);
    double tol = std::max(P_i, N_i) * R00 * std::numeric_limits<double>::epsilon();
    int r = 0;
    for (int k = 0; k < std::min(P_i, N_i); ++k) {
        double diag = std::abs(A_qr[k + k * N_i]);
        if (diag > tol) {
            ++r;
        } else {
            break;
        }
    }

    if (verbosity > 1) {
        std::cout << "　　[least_squares_with_constraints_gqr] C row-rank r = "
            << r << " / " << P_i << "\n";
    }

    // Step 3: Extract indices of the first r pivoted columns of C^T
    std::vector<int> independent_rows(r);
    for (int i = 0; i < r; ++i) {
        independent_rows[i] = jpvt[i] - 1; // Convert 1-based to 0-based
    }

    // Step 4: Build reduced C_red (r×N) and d_red (length r), both column-major
    std::vector<double> C_red(r * N);
    std::vector<double> d_red(r);
    for (int ii = 0; ii < r; ++ii) {
        int orig_row = independent_rows[ii];
        for (int col = 0; col < N_i; ++col) {
            C_red[ii + col * r] = C_mat[orig_row + col * P_i];
        }
        d_red[ii] = dvec[orig_row];
    }

    if (verbosity > 1) {
        std::cout << "　[least_squares_with_constraints_gqr] Calling dgglse_ with reduced C (" << r << "×" << N_i <<
            ")...\n";
    }

    // Step 5: Call dgglse to solve minimize ||A x - b||_2 subject to C_red x = d_red
    int M_i = static_cast<int>(M);
    int r_i = r;
    int LWORK_lse = r_i + std::min(M_i, N_i) + 10 * std::max(M_i, N_i);
    std::vector<double> WORK_lse(LWORK_lse);
    std::vector<double> X(N_i);

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
            &INFO_qr);

    if (verbosity > 1) {
        if (INFO_qr == 0) {
            std::cout << "　[least_squares_with_constraints_gqr] dgglse_ completed successfully.\n";
        } else {
            std::cerr << "　[least_squares_with_constraints_gqr] dgglse_ returned INFO = "
                << INFO_qr << "\n";
        }
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
        std::cout << '\n' << "  Residual sum of squares for the solution: "
            << sqrt(f_residual) << '\n';
        std::cout << "  Fitting error (%) : "
            << std::sqrt(f_residual / f_square) * 100.0 << '\n';
    }

    return INFO_qr;
}

int least_squares_with_constraints_svd(const size_t N,
                                       const size_t M,
                                       const size_t P,
                                       double *amat,              // A: (M×N) column-major, may be overwritten
                                       double *bvec,              // b: (M) column vector, may be overwritten
                                       double *param_out,         // output x (length N)
                                       const double *const *cmat, // C[i][j] pointer array (row i, col j)
                                       const double *dvec_orig,   // d: (P) column vector
                                       const int verbosity
    )
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
        dgemv_(&trans,
               &n,
               &p,
               &alpha,
               C_pinv.data(),
               &n,
               d_copy.data(),
               &inc,
               &beta,
               x0.data(),
               &inc);
    }
    // If rankC == 0, x0 remains all zeros

    // ------------------------------------------------------------
    // 4) Compute residual b' = b − A * x0
    // ------------------------------------------------------------
    std::vector<double> b_prime(m);
    for (int i = 0; i < m; ++i) b_prime[i] = bvec[i];
    const auto f_square = std::inner_product(b_prime.begin(),
                                             b_prime.end(),
                                             b_prime.begin(),
                                             0.0);
    {
        char trans = 'N';
        double alpha = -1.0, beta = 1.0;
        int inc = 1;
        dgemv_(&trans,
               &m,
               &n,
               &alpha,
               amat,
               &m,
               x0.data(),
               &inc,
               &beta,
               b_prime.data(),
               &inc);
    }
    // ------------------------------------------------------------
    // 5) Projector matrix N = I - C^+ * C (nxn)
    // ------------------------------------------------------------
    std::vector<double> Nproj(static_cast<size_t>(n) * n, 0.0);
    {
        char tA = 'N', tB = 'N';
        double alpha = 1.0, beta = 0.0;
        dgemm_(&tA,
               &tB,
               &n,
               &n,
               &p,
               &alpha,
               C_pinv.data(),
               &n,
               C_copy.data(),
               &p,
               &beta,
               Nproj.data(),
               &n);
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
        dgemm_(&tA,
               &tB,
               &m,
               &n,
               &n,
               &alpha,
               amat,
               &m,
               Nproj.data(),
               &n,
               &beta,
               A2.data(),
               &m);
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
        dgemv_(&trans,
               &n,
               &m,
               &alpha,
               A2_pinv.data(),
               &n,
               b_prime.data(),
               &inc,
               &beta,
               delta.data(),
               &inc);
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
        dgemv_(&trans,
               &m,
               &n,
               &alpha,
               amat,
               &m,
               param_out,
               &incx,
               &beta,
               Ax.data(),
               &incy);

        double rss = 0.0;
        for (int i = n - p; i < m; ++i) {
            double ri = Ax[i] - bvec[i];
            rss += ri * ri;
        }

        std::cout << '\n' << "  Residual sum of squares for the solution: "
            << sqrt(rss) << '\n';
        std::cout << "  Fitting error (%) : "
            << std::sqrt(rss / f_square) * 100.0 << '\n';

    }


    return 0;
}

int least_squares_eigen_sparse_solver(const Eigen::SparseMatrix<double> &sp_mat,
                                      const Eigen::VectorXd &sp_bvec,
                                      Eigen::VectorXd &x_out,
                                      const std::string &solver_type,
                                      const double tolerance_iteration,
                                      const int maxnum_iteration)
{
    const auto solver_type_lower = boost::algorithm::to_lower_copy(solver_type);

    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
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
