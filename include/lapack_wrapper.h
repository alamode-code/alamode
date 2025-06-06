//
// Created by Terumasa Tadano on 25/06/04.
//

#pragma once

extern "C"
{
    // LAPACK routines for GQR-based solver
    void dgeqp3_(const int *M, const int *N, double *A, const int *LDA, int *JPVT, double *TAU, double *WORK,
                 const int *LWORK, int *INFO);

    void dgglse_(const int *M, const int *N, const int *P, double *A, const int *LDA, double *B, const int *LDB,
                 double *C, double *D, double *X, double *WORK, const int *LWORK, int *INFO);

    // dlamch_ returns machine parameters; here we use it to get epsilon
    double dlamch_(char *);

    // DGESDD: compute the singular value decomposition of a matrix A (column-major)
    void dgesdd_(char *jobz,   // 'A' to compute all columns of U and rows of VT
                 int *m,       // number of rows of A
                 int *n,       // number of columns of A
                 double *a,    // input/output: the m×n matrix, column-major
                 int *lda,     // leading dimension of A (>= m)
                 double *s,    // output: singular values (length = min(m,n))
                 double *u,    // output: U matrix (m×m if jobz='A'), column-major
                 int *ldu,     // leading dimension of U (>= m)
                 double *vt,   // output: VT matrix (n×n if jobz='A'), column-major
                 int *ldvt,    // leading dimension of VT (>= n)
                 double *work, // workspace array
                 int *lwork,   // length of the WORK array
                 int *iwork,   // integer workspace (size = 8*min(m,n))
                 int *info     // output: error code (0 = success)
    );

    // DGEMV: y := alpha*A*x + beta*y  (A is column-major)
    void dgemv_(char *trans, // 'N' for A*x, 'T' for A^T*x
                int *m,      // number of rows of A
                int *n,      // number of columns of A
                double *alpha,
                double *a, // A: (m×n) column-major
                int *lda,  // leading dimension of A (>= m)
                double *x, // x: length n (if trans='N') or length m (if trans='T')
                int *incx, // increment for elements of x (usually 1)
                double *beta,
                double *y, // y: length m (if trans='N') or length n (if trans='T')
                int *incy  // increment for elements of y (usually 1)
    );

    // DGEMM: C := alpha*A*B + beta*C  (all matrices column-major)
    void dgemm_(char *transa, // 'N' or 'T' for A
                char *transb, // 'N' or 'T' for B
                int *m,       // number of rows of A and C
                int *n,       // number of columns of B and C
                int *k,       // number of columns of A and rows of B
                double *alpha,
                double *a, // A: (m×k) if transa='N', else (k×m), column-major
                int *lda,  // leading dimension of A (>= m if 'N', >= k if 'T')
                double *b, // B: (k×n) if transb='N', else (n×k), column-major
                int *ldb,  // leading dimension of B (>= k if 'N', >= n if 'T')
                double *beta,
                double *c, // C: (m×n), column-major
                int *ldc   // leading dimension of C (>= m)
    );

    // DGELS: solves least‐squares A*X = B or A^T*X = B for over/underdetermined systems
    void dgels_(char *trans,  // 'N' for A*X=B, 'T' for A^T*X=B
                int *m,       // number of rows of A
                int *n,       // number of columns of A
                int *nrhs,    // number of right-hand sides (columns of B)
                double *a,    // A: (m×n), column-major, overwritten
                int *lda,     // leading dimension of A (>= m)
                double *b,    // B: (max(m,n)×nrhs), column-major, overwritten with solution
                int *ldb,     // leading dimension of B (>= max(m,n))
                double *work, // workspace array
                int *lwork,   // length of WORK array
                int *info     // output: error code (0=success)
    );

    void dgesvd_(const char *JOBU, const char *JOBVT, const int *M, const int *N, double *A, const int *LDA, double *S,
                 double *U, const int *LDU, double *VT, const int *LDVT, double *WORK, const int *LWORK, int *INFO);


    // others


    void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);

    void dpotrs_(char *uplo, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info);

    void dgelss_(int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *s, double *rcond,
                 int *rank, double *work, int *lwork, int *info);
}
