//
// blas_wrapper.h
//
// Hand-rolled Fortran BLAS prototypes shared by the alm and anphon modules (dgemv_ / dgemm_ /
// zgemm_), plus a small typed wrapper (zgemm_cpx) for the complex GEMM.
//
// The raw prototypes are declared ONLY when the project must supply them itself, i.e.:
//   * NOT when Eigen's BLAS backend is active (EIGEN_USE_BLAS, set by the openblas / Accelerate
//     dense backends) -- Eigen then declares the same symbols from <Eigen/src/misc/blas.h>, and a
//     second, conflicting declaration here would be a hard C++ error; and
//   * NOT when MKL's headers are used (USE_MKL_BACKEND pulls in <mkl_blas.h>, which provides them).
//
// Centralizing them here (instead of inline in lapack_wrapper.h / anphon headers) is what lets an
// EIGEN_USE_BLAS build compile at all. LAPACK prototypes are intentionally NOT here: Eigen's LAPACKE
// backend uses LAPACKE_* C names, so the project's Fortran d*_/z*_ LAPACK names never collide and
// stay in their own headers.
//
// Complex GEMM follows the BLAS double-pair convention: Eigen's blas.h (and MKL) declare zgemm_ with
// double* arguments, since std::complex<double> is layout-compatible with double[2]. Call it through
// zgemm_cpx(), which accepts std::complex<double>* and reinterpret_casts, so call sites are identical
// across backends.
//
#pragma once

#include <complex>

#if !defined(EIGEN_USE_BLAS) && !defined(USE_MKL_BACKEND)

#ifdef __cplusplus
extern "C"
{
#endif

// DGEMV: y := alpha*A*x + beta*y   (A column-major)
void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta,
            double *y, int *incy);

// DGEMM: C := alpha*A*B + beta*C   (all column-major)
void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b,
            int *ldb, double *beta, double *c, int *ldc);

// ZGEMM: complex C := alpha*A*B + beta*C   (column-major; complex args as double* per BLAS convention)
void zgemm_(const char *transa, const char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda,
            double *b, int *ldb, double *beta, double *c, int *ldc);

#ifdef __cplusplus
}
#endif

#endif // !defined(EIGEN_USE_BLAS) && !defined(USE_MKL_BACKEND)

// Typed wrapper for complex GEMM. Forwards std::complex<double>* arguments to the Fortran zgemm_
// declared either above (generic) or by Eigen's blas.h / MKL when those backends are active -- all
// of which take the double-pair convention. reinterpret_cast is well-defined: std::complex<double>
// is guaranteed layout-compatible with double[2].
//
// NOTE: under EIGEN_USE_BLAS the raw zgemm_ above is suppressed and comes from Eigen's blas.h, so
// this header must be included AFTER the Eigen headers in that build (both alm, via lapack_wrapper.h,
// and anphon satisfy this).
inline void zgemm_cpx(const char *transa, const char *transb, int *m, int *n, int *k,
                      std::complex<double> *alpha, std::complex<double> *a, int *lda,
                      std::complex<double> *b, int *ldb, std::complex<double> *beta,
                      std::complex<double> *c, int *ldc)
{
    zgemm_(transa, transb, m, n, k, reinterpret_cast<double *>(alpha), reinterpret_cast<double *>(a), lda,
           reinterpret_cast<double *>(b), ldb, reinterpret_cast<double *>(beta), reinterpret_cast<double *>(c), ldc);
}
