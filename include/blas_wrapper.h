//
// blas_wrapper.h
//
// Hand-rolled Fortran BLAS prototypes shared by the alm and anphon modules (dgemv_ / dgemm_ /
// zgemm_), plus a small typed wrapper (zgemm_cpx) for the complex GEMM.
//
// Which prototypes this header supplies depends on the active dense backend:
//
//   * EIGEN_USE_BLAS (openblas / Accelerate dense backends): Eigen declares dgemv_ / dgemm_ / zgemm_
//     itself from <Eigen/src/misc/blas.h> (double-pair convention for the complex routine), so this
//     header declares NONE of them -- a second, conflicting declaration would be a hard C++ error.
//     This header must therefore be included AFTER the Eigen headers in that build.
//
//   * USE_MKL_BACKEND: <mkl_blas.h> exposes the real GEMMs as dgemm / dgemv (no trailing underscore)
//     and lapack_wrapper.h aliases dgemm_ -> dgemm / dgemv_ -> dgemv, so this header does NOT declare
//     dgemv_ / dgemm_. MKL does NOT, however, expose a zgemm_ (only zgemm), so we still declare the
//     generic Fortran zgemm_ here; it binds to MKL's Fortran zgemm_ library symbol at link time
//     (standard double-pair ABI). This also keeps zgemm_cpx below compilable in the MKL build even
//     though alm never calls it.
//
//   * otherwise (generic system BLAS/LAPACK): this header declares all three.
//
// Complex GEMM follows the BLAS double-pair convention: the zgemm_ seen here (ours, or Eigen's
// blas.h) takes double* arguments, since std::complex<double> is layout-compatible with double[2].
// Call it through zgemm_cpx(), which accepts std::complex<double>* and reinterpret_casts, so call
// sites are identical across backends.
//
// LAPACK prototypes are intentionally NOT here: Eigen's LAPACKE backend uses LAPACKE_* C names, so
// the project's Fortran d*_/z*_ LAPACK names never collide and stay in their own headers.
//
#pragma once

#include <complex>

// dgemv_ / dgemm_: declared only for the generic backend. EIGEN_USE_BLAS gets them from Eigen's
// blas.h; USE_MKL_BACKEND gets dgemm / dgemv from <mkl_blas.h>, with the dgemm_/dgemv_ aliases set
// up in lapack_wrapper.h.
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

#ifdef __cplusplus
}
#endif

#endif // !defined(EIGEN_USE_BLAS) && !defined(USE_MKL_BACKEND)

// zgemm_: declared for every backend EXCEPT EIGEN_USE_BLAS (where Eigen's blas.h declares it).
// MKL exposes the complex GEMM as zgemm (no trailing underscore) and does not provide zgemm_, so we
// declare the generic Fortran zgemm_ here too; it resolves to MKL's Fortran zgemm_ symbol at link
// time. Complex args are passed as double* per the BLAS double-pair convention.
#if !defined(EIGEN_USE_BLAS)

#ifdef __cplusplus
extern "C"
{
#endif

// ZGEMM: complex C := alpha*A*B + beta*C   (column-major; complex args as double* per BLAS convention)
void zgemm_(const char *transa, const char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda,
            double *b, int *ldb, double *beta, double *c, int *ldc);

#ifdef __cplusplus
}
#endif

#endif // !defined(EIGEN_USE_BLAS)

// Typed wrapper for complex GEMM. Forwards std::complex<double>* arguments to the Fortran zgemm_
// declared either above (generic / MKL) or by Eigen's blas.h when EIGEN_USE_BLAS is active -- all of
// which take the double-pair convention. reinterpret_cast is well-defined: std::complex<double> is
// guaranteed layout-compatible with double[2].
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
