/*
 dense_hermitian_eigen.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "dense_hermitian_eigen.h"
#include <mpi.h>
#include "error.h"
#include "ndarray.h"

extern "C"
{
    void zheev_(const char *jobz, const char *uplo, int *n, std::complex<double> *a, int *lda, double *w,
                std::complex<double> *work, int *lwork, double *rwork, int *info);
}

using namespace PHON_NS;

void PHON_NS::solve_dense_hermitian(int n, const std::complex<double> *const *mat_in, double *eval_out,
                                    std::complex<double> **evec_out, bool compute_evec, char uplo)
{
    int INFO;
    int LWORK = (2 * n - 1) * 10;
    NDArray<std::complex<double>, 1> amat(n * n);
    NDArray<std::complex<double>, 1> WORK(LWORK);
    NDArray<double, 1> RWORK(3 * n - 2);

    unsigned int k = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            amat[k++] = mat_in[i][j];
        }
    }

    char JOBZ = compute_evec ? 'V' : 'N';

    zheev_(&JOBZ, &uplo, &n, amat, &n, eval_out, WORK, &LWORK, RWORK, &INFO);
    if (INFO != 0) {
        exit("solve_dense_hermitian", "zheev failed to diagonalize the Hermitian matrix (INFO != 0).");
    }

    if (evec_out) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                evec_out[j][i] = amat[j * n + i];
            }
        }
    }
}
