/*
 dense_symmetric_eigen.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "dense_symmetric_eigen.h"
#include <mpi.h>
#include <string>
#include "error.h"

extern "C"
{
    void dsyev_(const char *jobz, const char *uplo, int *n, double *a, int *lda, double *w, double *work,
                int *lwork, int *info);
}

using namespace PHON_NS;

void PHON_NS::solve_dense_symmetric(int n, std::vector<double> &A, std::vector<double> &w, int num_lowest)
{
    // LAPACK backend: full spectrum regardless of num_lowest (see header).
    (void) num_lowest;

    if (n <= 0) return;
    w.resize(n);

    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    dsyev_("V", "L", &n, A.data(), &n, w.data(), &work_query, &lwork, &info);
    if (info != 0) {
        exit("solve_dense_symmetric", "dsyev workspace query failed");
    }
    lwork = static_cast<int>(work_query);
    std::vector<double> work(lwork);
    dsyev_("V", "L", &n, A.data(), &n, w.data(), work.data(), &lwork, &info);
    if (info != 0) {
        exit("solve_dense_symmetric",
             ("dsyev failed with info = " + std::to_string(info)).c_str());
    }
}
