//
// Created by Terumasa Tadano on 25/06/03.
//

#pragma once

#include <Eigen/Core>
#include "constraint.h"


std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd>
compute_svd_thin(const Eigen::MatrixXd &A, const bool use_eigen);


extern "C" void dgesvd_(
    const char *jobu,  // 'A','S','O' or 'N'
    const char *jobvt, // 'A','S','O' or 'N'
    const int *m,      // number of rows of A
    const int *n,      // number of cols of A
    double *A,         // pointer to A (column‐major, size >= m*n)
    const int *lda,    // leading dimension of A (>= max(1,m))
    double *S,         // output singular values (length = min(m,n))
    double *U,         // output U matrix (if requested)
    const int *ldu,    // leading dimension of U
    double *VT,        // output V^T matrix (if requested)
    const int *ldvt,   // leading dimension of VT
    double *work,      // workspace array, length >= lwork
    const int *lwork,  // length of work[]
    int *info          // output info; 0 ⇒ success
    );
