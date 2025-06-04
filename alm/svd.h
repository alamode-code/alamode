//
// Created by Terumasa Tadano on 25/06/03.
//

#pragma once

#include <Eigen/Core>


std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd>
compute_svd_thin(const Eigen::MatrixXd &A, const bool use_eigen);


int lapack_pseudoinverse(
    const int m,
    const int n,
    double *A,
    int &rank,
    double *A_pinv,
    const double tol_factor = -1.0
    );

Eigen::MatrixXd eigen_pseudoinverse(const Eigen::MatrixXd &A,
                                    int &rank,
                                    const double tol_factor = -1.0);
