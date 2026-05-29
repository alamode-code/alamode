//
// Created by Terumasa Tadano on 25/06/03.
//

#pragma once

#include <Eigen/Core>


auto compute_svd_thin(const Eigen::MatrixXd &A,
                      const bool use_eigen) -> std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd>;


auto lapack_pseudoinverse(const int m, const int n, double *A, int &rank, double *A_pinv,
                          const double tol_factor = -1.0) -> int;

auto eigen_pseudoinverse(const Eigen::MatrixXd &A, int &rank, const double tol_factor = -1.0) -> Eigen::MatrixXd;
