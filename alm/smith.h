//
// Created by Terumasa Tadano on 2023/02/09.
//

#ifndef ALAMODE_SMITH_H
#define ALAMODE_SMITH_H


#endif //ALAMODE_SMITH_H

#include <Eigen/Core>

[[nodiscard]] auto gcd(const int &a, const int &b) -> int;

[[nodiscard]] auto exgcd(const int &a, const int &b, int &x, int &y) -> int;

[[nodiscard]] auto is_lone(const Eigen::MatrixXi &A, const int s) -> bool;

[[nodiscard]] auto locate_minval_lower_right(const Eigen::MatrixXi &A, const int s, int &irow, int &icol) -> int;

[[nodiscard]] auto check_divide_subelements(const Eigen::MatrixXi &A, const int s, int &irow, int &icol) -> bool;

auto swap_rows(Eigen::MatrixXi &A, const int irow, const int jrow) -> void;

auto swap_cols(Eigen::MatrixXi &A, const int icol, const int jcol) -> void;

auto add_row_wise(Eigen::MatrixXi &A, const int irow, const int jrow, const int factor) -> void;

auto add_col_wise(Eigen::MatrixXi &A, const int icol, const int jcol, const int factor) -> void;

auto smith_decomposition(const Eigen::MatrixXi &A, Eigen::MatrixXi &D, Eigen::MatrixXi &U, Eigen::MatrixXi &V) -> void;
