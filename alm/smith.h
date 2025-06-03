//
// Created by Terumasa Tadano on 2023/02/09.
//

#ifndef ALAMODE_SMITH_H
#define ALAMODE_SMITH_H


#endif //ALAMODE_SMITH_H

#include <Eigen/Core>

[[nodiscard]] int gcd(const int &a, const int &b);

[[nodiscard]] int exgcd(const int &a, const int &b, int &x, int &y);

[[nodiscard]] bool is_lone(const Eigen::MatrixXi &A, const int s);

[[nodiscard]] int locate_minval_lower_right(const Eigen::MatrixXi &A, const int s, int &irow, int &icol);

[[nodiscard]] bool check_divide_subelements(const Eigen::MatrixXi &A, const int s, int &irow, int &icol);

void swap_rows(Eigen::MatrixXi &A, const int irow, const int jrow);

void swap_cols(Eigen::MatrixXi &A, const int icol, const int jcol);

void add_row_wise(Eigen::MatrixXi &A, const int irow, const int jrow, const int factor);

void add_col_wise(Eigen::MatrixXi &A, const int icol, const int jcol, const int factor);

void smith_decomposition(const Eigen::MatrixXi &A,
                         Eigen::MatrixXi &D,
                         Eigen::MatrixXi &U,
                         Eigen::MatrixXi &V);
