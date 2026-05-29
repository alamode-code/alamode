/*
 mathfunctions.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <cstdlib>
#include <iostream>
#include <vector>

// Square of a scalar: a single multiplication instead of a general power
// call (faster than raising to the power of two), with the argument
// evaluated exactly once. Works for double and std::complex<double>.
template <typename T>
inline auto pow2(const T x) -> T
{
    return x * x;
}

template <typename T>
inline auto matmul3(T ret[3][3], const T amat[3][3], const T bmat[3][3]) -> void
{
    int i, j, k;

    T ret_tmp[3][3];

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            ret_tmp[i][j] = 0.0;
            for (k = 0; k < 3; ++k)
                ret_tmp[i][j] += amat[i][k] * bmat[k][j];
        }
    }

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            ret[i][j] = ret_tmp[i][j];
        }
    }
}

inline auto transpose3(double ret[3][3], const double mat[3][3]) -> void
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ret[i][j] = mat[j][i];
        }
    }
}

template <typename T>
inline auto rotvec(T vec_out[3], const T vec_in[3], const double mat[3][3], char mode = 'N') -> void
{
    // Perform matrix x vector multiplication.
    //
    // vec_out = mat      * vec_in   (mode = 'N')
    //          (mat)^{t} * vec_in   (mode = 'T')
    //

    unsigned int i;
    T vec_tmp[3];

    for (i = 0; i < 3; ++i) {
        vec_tmp[i] = vec_in[i];
    }

    if (mode == 'N') {
        for (i = 0; i < 3; ++i) {
            vec_out[i] = mat[i][0] * vec_tmp[0] + mat[i][1] * vec_tmp[1] + mat[i][2] * vec_tmp[2];
        }
    } else if (mode == 'T') {
        for (i = 0; i < 3; ++i) {
            vec_out[i] = mat[0][i] * vec_tmp[0] + mat[1][i] * vec_tmp[1] + mat[2][i] * vec_tmp[2];
        }
    } else {
        std::cout << "Invalid mode " << mode << std::endl;
        exit(1);
    }
}

inline auto rotvec(double vec_out[3], double vec_in[3], double **mat, char mode = 'N') -> void
{
    // Perform matrix x vector multiplication.
    //
    // vec_out = mat      * vec_in   (mode = 'N')
    //          (mat)^{t} * vec_in   (mode = 'T')
    //

    unsigned int i;
    double vec_tmp[3];

    for (i = 0; i < 3; ++i) {
        vec_tmp[i] = vec_in[i];
    }

    if (mode == 'N') {
        for (i = 0; i < 3; ++i) {
            vec_out[i] = mat[i][0] * vec_tmp[0] + mat[i][1] * vec_tmp[1] + mat[i][2] * vec_tmp[2];
        }
    } else if (mode == 'T') {
        for (i = 0; i < 3; ++i) {
            vec_out[i] = mat[0][i] * vec_tmp[0] + mat[1][i] * vec_tmp[1] + mat[2][i] * vec_tmp[2];
        }
    } else {
        std::cout << "Invalid mode " << mode << std::endl;
        exit(1);
    }
}

inline auto rotvec(double vec_out[3], double vec_in[3], const Eigen::Matrix3d &mat_in, char mode = 'N') -> void
{
    Eigen::Vector3d vec_tmp;

    for (auto i = 0; i < 3; ++i)
        vec_tmp[i] = vec_in[i];

    if (mode == 'N') {
        vec_tmp = mat_in * vec_tmp;
    } else if (mode == 'T') {
        vec_tmp = mat_in.transpose() * vec_tmp;
    } else {
        std::cout << "Invalid mode " << mode << std::endl;
        exit(1);
    }
    for (auto i = 0; i < 3; ++i)
        vec_out[i] = vec_tmp[i];
}

inline auto rotvec(std::complex<double> vec_out[3], std::complex<double> vec_in[3], const Eigen::Matrix3d &mat_in,
                   char mode = 'N') -> void
{
    Eigen::Vector3cd vec_tmp;

    for (auto i = 0; i < 3; ++i)
        vec_tmp[i] = vec_in[i];

    if (mode == 'N') {
        vec_tmp = mat_in * vec_tmp;
    } else if (mode == 'T') {
        vec_tmp = mat_in.transpose() * vec_tmp;
    } else {
        std::cout << "Invalid mode " << mode << std::endl;
        exit(1);
    }
    for (auto i = 0; i < 3; ++i)
        vec_out[i] = vec_tmp[i];
}

inline auto invmat3(double invmat[3][3], const double mat[3][3]) -> void
{
    unsigned int i, j;
    double mat_tmp[3][3];

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            mat_tmp[i][j] = mat[i][j];
        }
    }

    double det = mat_tmp[0][0] * mat_tmp[1][1] * mat_tmp[2][2] + mat_tmp[1][0] * mat_tmp[2][1] * mat_tmp[0][2] +
                 mat_tmp[2][0] * mat_tmp[0][1] * mat_tmp[1][2] - mat_tmp[0][0] * mat_tmp[2][1] * mat_tmp[1][2] -
                 mat_tmp[2][0] * mat_tmp[1][1] * mat_tmp[0][2] - mat_tmp[1][0] * mat_tmp[0][1] * mat_tmp[2][2];

    if (std::abs(det) < 1.0e-12) {
        std::cout << "invmat3: Given matrix is singular" << std::endl;
        exit(1);
    }

    double factor = 1.0 / det;

    invmat[0][0] = (mat_tmp[1][1] * mat_tmp[2][2] - mat_tmp[1][2] * mat_tmp[2][1]) * factor;
    invmat[0][1] = (mat_tmp[0][2] * mat_tmp[2][1] - mat_tmp[0][1] * mat_tmp[2][2]) * factor;
    invmat[0][2] = (mat_tmp[0][1] * mat_tmp[1][2] - mat_tmp[0][2] * mat_tmp[1][1]) * factor;

    invmat[1][0] = (mat_tmp[1][2] * mat_tmp[2][0] - mat_tmp[1][0] * mat_tmp[2][2]) * factor;
    invmat[1][1] = (mat_tmp[0][0] * mat_tmp[2][2] - mat_tmp[0][2] * mat_tmp[2][0]) * factor;
    invmat[1][2] = (mat_tmp[0][2] * mat_tmp[1][0] - mat_tmp[0][0] * mat_tmp[1][2]) * factor;

    invmat[2][0] = (mat_tmp[1][0] * mat_tmp[2][1] - mat_tmp[1][1] * mat_tmp[2][0]) * factor;
    invmat[2][1] = (mat_tmp[0][1] * mat_tmp[2][0] - mat_tmp[0][0] * mat_tmp[2][1]) * factor;
    invmat[2][2] = (mat_tmp[0][0] * mat_tmp[1][1] - mat_tmp[0][1] * mat_tmp[1][0]) * factor;
}

inline auto invmat3_i(int invmat[3][3], int mat[3][3]) -> void
{

    int det = mat[0][0] * mat[1][1] * mat[2][2] + mat[1][0] * mat[2][1] * mat[0][2] +
              mat[2][0] * mat[0][1] * mat[1][2] - mat[0][0] * mat[2][1] * mat[1][2] -
              mat[2][0] * mat[1][1] * mat[0][2] - mat[1][0] * mat[0][1] * mat[2][2];

    if (std::abs(det) == 0) {
        std::cout << "invmat3_i: Given matrix is singular" << std::endl;
        exit(1);
    }

    invmat[0][0] = (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) / det;
    invmat[0][1] = (mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2]) / det;
    invmat[0][2] = (mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1]) / det;

    invmat[1][0] = (mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2]) / det;
    invmat[1][1] = (mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0]) / det;
    invmat[1][2] = (mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2]) / det;

    invmat[2][0] = (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]) / det;
    invmat[2][1] = (mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1]) / det;
    invmat[2][2] = (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]) / det;
}

inline auto nint(double x) -> int
{
    return int(x + 0.5 - (x < 0.0));
}

template <typename T>
auto insort(int n, T *arr) -> void
{
    int i, j;
    T tmp;

    for (i = 1; i < n; ++i) {
        tmp = arr[i];
        for (j = i - 1; j >= 0 && arr[j] > tmp; --j) {
            arr[j + 1] = arr[j];
        }
        arr[j + 1] = tmp;
    }
}

inline auto sort_tail(const int n, int *arr) -> void
{
    int i, m;

    m = n - 1;
    int *ind_tmp;

    ind_tmp = new int[m];

    for (i = 0; i < m; ++i) {
        ind_tmp[i] = arr[i + 1];
    }

    insort(m, ind_tmp);

    for (i = 0; i < m; ++i) {
        arr[i + 1] = ind_tmp[i];
    }
    delete[] ind_tmp;
}


inline auto distance(double *x1, double *x2) -> double
{
    auto dist = pow2(x1[0] - x2[0]) + pow2(x1[1] - x2[1]) + pow2(x1[2] - x2[2]);
    dist = std::sqrt(dist);

    return dist;
}
