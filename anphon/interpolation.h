/*
 interpolation.h

 Copyright (c) 2021 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <cmath>
#include <iomanip>
#include <vector>
#include "ndarray.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "memory.h"

namespace PHON_NS
{
class TriLinearInterpolator
{
public:
    TriLinearInterpolator() = default;

    TriLinearInterpolator(const unsigned int ngrid_coarse_in[3], const unsigned int ngrid_dense_in[3]);

    void setup();


    template <typename T>
    void interpolate(const T *val_c, T *val_f, const bool regular_grid = true)
    {

        T v_cubes[8];
        int corner_index[8];
        NDArray<double, 2> corner_coord;

        corner_coord.resize(8, 3);

        for (auto i = 0; i < ngrid_f; ++i) {

            if (regular_grid) {

                get_corners_regular(xf[i], corner_index, corner_coord);
            }

            for (auto j = 0; j < 8; ++j) {
                v_cubes[j] = val_c[corner_index[j]];
            }

            val_f[i] = TriLinearInterpolation(xf[i], corner_coord, v_cubes);
        }

    }

    template <typename T>
    void interpolate_avoidgamma(const T *val_c, T *val_f, const unsigned is, const bool regular_grid = true)
    {
        T v_cubes[8];
        bool contain_gamma;
        int corner_index[8];
        NDArray<double, 2> corner_coord;

        double limit = -100; // only for at acoustic branch at gamma

        corner_coord.resize(8, 3);

        const double sign[2] = {-1.0, 1.0};

        for (auto i = 0; i < ngrid_f; ++i) {

            if (i == 0 && is < 3) {
                val_f[i] = static_cast<T>(limit);
                continue;
            }

            contain_gamma = false;

            if (regular_grid) get_corners_regular(xf[i], corner_index, corner_coord);

            for (auto j = 0; j < 8; ++j) {
                if (corner_index[j] == 0) {
                    contain_gamma = true;
                    break;
                }
            }

            if (contain_gamma && is < 3) {

                // find the closest corner
                double closest[3];
                double dist = 1e10;
                for (auto j = 0; j < 8; ++j) {
                    if (corner_index[j] == 0) continue;

                    double tmp = 0.0;
                    for (auto k = 0; k < 3; ++k) {
                        tmp += std::pow(xf[i][k] - corner_coord[j][k], 2);
                    }
                    if (tmp < dist) {
                        dist = tmp;
                        for (auto k = 0; k < 3; ++k)
                            closest[k] = corner_coord[j][k];
                    }
                }

                T val_sum{};
                int counter = 0;

                int neigh_corner_index[8];
                NDArray<double, 2> neigh_corner_coord;
                neigh_corner_coord.resize(8, 3);

                for (auto tmpi = 0; tmpi < 2; ++tmpi) {
                    for (auto tmpj = 0; tmpj < 2; ++tmpj) {
                        for (auto tmpk = 0; tmpk < 2; ++tmpk) {
                            double shifted_center[3];
                            shifted_center[0] = closest[0] + sign[tmpi] * (xf[i][0] - closest[0]);
                            shifted_center[1] = closest[1] + sign[tmpj] * (xf[i][1] - closest[1]);
                            shifted_center[2] = closest[2] + sign[tmpk] * (xf[i][2] - closest[2]);

                            if (regular_grid)
                                get_corners_regular(shifted_center, neigh_corner_index, neigh_corner_coord);

                            bool still_contain_gamma = false;
                            for (auto j = 0; j < 8; ++j) {
                                if (neigh_corner_index[j] == 0) {
                                    still_contain_gamma = true;
                                    break;
                                }
                            }

                            if (!still_contain_gamma) {

                                for (auto j = 0; j < 8; ++j) {
                                    v_cubes[j] = val_c[neigh_corner_index[j]];
                                }
                                val_sum += TriLinearInterpolation(xf[i], neigh_corner_coord, v_cubes);
                                counter += 1;
                            }

                        } // tmpk
                    } // tmpj
                } // tmpi

                val_f[i] = val_sum / static_cast<T>(counter);

            } else {

                for (auto j = 0; j < 8; ++j) {
                    v_cubes[j] = val_c[corner_index[j]];
                }

                val_f[i] = TriLinearInterpolation(xf[i], corner_coord, v_cubes);
            }
        }
    }


    ~TriLinearInterpolator();

private:
    unsigned int grid_c[3]{};
    unsigned int grid_f[3]{};
    unsigned int ngrid_f, ngrid_c;
    NDArray<double, 2> xf; // coordinate of fine grid
    NDArray<double, 2> xc; // coordinate of coarse grid


    static void set_grid(const unsigned int ngrid_in[3], double **x_out);


    void get_corners_regular(double *xk_i, int *corner_index, double **corner_coord);


    template <typename T>
    T TriLinearInterpolation(double *center, double **corners_coord, const T *val_corner)
    {
        T tx = static_cast<T>(center[0] - corners_coord[0][0]) * static_cast<T>(grid_c[0]);
        T ty = static_cast<T>(center[1] - corners_coord[0][1]) * static_cast<T>(grid_c[1]);
        T tz = static_cast<T>(center[2] - corners_coord[0][2]) * static_cast<T>(grid_c[2]);

        const auto c0 = BiLinearInterpolation(tx, ty, val_corner[0], val_corner[1], val_corner[2], val_corner[3]);
        const auto c1 = BiLinearInterpolation(tx, ty, val_corner[4], val_corner[5], val_corner[6], val_corner[7]);
        return (c1 - c0) * tz + c0;
    }


    template <typename T>
    T BiLinearInterpolation(const T tx, const T ty, const T c00, const T c10, const T c01, const T c11)
    {
        return c00 + (c10 - c00) * tx + (c01 - c00) * ty + (c11 - c01 - c10 + c00) * tx * ty;
    }
};

class FourierInterpolator
{
public:
    FourierInterpolator() = default;
    ~FourierInterpolator() = default;

    FourierInterpolator(const KpointMeshUniform &kmesh_coarse_in, const KpointMeshUniform &kmesh_dense_in,
                        const bool subtract_harmonic_term_in = true);

private:
    // Borrowed views of the caller-owned meshes (copying KpointMeshUniform
    // is deleted -- it owns raw arrays and a shallow copy would double-free).
    const KpointMeshUniform &kmesh_coarse;
    const KpointMeshUniform &kmesh_dense;
    std::vector<int> map_corase_to_dense;
    bool subtract_harmonic_term = true;

    static void get_map_coarse_to_dense(const KpointMeshUniform &kmesh_coarse, const KpointMeshUniform &kmesh_dense,
                                        std::vector<int> &kmap);
};

} // namespace PHON_NS
