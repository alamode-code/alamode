/*
 interpolation.h

 Copyright (c) 2021 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <vector>
#include <cmath>
#include <iomanip>
#include "memory.h"
#include "mathfunctions.h"

namespace PHON_NS
{
class TriLinearInterpolator
{
public:
    TriLinearInterpolator()
    = default;

    TriLinearInterpolator(const unsigned int ngrid_coarse_in[3],
                          const unsigned int ngrid_dense_in[3])
    {
        for (auto i = 0; i < 3; ++i) {
            grid_c[i] = ngrid_coarse_in[i];
            grid_f[i] = ngrid_dense_in[i];
        }
        ngrid_c = grid_c[0] * grid_c[1] * grid_c[2];
        ngrid_f = grid_f[0] * grid_f[1] * grid_f[2];

        if (xf) deallocate(xf);
        if (xc) deallocate(xc);

        allocate(xf, ngrid_f, 3);
        allocate(xc, ngrid_c, 3);
    }


    void setup()
    {
        set_grid(grid_c, xc);
        set_grid(grid_f, xf);
    };


    template <typename T>
    void interpolate(const T *val_c, T *val_f, const bool regular_grid = true)
    {

        T v_cubes[8];
        int corner_index[8];
        double **corner_coord;

        allocate(corner_coord, 8, 3);

        for (auto i = 0; i < ngrid_f; ++i) {

            if (regular_grid) {

                get_corners_regular(xf[i], corner_index, corner_coord);

            }

            for (auto j = 0; j < 8; ++j) {
                v_cubes[j] = val_c[corner_index[j]];
            }

            val_f[i] = TriLinearInterpolation(xf[i], corner_coord, v_cubes);

        }

        deallocate(corner_coord);
    }

    template <typename T>
    void interpolate_avoidgamma(const T *val_c, T *val_f, const unsigned is, const bool regular_grid = true)
    {
        T v_cubes[8];
        bool contain_gamma;
        int corner_index[8];
        double **corner_coord;

        double limit = -100; // only for at acoustic branch at gamma

        allocate(corner_coord, 8, 3);

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
                        for (auto k = 0; k < 3; ++k) closest[k] = corner_coord[j][k];
                    }
                }

                T val_sum{};
                int counter = 0;

                int neigh_corner_index[8];
                double **neigh_corner_coord;
                allocate(neigh_corner_coord, 8, 3);

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
                    }     // tmpj
                }         // tmpi

                val_f[i] = val_sum / static_cast<T>(counter);
                deallocate(neigh_corner_coord);

            } else {

                for (auto j = 0; j < 8; ++j) {
                    v_cubes[j] = val_c[corner_index[j]];
                }

                val_f[i] = TriLinearInterpolation(xf[i], corner_coord, v_cubes);

            }
        }
        deallocate(corner_coord);
    }


    ~TriLinearInterpolator()
    {
        if (xf) deallocate(xf);
        if (xc) deallocate(xc);
    };

private:
    unsigned int grid_c[3]{};
    unsigned int grid_f[3]{};
    unsigned int ngrid_f, ngrid_c;
    double **xf = nullptr; // coordinate of fine grid
    double **xc = nullptr; // coordinate of coarse grid


    static void set_grid(const unsigned int ngrid_in[3], double **x_out)
    {
        // gamma point will always be index 0
        size_t ik;
        double invn[3];
        for (auto i = 0; i < 3; ++i) {
            invn[i] = 1.0 / static_cast<double>(ngrid_in[i]);
        }
        for (size_t ix = 0; ix < ngrid_in[0]; ++ix) {
            for (size_t iy = 0; iy < ngrid_in[1]; ++iy) {
                for (size_t iz = 0; iz < ngrid_in[2]; ++iz) {
                    ik = iz + iy * ngrid_in[2] + ix * ngrid_in[2] * ngrid_in[1];
                    x_out[ik][0] = static_cast<double>(ix) * invn[0];
                    x_out[ik][1] = static_cast<double>(iy) * invn[1];
                    x_out[ik][2] = static_cast<double>(iz) * invn[2];
                }
            }
        }
    }


    void get_corners_regular(double *xk_i, int *corner_index, double **corner_coord)
    {
        // get the index of the corner, as well as coordinates[8][3]
        int iloc[2], jloc[2], kloc[2];
        double dn_c[3];
        double tmp[3];
        int n23 = static_cast<int>(grid_c[1] * grid_c[2]);
        int igrid[3];

        for (auto i = 0; i < 3; ++i) {
            dn_c[i] = static_cast<double>(grid_c[i]);
            igrid[i] = static_cast<int>(grid_c[i]);
        }

        for (auto j = 0; j < 3; ++j) tmp[j] = xk_i[j] * dn_c[j];
        iloc[0] = nint(std::floor(tmp[0]));
        iloc[1] = nint(std::ceil(tmp[0]));
        jloc[0] = nint(std::floor(tmp[1]));
        jloc[1] = nint(std::ceil(tmp[1]));
        kloc[0] = nint(std::floor(tmp[2]));
        kloc[1] = nint(std::ceil(tmp[2]));

        if (iloc[1] == iloc[0]) ++iloc[1];
        if (jloc[1] == jloc[0]) ++jloc[1];
        if (kloc[1] == kloc[0]) ++kloc[1];

        corner_coord[0][0] = static_cast<double>(iloc[0]) / dn_c[0];
        corner_coord[0][1] = static_cast<double>(jloc[0]) / dn_c[1];
        corner_coord[0][2] = static_cast<double>(kloc[0]) / dn_c[2];

        corner_coord[1][0] = static_cast<double>(iloc[1]) / dn_c[0];
        corner_coord[1][1] = static_cast<double>(jloc[0]) / dn_c[1];
        corner_coord[1][2] = static_cast<double>(kloc[0]) / dn_c[2];

        corner_coord[2][0] = static_cast<double>(iloc[0]) / dn_c[0];
        corner_coord[2][1] = static_cast<double>(jloc[1]) / dn_c[1];
        corner_coord[2][2] = static_cast<double>(kloc[0]) / dn_c[2];

        corner_coord[3][0] = static_cast<double>(iloc[1]) / dn_c[0];
        corner_coord[3][1] = static_cast<double>(jloc[1]) / dn_c[1];
        corner_coord[3][2] = static_cast<double>(kloc[0]) / dn_c[2];

        corner_coord[4][0] = static_cast<double>(iloc[0]) / dn_c[0];
        corner_coord[4][1] = static_cast<double>(jloc[0]) / dn_c[1];
        corner_coord[4][2] = static_cast<double>(kloc[1]) / dn_c[2];

        corner_coord[5][0] = static_cast<double>(iloc[1]) / dn_c[0];
        corner_coord[5][1] = static_cast<double>(jloc[0]) / dn_c[1];
        corner_coord[5][2] = static_cast<double>(kloc[1]) / dn_c[2];

        corner_coord[6][0] = static_cast<double>(iloc[0]) / dn_c[0];
        corner_coord[6][1] = static_cast<double>(jloc[1]) / dn_c[1];
        corner_coord[6][2] = static_cast<double>(kloc[1]) / dn_c[2];

        corner_coord[7][0] = static_cast<double>(iloc[1]) / dn_c[0];
        corner_coord[7][1] = static_cast<double>(jloc[1]) / dn_c[1];
        corner_coord[7][2] = static_cast<double>(kloc[1]) / dn_c[2];

        iloc[0] = iloc[0] % igrid[0];
        iloc[1] = iloc[1] % igrid[0];
        jloc[0] = jloc[0] % igrid[1];
        jloc[1] = jloc[1] % igrid[1];
        kloc[0] = kloc[0] % igrid[2];
        kloc[1] = kloc[1] % igrid[2];

        corner_index[0] = kloc[0] + jloc[0] * igrid[2] + iloc[0] * n23; // index of c000
        corner_index[1] = kloc[0] + jloc[0] * igrid[2] + iloc[1] * n23; // index of c100
        corner_index[2] = kloc[0] + jloc[1] * igrid[2] + iloc[0] * n23; // index of c010
        corner_index[3] = kloc[0] + jloc[1] * igrid[2] + iloc[1] * n23; // index of c110
        corner_index[4] = kloc[1] + jloc[0] * igrid[2] + iloc[0] * n23; // index of c001
        corner_index[5] = kloc[1] + jloc[0] * igrid[2] + iloc[1] * n23; // index of c101
        corner_index[6] = kloc[1] + jloc[1] * igrid[2] + iloc[0] * n23; // index of c011
        corner_index[7] = kloc[1] + jloc[1] * igrid[2] + iloc[1] * n23; // index of c111

    }


    template <typename T>
    T TriLinearInterpolation(double *center, double **corners_coord, const T *val_corner)
    {
        T tx = static_cast<T>(center[0] - corners_coord[0][0]) * static_cast<T>(grid_c[0]);
        T ty = static_cast<T>(center[1] - corners_coord[0][1]) * static_cast<T>(grid_c[1]);
        T tz = static_cast<T>(center[2] - corners_coord[0][2]) * static_cast<T>(grid_c[2]);

        const auto c0 = BiLinearInterpolation(tx,
                                              ty,
                                              val_corner[0],
                                              val_corner[1],
                                              val_corner[2],
                                              val_corner[3]);
        const auto c1 = BiLinearInterpolation(tx,
                                              ty,
                                              val_corner[4],
                                              val_corner[5],
                                              val_corner[6],
                                              val_corner[7]);
        return (c1 - c0) * tz + c0;
    }


    template <typename T>
    T BiLinearInterpolation(const T tx, const T ty,
                            const T c00, const T c10,
                            const T c01, const T c11)
    {
        return c00 + (c10 - c00) * tx + (c01 - c00) * ty
               + (c11 - c01 - c10 + c00) * tx * ty;
    }

};

}

/* original interpolate function

    template<typename T>
    void interpolate(const T *val_c, T *val_f, const bool regular_grid = true)
    {

        T v_cubes[8];
        T tx, ty, tz;
        T invdel_x, invdel_y, invdel_z;

        if (regular_grid) {
            invdel_x = static_cast<T>(grid_c[0]);
            invdel_y = static_cast<T>(grid_c[1]);
            invdel_z = static_cast<T>(grid_c[2]);

            for (auto i = 0; i < ngrid_f; ++i) {
                for (auto j = 0; j < 8; ++j) {
                    v_cubes[j] = val_c[cubes[i][j]];
                }
                tx = static_cast<T>(xf[i][0] - xc[cubes[i][0]][0]) * invdel_x;
                ty = static_cast<T>(xf[i][1] - xc[cubes[i][0]][1]) * invdel_y;
                tz = static_cast<T>(xf[i][2] - xc[cubes[i][0]][2]) * invdel_z;

                const auto c0 = BiLinearInterpolation(tx, ty,
                                                      v_cubes[0], v_cubes[1],
                                                      v_cubes[2], v_cubes[3]);
                const auto c1 = BiLinearInterpolation(tx, ty,
                                                      v_cubes[4], v_cubes[5],
                                                      v_cubes[6], v_cubes[7]);
                val_f[i] = (c1 - c0) * tz + c0;
            }
        }
    }

*/
