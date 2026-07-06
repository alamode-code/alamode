/*
 interpolation.cpp

 Copyright (c) 2021 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "interpolation.h"
#include <cmath>
#include "error.h"

using namespace PHON_NS;

// TriLinearInterpolator implementations

TriLinearInterpolator::TriLinearInterpolator(const unsigned int ngrid_coarse_in[3],
                                             const unsigned int ngrid_dense_in[3])
{
    for (auto i = 0; i < 3; ++i) {
        grid_c[i] = ngrid_coarse_in[i];
        grid_f[i] = ngrid_dense_in[i];
    }
    ngrid_c = grid_c[0] * grid_c[1] * grid_c[2];
    ngrid_f = grid_f[0] * grid_f[1] * grid_f[2];

    xf.resize(ngrid_f, 3);
    xc.resize(ngrid_c, 3);
}

void TriLinearInterpolator::setup()
{
    set_grid(grid_c, xc);
    set_grid(grid_f, xf);
}

TriLinearInterpolator::~TriLinearInterpolator()
{

}

void TriLinearInterpolator::set_grid(const unsigned int ngrid_in[3], double **x_out)
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

void TriLinearInterpolator::get_corners_regular(double *xk_i, int *corner_index, double **corner_coord)
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

    for (auto j = 0; j < 3; ++j)
        tmp[j] = xk_i[j] * dn_c[j];
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

// FourierInterpolator implementations

FourierInterpolator::FourierInterpolator(const KpointMeshUniform &kmesh_coarse_in,
                                         const KpointMeshUniform &kmesh_dense_in, const bool subtract_harmonic_term_in)
    : kmesh_coarse(kmesh_coarse_in), kmesh_dense(kmesh_dense_in)
{
    subtract_harmonic_term = subtract_harmonic_term_in;

    get_map_coarse_to_dense(kmesh_coarse, kmesh_dense, map_corase_to_dense);
}

void FourierInterpolator::get_map_coarse_to_dense(const KpointMeshUniform &kmesh_coarse,
                                                  const KpointMeshUniform &kmesh_dense, std::vector<int> &kmap)
{
    kmap.resize(kmesh_coarse.nk);
    double xtmp[3];

    for (auto ik = 0; ik < kmesh_coarse.nk; ++ik) {
        for (auto i = 0; i < 3; ++i)
            xtmp[i] = kmesh_coarse.xk[ik][i];

        const auto loc = kmesh_dense.get_knum(xtmp);

        if (loc == -1) {
            exit("FourierInterpolator::get_map_coarse_to_dense",
                 "Cannot find the corresponding kpoint in the dense mesh");
        }

        kmap[ik] = loc;
    }
}
