/*
 cell_shift_table.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "ndarray.h"

#include <Eigen/Core>
#include <Eigen/QR>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

namespace PHON_NS
{
// Build the standard 27-image shift table: row 0 is the home cell, rows
// 1-26 the neighbor cells in {-1,0,1}^3 order (ix slowest, iz fastest).
inline void build_27cell_shift_table(NDArray<double, 2> &xshift_s)
{
    xshift_s.resize(27, 3);
    for (auto i = 0; i < 3; ++i) xshift_s[0][i] = 0.0;
    auto icell = 0;
    for (auto ix = -1; ix <= 1; ++ix) {
        for (auto iy = -1; iy <= 1; ++iy) {
            for (auto iz = -1; iz <= 1; ++iz) {
                if (ix == 0 && iy == 0 && iz == 0) continue;
                ++icell;
                xshift_s[icell][0] = static_cast<double>(ix);
                xshift_s[icell][1] = static_cast<double>(iy);
                xshift_s[icell][2] = static_cast<double>(iz);
            }
        }
    }
}

// Upper-triangular factor R of lat = Q R (lat: lattice vectors as columns), so that
// |lat * v| = |R * v|. Compute it once per lattice and pass it to images_beyond_27.
inline Eigen::Matrix3d lattice_r_factor(const Eigen::Matrix3d &lat)
{
    const Eigen::HouseholderQR<Eigen::Matrix3d> qr(lat);
    return qr.matrixQR().triangularView<Eigen::Upper>();
}

// Lattice images n outside {-1,0,1}^3 of the fractional vector dx with |lat * (dx + n)| < rmax,
// as (distance, n) pairs; rfac = lattice_r_factor(lat). The images are enumerated inside the
// sphere of radius rmax only (Fincke-Pohst: n_3 bounded by row 3 of R, then n_2 by the radius
// left over, then n_1), so the result is complete for any cell shape and the work stays
// proportional to the number of lattice points near the sphere, also for elongated cells. With
// rmax = (shortest of the 27 images) + tie tolerance, a non-empty result means a skewed cell whose
// 27-image search misses the minimum image or one of its equidistant partners; for orthogonal and
// near-orthogonal cells it is empty. n_visited (optional) returns the number of candidates tried.
inline std::vector<std::pair<double, std::array<int, 3>>>
images_beyond_27(const Eigen::Matrix3d &rfac, const double dx[3], const double rmax, long *n_visited = nullptr)
{
    // integer n with |diag * (dx_i + n) + offset| <= radius
    const auto bounds =
        [](const double diag, const double dxi, const double offset, const double radius, int &lo, int &hi) {
            const auto a = (-offset - radius) / diag;
            const auto b = (-offset + radius) / diag;
            lo = static_cast<int>(std::ceil(std::min(a, b) - dxi - 1.0e-9));
            hi = static_cast<int>(std::floor(std::max(a, b) - dxi + 1.0e-9));
        };
    const auto r2max = rmax * rmax;
    std::vector<std::pair<double, std::array<int, 3>>> out;
    long nv = 0;
    int lo3, hi3, lo2, hi2, lo1, hi1;
    bounds(rfac(2, 2), dx[2], 0.0, rmax, lo3, hi3);
    for (auto n3 = lo3; n3 <= hi3; ++n3) {
        const auto v3 = dx[2] + n3;
        const auto y3 = rfac(2, 2) * v3;
        const auto rem3 = r2max - y3 * y3;
        if (rem3 < 0.0) continue;
        bounds(rfac(1, 1), dx[1], rfac(1, 2) * v3, std::sqrt(rem3), lo2, hi2);
        for (auto n2 = lo2; n2 <= hi2; ++n2) {
            const auto v2 = dx[1] + n2;
            const auto y2 = rfac(1, 1) * v2 + rfac(1, 2) * v3;
            const auto rem2 = rem3 - y2 * y2;
            if (rem2 < 0.0) continue;
            bounds(rfac(0, 0), dx[0], rfac(0, 1) * v2 + rfac(0, 2) * v3, std::sqrt(rem2), lo1, hi1);
            for (auto n1 = lo1; n1 <= hi1; ++n1) {
                ++nv;
                if (std::abs(n1) <= 1 && std::abs(n2) <= 1 && std::abs(n3) <= 1) continue;
                const auto y1 = rfac(0, 0) * (dx[0] + n1) + rfac(0, 1) * v2 + rfac(0, 2) * v3;
                const auto d = std::sqrt(y1 * y1 + y2 * y2 + y3 * y3);
                if (d < rmax) out.push_back({d, {n1, n2, n3}});
            }
        }
    }
    if (n_visited) *n_visited = nv;
    return out;
}
} // namespace PHON_NS
