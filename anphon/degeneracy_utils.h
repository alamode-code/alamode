/*
 degeneracy_utils.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <cmath>
#include <vector>

namespace PHON_NS
{
inline void find_degenerate_groups(const unsigned int ns, const double *eval_at_k, std::vector<int> &degeneracy_out,
                                   const double tol_omega = 1.0e-7)
{
    degeneracy_out.clear();

    auto omega_prev = eval_at_k[0];
    auto ideg = 1;

    for (unsigned int is = 1; is < ns; ++is) {
        const auto omega_now = eval_at_k[is];

        if (std::abs(omega_now - omega_prev) < tol_omega) {
            ++ideg;
        } else {
            degeneracy_out.push_back(ideg);
            ideg = 1;
            omega_prev = omega_now;
        }
    }
    degeneracy_out.push_back(ideg);
}

// Replace per-branch data by its average over each degenerate subspace at
// one k point. eval_at_k holds the ns sorted eigenvalues; data is [ns][width]
// row-major (width = 1 for scalars, 3 for Cartesian vectors, ntemp for
// temperature rows) and is averaged column-wise within each group of
// consecutive eigenvalues closer than tol_omega (~0.01 cm^-1 by default).
inline void average_over_degenerate_modes(const int ns, const double *eval_at_k, const int width, double *data,
                                          const double tol_omega = 1.0e-7)
{
    std::vector<int> degeneracy_at_k;
    find_degenerate_groups(ns, eval_at_k, degeneracy_at_k, tol_omega);

    std::vector<double> data_sum(width);

    int is = 0;
    for (const auto ideg_now: degeneracy_at_k) {
        if (ideg_now > 1) {
            for (int l = 0; l < width; ++l) data_sum[l] = 0.0;
            for (int k = is; k < is + ideg_now; ++k) {
                for (int l = 0; l < width; ++l) {
                    data_sum[l] += data[k * width + l];
                }
            }
            for (int k = is; k < is + ideg_now; ++k) {
                for (int l = 0; l < width; ++l) {
                    data[k * width + l] = data_sum[l] / static_cast<double>(ideg_now);
                }
            }
        }
        is += ideg_now;
    }
}
} // namespace PHON_NS
