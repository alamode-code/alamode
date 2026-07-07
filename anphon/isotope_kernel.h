/*
 isotope_kernel.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <cmath>
#include <complex>

namespace PHON_NS
{
// Tamura mass-variance overlap: sum_iat g2[kind[iat]] * |<e_a(iat)|e_b(iat)>|^2,
// with e_a conjugated. Exact accumulation order of the historical loops.
inline double tamura_overlap(const unsigned int natmin, const std::complex<double> *evec_conj,
                             const std::complex<double> *evec_ref, const double *isotope_factor,
                             const int *kind)
{
    auto prod = 0.0;

    for (auto iat = 0; iat < natmin; ++iat) {

        auto dprod = std::complex<double>(0.0, 0.0);
        for (auto icrd = 0; icrd < 3; ++icrd) {
            dprod += std::conj(evec_conj[3 * iat + icrd]) * evec_ref[3 * iat + icrd];
        }
        prod += isotope_factor[kind[iat]] * std::norm(dprod);
    }

    return prod;
}

// Transposed [ns][nk] sibling of average_over_degenerate_modes in degeneracy_utils.h.
inline void average_degenerate_frequencies_transposed(const int nk, const int ns, const double *const *eval_in,
                                                      const double tol_degenerate, double *const *eval_avg_out)
{
    for (int ik = 0; ik < nk; ++ik) {
        auto begin = 0;
        auto omega_ref = eval_in[ik][0];
        auto omega_sum = eval_in[ik][0];
        for (int is = 1; is < ns; ++is) {
            const auto omega_now = eval_in[ik][is];
            if (std::abs(omega_now - omega_ref) < tol_degenerate) {
                omega_sum += omega_now;
            } else {
                const auto omega_avg = omega_sum / static_cast<double>(is - begin);
                for (auto js = begin; js < is; ++js) eval_avg_out[js][ik] = omega_avg;
                begin = is;
                omega_ref = omega_now;
                omega_sum = omega_now;
            }
        }
        const auto omega_avg = omega_sum / static_cast<double>(ns - begin);
        for (auto js = begin; js < ns; ++js) eval_avg_out[js][ik] = omega_avg;
    }
}

inline void average_tetra_weights_over_degenerate_modes(const int ns, const int ik,
                                                        const double *const *eval_avg,
                                                        double *const *weight_tetra,
                                                        const double tol_degenerate)
{
    auto begin = 0;
    auto omega_ref = eval_avg[0][ik];
    for (int is = 1; is <= ns; ++is) {
        if (is < ns && std::abs(eval_avg[is][ik] - omega_ref) < tol_degenerate) continue;
        if (is - begin > 1) {
            auto wsum = 0.0;
            for (auto js = begin; js < is; ++js) wsum += weight_tetra[js][ik];
            wsum /= static_cast<double>(is - begin);
            for (auto js = begin; js < is; ++js) weight_tetra[js][ik] = wsum;
        }
        if (is < ns) {
            begin = is;
            omega_ref = eval_avg[is][ik];
        }
    }
}
} // namespace PHON_NS
