/*
 grid_phase.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <vector>

namespace PHON_NS
{
inline int positive_modulo(const int a, const int b)
{
    const int m = a % b;
    return m < 0 ? m + b : m;
}

// exp(i sign k.R) for integer R and k = q / N on the uniform grid.
inline std::complex<double> phase_factor(const int *q, const int *R, const int sign, const int *ngrid,
                                         const std::vector<std::complex<double>> *table)
{
    std::complex<double> e(1.0, 0.0);
    for (auto icrd = 0; icrd < 3; ++icrd) {
        e *= table[icrd][positive_modulo(sign * q[icrd] * R[icrd], ngrid[icrd])];
    }
    return e;
}
} // namespace PHON_NS
