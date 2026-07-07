/*
 cell_shift_table.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "ndarray.h"

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
} // namespace PHON_NS
