/*
 dense_symmetric_eigen.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <vector>

namespace PHON_NS
{
// Backend seam for dense symmetric eigenproblems (used by SOLVER = DBTE;
// candidate later consumer: batched dynamical-matrix diagonalization).
//
// v1 backend: LAPACK dsyev on the calling rank. Planned backends behind the
// same call: ELPA/ScaLAPACK (memory-distributed; assembly is already
// row-distributed, so only a block-cyclic redistribution is needed) and
// MAGMA/cuSOLVER (single-node GPU). Distributed backends will take this
// call collectively.
//
// A is n x n column-major and is overwritten by the eigenvectors (column j
// = eigenvector of w[j]); eigenvalues are returned in ascending order.
// num_lowest >= 0 asks for only the lowest eigenpairs - the v1 backend
// computes the full spectrum and lets the caller truncate, but
// range-capable backends (dsyevr, ELPA partial, ...) may exploit it.
void solve_dense_symmetric(int n, std::vector<double> &A, std::vector<double> &w, int num_lowest = -1);
} // namespace PHON_NS
