/*
 dense_hermitian_eigen.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>

namespace PHON_NS
{
// Backend seam for dense Hermitian eigenproblems (used by dynamical-matrix
// diagonalization).
//
// v1 backend: LAPACK zheev on the calling rank. Candidate later backends
// behind the same call: ELPA and MAGMA.
//
// mat_in is row-pointer indexed [i][j] and is copied into column-major scratch
// internally. LWORK is fixed at (2n-1)*10 with no workspace query because zheev
// may take a different (blocked vs unblocked) path for different workspace
// sizes and bit-identical results with the historical call are required.
// compute_evec drives JOBZ ('V'/'N'); evec_out (nullable) independently gates
// the eigenvector write-back.
void solve_dense_hermitian(int n, const std::complex<double> *const *mat_in, double *eval_out,
                           std::complex<double> **evec_out, bool compute_evec, char uplo = 'U');
} // namespace PHON_NS
