/*
 v4_distributed.h

 Row-distributed storage of the reciprocal-space quartic coefficients

   v4[ik_prod][a*ns + b][c*ns + d],   ik_prod = ik_irred_coarse * nk_dense + jk,

 for the SCPH/QHA structural optimization. The full array is
 nk_irred*nk_dense x ns^2 x ns^2 complex (560 GB for a 144-atom Gamma-only cell);
 here every MPI rank keeps only a contiguous range of "units"

   u = ik_prod * ns + a   (the ns rows (a, b), b = 0..ns-1, of one slice),

 which is exactly the granularity in which the band-parallel builder
 (compute_V4_elements_mpi_over_band) computes the array. The k-point builder
 computes whole slices (ns units each), so its partition keeps slices intact.

 Consumers only ever need rows, never columns, of v4:
   - the SCP self-energy F_v4[ik_irred](a,b) = sum_jk sum_c v4[(ik_irred,jk)][a,b][c] D_jk[c]
     (accumulate_fmat below, rows b <= a),
   - the q0 renormalization sweep (q0_contraction.h, Options::unit_begin/end),
   - the diagonal elements v4[(ik_irred,knum)][(ns+1)a][(ns+1)a] (gathered once).
 Each rank contracts its own rows and the partial results are summed over the
 ranks by the caller (MPI stays outside this header).

 Load balance: the F_v4 contraction touches only the lower-triangle rows b <= a
 of unit (ik_prod, a), i.e. a + 1 rows; equal unit counts would give the last
 rank 1.75x the average work at four ranks. partition_units therefore balances
 the prefix sum of a caller-supplied weight (unit_weights).
*/

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
#include "ndarray.h"

namespace PHON_NS::v4_distributed
{

// Partition weighted units contiguously; rank r owns [bounds[r], bounds[r+1]).
// Choose prefixes nearest each target cumulative share, breaking ties toward
// earlier prefixes. Return nprocs+1 bounds; ranks may be empty.
inline std::vector<std::size_t> partition_units(const std::vector<double> &weight, const int nprocs)
{
    const std::size_t nunits = weight.size();
    std::vector<std::size_t> bounds(nprocs + 1, nunits);
    bounds[0] = 0;
    double total = 0.0;
    for (const auto w: weight) {
        total += w;
    }
    std::size_t u = 0;
    double cumulative = 0.0;
    for (int r = 0; r + 1 < nprocs; ++r) {
        const double target = total * static_cast<double>(r + 1) / static_cast<double>(nprocs);
        while (u < nunits && cumulative + 0.5 * weight[u] < target) {
            cumulative += weight[u];
            ++u;
        }
        bounds[r + 1] = u;
    }
    bounds[nprocs] = nunits;
    return bounds;
}

// Contiguous partition of whole slices (ns units each) over nprocs ranks, in
// units: rank r owns the slices [s0, s1) with the remainder given to the first
// ranks. Ranks beyond nslices own nothing.
inline std::vector<std::size_t> partition_slices(const std::size_t nslices, const std::size_t ns, const int nprocs)
{
    std::vector<std::size_t> bounds(nprocs + 1, nslices * ns);
    bounds[0] = 0;
    const std::size_t each = nslices / static_cast<std::size_t>(nprocs);
    const std::size_t rest = nslices - each * static_cast<std::size_t>(nprocs);
    std::size_t s = 0;
    for (int r = 0; r < nprocs; ++r) {
        s += each + (static_cast<std::size_t>(r) < rest ? 1 : 0);
        bounds[r + 1] = s * ns;
    }
    return bounds;
}

// Work of unit (ik_prod, a) per SCPH iteration plus per structure step:
// the F_v4 contraction reads a + 1 rows (offdiag) or 1 row (diagonal only);
// the q0 sweep reads every row of the Gamma families (slices (g, jk) and (ik, jg)).
// Weights are relative; the sweep runs once per step against tens of SCPH
// iterations, so its rows count with a small factor.
inline std::vector<double> unit_weights(const std::size_t ns, const std::size_t nk_dense, const std::size_t nk_irred,
                                        const std::size_t ik_gamma_irred, const std::size_t jk_gamma_dense,
                                        const bool offdiag, const double sweep_weight = 0.1)
{
    std::vector<double> weight(nk_irred * nk_dense * ns, 0.0);
    for (std::size_t ik = 0; ik < nk_irred; ++ik) {
        for (std::size_t jk = 0; jk < nk_dense; ++jk) {
            const std::size_t ik_prod = ik * nk_dense + jk;
            const bool gamma_family = (ik == ik_gamma_irred) || (jk == jk_gamma_dense);
            for (std::size_t a = 0; a < ns; ++a) {
                double w = offdiag ? static_cast<double>(a + 1) : 1.0;
                if (gamma_family) {
                    w += sweep_weight * static_cast<double>(ns);
                }
                weight[ik_prod * ns + a] = w;
            }
        }
    }
    return weight;
}

struct V4RowBlock
{
    std::size_t ns = 0, ns2 = 0, nk_dense = 0, nk_irred = 0;
    std::size_t unit_begin = 0, unit_end = 0; // owned units [unit_begin, unit_end)
    NDArray<std::complex<double>, 2> rows;    // (unit_end - unit_begin) * ns rows of ns2 complex

    void allocate(const std::size_t ns_in, const std::size_t nk_dense_in, const std::size_t nk_irred_in,
                  const std::size_t unit_begin_in, const std::size_t unit_end_in)
    {
        ns = ns_in;
        ns2 = ns * ns;
        nk_dense = nk_dense_in;
        nk_irred = nk_irred_in;
        unit_begin = std::min(unit_begin_in, nunits_total());
        unit_end = std::min(std::max(unit_end_in, unit_begin), nunits_total());
        rows.clear();
        if (nunits_local() > 0) {
            rows.resize(nunits_local() * ns, ns2);
        }
    }

    std::size_t nunits_total() const
    {
        return nk_irred * nk_dense * ns;
    }

    std::size_t nunits_local() const
    {
        return unit_end - unit_begin;
    }

    std::size_t nrows_local() const
    {
        return nunits_local() * ns;
    }

    double bytes_local() const
    {
        return static_cast<double>(nrows_local()) * static_cast<double>(ns2) * sizeof(std::complex<double>);
    }

    static std::size_t unit_of(const std::size_t ik_prod, const std::size_t a, const std::size_t ns)
    {
        return ik_prod * ns + a;
    }

    bool owns_unit(const std::size_t u) const
    {
        return u >= unit_begin && u < unit_end;
    }

    // Owned rows a of slice ik_prod: [a0, a1)
    void owned_a_range(const std::size_t ik_prod, std::size_t &a0, std::size_t &a1) const
    {
        const std::size_t base = ik_prod * ns;
        a0 = unit_begin > base ? std::min(unit_begin - base, ns) : 0;
        a1 = unit_end > base ? std::min(unit_end - base, ns) : 0;
        if (a1 < a0) {
            a1 = a0;
        }
    }

    // Row (a, b) of the owned unit u = (ik_prod, a)
    std::complex<double> *row(const std::size_t u, const std::size_t b)
    {
        return rows[(u - unit_begin) * ns + b];
    }

    const std::complex<double> *row(const std::size_t u, const std::size_t b) const
    {
        return rows[(u - unit_begin) * ns + b];
    }
};

// Accumulate the anharmonic part of the SCP self-energy from the owned rows:
//   offdiag:       F[ik_irred](a,b) += sum_jk sum_c v4[(ik_irred,jk)][a,b][c] dvec[jk][c],  b <= a
//   diagonal only: F[ik_irred](a,a) += sum_jk sum_ks v4[(ik_irred,jk)][a,a][(ns+1) ks] dvec[jk][(ns+1) ks]
// dvec[jk*ns2 + ks*ns + ls] = D_jk(ks, ls) (the D matrices flattened row by row,
// as Scph::update_fmat_with_v4 builds them). fmat_inout[ik_irred][a][b] is
// accumulated into (the caller seeds it with the harmonic F on one rank and with
// zero elsewhere); only the lower triangle b <= a is touched, the Hermitian
// completion is the caller's. The accumulation order is that of the
// single-process Scph::update_fmat_with_v4: off-diagonal, the jk sum of a row
// pair is formed in a local accumulator and added once; diagonal only, every
// product is added directly to F. A single rank owning everything therefore
// reproduces the single-process arithmetic (up to the vectorized row sum).
inline void accumulate_fmat(const V4RowBlock &blk, const std::complex<double> *dvec, const bool offdiag,
                            std::complex<double> ***fmat_inout)
{
    using namespace Eigen;
    const std::size_t ns = blk.ns;
    const std::size_t ns2 = blk.ns2;
    const std::size_t nk = blk.nk_dense;

    for (std::size_t ik_irred = 0; ik_irred < blk.nk_irred; ++ik_irred) {
        // any owned unit in this ik_irred's slices?
        const std::size_t first = ik_irred * nk * ns, last = (ik_irred + 1) * nk * ns;
        if (blk.unit_end <= first || blk.unit_begin >= last) {
            continue;
        }

        if (!offdiag) {
#pragma omp parallel for schedule(static)
            for (long a_l = 0; a_l < static_cast<long>(ns); ++a_l) {
                const auto a = static_cast<std::size_t>(a_l);
                std::complex<double> &f = fmat_inout[ik_irred][a][a];
                for (std::size_t jk = 0; jk < nk; ++jk) {
                    const std::size_t u = V4RowBlock::unit_of(ik_irred * nk + jk, a, ns);
                    if (!blk.owns_unit(u)) {
                        continue;
                    }
                    const std::complex<double> *r = blk.row(u, a);
                    const std::complex<double> *d = dvec + jk * ns2;
                    for (std::size_t ks = 0; ks < ns; ++ks) {
                        f += r[(ns + 1) * ks] * d[(ns + 1) * ks];
                    }
                }
            }
            continue;
        }

        const long npairs = static_cast<long>(ns * (ns + 1) / 2);
#pragma omp parallel for schedule(dynamic, 16)
        for (long ip = 0; ip < npairs; ++ip) {
            // lower-triangle pair (a, b), b <= a, enumerated row by row
            std::size_t a = 0, b = 0, count = 0;
            {
                // a = floor((sqrt(8 ip + 1) - 1) / 2), corrected for rounding
                a = static_cast<std::size_t>((std::sqrt(8.0 * static_cast<double>(ip) + 1.0) - 1.0) / 2.0);
                while (a * (a + 1) / 2 > static_cast<std::size_t>(ip)) {
                    --a;
                }
                while ((a + 1) * (a + 2) / 2 <= static_cast<std::size_t>(ip)) {
                    ++a;
                }
                count = a * (a + 1) / 2;
                b = static_cast<std::size_t>(ip) - count;
            }
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t jk = 0; jk < nk; ++jk) {
                const std::size_t u = V4RowBlock::unit_of(ik_irred * nk + jk, a, ns);
                if (!blk.owns_unit(u)) {
                    continue;
                }
                sum += Map<const VectorXcd>(blk.row(u, b), ns2)
                           .cwiseProduct(Map<const VectorXcd>(dvec + jk * ns2, ns2))
                           .sum();
            }
            fmat_inout[ik_irred][a][b] += sum;
        }
    }
}

// Batched, full-matrix form for linear-response solves (the SCPH Hessian):
//   F_m[ik_irred](a,b) += sum_jk sum_c v4[(ik_irred,jk)][a,b][c] D_m[jk][c],  every (a, b), m < nrhs,
// with D_m at dmat + m * nk_dense * ns2 (same row-by-row layout as dvec) and F_m
// at fout + m * nk_irred * ns2, fout[.. + ik_irred * ns2 + a * ns + b]. Unlike
// accumulate_fmat it computes the upper triangle too, so the map is
// complex-linear in D (a Hermitian completion is only real-linear), and each
// owned unit (ik_prod, a) is one dense product of its ns x ns2 rows with the
// ns2 x nrhs block of D_jk. Needs every row (full_tensor).
inline void accumulate_fmat_batch(const V4RowBlock &blk, const std::complex<double> *dmat, const std::size_t nrhs,
                                  std::complex<double> *fout)
{
    using namespace Eigen;
    using RowMat = Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>;
    const std::size_t ns = blk.ns;
    const std::size_t ns2 = blk.ns2;
    const std::size_t nk = blk.nk_dense;
    const auto dstride = static_cast<Index>(nk * ns2);
    const auto fstride = static_cast<Index>(blk.nk_irred * ns2);

    for (std::size_t ik_irred = 0; ik_irred < blk.nk_irred; ++ik_irred) {
        // Threads own rows a, so each block F_m[ik_irred](a, :) has one writer.
#pragma omp parallel for schedule(dynamic, 1)
        for (long a_l = 0; a_l < static_cast<long>(ns); ++a_l) {
            const auto a = static_cast<std::size_t>(a_l);
            Map<Matrix<std::complex<double>, Dynamic, Dynamic>, 0, OuterStride<>> F(fout + ik_irred * ns2 + a * ns,
                                                                                    static_cast<Index>(ns),
                                                                                    static_cast<Index>(nrhs),
                                                                                    OuterStride<>(fstride));
            for (std::size_t jk = 0; jk < nk; ++jk) {
                const std::size_t u = V4RowBlock::unit_of(ik_irred * nk + jk, a, ns);
                if (!blk.owns_unit(u)) {
                    continue;
                }
                const Map<const RowMat> V(blk.row(u, 0), static_cast<Index>(ns), static_cast<Index>(ns2));
                const Map<const Matrix<std::complex<double>, Dynamic, Dynamic>, 0, OuterStride<>> D(
                    dmat + jk * ns2,
                    static_cast<Index>(ns2),
                    static_cast<Index>(nrhs),
                    OuterStride<>(dstride));
                F.noalias() += V * D;
            }
        }
    }
}

} // namespace PHON_NS::v4_distributed
