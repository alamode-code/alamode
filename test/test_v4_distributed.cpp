/*
 test_v4_distributed.cpp

 Unit test for anphon/v4_distributed.h: the contiguous partitions of the
 quartic-coefficient units (weighted units for the band-parallel builder,
 whole slices for the k-point builder), the V4RowBlock accessors, and the
 row-local accumulation of the SCP self-energy (accumulate_fmat) against a
 naive implementation of the formula of Scph::update_fmat_with_v4, for the
 full array and for the partial sums of random partitions (off-diagonal and
 diagonal-only forms), the legacy accumulation order of the diagonal-only
 form (bitwise), and 1 versus several OpenMP threads on a restricted range
 (bitwise).

 Built by the anphon CMake project as `test_v4_distributed`. Exits 0 on
 success and prints the first failing check otherwise.
*/

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "ndarray.h"
#include "v4_distributed.h"

using cplx = std::complex<double>;
using namespace PHON_NS::v4_distributed;

namespace
{
int nfail = 0;

void check(const bool ok, const std::string &what)
{
    if (!ok) {
        ++nfail;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

bool close(const cplx &a, const cplx &b, const double tol)
{
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

void test_partitions()
{
    // weighted units: shares within one unit weight of the ideal, contiguous, complete
    for (const int nprocs: {1, 2, 3, 4, 7}) {
        const std::size_t ns = 9, nk = 2, nk_irred = 2;
        const auto w = unit_weights(ns, nk, nk_irred, 0, 0, true, 0.1);
        const auto bounds = partition_units(w, nprocs);
        check(bounds.size() == static_cast<std::size_t>(nprocs) + 1, "partition_units: size");
        check(bounds.front() == 0 && bounds.back() == w.size(), "partition_units: covers all units");
        double total = 0.0, wmax = 0.0;
        for (const auto x: w) {
            total += x;
            wmax = std::max(wmax, x);
        }
        for (int r = 0; r < nprocs; ++r) {
            check(bounds[r] <= bounds[r + 1], "partition_units: monotone");
            double share = 0.0;
            for (auto u = bounds[r]; u < bounds[r + 1]; ++u) {
                share += w[u];
            }
            check(std::abs(share - total / nprocs) <= wmax + 1.0e-12,
                  "partition_units: rank " + std::to_string(r) + " share off by more than one unit (nprocs " +
                      std::to_string(nprocs) + ")");
        }
    }
    // fewer units than ranks: some ranks (not necessarily the trailing ones) are
    // empty, nothing is lost; the nearest-prefix rule gives {0,1,1,2,2,3}
    {
        const std::vector<double> w(3, 1.0);
        const auto bounds = partition_units(w, 5);
        check(bounds == std::vector<std::size_t>({0, 1, 1, 2, 2, 3}), "partition_units: three units over five ranks");
    }
    // empty problem
    {
        const std::vector<double> w;
        const auto bounds = partition_units(w, 3);
        check(bounds == std::vector<std::size_t>({0, 0, 0, 0}), "partition_units: no units");
    }
    // unit weights: a + 1 rows (offdiag) or 1 (diagonal), plus 0.1 ns for the Gamma families
    {
        const std::size_t ns = 3, nk = 2, nk_irred = 2;
        const auto w = unit_weights(ns, nk, nk_irred, 0, 0, true, 0.1);
        // slice (0,0): Gamma-Gamma; (0,1): row-Gamma; (1,0): column-Gamma; (1,1): none
        check(w[0 * ns + 2] == 3.0 + 0.3 && w[1 * ns + 0] == 1.0 + 0.3 && w[2 * ns + 1] == 2.0 + 0.3 &&
                  w[3 * ns + 2] == 3.0,
              "unit_weights: offdiag values");
        const auto wd = unit_weights(ns, nk, nk_irred, 1, 1, false, 0.1);
        check(wd[0 * ns + 2] == 1.0 && wd[1 * ns + 1] == 1.3 && wd[3 * ns + 0] == 1.3, "unit_weights: diagonal values");
    }
    // block: clamping of the unit range and the owned a-range per slice
    {
        V4RowBlock blk;
        blk.allocate(3, 2, 2, 4, 100);
        check(blk.unit_begin == 4 && blk.unit_end == 12 && blk.nunits_local() == 8 && blk.nrows_local() == 24,
              "V4RowBlock::allocate clamps to the total number of units");
        std::size_t a0, a1;
        blk.owned_a_range(0, a0, a1);
        check(a0 == 3 && a1 == 3, "owned_a_range: slice fully below the range is empty");
        blk.owned_a_range(1, a0, a1);
        check(a0 == 1 && a1 == 3, "owned_a_range: slice cut by unit_begin");
        blk.owned_a_range(3, a0, a1);
        check(a0 == 0 && a1 == 3, "owned_a_range: slice fully owned");
        check(blk.owns_unit(4) && !blk.owns_unit(3) && !blk.owns_unit(12), "owns_unit");
        blk.allocate(3, 2, 2, 7, 5);
        check(blk.nunits_local() == 0, "V4RowBlock::allocate with end < begin is empty");
    }
    // slices: boundaries at multiples of ns, remainder to the first ranks, empty ranks beyond nslices
    {
        const std::size_t ns = 4;
        auto bounds = partition_slices(7, ns, 3);
        check(bounds == std::vector<std::size_t>({0, 12, 20, 28}), "partition_slices: 7 slices over 3 ranks (3,2,2)");
        bounds = partition_slices(1, ns, 3);
        check(bounds == std::vector<std::size_t>({0, 4, 4, 4}), "partition_slices: 1 slice over 3 ranks");
    }
}

// naive reference of update_fmat_with_v4 (anharmonic part only)
void reference_fmat(const NDArray<cplx, 3> &v4, const std::size_t ns, const std::size_t nk, const std::size_t nk_irred,
                    const std::vector<cplx> &dvec, const bool offdiag, NDArray<cplx, 3> &f)
{
    const auto ns2 = ns * ns;
    for (std::size_t ik = 0; ik < nk_irred; ++ik) {
        for (std::size_t a = 0; a < ns; ++a) {
            for (std::size_t b = 0; b < ns; ++b) {
                f[ik][a][b] = cplx(0.0, 0.0);
            }
        }
        if (offdiag) {
            for (std::size_t a = 0; a < ns; ++a) {
                for (std::size_t b = 0; b <= a; ++b) {
                    cplx sum(0.0, 0.0);
                    for (std::size_t jk = 0; jk < nk; ++jk) {
                        for (std::size_t c = 0; c < ns2; ++c) {
                            sum += v4[ik * nk + jk][a * ns + b][c] * dvec[jk * ns2 + c];
                        }
                    }
                    f[ik][a][b] = sum;
                }
            }
        } else {
            for (std::size_t a = 0; a < ns; ++a) {
                cplx sum(0.0, 0.0);
                for (std::size_t jk = 0; jk < nk; ++jk) {
                    for (std::size_t ks = 0; ks < ns; ++ks) {
                        sum += v4[ik * nk + jk][(ns + 1) * a][(ns + 1) * ks] * dvec[jk * ns2 + (ns + 1) * ks];
                    }
                }
                f[ik][a][a] = sum;
            }
        }
    }
}

void fill_block_from_full(V4RowBlock &blk, const NDArray<cplx, 3> &v4)
{
    for (std::size_t u = blk.unit_begin; u < blk.unit_end; ++u) {
        const std::size_t ik_prod = u / blk.ns, a = u % blk.ns;
        for (std::size_t b = 0; b < blk.ns; ++b) {
            std::copy(v4[ik_prod][a * blk.ns + b], v4[ik_prod][a * blk.ns + b] + blk.ns2, blk.row(u, b));
        }
    }
}

void test_fmat(const std::size_t ns, const std::size_t nk, const std::size_t nk_irred, const bool offdiag,
               const unsigned seed, const std::string &label)
{
    const auto ns2 = ns * ns;
    const double tol = 1.0e-13;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    NDArray<cplx, 3> v4(nk_irred * nk, ns2, ns2);
    for (std::size_t i = 0; i < v4.size(); ++i) {
        (&v4[0][0][0])[i] = cplx(dist(rng), dist(rng));
    }
    std::vector<cplx> dvec(nk * ns2);
    for (auto &x: dvec) {
        x = cplx(dist(rng), dist(rng));
    }
    NDArray<cplx, 3> f_ref(nk_irred, ns, ns);
    reference_fmat(v4, ns, nk, nk_irred, dvec, offdiag, f_ref);

    const std::size_t nunits = nk_irred * nk * ns;
    for (const int nparts: {1, 2, 3, 6}) {
        std::vector<std::size_t> bounds{0};
        std::uniform_int_distribution<std::size_t> pick(0, nunits);
        for (int ip = 1; ip < nparts; ++ip) {
            bounds.push_back(pick(rng));
        }
        bounds.push_back(nunits);
        std::sort(bounds.begin(), bounds.end());

        NDArray<cplx, 3> f_sum(nk_irred, ns, ns);
        std::fill(&f_sum[0][0][0], &f_sum[0][0][0] + f_sum.size(), cplx(0.0, 0.0));
        for (int ip = 0; ip < nparts; ++ip) {
            V4RowBlock blk;
            blk.allocate(ns, nk, nk_irred, bounds[ip], bounds[ip + 1]);
            check(blk.nunits_local() == bounds[ip + 1] - bounds[ip], label + ": nunits_local");
            fill_block_from_full(blk, v4);
            NDArray<cplx, 3> f_part(nk_irred, ns, ns);
            std::fill(&f_part[0][0][0], &f_part[0][0][0] + f_part.size(), cplx(0.0, 0.0));
            accumulate_fmat(blk, dvec.data(), offdiag, f_part);
            for (std::size_t i = 0; i < f_sum.size(); ++i) {
                (&f_sum[0][0][0])[i] += (&f_part[0][0][0])[i];
            }
        }
        bool ok = true;
        for (std::size_t ik = 0; ik < nk_irred && ok; ++ik) {
            for (std::size_t a = 0; a < ns && ok; ++a) {
                for (std::size_t b = 0; b < ns && ok; ++b) {
                    ok = close(f_sum[ik][a][b], f_ref[ik][a][b], tol);
                }
            }
        }
        check(ok, label + " [" + std::to_string(nparts) + " parts]: partial Fmat sums != reference");
    }
    // diagonal-only: the legacy order (every product added directly to the seeded F) must be
    // reproduced bitwise by a single block owning everything
    if (!offdiag) {
        V4RowBlock blk;
        blk.allocate(ns, nk, nk_irred, 0, nunits);
        fill_block_from_full(blk, v4);
        NDArray<cplx, 3> f(nk_irred, ns, ns), f_legacy(nk_irred, ns, ns);
        for (std::size_t i = 0; i < f.size(); ++i) {
            (&f[0][0][0])[i] = cplx(1.0e3, -1.0e3);
            (&f_legacy[0][0][0])[i] = cplx(1.0e3, -1.0e3);
        }
        for (std::size_t ik = 0; ik < nk_irred; ++ik) {
            for (std::size_t a = 0; a < ns; ++a) {
                for (std::size_t jk = 0; jk < nk; ++jk) {
                    for (std::size_t ks = 0; ks < ns; ++ks) {
                        f_legacy[ik][a][a] +=
                            v4[ik * nk + jk][(ns + 1) * a][(ns + 1) * ks] * dvec[jk * ns2 + (ns + 1) * ks];
                    }
                }
            }
        }
        accumulate_fmat(blk, dvec.data(), offdiag, f);
        check(std::memcmp(&f[0][0][0], &f_legacy[0][0][0], f.size() * sizeof(cplx)) == 0,
              label + ": diagonal-only accumulation order differs from the legacy loop");
    }
#ifdef _OPENMP
    // thread-count independence on a restricted range (bitwise)
    {
        V4RowBlock blk;
        blk.allocate(ns, nk, nk_irred, nunits / 3, (2 * nunits) / 3 + 1);
        fill_block_from_full(blk, v4);
        NDArray<cplx, 3> f1(nk_irred, ns, ns), fn(nk_irred, ns, ns);
        std::fill(&f1[0][0][0], &f1[0][0][0] + f1.size(), cplx(0.0, 0.0));
        std::fill(&fn[0][0][0], &fn[0][0][0] + fn.size(), cplx(0.0, 0.0));
        const int nthreads_saved = omp_get_max_threads();
        omp_set_num_threads(1);
        accumulate_fmat(blk, dvec.data(), offdiag, f1);
        omp_set_num_threads(std::max(nthreads_saved, 4));
        accumulate_fmat(blk, dvec.data(), offdiag, fn);
        omp_set_num_threads(nthreads_saved);
        check(std::memcmp(&f1[0][0][0], &fn[0][0][0], f1.size() * sizeof(cplx)) == 0,
              label + ": 1-thread and multi-thread results differ on a restricted range");
    }
#endif
    // accumulation semantics: a seeded input is added to, not overwritten
    {
        V4RowBlock blk;
        blk.allocate(ns, nk, nk_irred, 0, nunits);
        fill_block_from_full(blk, v4);
        NDArray<cplx, 3> f(nk_irred, ns, ns);
        for (std::size_t i = 0; i < f.size(); ++i) {
            (&f[0][0][0])[i] = cplx(1.0, -1.0);
        }
        accumulate_fmat(blk, dvec.data(), offdiag, f);
        bool ok = true;
        for (std::size_t ik = 0; ik < nk_irred && ok; ++ik) {
            for (std::size_t a = 0; a < ns && ok; ++a) {
                for (std::size_t b = 0; b < ns && ok; ++b) {
                    ok = close(f[ik][a][b], f_ref[ik][a][b] + cplx(1.0, -1.0), tol);
                }
            }
        }
        check(ok, label + ": accumulate_fmat does not add to the seeded input (or touches the upper triangle)");
    }
}
// accumulate_fmat_batch: every (a, b), every right-hand side, against the naive
// formula; the partial sums of a random 3-rank partition must add up to it.
void test_fmat_batch(const std::size_t ns, const std::size_t nk, const std::size_t nk_irred, const std::size_t nrhs,
                     const unsigned seed, const std::string &label)
{
    const auto ns2 = ns * ns;
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    NDArray<cplx, 3> v4(nk_irred * nk, ns2, ns2);
    for (std::size_t i = 0; i < v4.size(); ++i) (&v4[0][0][0])[i] = cplx(g(rng), g(rng));
    std::vector<cplx> dmat(nrhs * nk * ns2);
    for (auto &x: dmat) x = cplx(g(rng), g(rng));

    std::vector<cplx> want(nrhs * nk_irred * ns2, cplx(0.0, 0.0));
    for (std::size_t m = 0; m < nrhs; ++m) {
        for (std::size_t ik = 0; ik < nk_irred; ++ik) {
            for (std::size_t a = 0; a < ns; ++a) {
                for (std::size_t b = 0; b < ns; ++b) {
                    cplx sum(0.0, 0.0);
                    for (std::size_t jk = 0; jk < nk; ++jk) {
                        for (std::size_t c = 0; c < ns2; ++c) {
                            sum += v4[ik * nk + jk][a * ns + b][c] * dmat[m * nk * ns2 + jk * ns2 + c];
                        }
                    }
                    want[m * nk_irred * ns2 + ik * ns2 + a * ns + b] = sum;
                }
            }
        }
    }

    const auto nunits = nk_irred * nk * ns;
    std::uniform_int_distribution<std::size_t> cut(0, nunits);
    std::size_t c1 = cut(rng), c2 = cut(rng);
    if (c1 > c2) std::swap(c1, c2);
    const std::vector<std::size_t> bounds{0, c1, c2, nunits};
    std::vector<cplx> got(want.size(), cplx(0.0, 0.0));
    for (int r = 0; r < 3; ++r) {
        V4RowBlock blk;
        blk.allocate(ns, nk, nk_irred, bounds[r], bounds[r + 1]);
        fill_block_from_full(blk, v4);
        std::vector<cplx> part(want.size(), cplx(0.0, 0.0));
        accumulate_fmat_batch(blk, dmat.data(), nrhs, part.data());
        for (std::size_t i = 0; i < got.size(); ++i) got[i] += part[i];
    }
    bool ok = true;
    for (std::size_t i = 0; i < want.size(); ++i) ok = ok && close(got[i], want[i], 1.0e-11);
    check(ok, label + ": accumulate_fmat_batch vs the naive formula (partitioned)");
}
} // namespace

int main()
{
    test_partitions();
    test_fmat(6, 3, 2, true, 3u, "offdiag nk=3 nk_irred=2");
    test_fmat(5, 1, 1, true, 5u, "offdiag gamma-only");
    test_fmat(7, 2, 3, false, 8u, "diagonal-only nk=2 nk_irred=3");
    test_fmat(4, 4, 1, false, 13u, "diagonal-only nk=4");
    test_fmat_batch(5, 3, 2, 4, 21u, "batch nk=3 nk_irred=2 nrhs=4");
    test_fmat_batch(6, 1, 1, 7, 34u, "batch gamma-only nrhs=7");

    if (nfail == 0) {
        std::printf("test_v4_distributed: all checks passed\n");
        return EXIT_SUCCESS;
    }
    std::printf("test_v4_distributed: %d check(s) failed\n", nfail);
    return EXIT_FAILURE;
}
