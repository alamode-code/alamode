/*
 test_ndarray.cpp

 Unit test for include/ndarray.h: verifies that NDArray<T, N> reproduces the
 memory.h allocate() layout exactly (contiguous payload + row-pointer tables),
 that the implicit view conversions behave as designed, and that the
 compile-time safety properties hold (ownership operations cannot bind, const
 objects vend only const views, owners are move-only).

 Built by the anphon CMake project as `test_ndarray`; run via
 test/test_ndarray.py or directly. Exits 0 on success.
*/

#include "memory.h"
#include "ndarray.h"
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

// ---- compile-time properties -------------------------------------------

// The views are reachable implicitly, including the kernel view convention.
static_assert(std::is_convertible<NDArray<double, 1> &, double *>::value, "1D view");
static_assert(std::is_convertible<NDArray<double, 2> &, double **>::value, "2D view");
static_assert(std::is_convertible<NDArray<double, 3> &, double ***>::value, "3D view");
static_assert(std::is_convertible<NDArray<std::complex<double>, 4> &, std::complex<double> ****>::value, "4D view");
static_assert(std::is_convertible<NDArray<std::complex<double>, 3> &,
                                  std::complex<double> *const *const *>::value,
              "kernel view convention (T *const *const *) reachable in one conversion sequence");
static_assert(std::is_convertible<const NDArray<double, 2> &, const double *const *>::value, "const 2D view");
static_assert(std::is_convertible<const NDArray<double, 3> &, const double *const *const *>::value, "const 3D view");

// A const object must NOT vend a mutable view.
static_assert(!std::is_convertible<const NDArray<double, 2> &, double **>::value, "no mutable view from const");
static_assert(!std::is_convertible<const NDArray<double, 3> &, double ***>::value, "no mutable view from const");

// Ownership operations cannot bind: allocate()/deallocate() take T**& and a
// conversion result is a prvalue. Leftover ownership call sites are compile
// errors during migration, never silent double frees.
static_assert(!std::is_convertible<NDArray<double, 2> &, double **&>::value, "cannot bind to T**&");
static_assert(!std::is_convertible<NDArray<double, 3> &, double ***&>::value, "cannot bind to T***&");

// Owners are move-only.
static_assert(!std::is_copy_constructible<NDArray<double, 3>>::value, "non-copyable");
static_assert(!std::is_copy_assignable<NDArray<double, 3>>::value, "non-copy-assignable");
static_assert(std::is_nothrow_move_constructible<NDArray<double, 3>>::value, "movable");
static_assert(std::is_nothrow_move_assignable<NDArray<double, 3>>::value, "move-assignable");

// ---- runtime checks -----------------------------------------------------

namespace
{
int nfail = 0;

void check(const bool ok, const char *what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++nfail;
    }
}

// A legacy-style consumer taking the raw view.
void fill_through_view(double **v, const std::size_t n1, const std::size_t n2)
{
    for (std::size_t i = 0; i < n1; ++i)
        for (std::size_t j = 0; j < n2; ++j)
            v[i][j] = static_cast<double>(100 * i + j);
}

double sum_through_const_view(const double *const *const *v, const std::size_t n1, const std::size_t n2,
                              const std::size_t n3)
{
    double s = 0.0;
    for (std::size_t i = 0; i < n1; ++i)
        for (std::size_t j = 0; j < n2; ++j)
            for (std::size_t k = 0; k < n3; ++k)
                s += v[i][j][k];
    return s;
}
} // namespace

int main()
{
    constexpr std::size_t n1 = 4, n2 = 5, n3 = 6, n4 = 3;

    // Layout identity vs memory.h, 3D.
    {
        NDArray<double, 3> a(n1, n2, n3);
        double ***b = nullptr;
        allocate(b, n1, n2, n3);
        bool same = true, contiguous = true;
        for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t j = 0; j < n2; ++j)
                for (std::size_t k = 0; k < n3; ++k) {
                    const auto off_a = &a[i][j][k] - a.data();
                    const auto off_b = &b[i][j][k] - &b[0][0][0];
                    same = same && (off_a == off_b);
                    contiguous = contiguous &&
                                 (off_a == static_cast<std::ptrdiff_t>(i * n2 * n3 + j * n3 + k));
                }
        check(same, "3D offsets identical to memory.h");
        check(contiguous, "3D payload contiguous");
        deallocate(b);
    }

    // Layout identity, 2D and 4D spot checks.
    {
        NDArray<double, 2> a(n1, n2);
        check(&a[n1 - 1][n2 - 1] - a.data() == static_cast<std::ptrdiff_t>(n1 * n2 - 1), "2D contiguous");

        NDArray<std::complex<double>, 4> c(n1, n2, n3, n4);
        check(&c[n1 - 1][n2 - 1][n3 - 1][n4 - 1] - c.data() ==
                  static_cast<std::ptrdiff_t>(n1 * n2 * n3 * n4 - 1),
              "4D contiguous");
        check(&c[2][3][4][1] - c.data() ==
                  static_cast<std::ptrdiff_t>(((2 * n2 + 3) * n3 + 4) * n4 + 1),
              "4D index arithmetic");
    }

    // View round-trip: write through the mutable view, read through data()
    // and through the deep-const 3D view.
    {
        NDArray<double, 2> a(n1, n2);
        fill_through_view(a, n1, n2);
        check(a.data()[1 * n2 + 3] == 103.0, "write through T** view lands in payload");

        NDArray<double, 3> t(2, 2, 2);
        for (std::size_t i = 0; i < t.size(); ++i) t.data()[i] = 1.0;
        check(sum_through_const_view(t, 2, 2, 2) == 8.0, "read through const view");
    }

    // resize() no-op on identical dims; reallocation on change.
    {
        NDArray<double, 2> a(n1, n2);
        const double *p0 = a.data();
        a.resize(n1, n2);
        check(a.data() == p0, "resize with same dims is a no-op");
        a.resize(n1 + 1, n2);
        check(a.dim(0) == n1 + 1 && a.size() == (n1 + 1) * n2, "resize updates shape");
    }

    // clear(), empty(), boolean context, zero leading dimension.
    {
        NDArray<double, 3> a(n1, n2, n3);
        check(static_cast<bool>(a), "non-empty is truthy via view conversion");
        a.clear();
        check(a.empty() && !static_cast<double ***>(a), "clear -> empty, null view");

        NDArray<double, 2> z(0, 7);
        check(!static_cast<double **>(z), "zero leading dim -> null table (memory.h convention)");
    }

    // Move semantics.
    {
        NDArray<double, 2> a(n1, n2);
        a.data()[0] = 42.0;
        const double *p0 = a.data();
        NDArray<double, 2> b(std::move(a));
        check(b.data() == p0 && b.data()[0] == 42.0, "move transfers ownership");
        check(a.empty(), "moved-from is empty");
    }

    // 1D.
    {
        NDArray<int, 1> v(10);
        int *raw = v;
        raw[9] = 7;
        check(v.data()[9] == 7 && v.size() == 10, "1D view and size");
    }

    if (nfail == 0) {
        std::printf("ndarray_test: all checks passed\n");
        return EXIT_SUCCESS;
    }
    std::printf("ndarray_test: %d check(s) FAILED\n", nfail);
    return EXIT_FAILURE;
}
