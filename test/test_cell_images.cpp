/*
 test_cell_images.cpp

 Unit test for images_beyond_27 (anphon/cell_shift_table.h), which extends the
 {-1,0,1}^3 minimum-image search of System::get_minimum_distances to skewed
 supercells.

 - For random fractional vectors (half of them on a quarter grid, i.e. exact
   ties) in an orthogonal, a hexagonal and two skewed cells, the minimum images
   found by the 27 cells plus images_beyond_27 must equal those of a brute-force
   search over [-6,6]^3.
 - The orthogonal and hexagonal cells must never need the extension (their
   lists stay as before). The skewed cell a(0,0,2), a(2,0,0), a(2,2,0) misses
   equidistant images only; the 4x4x4 supercell of the basis (1,0,0), (2,1,0),
   (0,0,1) also misses the minimum image itself.
 - Cost: for an elongated orthogonal cell diag(10, 10, 10000) bohr and
   dx = (0, 0, 0.5), where no image beyond the 27 qualifies, the number of
   candidates tried must stay small (a box bound would try ~2e6).

 Built by the anphon CMake project as `test_cell_images`. Exits 0 on success.
*/

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstdio>
#include <random>
#include <set>
#include "cell_shift_table.h"

using PHON_NS::images_beyond_27;
using PHON_NS::lattice_r_factor;
using Shift = std::array<int, 3>;

static double dist(const Eigen::Matrix3d &lat, const double dx[3], const Shift &n)
{
    return (lat * Eigen::Vector3d(dx[0] + n[0], dx[1] + n[1], dx[2] + n[2])).norm();
}

// Minimum distance and the minimum images (within tol) among the candidate shifts.
static std::pair<double, std::set<Shift>> min_images(const std::vector<std::pair<double, Shift>> &cand,
                                                     const double tol)
{
    double dmin = 1.0e100;
    for (const auto &c: cand) dmin = std::min(dmin, c.first);
    std::set<Shift> out;
    for (const auto &c: cand) {
        if (c.first - dmin < tol) out.insert(c.second);
    }
    return {dmin, out};
}

struct Counts
{
    int extended = 0;    // vectors whose 27-image list was incomplete
    int new_minimum = 0; // ... whose true minimum lies beyond the 27 cells
    bool ok = true;
};

static Counts check_cell(const char *name, const Eigen::Matrix3d &lat, std::mt19937 &rng)
{
    constexpr double tol = 1.0e-3;
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::uniform_int_distribution<int> grid(-4, 4);
    const Eigen::Matrix3d rfac = lattice_r_factor(lat);
    Counts c;
    for (auto trial = 0; trial < 4000; ++trial) {
        double dx[3];
        for (auto &x: dx) x = trial % 2 ? uni(rng) : 0.25 * grid(rng);
        std::vector<std::pair<double, Shift>> brute, fast;
        for (auto i = -6; i <= 6; ++i) {
            for (auto j = -6; j <= 6; ++j) {
                for (auto k = -6; k <= 6; ++k) {
                    const Shift n{i, j, k};
                    const auto d = dist(lat, dx, n);
                    brute.emplace_back(d, n);
                    if (std::abs(i) <= 1 && std::abs(j) <= 1 && std::abs(k) <= 1) fast.emplace_back(d, n);
                }
            }
        }
        const auto d27 = min_images(fast, tol).first;
        const auto extra = images_beyond_27(rfac, dx, d27 + tol);
        fast.insert(fast.end(), extra.begin(), extra.end());
        const auto ref = min_images(brute, tol);
        if (!extra.empty()) ++c.extended;
        if (ref.first < d27 - tol) ++c.new_minimum;
        if (min_images(fast, tol).second != ref.second) {
            std::printf("FAIL %s: dx = %g %g %g\n", name, dx[0], dx[1], dx[2]);
            c.ok = false;
            return c;
        }
    }
    std::printf("%-12s: %4d of 4000 vectors needed images beyond the 27 cells (%d with a new minimum)\n",
                name,
                c.extended,
                c.new_minimum);
    return c;
}

int main()
{
    std::mt19937 rng(12345);
    const double a = 7.363;
    Eigen::Matrix3d ortho, hexa, skew, skew2, elong;
    ortho << 2 * a, 0, 0, 0, 2.2 * a, 0, 0, 0, 2.5 * a;
    // columns a1 = a(1,0,0), a2 = a(-1/2, sqrt3/2, 0), a3 = c(0,0,1)
    hexa << a, -0.5 * a, 0, 0, 0.8660254037844386 * a, 0, 0, 0, 1.6 * a;
    // columns a(0,0,2), a(2,0,0), a(2,2,0)
    skew << 0, 2 * a, 2 * a, 0, 0, 2 * a, 2 * a, 0, 0;
    // columns 4a(1,0,0), 4a(2,1,0), 4a(0,0,1)
    skew2 << 4 * a, 8 * a, 0, 0, 4 * a, 0, 0, 0, 4 * a;

    const auto c_ortho = check_cell("orthogonal", ortho, rng);
    const auto c_hexa = check_cell("hexagonal", hexa, rng);
    const auto c_skew = check_cell("skewed", skew, rng);
    const auto c_skew2 = check_cell("skewed-4x4x4", skew2, rng);
    if (!c_ortho.ok || !c_hexa.ok || !c_skew.ok || !c_skew2.ok) return 1;
    if (c_ortho.extended != 0 || c_hexa.extended != 0 || c_skew.extended == 0 || c_skew2.new_minimum == 0) {
        std::printf("FAIL: expected no extension for the orthogonal/hexagonal cells, ties for the skewed one\n"
                    "      and new minimum images for the skewed 4x4x4 one\n");
        return 1;
    }

    elong << 10, 0, 0, 0, 10, 0, 0, 0, 10000;
    const double dx[3] = {0.0, 0.0, 0.5};
    long nvisit = 0;
    const auto extra = images_beyond_27(lattice_r_factor(elong), dx, 5000.0 + 1.0e-3, &nvisit);
    std::printf("elongated   : %ld candidates tried, %zu images beyond the 27 cells\n", nvisit, extra.size());
    if (!extra.empty() || nvisit > 100) {
        std::printf("FAIL: elongated cell\n");
        return 1;
    }
    std::printf("test_cell_images: pass\n");
    return 0;
}
