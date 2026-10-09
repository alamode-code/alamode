// Unit test of the real-space dynamical-matrix projection (anphon/dymat_projection.h)
// used by ScphQhaCommon::symmetrize_delta_dymat: group check, idempotence,
// invariance under every operation, an anisotropic supercell and a
// nonsymmorphic group.

#include <cstdio>
#include <random>
#include <vector>
#include "dymat_projection.h"

using namespace PHON_NS;

namespace
{
int nfail = 0;

void check(const bool ok, const char *what)
{
    std::printf("%-70s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) ++nfail;
}

// Contiguous [ns][ns][nk] complex array with pointer tables.
struct Dymat
{
    unsigned int ns, nk;
    std::vector<std::complex<double>> data;
    std::vector<std::complex<double> *> rows;
    std::vector<std::complex<double> **> tab;

    Dymat(const unsigned int ns_in, const unsigned int nk_in) :
        ns(ns_in), nk(nk_in), data(static_cast<size_t>(ns_in) * ns_in * nk_in), rows(ns_in * ns_in), tab(ns_in)
    {
        for (unsigned int i = 0; i < ns * ns; ++i) rows[i] = data.data() + static_cast<size_t>(i) * nk;
        for (unsigned int i = 0; i < ns; ++i) tab[i] = rows.data() + i * ns;
    }

    Dymat(const Dymat &o) : Dymat(o.ns, o.nk)
    {
        data = o.data;
    }

    std::complex<double> ***ptr()
    {
        return tab.data();
    }

    double maxdiff(const Dymat &o) const
    {
        double d = 0.0;
        for (size_t i = 0; i < data.size(); ++i) d = std::max(d, std::abs(data[i] - o.data[i]));
        return d;
    }

    void randomize(const unsigned int seed)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (auto &x: data) x = std::complex<double>(u(gen), 0.0);
    }
};

// Cubic cell (lattice = identity): Cartesian and fractional rotations coincide.
std::vector<DymatSymOp> cubic_ops(const Eigen::MatrixXd &xf, const int N[3])
{
    std::vector<DymatSymOp> ops;
    const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    for (const auto &p: perms) {
        for (auto signs = 0; signs < 8; ++signs) {
            DymatSymOp op;
            op.T.setZero();
            for (auto i = 0; i < 3; ++i) op.T(i, p[i]) = (signs >> i & 1) ? -1 : 1;
            op.S = op.T.cast<double>();
            op.t.setZero();
            if (!dymat_symop_keeps_grid(op, N)) continue;
            if (!set_dymat_symop_mapping(op, xf)) continue;
            ops.push_back(op);
        }
    }
    return ops;
}

std::vector<const DymatSymOp *> pointers(const std::vector<DymatSymOp> &ops)
{
    std::vector<const DymatSymOp *> p;
    for (const auto &op: ops) p.push_back(&op);
    return p;
}

// Idempotence and invariance of P(x) under every operation of ops.
void check_projection(const std::vector<const DymatSymOp *> &ops, const unsigned int nat, const int N[3],
                      const char *label)
{
    const auto nk = static_cast<unsigned int>(N[0] * N[1] * N[2]);
    Dymat d(3 * nat, nk);
    d.randomize(12345);
    const Dymat x(d);
    double moved = 0.0; // the operations act nontrivially on a random x
    for (const auto *op: ops) {
        Dymat g(x);
        project_dymat_r(g.ptr(), nat, N, {op});
        moved = std::max(moved, g.maxdiff(x));
    }
    project_dymat_r(d.ptr(), nat, N, ops);
    const Dymat zero(3 * nat, nk);
    char buf0[200];
    std::snprintf(buf0, sizeof buf0, "%s: nontrivial (g.x != x, P(x) != 0)", label);
    check((ops.size() == 1 || moved > 0.1) && d.maxdiff(zero) > 0.1, buf0);
    Dymat d2(d);
    project_dymat_r(d2.ptr(), nat, N, ops);
    char buf[200];
    std::snprintf(buf, sizeof buf, "%s: P(P(x)) = P(x)", label);
    check(d2.maxdiff(d) < 1.0e-13, buf);
    double worst = 0.0;
    for (const auto *op: ops) {
        Dymat g(d);
        project_dymat_r(g.ptr(), nat, N, {op}); // a single operation: g.P(x)
        worst = std::max(worst, g.maxdiff(d));
    }
    std::snprintf(buf, sizeof buf, "%s: g.P(x) = P(x) for every g", label);
    check(worst < 1.0e-13, buf);
}
} // namespace

int main()
{
    // ABO3 perovskite, A at the origin
    Eigen::MatrixXd xf(5, 3);
    xf << 0, 0, 0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0;

    {
        const int N[3] = {2, 2, 2};
        const auto ops = cubic_ops(xf, N);
        check(ops.size() == 48 && dymat_symops_form_group(pointers(ops)), "Oh on 2x2x2: 48 operations, a group");
        check_projection(pointers(ops), 5, N, "Oh, 2x2x2");

        // Componentwise selection with a tolerance (the former selection): an
        // optical displacement (0.0006, 0, 0) of B with tol = 0.001 admits the
        // 40 operations that do not send x to -x, which are no group.
        std::vector<const DymatSymOp *> admitted;
        for (const auto &op: ops) {
            const Eigen::Vector3d u(0.0006, 0.0, 0.0);
            if ((op.S * u - u).cwiseAbs().maxCoeff() < 0.001) admitted.push_back(&op);
        }
        check(admitted.size() == 40 && !dymat_symops_form_group(admitted),
              "near-threshold componentwise selection (40 of 48) is rejected");
    }
    {
        // anisotropic supercell: only the operations keeping z keep the grid
        const int N[3] = {2, 2, 3};
        const auto ops = cubic_ops(xf, N);
        check(ops.size() == 16 && dymat_symops_form_group(pointers(ops)), "2x2x3 keeps D4h (16 operations)");
        check_projection(pointers(ops), 5, N, "D4h, 2x2x3");
    }
    {
        // nonsymmorphic P2_1: E and the screw (-x, -y, z + 1/2)
        Eigen::MatrixXd x2(2, 3);
        x2 << 0.1, 0.2, 0.0, -0.1, -0.2, 0.5;
        const int N[3] = {2, 3, 4};
        std::vector<DymatSymOp> ops(2);
        ops[0].T.setIdentity();
        ops[0].t.setZero();
        ops[1].T = Eigen::Vector3i(-1, -1, 1).asDiagonal();
        ops[1].t = Eigen::Vector3d(0.0, 0.0, 0.5);
        auto mapped = true;
        for (auto &op: ops) {
            op.S = op.T.cast<double>();
            mapped = mapped && set_dymat_symop_mapping(op, x2);
        }
        check(mapped && ops[1].map[0] == 1 && dymat_symops_form_group(pointers(ops)),
              "P2_1 screw: atoms exchanged, a group");
        check_projection(pointers(ops), 2, N, "P2_1, 2x3x4");
        check(!dymat_symops_form_group({&ops[1]}), "a set without the identity is rejected");
    }

    std::printf(nfail ? "test_dymat_projection --> failed\n" : "test_dymat_projection --> pass\n");
    return nfail ? 1 : 0;
}
