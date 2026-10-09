/*
 test_velocity_symmetry.cpp

 Unit test for anphon/velocity_symmetry.h: the little-group average of the phonon
 velocity operator in the atomic Cartesian basis.

 Synthetic crystals: (i) a hexagonal lattice with point group 3m (E, C3, C3^2 and three
 mirrors) and three atoms, two of which sit on the threefold axes through K, so that the
 cell phases of T(R) are nontrivial, at Gamma, K, M, a point on Gamma-A, a point on the
 K-M edge and a general point; (ii) the nonsymmorphic orthorhombic Pmc2_1 (2_1 screw and
 c glide with a fractional translation (0, 0, 1/2)), four atoms in the general position,
 at Gamma, on the kz = 1/2 face (Z, U, R and two lines) and inside the zone. With time
 reversal, it checks
   - that D(k) and dD/dk of a symmetric real-space force-constant model are fixed points
     of the averages (this is what pins the conventions: atom map, cell phase, placement
     of R and the sign of the time-reversal term);
 and on random Hermitian input:
   - the operations found form the little group (R k = +-k + G) and the Cartesian
     rotations are orthogonal;
   - the average is idempotent and its result is invariant under every element
     (the test the per-block polar construction failed, e.g. for a tilted 2D block at K);
   - the vector projector (finite-difference path) is idempotent and invariant;
   - with eigenvectors of a little-group symmetric dynamical matrix, the block traces
     Tr(P_B V^a P_B V^b) and the cross-block sums used by the coherent term do not change
     under random unitary rotations inside the degenerate blocks, and the block traces
     are invariant Cartesian tensors;
   - the result does not depend on the eigenvectors at all: the average of the same
     operator is unchanged when the eigenvectors come from a non-symmetric (tilted)
     dynamical matrix, which mixes the blocks.

 Built by the anphon CMake project as `test_velocity_symmetry`. Exits 0 on success.
*/

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <map>
#include <random>
#include <tuple>
#include <vector>
#include "velocity_symmetry.h"

using namespace PHON_NS::velocity_symmetry;
using Eigen::Matrix3d;
using Eigen::MatrixXcd;
using cplx = std::complex<double>;

static int nfail = 0;

static void check(const bool ok, const char *what, const double val)
{
    if (!ok) {
        ++nfail;
        std::printf("FAIL  %-60s %.3e\n", what, val);
    }
}

static double maxabs(const MatrixXcd (&a)[3], const MatrixXcd (&b)[3])
{
    double m = 0.0;
    for (auto mu = 0; mu < 3; ++mu) m = std::max(m, (a[mu] - b[mu]).cwiseAbs().maxCoeff());
    return m;
}

static MatrixXcd random_hermitian(const int n, std::mt19937 &rng)
{
    std::normal_distribution<double> g;
    MatrixXcd a(n, n);
    for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < n; ++j) a(i, j) = cplx(g(rng), g(rng));
    }
    return 0.5 * (a + a.adjoint());
}

static MatrixXcd random_unitary(const int n, std::mt19937 &rng)
{
    std::normal_distribution<double> g;
    MatrixXcd a(n, n);
    for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < n; ++j) a(i, j) = cplx(g(rng), g(rng));
    }
    return Eigen::HouseholderQR<MatrixXcd>(a).householderQ();
}

// Real-space force-constant model with the symmetry of the crystal: random real 3x3 blocks
// Phi(i, j, L) for every lattice vector L in {-1, 0, 1}^3, averaged over the group
// (Phi(map i, map j, L') = R Phi(i, j, L) R^T with x_j + L - x_i mapped by the operation)
// and over the transpose partner Phi(j, i, -L)^T.
using Key = std::tuple<int, int, int, int, int>;
static std::map<Key, Matrix3d> symmetric_model(const std::vector<Matrix3d> &rot_frac,
                                               const std::vector<std::vector<unsigned int>> &mapping,
                                               const Eigen::MatrixXd &xf, const Matrix3d &lavec, std::mt19937 &rng)
{
    std::normal_distribution<double> g;
    const auto nat = static_cast<int>(xf.rows());
    std::map<Key, Matrix3d> raw, sym;
    for (auto i = 0; i < nat; ++i) {
        for (auto j = 0; j < nat; ++j) {
            for (auto l0 = -1; l0 <= 1; ++l0) {
                for (auto l1 = -1; l1 <= 1; ++l1) {
                    for (auto l2 = -1; l2 <= 1; ++l2) {
                        Matrix3d f;
                        for (auto a = 0; a < 9; ++a) f(a / 3, a % 3) = g(rng);
                        raw[Key(i, j, l0, l1, l2)] = f;
                    }
                }
            }
        }
    }
    const Matrix3d linv = lavec.inverse();
    for (const auto &[key, f]: raw) {
        const auto [i, j, l0, l1, l2] = key;
        for (size_t s = 0; s < rot_frac.size(); ++s) {
            const Matrix3d r = lavec * rot_frac[s] * linv;
            const Eigen::Vector3d d = xf.row(j).transpose() + Eigen::Vector3d(l0, l1, l2) - xf.row(i).transpose();
            const int mi = static_cast<int>(mapping[s][i]), mj = static_cast<int>(mapping[s][j]);
            const Eigen::Vector3d lp = rot_frac[s] * d - (xf.row(mj) - xf.row(mi)).transpose();
            const Matrix3d img = r * f * r.transpose() / (2.0 * rot_frac.size());
            const Key k1(mi, mj, int(std::lround(lp[0])), int(std::lround(lp[1])), int(std::lround(lp[2])));
            const Key k2(mj, mi, -int(std::lround(lp[0])), -int(std::lround(lp[1])), -int(std::lround(lp[2])));
            if (!sym.count(k1)) sym[k1] = Matrix3d::Zero();
            if (!sym.count(k2)) sym[k2] = Matrix3d::Zero();
            sym[k1] += img;
            sym[k2] += img.transpose();
        }
    }
    return sym;
}

// D(k) (cell-phase gauge, unit masses) and the velocity operator in the convention of
// PhononVelocity::velocity_operator: the derivative of the displacement-aware phase,
// i 2 pi (x_j + L - x_i), taken to Cartesian components with the lattice vectors.
static void model_dk(const std::map<Key, Matrix3d> &fc, const Eigen::MatrixXd &xf, const Matrix3d &lavec,
                     const double k[3], MatrixXcd &d, MatrixXcd (&m)[3])
{
    const auto ns = 3 * static_cast<int>(xf.rows());
    d = MatrixXcd::Zero(ns, ns);
    for (auto &x: m) x = MatrixXcd::Zero(ns, ns);
    const double tpi = 6.283185307179586;
    for (const auto &[key, f]: fc) {
        const auto [i, j, l0, l1, l2] = key;
        const Eigen::Vector3d l(l0, l1, l2);
        const cplx ph = std::exp(cplx(0.0, tpi * (k[0] * l0 + k[1] * l1 + k[2] * l2)));
        const Eigen::Vector3d r = lavec * (xf.row(j).transpose() + l - xf.row(i).transpose());
        d.block(3 * i, 3 * j, 3, 3) += f.cast<cplx>() * ph;
        for (auto mu = 0; mu < 3; ++mu)
            m[mu].block(3 * i, 3 * j, 3, 3) += f.cast<cplx>() * (ph * cplx(0.0, tpi * r[mu]));
    }
}

// Degenerate blocks [lo, hi) of sorted eigenvalues.
static std::vector<std::pair<int, int>> blocks_of(const Eigen::VectorXd &w, const double tol)
{
    std::vector<std::pair<int, int>> b;
    for (int lo = 0; lo < w.size();) {
        auto hi = lo + 1;
        while (hi < w.size() && std::abs(w[hi] - w[lo]) < tol) ++hi;
        b.emplace_back(lo, hi);
        lo = hi;
    }
    return b;
}

// Tr(P_B V^a P_B V^b) for every block, and sum over pairs of different blocks (B, B')
// of sum_{i in B, j in B'} V^a_ij V^b_ji.
static void block_quantities(const MatrixXcd (&v)[3], const std::vector<std::pair<int, int>> &blk,
                             std::vector<Matrix3d> &tr, std::vector<Matrix3d> &cross)
{
    tr.clear();
    cross.clear();
    for (const auto &[lo, hi]: blk) {
        Matrix3d t;
        for (auto a = 0; a < 3; ++a) {
            for (auto b = 0; b < 3; ++b) {
                t(a, b) = (v[a].block(lo, lo, hi - lo, hi - lo) * v[b].block(lo, lo, hi - lo, hi - lo)).trace().real();
            }
        }
        tr.push_back(t);
    }
    for (const auto &[lo, hi]: blk) {
        for (const auto &[lo2, hi2]: blk) {
            if (lo == lo2) continue;
            Matrix3d t;
            for (auto a = 0; a < 3; ++a) {
                for (auto b = 0; b < 3; ++b) {
                    t(a, b) = (v[a].block(lo, lo2, hi - lo, hi2 - lo2) * v[b].block(lo2, lo, hi2 - lo2, hi - lo))
                                  .trace()
                                  .real();
                }
            }
            cross.push_back(t);
        }
    }
}

// Atom map of x -> R x + t (fractional) on the positions xf (rows); empty if not a symmetry.
static std::vector<std::vector<unsigned int>>
atom_maps(const std::vector<Matrix3d> &rot_frac, const std::vector<Eigen::Vector3d> &trans, const Eigen::MatrixXd &xf)
{
    const auto natmin = static_cast<int>(xf.rows());
    std::vector<std::vector<unsigned int>> mapping;
    for (size_t s = 0; s < rot_frac.size(); ++s) {
        std::vector<unsigned int> map(natmin);
        for (auto j = 0; j < natmin; ++j) {
            const Eigen::Vector3d x = rot_frac[s] * xf.row(j).transpose() + trans[s];
            auto found = -1;
            for (auto i = 0; i < natmin; ++i) {
                Eigen::Vector3d d = x - xf.row(i).transpose();
                for (auto c = 0; c < 3; ++c) d[c] -= std::round(d[c]);
                if (d.norm() < 1.0e-10) found = i;
            }
            if (found < 0) return {};
            map[j] = static_cast<unsigned int>(found);
        }
        mapping.push_back(map);
    }
    return mapping;
}

// All checks for one crystal (lattice vectors as columns, fractional positions as rows,
// space-group operations {R|t} in fractional coordinates) at the given k points.
static void check_crystal(const Matrix3d &lavec, const Eigen::MatrixXd &xf, const std::vector<Matrix3d> &rot_frac,
                          const std::vector<Eigen::Vector3d> &trans, const std::vector<std::array<double, 3>> &kpts,
                          std::mt19937 &rng)
{
    const auto ns = 3 * static_cast<int>(xf.rows());
    const auto mapping = atom_maps(rot_frac, trans, xf);
    if (mapping.empty()) {
        check(false, "synthetic crystal: operation does not map the atoms", 0.0);
        return;
    }
    const double tol = 1.0e-11;
    const auto model = symmetric_model(rot_frac, mapping, xf, lavec, rng);

    for (const auto &kv: kpts) {
        const double *k = kv.data();
        const auto ops = little_group(k, rot_frac, mapping, xf, lavec, true);
        char tag[128];

        // Little group and orthogonality.
        for (const auto &op: ops) {
            check((op.rot * op.rot.transpose() - Matrix3d::Identity()).norm() < 1.0e-12, "orthogonal rotation", 0.0);
        }
        check(!ops.empty() && !ops[0].antiunitary && (ops[0].rot - Matrix3d::Identity()).norm() < 1e-14,
              "identity first",
              0.0);

        // Conventions (atom map, cell phase, R placement, time-reversal sign): the
        // dynamical matrix and velocity operator of a symmetric model are fixed points.
        {
            MatrixXcd dk, mk[3], dks;
            model_dk(model, xf, lavec, k, dk, mk);
            dks = dk;
            symmetrize_scalar_operator(ops, dks);
            MatrixXcd mks[3] = {mk[0], mk[1], mk[2]};
            symmetrize_vector_operator(ops, mks);
            const auto sc = mk[0].cwiseAbs().maxCoeff();
            std::snprintf(tag, sizeof tag, "model D(k) fixed point at k=(%.3f %.3f %.3f)", k[0], k[1], k[2]);
            check((dks - dk).cwiseAbs().maxCoeff() < 1.0e-12 * sc, tag, (dks - dk).cwiseAbs().maxCoeff());
            std::snprintf(tag, sizeof tag, "model dD/dk fixed point at k=(%.3f %.3f %.3f)", k[0], k[1], k[2]);
            check(maxabs(mks, mk) < 1.0e-12 * sc, tag, maxabs(mks, mk));
        }

        // Idempotence and invariance on random Hermitian input.
        MatrixXcd m[3], pm[3], ppm[3], img[3];
        for (auto &x: m) x = random_hermitian(ns, rng);
        for (auto mu = 0; mu < 3; ++mu) pm[mu] = m[mu];
        symmetrize_vector_operator(ops, pm);
        for (auto mu = 0; mu < 3; ++mu) ppm[mu] = pm[mu];
        symmetrize_vector_operator(ops, ppm);
        std::snprintf(tag, sizeof tag, "idempotence at k=(%.3f %.3f %.3f), |G|=%zu", k[0], k[1], k[2], ops.size());
        check(maxabs(pm, ppm) < tol, tag, maxabs(pm, ppm));
        for (const auto &op: ops) {
            apply(op, pm, img);
            std::snprintf(tag, sizeof tag, "invariance at k=(%.3f %.3f %.3f)", k[0], k[1], k[2]);
            check(maxabs(pm, img) < tol, tag, maxabs(pm, img));
        }
        for (const auto &x: pm) check((x - x.adjoint()).cwiseAbs().maxCoeff() < tol, "hermiticity", 0.0);

        // Vector projector.
        const Matrix3d s = vector_projector(ops);
        check((s * s - s).norm() < 1.0e-12, "vector projector idempotent", (s * s - s).norm());
        for (const auto &op: ops) {
            const Matrix3d r = op.antiunitary ? Matrix3d(-op.rot) : op.rot;
            check((r * s - s).norm() < 1.0e-12, "vector projector invariant", (r * s - s).norm());
        }

        // Eigenvectors of a symmetric dynamical matrix: block quantities are gauge invariant.
        MatrixXcd d = random_hermitian(ns, rng);
        symmetrize_scalar_operator(ops, d);
        Eigen::SelfAdjointEigenSolver<MatrixXcd> es(d);
        const auto blk = blocks_of(es.eigenvalues(), 1.0e-9);
        MatrixXcd e = es.eigenvectors();
        MatrixXcd v[3];
        for (auto mu = 0; mu < 3; ++mu) v[mu] = e.adjoint() * pm[mu] * e;
        std::vector<Matrix3d> tr0, cr0, tr1, cr1;
        block_quantities(v, blk, tr0, cr0);
        for (const auto &[lo, hi]: blk) e.middleCols(lo, hi - lo) *= random_unitary(hi - lo, rng);
        for (auto mu = 0; mu < 3; ++mu) v[mu] = e.adjoint() * pm[mu] * e;
        block_quantities(v, blk, tr1, cr1);
        auto dg = 0.0, inv = 0.0;
        for (size_t b = 0; b < tr0.size(); ++b) {
            dg = std::max(dg, (tr0[b] - tr1[b]).cwiseAbs().maxCoeff());
            for (const auto &op: ops)
                inv = std::max(inv, (op.rot * tr0[b] * op.rot.transpose() - tr0[b]).cwiseAbs().maxCoeff());
        }
        for (size_t b = 0; b < cr0.size(); ++b) dg = std::max(dg, (cr0[b] - cr1[b]).cwiseAbs().maxCoeff());
        std::snprintf(tag,
                      sizeof tag,
                      "gauge invariance at k=(%.3f %.3f %.3f), %zu blocks",
                      k[0],
                      k[1],
                      k[2],
                      blk.size());
        check(dg < 1.0e-10, tag, dg);
        std::snprintf(tag, sizeof tag, "block traces invariant at k=(%.3f %.3f %.3f)", k[0], k[1], k[2]);
        check(inv < 1.0e-10, tag, inv);

        // Tilted blocks: an operator that is block diagonal in the eigenvectors of a
        // non-symmetric dynamical matrix (pairs of modes grouped across the symmetry
        // multiplets). The average works on the operator alone, so it stays an exact
        // projector; the per-block polar construction was not idempotent here.
        MatrixXcd dt = d + 0.3 * random_hermitian(ns, rng);
        Eigen::SelfAdjointEigenSolver<MatrixXcd> est(dt);
        const MatrixXcd et = est.eigenvectors();
        MatrixXcd mt[3], pmt[3], ppmt[3];
        for (auto mu = 0; mu < 3; ++mu) {
            MatrixXcd bd = MatrixXcd::Zero(ns, ns);
            for (auto lo = 0; lo < ns; lo += 2) {
                const auto w = std::min(2, ns - lo);
                bd.block(lo, lo, w, w) = random_hermitian(w, rng);
            }
            mt[mu] = et * bd * et.adjoint();
            pmt[mu] = mt[mu];
        }
        symmetrize_vector_operator(ops, pmt);
        for (auto mu = 0; mu < 3; ++mu) ppmt[mu] = pmt[mu];
        symmetrize_vector_operator(ops, ppmt);
        std::snprintf(tag, sizeof tag, "tilted blocks: idempotence at k=(%.3f %.3f %.3f)", k[0], k[1], k[2]);
        check(maxabs(pmt, ppmt) < tol, tag, maxabs(pmt, ppmt));
        for (const auto &op: ops) {
            apply(op, pmt, img);
            std::snprintf(tag, sizeof tag, "tilted blocks: invariance at k=(%.3f %.3f %.3f)", k[0], k[1], k[2]);
            check(maxabs(pmt, img) < tol, tag, maxabs(pmt, img));
        }
    }
}

int main()
{
    std::mt19937 rng(20261009);

    // Uniform translations at Gamma: D = (1 - P_T) H (1 - P_T) + 1e-14 (an acoustic-sum-rule
    // residual), with H positive definite. Its three lowest modes span the mass-weighted
    // translations; translational_modes must find them after the columns are shuffled.
    {
        const int nat = 4, ns = 3 * nat;
        std::uniform_real_distribution<double> um(1.0, 200.0);
        std::vector<double> mass(nat);
        for (auto &m: mass) m = um(rng);
        auto msum = 0.0;
        for (const auto m: mass) msum += m;
        MatrixXcd t = MatrixXcd::Zero(ns, 3);
        for (auto k = 0; k < nat; ++k) {
            for (auto a = 0; a < 3; ++a) t(3 * k + a, a) = std::sqrt(mass[k] / msum);
        }
        const MatrixXcd q = MatrixXcd::Identity(ns, ns) - t * t.adjoint();
        const MatrixXcd h = random_hermitian(ns, rng);
        MatrixXcd d = q * (h * h.adjoint() + MatrixXcd::Identity(ns, ns)) * q;
        d += 1.0e-14 * MatrixXcd::Identity(ns, ns);
        Eigen::SelfAdjointEigenSolver<MatrixXcd> es(d);
        std::vector<int> perm = {7, 2, 11, 0, 5, 9, 1, 3, 10, 4, 8, 6};
        MatrixXcd e(ns, ns);
        for (auto n = 0; n < ns; ++n) e.col(perm[n]) = es.eigenvectors().col(n);
        auto leak = 1.0;
        const auto idx = translational_modes(e, mass, &leak);
        check(std::abs(leak) < 1.0e-10, "translational_modes: no leak for separate translations", leak);
        // Mix a translation with an optical mode (as a degeneracy would allow): leak ~ 1/2.
        MatrixXcd e2 = e;
        e2.col(2) = (e.col(2) + e.col(5)) / std::sqrt(2.0);
        e2.col(5) = (e.col(2) - e.col(5)) / std::sqrt(2.0);
        translational_modes(e2, mass, &leak);
        check(leak > 0.4, "translational_modes: leak detected for a mixed basis", leak);
        check(idx == std::vector<int>({2, 7, 11}), "translational_modes picks the uniform translations", idx.size());
        const double g0[3] = {0.0, 0.0, 0.0}, g1[3] = {1.0, -2.0, 0.0}, k1[3] = {0.25, 0.0, 0.0};
        check(is_gamma(g0) && is_gamma(g1) && !is_gamma(k1), "is_gamma", 0.0);
    }

    // Symmorphic: hexagonal lattice, point group 3m, A at the origin and B, B' on the
    // threefold axes through K (nontrivial cell phases at K).
    {
        Matrix3d lavec;
        lavec << 1.0, -0.5, 0.0, 0.0, std::sqrt(3.0) / 2.0, 0.0, 0.0, 0.0, 1.6;
        Eigen::MatrixXd xf(3, 3);
        xf << 0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 0.25, 2.0 / 3.0, 1.0 / 3.0, 0.25;
        // C3 (x, y) -> (-y, x - y), mirror (x, y) -> (y, x).
        Matrix3d c3, sg;
        c3 << 0, -1, 0, 1, -1, 0, 0, 0, 1;
        sg << 0, 1, 0, 1, 0, 0, 0, 0, 1;
        const std::vector<Matrix3d> rot_frac = {Matrix3d::Identity(), c3, c3 * c3, sg, sg * c3, sg * c3 * c3};
        const std::vector<Eigen::Vector3d> trans(rot_frac.size(), Eigen::Vector3d::Zero());
        check_crystal(lavec,
                      xf,
                      rot_frac,
                      trans,
                      {{0.0, 0.0, 0.0},
                       {1.0 / 3.0, 1.0 / 3.0, 0.0},
                       {0.5, 0.0, 0.0},
                       {0.0, 0.0, 0.3},
                       {0.5, 0.0, 0.5},
                       {0.4, 0.2, 0.0},
                       {0.13, 0.29, 0.37}},
                      rng);
    }

    // Nonsymmorphic: orthorhombic Pmc2_1 (No. 26), {E|0}, {2_z|0 0 1/2} (screw),
    // {m_y|0 0 1/2} (c glide), {m_x|0}, four atoms in the general position. On the kz = 1/2
    // face the fractional translations give phases -1 and the bands stick in pairs.
    {
        const Matrix3d lavec = Eigen::Vector3d(1.0, 1.3, 1.7).asDiagonal();
        const Eigen::Vector3d x0(0.11, 0.23, 0.07);
        const std::vector<Matrix3d> rot_frac = {Matrix3d::Identity(),
                                                Eigen::Vector3d(-1, -1, 1).asDiagonal(),
                                                Eigen::Vector3d(1, -1, 1).asDiagonal(),
                                                Eigen::Vector3d(-1, 1, 1).asDiagonal()};
        const Eigen::Vector3d half(0.0, 0.0, 0.5);
        const std::vector<Eigen::Vector3d> trans = {Eigen::Vector3d::Zero(), half, half, Eigen::Vector3d::Zero()};
        Eigen::MatrixXd xf(4, 3);
        for (auto s = 0; s < 4; ++s) xf.row(s) = (rot_frac[s] * x0 + trans[s]).transpose();
        check_crystal(lavec,
                      xf,
                      rot_frac,
                      trans,
                      {{0.0, 0.0, 0.0},
                       {0.0, 0.0, 0.5},
                       {0.5, 0.0, 0.5},
                       {0.5, 0.5, 0.5},
                       {0.0, 0.3, 0.5},
                       {0.2, 0.0, 0.5},
                       {0.0, 0.0, 0.2},
                       {0.1, 0.2, 0.3}},
                      rng);
    }

    if (nfail == 0) {
        std::printf("test_velocity_symmetry: all checks passed\n");
        return 0;
    }
    std::printf("test_velocity_symmetry: %d check(s) failed\n", nfail);
    return 1;
}
