/*
 velocity_symmetry.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

// Little-group symmetrization of the phonon velocity operator at one k point,
// in the atomic Cartesian basis (before any projection onto eigenvectors).
//
// T(R) maps a displacement field at k onto one at R k: atom j goes to map[j],
// its Cartesian components are rotated by R and multiplied by the cell phase
// exp(i 2 pi k.(S^-1 x_map(j) - x_j)) (the matrix of get_symmetry_gamma_dynamical).
// The velocity operator M^mu = dD~/dk_mu then satisfies, for R k = k + G,
//     M^mu = sum_nu R_{mu nu} T M^nu T^dagger,
// and, for R k = -k + G combined with time reversal (D(-k) = D(k)^*),
//     M^mu = -sum_nu R_{mu nu} (T M^nu T^dagger)^*.
// The T(R) form a representation of the little group up to phases, which cancel
// in T M T^dagger, so the average over the little group is an exact projector,
// independent of eigenvectors and degeneracies. Header-only (Eigen), so that it
// can be unit tested (test/test_velocity_symmetry.cpp).

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace PHON_NS
{
namespace velocity_symmetry
{
struct LittleGroupOp
{
    Eigen::Matrix3d rot;                     // Cartesian rotation R
    bool antiunitary = false;                // R k = -k + G, combined with time reversal
    std::vector<unsigned int> map;           // atom j -> map[j]
    std::vector<std::complex<double>> phase; // phase of source atom j
};

// Little group of xk (fractional reciprocal coordinates). rot_frac[i] is the rotation
// in fractional real-space coordinates and mapping[i] the atom map of operation i;
// xf holds the fractional atomic positions (rows) and lavec the lattice vectors (columns).
inline std::vector<LittleGroupOp> little_group(const double xk[3], const std::vector<Eigen::Matrix3d> &rot_frac,
                                               const std::vector<std::vector<unsigned int>> &mapping,
                                               const Eigen::MatrixXd &xf, const Eigen::Matrix3d &lavec,
                                               const bool time_reversal, const double tol = 1.0e-8)
{
    std::vector<LittleGroupOp> ops;
    const Eigen::Vector3d k(xk[0], xk[1], xk[2]);
    const auto natmin = static_cast<unsigned int>(xf.rows());
    const Eigen::Matrix3d lavec_inv = lavec.inverse();

    for (size_t isym = 0; isym < rot_frac.size(); ++isym) {
        const Eigen::Matrix3d rinv = rot_frac[isym].inverse();
        const Eigen::Vector3d sk = rinv.transpose() * k; // reciprocal-space action
        auto plus = true;
        auto minus = time_reversal;
        for (auto i = 0; i < 3; ++i) {
            const auto dp = sk[i] - k[i];
            const auto dm = sk[i] + k[i];
            if (std::abs(dp - std::round(dp)) > tol) plus = false;
            if (std::abs(dm - std::round(dm)) > tol) minus = false;
        }
        for (const auto anti: {false, true}) {
            if ((!anti && !plus) || (anti && !minus)) continue;
            LittleGroupOp op;
            op.rot = lavec * rot_frac[isym] * lavec_inv;
            op.antiunitary = anti;
            op.map = mapping[isym];
            op.phase.resize(natmin);
            for (unsigned int j = 0; j < natmin; ++j) {
                const Eigen::Vector3d xd = rinv * xf.row(op.map[j]).transpose() - xf.row(j).transpose();
                op.phase[j] = std::exp(std::complex<double>(0.0, 6.283185307179586 * k.dot(xd)));
            }
            ops.push_back(op);
        }
    }
    return ops;
}

// T M T^dagger for a matrix M in the atomic Cartesian basis (3 natmin x 3 natmin).
inline Eigen::MatrixXcd transform(const LittleGroupOp &op, const Eigen::MatrixXcd &m)
{
    const auto natmin = static_cast<unsigned int>(op.map.size());
    const Eigen::Matrix3cd r = op.rot.cast<std::complex<double>>();
    Eigen::MatrixXcd out(m.rows(), m.cols());
    for (unsigned int i = 0; i < natmin; ++i) {
        for (unsigned int j = 0; j < natmin; ++j) {
            out.block<3, 3>(3 * op.map[i], 3 * op.map[j]) =
                (op.phase[i] * std::conj(op.phase[j])) * (r * m.block<3, 3>(3 * i, 3 * j) * r.transpose());
        }
    }
    return out;
}

// Image of the vector operator m[3] under one operation:
// out^mu = sum_nu R_{mu nu} X^nu, X = T m T^dagger, or -(T m T^dagger)^* if antiunitary.
inline void apply(const LittleGroupOp &op, const Eigen::MatrixXcd (&m)[3], Eigen::MatrixXcd (&out)[3])
{
    for (auto &o: out) o = Eigen::MatrixXcd::Zero(m[0].rows(), m[0].cols());
    for (auto nu = 0; nu < 3; ++nu) {
        Eigen::MatrixXcd x = transform(op, m[nu]);
        if (op.antiunitary) x = -x.conjugate().eval();
        for (auto mu = 0; mu < 3; ++mu) out[mu] += op.rot(mu, nu) * x;
    }
}

// m[3] <- (1/|G_k|) sum_R image_R(m). No-op for a trivial little group.
inline void symmetrize_vector_operator(const std::vector<LittleGroupOp> &ops, Eigen::MatrixXcd (&m)[3])
{
    if (ops.size() <= 1) return;
    Eigen::MatrixXcd acc[3], img[3];
    for (auto &a: acc) a = Eigen::MatrixXcd::Zero(m[0].rows(), m[0].cols());
    for (const auto &op: ops) {
        apply(op, m, img);
        for (auto mu = 0; mu < 3; ++mu) acc[mu] += img[mu];
    }
    for (auto mu = 0; mu < 3; ++mu) m[mu] = acc[mu] / static_cast<double>(ops.size());
}

// d <- (1/|G_k|) sum_R T d T^dagger (conjugated for the antiunitary elements): the
// little-group average of a scalar operator such as the dynamical matrix.
inline void symmetrize_scalar_operator(const std::vector<LittleGroupOp> &ops, Eigen::MatrixXcd &d)
{
    if (ops.size() <= 1) return;
    Eigen::MatrixXcd acc = Eigen::MatrixXcd::Zero(d.rows(), d.cols());
    for (const auto &op: ops) {
        const Eigen::MatrixXcd x = transform(op, d);
        acc += op.antiunitary ? Eigen::MatrixXcd(x.conjugate()) : x;
    }
    d = acc / static_cast<double>(ops.size());
}

// True if xk (fractional reciprocal coordinates) is a reciprocal lattice vector.
inline bool is_gamma(const double xk[3], const double tol = 1.0e-8)
{
    for (auto i = 0; i < 3; ++i) {
        if (std::abs(xk[i] - std::round(xk[i])) > tol) return false;
    }
    return true;
}

// The three modes that span the uniform translations at k = 0: the eigenvectors (columns
// of evec, atomic Cartesian basis 3 kappa + i) with the largest weight
// sum_alpha |<t_alpha|e_n>|^2 in the span of the mass-weighted translations
// t_alpha(3 kappa + i) = sqrt(m_kappa / sum m) delta_{i alpha}. The projector has trace 3,
// so the weights sum to 3 and are ~1 for these modes whatever their frequencies (an
// acoustic-sum-rule residual leaves them slightly finite). Their velocity <n|M|n> / 2w is
// 0/0 there; callers set it, and their velocity-matrix rows and columns, to zero.
// Selection by weight assumes no optical mode is degenerate with them, true unless an
// optical frequency is also ~0: then the translations can mix with that mode and the choice
// depends on the basis. *leak (if given) receives 3 minus the weight of the chosen modes,
// ~0 normally and of order 1 in that case, so that callers can warn.
inline std::vector<int> translational_modes(const Eigen::MatrixXcd &evec, const std::vector<double> &mass,
                                            double *leak = nullptr)
{
    if (leak) *leak = 0.0;
    const auto nmode = evec.rows();
    const auto natmin = static_cast<Eigen::Index>(mass.size());
    if (nmode != 3 * natmin || nmode < 3) return {};
    auto msum = 0.0;
    for (const auto m: mass) msum += m;
    Eigen::MatrixXcd t = Eigen::MatrixXcd::Zero(nmode, 3);
    for (Eigen::Index k = 0; k < natmin; ++k) {
        for (auto a = 0; a < 3; ++a) t(3 * k + a, a) = std::sqrt(mass[k] / msum);
    }
    const Eigen::VectorXd w = (t.adjoint() * evec).cwiseAbs2().colwise().sum().transpose();
    std::vector<int> idx(nmode);
    for (Eigen::Index n = 0; n < nmode; ++n) idx[n] = static_cast<int>(n);
    std::partial_sort(idx.begin(), idx.begin() + 3, idx.end(), [&w](const int a, const int b) { return w[a] > w[b]; });
    idx.resize(3);
    if (leak) *leak = 3.0 - (w[idx[0]] + w[idx[1]] + w[idx[2]]);
    std::sort(idx.begin(), idx.end());
    return idx;
}

// Projector onto the Cartesian vectors invariant under the little group:
// (1/|G_k|) sum_R (+-R), with - for the antiunitary elements.
inline Eigen::Matrix3d vector_projector(const std::vector<LittleGroupOp> &ops)
{
    if (ops.empty()) return Eigen::Matrix3d::Identity();
    Eigen::Matrix3d s = Eigen::Matrix3d::Zero();
    for (const auto &op: ops) s += op.antiunitary ? Eigen::Matrix3d(-op.rot) : op.rot;
    return s / static_cast<double>(ops.size());
}
} // namespace velocity_symmetry
} // namespace PHON_NS
