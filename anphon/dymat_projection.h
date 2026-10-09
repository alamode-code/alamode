/*
 dymat_projection.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

// Projection of a real-space dynamical matrix D[is][js][icell] on the
// N1 x N2 x N3 supercell (atom j in cell +c of atom i in cell 0, icell =
// (c1 * N2 + c2) * N3 + c3) onto a space group. Header-only (Eigen); no MPI.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <complex>
#include <vector>

namespace PHON_NS
{
struct DymatSymOp
{
    Eigen::Matrix3d S;              // Cartesian rotation
    Eigen::Matrix3i T;              // fractional rotation
    Eigen::Vector3d t;              // fractional translation
    std::vector<unsigned int> map;  // atom j -> map[j]
    std::vector<Eigen::Vector3i> L; // T x_j + t = x_map(j) + L_j
};

// Fill map and L from the fractional positions xf (natom x 3); false when
// the operation does not map the atoms onto each other within tol.
inline bool set_dymat_symop_mapping(DymatSymOp &op, const Eigen::MatrixXd &xf, const double tol = 1.0e-3)
{
    const auto nat = static_cast<size_t>(xf.rows());
    op.map.assign(nat, 0);
    op.L.assign(nat, Eigen::Vector3i::Zero());
    for (size_t j = 0; j < nat; ++j) {
        const Eigen::Vector3d x = op.T.cast<double>() * xf.row(j).transpose() + op.t;
        auto found = false;
        for (size_t i = 0; i < nat && !found; ++i) {
            const Eigen::Vector3d d = x - xf.row(i).transpose();
            const Eigen::Vector3d dr = d.array().round().matrix();
            if ((d - dr).cwiseAbs().maxCoeff() < tol) {
                op.map[j] = static_cast<unsigned int>(i);
                op.L[j] = dr.cast<int>();
                found = true;
            }
        }
        if (!found) return false;
    }
    return true;
}

// The operation maps the lattice of the N supercell onto itself.
inline bool dymat_symop_keeps_grid(const DymatSymOp &op, const int N[3])
{
    for (auto i = 0; i < 3; ++i) {
        for (auto k = 0; k < 3; ++k) {
            if ((op.T(k, i) * N[i]) % N[k] != 0) return false;
        }
    }
    return true;
}

// Closure of the set (it contains the identity and every product a * b),
// which makes the average over it a projection.
inline bool dymat_symops_form_group(const std::vector<const DymatSymOp *> &ops, const double tol = 1.0e-5)
{
    const auto same = [tol](const Eigen::Matrix3i &T1, const Eigen::Vector3d &t1, const DymatSymOp &b) {
        if (T1 != b.T) return false;
        const Eigen::Vector3d d = t1 - b.t;
        return (d - d.array().round().matrix()).cwiseAbs().maxCoeff() < tol;
    };
    auto has_identity = false;
    for (const auto *op: ops) {
        if (same(Eigen::Matrix3i::Identity(), Eigen::Vector3d::Zero(), *op)) has_identity = true;
    }
    if (!has_identity) return false;
    for (const auto *a: ops) {
        for (const auto *b: ops) {
            const Eigen::Matrix3i T = a->T * b->T;
            const Eigen::Vector3d t = a->T.cast<double>() * b->t + a->t;
            auto found = false;
            for (const auto *c: ops) {
                if (same(T, t, *c)) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
    }
    return true;
}

// D <- (1/|G|) sum_g g.D with (g.D)(g i, g j, T c + L_j - L_i) = S D(i, j, c) S^T.
// ops must form a group (dymat_symops_form_group) and keep the grid.
inline void project_dymat_r(std::complex<double> ***dymat, const unsigned int natom, const int N[3],
                            const std::vector<const DymatSymOp *> &ops)
{
    const auto ns = 3 * natom;
    const auto nk = static_cast<unsigned int>(N[0] * N[1] * N[2]);
    std::vector<std::complex<double>> sum(static_cast<size_t>(ns) * ns * nk, 0.0);
    const auto at = [ns, nk](const unsigned int is, const unsigned int js, const unsigned int ir) {
        return (static_cast<size_t>(is) * ns + js) * nk + ir;
    };
    for (const auto *op: ops) {
        for (unsigned int iat = 0; iat < natom; ++iat) {
            for (unsigned int jat = 0; jat < natom; ++jat) {
                const auto gi = op->map[iat];
                const auto gj = op->map[jat];
                for (unsigned int ir = 0; ir < nk; ++ir) {
                    const Eigen::Vector3i c(ir / (N[1] * N[2]), (ir / N[2]) % N[1], ir % N[2]);
                    Eigen::Vector3i c2 = op->T * c + op->L[jat] - op->L[iat];
                    for (auto k = 0; k < 3; ++k) c2[k] = ((c2[k] % N[k]) + N[k]) % N[k];
                    const auto ir2 = static_cast<unsigned int>((c2[0] * N[1] + c2[1]) * N[2] + c2[2]);
                    Eigen::Matrix3cd B;
                    for (auto a = 0; a < 3; ++a) {
                        for (auto b = 0; b < 3; ++b) B(a, b) = dymat[3 * iat + a][3 * jat + b][ir];
                    }
                    const Eigen::Matrix3cd B2 = op->S * B * op->S.transpose();
                    for (auto a = 0; a < 3; ++a) {
                        for (auto b = 0; b < 3; ++b) sum[at(3 * gi + a, 3 * gj + b, ir2)] += B2(a, b);
                    }
                }
            }
        }
    }
    const auto inv = 1.0 / static_cast<double>(ops.size());
    for (unsigned int is = 0; is < ns; ++is) {
        for (unsigned int js = 0; js < ns; ++js) {
            for (unsigned int ir = 0; ir < nk; ++ir) dymat[is][js][ir] = sum[at(is, js, ir)] * inv;
        }
    }
}
} // namespace PHON_NS
