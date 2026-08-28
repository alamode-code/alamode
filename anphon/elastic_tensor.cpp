/*
 elastic_tensor.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "elastic_tensor.h"
#include <Eigen/Dense>
#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "constants.h"
#include "error.h"
#include "ewald.h"
#include "ifc_derivative.h"
#include "system.h"

using namespace PHON_NS;

ElasticTensor::ElasticTensor(const System &system_in) : system_(system_in)
{}

void ElasticTensor::read_C1_array(double *C1_array)
{
    std::fstream fin_C1_array;
    std::string str_tmp;

    // initialize elastic constants
    for (auto i1 = 0; i1 < 9; i1++) {
        C1_array[i1] = 0.0;
    }

    fin_C1_array.open("C1_array.in");

    if (!fin_C1_array) {
        std::cout << "  Warning: file C1_array.in could not be open.\n";
        std::cout << "  The stress tensor at the reference structure is set zero.\n";
        return;
    }

    fin_C1_array >> str_tmp;
    for (auto i1 = 0; i1 < 9; i1++) {
        fin_C1_array >> C1_array[i1];
    }
}

void ElasticTensor::read_elastic_constants(double *const *C2_array, double *const *const *C3_array,
                                           const std::string &strain_ifc_dir)
{
    std::fstream fin_elastic_constants;
    std::string str_tmp;
    int i1, i2;

    // read elastic_constants.in from strain_ifc_dir directory
    fin_elastic_constants.open(strain_ifc_dir + "elastic_constants.in");

    if (!fin_elastic_constants) {
        exit("read_elastic_constants", "could not open file elastic_constants.in");
    }

    fin_elastic_constants >> str_tmp;
    for (i1 = 0; i1 < 9; i1++) {
        for (i2 = 0; i2 < 9; i2++) {
            fin_elastic_constants >> C2_array[i1][i2];
        }
    }
    fin_elastic_constants >> str_tmp;
    for (i1 = 0; i1 < 9; i1++) {
        for (i2 = 0; i2 < 9; i2++) {
            for (int i3 = 0; i3 < 9; i3++) {
                fin_elastic_constants >> C3_array[i1][i2][i3];
            }
        }
    }
}

void ElasticTensor::set_dummy_elastic_constants(double *C1_array, double *const *C2_array,
                                                double *const *const *C3_array)
{
    int i1, i2;
    std::fill_n(C1_array, 9, 0.0);

    // The elastic constant should be positive-definite
    // except for the rotational degrees of freedom
    for (i1 = 0; i1 < 3; i1++) {
        for (i2 = 0; i2 < 3; i2++) {
            for (int i3 = 0; i3 < 3; i3++) {
                for (int i4 = 0; i4 < 3; i4++) {
                    if ((i1 == i3 && i2 == i4) || (i1 == i4 && i2 == i3)) {
                        C2_array[i1 * 3 + i2][i3 * 3 + i4] = 10.0; // This dummy value can be any positive value
                    } else {
                        C2_array[i1 * 3 + i2][i3 * 3 + i4] = 0.0;
                    }
                }
            }
        }
    }

    for (i1 = 0; i1 < 9; i1++) {
        for (i2 = 0; i2 < 9; i2++) {
            std::fill_n(C3_array[i1][i2], 9, 0.0);
        }
    }
}

void ElasticTensor::calc_longwave_brackets(const std::vector<FcsArrayWithCell> &fcs_in, NDArray<double, 4> &ret) const
{
    // The relative vector of each pair is taken from relvecs_velocity (stored
    // in the primitive-lattice basis), the same geometric input used by the
    // strain-derivative kernels of DerivativeIFC.
    const auto convmat = system_.get_primcell().lattice_vector;

    ret.resize(3, 3, 3, 3);
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            for (auto k = 0; k < 3; ++k) {
                for (auto l = 0; l < 3; ++l) {
                    ret[i][j][k][l] = 0.0;
                }
            }
        }
    }

    for (const auto &it: fcs_in) {

        const Eigen::Vector3d rel = convmat * it.relvecs_velocity[0];

        const auto crd0 = it.pairs[0].index % 3;
        const auto crd1 = it.pairs[1].index % 3;

        for (auto k = 0; k < 3; ++k) {
            for (auto l = 0; l < 3; ++l) {
                ret[crd0][crd1][k][l] += it.fcs_val * rel[k] * rel[l];
            }
        }
    }

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            for (auto k = 0; k < 3; ++k) {
                for (auto l = 0; l < 3; ++l) {
                    ret[i][j][k][l] *= -0.5;
                }
            }
        }
    }
}

void ElasticTensor::brackets_to_elastic(const NDArray<double, 4> &A, NDArray<double, 4> &C_gpa,
                                        const bool symmetrize) const
{
    const auto volume = system_.get_primcell().volume * std::pow(Bohr_in_Angstrom, 3) * 1.0e-30; // in m^3
    const auto factor = 1.0e-9 * Ryd / volume;

    C_gpa.resize(3, 3, 3, 3);

    // This corresponds to Eq. (7.30) of Wallace's "Thermodynamics of Crystals" (1972)
    // when the initial stress is zero. If initial stress is nonzero, the formula would be
    // C_{abcd} = A_{acbd} + A_{bcad} - A_{abcd} - sigma_{bd} delta_{ac} - sigma_{ad} delta_{bc} + sigma_{cd} delta_{ab}
    // where sigma is the initial stress tensor. The initial stress is not considered here.
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            for (auto k = 0; k < 3; ++k) {
                for (auto l = 0; l < 3; ++l) {
                    C_gpa[i][j][k][l] = (A[i][k][j][l] + A[j][k][i][l] - A[i][j][k][l]) * factor;
                }
            }
        }
    }

    if (symmetrize) {
        symmetrize_elastic_tensor2(C_gpa);
    }
}

void ElasticTensor::calc_elastic_tensor(const std::vector<FcsArrayWithCell> &fcs_harmonic, NDArray<double, 4> &C_gpa,
                                        const bool symmetrize) const
{
    NDArray<double, 4> A;
    calc_longwave_brackets(fcs_harmonic, A);
    brackets_to_elastic(A, C_gpa, symmetrize);
}

void ElasticTensor::calc_longwave_brackets_dipole(Ewald &ewald_in, NDArray<double, 4> &ret) const
{
    // A^DD_{ab,cd} = 1/2 d^2/dq_c dq_d [sum_{kappa kappa'} Phi^DD(q)]_{ab} at
    // q -> 0, with the macroscopic (G = 0) term excluded so that the result is
    // the fixed-E (clamped-ion) dipole contribution (Born & Huang, Sec. 26-27).
    // Phi^DD(q) without G = 0 is analytic around q = 0, so central finite
    // differences with Richardson extrapolation converge rapidly.
    const auto natmin = static_cast<int>(system_.get_primcell().number_of_atoms);

    auto eval = [&](const Eigen::Vector3d &q) {
        Eigen::MatrixXcd phi;
        ewald_in.calc_dipole_fcs_q(q, phi);
        Eigen::Matrix3cd M = Eigen::Matrix3cd::Zero();
        for (int iat = 0; iat < natmin; ++iat) {
            for (int jat = 0; jat < natmin; ++jat) {
                for (auto a = 0; a < 3; ++a) {
                    for (auto b = 0; b < 3; ++b) {
                        M(a, b) += phi(3 * iat + a, 3 * jat + b);
                    }
                }
            }
        }
        return M;
    };

    auto second_derivs = [&](const double delta) {
        std::array<std::array<Eigen::Matrix3cd, 3>, 3> d2;
        const Eigen::Matrix3cd f0 = eval(Eigen::Vector3d::Zero());
        const auto id = Eigen::Matrix3d::Identity();

        for (auto c = 0; c < 3; ++c) {
            const Eigen::Matrix3cd fp = eval(delta * id.col(c));
            const Eigen::Matrix3cd fm = eval(-delta * id.col(c));
            d2[c][c] = (fp + fm - 2.0 * f0) / (delta * delta);
        }
        for (auto c = 0; c < 3; ++c) {
            for (auto d = c + 1; d < 3; ++d) {
                const Eigen::Vector3d ep = delta * (id.col(c) + id.col(d));
                const Eigen::Vector3d em = delta * (id.col(c) - id.col(d));
                d2[c][d] = (eval(ep) + eval(-ep) - eval(em) - eval(-em)) / (4.0 * delta * delta);
                d2[d][c] = d2[c][d];
            }
        }
        return d2;
    };

    constexpr double delta_q = 2.0e-2; // 1/bohr
    const auto d2_full = second_derivs(delta_q);
    const auto d2_half = second_derivs(0.5 * delta_q);

    ret.resize(3, 3, 3, 3);
    for (auto a = 0; a < 3; ++a) {
        for (auto b = 0; b < 3; ++b) {
            for (auto c = 0; c < 3; ++c) {
                for (auto d = 0; d < 3; ++d) {
                    const auto richardson = (4.0 * d2_half[c][d](a, b) - d2_full[c][d](a, b)) / 3.0;
                    ret[a][b][c][d] = 0.5 * richardson.real();
                }
            }
        }
    }
}

void ElasticTensor::calc_elastic_tensor_longrange(const std::vector<FcsArrayWithCell> &fcs_short, Ewald &ewald_in,
                                                  NDArray<double, 4> &C_gpa, const bool symmetrize) const
{
    NDArray<double, 4> A, A_dd;
    calc_longwave_brackets(fcs_short, A);
    calc_longwave_brackets_dipole(ewald_in, A_dd);

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            for (auto k = 0; k < 3; ++k) {
                for (auto l = 0; l < 3; ++l) {
                    A[i][j][k][l] += A_dd[i][j][k][l];
                }
            }
        }
    }

    brackets_to_elastic(A, C_gpa, symmetrize);
}

void ElasticTensor::calc_sublattice_response(const std::vector<FcsArrayWithCell> &fcs_harmonic,
                                             Eigen::MatrixXd &X) const
{
    Eigen::MatrixXd Lambda;
    calc_force_strain_coupling(fcs_harmonic, Lambda, X);
}

void ElasticTensor::calc_force_strain_coupling(const std::vector<FcsArrayWithCell> &fcs_harmonic,
                                               Eigen::MatrixXd &Lambda, Eigen::MatrixXd &X) const
{
    const auto ns = 3 * static_cast<int>(system_.get_primcell().number_of_atoms);

    // Force-strain coupling Lambda_{I, mu nu} = sum_{l kappa'} Phi_{lambda mu}(0 kappa; l kappa') R_nu
    // from the single-pass strain-derivative kernel, then symmetrized over (mu, nu).
    auto fcs_aligned = fcs_harmonic;
    sort_by_heading_indices const operator1(1);
    boost::sort::block_indirect_sort(fcs_aligned.begin(), fcs_aligned.end(), operator1);

    // Compute \sum_{l kappa'} Phi_{lambda mu}(0 kappa; l kappa') [R_nu(l kappa') - R_nu(0 kappa)] for all 9 strain components
    // This is equivalent to \sum_{l kappa'} Phi_{lambda mu}(0 kappa; l kappa') R_nu(l kappa') because of ASR
    std::vector<DeltaFcsStrainComponents> groups;
    DerivativeIFC::compute_dV_dumn_all_real_space(fcs_aligned, groups, 1, system_.get_primcell().lattice_vector);

    // Force symmetrize the strain components: Lambda_{I, (mu nu)} = (Lambda_{I, mu nu} + Lambda_{I, nu mu}) / 2
    // This is not necessary if the input IFC2 satisfies the rotational invariance,
    // but we usually do not impose the rotational invariance on the IFC2, so we need to symmetrize it here.
    Lambda = Eigen::MatrixXd::Zero(ns, 9);
    for (const auto &group: groups) {
        const auto row = static_cast<int>(group.pairs[0].index);
        for (auto mu = 0; mu < 3; ++mu) {
            for (auto nu = 0; nu < 3; ++nu) {
                Lambda(row, mu * 3 + nu) = 0.5 * (group.values[mu * 3 + nu] + group.values[nu * 3 + mu]);
            }
        }
    }

    // Zone-center harmonic matrix K_{I J} = sum_l Phi(0 kappa; l kappa').
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(ns, ns);
    for (const auto &it: fcs_harmonic) {
        K(it.pairs[0].index, it.pairs[1].index) += it.fcs_val;
    }

    // Pseudoinverse excluding the acoustic translations.
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(0.5 * (K + K.transpose()));
    const auto &eigs = solver.eigenvalues();
    const auto eig_max = eigs.cwiseAbs().maxCoeff();
    Eigen::VectorXd inv_eigs = Eigen::VectorXd::Zero(ns);
    for (auto i = 0; i < ns; ++i) {
        if (std::abs(eigs[i]) > eig_max * 1.0e-8) {
            inv_eigs[i] = 1.0 / eigs[i];
        }
    }

    // X = -K^+ Lambda, where K^+ is the pseudoinverse of K.
    X = -solver.eigenvectors() * inv_eigs.asDiagonal() * solver.eigenvectors().transpose() * Lambda;
}

void ElasticTensor::calc_elastic_tensor_relaxed(const std::vector<FcsArrayWithCell> &fcs_harmonic,
                                                NDArray<double, 4> &C_gpa) const
{
    // Clamped-ion part from the long-wave brackets.
    calc_elastic_tensor(fcs_harmonic, C_gpa);

    // Internal-strain correction, entirely in real space:
    // dC_{ab} = (1/Vcell) Lambda_{I,a} X_{I,b} = -(1/Vcell) Lambda^T K^+ Lambda.
    // The mass-weighted (mode-basis) form is mathematically identical; the
    // masses cancel, so the static real-space form is used here.
    Eigen::MatrixXd Lambda, X;
    calc_force_strain_coupling(fcs_harmonic, Lambda, X);

    const auto volume = system_.get_primcell().volume * std::pow(Bohr_in_Angstrom, 3) * 1.0e-30; // in m^3
    const auto factor = 1.0e-9 * Ryd / volume;

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            for (auto k = 0; k < 3; ++k) {
                for (auto l = 0; l < 3; ++l) {
                    C_gpa[i][j][k][l] += factor * Lambda.col(i * 3 + j).dot(X.col(k * 3 + l));
                }
            }
        }
    }
}


void ElasticTensor::calc_longwave_brackets3(const std::vector<FcsArrayWithCell> &fcs_cubic, const Eigen::MatrixXd &X,
                                            Tensor6 &A_hat) const
{
    // Wallace's restricted surface-free coefficient, stored as
    // A_hat(mu1, mu2, nu1, nu2, mu3, nu3):
    //   A^[0] (RRR)      = -1/2 sum Phi_{mu1 mu2 mu3} R12_nu1 R12_nu2 R13_nu3
    //   K^[1..3] (X groups) contracted with the sublattice response and
    //   projected with P12 (nu1 <-> nu2).
    // Volume normalization and unit conversion are applied by the caller.
    const auto convmat = system_.get_primcell().lattice_vector;
    const bool with_x = X.size() != 0;

    A_hat.setZero();
    Tensor6 K_sum; // unprojected, index order (mu1, nu1, mu2, nu2, mu3, nu3)

    for (const auto &it: fcs_cubic) {

        const auto phi = it.fcs_val;
        const auto index1 = it.pairs[0].index;
        const auto index2 = it.pairs[1].index;
        const auto index3 = it.pairs[2].index;
        const auto c1 = index1 % 3;
        const auto c2 = index2 % 3;
        const auto c3 = index3 % 3;

        const Eigen::Vector3d R12 = convmat * it.relvecs_velocity[0];
        const Eigen::Vector3d R13 = convmat * it.relvecs_velocity[1];

        // RRR group [Wallace Eq. (8.42)], already symmetric in (nu1, nu2)
        for (auto nu1 = 0; nu1 < 3; ++nu1) {
            for (auto nu2 = 0; nu2 < 3; ++nu2) {
                for (auto nu3 = 0; nu3 < 3; ++nu3) {
                    A_hat(c1, c2, nu1, nu2, c3, nu3) += -0.5 * phi * R12[nu1] * R12[nu2] * R13[nu3];
                }
            }
        }

        if (!with_x) continue;

        const Eigen::Vector3d R21 = -R12;
        const Eigen::Vector3d R23 = R13 - R12;
        const Eigen::Vector3d R31 = -R13;
        const Eigen::Vector3d R32 = R12 - R13;

        // X rows of the three legs (strain label a = mu*3+nu)
        const auto X1 = X.row(index1);
        const auto X2 = X.row(index2);
        const auto X3 = X.row(index3);

        for (auto mu = 0; mu < 3; ++mu) {
            for (auto nu = 0; nu < 3; ++nu) {
                const auto a = mu * 3 + nu;

                // K^[1]: one X leg, two relative vectors from that leg
                for (auto p = 0; p < 3; ++p) {
                    for (auto q = 0; q < 3; ++q) {
                        K_sum(mu, nu, c2, p, c3, q) += phi * X1(a) * R12[p] * R13[q];
                        K_sum(c1, p, mu, nu, c3, q) += phi * X2(a) * R21[p] * R23[q];
                        K_sum(c1, p, c2, q, mu, nu) += phi * X3(a) * R31[p] * R32[q];
                    }
                }

                // K^[2]: two X legs, one relative vector
                for (auto mu2 = 0; mu2 < 3; ++mu2) {
                    for (auto nu2 = 0; nu2 < 3; ++nu2) {
                        const auto b = mu2 * 3 + nu2;
                        for (auto p = 0; p < 3; ++p) {
                            K_sum(mu, nu, mu2, nu2, c3, p) += phi * X1(a) * X2(b) * R13[p];
                            K_sum(mu, nu, c2, p, mu2, nu2) += phi * X1(a) * X3(b) * R12[p];
                            K_sum(c1, p, mu, nu, mu2, nu2) += phi * X2(a) * X3(b) * R21[p];
                        }

                        // K^[3]: three X legs
                        for (auto mu3 = 0; mu3 < 3; ++mu3) {
                            for (auto nu3 = 0; nu3 < 3; ++nu3) {
                                K_sum(mu, nu, mu2, nu2, mu3, nu3) += phi * X1(a) * X2(b) * X3(mu3 * 3 + nu3);
                            }
                        }
                    }
                }
            }
        }
    }

    if (with_x) {
        // Restricted projection P12 (nu1 <-> nu2) of the X groups
        for (auto mu1 = 0; mu1 < 3; ++mu1)
            for (auto mu2 = 0; mu2 < 3; ++mu2)
                for (auto nu1 = 0; nu1 < 3; ++nu1)
                    for (auto nu2 = 0; nu2 < 3; ++nu2)
                        for (auto mu3 = 0; mu3 < 3; ++mu3)
                            for (auto nu3 = 0; nu3 < 3; ++nu3) {
                                A_hat(mu1, mu2, nu1, nu2, mu3, nu3) +=
                                    0.5 * (K_sum(mu1, nu1, mu2, nu2, mu3, nu3) + K_sum(mu1, nu2, mu2, nu1, mu3, nu3));
                            }
    }
}

double ElasticTensor::symmetrize_elastic_tensor3(Tensor6 &C3)
{
    // Average over the 6 permutations of the three index pairs and the 2^3
    // transpositions within each pair (48 operations in total).
    static constexpr int pair_perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

    Tensor6 sym;
    sym.setZero();

    int idx[3][2];
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            for (auto k = 0; k < 3; ++k) {
                for (auto l = 0; l < 3; ++l) {
                    for (auto m = 0; m < 3; ++m) {
                        for (auto n = 0; n < 3; ++n) {
                            idx[0][0] = i;
                            idx[0][1] = j;
                            idx[1][0] = k;
                            idx[1][1] = l;
                            idx[2][0] = m;
                            idx[2][1] = n;
                            auto avg = 0.0;
                            for (const auto &perm: pair_perms) {
                                for (auto flip = 0; flip < 8; ++flip) {
                                    int a[6];
                                    for (auto p = 0; p < 3; ++p) {
                                        const auto swap = (flip >> p) & 1;
                                        a[2 * p] = idx[perm[p]][swap];
                                        a[2 * p + 1] = idx[perm[p]][1 - swap];
                                    }
                                    avg += C3(a[0], a[1], a[2], a[3], a[4], a[5]);
                                }
                            }
                            sym(i, j, k, l, m, n) = avg / 48.0;
                        }
                    }
                }
            }
        }
    }

    auto max_change = 0.0;
    for (auto i = 0; i < 729; ++i) {
        max_change = std::max(max_change, std::abs(sym.data[i] - C3.data[i]));
    }
    C3 = sym;
    return max_change;
}

double ElasticTensor::symmetrize_elastic_tensor2(NDArray<double, 4> &C2)
{
    // Average over the pair exchange and the transpositions within each pair
    // (8 operations).
    NDArray<double, 4> sym(3, 3, 3, 3);
    auto max_change = 0.0;

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            for (auto k = 0; k < 3; ++k) {
                for (auto l = 0; l < 3; ++l) {
                    const auto avg = 0.125 * (C2[i][j][k][l] + C2[j][i][k][l] + C2[i][j][l][k] + C2[j][i][l][k] +
                                              C2[k][l][i][j] + C2[l][k][i][j] + C2[k][l][j][i] + C2[l][k][j][i]);
                    sym[i][j][k][l] = avg;
                    max_change = std::max(max_change, std::abs(avg - C2[i][j][k][l]));
                }
            }
        }
    }
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            for (auto k = 0; k < 3; ++k) {
                for (auto l = 0; l < 3; ++l) {
                    C2[i][j][k][l] = sym[i][j][k][l];
                }
            }
        }
    }
    return max_change;
}

double ElasticTensor::stress_tensor_asymmetry(const double *C1_array)
{
    auto dev = 0.0;
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            dev = std::max(dev, std::abs(C1_array[i * 3 + j] - C1_array[j * 3 + i]));
        }
    }
    return dev;
}

double ElasticTensor::elastic_tensor2_asymmetry(const double *const *C2_array)
{
    auto dev = 0.0;
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            for (auto k = 0; k < 3; ++k) {
                for (auto l = 0; l < 3; ++l) {
                    const auto ref = C2_array[i * 3 + j][k * 3 + l];
                    dev = std::max(dev, std::abs(ref - C2_array[j * 3 + i][k * 3 + l]));
                    dev = std::max(dev, std::abs(ref - C2_array[i * 3 + j][l * 3 + k]));
                    dev = std::max(dev, std::abs(ref - C2_array[k * 3 + l][i * 3 + j]));
                }
            }
        }
    }
    return dev;
}

double ElasticTensor::elastic_tensor3_asymmetry(const double *const *const *C3_array)
{
    Tensor6 tmp;
    for (auto i = 0; i < 3; ++i)
        for (auto j = 0; j < 3; ++j)
            for (auto k = 0; k < 3; ++k)
                for (auto l = 0; l < 3; ++l)
                    for (auto m = 0; m < 3; ++m)
                        for (auto n = 0; n < 3; ++n) tmp(i, j, k, l, m, n) = C3_array[i * 3 + j][k * 3 + l][m * 3 + n];
    return symmetrize_elastic_tensor3(tmp);
}

void ElasticTensor::calc_elastic_tensor3(const std::vector<FcsArrayWithCell> &fcs_harmonic,
                                         const std::vector<FcsArrayWithCell> &fcs_cubic, const bool relax_ions,
                                         Tensor6 &C3_gpa, const bool symmetrize) const
{
    Eigen::MatrixXd Lambda, X;
    if (relax_ions) {
        calc_force_strain_coupling(fcs_harmonic, Lambda, X);
    }

    Tensor6 A_hat;
    calc_longwave_brackets3(fcs_cubic, X, A_hat);

    NDArray<double, 4> C2;
    if (relax_ions) {
        calc_elastic_tensor_relaxed(fcs_harmonic, C2);
    } else {
        calc_elastic_tensor(fcs_harmonic, C2);
    }

    const auto volume = system_.get_primcell().volume * std::pow(Bohr_in_Angstrom, 3) * 1.0e-30; // in m^3
    const auto factor = 1.0e-9 * Ryd / volume;

    // Wallace Eq. (8.14): C3_{ij kl mn} from the restricted brackets and the
    // second-order tensor of the same path.
    const auto d = [](const int a, const int b) { return a == b ? 1.0 : 0.0; };

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            for (auto k = 0; k < 3; ++k) {
                for (auto l = 0; l < 3; ++l) {
                    for (auto m = 0; m < 3; ++m) {
                        for (auto n = 0; n < 3; ++n) {
                            C3_gpa(i, j, k, l, m, n) =
                                factor * (A_hat(i, k, j, l, m, n) + A_hat(j, k, i, l, m, n) - A_hat(i, j, k, l, m, n)) +
                                C2[k][l][m][n] * d(i, j) - C2[j][l][m][n] * d(i, k) - C2[i][l][m][n] * d(j, k) -
                                0.5 * (2.0 * C2[i][j][l][n] + C2[i][l][j][n] + C2[j][l][i][n]) * d(k, m) +
                                0.5 * (C2[j][l][k][n] - C2[k][l][j][n]) * d(i, m) +
                                0.5 * (C2[i][l][k][n] - C2[k][l][i][n]) * d(j, m);
                        }
                    }
                }
            }
        }
    }

    if (symmetrize) {
        symmetrize_elastic_tensor3(C3_gpa);
    }
}

void ElasticTensor::print_elastic_tensor(const std::vector<FcsArrayWithCell> &fcs_harmonic) const
{
    NDArray<double, 4> A, C;

    calc_longwave_brackets(fcs_harmonic, A);
    calc_elastic_tensor(fcs_harmonic, C);

    unsigned int i, j, k, l;

    std::cout << "# A [Ryd]" << '\n';

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            for (k = 0; k < 3; ++k) {
                for (l = 0; l < 3; ++l) {
                    std::cout << std::setw(3) << i + 1;
                    std::cout << std::setw(3) << j + 1;
                    std::cout << std::setw(3) << k + 1;
                    std::cout << std::setw(3) << l + 1;
                    std::cout << std::setw(15) << std::fixed << A[i][j][k][l];
                    std::cout << '\n';
                }
            }
        }
    }

    std::cout << '\n';
    std::cout << "# C [GPa]" << '\n';

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            for (k = 0; k < 3; ++k) {
                for (l = 0; l < 3; ++l) {
                    std::cout << std::setw(3) << i + 1;
                    std::cout << std::setw(3) << j + 1;
                    std::cout << std::setw(3) << k + 1;
                    std::cout << std::setw(3) << l + 1;
                    std::cout << std::setw(15) << std::fixed << C[i][j][k][l];
                    std::cout << '\n';
                }
            }
        }
    }

    std::cout << "Bulk Modulus [GPa] = " << (C[0][0][0][0] + 2.0 * C[0][0][1][1]) / 3.0 << '\n';
}
