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

void ElasticTensor::calc_elastic_tensor(const std::vector<FcsArrayWithCell> &fcs_harmonic,
                                        NDArray<double, 4> &C_gpa) const
{
    NDArray<double, 4> A;
    calc_longwave_brackets(fcs_harmonic, A);

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
