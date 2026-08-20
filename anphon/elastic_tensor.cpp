/*
 elastic_tensor.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "elastic_tensor.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "cell_shift_table.h"
#include "constants.h"
#include "error.h"
#include "system.h"

using namespace PHON_NS;

ElasticTensor::ElasticTensor(const System &system_in) : system_(system_in)
{
    build_27cell_shift_table(xshift_s_);
}

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
    const auto &cell = system_.get_supercell(0);
    const auto &map_p2s = system_.get_map_p2s(0);

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

    Eigen::Vector3d xf_pair, xf_ref;

    for (const auto &it: fcs_in) {

        const auto atom_pair = map_p2s[it.pairs[1].index / 3][it.pairs[1].tran];
        const auto atom_ref = map_p2s[it.pairs[0].index / 3][0];

        for (auto j = 0; j < 3; ++j) {
            xf_pair[j] = cell.x_fractional(atom_pair, j) + xshift_s_[it.pairs[1].cell_s][j];
            xf_ref[j] = cell.x_fractional(atom_ref, j);
        }
        const Eigen::Vector3d rel = cell.lattice_vector * (xf_pair - xf_ref);

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
